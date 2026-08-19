// Dedicated Web Worker: ALL onnxruntime-web code lives here. The main thread
// never touches ORT directly — it posts plain messages and receives results,
// so GPU shader compilation and inference can never freeze the UI.
import * as ort from 'onnxruntime-web';
import { MODEL_BY_ID } from '@/lib/constants';
import type { ModelId } from '@/types';

// self is typed as Window under the DOM lib; workers accept
// postMessage(message, transfer) — declare the scope structurally.
const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
};

type InboundMessage =
  | { type: 'load'; modelId: ModelId; forceWasm?: boolean }
  | { type: 'run'; runId: number; modelId: ModelId; data: ArrayBuffer; h: number; w: number };

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'webgpu' | 'wasm';
  dtype: 'fp16' | 'fp32';
}

const sessions = new Map<ModelId, SessionEntry>();
const loadingModels = new Set<ModelId>();

const threadCount = crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 4)
  : 1;
ort.env.wasm.numThreads = threadCount;

// Warm-up inference timeout: if the first WebGPU run (kernel compilation)
// never settles, fall back to WASM instead of hanging the load forever.
const WARMUP_TIMEOUT_MS = 60000;

// Cache whether the GPU adapter supports fp16 shaders. Fetching the fp16
// model without this check wastes ~35 MB on devices without shader-f16
// (older iGPUs, some virtualized GPUs, SwiftShader).
let fp16SupportedPromise: Promise<boolean> | null = null;
function supportsFp16(): Promise<boolean> {
  fp16SupportedPromise ??= (async () => {
    try {
      if (typeof navigator.gpu === 'undefined') return false;
      const adapter = await navigator.gpu.requestAdapter();
      return adapter ? adapter.features.has('shader-f16') : false;
    } catch {
      return false;
    }
  })();
  return fp16SupportedPromise;
}

async function fetchModelBuffer(
  url: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Model file not found at ${url}. Place the ONNX model in public/models/.`
    );
  }
  const body = response.body;
  const contentLength = response.headers.get('Content-Length');
  if (body && contentLength) {
    const total = Number(contentLength);
    const reader = body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loaded += value.byteLength;
        onProgress?.(loaded, total);
      }
    }
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return buffer.buffer;
  }
  return response.arrayBuffer();
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(label)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

// One dummy inference on a 64x64 input so the WebGPU kernels compile during
// model loading instead of during the user's first tile.
async function runWarmUp(session: ort.InferenceSession): Promise<void> {
  const size = 64;
  const input = new Float32Array(3 * size * size); // all zeros is fine
  const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  results[session.outputNames[0]].dispose();
}

async function createWebGpuSession(
  buffer: ArrayBuffer,
  dtype: 'fp16' | 'fp32',
  modelId: ModelId
): Promise<SessionEntry | null> {
  let session: ort.InferenceSession;
  try {
    session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
  } catch {
    return null;
  }
  scope.postMessage({
    type: 'load-message',
    modelId,
    message: 'Preparing GPU kernels (first run compiles shaders)...',
  });
  try {
    await withTimeout(runWarmUp(session), WARMUP_TIMEOUT_MS, 'GPU warm-up timed out');
  } catch {
    // GPU unusable (hung or failed kernel) — release and fall back to WASM.
    void session.release().catch(() => {});
    return null;
  }
  return { session, backend: 'webgpu', dtype };
}

async function createWasmSession(buffer: ArrayBuffer): Promise<SessionEntry> {
  const session = await ort.InferenceSession.create(buffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    intraOpNumThreads: threadCount,
  });
  return { session, backend: 'wasm', dtype: 'fp32' };
}

async function loadModel(modelId: ModelId, forceWasm: boolean): Promise<void> {
  // Dedupe: loads of the same model that are already in flight or complete
  // are no-ops. Loads of different models serialize via the message queue.
  if (sessions.has(modelId) || loadingModels.has(modelId)) return;
  loadingModels.add(modelId);
  try {
    const config = MODEL_BY_ID[modelId];
    let entry: SessionEntry | null = null;

    if (!forceWasm && (await supportsFp16())) {
      try {
        const buffer = await fetchModelBuffer(config.urlFp16, (loaded, total) =>
          scope.postMessage({ type: 'load-progress', modelId, loaded, total })
        );
        entry = await createWebGpuSession(buffer, 'fp16', modelId);
      } catch {
        // fp16 fetch failed — fall back to fp32 below.
      }
    }

    if (!entry) {
      const buffer = await fetchModelBuffer(config.url, (loaded, total) =>
        scope.postMessage({ type: 'load-progress', modelId, loaded, total })
      );
      if (!forceWasm) {
        entry = await createWebGpuSession(buffer, 'fp32', modelId);
      }
      if (!entry) {
        entry = await createWasmSession(buffer);
      }
    }

    sessions.set(modelId, entry);
    scope.postMessage({ type: 'load-message', modelId, message: 'Ready' });
    scope.postMessage({ type: 'loaded', modelId, backend: entry.backend, dtype: entry.dtype });
  } catch (err) {
    scope.postMessage({
      type: 'load-error',
      modelId,
      message: err instanceof Error ? err.message : String(err),
    });
  } finally {
    loadingModels.delete(modelId);
  }
}

async function runOne(
  runId: number,
  modelId: ModelId,
  data: ArrayBuffer,
  h: number,
  w: number
): Promise<void> {
  const entry = sessions.get(modelId);
  if (!entry) {
    scope.postMessage({ type: 'run-error', runId, message: `Model ${modelId} not loaded` });
    return;
  }
  try {
    const tensor = new ort.Tensor('float32', new Float32Array(data), [1, 3, h, w]);
    const feeds: Record<string, ort.Tensor> = { [entry.session.inputNames[0]]: tensor };
    const results = await entry.session.run(feeds);
    const output = results[entry.session.outputNames[0]];
    // Copy out before dispose — .data may be a CPU view of the tensor.
    const out = new Float32Array((await output.getData()) as Float32Array);
    output.dispose();
    scope.postMessage({ type: 'run-result', runId, data: out.buffer }, [out.buffer]);
  } catch (err) {
    scope.postMessage({
      type: 'run-error',
      runId,
      message: err instanceof Error ? err.message : String(err),
    });
  }
}

// Serialize all work: one task at a time, so a second run waits for the
// first and loads never interleave. A failing task never breaks the chain.
let queueTail: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void>): void {
  queueTail = queueTail.then(task, task).catch(() => {});
}

scope.onmessage = (event: MessageEvent) => {
  const message = event.data as InboundMessage;
  if (message.type === 'load') {
    enqueue(() => loadModel(message.modelId, message.forceWasm === true));
  } else if (message.type === 'run') {
    enqueue(() => runOne(message.runId, message.modelId, message.data, message.h, message.w));
  }
};

scope.onerror = (event: ErrorEvent) => {
  scope.postMessage({ type: 'worker-error', message: event.message || 'Unknown worker error' });
  // Handled here — don't propagate to the main thread's onerror.
  event.preventDefault();
};
