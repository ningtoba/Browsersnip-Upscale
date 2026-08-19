// Dedicated Web Worker: ALL onnxruntime-web code lives here. The main thread
// never touches ORT directly — it posts plain messages and receives results,
// so inference can never freeze the UI.
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
  | { type: 'load'; modelId: ModelId }
  | { type: 'run'; runId: number; modelId: ModelId; data: ArrayBuffer; h: number; w: number };

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'wasm';
  dtype: 'fp32';
}

const sessions = new Map<ModelId, SessionEntry>();
const loadingModels = new Set<ModelId>();

// Single-threaded WASM, like the sibling BrowserSnip-Blurred project: ORT's
// threaded runtime spawns its own worker scripts that break inside bundled
// module workers (hangs in session creation, stray worker boot loops). The
// single-threaded path is battle-tested on every device.
const threadCount = 1;
ort.env.wasm.numThreads = threadCount;
console.log(`[upscale-worker] boot (threads=${threadCount}, isolated=${crossOriginIsolated})`);

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

async function loadModel(modelId: ModelId): Promise<void> {
  console.log(`[upscale-worker] loadModel start: ${modelId}`);
  if (sessions.has(modelId) || loadingModels.has(modelId)) return;
  loadingModels.add(modelId);
  try {
    const config = MODEL_BY_ID[modelId];
    console.log(`[upscale-worker] fetching ${config.url}`);
    const buffer = await fetchModelBuffer(config.url, (loaded, total) =>
      scope.postMessage({ type: 'load-progress', modelId, loaded, total })
    );
    console.log(`[upscale-worker] fetched ${(buffer.byteLength / 1048576).toFixed(1)} MB — creating wasm session`);
    // A session-creation hang must surface as a load-error, never stall.
    const session = await withTimeout(
      ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
        intraOpNumThreads: threadCount,
      }),
      90000,
      'WASM session creation timed out'
    );
    console.log(`[upscale-worker] session ready: ${modelId}`);
    const entry: SessionEntry = { session, backend: 'wasm', dtype: 'fp32' };
    sessions.set(modelId, entry);
    scope.postMessage({ type: 'load-message', modelId, message: 'Ready' });
    scope.postMessage({ type: 'loaded', modelId, backend: entry.backend, dtype: entry.dtype });
  } catch (err) {
    console.error(`[upscale-worker] load failed: ${modelId}`, err);
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
  console.log(`[upscale-worker] message: ${message.type}${message.type === 'run' ? ` #${message.runId} ${message.w}x${message.h}` : ''}`);
  if (message.type === 'load') {
    enqueue(() => loadModel(message.modelId));
  } else if (message.type === 'run') {
    enqueue(() => runOne(message.runId, message.modelId, message.data, message.h, message.w));
  }
};

scope.onerror = (event: ErrorEvent) => {
  scope.postMessage({ type: 'worker-error', message: event.message || 'Unknown worker error' });
  // Handled here — don't propagate to the main thread's onerror.
  event.preventDefault();
};
