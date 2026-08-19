// Main-thread inference client. Owns the inference Web Worker lifecycle,
// per-run timeouts, and the WebGPU → WASM fallback policy. No ORT code here —
// everything onnxruntime-related lives in inference.worker.ts.
import type { ModelId } from '@/types';

export class InferenceTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InferenceTimeoutError';
  }
}

type Backend = 'webgpu' | 'wasm';
type Dtype = 'fp16' | 'fp32';

export interface LoadedModel {
  backend: Backend;
  dtype: Dtype;
}

interface LoadWaiter {
  onProgress?: (loaded: number, total: number) => void;
  onMessage?: (message: string) => void;
  resolve: (entry: LoadedModel) => void;
  reject: (err: unknown) => void;
  cleanup: () => void;
}

interface InFlightLoad {
  modelId: ModelId;
  waiters: Set<LoadWaiter>;
}

interface PendingRun {
  runId: number;
  timer: ReturnType<typeof setTimeout> | null;
  cleanup: () => void;
  resolve: (data: Float32Array) => void;
  reject: (err: unknown) => void;
}

type WorkerOutboundMessage =
  | { type: 'load-progress'; modelId: ModelId; loaded: number; total: number }
  | { type: 'load-message'; modelId: ModelId; message: string }
  | { type: 'loaded'; modelId: ModelId; backend: Backend; dtype: Dtype }
  | { type: 'load-error'; modelId: ModelId; message: string }
  | { type: 'run-result'; runId: number; data: ArrayBuffer }
  | { type: 'run-error'; runId: number; message: string }
  | { type: 'worker-error'; message: string };

export interface EnsureModelOptions {
  signal?: AbortSignal;
  forceWasm?: boolean;
  onProgress?: (loaded: number, total: number) => void;
  onMessage?: (message: string) => void;
}

export interface RunTileOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const loadedModels = new Map<ModelId, LoadedModel>();
// Models marked forceWasm stay on WASM for every future load, even when a
// later ensureModel call doesn't pass the flag.
const forceWasmModels = new Map<ModelId, boolean>();
const pendingLoads = new Map<ModelId, InFlightLoad>();
const pendingRuns = new Map<number, PendingRun>();

let workerPromise: Promise<Worker> | null = null;
let workerGeneration = 0;
let runIdCounter = 0;

export function hasWebGPU(): boolean {
  return typeof navigator.gpu !== 'undefined';
}

// Cache whether the GPU adapter supports fp16 shaders. Fetching the fp16
// model without this check wastes ~35 MB on devices without shader-f16
// (older iGPUs, some virtualized GPUs, SwiftShader).
let fp16SupportedPromise: Promise<boolean> | null = null;
export function supportsFp16(): Promise<boolean> {
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

export function getBackend(id: ModelId): Backend | null {
  return loadedModels.get(id)?.backend ?? null;
}

export function getDtype(id: ModelId): Dtype | null {
  return loadedModels.get(id)?.dtype ?? null;
}

export function isModelLoaded(id: ModelId): boolean {
  return loadedModels.has(id);
}

function spawnWorker(): Worker {
  const generation = ++workerGeneration;
  const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
    type: 'module',
  });
  worker.onmessage = (event: MessageEvent) => {
    if (generation !== workerGeneration) return; // stale worker — ignore
    handleWorkerMessage(event.data as WorkerOutboundMessage);
  };
  worker.onerror = (event: ErrorEvent) => {
    if (generation !== workerGeneration) return;
    workerGeneration++; // invalidate this worker
    workerPromise = null;
    rejectAllPending(new Error(event.message || 'Inference worker crashed'));
  };
  return worker;
}

function getWorker(): Promise<Worker> {
  workerPromise ??= Promise.resolve(spawnWorker());
  return workerPromise;
}

function rejectLoad(load: InFlightLoad, err: unknown): void {
  pendingLoads.delete(load.modelId);
  for (const waiter of load.waiters) {
    waiter.cleanup();
    waiter.reject(err);
  }
  load.waiters.clear();
}

function rejectRun(run: PendingRun, err: unknown): void {
  pendingRuns.delete(run.runId);
  if (run.timer !== null) clearTimeout(run.timer);
  run.cleanup();
  run.reject(err);
}

function resolveRun(run: PendingRun, data: Float32Array): void {
  pendingRuns.delete(run.runId);
  if (run.timer !== null) clearTimeout(run.timer);
  run.cleanup();
  run.resolve(data);
}

function rejectAllPending(err: unknown): void {
  for (const load of [...pendingLoads.values()]) rejectLoad(load, err);
  for (const run of [...pendingRuns.values()]) rejectRun(run, err);
}

// Kill the worker (a run may never settle — terminate instead of waiting),
// drop every pending operation, and force the next spawn to respawn clean.
function killWorker(): void {
  workerGeneration++;
  if (workerPromise) {
    const wp = workerPromise;
    workerPromise = null;
    void wp.then((worker) => worker.terminate()).catch(() => {});
  }
  rejectAllPending(new Error('Inference worker was terminated'));
}

function handleWorkerMessage(message: WorkerOutboundMessage): void {
  switch (message.type) {
    case 'load-progress': {
      const load = pendingLoads.get(message.modelId);
      if (!load) return;
      for (const waiter of load.waiters) {
        waiter.onProgress?.(message.loaded, message.total);
      }
      break;
    }
    case 'load-message': {
      const load = pendingLoads.get(message.modelId);
      if (!load) return;
      for (const waiter of load.waiters) {
        waiter.onMessage?.(message.message);
      }
      break;
    }
    case 'loaded': {
      const load = pendingLoads.get(message.modelId);
      if (!load) return;
      const entry: LoadedModel = { backend: message.backend, dtype: message.dtype };
      loadedModels.set(message.modelId, entry);
      pendingLoads.delete(message.modelId);
      for (const waiter of load.waiters) {
        waiter.cleanup();
        waiter.resolve(entry);
      }
      load.waiters.clear();
      break;
    }
    case 'load-error': {
      const load = pendingLoads.get(message.modelId);
      if (!load) return;
      rejectLoad(load, new Error(message.message));
      break;
    }
    case 'run-result': {
      const run = pendingRuns.get(message.runId);
      if (!run) return;
      resolveRun(run, new Float32Array(message.data));
      break;
    }
    case 'run-error': {
      const run = pendingRuns.get(message.runId);
      if (!run) return;
      rejectRun(run, new Error(message.message));
      break;
    }
    case 'worker-error': {
      rejectAllPending(new Error(message.message));
      break;
    }
  }
}

function joinLoad(
  id: ModelId,
  load: InFlightLoad,
  opts?: EnsureModelOptions
): Promise<LoadedModel> {
  return new Promise<LoadedModel>((resolve, reject) => {
    const waiter: LoadWaiter = {
      onProgress: opts?.onProgress,
      onMessage: opts?.onMessage,
      resolve,
      reject,
      cleanup: () => {},
    };
    const onAbort = () => {
      load.waiters.delete(waiter);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    if (opts?.signal) {
      if (opts.signal.aborted) {
        onAbort();
        return;
      }
      opts.signal.addEventListener('abort', onAbort, { once: true });
      waiter.cleanup = () => opts.signal!.removeEventListener('abort', onAbort);
    }
    load.waiters.add(waiter);
  });
}

function beginLoad(id: ModelId, opts?: EnsureModelOptions): Promise<LoadedModel> {
  const load: InFlightLoad = { modelId: id, waiters: new Set() };
  pendingLoads.set(id, load);
  void getWorker().then((worker) => {
    // Superseded (worker terminated while spawning) — a new load was started.
    if (pendingLoads.get(id) !== load) return;
    worker.postMessage({
      type: 'load',
      modelId: id,
      forceWasm: forceWasmModels.get(id) === true,
    });
  }).catch((err) => {
    if (pendingLoads.get(id) === load) rejectLoad(load, err);
  });
  return joinLoad(id, load, opts);
}

export function ensureModel(id: ModelId, opts?: EnsureModelOptions): Promise<LoadedModel> {
  if (opts?.forceWasm) forceWasmModels.set(id, true);
  const cached = loadedModels.get(id);
  if (cached) return Promise.resolve(cached);

  const inFlight = pendingLoads.get(id);
  if (inFlight) return joinLoad(id, inFlight, opts);

  return beginLoad(id, opts);
}

export async function runTile(
  modelId: ModelId,
  chw: Float32Array,
  h: number,
  w: number,
  opts?: RunTileOptions
): Promise<Float32Array> {
  const timeoutMs = opts?.timeoutMs ?? 60000;
  const worker = await getWorker();
  const runId = ++runIdCounter;

  return new Promise<Float32Array>((resolve, reject) => {
    // Always transfer a copy: the caller may reuse chw after this resolves
    // (TTA re-runs the same tile under the 8 transforms), and transferring
    // the buffer in place would detach it.
    const data = new Float32Array(chw);
    const buffer = data.buffer;

    const run: PendingRun = {
      runId,
      timer: null,
      cleanup: () => {},
      resolve,
      reject,
    };
    pendingRuns.set(runId, run);

    run.timer = setTimeout(() => {
      pendingRuns.delete(runId);
      run.cleanup();
      forceWasmModels.set(modelId, true);
      killWorker();
      reject(
        new InferenceTimeoutError(`WebGPU/CPU inference timed out after ${timeoutMs / 1000}s`)
      );
    }, timeoutMs);

    const onAbort = () => {
      if (!pendingRuns.has(runId)) return;
      rejectRun(run, new DOMException('Aborted', 'AbortError'));
    };
    if (opts?.signal) {
      if (opts.signal.aborted) {
        rejectRun(run, new DOMException('Aborted', 'AbortError'));
        return;
      }
      opts.signal.addEventListener('abort', onAbort, { once: true });
      run.cleanup = () => opts.signal!.removeEventListener('abort', onAbort);
    }

    try {
      worker.postMessage({ type: 'run', runId, modelId, data: buffer, h, w }, [buffer]);
    } catch (err) {
      // Worker gone — reject and respawn on the next use.
      rejectRun(run, err);
      workerPromise = null;
    }
  });
}

export function forceWasmFallback(modelId: ModelId): void {
  forceWasmModels.set(modelId, true);
  killWorker();
}

export function disposeAll(): void {
  killWorker();
  loadedModels.clear();
  forceWasmModels.clear();
}
