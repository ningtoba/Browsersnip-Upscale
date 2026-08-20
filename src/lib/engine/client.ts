// Main-thread inference client. Owns the inference Web Worker pool lifecycle,
// per-run timeouts, and a load-phase timeout so no stage can stall silently.
// No ORT code here — everything onnxruntime-related lives in inference.worker.ts.
import type { ModelId } from '@/types';
import { MODEL_TIER } from '@/lib/constants';

export class InferenceTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InferenceTimeoutError';
  }
}

type Backend = 'wasm';
type Dtype = 'fp32';

export interface LoadedModel {
  backend: Backend;
  dtype: Dtype;
  /** Pool generation at load time — invalidated when the pool is rebuilt. */
  epoch: number;
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
  timer: ReturnType<typeof setTimeout> | undefined;
  // How many workers in the pool have posted 'loaded' for this model. The
  // load only resolves once every worker has its own session — runTile
  // round-robins across the whole pool.
  loadedWorkers: number;
}

interface PendingRun {
  runId: number;
  timer: ReturnType<typeof setTimeout> | undefined;
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
  onProgress?: (loaded: number, total: number) => void;
  onMessage?: (message: string) => void;
}

export interface RunTileOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const loadedModels = new Map<ModelId, LoadedModel>();
const pendingLoads = new Map<ModelId, InFlightLoad>();
const pendingRuns = new Map<number, PendingRun>();

// Worker pool: sized per model tier (2 workers for the big standard models,
// up to 4 for compact ones). The pool is rebuilt only when the tier changes
// (model switches are rare); poolGeneration invalidates stale workers.
let workers: Worker[] = [];
let poolTier: 'compact' | 'standard' | null = null;
let poolCount = 0;
let poolGeneration = 0;
let runIdCounter = 0;

export function getBackend(id: ModelId): Backend | null {
  return loadedModels.get(id)?.backend ?? null;
}

export function getDtype(id: ModelId): Dtype | null {
  return loadedModels.get(id)?.dtype ?? null;
}

export function isModelLoaded(id: ModelId): boolean {
  const entry = loadedModels.get(id);
  return entry !== undefined && entry.epoch === poolGeneration;
}

export function getWorkerCount(id: ModelId): number {
  return MODEL_TIER[id] === 'compact'
    ? Math.min(4, navigator.hardwareConcurrency || 4)
    : Math.min(2, navigator.hardwareConcurrency || 2);
}

function spawnWorker(index: number): Worker {
  const generation = poolGeneration;
  console.log(`[upscale-client] spawning inference worker ${index} (gen ${generation})`);
  const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
    type: 'module',
  });
  worker.onmessage = (event: MessageEvent) => {
    if (generation !== poolGeneration) return; // stale worker — ignore
    handleWorkerMessage(event.data as WorkerOutboundMessage, index);
  };
  worker.onerror = (event: ErrorEvent) => {
    if (generation !== poolGeneration) return;
    teardownPool();
    rejectAllPending(new Error(event.message || 'Inference worker crashed'));
  };
  return worker;
}

// Terminate every worker, invalidate the pool, and force the next use to
// spawn a fresh one. In-flight messages from terminated workers are dropped
// by the generation guard above.
function teardownPool(): void {
  poolGeneration++;
  for (const worker of workers) worker.terminate();
  workers = [];
  poolTier = null;
  poolCount = 0;
}

function ensurePool(modelId: ModelId): void {
  const tier = MODEL_TIER[modelId];
  const count = getWorkerCount(modelId);
  if (poolTier === tier && poolCount === count && workers.length > 0) return;

  // Loads in flight when the pool is rebuilt (model switch mid-load) would
  // otherwise be orphaned: their 'loaded' messages are dropped by the
  // generation guard and their waiters would hang. Re-post them to the
  // fresh pool so every waiter still resolves.
  const inFlight = [...pendingLoads.keys()];

  teardownPool();
  poolTier = tier;
  poolCount = count;
  for (let i = 0; i < count; i++) {
    workers.push(spawnWorker(i));
  }
  for (const id of inFlight) {
    const load = pendingLoads.get(id);
    if (!load) continue;
    load.loadedWorkers = 0;
    console.log(`[upscale-client] re-posting in-flight load: ${id} to ${workers.length} worker(s)`);
    for (const worker of workers) {
      worker.postMessage({ type: 'load', modelId: id });
    }
  }
}

function rejectLoad(load: InFlightLoad, err: unknown): void {
  pendingLoads.delete(load.modelId);
  clearTimeout(load.timer);
  for (const waiter of load.waiters) {
    waiter.cleanup();
    waiter.reject(err);
  }
  load.waiters.clear();
}

function rejectRun(run: PendingRun, err: unknown): void {
  pendingRuns.delete(run.runId);
  clearTimeout(run.timer);
  run.cleanup();
  run.reject(err);
}

function resolveRun(run: PendingRun, data: Float32Array): void {
  pendingRuns.delete(run.runId);
  clearTimeout(run.timer);
  run.cleanup();
  run.resolve(data);
}

function rejectAllPending(err: unknown): void {
  for (const load of [...pendingLoads.values()]) rejectLoad(load, err);
  for (const run of [...pendingRuns.values()]) rejectRun(run, err);
}

// Restart the whole pool (a run may never settle — terminate instead of
// waiting), drop every pending operation, and force the next spawn to
// respawn clean.
export function restartWorker(): void {
  teardownPool();
  rejectAllPending(new Error('Inference worker was terminated'));
}

function handleWorkerMessage(message: WorkerOutboundMessage, workerIndex: number): void {
  switch (message.type) {
    case 'load-progress': {
      // Relay progress only from worker 0 — secondary workers re-fetch the
      // same bytes from the HTTP cache and their progress would jitter.
      if (workerIndex !== 0) return;
      const load = pendingLoads.get(message.modelId);
      if (!load) return;
      for (const waiter of load.waiters) {
        waiter.onProgress?.(message.loaded, message.total);
      }
      break;
    }
    case 'load-message': {
      if (workerIndex !== 0) return;
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
      load.loadedWorkers++;
      if (load.loadedWorkers < workers.length) return; // wait for the rest of the pool
      const entry: LoadedModel = { backend: message.backend, dtype: message.dtype, epoch: poolGeneration };
      loadedModels.set(message.modelId, entry);
      pendingLoads.delete(message.modelId);
      clearTimeout(load.timer);
      console.log(`[upscale-client] model loaded: ${message.modelId} (${entry.backend} ${entry.dtype}, ${load.loadedWorkers} worker(s))`);
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
      console.error(`[upscale-client] model load failed: ${message.modelId} — ${message.message}`);
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

const LOAD_TIMEOUT_MS = 180000;

function beginLoad(id: ModelId, opts?: EnsureModelOptions): Promise<LoadedModel> {
  console.log(`[upscale-client] beginLoad(${id})`);
  const load: InFlightLoad = { modelId: id, waiters: new Set(), timer: undefined, loadedWorkers: 0 };
  pendingLoads.set(id, load);

  // The load phase must never stall silently: if no worker posts either
  // 'loaded' nor 'load-error' in time, terminate the pool, fail every
  // waiter, and let the next use respawn fresh workers.
  load.timer = setTimeout(() => {
    if (pendingLoads.get(id) !== load) return;
    rejectLoad(load, new Error('Model load timed out — check your connection or refresh'));
    restartWorker();
  }, LOAD_TIMEOUT_MS);

  try {
    ensurePool(id);
    console.log(`[upscale-client] posting load: ${id} to ${workers.length} worker(s)`);
    for (const worker of workers) {
      worker.postMessage({ type: 'load', modelId: id });
    }
  } catch (err) {
    rejectLoad(load, err);
    return Promise.reject(err);
  }

  return joinLoad(id, load, opts);
}

export function ensureModel(id: ModelId, opts?: EnsureModelOptions): Promise<LoadedModel> {
  const cached = loadedModels.get(id);
  if (cached && cached.epoch === poolGeneration) return Promise.resolve(cached);

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
  const timeoutMs = opts?.timeoutMs ?? 300000;
  // Rebuild the pool if it is sized for a different model tier (switching
  // models is rare, so keeping per-tier pools is not worth the memory).
  // A rebuild discards every session, so re-ensure the model afterwards:
  // this is a no-op when the pool was not rebuilt (cached entry, matching
  // epoch) and loads the model on the fresh pool otherwise.
  ensurePool(modelId);
  await ensureModel(modelId);
  const runId = ++runIdCounter;
  const worker = workers[runId % workers.length];

  return new Promise<Float32Array>((resolve, reject) => {
    // Always transfer a copy: the caller may reuse chw after this resolves
    // (TTA re-runs the same tile under the 8 transforms), and transferring
    // the buffer in place would detach it.
    const data = new Float32Array(chw);
    const buffer = data.buffer;

    const run: PendingRun = {
      runId,
      timer: undefined,
      cleanup: () => {},
      resolve,
      reject,
    };
    pendingRuns.set(runId, run);

    run.timer = setTimeout(() => {
      pendingRuns.delete(runId);
      run.cleanup();
      restartWorker();
      reject(
        new InferenceTimeoutError(`Inference timed out after ${timeoutMs / 1000}s`)
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
      // Worker gone — reject and rebuild the pool on next use.
      rejectRun(run, err);
      teardownPool();
    }
  });
}

export function disposeAll(): void {
  restartWorker();
  loadedModels.clear();
  pendingLoads.clear();
  pendingRuns.clear();
}
