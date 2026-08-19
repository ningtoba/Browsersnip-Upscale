import * as ort from 'onnxruntime-web';
import { MODEL_BY_ID } from '@/lib/constants';
import type { ModelId } from '@/types';

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'webgpu' | 'wasm';
}

const sessions = new Map<ModelId, SessionEntry>();
const loading = new Map<ModelId, Promise<void>>();
let generation = 0;

const threadCount = crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 4)
  : 1;
ort.env.wasm.numThreads = threadCount;

export function hasWebGPU(): boolean {
  return typeof navigator.gpu !== 'undefined';
}

async function fetchModelBuffer(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(
      `Model file not found at ${url}. Place the ONNX model in public/models/.`
    );
  }
  return response.arrayBuffer();
}

async function createSession(
  modelBuffer: ArrayBuffer
): Promise<SessionEntry> {
  try {
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    return { session, backend: 'webgpu' };
  } catch {
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      intraOpNumThreads: threadCount,
    });
    return { session, backend: 'wasm' };
  }
}

export async function initSession(id: ModelId, signal?: AbortSignal): Promise<void> {
  if (sessions.has(id)) return;

  const inFlight = loading.get(id);
  if (inFlight) {
    try {
      await inFlight;
      return;
    } catch {
      // Previous load failed — retry below.
    }
  }

  const config = MODEL_BY_ID[id];
  const gen = generation;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });

  const promise = (async () => {
    const buffer = await fetchModelBuffer(config.url, controller.signal);
    if (gen !== generation) return; // disposed while loading

    const entry = await createSession(buffer);
    if (gen !== generation) {
      void entry.session.release().catch(() => {});
      return;
    }

    sessions.set(id, entry);
  })();

  loading.set(id, promise);

  try {
    await promise;
  } finally {
    loading.delete(id);
    signal?.removeEventListener('abort', onAbort);
  }
}

export function getSession(id: ModelId): ort.InferenceSession | null {
  return sessions.get(id)?.session ?? null;
}

export function getBackend(id: ModelId): 'webgpu' | 'wasm' | null {
  return sessions.get(id)?.backend ?? null;
}

export function getModelScale(id: ModelId): number {
  return MODEL_BY_ID[id].scale;
}

export async function disposeAll(): Promise<void> {
  generation++;
  const entries = [...sessions.values()];
  sessions.clear();
  loading.clear();
  await Promise.all(
    entries.map((entry) => entry.session.release().catch(() => {}))
  );
}
