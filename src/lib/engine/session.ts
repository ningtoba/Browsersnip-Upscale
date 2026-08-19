import * as ort from 'onnxruntime-web';
import { MODEL_BY_ID } from '@/lib/constants';
import type { ModelId } from '@/types';

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'webgpu' | 'wasm';
  dtype: 'fp16' | 'fp32';
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

async function fetchModelBuffer(
  url: string,
  signal: AbortSignal,
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
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

export async function ensureModel(
  id: ModelId,
  opts?: { signal?: AbortSignal; onProgress?: (loaded: number, total: number) => void }
): Promise<void> {
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
  opts?.signal?.addEventListener('abort', onAbort, { once: true });

  const promise = (async () => {
    const useFp16 = await supportsFp16();
    const signal = controller.signal;
    const onProgress = opts?.onProgress;

    let entry: SessionEntry | null = null;

    if (useFp16) {
      try {
        const buffer = await fetchModelBuffer(config.urlFp16, signal, onProgress);
        if (gen !== generation) return;
        try {
          const session = await ort.InferenceSession.create(buffer, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
          });
          entry = { session, backend: 'webgpu', dtype: 'fp16' };
        } catch {
          // WebGPU create failed — fall back to fp32 below.
        }
      } catch {
        // fp16 fetch failed — fall back to fp32 below.
      }
    }

    if (!entry) {
      const buffer = await fetchModelBuffer(config.url, signal, onProgress);
      if (gen !== generation) return;
      try {
        const session = await ort.InferenceSession.create(buffer, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        entry = { session, backend: 'webgpu', dtype: 'fp32' };
      } catch {
        const session = await ort.InferenceSession.create(buffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: true,
          enableMemPattern: true,
          intraOpNumThreads: threadCount,
        });
        entry = { session, backend: 'wasm', dtype: 'fp32' };
      }
    }

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
    opts?.signal?.removeEventListener('abort', onAbort);
  }
}

export function getDtype(id: ModelId): 'fp16' | 'fp32' | null {
  return sessions.get(id)?.dtype ?? null;
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
