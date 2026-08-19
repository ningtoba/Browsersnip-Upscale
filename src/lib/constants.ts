import type { ModelConfig, ModelId, PipelinePhase } from '@/types';

export const MODELS: ModelConfig[] = [
  {
    id: 'realesrgan-x4plus',
    label: 'General',
    description: 'Real-ESRGAN x4plus — 23-block RRDBNet, best quality for photos',
    url: '/models/realesrgan-x4plus.onnx',
    sizeMB: 65.5,
    scale: 4,
  },
  {
    id: 'realesrgan-x4plus-anime',
    label: 'Anime',
    description: 'Real-ESRGAN x4plus anime 6B — tuned for illustrations and anime',
    url: '/models/realesrgan-x4plus-anime.onnx',
    sizeMB: 17.5,
    scale: 4,
  },
  {
    id: 'realesr-animevideov3',
    label: 'Anime Fast',
    description: 'Real-ESRGAN animevideov3 — compact, fastest model',
    url: '/models/realesr-animevideov3.onnx',
    sizeMB: 2.4,
    scale: 4,
  },
];

export const MODEL_BY_ID: Record<ModelId, ModelConfig> = MODELS.reduce(
  (acc, model) => {
    acc[model.id] = model;
    return acc;
  },
  {} as Record<ModelId, ModelConfig>
);

export const TILE_OVERLAP = 12; // input-px overlap between adjacent tiles

export const DEFAULT_TILE: Record<ModelId, { webgpu: number; wasm: number }> = {
  'realesrgan-x4plus': { webgpu: 256, wasm: 128 },
  'realesrgan-x4plus-anime': { webgpu: 512, wasm: 256 },
  'realesr-animevideov3': { webgpu: 512, wasm: 256 },
};

export const TILE_CHOICES: number[] = [128, 256, 512];

export const PHASE_WEIGHTS: Record<PipelinePhase, number> = {
  idle: 0,
  upscaling: 90,
  postprocessing: 10,
  done: 0,
};

export const PHASE_DESCRIPTIONS: Record<PipelinePhase, string> = {
  idle: '',
  upscaling: 'Upscaling image with AI...',
  postprocessing: 'Encoding output image...',
  done: 'Done',
};
