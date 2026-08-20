import type { ModelConfig, ModelId, PipelinePhase } from '@/types';

export const MODELS: ModelConfig[] = [
  {
    id: 'realesr-general-x4v3',
    label: 'Fast',
    description: 'Real-ESRGAN general-x4v3 — compact model, ~8x faster, great all-round quality',
    url: '/models/realesr-general-x4v3.onnx',
    sizeMB: 4.7,
    scale: 4,
  },
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

export const DEFAULT_TILE: Record<ModelId, number> = {
  'realesr-general-x4v3': 256,
  'realesrgan-x4plus': 128,
  'realesrgan-x4plus-anime': 256,
  'realesr-animevideov3': 256,
};

export const MODEL_TIER: Record<ModelId, 'compact' | 'standard'> = {
  'realesr-general-x4v3': 'compact',
  'realesrgan-x4plus': 'standard',
  'realesrgan-x4plus-anime': 'standard',
  'realesr-animevideov3': 'compact',
};

export const TILE_CHOICES: number[] = [128, 256, 512];

export const PHASE_WEIGHTS: Record<PipelinePhase, number> = {
  idle: 0,
  'loading-model': 15,
  upscaling: 75,
  postprocessing: 10,
  done: 0,
};

export const PHASE_DESCRIPTIONS: Record<PipelinePhase, string> = {
  idle: '',
  'loading-model': 'Downloading AI model...',
  upscaling: 'Upscaling image with AI...',
  postprocessing: 'Encoding output image...',
  done: 'Done',
};
