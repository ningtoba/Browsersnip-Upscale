export type ModelId =
  | 'realesrgan-x4plus'
  | 'realesrgan-x4plus-anime'
  | 'realesr-animevideov3'
  | 'realesr-general-x4v3';

export type OutputFormat = 'png' | 'jpeg' | 'webp';
export type ScaleOption = '2' | '3' | '4' | 'custom';
export type TileOption = 'auto' | '128' | '256' | '512';
export type PipelinePhase = 'idle' | 'loading-model' | 'upscaling' | 'postprocessing' | 'done';

export interface ImageMetadata {
  width: number;
  height: number;
  fileSize: number;
  fileName: string;
  hasAlpha: boolean;
}

export interface UpscaleOptions {
  model: ModelId;
  scale: ScaleOption;
  customWidth: number;
  format: OutputFormat;
  quality: number;
  tileSize: TileOption;
  tta: boolean;
}

export interface PipelineProgress {
  phase: PipelinePhase;
  phaseDescription: string;
  phasePercent: number;
  overallPercent: number;
  detail?: string;
}

export interface ModelConfig {
  id: ModelId;
  label: string;
  description: string;
  url: string;
  sizeMB: number;
  scale: number;
}

export interface DecodedImage {
  bitmap: ImageBitmap;
  width: number;
  height: number;
  hasAlpha: boolean;
}

export interface UpscaleInput {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface UpscaleRunOptions {
  tileSize: number;
  overlap?: number;
  tta?: boolean;
  /** Max tiles in flight concurrently (default 2). */
  maxConcurrency?: number;
  onProgress?: (done: number, total: number, detail?: string) => void;
  signal?: AbortSignal;
}

export interface UpscaleOutput {
  width: number;
  height: number;
  data: Float32Array; // CHW RGB [0,1]
}
