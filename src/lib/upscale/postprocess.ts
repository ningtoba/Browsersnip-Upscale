import type { OutputFormat } from '@/types';

export function floatCHWToImageData(
  rgb: Float32Array,
  width: number,
  height: number
): ImageData {
  const n = width * height;
  const out = new Uint8ClampedArray(n * 4);
  const plane = n;

  for (let i = 0; i < n; i++) {
    const r = rgb[i];
    const g = rgb[plane + i];
    const b = rgb[2 * plane + i];
    const o = i * 4;
    out[o] = (r < 0 ? 0 : r > 1 ? 1 : r) * 255;
    out[o + 1] = (g < 0 ? 0 : g > 1 ? 1 : g) * 255;
    out[o + 2] = (b < 0 ? 0 : b > 1 ? 1 : b) * 255;
    out[o + 3] = 255;
  }

  return new ImageData(out, width, height);
}

export function upscaleAlpha(
  alpha: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8ClampedArray {
  // Pack alpha as a grayscale RGBA image (mask in RGB *and* alpha so the
  // drawn result's alpha channel carries the interpolated mask), upscale it
  // with high-quality smoothing, then read the alpha channel back.
  const src = new Uint8ClampedArray(srcW * srcH * 4);
  for (let i = 0; i < alpha.length; i++) {
    const o = i * 4;
    src[o] = alpha[i];
    src[o + 1] = alpha[i];
    src[o + 2] = alpha[i];
    src[o + 3] = alpha[i];
  }

  const srcCanvas = new OffscreenCanvas(srcW, srcH);
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) throw new Error('Unable to get 2D context');
  srcCtx.putImageData(new ImageData(src, srcW, srcH), 0, 0);

  const dstCanvas = new OffscreenCanvas(dstW, dstH);
  const dstCtx = dstCanvas.getContext('2d');
  if (!dstCtx) throw new Error('Unable to get 2D context');
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  dstCtx.drawImage(srcCanvas, 0, 0, srcW, srcH, 0, 0, dstW, dstH);

  const sampled = dstCtx.getImageData(0, 0, dstW, dstH).data;
  const result = new Uint8ClampedArray(dstW * dstH);
  for (let i = 0; i < result.length; i++) {
    result[i] = sampled[i * 4 + 3];
  }
  return result;
}

export function canvasFromImageData(imageData: ImageData): OffscreenCanvas {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Unable to get 2D context');
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export function resizeCanvas(
  canvas: OffscreenCanvas,
  targetW: number,
  targetH: number
): OffscreenCanvas {
  const w = Math.round(targetW);
  const h = Math.round(targetH);
  const out = new OffscreenCanvas(w, h);
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('Unable to get 2D context');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, w, h);
  return out;
}

export async function encodeCanvas(
  canvas: OffscreenCanvas,
  format: OutputFormat,
  quality: number
): Promise<Blob> {
  const type = format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
  const options: ImageEncodeOptions =
    format === 'png' ? { type } : { type, quality: quality / 100 };
  return canvas.convertToBlob(options);
}
