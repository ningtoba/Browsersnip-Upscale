import type { DecodedImage } from '@/types';

const MAX_PROBE_DIM = 128;
const INV_255 = 1 / 255;

export async function decodeImageFile(file: File): Promise<DecodedImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    bitmap = await decodeViaImageElement(file);
  }

  return {
    bitmap,
    width: bitmap.width,
    height: bitmap.height,
    hasAlpha: probeAlpha(bitmap),
  };
}

async function decodeViaImageElement(file: File): Promise<ImageBitmap> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('Failed to decode image file'));
    });

    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (width === 0 || height === 0) {
      throw new Error('Decoded image has zero dimensions');
    }

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Unable to get 2D context');
    ctx.drawImage(img, 0, 0);

    try {
      return await createImageBitmap(canvas);
    } catch {
      const blob = await canvas.convertToBlob();
      return await createImageBitmap(blob);
    }
  } finally {
    URL.revokeObjectURL(url);
  }
}

function probeAlpha(bitmap: ImageBitmap): boolean {
  const maxDim = Math.max(bitmap.width, bitmap.height);
  const scale = maxDim > MAX_PROBE_DIM ? MAX_PROBE_DIM / maxDim : 1;
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;
  ctx.drawImage(bitmap, 0, 0, w, h);

  return hasAlphaChannel(ctx.getImageData(0, 0, w, h).data);
}

export function getImageData(bitmap: ImageBitmap): ImageData {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Unable to get 2D context');
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

export function hasAlphaChannel(data: Uint8ClampedArray): boolean {
  // Check every 7th pixel for speed on huge arrays.
  for (let i = 3; i < data.length; i += 28) {
    if (data[i] < 255) return true;
  }
  return false;
}

function extractRectCHW(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  x0: number,
  y0: number,
  tw: number,
  th: number
): Float32Array {
  const out = new Float32Array(tw * th * 3);
  if (imgW <= 0 || imgH <= 0) return out;

  const plane = tw * th;

  if (x0 >= 0 && y0 >= 0 && x0 + tw <= imgW && y0 + th <= imgH) {
    // Fast interior path: every pixel is in-bounds, no clamping.
    for (let r = 0; r < th; r++) {
      let s = ((y0 + r) * imgW + x0) * 4;
      const row = r * tw;
      for (let c = 0; c < tw; c++) {
        const d = row + c;
        out[d] = data[s] * INV_255;
        out[plane + d] = data[s + 1] * INV_255;
        out[2 * plane + d] = data[s + 2] * INV_255;
        s += 4;
      }
    }
  } else {
    // Clamped path: replicate edge pixels for out-of-bounds tiles.
    for (let r = 0; r < th; r++) {
      const sy = y0 + r < 0 ? 0 : y0 + r >= imgH ? imgH - 1 : y0 + r;
      const row = r * tw;
      for (let c = 0; c < tw; c++) {
        const sx = x0 + c < 0 ? 0 : x0 + c >= imgW ? imgW - 1 : x0 + c;
        const s = (sy * imgW + sx) * 4;
        const d = row + c;
        out[d] = data[s] * INV_255;
        out[plane + d] = data[s + 1] * INV_255;
        out[2 * plane + d] = data[s + 2] * INV_255;
      }
    }
  }

  return out;
}

export function extractTileCHW(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  x0: number,
  y0: number,
  size: number
): Float32Array {
  return extractRectCHW(data, imgW, imgH, x0, y0, size, size);
}

export function extractTileRectCHW(
  data: Uint8ClampedArray,
  imgW: number,
  imgH: number,
  x0: number,
  y0: number,
  tw: number,
  th: number
): Float32Array {
  return extractRectCHW(data, imgW, imgH, x0, y0, tw, th);
}
