import type { UpscaleInput, UpscaleOutput, UpscaleRunOptions } from '@/types';
import { TILE_OVERLAP } from '@/lib/constants';
import { extractTileRectCHW } from '@/lib/upscale/preprocess';

const MODEL_SCALE = 4;
const CHANNELS = 3;

// The 8 symmetries of the square: { r: clockwise 90° rotations,
// f: horizontal flip applied after rotation }.
const TTA_TRANSFORMS = [
  { r: 0, f: 0 }, // identity
  { r: 0, f: 1 }, // hflip
  { r: 2, f: 1 }, // vflip (= rot180 + hflip)
  { r: 1, f: 0 }, // rot90 cw
  { r: 2, f: 0 }, // rot180
  { r: 3, f: 0 }, // rot270
  { r: 1, f: 1 }, // rot90 + hflip
  { r: 3, f: 1 }, // rot270 + hflip
];

// Rotate a CHW image 90° clockwise: (C,H,W) -> (C,W,H).
// dst[c][x][y] = src[c][H-1-y][x]
function rotateCW(src: Float32Array, h: number, w: number): Float32Array {
  const plane = h * w;
  const dst = new Float32Array(src.length);
  for (let p = 0; p < CHANNELS; p++) {
    const sp = p * plane;
    for (let x = 0; x < w; x++) {
      const dstRow = sp + x * h;
      const srcCol = sp + x;
      for (let y = 0; y < h; y++) {
        dst[dstRow + y] = src[srcCol + (h - 1 - y) * w];
      }
    }
  }
  return dst;
}

// Horizontally flip a CHW image: (C,H,W) -> (C,H,W).
function flipH(src: Float32Array, h: number, w: number): Float32Array {
  const plane = h * w;
  const dst = new Float32Array(src.length);
  for (let p = 0; p < CHANNELS; p++) {
    const sp = p * plane;
    for (let y = 0; y < h; y++) {
      const row = sp + y * w;
      for (let x = 0; x < w; x++) {
        dst[row + (w - 1 - x)] = src[row + x];
      }
    }
  }
  return dst;
}

// Per-pixel feather weight along one axis of a tile: linear ramp from the
// edges up to a plateau of 1. The minimum weight is clamped to 1/featherPx
// so image-border pixels (covered by exactly one tile) keep weight > 0.
function edgeWeights(len: number, featherPx: number): Float32Array {
  const w = new Float32Array(len);
  for (let d = 0; d < len; d++) {
    let e = Math.min(d, len - 1 - d);
    if (e > featherPx) e = featherPx;
    if (e < 1) e = 1;
    w[d] = e / featherPx;
  }
  return w;
}

export async function upscaleImage(
  input: UpscaleInput,
  opts: UpscaleRunOptions & { run: (chw: Float32Array, h: number, w: number) => Promise<Float32Array> }
): Promise<UpscaleOutput> {
  if (input.width === 0 || input.height === 0) {
    throw new Error('Cannot upscale an empty image');
  }

  const overlap = opts.overlap ?? TILE_OVERLAP;
  const tileSize = opts.tileSize;
  const tta = opts.tta ?? false;
  const stride = Math.max(1, tileSize - overlap);
  const runsPerTile = tta ? TTA_TRANSFORMS.length : 1;

  // Tile grid: edge tiles may be smaller than tileSize — no padding,
  // the model has dynamic H/W.
  const xs: number[] = [];
  for (let x = 0; x < input.width; x += stride) xs.push(x);
  const ys: number[] = [];
  for (let y = 0; y < input.height; y += stride) ys.push(y);
  const tileCount = xs.length * ys.length;
  const totalRuns = tileCount * runsPerTile;

  const outW = input.width * MODEL_SCALE;
  const outH = input.height * MODEL_SCALE;
  const out = new Float32Array(CHANNELS * outW * outH);
  const weights = new Float32Array(outW * outH);
  const planeSize = outW * outH;
  const feather = Math.max(1, overlap * MODEL_SCALE);

  let done = 0;

  const tick = (): void => {
    done++;
    if (opts.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    opts.onProgress?.(
      done,
      totalRuns,
      `Tile ${Math.min(Math.floor(done / runsPerTile) + 1, tileCount)}/${tileCount}`
    );
  };

  const accumulate = (
    tileOut: Float32Array,
    th: number,
    tw: number,
    wx: Float32Array,
    wy: Float32Array,
    baseY: number,
    baseX: number
  ): void => {
    const th4 = th * MODEL_SCALE;
    const tw4 = tw * MODEL_SCALE;
    const tPlane = th4 * tw4;
    for (let dy = 0; dy < th4; dy++) {
      const wyv = wy[dy];
      const rowBase = (baseY + dy) * outW + baseX;
      const tRowBase = dy * tw4;
      for (let dx = 0; dx < tw4; dx++) {
        const wv = wx[dx] < wyv ? wx[dx] : wyv;
        const oi = rowBase + dx;
        const ti = tRowBase + dx;
        weights[oi] += wv;
        out[oi] += tileOut[ti] * wv;
        out[planeSize + oi] += tileOut[tPlane + ti] * wv;
        out[2 * planeSize + oi] += tileOut[2 * tPlane + ti] * wv;
      }
    }
  };

  for (const y0 of ys) {
    for (const x0 of xs) {
      const tw = Math.min(tileSize, input.width - x0);
      const th = Math.min(tileSize, input.height - y0);
      const chw = extractTileRectCHW(input.data, input.width, input.height, x0, y0, tw, th);
      const wx = edgeWeights(tw * MODEL_SCALE, feather);
      const wy = edgeWeights(th * MODEL_SCALE, feather);
      const baseX = x0 * MODEL_SCALE;
      const baseY = y0 * MODEL_SCALE;

      if (!tta) {
        const raw = await opts.run(chw, th, tw);
        accumulate(raw, th, tw, wx, wy, baseY, baseX);
        tick();
        continue;
      }

      for (const t of TTA_TRANSFORMS) {
        // Transform the input tile.
        let tIn = chw;
        let tH = th;
        let tW = tw;
        for (let i = 0; i < t.r; i++) {
          tIn = rotateCW(tIn, tH, tW);
          const tmp = tH;
          tH = tW;
          tW = tmp;
        }
        if (t.f) tIn = flipH(tIn, tH, tW);

        const raw = await opts.run(tIn, tH, tW);

        // Inverse-transform the output back to the tile's orientation:
        // hflip first, then rotate counter-clockwise ((4 - r) cw turns).
        let tOut = raw;
        let oH = tH * MODEL_SCALE;
        let oW = tW * MODEL_SCALE;
        if (t.f) tOut = flipH(tOut, oH, oW);
        for (let i = 0; i < (4 - t.r) % 4; i++) {
          tOut = rotateCW(tOut, oH, oW);
          const tmp = oH;
          oH = oW;
          oW = tmp;
        }

        accumulate(tOut, th, tw, wx, wy, baseY, baseX);
        tick();
      }
    }
  }

  // Normalize by accumulated weights. Every output pixel is covered by at
  // least one tile (weights > 0), but guard division by zero anyway.
  for (let i = 0; i < planeSize; i++) {
    const w = weights[i];
    if (w > 0) {
      const inv = 1 / w;
      out[i] *= inv;
      out[planeSize + i] *= inv;
      out[2 * planeSize + i] *= inv;
    }
  }

  return { width: outW, height: outH, data: out };
}
