import { useCallback, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { decodeImageFile, getImageData, hasAlphaChannel } from '@/lib/upscale/preprocess';
import { upscaleImage } from '@/lib/upscale/upscale';
import {
  floatCHWToImageData,
  upscaleAlpha,
  canvasFromImageData,
  resizeCanvas,
  encodeCanvas,
} from '@/lib/upscale/postprocess';
import {
  ensureModel,
  getBackend,
  getDtype,
  hasWebGPU,
  isModelLoaded,
  runTile,
  forceWasmFallback,
} from '@/lib/engine/client';
import type { UpscaleInput, UpscaleOutput } from '@/types';
import { MODELS, DEFAULT_TILE, TILE_OVERLAP } from '@/lib/constants';

export function usePipeline() {
  const abortRef = useRef<AbortController | null>(null);

  const setPhase = useProcessStore((s) => s.setPhase);
  const updateProgress = useProcessStore((s) => s.updateProgress);
  const startProcessing = useProcessStore((s) => s.startProcessing);
  const setOutput = useProcessStore((s) => s.setOutput);
  const setError = useProcessStore((s) => s.setError);
  const appendLog = useProcessStore((s) => s.appendLog);

  const startUpscale = useCallback(async () => {
    const file = useFileStore.getState().file;
    const state = useProcessStore.getState();
    if (!file || state.isProcessing) return;

    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    try {
      startProcessing();
      const options = state.options;

      // Lazy model loading: download on first use (skipped when the model
      // is already cached, e.g. prefetched by the app shell).
      if (!isModelLoaded(options.model)) {
        const model = MODELS.find((m) => m.id === options.model) ?? MODELS[0];
        setPhase('loading-model');
        appendLog(`Downloading ${model.label} model...`);
        await ensureModel(options.model, {
          signal,
          onProgress: (loaded, total) => {
            const pct = total > 0 ? (loaded / total) * 100 : 0;
            updateProgress({
              phaseDescription: `Downloading ${model.label} model (${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB)...`,
              phasePercent: pct,
              overallPercent: (15 * pct) / 100,
              detail: `${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`,
            });
          },
          // Warm-up status (GPU kernel compilation) overrides the download
          // description while it runs.
          onMessage: (msg) => {
            updateProgress({ phaseDescription: msg });
          },
        });
        if (signal.aborted) {
          useProcessStore.getState().reset();
          return;
        }
      }

      // Resolve the backend AFTER loading so logs and tile defaults reflect
      // the session that will actually run (webgpu load may have fallen back
      // to wasm).
      const backend = getBackend(options.model) ?? (hasWebGPU() ? 'webgpu' : 'wasm');
      const tileSize =
        options.tileSize === 'auto'
          ? DEFAULT_TILE[options.model][backend]
          : Number(options.tileSize);

      // Decode the image and fill in metadata if the dropzone/paste path
      // didn't already probe it.
      const decoded = await decodeImageFile(file);
      const imageData = getImageData(decoded.bitmap);
      const w = imageData.width;
      const h = imageData.height;
      const hasAlpha = decoded.hasAlpha;

      if (!useFileStore.getState().metadata) {
        useFileStore.getState().setMetadata({
          width: w,
          height: h,
          fileSize: file.size,
          fileName: file.name,
          hasAlpha: hasAlphaChannel(imageData.data),
        });
      }

      const model = MODELS.find((m) => m.id === options.model) ?? MODELS[0];
      setPhase('upscaling');
      appendLog(`Backend: ${backend} (${getDtype(options.model) ?? 'fp32'}, ${tileSize}px tiles, overlap ${TILE_OVERLAP}px)`);
      appendLog(`Model: ${model.label} — upscaling ${w}x${h} → ${w * 4}x${h * 4}`);

      const tileCount = Math.ceil(w / tileSize) * Math.ceil(h / tileSize);
      const ttaFactor = options.tta ? 8 : 1;
      appendLog(`Starting inference — ${tileCount} tile(s), TTA ${options.tta ? 'on (8x)' : 'off'}`);

      const input: UpscaleInput = { data: imageData.data, width: w, height: h };

      // Any WebGPU run failure (error or timeout) is retried once on WASM:
      // terminate the worker, force a WASM reload, restart the tiling.
      let fellBack = false;
      const runWithFallback = async (): Promise<UpscaleOutput> => {
        try {
          return await upscaleImage(input, {
            tileSize,
            overlap: TILE_OVERLAP,
            tta: options.tta,
            signal,
            onProgress: (done: number, total: number) => {
              const phasePercent = (done / total) * 100;
              updateProgress({
                phaseDescription: 'Upscaling image with AI...',
                phasePercent,
                overallPercent: 15 + (75 * phasePercent) / 100,
                detail: `Tile ${Math.floor(done / ttaFactor) + 1}/${tileCount}`,
              });
            },
            run: (chw, h, w) =>
              runTile(options.model, chw, h, w, {
                signal,
                timeoutMs: getBackend(options.model) === 'webgpu' ? 60000 : 300000,
              }),
          });
        } catch (err) {
          const isWebGpu = getBackend(options.model) === 'webgpu';
          if (isWebGpu && !fellBack) {
            // ANY webgpu run failure (error or timeout) → wasm retry
            fellBack = true;
            appendLog(
              'WebGPU inference failed or timed out — retrying on CPU (WASM). This is slower but reliable.'
            );
            forceWasmFallback(options.model);
            await ensureModel(options.model, { signal }); // respawn + wasm load (browser-cached fetch)
            if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
            setPhase('upscaling'); // keep progress UI coherent
            return runWithFallback();
          }
          throw err;
        }
      };

      const result = await runWithFallback();

      const outW = result.width;
      const outH = result.height;
      setPhase('postprocessing');
      updateProgress({
        phasePercent: 0,
        overallPercent: 90,
        detail: 'Encoding output image...',
      });

      const outImageData = floatCHWToImageData(result.data, outW, outH);

      if (hasAlpha) {
        const alphaData = new Uint8ClampedArray((w * h));
        for (let i = 0; i < w * h; i++) {
          alphaData[i] = imageData.data[i * 4 + 3];
        }
        const alpha = upscaleAlpha(alphaData, w, h, outW, outH);
        const pixels = outImageData.data.length / 4;
        for (let i = 0; i < pixels; i++) {
          outImageData.data[i * 4 + 3] = alpha[i];
        }
      }

      let canvas = canvasFromImageData(outImageData);
      if (options.scale === '2' || options.scale === '3') {
        canvas = resizeCanvas(canvas, Math.round(w * Number(options.scale)), Math.round(h * Number(options.scale)));
      } else if (options.scale === 'custom' && options.customWidth > 0) {
        canvas = resizeCanvas(canvas, options.customWidth, Math.round((h * options.customWidth) / w));
      }

      const blob = await encodeCanvas(canvas, options.format, options.quality);
      const stem = file.name.replace(/\.[^.]+$/, '');
      const scalePart = options.scale === 'custom' ? `${options.customWidth}px` : `${options.scale}x`;
      const filename = `${stem}_upscayl_${scalePart}_${options.model}.${options.format}`;

      setOutput(blob, URL.createObjectURL(blob));
      appendLog(`Done — ${(blob.size / 1048576).toFixed(1)} MB`);
      setPhase('done');
      updateProgress({ phasePercent: 100, overallPercent: 100 });
    } catch (err) {
      if (signal.aborted) {
        useProcessStore.getState().reset();
        return;
      }
      const message = err instanceof Error ? err.message : 'Upscaling failed';
      setError(message);
      appendLog(message);
    }
  }, [setPhase, updateProgress, startProcessing, setOutput, setError, appendLog]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { startUpscale, cancel };
}
