import { useState } from 'react';
import { useProcessStore } from '@/stores/process-store';
import { useFileStore } from '@/stores/file-store';
import { downloadBlob } from '@/lib/utils/download';

export function OutputActions() {
  const outputBlob = useProcessStore((s) => s.outputBlob);
  const outputUrl = useProcessStore((s) => s.outputUrl);
  const options = useProcessStore((s) => s.options);
  const file = useFileStore((s) => s.file);
  const previewUrl = useFileStore((s) => s.previewUrl);
  const metadata = useFileStore((s) => s.metadata);
  const [sliderValue, setSliderValue] = useState(50);

  if (!outputBlob || !outputUrl || !metadata) return null;

  const sizeMB = (outputBlob.size / (1024 * 1024)).toFixed(1);
  const { width, height } = metadata;

  // Mirror the pipeline's output sizing so the caption matches the result
  const scale = options.scale;
  let outputW: number;
  let outputH: number;
  if (scale === 'custom' && options.customWidth > 0) {
    outputW = options.customWidth;
    outputH = Math.round((height * options.customWidth) / width);
  } else if (scale === 'custom') {
    outputW = width * 4;
    outputH = height * 4;
  } else {
    outputW = Math.round(width * Number(scale));
    outputH = Math.round(height * Number(scale));
  }

  const originalName = file?.name ?? 'image.png';
  const stem = originalName.replace(/\.[^.]+$/, '');
  const scalePart = scale === 'custom' ? `${options.customWidth}px` : `${scale}x`;
  const outName = `${stem}_upscayl_${scalePart}_${options.model}.${options.format}`;

  return (
    <div className="space-y-4 animate-doodle-pop">
      <div className="doodle-section space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-ink">
              Upscaled Image Ready
            </h3>
            <p className="text-[11px] text-ink-muted">
              {sizeMB} MB — {outputW}×{outputH}px
            </p>
          </div>
        </div>

        <div className="sketch-border">
          <div className="relative select-none">
            <img
              src={outputUrl}
              alt="Upscaled result"
              draggable={false}
              className="w-full max-h-[300px] object-contain bg-black"
            />
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Original image"
                draggable={false}
                className="absolute inset-0 w-full h-full object-contain"
                style={{ clipPath: `inset(0 ${100 - sliderValue}% 0 0)` }}
              />
            )}
            <div
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-accent"
              style={{ left: `${sliderValue}%` }}
            />
            <span className="pointer-events-none absolute top-2 left-2 text-[10px] font-medium px-1.5 py-0.5 rounded bg-black/60 text-ink">
              Original
            </span>
            <span className="pointer-events-none absolute top-2 right-2 text-[10px] font-medium px-1.5 py-0.5 rounded bg-black/60 text-ink">
              4× AI
            </span>
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[10px] text-ink-muted">
            <span>Compare</span>
            <span className="font-mono tabular-nums">{sliderValue}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={sliderValue}
            onChange={(e) => setSliderValue(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <button
          onClick={() => downloadBlob(outputBlob, outName)}
          className="doodle-btn"
        >
          Download {outName}
        </button>
      </div>
    </div>
  );
}
