import { useProcessStore } from '@/stores/process-store';
import { MODELS, TILE_CHOICES } from '@/lib/constants';
import { usePipeline } from '@/hooks/usePipeline';
import type { OutputFormat, ScaleOption, TileOption } from '@/types';

const SCALE_CHOICES: { label: string; value: ScaleOption }[] = [
  { label: '2x', value: '2' },
  { label: '3x', value: '3' },
  { label: '4x', value: '4' },
  { label: 'Custom', value: 'custom' },
];

const FORMAT_CHOICES: { label: string; value: OutputFormat }[] = [
  { label: 'PNG', value: 'png' },
  { label: 'JPEG', value: 'jpeg' },
  { label: 'WebP', value: 'webp' },
];

const TILE_OPTIONS: { label: string; value: TileOption }[] = [
  { label: 'Auto', value: 'auto' },
  ...TILE_CHOICES.map((size) => ({
    label: String(size),
    value: String(size) as TileOption,
  })),
];

export function OptionsPanel() {
  const options = useProcessStore((s) => s.options);
  const setOptions = useProcessStore((s) => s.setOptions);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const { startUpscale } = usePipeline();

  const chipClass = (active: boolean) =>
    `doodle-chip ${
      active ? 'doodle-chip-active' : 'doodle-chip-inactive'
    } disabled:opacity-40 disabled:cursor-not-allowed`;

  return (
    <div className="doodle-section space-y-5 animate-fade-in">
      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-soft">Model</p>
        <div className="flex flex-wrap gap-2">
          {MODELS.map((m) => (
            <button
              key={m.id}
              disabled={isProcessing}
              onClick={() => setOptions({ model: m.id })}
              className={chipClass(options.model === m.id)}
            >
              {m.label} ({m.sizeMB} MB)
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-soft">Output Scale</p>
        <div className="flex gap-2">
          {SCALE_CHOICES.map((c) => (
            <button
              key={c.value}
              disabled={isProcessing}
              onClick={() => setOptions({ scale: c.value })}
              className={chipClass(options.scale === c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
        {options.scale === 'custom' && (
          <input
            type="number"
            min={64}
            max={8192}
            value={options.customWidth || ''}
            disabled={isProcessing}
            onChange={(e) =>
              setOptions({
                customWidth: Math.min(
                  8192,
                  Math.max(0, Number(e.target.value) || 0)
                ),
              })
            }
            placeholder="Output width in pixels (64–8192)"
            className="doodle-input"
          />
        )}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-soft">Format</p>
        <div className="flex gap-2">
          {FORMAT_CHOICES.map((c) => (
            <button
              key={c.value}
              disabled={isProcessing}
              onClick={() => setOptions({ format: c.value })}
              className={chipClass(options.format === c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-soft">
          Quality — {options.quality}%
        </p>
        <input
          type="range"
          min={1}
          max={100}
          value={options.quality}
          disabled={isProcessing || options.format === 'png'}
          onChange={(e) => setOptions({ quality: Number(e.target.value) })}
          className="w-full disabled:opacity-40"
        />
        {options.format === 'png' && (
          <p className="text-[10px] text-ink-muted">
            Lossless — quality ignored for PNG
          </p>
        )}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-soft">Tile Size</p>
        <div className="flex gap-2">
          {TILE_OPTIONS.map((t) => (
            <button
              key={t.value}
              disabled={isProcessing}
              onClick={() => setOptions({ tileSize: t.value })}
              className={chipClass(options.tileSize === t.value)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-ink-muted">
          Smaller tiles use less memory, larger tiles are faster
        </p>
      </div>

      <div className="space-y-2">
        <button
          disabled={isProcessing}
          onClick={() => setOptions({ tta: !options.tta })}
          className={chipClass(options.tta)}
        >
          TTA — test-time augmentation (8× slower, slightly better detail)
        </button>
      </div>

      <button
        onClick={() => startUpscale()}
        disabled={isProcessing}
        className="doodle-btn"
      >
        {isProcessing ? 'Upscaling...' : 'Upscale Image'}
      </button>
    </div>
  );
}
