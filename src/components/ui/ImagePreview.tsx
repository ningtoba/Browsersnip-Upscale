import { useFileStore } from '@/stores/file-store';

export function ImagePreview() {
  const previewUrl = useFileStore((s) => s.previewUrl);
  const metadata = useFileStore((s) => s.metadata);

  if (!previewUrl || !metadata) return null;

  return (
    <div className="doodle-section space-y-3 animate-fade-in">
      <div className="sketch-border">
        <img
          src={previewUrl}
          alt="Input image"
          className="w-full max-h-[300px] object-contain bg-black"
        />
      </div>
      <div className="space-y-0.5">
        <p className="text-xs text-ink-soft font-medium">
          {metadata.width} × {metadata.height}px —{' '}
          {(metadata.fileSize / 1048576).toFixed(1)} MB
        </p>
        <p className="text-[10px] text-ink-muted">
          Upscale will produce 4× the resolution
        </p>
      </div>
    </div>
  );
}
