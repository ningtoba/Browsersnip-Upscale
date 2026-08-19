import { useCallback, useState, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { decodeImageFile, getImageData, hasAlphaChannel } from '@/lib/upscale/preprocess';
import type { ImageMetadata } from '@/types';

/**
 * Shared entry point for loading an image file into the app: validates the
 * type, probes dimensions/alpha, and populates the file store. Used by the
 * dropzone and by App's clipboard paste handler.
 */
export async function handleImageFile(file: File): Promise<void> {
  if (!file.type.startsWith('image/')) {
    throw new Error('Please upload an image file.');
  }

  const decoded = await decodeImageFile(file);
  const imageData = getImageData(decoded.bitmap);

  const metadata: ImageMetadata = {
    width: decoded.width,
    height: decoded.height,
    fileSize: file.size,
    fileName: file.name,
    hasAlpha: hasAlphaChannel(imageData.data),
  };

  useFileStore.getState().setFile(file);
  useFileStore.getState().setMetadata(metadata);
}

export function FileDropZone() {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);

    try {
      await handleImageFile(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load image');
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="space-y-4 animate-fade-in">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative rounded-doodle-lg border-2 border-dashed p-10 text-center
          cursor-pointer transition-all duration-200 select-none
          ${
            dragOver
              ? 'border-accent bg-accent/5 scale-[1.01]'
              : 'border-cream-border hover:border-cream-border/60'
          }
        `}
      >
        <div className="space-y-3">
          <div className="w-12 h-12 mx-auto rounded-full bg-accent/10 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-accent"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-ink-soft">
              Drop an image to upscale
            </p>
            <p className="text-[11px] text-ink-muted mt-1">
              PNG, JPEG, WebP — any size
            </p>
          </div>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
      </div>

      {error && (
        <div className="rounded-md p-3 text-xs border border-danger/20 bg-danger/5 text-danger animate-slide-up">
          {error}
        </div>
      )}
    </div>
  );
}
