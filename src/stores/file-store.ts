import { create } from 'zustand';
import type { ImageMetadata } from '@/types';

const LARGE_IMAGE_THRESHOLD_PX = 16_000_000;

interface FileState {
  file: File | null;
  metadata: ImageMetadata | null;
  previewUrl: string | null;
  isLargeImage: boolean;

  setFile: (file: File | null) => void;
  setMetadata: (meta: ImageMetadata | null) => void;
  reset: () => void;
}

export const useFileStore = create<FileState>((set, get) => ({
  file: null,
  metadata: null,
  previewUrl: null,
  isLargeImage: false,

  setFile: (file) => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      file,
      metadata: null,
      isLargeImage: false,
      previewUrl: file ? URL.createObjectURL(file) : null,
    });

    if (file) {
      // Probe real dimensions without blocking; the pipeline's decode will
      // also refresh isLargeImage via setMetadata.
      void probeDimensions(file)
        .then(({ width, height }) => {
          if (get().file !== file) return;
          set({ isLargeImage: width * height > LARGE_IMAGE_THRESHOLD_PX });
        })
        .catch(() => {
          // Best-effort probe — isLargeImage stays false until setMetadata.
        });
    }
  },

  setMetadata: (meta) =>
    set((s) => ({
      metadata: meta,
      isLargeImage: meta
        ? meta.width * meta.height > LARGE_IMAGE_THRESHOLD_PX
        : s.isLargeImage,
    })),

  reset: () => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      file: null,
      metadata: null,
      previewUrl: null,
      isLargeImage: false,
    });
  },
}));

async function probeDimensions(file: File): Promise<{ width: number; height: number }> {
  const bitmap = await createImageBitmap(file);
  const { width, height } = bitmap;
  bitmap.close();
  return { width, height };
}
