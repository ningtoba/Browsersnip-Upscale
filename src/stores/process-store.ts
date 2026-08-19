import { create } from 'zustand';
import type { PipelinePhase, PipelineProgress, UpscaleOptions } from '@/types';
import { PHASE_DESCRIPTIONS } from '@/lib/constants';

const DEFAULT_OPTIONS: UpscaleOptions = {
  model: 'realesrgan-x4plus',
  scale: '4',
  customWidth: 0,
  format: 'png',
  quality: 90,
  tileSize: 'auto',
  tta: false,
};

const DEFAULT_PROGRESS: PipelineProgress = {
  phase: 'idle',
  phaseDescription: '',
  phasePercent: 0,
  overallPercent: 0,
};

interface ProcessState {
  phase: PipelinePhase;
  progress: PipelineProgress;
  isProcessing: boolean;
  outputBlob: Blob | null;
  outputUrl: string | null;
  error: string | null;
  logs: string[];
  options: UpscaleOptions;

  setPhase: (phase: PipelinePhase) => void;
  updateProgress: (p: Partial<PipelineProgress>) => void;
  setOptions: (p: Partial<UpscaleOptions>) => void;
  startProcessing: () => void;
  setOutput: (blob: Blob, url: string) => void;
  setError: (msg: string) => void;
  appendLog: (line: string) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>((set, get) => ({
  phase: 'idle',
  progress: { ...DEFAULT_PROGRESS },
  isProcessing: false,
  outputBlob: null,
  outputUrl: null,
  error: null,
  logs: [],
  options: { ...DEFAULT_OPTIONS },

  setPhase: (phase) =>
    set((s) => ({
      phase,
      progress: {
        ...s.progress,
        phase,
        phaseDescription: PHASE_DESCRIPTIONS[phase],
        phasePercent: 0,
        overallPercent: 0,
      },
    })),

  updateProgress: (p) =>
    set((s) => ({
      progress: { ...s.progress, ...p },
    })),

  setOptions: (p) =>
    set((s) => ({
      options: { ...s.options, ...p },
    })),

  startProcessing: () => {
    const prev = get().outputUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      isProcessing: true,
      error: null,
      outputBlob: null,
      outputUrl: null,
      logs: [],
      progress: { ...DEFAULT_PROGRESS },
    });
  },

  setOutput: (blob, url) => {
    const prev = get().outputUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      isProcessing: false,
      phase: 'done',
      outputBlob: blob,
      outputUrl: url,
    });
  },

  setError: (msg) =>
    set({ isProcessing: false, error: msg }),

  appendLog: (line) =>
    set((s) => ({ logs: [...s.logs, line] })),

  reset: () => {
    const prev = get().outputUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      phase: 'idle',
      progress: { ...DEFAULT_PROGRESS },
      isProcessing: false,
      outputBlob: null,
      outputUrl: null,
      error: null,
      logs: [],
      options: { ...DEFAULT_OPTIONS },
    });
  },
}));
