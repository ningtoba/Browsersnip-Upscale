import { useEffect } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { useUIStore } from '@/stores/ui-store';
import { FileDropZone, handleImageFile } from '@/components/ui/FileDropZone';
import { ImagePreview } from '@/components/ui/ImagePreview';
import { OptionsPanel } from '@/components/ui/OptionsPanel';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { PhaseIndicator } from '@/components/ui/PhaseIndicator';
import { OutputActions } from '@/components/ui/OutputActions';
import { MemoryWarning } from '@/components/ui/MemoryWarning';
import { LogMonitor } from '@/components/ui/LogMonitor';
import { ensureModel, isModelLoaded, disposeAll } from '@/lib/engine/client';
import { usePipeline } from '@/hooks/usePipeline';

export default function App() {
  const file = useFileStore((s) => s.file);
  const phase = useProcessStore((s) => s.phase);
  const outputBlob = useProcessStore((s) => s.outputBlob);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const showLogMonitor = useUIStore((s) => s.showLogMonitor);
  const toggleLogMonitor = useUIStore((s) => s.toggleLogMonitor);
  const model = useProcessStore((s) => s.options.model);
  const { cancel } = usePipeline();

  // Background prefetch: start the selected model's download as soon as an
  // image is dropped, while the user still inspects the options.
  useEffect(() => {
    if (!file) return;
    if (isModelLoaded(model)) return;
    void ensureModel(model).catch(() => {});
  }, [file, model]);

  // Release all inference sessions when the app unmounts.
  useEffect(
    () => () => {
      void disposeAll();
    },
    []
  );

  // Clipboard paste support: drop an image straight into the app
  useEffect(() => {
    const onPaste = async (e: ClipboardEvent) => {
      if (useFileStore.getState().file) return;
      const item = e.clipboardData?.files?.[0];
      if (!item) return;
      if (!item.type.startsWith('image/')) return;
      try {
        await handleImageFile(item);
      } catch {
        // ignore invalid pastes
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, []);

  const handleReset = () => {
    if (isProcessing) return;
    cancel();
    // Revoke any output URLs to free memory
    const outputUrl = useProcessStore.getState().outputUrl;
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    // Reset all stores
    useProcessStore.getState().reset();
    useFileStore.getState().reset();
  };

  const showOutput = phase === 'done' && outputBlob;
  const showProcessing =
    isProcessing || (phase !== 'idle' && phase !== 'done');

  return (
    <div className="flex flex-col h-screen-safe">
      <header className="h-[44px] shrink-0 flex items-center justify-between px-4 border-b border-cream-border bg-glass z-20">
        <div className="flex items-center gap-3">
          <a
            href="https://www.browsersnip.com"
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink-soft transition-colors"
            title="Back to BrowserSnip"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="hidden sm:inline">BrowserSnip</span>
          </a>
          <span className="text-cream-border">|</span>
          <span className="text-sm font-semibold tracking-wide text-ink">
            Image Upscaler
          </span>
          {phase !== 'idle' && phase !== 'done' && (
            <span className="text-[11px] text-ink-muted hidden sm:inline">
              {phase.replace(/-/g, ' ')}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {file && !isProcessing && (
            <button
              onClick={handleReset}
              className="text-xs text-ink-muted hover:text-accent transition-colors"
              title="Start over with a new image"
            >
              New Image
            </button>
          )}
          <button
            onClick={toggleLogMonitor}
            className="text-xs text-ink-muted hover:text-ink-soft transition-colors"
          >
            {showLogMonitor ? 'Hide Logs' : 'Logs'}
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
          {!file && <FileDropZone />}

          {file && (
            <>
              <MemoryWarning />
              <ImagePreview />

              {!isProcessing && phase !== 'done' && <OptionsPanel />}

              {showProcessing && (
                <div className="space-y-3 animate-fade-in">
                  <PhaseIndicator />
                  <ProgressBar />
                </div>
              )}

              {showOutput && <OutputActions />}
            </>
          )}
        </div>
      </main>

      {showLogMonitor && <LogMonitor />}
    </div>
  );
}
