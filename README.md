# BrowserSnip Upscale

AI image upscaling, 100% in the browser. BrowserSnip Upscale runs the
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model family locally in
your browser with [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
(WASM/CPU inference) — your images never leave your machine. No
uploads, no servers, no accounts.

BrowserSnip Upscale is part of the [BrowserSnip](https://github.com/ningtoba/BrowserSnip)
family of client-side browser tools, sharing its design system and
architecture with BrowserSnip Face Blur. The upscale method and model family
are modeled on [Upscayl](https://github.com/upscayl/upscayl); see
[Credits](#credits) for details.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Models](#models)
- [Browser Requirements](#browser-requirements)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Reproducing the ONNX Exports](#reproducing-the-onnx-exports)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Credits](#credits)
- [License](#license)

## Features

- **Fully client-side inference.** Real-ESRGAN super-resolution runs in the
  browser via ONNX Runtime Web on the WASM (CPU) backend — no GPU required, so
  it works in every modern browser.
- **Parallel inference.** Tiles are processed concurrently across a small
  Web Worker pool (up to 4 workers for compact models, 2 for the big ones),
  so CPU cores are used instead of one worker serializing every tile.
- **Lazy model loading.** Models no longer download at page load — the app
  renders instantly. The first time a model is used it downloads with live
  progress in the UI, and the selected model is prefetched in the background
  as soon as an image is dropped. Models are fp32 only.
- **Privacy by construction.** Images are processed in memory locally. There is
  no upload step and no server to receive one.
- **Hang-proof inference.** Inference runs in a dedicated Web Worker with
  timeouts on every stage — session creation, model loading, and each tile
  run — so no phase can silently stall. A timed-out run restarts the engine
  and retries once automatically.
- **Four models** tuned for different content: a fast general-purpose model
  (the default), photos/general, anime, and a fast anime/illustration model
  (see [Models](#models)).
- **Tiled inference with overlap blending.** Large images are processed in
  tiles with 12 px overlaps blended by a linear weight ramp, keeping memory
  usage bounded. Tiles are sized per model (128 px for the 23-block
  general model; 256 px for the compact models), and edge tiles run at their
  native size thanks to the models' dynamic input shapes.
- **Optional TTA (test-time augmentation).** Average the model's output over
  all 8 flip/rotation combinations for higher quality at 8x the inference cost.
- **Flexible output sizing.** 2x, 3x, or 4x scale, or a custom output width.
  The models are 4x-only; lower scales are produced by resizing the 4x output —
  the same approach Upscayl uses for 4x-only models.
- **PNG / JPEG / WebP output** with configurable quality.
- **Transparency preserved.** The alpha channel is upscaled separately with
  high-quality canvas scaling and recombined with the upscaled RGB.
- **Upscayl-style output naming.** Files are saved as
  `<name>_upscayl_<scale>x_<model>.<ext>`.

## How It Works

The processing pipeline is:

1. **Drop an image** onto the page.
2. **Choose a model and options** — model, scale (2x/3x/4x or custom width),
   TTA, output format and quality.
3. **Tiled inference.** The image is split into overlapping tiles and run
   through the selected Real-ESRGAN model on the WASM (CPU) backend.
   Tile seams are eliminated by blending the 12 px overlap regions with linear
   weights; edge tiles are inferred at native size via the models' dynamic
   input shapes.
4. **Optional TTA.** When enabled, all 8 flip/rotation variants of each tile
   are inferred and averaged.
5. **Postprocessing.** The alpha channel is upscaled separately (high-quality
   canvas scaling) and merged with the RGB result.
6. **Resize.** The 4x model output is downscaled to the requested scale or
   custom output width.
7. **Encode and download.** The result is encoded as PNG, JPEG, or WebP and
   saved as `<name>_upscayl_<scale>x_<model>.<ext>`.

## Performance

On a typical desktop CPU, the default **Fast** model (`realesr-general-x4v3`)
runs about **~1 s per 256 px tile** while the big **General** model
(`realesrgan-x4plus`) takes **~10 s per 128 px tile**. Tiles run concurrently
across the worker pool, so a 121-tile image completes in roughly **1–2
minutes** with the default Fast model, versus **10–20 minutes** with x4plus —
pick per patience/quality: Fast for everyday upscaling, General when the best
photo quality is worth the wait.

## Models

The models are committed under `public/models/` and are exported 1:1 from the
canonical weights in the [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
releases by `scripts/export_onnx.py`.

| Model                    | Architecture       | Size    | Use case                  |
| ------------------------ | ------------------ | ------- | ------------------------- |
| `realesr-general-x4v3`   | SRVGGNetCompact (32 conv) | 4.7 MB | Fast general — default model |
| `realesrgan-x4plus`      | RRDBNet, 23 blocks | 68.7 MB | Photos, general content  |
| `realesrgan-x4plus-anime` | RRDBNet, 6 blocks | 18.4 MB | Anime, illustrations     |
| `realesr-animevideov3`   | SRVGGNetCompact    | 2.5 MB  | Fast anime / illustrations |

All models share the same contract:

- 4x scale, fp32, ONNX opset 18
- Dynamic input `[1, 3, H, W]` → output `[1, 3, 4H, 4W]`
- RGB in `[0, 1]`, no normalization

Exports are verified by a torch-vs-onnxruntime parity check (PSNR > 130 dB).

## Browser Requirements

- **Any modern browser.** Inference runs on the WASM (CPU) execution provider
  of ONNX Runtime Web — no WebGPU and no GPU hardware are required. Each
  worker runs WASM single-threaded (the same configuration the sibling
  BrowserSnip Face Blur project ships, for maximum device compatibility);
  tiles are parallelized across a small pool of workers instead.
- **Cross-origin isolation** (COOP/COEP) is enabled by `coi-serviceworker.js`
  and the Vite dev server headers; it is not required for single-threaded
  inference but keeps the door open for faster multithreaded WASM later.
- **First download ~94 MB** of ONNX models (cached by the browser afterwards),
  plus the ~13 MB ONNX Runtime WASM build fetched from its CDN.

## Quick Start

```bash
npm install
npm run dev      # start the Vite dev server
npm run build    # typecheck (tsc -b) and production build
npm run preview  # preview the production build locally
```

## Deployment

The app is a static site. A GitHub Actions workflow
(`.github/workflows/deploy.yml`) builds and publishes it to GitHub Pages —
push to the default branch (or trigger the workflow) to deploy.

The models (~94 MB total) are served from `public/models/` and are cached by
the browser, so repeat visits only fetch them once.

## Reproducing the ONNX Exports

`scripts/export_onnx.py` reproduces the ONNX models from the official PyTorch
weights. It re-implements the RRDBNet and SRVGGNetCompact architectures 1:1
from `xinntao/Real-ESRGAN` and loads the canonical weights — download them
from the `xinntao/Real-ESRGAN` GitHub releases (v0.1.0 for x4plus, v0.2.2.4 for
x4plus-anime 6B, v0.2.5.0 for animevideov3 and general-x4v3). The script
requires `torch` and `onnxruntime`:

```bash
python3 scripts/export_onnx.py --arch srvgg --num-conv 32 \
    --input realesr-general-x4v3.pth --output public/models/realesr-general-x4v3.onnx
python3 scripts/export_onnx.py --arch rrdb  --blocks 23 \
    --input RealESRGAN_x4plus.pth --output public/models/realesrgan-x4plus.onnx
python3 scripts/export_onnx.py --arch rrdb  --blocks 6 \
    --input RealESRGAN_x4plus_anime_6B.pth --output public/models/realesrgan-x4plus-anime.onnx
python3 scripts/export_onnx.py --arch srvgg \
    --input realesr-animevideov3.pth --output public/models/realesr-animevideov3.onnx
```

The script also supports `--fp16` (half-size WebGPU variants) for future GPU
work; the app itself runs fp32 only.

Each export runs a torch-vs-onnxruntime parity check and fails unless the
outputs match.

## Project Structure

```
├── public/
│   ├── models/                  # ONNX models (served statically, cached by the browser)
│   └── coi-serviceworker.js     # COOP/COEP for cross-origin isolation
├── scripts/
│   └── export_onnx.py           # Reproduces the ONNX exports from official .pth weights
├── src/
│   ├── App.tsx                  # App shell
│   ├── components/
│   │   └── ui/                  # UI components
│   ├── hooks/
│   │   └── usePipeline.ts       # Upscale pipeline orchestration
│   ├── stores/                  # zustand stores
│   ├── lib/
│   │   ├── constants.ts         # Models, tile sizes, options
│   │   ├── engine/
│   │   │   ├── client.ts         # Main-thread inference client (worker lifecycle, run timeouts, retry)
│   │   │   └── inference.worker.ts # Web Worker running onnxruntime-web (WASM)
│   │   └── upscale/
│   │       ├── preprocess.ts    # Tile extraction, normalization
│   │       ├── upscale.ts       # Tiled inference, overlap blending, TTA
│   │       └── postprocess.ts   # Alpha upscale, resize, encoding
│   └── main.tsx
├── index.html
├── vite.config.ts
└── package.json
```

## Limitations

- **GAN upscaling synthesizes plausible detail.** Real-ESRGAN invents
  high-frequency detail consistent with what it learned; it cannot recover
  detail that was destroyed by compression or downscaling.
- **Model/content mismatch looks wrong.** Anime models produce poor results on
  photos and vice versa — pick the model that matches your content.
- **Memory bounds.** Browsers impose memory limits that bound tile size on very
  large images; tiling keeps processing feasible but very large inputs may
  still be constrained.

## Credits

This project explicitly credits its influences:

- **[Upscayl](https://github.com/upscayl/upscayl)** — the upscale method and
  model family this project is modeled on (Real-ESRGAN + tiled AI inference).
  Upscayl is AGPLv3/community-licensed; this project reimplements the approach
  in the browser and uses none of Upscayl's code or binaries.
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** (xinntao, BSD-3-Clause)
  — the models and weights.
- **[ONNX Runtime Web](https://github.com/microsoft/onnxruntime)** — the
  in-browser inference engine (WASM execution provider).
- **[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)**
  (gzuidhof) — cross-origin isolation support.
- **[BrowserSnip](https://github.com/ningtoba/BrowserSnip)** and
  **BrowserSnip-Blurred** — the design system and architecture of the
  BrowserSnip project family.
- **[SceneWorks/real-esrgan-onnx](https://huggingface.co/SceneWorks/real-esrgan-onnx)**
  (BSD-3-Clause) — a reference for the dynamic-shape ONNX export approach.

## License

- **App code:** MIT.
- **Models:** BSD-3-Clause (Real-ESRGAN).
- **Upscayl is NOT vendored** — no Upscayl code is copied into this project.
