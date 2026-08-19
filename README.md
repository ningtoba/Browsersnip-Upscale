# BrowserSnip Upscale

AI image upscaling, 100% in the browser. BrowserSnip Upscale runs the
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model family locally in
your browser with [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
(WebGPU with a WASM fallback) — your images never leave your machine. No
uploads, no servers, no accounts.

BrowserSnip Upscale is part of the [BrowserSnip](https://github.com/ningtoba/BrowserSnip)
family of client-side browser tools, sharing its design system and
architecture with BrowserSnip Face Blur. The upscale method and model family
are modeled on [Upscayl](https://github.com/upscayl/upscayl); see
[Credits](#credits) for details.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
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
  browser via ONNX Runtime Web. Primary execution on the WebGPU backend, with
  an automatic WASM fallback for browsers without WebGPU.
- **Lazy model loading.** Models no longer download at page load — the app
  renders instantly. The first time a model is used it downloads with live
  progress in the UI, and the selected model is prefetched in the background
  as soon as an image is dropped. WebGPU devices with fp16 support
  (`shader-f16`) download fp16 models (~34 MB General / ~9 MB Anime / ~1.3 MB
  Anime Fast); everything else uses the fp32 variants.
- **Privacy by construction.** Images are processed in memory locally. There is
  no upload step and no server to receive one.
- **Hang-proof inference.** Inference runs in a dedicated Web Worker, so the
  UI never freezes during GPU kernel compilation. A warm-up inference compiles
  the WebGPU shaders while the model loads, and WebGPU runs are timed out and
  automatically retried on the WASM backend if they fail or hang.
- **Three models** tuned for different content: photos/general, anime, and a
  fast anime/illustration model (see [Models](#models)).
- **Tiled inference with overlap blending.** Large images are processed in
  tiles with 12 px overlaps blended by a linear weight ramp, keeping memory
  usage bounded. Tiles are sized per backend (256 px WebGPU / 128 px WASM for
  the general model; 512 / 256 for the smaller models), and edge tiles run at
  their native size thanks to the models' dynamic input shapes.
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
   through the selected Real-ESRGAN model on the WebGPU (or WASM) backend.
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

## Models

The models are committed under `public/models/` and are exported 1:1 from the
canonical weights in the [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
releases by `scripts/export_onnx.py`.

| Model                    | Architecture       | Size    | WebGPU (fp16) | Use case                  |
| ------------------------ | ------------------ | ------- | ------------- | ------------------------- |
| `realesrgan-x4plus`      | RRDBNet, 23 blocks | 68.7 MB | 32.8 MB       | Photos, general content  |
| `realesrgan-x4plus-anime` | RRDBNet, 6 blocks | 18.4 MB | 8.8 MB        | Anime, illustrations     |
| `realesr-animevideov3`   | SRVGGNetCompact    | 2.5 MB  | 1.2 MB        | Fast anime / illustrations |

fp16 variants (roughly half the size) are used on WebGPU devices with fp16
support (`shader-f16`); everything else uses fp32.

All models share the same contract:

- 4x scale, fp32 or fp16, ONNX opset 18
- Dynamic input `[1, 3, H, W]` → output `[1, 3, 4H, 4W]`
- RGB in `[0, 1]`, no normalization

Exports are verified by a torch-vs-onnxruntime parity check (PSNR > 130 dB).

## Browser Requirements

- **WebGPU-capable browser (Chrome/Edge 113+) is recommended.** Inference runs
  on the GPU via the WebGPU execution provider.
- **WASM fallback** works in any modern browser but is 3–6x slower.
- **Cross-origin isolation** (COOP/COEP) is required for multithreaded WASM and
  for WebGPU. The app installs `coi-serviceworker.js`, which sets the
  `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers
  service-worker-side; the Vite dev server also sets the headers directly.

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

The models (~90 MB total) are served from `public/models/` and are cached by
the browser, so repeat visits only fetch them once.

## Reproducing the ONNX Exports

`scripts/export_onnx.py` reproduces the ONNX models from the official PyTorch
weights. It re-implements the RRDBNet and SRVGGNetCompact architectures 1:1
from `xinntao/Real-ESRGAN` and loads the canonical weights — download them
from the `xinntao/Real-ESRGAN` GitHub releases (v0.1.0 for x4plus, v0.2.2.4 for
x4plus-anime 6B, v0.2.5.0 for animevideov3). The script requires `torch` and
`onnxruntime`:

```bash
python3 scripts/export_onnx.py --arch rrdb  --blocks 23 \
    --input RealESRGAN_x4plus.pth --output public/models/realesrgan-x4plus.onnx
python3 scripts/export_onnx.py --arch rrdb  --blocks 6 \
    --input RealESRGAN_x4plus_anime_6B.pth --output public/models/realesrgan-x4plus-anime.onnx
python3 scripts/export_onnx.py --arch srvgg \
    --input realesr-animevideov3.pth --output public/models/realesr-animevideov3.onnx
```

fp16 exports for WebGPU users are produced the same way with `--fp16`:

```bash
python3 scripts/export_onnx.py --arch rrdb  --blocks 23 --fp16 \
    --input RealESRGAN_x4plus.pth --output public/models/realesrgan-x4plus-fp16.onnx
```

Each export runs a torch-vs-onnxruntime parity check and fails unless the
outputs match. fp16 fidelity is checked against the fp32 export at >45 dB
PSNR.

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
│   │   │   ├── client.ts         # Main-thread inference client (worker lifecycle, run timeouts, WASM fallback)
│   │   │   └── inference.worker.ts # ONNX Runtime sessions + inference (WebGPU/WASM), runs off the main thread
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
  in-browser inference engine (WebGPU / WASM execution providers).
- **[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)**
  (gzuidhof) — cross-origin isolation for multithreaded WASM and WebGPU.
- **[BrowserSnip](https://github.com/ningtoba/BrowserSnip)** and
  **BrowserSnip-Blurred** — the design system and architecture of the
  BrowserSnip project family.
- **[SceneWorks/real-esrgan-onnx](https://huggingface.co/SceneWorks/real-esrgan-onnx)**
  (BSD-3-Clause) — a reference for the dynamic-shape ONNX export approach.

## License

- **App code:** MIT.
- **Models:** BSD-3-Clause (Real-ESRGAN).
- **Upscayl is NOT vendored** — no Upscayl code is copied into this project.
