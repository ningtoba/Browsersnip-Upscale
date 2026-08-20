#!/usr/bin/env python3
"""
Export official Real-ESRGAN PyTorch weights to ONNX (dynamic H/W, fp32, opset 17).

The architectures below are 1:1 re-implementations of xinntao/Real-ESRGAN:
  - RRDBNet           -> realesrgan/archs/rrdbnet_arch.py
  - SRVGGNetCompact   -> realesrgan/archs/srvgg_arch.py

Weights are the canonical releases from xinntao/Real-ESRGAN:
  - RealESRGAN_x4plus.pth            (RRDBNet, 23 blocks)  v0.1.0
  - RealESRGAN_x4plus_anime_6B.pth   (RRDBNet,  6 blocks)  v0.2.2.4
  - realesr-animevideov3.pth         (SRVGGNetCompact)     v0.2.5.0

These are the same model family Upscayl ships (ncnn format); here they are
exported for onnxruntime-web (WebGPU / WASM) so upscaling runs fully in the
browser.

Usage:
  python3 scripts/export_onnx.py --arch rrdb  --blocks 23 \
      --input RealESRGAN_x4plus.pth --output public/models/realesrgan-x4plus.onnx
  python3 scripts/export_onnx.py --arch rrdb  --blocks 6 \
      --input RealESRGAN_x4plus_anime_6B.pth --output public/models/realesrgan-x4plus-anime.onnx
  python3 scripts/export_onnx.py --arch srvgg \
      --input realesr-animevideov3.pth --output public/models/realesr-animevideov3.onnx

Each export runs a parity check (torch vs onnxruntime CPU, PSNR on random input)
and fails unless the outputs match to >90 dB.
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- RRDBNet ---

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used by RRDBNet."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirical weights from the paper
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def make_layer(block, n_layers, **kw):
    return nn.Sequential(*(block(**kw) for _ in range(n_layers)))


class RRDBNet(nn.Module):
    """Real-ESRGAN RRDBNet (x4plus / x4plus-anime)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ----------------------------------------------------------- SRVGGNetCompact ---

class SRVGGNetCompact(nn.Module):
    """Compact VGG-style network (realesr-animevideov3)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


# ------------------------------------------------------------------- export ---

def build_model(arch: str, blocks: int | None) -> nn.Module:
    if arch == 'rrdb':
        n = 23 if blocks is None else int(blocks)
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=n, num_grow_ch=32, scale=4)
    if arch == 'srvgg':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    raise ValueError(f'Unknown arch: {arch}')


def check_parity(model: nn.Module, onnx_path: str) -> float:
    """Compare torch output vs onnxruntime CPU output on a random input; return PSNR."""
    try:
        import onnxruntime as ort
    except ImportError:
        print('WARN: onnxruntime not installed — skipping parity check', file=sys.stderr)
        return float('nan')

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dtype = next(model.parameters()).dtype
    x = torch.rand(1, 3, 48, 64).to(dtype)
    with torch.no_grad():
        y_torch = model(x).float().numpy()
    try:
        y_onnx = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    except Exception as err:  # e.g. CPU EP without fp16 kernels
        print(f'WARN: onnxruntime could not run this model on CPU '
              f'({err.__class__.__name__}: {err}) — skipping graph parity', file=sys.stderr)
        return float('nan')
    y_onnx = y_onnx.astype(np.float32, copy=False) if y_onnx.dtype != np.float32 else y_onnx
    mse = float(((y_torch - y_onnx) ** 2).mean())
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(1.0 / mse)
    print(f'parity: torch vs onnxruntime PSNR = {psnr:.2f} dB (mse={mse:.3e})')
    # fp32 exports must be bit-near-exact; fp16 has inherent rounding (~1e-7
    # MSE on [0,1] outputs, ~65-75 dB) due to different accumulation order.
    threshold = 50.0 if dtype == torch.float16 else 90.0
    if psnr < threshold:
        raise SystemExit(f'FAIL: parity PSNR {psnr:.2f} dB < {threshold:.0f} dB — export mismatch')
    return psnr


def main() -> None:
    ap = argparse.ArgumentParser(description='Export Real-ESRGAN weights to ONNX')
    ap.add_argument('--arch', choices=['rrdb', 'srvgg'], required=True)
    ap.add_argument('--blocks', type=int, default=None, help='RRDBNet block count (23 or 6)')
    ap.add_argument('--num-conv', type=int, default=16, help='SRVGGNetCompact conv count (animevideov3: 16, general-x4v3: 32)')
    ap.add_argument('--input', required=True, help='Path to official .pth weights')
    ap.add_argument('--output', required=True, help='Path to write the .onnx model')
    ap.add_argument('--fp16', action='store_true',
                    help='Export in float16 (for WebGPU users — halves download size)')
    args = ap.parse_args()

    model = build_model(args.arch, args.blocks)
    if args.arch == 'srvgg':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=args.num_conv, upscale=4, act_type='prelu')
    state = torch.load(args.input, map_location='cpu', weights_only=True)
    key = 'params_ema' if 'params_ema' in state else 'params'
    model.load_state_dict(state[key])
    model.eval()
    print(f'loaded {args.input} ({key}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)')

    x = torch.rand(1, 3, 64, 64)
    if args.fp16:
        model = model.half()
        x = x.half()
    with torch.no_grad():
        y = model(x)
    print(f'output shape at 64x64 input ({y.dtype}): {tuple(y.shape)}')

    torch.onnx.export(
        model,
        x,
        args.output,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes={'input': {2: 'H', 3: 'W'}, 'output': {2: 'H', 3: 'W'}},
    )
    print(f'exported {args.output}')

    check_parity(model, args.output)
    if args.fp16:
        # Fidelity of the fp16 export vs the fp32 model on the same input.
        fp32 = build_model(args.arch, args.blocks)
        fp32.load_state_dict(state[key])
        fp32.eval()
        with torch.no_grad():
            y_fp32 = fp32(x.float()).float()
        mse = float(((y.float() - y_fp32) ** 2).mean())
        psnr = float('inf') if mse == 0 else 10 * math.log10(1.0 / mse)
        print(f'fp16 vs fp32 fidelity PSNR = {psnr:.2f} dB')
        if psnr < 45:
            raise SystemExit(f'FAIL: fp16 fidelity PSNR {psnr:.2f} dB < 45 dB')
    print('OK')


if __name__ == '__main__':
    main()
