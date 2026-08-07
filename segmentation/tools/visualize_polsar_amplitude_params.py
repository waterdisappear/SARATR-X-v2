# -*- coding: utf-8 -*-
"""对比 LoadPolSARAmplitudeRGB 的 clip_percentile / log_gain 效果（伪 RGB：R=HH,G=HV,B=VV）。

用法示例（Linux，任选一张 HH 幅度图）::

    python tools/visualize_polsar_amplitude_params.py \\
        --hh /mnt/data7t/lwj/mmseg_lwj/data/AIR-PolarSAR-Seg-2.0/val/hh/xxx_hh_amp.tiff \\
        -o work_dirs/polsar_amp_param_grid.png

默认会画多组参数网格；也可用 --clips / --gains 自定义。
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image


def _build_pair_path(hh_path: str, pol: str) -> str:
    """与 mmcv_custom.resize_transform.LoadPolSARAmplitudeRGB._build_pair_path 一致。"""
    hh_name = osp.basename(hh_path)
    hh_dir = osp.dirname(hh_path)
    if hh_name.endswith('_hh_amp.tiff'):
        split_dir = osp.dirname(hh_dir)
        patch_id = hh_name[: -len('_hh_amp.tiff')]
        return osp.join(split_dir, pol, f'{patch_id}_{pol}_amp.tiff')
    if hh_name.endswith('_HH.tiff'):
        return osp.join(hh_dir, hh_name.replace('_HH.tiff', f'_{pol.upper()}.tiff'))
    raise ValueError(
        f'Unexpected HH filename: {hh_name}, expected *_hh_amp.tiff or *_HH.tiff')


def _nonlinear_quantize(img_2d: np.ndarray, clip_percentile: float,
                        log_gain: float) -> np.ndarray:
    """与 LoadPolSARAmplitudeRGB._nonlinear_quantize 一致（单通道）。"""
    img = img_2d.astype(np.float32)
    upper = float(np.percentile(img, clip_percentile))
    upper = max(upper, 1e-6)
    img = np.clip(img, 0, upper) / upper
    img = np.log1p(log_gain * img) / np.log1p(log_gain)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def load_hhhvvv(hh_path: str) -> np.ndarray:
    """(H,W,3) float/uint，顺序 HH,HV,VV。"""
    hv_path = _build_pair_path(hh_path, 'hv')
    vv_path = _build_pair_path(hh_path, 'vv')
    hh = np.array(Image.open(hh_path))
    hv = np.array(Image.open(hv_path))
    vv = np.array(Image.open(vv_path))
    for name, arr in [('hh', hh), ('hv', hv), ('vv', vv)]:
        if arr.ndim == 3:
            arr = arr[..., 0]
    if hh.ndim == 3:
        hh = hh[..., 0]
    if hv.ndim == 3:
        hv = hv[..., 0]
    if vv.ndim == 3:
        vv = vv[..., 0]
    return np.stack([hh, hv, vv], axis=-1)


def apply_clip_log(img_hw3: np.ndarray, clip_percentile: float,
                   log_gain: float) -> np.ndarray:
    """三通道各自做分位裁剪 + 对数量化，输出 uint8 HWC。"""
    out = []
    for i in range(3):
        out.append(
            _nonlinear_quantize(img_hw3[..., i], clip_percentile, log_gain))
    return np.stack(out, axis=-1)


def minmax_rgb_u8(img_hw3: np.ndarray) -> np.ndarray:
    """无 clip 时：每通道 min-max 到 0-255，仅作对照可视化。"""
    x = img_hw3.astype(np.float32)
    out = []
    for i in range(3):
        c = x[..., i]
        lo, hi = float(c.min()), float(c.max())
        if hi <= lo + 1e-12:
            out.append(np.zeros_like(c, dtype=np.uint8))
        else:
            t = (c - lo) / (hi - lo)
            out.append(np.clip(np.round(t * 255.0), 0, 255).astype(np.uint8))
    return np.stack(out, axis=-1)


def main():
    p = argparse.ArgumentParser(
        description='Visualize PolSAR amplitude pseudo-RGB under different '
        'clip_percentile / log_gain (same as LoadPolSARAmplitudeRGB).')
    p.add_argument(
        '--hh',
        default=r'data/air-polarsar-seg/train_set/AIR-PolarSAR-Seg-1_HH.tiff',
        help='HH 幅度图路径（同训练集，如 *_hh_amp.tiff 或 *_HH.tiff）')
    p.add_argument(
        '-o',
        '--out',
        default='polsar_amp_param_grid.png',
        help='输出 PNG 路径')
    p.add_argument(
        '--clips',
        type=float,
        nargs='+',
        default=[98.0, 99.0, 99.5, 99.9],
        help='clip_percentile 列表（每行一种）')
    p.add_argument(
        '--gains',
        type=float,
        nargs='+',
        default=[8.0, 15.0, 25.0],
        help='log_gain 列表（每列一种）')
    p.add_argument(
        '--dpi',
        type=int,
        default=120,
        help='输出 DPI')
    p.add_argument(
        '--show-minmax',
        action='store_true',
        help='在网格左上角额外加一列：各通道 min-max 伪彩色（无 clip/log）')
    args = p.parse_args()

    import matplotlib.pyplot as plt

    img = load_hhhvvv(args.hh)
    clips = list(args.clips)
    gains = list(args.gains)
    nrows = len(clips)
    ncols = len(gains) + (1 if args.show_minmax else 0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        squeeze=False)

    col0 = 0
    if args.show_minmax:
        ref = minmax_rgb_u8(img)
        for r in range(nrows):
            ax = axes[r, 0]
            ax.imshow(ref)
            ax.set_title('min-max / ch\n(no clip+log)' if r == 0 else '')
            ax.axis('off')
        col0 = 1

    for ri, clip_p in enumerate(clips):
        for gj, gain in enumerate(gains):
            vis = apply_clip_log(img, clip_p, gain)
            ax = axes[ri, col0 + gj]
            ax.imshow(vis)
            if ri == 0:
                ax.set_title(f'log_gain={gain:g}', fontsize=10)
            # axis('off') 会吞掉 ylabel，故用叠加文字标明本行 clip_percentile
            ax.text(
                0.02,
                0.98,
                f'clip_pct\n{clip_p:g}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                color='white',
                linespacing=0.95,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.55))
            ax.axis('off')

    clips_str = ', '.join(f'{c:g}' for c in clips)
    gains_str = ', '.join(f'{g:g}' for g in gains)
    fig.suptitle(
        f'LoadPolSARAmplitudeRGB-style  |  {osp.basename(args.hh)}\n'
        f'rows: clip_percentile = [{clips_str}]\n'
        f'cols: log_gain = [{gains_str}]',
        fontsize=10,
        y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = osp.abspath(args.out)
    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
