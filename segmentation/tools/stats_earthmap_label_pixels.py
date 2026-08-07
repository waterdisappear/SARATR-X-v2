#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 EarthMap OEM 风格标签（trainId 0=无效, 1..8=类）在各 split 的像素占比。

用于排查某类（如 Bareland=1）是否极少、或映射后是否几乎全为 ignore。

用法:
  python tools/stats_earthmap_label_pixels.py --data-root /path/to/EarthMap
  python tools/stats_earthmap_label_pixels.py --data-root D:\\Data\\SAR\\地物\\EarthMap --max-files 500
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Optional

import numpy as np
from PIL import Image

# 与 configs 中 classes 顺序一致：索引 0..7 对应 OEM trainId 1..8
CLASS_NAMES = (
    'Bareland',
    'Rangeland',
    'Developed Space',
    'Road',
    'Tree',
    'Water',
    'Agriculture Land',
    'Building',
)


def _apply_reduce_zero_label(arr: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """与 mmseg LoadAnnotations / eval_metrics 一致。"""
    out = arr.astype(np.int32, copy=True)
    out[out == 0] = ignore_index
    out = out - 1
    out[out == 254] = ignore_index
    return out


def _iter_label_files(labels_dir: str, suffixes=('.tif', '.tiff', '.png')):
    if not osp.isdir(labels_dir):
        return
    for name in sorted(os.listdir(labels_dir)):
        low = name.lower()
        if any(low.endswith(s) for s in suffixes):
            yield osp.join(labels_dir, name)


def accumulate_split(labels_dir: str, max_files: Optional[int]):
    raw_hist = np.zeros(256, dtype=np.int64)
    mapped_hist = np.zeros(9, dtype=np.int64)  # 0..7 class, 8 = ignore(255)
    n_files = 0
    for path in _iter_label_files(labels_dir):
        if max_files is not None and n_files >= max_files:
            break
        a = np.array(Image.open(path))
        if a.ndim == 3:
            a = a[..., 0]
        a = a.astype(np.uint8, copy=False)
        for v in range(256):
            raw_hist[v] += int((a == v).sum())
        m = _apply_reduce_zero_label(a)
        for c in range(8):
            mapped_hist[c] += int((m == c).sum())
        mapped_hist[8] += int((m == 255).sum())
        n_files += 1
    return raw_hist, mapped_hist, n_files


def print_split(name: str, labels_dir: str, max_files: Optional[int]):
    if not osp.isdir(labels_dir):
        print(f'[{name}] 跳过（无目录）: {labels_dir}')
        return
    raw_hist, mapped_hist, n_files = accumulate_split(labels_dir, max_files)
    total_raw = raw_hist.sum()
    total_m = mapped_hist.sum()
    print(f'\n=== {name} ===')
    print(f'  标签目录: {labels_dir}')
    print(f'  读取文件数: {n_files}' + (f'（--max-files 截断）' if max_files and n_files == max_files else ''))
    if total_raw == 0:
        print('  （无像素）')
        return
    print('  原始像素值占比（OEM trainId，0=无效）:')
    for v in range(9):
        cnt = int(raw_hist[v])
        pct = 100.0 * cnt / total_raw
        if v == 0:
            tag = 'ignore/void'
        else:
            tag = CLASS_NAMES[v - 1]
        print(f'    id={v} ({tag}): {cnt} px ({pct:.4f}%)')
    for v in range(9, 256):
        cnt = raw_hist[v]
        if cnt > 0:
            print(f'    id={v} (非预期): {cnt} px ({100.0 * cnt / total_raw:.4f}%)')

    print('  映射后（与训练 evaluate 一致，0..7=类，255=忽略）:')
    for c in range(8):
        cnt = mapped_hist[c]
        pct = 100.0 * cnt / total_m if total_m else 0.0
        print(f'    class {c} ({CLASS_NAMES[c]}): {cnt} px ({pct:.4f}%)')
    ign = mapped_hist[8]
    print(f'    ignore (255): {ign} px ({100.0 * ign / total_m if total_m else 0:.4f}%)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True, help='EarthMap 根目录（含 train/val/test）')
    p.add_argument('--max-files', type=int, default=None, help='每个 split 最多读多少张（调试用）')
    args = p.parse_args()
    root = osp.abspath(args.data_root)
    for split in ('train', 'val', 'test'):
        ld = osp.join(root, split, 'labels')
        print_split(split, ld, args.max_files)
    print('\n说明: Bareland 对应原始标签 id=1；若 id=1 占比极低，mIoU 中该类会接近 0。')


if __name__ == '__main__':
    main()
