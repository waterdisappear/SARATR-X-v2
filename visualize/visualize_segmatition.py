# -*- coding: utf-8 -*-
"""
随机选取 AIR-PolarSAR-Seg-2.0 train 中一对 hh（SAR）与 gt_color（彩色语义标签），
叠加展示语义分割示意；标签层使用可调透明度，便于看清底层 SAR。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    """Windows 上含中文路径时用字节读取 + imdecode，避免 cv2.imread 失败。"""
    if not path.is_file():
        return None
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def collect_images(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            out[p.stem] = p
    return out


def gray_to_u8(gray: np.ndarray) -> np.ndarray:
    """将单通道 SAR 压到 0–255 uint8，便于与彩色标签叠加显示。"""
    g = gray.astype(np.float32)
    if g.size == 0:
        return g.astype(np.uint8)
    gmin, gmax = float(g.min()), float(g.max())
    if gmax <= gmin:
        return np.zeros_like(gray, dtype=np.uint8)
    u8 = np.clip((g - gmin) / (gmax - gmin) * 255.0, 0.0, 255.0).astype(np.uint8)
    return u8


def ensure_hwc_bgr(img: np.ndarray) -> np.ndarray:
    """统一为 HxWx3 BGR uint8。"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_like(src: np.ndarray, ref_hw: Tuple[int, int]) -> np.ndarray:
    h, w = ref_hw
    if src.shape[0] == h and src.shape[1] == w:
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_NEAREST)


def _strip_known_suffix(stem: str, suffixes: Tuple[str, ...]) -> str:
    low = stem.lower()
    for suf in suffixes:
        s = suf.lower()
        if len(stem) > len(suf) and low.endswith(s):
            return stem[: -len(suf)]
    return stem


def norm_hh_stem(stem: str) -> str:
    """
    与 gt_color stem 对齐。该数据集常见形式为：
    ``{patch_id}_hh_amp`` / ``_hh_real`` / ``_hh_pha`` / ``_hh_imgy``，
    对应标签 ``{patch_id}``。
    """
    low = stem.lower()
    if "_hh_" in low:
        return stem[: low.index("_hh_")]
    return _strip_known_suffix(
        stem,
        ("_hh", "-hh", "_HH", "_hv", "-hv", "_HV", "_vv", "-vv", "_VV", "_vh", "-vh", "_VH"),
    )


def norm_gt_stem(stem: str) -> str:
    """去掉彩色标签常见后缀。"""
    return _strip_known_suffix(
        stem,
        ("_gt_color", "_gtcolor", "_color", "_gt", "_label", "_mask", "_seg"),
    )


def filter_hh_map(hh_map: Dict[str, Path], hh_mode: str) -> Dict[str, Path]:
    """按复数分量筛选 hh（默认幅度 amp，便于目视解译）。"""
    m = hh_mode.strip().lower()
    if m in ("any", "all", "*"):
        return dict(hh_map)
    if m not in ("amp", "real", "pha", "imgy"):
        raise ValueError(f"未知 hh_mode: {hh_mode!r}，请用 amp|real|pha|imgy|any")
    suf = f"_hh_{m}"
    out = {k: v for k, v in hh_map.items() if k.lower().endswith(suf)}
    return out


def build_hh_gt_pairs(hh_map: Dict[str, Path], gt_map: Dict[str, Path]) -> List[Tuple[str, str, Path, Path]]:
    """
    返回 (pair_key, hh_stem, hh_path, gt_path)；
    先尝试 stem 完全一致，再按去后缀后的公共 key 配对（适配 hh 与 gt 命名规则不同）。
    """
    pairs: List[Tuple[str, str, Path, Path]] = []
    for s in set(hh_map.keys()) & set(gt_map.keys()):
        pairs.append((s, s, hh_map[s], gt_map[s]))
    if pairs:
        return pairs

    gt_by_key: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for gst, gp in gt_map.items():
        gt_by_key[norm_gt_stem(gst)].append((gst, gp))

    seen: set[Tuple[str, str]] = set()
    for hh_st, hp in hh_map.items():
        key = norm_hh_stem(hh_st)
        if key not in gt_by_key:
            continue
        for _gst, gp in gt_by_key[key]:
            sig = (hh_st, str(gp))
            if sig in seen:
                continue
            seen.add(sig)
            pairs.append((key, hh_st, hp, gp))
    return pairs


def blend_sar_overlay(
    sar_bgr: np.ndarray,
    seg_bgr: np.ndarray,
    overlay_alpha: float,
) -> np.ndarray:
    """
    result = sar * (1 - a) + seg * a
    overlay_alpha 越大，彩色标签越显眼；越小则 SAR 越清晰。
    """
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    s = sar_bgr.astype(np.float32)
    v = seg_bgr.astype(np.float32)
    out = s * (1.0 - a) + v * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    default_train = "dataset/segmentation/air-polarsar-seg-2.0/train"
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="随机 hh + gt_color 叠加可视化语义分割示意。")
    parser.add_argument("--train_dir", type=str, default=default_train, help="train 根目录（内含 hh、gt_color）。")
    parser.add_argument(
        "--hh_mode",
        type=str,
        default="amp",
        choices=("amp", "real", "pha", "imgy", "any"),
        help="hh 文件名中的复数分量：amp/real/pha/imgy；any 表示四种均可随机抽到。",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.38,
        help="语义彩色层不透明度，约 0.25–0.45：越小 SAR 越清楚，越大色块越实。",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不设则每次不同。")
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(here / "seg_overlay_demo.png"),
        help="叠加结果保存路径。",
    )
    parser.add_argument("--dpi", type=int, default=200, help="保存图像 DPI。")
    parser.add_argument("--show", default=False, help="弹窗显示 matplotlib 图。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = Path(args.train_dir)
    hh_dir = train / "hh"
    gt_dir = train / "gt_color"

    hh_map = filter_hh_map(collect_images(hh_dir), args.hh_mode)
    if not hh_map:
        raise SystemExit(
            f"hh_mode={args.hh_mode!r} 下未匹配到任何 hh 文件（期望 stem 以 _hh_{args.hh_mode} 结尾）。请改用 --hh_mode any。"
        )
    gt_map = collect_images(gt_dir)
    pairs = build_hh_gt_pairs(hh_map, gt_map)
    if not pairs:
        raise SystemExit(
            f"无法在 {hh_dir} 与 {gt_dir} 之间配对图像（同名 stem 与去后缀规则均未命中）。\n"
            f"hh 数量={len(hh_map)}, gt_color 数量={len(gt_map)}，请检查命名或联系维护者扩展规则。"
        )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pair_key, hh_stem, hh_path, gt_path = random.choice(pairs)

    hh_raw = imread_unicode(hh_path, cv2.IMREAD_UNCHANGED)
    gt_raw = imread_unicode(gt_path, cv2.IMREAD_UNCHANGED)
    if hh_raw is None:
        raise SystemExit(f"无法读取 SAR: {hh_path}")
    if gt_raw is None:
        raise SystemExit(f"无法读取标签: {gt_path}")

    if hh_raw.ndim == 3:
        hh_gray = cv2.cvtColor(hh_raw, cv2.COLOR_BGR2GRAY)
    else:
        hh_gray = hh_raw
    if hh_gray.dtype != np.uint8:
        hh_u8 = gray_to_u8(hh_gray)
    else:
        hh_u8 = hh_gray
    sar_bgr = cv2.cvtColor(hh_u8, cv2.COLOR_GRAY2BGR)

    seg_bgr = ensure_hwc_bgr(gt_raw)
    seg_bgr = resize_like(seg_bgr, (sar_bgr.shape[0], sar_bgr.shape[1]))

    blended_bgr = blend_sar_overlay(sar_bgr, seg_bgr, args.overlay_alpha)
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    sar_rgb = cv2.cvtColor(sar_bgr, cv2.COLOR_BGR2RGB)
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(save_path.suffix if save_path.suffix else ".png", blended_bgr)
    if not ok:
        raise SystemExit("保存叠加图失败（imencode）。")
    enc.tofile(str(save_path))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=120)
    titles = ("HH (SAR)", "gt_color (mask)", f"overlay (α={args.overlay_alpha:.2f})")
    imgs = (sar_rgb, seg_rgb, blended_rgb)
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"pair_key={pair_key}\n{hh_path.name} + {gt_path.name}", fontsize=10, y=1.02)
    fig.tight_layout()
    # plt.savefig(save_path.with_name(save_path.stem + "_panel" + save_path.suffix), dpi=args.dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    # print(f"三联图: {save_path.with_name(save_path.stem + '_panel' + save_path.suffix)}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
