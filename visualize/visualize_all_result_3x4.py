# -*- coding: utf-8 -*-
"""
3×4 horizontal bar chart for 12 SAR benchmarks
（由 4×3 转置：行=任务，列=增量降序）

- 第 1/2/3 行：分类 / 检测 / 分割；每行自左至右按 SARATR-X-v2 表内增量 (delta) 从大到小
- top = best
- previous methods: blue palette
- ours (SARATR-X-v2): red palette
- best / second: bold / underline（与论文表一致）
- 数据与样式复用 visualize_all_result.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import visualize_all_result as base


# =============================================================================
# 3×4 布局（行=任务，列=增量降序）
# =============================================================================
NROWS, NCOLS = 3, 4
CELL_W, CELL_H = 4.13, 2.6
FIGSIZE = (NCOLS * CELL_W, NROWS * CELL_H)
DPI = base.DPI
_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
_SAVE_BASE = "sar_benchmark_3x4_horizontal_bar"
SAVE_PATH = os.path.join(_SAVE_DIR, f"{_SAVE_BASE}.png")
SAVE_PATH_PDF = os.path.join(_SAVE_DIR, f"{_SAVE_BASE}.pdf")


def build_panel_grid_3x4() -> List[Dict]:
    """
    3 行 × 4 列：行 0/1/2 = cls / det / seg；
    每行第 1–4 列按表内增量 (pp) 从大到小。
    flatten 顺序：行优先 → [cls_c0..c3, det_c0..c3, seg_c0..c3]。
    """
    cls = sorted(base._PANELS_CLS, key=base._panel_delta_for_sort, reverse=True)
    det = sorted(base._PANELS_DET, key=base._panel_delta_for_sort, reverse=True)
    seg = sorted(base._PANELS_SEG, key=base._panel_delta_for_sort, reverse=True)
    if len(cls) != NCOLS or len(det) != NCOLS or len(seg) != NCOLS:
        raise ValueError(
            f"每组需 {NCOLS} 个子图：cls={len(cls)} det={len(det)} seg={len(seg)}"
        )
    return cls + det + seg


PANELS: List[Dict] = build_panel_grid_3x4()


def main() -> None:
    base.apply_style()
    if len(PANELS) != NROWS * NCOLS:
        raise ValueError(f"PANELS 数量 {len(PANELS)} 与 {NROWS}×{NCOLS} 不一致")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, dpi=DPI, facecolor=base.FIG_FACE)
    axes = axes.flatten()

    underline_targets = []
    letters = list("abcdefghijkl")

    for ax, panel, letter in zip(axes, PANELS, letters):
        underline_texts = base.draw_panel(ax, panel, letter)
        underline_targets.extend(underline_texts)

    legend_handles = [
        Patch(
            facecolor=plt.get_cmap(base.PREV_CMAP_NAME)(0.58),
            edgecolor=plt.get_cmap(base.PREV_CMAP_NAME)(0.58),
            label="Previous methods",
        ),
        Patch(
            facecolor=base.OURS_COLOR_L,
            edgecolor=base.OURS_EDGE_L,
            label="SARATR-X-v2 (ours)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.01),
        ncol=2,
        frameon=False,
        fontsize=base.LEGEND_FONTSIZE,
    )

    fig.tight_layout(
        rect=(0.055, 0.055, 0.995, 0.995),
        pad=0.35,
        h_pad=0.45,
        w_pad=0.4,
    )
    fig.canvas.draw()

    for txt in underline_targets:
        base.add_underline(fig, txt, lw=base.SECOND_UNDERLINE_LW, pad_px=1.2)

    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", facecolor=base.FIG_FACE)
    plt.savefig(SAVE_PATH_PDF, dpi=DPI, bbox_inches="tight", facecolor=base.FIG_FACE, format="pdf")
    print(f"Saved: {SAVE_PATH}")
    print(f"Saved: {SAVE_PATH_PDF}")


if __name__ == "__main__":
    main()
