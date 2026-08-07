# -*- coding: utf-8 -*-
"""
4×3 horizontal bar chart for 12 SAR benchmarks
- 第 1/2/3 列：分类 / 检测 / 分割；每列自上而下按 SARATR-X-v2 表内增量 (delta) 从大到小
- top = best
- previous methods: blue palette
- ours (SARATR-X-v2): red palette
- best / second: bold / underline（与论文表一致）
- WHU-OPT-SAR 子图 x 轴为 PA (%)；其余 seg. 为 mIoU (%)
- Accuracy (%) 子图横轴上限不超过 100
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.transforms import blended_transform_factory


# =============================================================================
# 可修改的全局样式
# =============================================================================
NROWS, NCOLS = 4, 3
CELL_W, CELL_H = 4.13, 2.6
# 12 子图同尺寸：宽 3 列、高 4 行
FIGSIZE = (NCOLS * CELL_W, NROWS * CELL_H)
DPI = 300
_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
_SAVE_BASE = "sar_benchmark_4x3_horizontal_bar"
SAVE_PATH = os.path.join(_SAVE_DIR, f"{_SAVE_BASE}.png")
SAVE_PATH_PDF = os.path.join(_SAVE_DIR, f"{_SAVE_BASE}.pdf")

FONT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
}

TITLE_FONTSIZE = 11.5
XLABEL_FONTSIZE = 9.5
YTICK_FONTSIZE = 8.8
# y 轴方法名：负角度为顺时针，文字整体向右下方倾斜
YTICK_ROTATION = 25
VALUE_FONTSIZE = 8.4
DELTA_FONTSIZE = 6.8
BACKBONE_FONTSIZE = 7.0
BACKBONE_COLOR = "#000000"
LEGEND_FONTSIZE = 11.0

BAR_HEIGHT = 0.68
OURS_EDGEWIDTH = 1.0
PREV_EDGEWIDTH = 0.8

GRID_COLOR = "#D7D7D7"
GRID_ALPHA = 0.75
GRID_LS = ":"

SPINE_COLOR = "#666666"

FIG_FACE = "white"
AX_FACE = "white"

# 其他方法：蓝色系（淡雅）
PREV_CMAP_NAME = "Blues"
PREV_CMAP_T0 = 0.35
PREV_CMAP_T1 = 0.82

# 我们的方法：红色系
OURS_COLOR_B = "#E88C8D"   # iTPN-B
OURS_COLOR_L = "#C94B5F"   # iTPN-L
OURS_EDGE_B = "#A85E5F"
OURS_EDGE_L = "#8F2437"

BEST_TEXT_WEIGHT = "bold"
SECOND_UNDERLINE_LW = 1.1

# 柱旁数值标注水平位置（均按子图 x 轴跨度 span 的比例换算，与具体数据集无关）
VALUE_DX_FRAC = 0.018          # 柱末端 → 主分数左缘；越大分数离柱越远
DELTA_DX_FRAC = 0.05          # 主分数 → delta 的额外间距；越大「(+x.x)」离主分数越远
DELTA_CHAR_DX_FRAC = 0.0085    # 按主分数字符数补偿（如 97.8 比 68.7 宽）；越大间距随位数增加越多

# 子图标题：数据集名 + 任务类型缩写（与论文表 caption 一致）
TITLE_FMT = "({letter}) {dataset} ({task})"


# =============================================================================
# 数据结构
# =============================================================================
Emphasis = Literal["best", "second"]


@dataclass
class Entry:
    method: str
    year: int
    score: float
    backbone: str = ""
    ours: bool = False
    variant: str = ""              # "B" / "L"，仅用于配色；y 轴统一显示方法名+年份
    delta: Optional[float] = None  # 与论文表一致；None 则不显示括号
    # 与 LaTeX 表一致：\textbf{最佳}、\underline{次优}（按各子图指标列，非自动按排序）
    emphasis: Optional[Emphasis] = None

    def label(self) -> str:
        # ours 与基线一致：方法名 + 年份（B/L 由柱内骨干 iTPN-B / iTPN-L 区分）
        return f"{self.method} ({self.year})"


# =============================================================================
# 12 个子图数据（按任务分三组；绘图前自动排成 4×3：列=任务，行=增量降序）
# =============================================================================
_PANELS_CLS: List[Dict] = [
    {
        "dataset": "ATRNet-STAR",
        "task": "cls.",
        "xlabel": "Accuracy (%)",
        "entries": [
            Entry("ViT", 2020, 59.2, backbone="ViT-B"),
            Entry("HDANet", 2023, 63.7, backbone="CNN"),
            Entry("ConvNeXt", 2022, 81.6, backbone="ConvNeXt-B"),
            Entry("SARATR-X", 2025, 85.2, backbone="HiViT-B"),
            Entry("SARMAE", 2025, 81.4, backbone="ViT-L"),
            Entry("AFRL-DINOv2", 2025, 94.9, backbone="ViT-B", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 97.8, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 98.4, backbone="iTPN-L", ours=True, variant="L", delta=+3.5, emphasis="best"),
        ],
    },
    {
        "dataset": "MSTAR",
        "task": "cls.",
        "xlabel": "Accuracy (%)",
        "entries": [
            Entry("BIDFC", 2022, 90.3, backbone="ResNet-18"),
            Entry("CRID", 2023, 73.3, backbone="Swin"),
            Entry("EURAPS", 2023, 88.7, backbone="WRN-28-2"),
            Entry("SAR-JEPA", 2024, 56.5, backbone="ViT-B"),
            Entry("PD", 2024, 70.2, backbone="A-ConvNet"),
            Entry("SARATR-X", 2025, 95.9, backbone="HiViT-B", emphasis="second"),
            Entry("SAMBA", 2026, 95.0, backbone="Mamba"),
            Entry("SARATR-X-v2", 2026, 96.3, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 98.7, backbone="iTPN-L", ours=True, variant="L", delta=+2.8, emphasis="best"),
        ],
    },
    {
        "dataset": "SAR-VSA",
        "task": "cls.",
        "xlabel": "Accuracy (%)",
        "xmax": 92,
        "entries": [
            Entry("OpenCLIP", 2023, 81.6, backbone="ViT-L"),
            Entry("GeoRSCLIP", 2024, 80.1, backbone="ViT-L"),
            Entry("RemoteCLIP", 2024, 80.0, backbone="ViT-L"),
            Entry("SARCLIP", 2025, 89.8, backbone="ViT-L", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 90.8, backbone="iTPN-B", ours=True, variant="B", delta=+1.0, emphasis="best"),
            Entry("SARATR-X-v2", 2026, 90.6, backbone="iTPN-L", ours=True, variant="L"),
        ],
    },
    {
        "dataset": "FUSAR-Ship",
        "task": "cls.",
        "xlabel": "Accuracy (%)",
        "xmin": 52,
        "xmax": 95,
        "entries": [
            # Entry("ResNet", 2016, 58.4, backbone="ResNet-50"),
            Entry("Swin", 2021, 60.8, backbone="Swin-B"),
            Entry("Beit", 2021, 71.1, backbone="ViT-B"),
            Entry("SUMMIT", 2025, 71.9, backbone="ViT-B"),
            Entry("SARMAE", 2025, 92.9, backbone="ViT-B", emphasis="second"),
            Entry("SAMBA", 2026, 85.0, backbone="Mamba"),
            Entry("SARATR-X-v2", 2026, 93.0, backbone="iTPN-B", ours=True, variant="B", delta=+0.1, emphasis="best"),
            Entry("SARATR-X-v2", 2026, 91.7, backbone="iTPN-L", ours=True, variant="L"),
        ],
    },
]

_PANELS_DET: List[Dict] = [
    {
        "dataset": "SARDet-100K",
        "task": "det.",
        "xlabel": "mAP (%)",
        "xmax": 72,
        "entries": [
            Entry("MSFA", 2025, 56.4, backbone="ConvNext-B"),
            Entry("SARATR-X", 2025, 57.2, backbone="HiViT-B"),
            Entry("SUMMIT", 2025, 57.0, backbone="ViT-B"),
            Entry("SARMAE", 2025, 63.1, backbone="ViT-L"),
            Entry("ViTP", 2025, 59.7, backbone="ViT-L"),
            Entry("MaRS", 2026, 55.4, backbone="SwinV2-B"),
            Entry("SAMBA", 2026, 59.7, backbone="Mamba"),
            Entry("BabelRS", 2026, 63.3, backbone="ViT-L", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 63.4, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 68.7, backbone="iTPN-L", ours=True, variant="L", delta=+5.4, emphasis="best"),
        ],
    },
    {
        "dataset": "RSAR",
        "task": "det.",
        "xlabel": "mAP@50 (%)",
        "xmax": 78,
        "entries": [
            # Entry("RoI-Transformer", 2019, 67.0, backbone="ResNet-50"),
            Entry("OrientedFormer", 2024, 69.8, backbone="ResNet-50"),
            Entry("GSDet", 2025, 68.0, backbone="ResNet-50"),
            Entry("SARMAE", 2025, 72.2, backbone="ViT-L"),
            Entry("ViTP", 2025, 72.3, backbone="ViT-L"),
            Entry("O2-RTDETR", 2026, 73.1, backbone="Res-18vd", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 74.2, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 77.3, backbone="iTPN-L", ours=True, variant="L", delta=+4.2, emphasis="best"),
        ],
    },
    {
        "dataset": "SSDD",
        "task": "det.",
        "xlabel": "mAP (%)",
        "xmax": 73,
        "entries": [
            Entry("FEPS-Net", 2023, 59.9, backbone="ResNet-50"),
            Entry("CS\u207fNet", 2023, 64.9, backbone="CSPDarkNet-53"),
            Entry("SARATR-X", 2025, 67.5, backbone="HiViT-B"),
            Entry("SUMMIT", 2025, 70.4, backbone="ViT-B"),
            Entry("SARMAE", 2025, 69.3, backbone="ViT-L"),
            Entry("SAMBA", 2026, 71.3, backbone="Mamba", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 71.0, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 71.8, backbone="iTPN-L", ours=True, variant="L", delta=+0.5, emphasis="best"),
        ],
    },
    {
        "dataset": "HRSID",
        "task": "det.",
        "xlabel": "mAP@50 (%)",
        "xmax": 95,
        "entries": [
            Entry("HRLE-SARDet", 2023, 92.5, backbone="LSFEBackbone"),
            Entry("MLDet", 2023, 92.8, backbone="CSPDarkNet"),
            Entry("LFer-Net", 2024, 90.6, backbone="SIDConv"),
            Entry("HGNet", 2024, 93.0, backbone="CSPDarkNet"),
            Entry("CV-SAR-FM", 2025, 93.9, backbone="Swin-B"),
            Entry("RingMoE", 2026, 94.2, backbone="RingMoE-KC", emphasis="best"),
            Entry("SARATR-X-v2", 2026, 93.1, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 94.0, backbone="iTPN-L", ours=True, variant="L", delta=-0.2, emphasis="second"),
        ],
    },
]

_PANELS_SEG: List[Dict] = [
    {
        "dataset": "AIR-PolSAR-Seg-2.0",
        "task": "seg.",
        "xlabel": "mIoU (%)",
        "xmax": 98,
        "entries": [
            Entry("PSPNet", 2017, 82.2, backbone="ResNet-101"),
            Entry("DeepLabV3+", 2018, 83.1, backbone="Xception"),
            Entry("PointRend", 2020, 84.5, backbone="ResNet-101"),
            Entry("DANet", 2019, 86.0, backbone="ResNet-101"),
            Entry("TerraSegNet", 2025, 92.1, backbone="EfficientNetV2-S", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 90.8, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 95.3, backbone="iTPN-L", ours=True, variant="L", delta=+3.2, emphasis="best"),
        ],
    },
    {
        "dataset": "OpenEarthMap-SAR",
        "task": "seg.",
        "xlabel": "mIoU (%)",
        "xmax": 37.0,
        "entries": [
            Entry("U-Net", 2015, 35.1, backbone="U-Net"),
            Entry("VMamba", 2024, 34.7, backbone="VSS"),
            Entry("SegFormer", 2021, 35.8, backbone="MiT", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 35.8, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 36.2, backbone="iTPN-L", ours=True, variant="L", delta=+0.4, emphasis="best"),
        ],
    },
    {
        "dataset": "DDHR-SK",
        "task": "seg.",
        "xlabel": "mIoU (%)",
        "xmax": 87,
        "entries": [
            Entry("OCRNet", 2020, 74.2, backbone="HRNetV2-W48"),
            Entry("SegFormer", 2021, 78.4, backbone="MiT-B4"),
            Entry("PIDNet", 2023, 71.3, backbone="PIDNet-S"),
            Entry("TransUNet", 2024, 74.1, backbone="ResNet-50+ViT"),
            Entry("Mask2Former", 2022, 81.9, backbone="ResNet-101"),
            Entry("SETR", 2021, 82.8, backbone="ViT-L/16", emphasis="second"),
            Entry("SARATR-X-v2", 2026, 83.4, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 83.6, backbone="iTPN-L", ours=True, variant="L", delta=+0.8, emphasis="best"),
        ],
    },
    {
        "dataset": "WHU-OPT-SAR",
        "task": "seg.",
        "xlabel": "PA (%)",
        "xmax": 81,
        "entries": [
            Entry("PIDNet", 2023, 78.4, backbone="PIDNet-S"),
            Entry("OCRNet", 2020, 78.9, backbone="HRNetV2-W48"),
            Entry("SETR", 2021, 79.1, backbone="ViT-L"),
            Entry("SegFormer", 2021, 79.3, backbone="MiT-B4"),
            Entry("Mask2Former", 2022, 79.3, backbone="ResNet-101"),
            Entry("SARATR-X-v2", 2026, 79.8, backbone="iTPN-B", ours=True, variant="B"),
            Entry("SARATR-X-v2", 2026, 79.8, backbone="iTPN-L", ours=True, variant="L", delta=-0.3, emphasis="second"),
            Entry("TransUNet", 2024, 80.1, backbone="ResNet-50+ViT", emphasis="best"),
        ],
    },
]


def _panel_delta_for_sort(panel: Dict) -> float:
    """用于列内排序：优先取 emphasis=best 的 delta，否则取我方条目最大 delta。"""
    entries: List[Entry] = panel["entries"]
    for e in entries:
        if e.ours and e.emphasis == "best" and e.delta is not None:
            return float(e.delta)
    ours = [float(e.delta) for e in entries if e.ours and e.delta is not None]
    return max(ours) if ours else float("-inf")


def build_panel_grid() -> List[Dict]:
    """
    4 行 × 3 列：列 0/1/2 = cls / det / seg；
    每列第 1–4 行按表内增量 (pp) 从大到小（与 sar_gain 图一致）。
    flatten 顺序：行优先 → [cls_r, det_r, seg_r] × 4 行。
    """
    cls = sorted(_PANELS_CLS, key=_panel_delta_for_sort, reverse=True)
    det = sorted(_PANELS_DET, key=_panel_delta_for_sort, reverse=True)
    seg = sorted(_PANELS_SEG, key=_panel_delta_for_sort, reverse=True)
    if len(cls) != NROWS or len(det) != NROWS or len(seg) != NROWS:
        raise ValueError(f"每组需 {NROWS} 个子图：cls={len(cls)} det={len(det)} seg={len(seg)}")
    grid: List[Dict] = []
    for r in range(NROWS):
        grid.extend([cls[r], det[r], seg[r]])
    return grid


PANELS: List[Dict] = build_panel_grid()


# =============================================================================
# 工具函数
# =============================================================================
def apply_style() -> None:
    rcParams.update(FONT_RC)
    rcParams["axes.unicode_minus"] = False


def sort_entries(entries: List[Entry]) -> List[Entry]:
    return sorted(entries, key=lambda x: x.score, reverse=True)


def get_panel_xlim(values: List[float]) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1.0)

    xmin = math.floor((vmin - 0.18 * span) / 5.0) * 5.0
    xmax = math.ceil((vmax + 0.28 * span) / 5.0) * 5.0

    xmin = max(0.0, xmin)
    return xmin, xmax


def choose_integer_x_step(xmin: float, xmax: float, max_ticks: int = 7) -> int:
    """在 [1,2,5,10,20,25] 中选步长，使刻度个数适中且均为整数坐标。"""
    span = max(float(xmax) - float(xmin), 1.0)
    for step in (1, 2, 5, 10, 20, 25):
        if span / step <= max_ticks - 1:
            return step
    return 25


def apply_integer_x_axis(ax: plt.Axes, xmin: float, xmax: float) -> None:
    """
    x 轴刻度位置与标签均为整数（如 34、36），不会出现「线在 36.5、字写 36」。
    柱端数值标注仍用 entry.score 的真实小数，与此无关。
    """
    step = choose_integer_x_step(xmin, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def get_prev_colors(n_prev: int) -> List:
    if n_prev <= 0:
        return []
    cmap = plt.get_cmap(PREV_CMAP_NAME)
    ts = np.linspace(PREV_CMAP_T1, PREV_CMAP_T0, n_prev)  # 好的方法更深一些
    return [cmap(float(t)) for t in ts]


def format_score(entry: Entry) -> str:
    """仅主分数（不含 delta）；delta 单独绘制为小字号。"""
    return f"{entry.score:.1f}"


def add_underline(fig: plt.Figure, text_obj, color=None, lw=1.0, pad_px=1.0) -> None:
    """
    给某个 Text 对象添加下划线（matplotlib 原生不太方便，手动画一条线）。
    """
    renderer = fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer=renderer)

    x0, x1 = bbox.x0, bbox.x1
    y = bbox.y0 - pad_px

    inv = fig.transFigure.inverted()
    (x0f, y0f) = inv.transform((x0, y))
    (x1f, y1f) = inv.transform((x1, y))

    line = Line2D(
        [x0f, x1f],
        [y0f, y1f],
        transform=fig.transFigure,
        color=color if color is not None else text_obj.get_color(),
        linewidth=lw,
        solid_capstyle="butt",
        zorder=1000,
    )
    fig.add_artist(line)


def draw_panel(ax: plt.Axes, panel: Dict, letter: str):
    entries = sort_entries(panel["entries"])
    n = len(entries)
    y = np.arange(n)

    n_prev = sum(0 if e.ours else 1 for e in entries)
    prev_colors = get_prev_colors(n_prev)
    prev_idx = 0

    colors = []
    edgecolors = []
    for e in entries:
        if e.ours:
            if e.variant.upper() == "L":
                colors.append(OURS_COLOR_L)
                edgecolors.append(OURS_EDGE_L)
            else:
                colors.append(OURS_COLOR_B)
                edgecolors.append(OURS_EDGE_B)
        else:
            c = prev_colors[prev_idx]
            colors.append(c)
            edgecolors.append(c)
            prev_idx += 1

    scores = [e.score for e in entries]
    xmin, xmax = get_panel_xlim(scores)
    if panel.get("xmin") is not None:
        xmin = float(panel["xmin"])
    if panel.get("xmax") is not None:
        xmax = float(panel["xmax"])
    elif panel.get("xlabel") == "Accuracy (%)":
        # 准确率合法上界为 100%，避免出现 105 等刻度
        xmax = min(xmax, 100.0)
    span = xmax - xmin

    bars = ax.barh(
        y,
        scores,
        height=BAR_HEIGHT,
        color=colors,
        edgecolor=edgecolors,
        linewidth=[OURS_EDGEWIDTH if e.ours else PREV_EDGEWIDTH for e in entries],
        zorder=3,
    )

    for bi, (bar, e) in enumerate(zip(bars, entries)):
        if e.emphasis == "best":
            bar.set_linewidth(1.4)
            bar.set_edgecolor("#2E2E2E")
        elif e.emphasis == "second":
            bar.set_linewidth(1.15)
            bar.set_edgecolor("#5A5A5A")

    ax.set_xlim(xmin, xmax)
    ax.set_facecolor(AX_FACE)
    ax.invert_yaxis()

    # 骨干：绘图区内左侧（横轴为 axes 比例，避免与柱子刻度冲突）
    trans_xaxes_ydata = blended_transform_factory(ax.transAxes, ax.transData)
    for i, e in enumerate(entries):
        if not e.backbone:
            continue
        ax.text(
            0.02,
            y[i],
            e.backbone,
            transform=trans_xaxes_ydata,
            ha="left",
            va="center",
            fontsize=BACKBONE_FONTSIZE,
            color=BACKBONE_COLOR,
            clip_on=False,
            zorder=4,
        )

    # y 轴标签
    labels = [e.label() for e in entries]
    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        rotation=YTICK_ROTATION,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_fontstyle("italic")
        tick.set_fontsize(YTICK_FONTSIZE)
        if entries[i].ours:
            tick.set_color("#7B2332")
        else:
            tick.set_color("#26384A")
        if entries[i].emphasis == "best":
            tick.set_fontweight("bold")

    # x 轴：刻度位置 = 整数（MultipleLocator），与标签一致；柱旁分数仍保留一位小数
    ax.set_xlabel(panel["xlabel"], fontsize=XLABEL_FONTSIZE)
    apply_integer_x_axis(ax, xmin, xmax)
    ax.tick_params(axis="x", labelsize=10.0)
    ax.tick_params(axis="y", pad=6)

    # 网格
    ax.xaxis.grid(True, linestyle=GRID_LS, color=GRID_COLOR, alpha=GRID_ALPHA, zorder=0)
    ax.yaxis.grid(False)

    # 标题：数据集 + 任务类型（cls. / det. / seg.）
    ax.set_title(
        TITLE_FMT.format(letter=letter, dataset=panel["dataset"], task=panel["task"]),
        fontsize=TITLE_FONTSIZE,
        loc="center",
        pad=4,
    )

    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)

    # 数值标注：主分数 + 更小字号的 delta（与论文 \tiny 一致）
    underline_texts: List = []
    for i, e in enumerate(entries):
        x0 = e.score + VALUE_DX_FRAC * span
        col = "#7B2332" if e.ours else "#23364D"
        fw = BEST_TEXT_WEIGHT if e.emphasis == "best" else "normal"
        t_main = ax.text(
            x0,
            y[i],
            format_score(e),
            va="center",
            ha="left",
            fontsize=VALUE_FONTSIZE,
            color=col,
            clip_on=False,
            zorder=5,
            fontweight=fw,
        )
        if e.emphasis == "second":
            underline_texts.append(t_main)
        if e.ours and e.delta is not None:
            main_s = format_score(e)
            x1 = x0 + (DELTA_CHAR_DX_FRAC * len(main_s) + DELTA_DX_FRAC) * span
            ax.text(
                x1,
                y[i],
                f" ({e.delta:+.1f})",
                va="center",
                ha="left",
                fontsize=DELTA_FONTSIZE,
                color=col,
                clip_on=False,
                zorder=5,
                fontweight="normal",
            )

    return underline_texts


# =============================================================================
# 主函数
# =============================================================================
def main():
    apply_style()
    if len(PANELS) != NROWS * NCOLS:
        raise ValueError(f"PANELS 数量 {len(PANELS)} 与 {NROWS}×{NCOLS} 不一致")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, dpi=DPI, facecolor=FIG_FACE)
    axes = axes.flatten()

    underline_targets = []
    letters = list("abcdefghijkl")

    for ax, panel, letter in zip(axes, PANELS, letters):
        underline_texts = draw_panel(ax, panel, letter)
        underline_targets.extend(underline_texts)

    # 总图例
    legend_handles = [
        Patch(facecolor=plt.get_cmap(PREV_CMAP_NAME)(0.58), edgecolor=plt.get_cmap(PREV_CMAP_NAME)(0.58),
              label="Previous methods"),
        Patch(facecolor=OURS_COLOR_L, edgecolor=OURS_EDGE_L, label="SARATR-X-v2 (ours)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.01),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout(
        rect=(0.055, 0.055, 0.995, 0.995),
        pad=0.35,
        h_pad=0.45,
        w_pad=0.4,
    )
    fig.canvas.draw()

    # 次佳结果加下划线
    for txt in underline_targets:
        add_underline(fig, txt, lw=SECOND_UNDERLINE_LW, pad_px=1.2)

    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", facecolor=FIG_FACE)
    plt.savefig(SAVE_PATH_PDF, dpi=DPI, bbox_inches="tight", facecolor=FIG_FACE, format="pdf")
    # plt.show()


if __name__ == "__main__":
    main()