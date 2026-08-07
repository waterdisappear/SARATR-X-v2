# -*- coding: utf-8 -*-
"""
Single wide horizontal lollipop chart
Style: clean publication-quality scientific figure
"""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # 如果本机有 Times New Roman 可改成它
    "font.size": 12.0,
    "axes.unicode_minus": False,
})
plt.rcParams["font.family"] = "Times New Roman"
# Colors
POS_COLOR = "#A31010"   # dark red
NEG_COLOR = "#B0B0B0"   # gray
AXIS_COLOR = "#333333"
SEP_COLOR = "#BDBDBD"
GROUP_TITLE_COLOR = "#A31010"  # 与正增益一致：Classification / Detection / Segmentation 用红
TEXT_COLOR = "#111111"
SUBTEXT_COLOR = "#7A7A7A"

# 顶部数据集 / 指标字号（各缩小约 1 级）
GROUP_TITLE_FONTSIZE = 11.0
DATASET_NAME_FONTSIZE = 8.4
METRIC_FONTSIZE = 6.7
VALUE_FONTSIZE = 9.5   # 圆旁 +x.x 数字

# 顶部数据集 / 指标：手动旋转坐标 + ax.text，列内居中对齐
DATASET_LABEL_ROTATION = 24   # 顺时针；仍重叠可略加大到 38
LABEL_CENTERS_GAP_PT = 50       # 两行中心间距（points）；越大两行越远

# 圆点 ↔ 数字：用「点(points)」偏移，与 DPI/图尺寸无关，避免小圆却空一大段
# 圆半径(点) = sqrt(s/π)；文字锚点距圆心 = 半径 + CIRCLE_TEXT_PAD_PT
#   · 仍太远 → 减小 CIRCLE_TEXT_PAD_PT（如 2.0）
#   · 仍重叠 → 增大 CIRCLE_TEXT_PAD_PT（如 4.5）
#   · 极小增量(|y|<0.35) 再乘 SMALL_Y_PAD_SCALE 略收紧
CIRCLE_TEXT_PAD_PT = 2.8
SMALL_Y_PAD_SCALE = 0.82
SMALL_Y_THRESH = 0.35

# =========================
# Data
# =========================
groups = {
    "Classification": [
        ("ATRNet-STAR",  "Acc.",   3.5),
        ("MSTAR",        "Acc.",   2.8),
        ("SAR-VSA",      "Acc.",   1.0),
        ("FUSAR-Ship",   "Acc.",   0.1),
    ],
    "Detection": [
        ("SARDet-100K",  "mAP",    5.4),
        ("RSAR",         "mAP@50",    4.2),
        ("SSDD", "mAP", 0.5),
        ("HRSID",        "mAP@50", -0.2),
    ],
    "Segmentation": [
        ("AIR-PolSAR-Seg-2.0", "mIoU", 3.2),
        ("DDHR-SK", "mIoU", 0.8),
        ("OpenEarthMap-SAR", "mIoU", 0.4),
        ("WHU-OPT-SAR", "PA", -0.3),
    ],
}

# =========================
# Layout positions
# =========================
gap = 1.0  # gap between groups
x_positions = []
labels_main = []
labels_sub = []
values = []
group_ranges = {}

x = 0
for gname, items in groups.items():
    start = x
    for name, metric, val in items:
        x_positions.append(x)
        labels_main.append(name)
        labels_sub.append(metric)
        values.append(val)
        x += 1
    end = x - 1
    group_ranges[gname] = (start, end)
    x += gap

x_positions = np.array(x_positions)
values = np.array(values)
abs_values = np.abs(values)

# 散点面积：按 |增量| 线性映射；最小 |Δ|（如 0.1）相对 s_min 先 ×0.5 再 ×0.7（再缩 30%），其余线性映射到 s_max
s_min, s_max = 85, 340
_AREA_MIN_REL_S = 0.5 * 0.7
s_floor = s_min * _AREA_MIN_REL_S
v_abs_min = float(abs_values.min())
v_abs_max = float(abs_values.max())
if v_abs_max - v_abs_min < 1e-9:
    scatter_sizes = np.full(len(values), (s_floor + s_max) / 2.0)
else:
    scatter_sizes = s_floor + (abs_values - v_abs_min) / (v_abs_max - v_abs_min) * (s_max - s_floor)

# 纵向排版：整体上移，与最高数值标注（约 5.85）拉开；任务类型 / 灰线 / 数据集 / 指标逐级拉开
Y_BOTTOM, Y_TOP = -1.4, 8.35
Y_GROUP_TITLE = 7.82 + 0.7
Y_GROUP_RULE = 7.58 + 0.35
Y_DATASET = 7.0 + 0.3
Y_LABEL = Y_DATASET - 0.4   # 两行标签组的几何中心锚点（灰线下方）
LW_SCATTER = 2.15


def _circle_radius_pt(s: float) -> float:
    """matplotlib scatter 的 s 为面积(点²)，半径 = sqrt(s/π)。"""
    return float(np.sqrt(max(float(s), 1.0) / np.pi))


def _value_label_offset_pt(y: float, s: float) -> Tuple[int, int]:
    """相对圆心的 offset points：(0, +dy) 正增益在上，(0, -dy) 负增益在下。"""
    pad = CIRCLE_TEXT_PAD_PT
    if abs(y) < SMALL_Y_THRESH:
        pad *= SMALL_Y_PAD_SCALE
    dy = _circle_radius_pt(s) + pad
    return (0, int(round(dy))) if y >= 0 else (0, int(round(-dy)))


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(9, 3.8), dpi=600)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Baseline y=0
ax.axhline(0, color=AXIS_COLOR, lw=1.1, zorder=1)

# Lollipop stems + 白色实心圆（彩色描边表示正负；面积随 |增量|）
for x, y, s in zip(x_positions, values, scatter_sizes):
    color = POS_COLOR if y > 0 else NEG_COLOR
    ax.vlines(x, 0, y, color=color, lw=2.2, zorder=2)
    lw = max(1.35, min(LW_SCATTER, 0.9 + 1.4 * (float(s) / s_max)))
    ax.scatter(
        x,
        y,
        s=s,
        facecolors="white",
        edgecolors=color,
        linewidths=lw,
        zorder=3,
    )

# Value annotations：offset points 贴在圆外缘，间距随圆半径自动缩放
for x, y, s in zip(x_positions, values, scatter_sizes):
    color = POS_COLOR if y > 0 else NEG_COLOR
    dx, dy = _value_label_offset_pt(y, s)
    ax.annotate(
        f"{y:+.1f}",
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="bottom" if y >= 0 else "top",
        color=color,
        fontsize=VALUE_FONTSIZE,
        fontweight="bold",
        annotation_clip=False,
    )

# 数据集名与指标：放在各组灰横线下方（不用 x 轴刻度）
ax.set_xticks([])
ax.tick_params(axis="x", length=0, labelbottom=False)

# Y 轴：下界留空；刻度仅 0～+6（不显示 -1）
ax.set_ylim(Y_BOTTOM, Y_TOP)
ax.set_yticks(list(range(0, 7)))
ax.set_yticklabels(["0", "+1", "+2", "+3", "+4", "+5", "+6"])
ax.tick_params(axis="y", length=4, width=1.0, color=AXIS_COLOR, labelsize=12.5, pad=8)

# Gain (pp) 作为 y 轴标签；loc='center' 沿 y 轴垂直居中
ax.set_ylabel(
    "Gain (pp)",
    fontsize=13.5,
    fontweight="bold",
    color=TEXT_COLOR,
    labelpad=12,
    loc="center",
)

# 分组之间不再用竖向虚线分隔（改为顶部分组灰横线）

def draw_group_header(ax, x0: float, x1: float, y_text: float, y_rule: float, text: str) -> None:
    """红色任务类型 + 其下灰横线。"""
    mid = (x0 + x1) / 2
    ax.text(
        mid,
        y_text,
        text,
        ha="center",
        va="top",
        fontsize=GROUP_TITLE_FONTSIZE,
        color=GROUP_TITLE_COLOR,
        fontweight="bold",
        clip_on=False,
    )
    ax.plot(
        [x0, x1],
        [y_rule, y_rule],
        color=SEP_COLOR,
        lw=1.35,
        solid_capstyle="round",
        clip_on=False,
        zorder=2,
    )


def format_metric_bracket(metric: str) -> str:
    m = (metric or "").strip()
    if m.startswith("(") and m.endswith(")"):
        return m
    return f"({m})"


def _rotate_around(x_in: float, y_in: float, cx: float, cy: float, rad: float) -> Tuple[float, float]:
    """绕 (cx, cy) 旋转 rad 弧度。"""
    dx = x_in - cx
    dy = y_in - cy
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def draw_column_labels(ax, x: float, name: str, metric: str) -> None:
    """
    双色标签：数据集名（上）+ 指标（下），绕 (x, Y_LABEL) 旋转，水平居中。
    仅用 ax.text + transData 换算间距，无 offsetbox。
    """
    inv = ax.transData.inverted()
    dy_data = abs(inv.transform((0, LABEL_CENTERS_GAP_PT))[1] - inv.transform((0, 0))[1])

    center_name = (x, Y_LABEL + dy_data / 2)
    center_metric = (x, Y_LABEL - dy_data / 2)

    theta = DATASET_LABEL_ROTATION
    rad = math.radians(theta)
    x_name, y_name = _rotate_around(*center_name, x, Y_LABEL, rad)
    x_metric, y_metric = _rotate_around(*center_metric, x, Y_LABEL, rad)

    ax.text(
        x_name,
        y_name,
        name,
        ha="center",
        va="center",
        rotation=theta,
        rotation_mode="anchor",
        fontsize=DATASET_NAME_FONTSIZE,
        color=TEXT_COLOR,
        clip_on=False,
        zorder=4,
    )
    ax.text(
        x_metric,
        y_metric,
        format_metric_bracket(metric),
        ha="center",
        va="center",
        rotation=theta,
        rotation_mode="anchor",
        fontsize=METRIC_FONTSIZE,
        color=SUBTEXT_COLOR,
        clip_on=False,
        zorder=4,
    )


pad = 0.15
for gname, (start, end) in group_ranges.items():
    draw_group_header(ax, start - pad, end + pad, Y_GROUP_TITLE, Y_GROUP_RULE, gname)

# 灰线下方：倾斜 + 列内居中对齐的数据集名 / 指标
for x, name, metric in zip(x_positions, labels_main, labels_sub):
    draw_column_labels(ax, float(x), name, metric)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_color(AXIS_COLOR)
ax.spines["left"].set_linewidth(1.0)

# X limits
ax.set_xlim(-0.85, x_positions.max() + 0.7)

# Optional summary strip (可删)
# summary = "7/9 wins    |    mean gain: +1.9 pp    |    Cls. 4/4    |    Det. 2/4    |    Seg. 1/1"
# ax.text(
#     0.5, -0.22, summary,
#     transform=ax.transAxes,
#     ha="center", va="top",
#     fontsize=10.5, color=TEXT_COLOR
# )

plt.tight_layout(rect=[0.03, 0.06, 0.995, 0.94])

# Save
plt.savefig("./visualize/sar_gain_single_wide_lollipop.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("./visualize/sar_gain_single_wide_lollipop.pdf", dpi=300, bbox_inches="tight", facecolor="white")

# if plt.get_backend().lower() != "agg":
    # plt.show()