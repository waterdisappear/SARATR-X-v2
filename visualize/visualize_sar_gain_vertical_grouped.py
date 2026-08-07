# -*- coding: utf-8 -*-
"""
Vertical grouped bar chart (reference-style layout)
- benchmark labels on the left
- gain on x-axis
- portrait layout
- keep original colors and values
"""

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Global style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12.0,
    "axes.unicode_minus": False,
})

# Colors
POS_COLOR = "#A31010"        # original dark red
NEG_COLOR = "#B0B0B0"        # original gray
POS_EDGE = "#7F0C0C"
NEG_EDGE = "#8E8E8E"
AXIS_COLOR = "#333333"
GRID_COLOR = "#E6E6E6"
GROUP_COLOR = "#1E4E9E"
TEXT_COLOR = "#111111"
SUBTEXT_COLOR = "#4F4F4F"

# Font sizes
DATASET_FONTSIZE = 11.6
METRIC_FONTSIZE = 9.8
VALUE_FONTSIZE = 10.8
GROUP_FONTSIZE = 12.2
TOP_TITLE_FONTSIZE = 13.2
X_LABEL_FONTSIZE = 12.8

# =========================
# Data
# 排序/排版参考示意图；颜色与数值保持原始设置
# =========================
groups = {
    "Cls.": [
        ("ATRNet-STAR", "Acc.", 3.5),
        ("MSTAR", "Acc.", 2.8),
        ("FUSAR-Ship", "Acc.", 0.1),
        ("SAR-VSA", "Acc.", 1.0),
    ],
    "Det.": [
        ("SARDet-100K", "mAP", 5.4),
        ("HRSID", "mAP@50", -0.2),
        ("SSDD", "mAP", 1.4),
        ("RSAR", "mAP@50", 4.4),
    ],
    "Seg.": [
        ("AIR-PolSAR-\nSeg-2.0", "mIoU", 3.2),
        ("DDHR-SK", "mIoU", 0.8),
        ("OpenEarthMap-SAR", "mIoU", 0.4),
        ("WHU-OPT-SAR", "PA", -0.3),
    ],
}

# =========================
# Layout positions
# =========================
bar_h = 0.72
row_gap = 1.05
group_gap = 0.90

y_positions = []
labels_main = []
labels_sub = []
values = []
group_ranges = {}

y = 0.0
for gname, items in groups.items():
    start_y = y
    for name, metric, val in items:
        y_positions.append(y)
        labels_main.append(name)
        labels_sub.append(metric)
        values.append(val)
        y += row_gap
    end_y = y - row_gap
    group_ranges[gname] = (start_y, end_y)
    y += group_gap

y_positions = np.array(y_positions, dtype=float)
values = np.array(values, dtype=float)

# =========================
# Figure / axes
# =========================
fig, ax = plt.subplots(figsize=(5.3, 9.8), dpi=600)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Axis range
x_min = min(-2.7, float(values.min()) - 0.55)
x_max = max(6.2, float(values.max()) + 0.70)

# Left label / bracket region
x_text = x_min - 1.45
x_bracket = x_text - 0.40
x_left_lim = x_bracket - 0.48
x_right_lim = x_max + 0.22

# =========================
# Horizontal guide lines
# =========================
for yy in y_positions:
    ax.hlines(
        yy, x_min, x_max,
        color=GRID_COLOR,
        lw=0.8,
        linestyles=(0, (2, 2)),
        zorder=0
    )

# Zero line
ax.axvline(0, color=AXIS_COLOR, lw=1.05, zorder=1)

# =========================
# Bars
# =========================
for yy, val in zip(y_positions, values):
    if val >= 0:
        ax.barh(
            yy,
            val,
            left=0,
            height=bar_h,
            color=POS_COLOR,
            edgecolor=POS_EDGE,
            linewidth=1.2,
            zorder=2,
        )
    else:
        ax.barh(
            yy,
            val,
            left=0,
            height=bar_h,
            color=NEG_COLOR,
            edgecolor=NEG_EDGE,
            linewidth=1.2,
            zorder=2,
        )

# =========================
# Value labels
# =========================
for yy, val in zip(y_positions, values):
    if val >= 0:
        ax.text(
            val + 0.14, yy,
            f"{val:+.1f}",
            ha="left", va="center",
            fontsize=VALUE_FONTSIZE,
            color=POS_COLOR,
            fontweight="bold",
        )
    else:
        ax.text(
            val + 0.18, yy,
            f"{val:+.1f}",
            ha="left", va="center",
            fontsize=VALUE_FONTSIZE,
            color=NEG_EDGE,
            fontweight="bold",
        )

# =========================
# Left labels: dataset + metric
# =========================
for yy, name, metric in zip(y_positions, labels_main, labels_sub):
    ax.text(
        x_text, yy - 0.16,
        name,
        ha="left", va="center",
        fontsize=DATASET_FONTSIZE,
        color=TEXT_COLOR,
    )
    ax.text(
        x_text, yy + 0.26,
        f"({metric})",
        ha="left", va="center",
        fontsize=METRIC_FONTSIZE,
        color=SUBTEXT_COLOR,
        fontstyle="italic",
    )

# =========================
# Group bracket
# =========================
def draw_group_bracket(ax, x, y0, y1, label, hook=0.18):
    top = y0 - bar_h * 0.92
    bottom = y1 + bar_h * 0.92

    ax.plot([x + hook, x], [top, top], color=GROUP_COLOR, lw=1.35, solid_capstyle="round", clip_on=False)
    ax.plot([x, x], [top, bottom], color=GROUP_COLOR, lw=1.35, solid_capstyle="round", clip_on=False)
    ax.plot([x, x + hook], [bottom, bottom], color=GROUP_COLOR, lw=1.35, solid_capstyle="round", clip_on=False)

    ax.text(
        x - 0.05, (top + bottom) / 2,
        label,
        ha="right", va="center",
        fontsize=GROUP_FONTSIZE,
        color=GROUP_COLOR,
        fontweight="bold",
    )

for gname, (y0, y1) in group_ranges.items():
    draw_group_bracket(ax, x_bracket, y0, y1, gname)

# =========================
# Top title
# =========================
ax.text(
    (0 + x_max) / 2,
    y_positions.min() - 1.02,
    "Gain (pp)",
    ha="center", va="center",
    fontsize=TOP_TITLE_FONTSIZE,
    color=POS_COLOR,
    fontweight="bold",
    clip_on=False,
)

# =========================
# Axes settings
# =========================
ax.set_xlim(x_left_lim, x_right_lim)
ax.set_ylim(y_positions.min() - 1.10, y_positions.max() + 1.00)
ax.invert_yaxis()

ax.set_yticks([])
ax.set_xticks([-2, 0, 2, 4, 6])
ax.tick_params(axis="x", labelsize=12, width=1.0, length=4, color=AXIS_COLOR)

ax.set_xlabel(
    "Absolute gain (percentage points)",
    fontsize=X_LABEL_FONTSIZE,
    color=TEXT_COLOR,
    labelpad=8,
)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(AXIS_COLOR)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout(rect=[0.06, 0.04, 0.99, 0.98])

# Save
plt.savefig("./visualize/sar_gain_vertical_grouped.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("./visualize/sar_gain_vertical_grouped.pdf", dpi=300, bbox_inches="tight", facecolor="white")

# if plt.get_backend().lower() != "agg":
#     plt.show()