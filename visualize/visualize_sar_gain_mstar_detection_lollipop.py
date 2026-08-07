# -*- coding: utf-8 -*-
"""
MSTAR classification + SAR detection gain lollipop (single wide)
Data: Table B.3 — SARATR-X vs. second-best (underlined) in pp
Style: visualize_sar_gain_single_wide_lollipop.py
"""

import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.0,
    "axes.unicode_minus": False,
})
plt.rcParams["font.family"] = "Times New Roman"

POS_COLOR = "#A31010"
NEG_COLOR = "#B0B0B0"
AXIS_COLOR = "#333333"
SEP_COLOR = "#BDBDBD"
GROUP_TITLE_COLOR = "#A31010"
TEXT_COLOR = "#111111"
SUBTEXT_COLOR = "#7A7A7A"

GROUP_TITLE_FONTSIZE = 11.0
DATASET_NAME_FONTSIZE = 8.0
METRIC_FONTSIZE = 6.4
VALUE_FONTSIZE = 9.0

DATASET_LABEL_ROTATION = 24
LABEL_CENTERS_GAP_PT = 50

CIRCLE_TEXT_PAD_PT = 2.8
SMALL_Y_PAD_SCALE = 0.82
SMALL_Y_THRESH = 0.35

# =========================
# Data — SARATR-X gains over second-best (scriptsize in tables)
# =========================
groups = {
    "Classification": [
        ("SOCs", "1-shot", 4.5),
        ("EOC-Dep.", "1-shot", 31.3),   # 俯仰角
        ("EOC-Cfg.", "1-shot", 2.2),    # 构型
        ("EOC-Ver.", "1-shot", 11.8),   # 版本
    ],
    "Detection": [
        ("SARDet-100K", "mAP", 0.9),
        ("OGSOD", "mAP", 6.9),
        ("SSDD", "mAP", 2.6),           # 表为 AP，统一标 mAP
        ("SAR-Aircraft", "mAP@50", 5.7),
    ],
}

# =========================
# Layout
# =========================
gap = 1.0
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

s_min, s_max = 70, 360
_AREA_MIN_REL_S = 0.5 * 0.7
s_floor = s_min * _AREA_MIN_REL_S
v_abs_min = float(abs_values.min())
v_abs_max = float(abs_values.max())
if v_abs_max - v_abs_min < 1e-9:
    scatter_sizes = np.full(len(values), (s_floor + s_max) / 2.0)
else:
    scatter_sizes = s_floor + (abs_values - v_abs_min) / (v_abs_max - v_abs_min) * (s_max - s_floor)

v_max = float(values.max())
v_min = float(values.min())
y_data_top = max(8.0, math.ceil(v_max * 1.12))
y_bottom = -1.35 if v_min >= 0 else min(-1.4, v_min - 0.8)

# Y ticks: 0 .. y_data_top, step 5 if range > 15 else step 2
if y_data_top > 15:
    y_tick_step = 5
else:
    y_tick_step = 2
y_ticks = list(range(0, int(y_data_top) + 1, y_tick_step))
if y_ticks[-1] < y_data_top:
    y_ticks.append(int(math.ceil(y_data_top / y_tick_step) * y_tick_step))
y_tick_labels = ["0"] + [f"+{t}" for t in y_ticks[1:]]

# 2 组 × 4 项，最大增益 31.3（EOC-Dep.）；顶栏比例对齐 single_wide，并按 v_max 抬高 ylim
_TOP_BAND = 2.35
Y_TOP = max(float(y_ticks[-1]) + _TOP_BAND, v_max + 6.2)
Y_GROUP_TITLE = Y_TOP + 3.52
Y_GROUP_RULE = Y_TOP + 1.02
Y_DATASET = Y_TOP - 1.05
Y_LABEL = Y_DATASET - 0.4
LW_SCATTER = 2.15

n_cols = len(values)
fig_w = 9.0


def _circle_radius_pt(s: float) -> float:
    return float(np.sqrt(max(float(s), 1.0) / np.pi))


def _value_label_offset_pt(y: float, s: float) -> Tuple[int, int]:
    pad = CIRCLE_TEXT_PAD_PT
    if abs(y) < SMALL_Y_THRESH:
        pad *= SMALL_Y_PAD_SCALE
    dy = _circle_radius_pt(s) + pad
    return (0, int(round(dy))) if y >= 0 else (0, int(round(-dy)))


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(fig_w, 3.9), dpi=600)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.axhline(0, color=AXIS_COLOR, lw=1.1, zorder=1)

for xi, y, s in zip(x_positions, values, scatter_sizes):
    color = POS_COLOR if y > 0 else NEG_COLOR
    ax.vlines(xi, 0, y, color=color, lw=2.2, zorder=2)
    lw = max(1.35, min(LW_SCATTER, 0.9 + 1.4 * (float(s) / s_max)))
    ax.scatter(
        xi, y, s=s, facecolors="white", edgecolors=color,
        linewidths=lw, zorder=3,
    )

for xi, y, s in zip(x_positions, values, scatter_sizes):
    color = POS_COLOR if y > 0 else NEG_COLOR
    dx, dy = _value_label_offset_pt(y, s)
    ax.annotate(
        f"{y:+.1f}",
        xy=(xi, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="bottom" if y >= 0 else "top",
        color=color,
        fontsize=VALUE_FONTSIZE,
        fontweight="bold",
        annotation_clip=False,
    )

ax.set_xticks([])
ax.tick_params(axis="x", length=0, labelbottom=False)

ax.set_ylim(y_bottom, Y_TOP)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels[: len(y_ticks)])
ax.tick_params(axis="y", length=4, width=1.0, color=AXIS_COLOR, labelsize=12.0, pad=8)

ax.set_ylabel(
    "Gain (pp)",
    fontsize=13.5,
    fontweight="bold",
    color=TEXT_COLOR,
    labelpad=12,
    loc="center",
)


def draw_group_header(ax, x0: float, x1: float, y_text: float, y_rule: float, text: str) -> None:
    mid = (x0 + x1) / 2
    ax.text(
        mid, y_text, text,
        ha="center", va="top",
        fontsize=GROUP_TITLE_FONTSIZE,
        color=GROUP_TITLE_COLOR,
        fontweight="bold",
        clip_on=False,
    )
    ax.plot(
        [x0, x1], [y_rule, y_rule],
        color=SEP_COLOR, lw=1.35, solid_capstyle="round",
        clip_on=False, zorder=2,
    )


def format_metric_bracket(metric: str) -> str:
    m = (metric or "").strip()
    if m.startswith("(") and m.endswith(")"):
        return m
    return f"({m})"


def _rotate_around(x_in: float, y_in: float, cx: float, cy: float, rad: float) -> Tuple[float, float]:
    dx = x_in - cx
    dy = y_in - cy
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def draw_column_labels(ax, x: float, name: str, metric: str) -> None:
    inv = ax.transData.inverted()
    dy_data = abs(inv.transform((0, LABEL_CENTERS_GAP_PT))[1] - inv.transform((0, 0))[1])

    center_name = (x, Y_LABEL + dy_data / 2)
    center_metric = (x, Y_LABEL - dy_data / 2)

    theta = DATASET_LABEL_ROTATION
    rad = math.radians(theta)
    x_name, y_name = _rotate_around(*center_name, x, Y_LABEL, rad)
    x_metric, y_metric = _rotate_around(*center_metric, x, Y_LABEL, rad)

    ax.text(
        x_name, y_name, name,
        ha="center", va="center",
        rotation=theta, rotation_mode="anchor",
        fontsize=DATASET_NAME_FONTSIZE,
        color=TEXT_COLOR, clip_on=False, zorder=4,
    )
    ax.text(
        x_metric, y_metric, format_metric_bracket(metric),
        ha="center", va="center",
        rotation=theta, rotation_mode="anchor",
        fontsize=METRIC_FONTSIZE,
        color=SUBTEXT_COLOR, clip_on=False, zorder=4,
    )


pad = 0.18
for gname, (start, end) in group_ranges.items():
    draw_group_header(ax, start - pad, end + pad, Y_GROUP_TITLE, Y_GROUP_RULE, gname)

for xi, name, metric in zip(x_positions, labels_main, labels_sub):
    draw_column_labels(ax, float(xi), name, metric)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_color(AXIS_COLOR)
ax.spines["left"].set_linewidth(1.0)

ax.set_xlim(-0.9, x_positions.max() + 0.75)

plt.tight_layout(rect=[0.03, 0.06, 0.995, 0.94])

_save_dir = os.path.dirname(os.path.abspath(__file__))
_out_base = os.path.join(_save_dir, "sar_gain_mstar_detection_lollipop")
plt.savefig(f"{_out_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(f"{_out_base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {_out_base}.png / .pdf")
