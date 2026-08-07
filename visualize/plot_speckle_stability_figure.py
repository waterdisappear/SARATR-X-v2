"""
单独运行：根据 visualize_speckle_target_stability.py 生成的 CSV / JSON 重画折线图。
请在文件顶部修改 FIGSIZE、DPI、各类字体大小与 rcParams。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib import rcParams
from matplotlib.ticker import LogFormatter, LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np

# =============================================================================
# 图像与字体配置（按需修改）
# =============================================================================
FIGSIZE = (3.3, 4.4)
DPI = 300

FONT_RC = {
    "font.family": "Times New Roman",
    "font.sans-serif": "Times New Roman",
    "font.size": 9.0,
    "mathtext.fontset": "stix",
    "font.serif": "Times New Roman",
}

TITLE_FONTSIZE = 10.5
XLABEL_FONTSIZE = 9.0
YLABEL_FONTSIZE = 9.0
# y 轴标题与轴线距离（pt）；默认约 4，减小则标签靠近坐标轴、左侧留白可略减
YLABEL_LABELPAD = 1.0
LEGEND_FONTSIZE = 6.5
TICK_LABELSIZE = 8.0

# tight_layout 之后收紧左侧（figure 坐标，越小绘图区越靠左；勿过小以免纵轴标签裁切）
FIG_SUBPLOTS_ADJUST_LEFT = 0.05

# 不在图上显示标题（若需打开可改为 True）
SHOW_TITLE = False

# 在该扰动强度（如 σ=0.25）处为每条曲线标注名称；找不到则取最接近且误差 ≤ TOL 的点
ANNOTATE_AT_PERTURB = 0.25
ANNOTATE_MATCH_TOL = 0.02
ANNOTATION_FONTSIZE = 7.0
# Pixel / Multi：每个 σ（或 L）数据点旁写数值（科学计数法，三位有效数字）
SHOW_PIXEL_MULTI_POINT_VALUES = True
POINT_VALUE_FONTSIZE = 5.5
# 相对圆点的屏幕偏移（pt），避免盖住 marker
POINT_VALUE_OFFSET_PIXEL_PT = (0.0, 4.0)
POINT_VALUE_OFFSET_MULTI_PT = (0.0, -5.0)
# 标注基准：全体共用同一 x（xj + 偏移），仅 y 随曲线取值对齐
ANNOT_DX_DATA = 0.08
# S1/S3、S6/Multi 竖向略重叠：仅在文字锚点的 y 上做 log 域对称位移（ha/va 不变）
ANNOT_PAIR_Y_LOG_SHIFT = 0.018

# y 轴：在 log10 域相对数据 min/max 留白（加在 log 上，即上下为乘性边距）
# 底部略大：Multi 点在下方写科学计数法时易贴底裁切，需多留 log 域边距
Y_LIM_LOG_PAD_BOTTOM = 0.09
# 顶部略大：避免 Pixel 在最大 σ 处 marker/标注贴顶被裁切，并与上方图例留出空隙
Y_LIM_LOG_PAD_TOP = 0.095
# 每个 10^k 区间内画 2×10^k … 9×10^k 次刻度，网格更密，便于区分 S1/S3、S6/Multi 等接近的量级
Y_LOG_MINOR_SUBS = tuple(range(2, 10))
# True 时在次刻度也写数值（较密，易与主刻度叠字）；False 仅更密的网格线
Y_SHOW_MINOR_LABELS = False

# y 轴非线性：在 log10 域分段，压缩 Pixel～S2 之间「无曲线」的大间隔（仍标真实数值，仅视觉比例改变）
Y_USE_PIECEWISE_LOG_COMPRESS = True
# log10 分界：L<=split 保持标准 log 间距；L>split（更靠近 Pixel）乘 COMPRESS 压缩。None 则用 median(pixel) 与 median(S2) 的中点
Y_LOG_COMPRESS_SPLIT: Optional[float] = None
Y_LOG_COMPRESS_FACTOR = 0.24

# 线宽与标记（再缩小一档）
LINE_WIDTH_PIXEL = 0.95
LINE_WIDTH_SCALES = 1.05
LINE_WIDTH_MULTI = 1.55
MARKER_SIZE = 3.1
MARKER_SIZE_MULTI = 3.45
MARKER_EDGEWIDTH = 0.38
MARKER_EDGEWIDTH_MULTI = 0.45

# Pixel–S2 之间浅灰带：Patch 输入目标 vs 早期特征目标的漂移差（paired speckle）
SHOW_PIXEL_S2_BAND = True
PIXEL_S2_BAND_FACE = "#F2F2F2"
PIXEL_S2_BAND_ALPHA = 0.52
PIXEL_S2_BAND_ZORDER = 1
# 上行：灰带 Pixel–S2；下行：对应浅蓝带 S2–Multi（相对单层尺度）
PIXEL_S2_BAND_CAPTION = (
    "target features drift less than pixel inputs under perturbation\n"
    "fused multi-scale targets better than single-scale alone"
)
PIXEL_S2_BAND_CAPTION_FONTSIZE = 7.0
# 横坐标为 0.5 时与 ha="center" 搭配，整段在子图内水平居中
PIXEL_S2_BAND_CAPTION_AXES_POS = (0.5, 0.675)

# S2–Multi 之间浅蓝带（与 Blues 曲线系协调的冷灰蓝）
SHOW_S2_MULTI_BAND = True
S2_MULTI_BAND_FACE = "#E4EEF5"
S2_MULTI_BAND_ALPHA = 0.55
S2_MULTI_BAND_ZORDER = 1

# Pixel：灰色虚线（表示输入随扰动的变化）
PIXEL_LINE_COLOR = "#5C5C5C"
PIXEL_MARKER_FACE = "#6E6E6E"
PIXEL_MARKER_EDGE = "#404040"

# multi-scale：红色 + 实心圆点
MULTI_LINE_COLOR = "#B22222"
MULTI_MARKER_FACE = "#C41E3A"
MULTI_MARKER_EDGE = "#7A0A0A"

# S1..S6：蓝色渐变（拉开 t 区间，提高区分度）
BLUE_CMAP_NAME = "Blues"
BLUE_CMAP_T0 = 0.38
BLUE_CMAP_T1 = 0.98

NAMES = ["pixel"] + [f"S{k}" for k in range(1, 7)] + ["multi"]
NON_MULTI = ["pixel"] + [f"S{k}" for k in range(1, 7)]
PLOT_ORDER = NON_MULTI + ["multi"]  # 与绘制顺序一致，用于标注错位


def _fmt_sci_3sf(v: float) -> str:
    """科学计数法字符串，约三位有效数字（尾数两位小数 + 一位整数部分）。"""
    x = float(v)
    if not np.isfinite(x) or x <= 0:
        return ""
    return np.format_float_scientific(x, precision=2, exp_digits=1, min_digits=1)


def legend_display_name(internal: str) -> str:
    """图例与点旁标注：Pixel / S1..S6 / Multi。"""
    if internal == "pixel":
        return "Pixel"
    if internal == "multi":
        return "Multi"
    return internal  # S1, S2, ...


def _index_at_perturb(levels: List[float], target: float, tol: float) -> Optional[int]:
    for i, v in enumerate(levels):
        if abs(float(v) - target) < 1e-9:
            return i
    arr = np.array(levels, dtype=float)
    j = int(np.argmin(np.abs(arr - target)))
    if abs(arr[j] - target) <= tol:
        return j
    return None


def _blue_gradient_colors(n: int) -> List[tuple]:
    cmap = mcm.get_cmap(BLUE_CMAP_NAME)
    if n <= 1:
        return [cmap(BLUE_CMAP_T1)]
    t = np.linspace(BLUE_CMAP_T0, BLUE_CMAP_T1, n)
    return [cmap(float(x)) for x in t]


def _annot_xt_va(_name: str, xj: float) -> Tuple[float, str]:
    """标注锚点：Pixel / S1..S6 / Multi 共用同一 xt，yv 为各曲线数据。"""
    return xj + ANNOT_DX_DATA, "center"


def _annot_y_text_shifted(name: str, yv: float) -> float:
    """标签竖直位置：略上移 S1/S6、略下移 S3/Multi，拉开两对间距；曲线锚点 xy 仍为真实 yv。"""
    L = np.log10(max(yv, 1e-300))
    if name in ("S1", "S6"):
        return float(10 ** (L + ANNOT_PAIR_Y_LOG_SHIFT))
    if name in ("S3", "multi"):
        return float(10 ** (L - ANNOT_PAIR_Y_LOG_SHIFT))
    return float(yv)


def apply_style() -> None:
    rcParams.update(FONT_RC)


def _auto_compress_split_log10(series: Dict[str, List[float]]) -> float:
    """Pixel 与 S2 在 log 域的中点，作为分段压缩分界。"""
    p = np.clip(np.asarray(series["pixel"], dtype=float), 1e-300, None)
    s2 = np.clip(np.asarray(series["S2"], dtype=float), 1e-300, None)
    return 0.5 * (float(np.log10(np.median(p))) + float(np.log10(np.median(s2))))


def _make_compress_forward_inverse(L_split: float, compress: float):
    """f: y->z 在 log10 域分段线性，z 作为内部坐标；inverse 回到 y。"""

    def forward(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        L = np.log10(np.maximum(y, 1e-300))
        out = np.empty_like(L)
        lo = L <= L_split
        hi = ~lo
        out[lo] = L[lo]
        out[hi] = L_split + compress * (L[hi] - L_split)
        return out

    def inverse(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        lo = z <= L_split
        hi = ~lo
        out[lo] = z[lo]
        out[hi] = L_split + (z[hi] - L_split) / compress
        return 10.0 ** out

    return forward, inverse


def _apply_log_y_axis_ticks(
    ax: plt.Axes,
    series: Dict[str, List[float]],
    names: List[str],
) -> None:
    """收紧 y 范围并设置对数主次刻度，使接近的曲线在纵向上占满可用跨度。"""
    y_flat = np.concatenate(
        [np.clip(np.asarray(series[n], dtype=float), 1e-300, None) for n in names]
    )
    log_lo = float(np.log10(np.min(y_flat)))
    log_hi = float(np.log10(np.max(y_flat)))
    ax.set_ylim(
        10 ** (log_lo - Y_LIM_LOG_PAD_BOTTOM),
        10 ** (log_hi + Y_LIM_LOG_PAD_TOP),
    )

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=Y_LOG_MINOR_SUBS))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    if Y_SHOW_MINOR_LABELS:
        ax.yaxis.set_minor_formatter(
            LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.12))
        )
        ax.tick_params(axis="y", which="minor", labelsize=max(TICK_LABELSIZE - 1.0, 5.5))
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def _png_pdf_paths(out_path: str) -> Tuple[str, str]:
    """同一主文件名下写出 PNG 与 PDF（若给定 .pdf 则 PNG 为同名 .png，反之亦然）。"""
    low = out_path.lower()
    if low.endswith(".pdf"):
        base = out_path[:-4]
        return base + ".png", out_path
    if low.endswith(".png"):
        return out_path, out_path[:-4] + ".pdf"
    return out_path + ".png", out_path + ".pdf"


def plot_stability_lines(
    x_values: List[float],
    x_tick_labels: List[str],
    xlabel: str,
    series: Dict[str, List[float]],
    out_path: str,
    title: str = "",
    dpi: Optional[float] = None,
) -> None:
    apply_style()
    save_dpi = float(DPI if dpi is None else dpi)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    x = np.arange(len(x_values), dtype=float)

    scale_blues = _blue_gradient_colors(6)
    colors_by_name: Dict[str, object] = {"pixel": PIXEL_LINE_COLOR}
    for si, name in enumerate([f"S{k}" for k in range(1, 7)]):
        colors_by_name[name] = scale_blues[si]
    colors_by_name["multi"] = MULTI_LINE_COLOR

    y_p = np.clip(np.array(series["pixel"], dtype=float), 1e-16, None)
    y_s2 = np.clip(np.array(series["S2"], dtype=float), 1e-16, None)
    y_m = np.clip(np.array(series["multi"], dtype=float), 1e-16, None)

    if SHOW_PIXEL_S2_BAND:
        y_lo = np.minimum(y_p, y_s2)
        y_hi = np.maximum(y_p, y_s2)
        ax.fill_between(
            x,
            y_lo,
            y_hi,
            facecolor=PIXEL_S2_BAND_FACE,
            alpha=PIXEL_S2_BAND_ALPHA,
            edgecolor="none",
            linewidth=0,
            zorder=PIXEL_S2_BAND_ZORDER,
        )

    if SHOW_S2_MULTI_BAND:
        y_lo_m = np.minimum(y_s2, y_m)
        y_hi_m = np.maximum(y_s2, y_m)
        ax.fill_between(
            x,
            y_lo_m,
            y_hi_m,
            facecolor=S2_MULTI_BAND_FACE,
            alpha=S2_MULTI_BAND_ALPHA,
            edgecolor="none",
            linewidth=0,
            zorder=S2_MULTI_BAND_ZORDER,
        )

    # Pixel：灰色虚线（输入目标随 speckle 的变化）
    ax.plot(
        x,
        y_p,
        marker="o",
        ms=MARKER_SIZE,
        mfc=PIXEL_MARKER_FACE,
        mec=PIXEL_MARKER_EDGE,
        mew=MARKER_EDGEWIDTH,
        label=legend_display_name("pixel"),
        color=PIXEL_LINE_COLOR,
        linestyle="--",
        linewidth=LINE_WIDTH_PIXEL,
        zorder=2,
    )

    # S1..S6：蓝色渐变实线
    for si, name in enumerate([f"S{k}" for k in range(1, 7)]):
        y = np.clip(np.array(series[name], dtype=float), 1e-16, None)
        c = scale_blues[si]
        ax.plot(
            x,
            y,
            marker="o",
            ms=MARKER_SIZE,
            mfc=c,
            mec=c,
            mew=MARKER_EDGEWIDTH,
            label=legend_display_name(name),
            color=c,
            linestyle="-",
            linewidth=LINE_WIDTH_SCALES,
            zorder=3 + si,
        )

    # multi-scale：红色 + 实心点（最后画，置顶）
    ax.plot(
        x,
        y_m,
        marker="o",
        ms=MARKER_SIZE_MULTI,
        mfc=MULTI_MARKER_FACE,
        mec=MULTI_MARKER_EDGE,
        mew=MARKER_EDGEWIDTH_MULTI,
        label=legend_display_name("multi"),
        color=MULTI_LINE_COLOR,
        linestyle="-",
        linewidth=LINE_WIDTH_MULTI,
        zorder=20,
    )

    if Y_USE_PIECEWISE_LOG_COMPRESS:
        L_split = (
            float(Y_LOG_COMPRESS_SPLIT)
            if Y_LOG_COMPRESS_SPLIT is not None
            else _auto_compress_split_log10(series)
        )
        fwd, inv = _make_compress_forward_inverse(L_split, Y_LOG_COMPRESS_FACTOR)
        ax.set_yscale("function", functions=(fwd, inv))
    else:
        ax.set_yscale("log")
    _apply_log_y_axis_ticks(ax, series, NAMES)

    if SHOW_PIXEL_S2_BAND and PIXEL_S2_BAND_CAPTION.strip():
        ax.text(
            PIXEL_S2_BAND_CAPTION_AXES_POS[0],
            PIXEL_S2_BAND_CAPTION_AXES_POS[1],
            PIXEL_S2_BAND_CAPTION,
            transform=ax.transAxes,
            fontsize=PIXEL_S2_BAND_CAPTION_FONTSIZE,
            color="#000000",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=25,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels, fontsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.set_xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel(
        r"Mean $\ell_1$ difference",
        fontsize=YLABEL_FONTSIZE,
        labelpad=YLABEL_LABELPAD,
    )
    if SHOW_TITLE and title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # 在指定扰动（如 σ=0.25）处标注每条曲线类别
    j = _index_at_perturb(list(x_values), ANNOTATE_AT_PERTURB, ANNOTATE_MATCH_TOL)
    if j is not None:
        xj = float(x[j])
        for k, name in enumerate(PLOT_ORDER):
            yv = float(np.clip(series[name][j], 1e-16, None))
            txt_color = colors_by_name[name]
            xt, va = _annot_xt_va(name, xj)
            yt = _annot_y_text_shifted(name, yv)
            ax.annotate(
                legend_display_name(name),
                xy=(xj, yv),
                xytext=(xt, yt),
                textcoords="data",
                fontsize=ANNOTATION_FONTSIZE,
                color=txt_color,
                ha="left",
                va=va,
                zorder=30 + k,
            )

    if SHOW_PIXEL_MULTI_POINT_VALUES:
        npt = len(x_values)
        for i in range(npt):
            xi = float(x[i])
            y_pix = float(np.clip(series["pixel"][i], 1e-300, None))
            y_mul = float(np.clip(series["multi"][i], 1e-300, None))
            ax.annotate(
                _fmt_sci_3sf(y_pix),
                xy=(xi, y_pix),
                xytext=POINT_VALUE_OFFSET_PIXEL_PT,
                textcoords="offset points",
                fontsize=POINT_VALUE_FONTSIZE,
                color=PIXEL_LINE_COLOR,
                ha="center",
                va="bottom",
                zorder=38,
                clip_on=False,
            )
            ax.annotate(
                _fmt_sci_3sf(y_mul),
                xy=(xi, y_mul),
                xytext=POINT_VALUE_OFFSET_MULTI_PT,
                textcoords="offset points",
                fontsize=POINT_VALUE_FONTSIZE,
                color=MULTI_LINE_COLOR,
                ha="center",
                va="top",
                zorder=38,
                clip_on=False,
            )

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45, color="#B0B0B0")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35, color="#C8C8C8")
    ax.legend(
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#999999",
        facecolor="#FAFAFA",
        fontsize=LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.75),
        borderaxespad=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.subplots_adjust(left=FIG_SUBPLOTS_ADJUST_LEFT)
    png_path, pdf_path = _png_pdf_paths(out_path)
    save_kw = {"bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(png_path, dpi=save_dpi, **save_kw)
    fig.savefig(pdf_path, format="pdf", **save_kw)
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    print(f"              {pdf_path}")


def load_from_json(path: str) -> Tuple[List[float], List[str], str, Dict[str, List[float]], str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_condition = data["by_condition"]
    mode = data.get("speckle_mode", "lognormal")
    if not by_condition:
        raise ValueError("Empty by_condition in JSON")

    if "speckle_std" in by_condition[0]:
        param_key = "speckle_std"
        xlabel = r"Speckle strength $\sigma$ (log-normal multiplicative)"
    elif "speckle_L" in by_condition[0]:
        param_key = "speckle_L"
        xlabel = r"Effective looks $L$ (Gamma noise, mean$=$1)"
    else:
        raise KeyError("JSON row must contain speckle_std or speckle_L")

    levels = [float(row[param_key]) for row in by_condition]
    x_tick_labels = [f"{v:g}" for v in levels]
    series: Dict[str, List[float]] = {n: [] for n in NAMES}
    for row in by_condition:
        m = row["mean_over_images"]
        for n in NAMES:
            series[n].append(float(m[n]))

    K = data.get("num_realizations_K", "?")
    patch = data.get("patch_size", "?")
    title = rf"Target stability under speckle perturbation ($K$={K}, patch={patch}, mode={mode})"
    return levels, x_tick_labels, xlabel, series, title


def load_from_csv(path: str) -> Tuple[List[float], List[str], str, Dict[str, List[float]], str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV")
    if fields is None:
        raise ValueError("No header")
    if "speckle_std" in fields:
        param_key = "speckle_std"
        xlabel = r"Speckle strength $\sigma$ (log-normal multiplicative)"
    elif "speckle_L" in fields:
        param_key = "speckle_L"
        xlabel = r"Effective looks $L$ (Gamma noise, mean$=$1)"
    else:
        raise KeyError("CSV must have column speckle_std or speckle_L")

    levels = [float(row[param_key]) for row in rows]
    x_tick_labels = [f"{v:g}" for v in levels]
    series: Dict[str, List[float]] = {n: [] for n in NAMES}
    for row in rows:
        for n in NAMES:
            series[n].append(float(row[n]))
    title = "Target stability under speckle perturbation (from CSV)"
    return levels, x_tick_labels, xlabel, series, title


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot speckle stability lines from JSON or CSV.")
    p.add_argument(
        "--input",
        type=str,
        default=r"../results/visualize/speckle_stability_out/speckle_stability_summary.json",
        help="Path to speckle_stability_summary.json or speckle_stability_per_condition.csv",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output base path: writes both .png and .pdf with the same stem (default: .../speckle_stability_lines_replot.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input
    if not os.path.isfile(inp):
        raise FileNotFoundError(inp)

    ext = os.path.splitext(inp)[1].lower()
    if ext == ".json":
        levels, x_tick_labels, xlabel, series, title = load_from_json(inp)
    elif ext == ".csv":
        levels, x_tick_labels, xlabel, series, title = load_from_csv(inp)
    else:
        raise ValueError("Use .json or .csv")

    out = args.output
    if not out:
        d = os.path.dirname(os.path.abspath(inp))
        out = os.path.join(d, "speckle_stability_lines_replot.png")

    plot_stability_lines(levels, x_tick_labels, xlabel, series, out, title=title)


if __name__ == "__main__":
    main()
