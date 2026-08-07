# -*- coding: utf-8 -*-
"""
Stability vs transfer scatter: ℓ₁ drift (speckle) × SOC 10-shot linear accuracy.

绘图/保存风格与 visualize_speckle_target_stability.py → plot_speckle_stability_figure.py 一致。

================================================================================
数据来源（脚本运行时从 JSON 读取，非硬编码）
================================================================================

【x 轴：Mean ℓ₁ difference】
  文件 : visualize/speckle_stability_out_50ep/speckle_stability_summary.json
  生成 : python visualize/visualize_speckle_target_stability.py
         --checkpoint D:\\2024_SARatrX_2\\MIM_weight\\500K\\base\\old_2\\jiaquan_simple\\checkpoint-50.pth
         --output_dir visualize/speckle_stability_out_50ep
         （默认 64 张 held-out SAR、K=5 speckle 对、patch=16、σ  sweep）
  取点 : by_condition 中 speckle_std=0.15 的 mean_over_images.{pixel,S1..S6,multi}
  说明 : 与 50-epoch 转移实验配套：Multi 的融合权重 α 取自上述 checkpoint-50；
         Pixel/S1–S6 不依赖 α（与 1200-epoch Fig.4 数值相同）。
         Fig.4 仍用 visualize/speckle_stability_out/（checkpoint-1200）。

【y 轴：10-shot accuracy (%)】
  文件 : visualize/target_ablation_soc_10shot_summary.json
  生成 : python visualize/run_linear_eval_target_ablation.py
         （finetune_HiVit / MIM_linear_itpn / SOC_50classes / 10-shot / 3 seeds）
  取点 : results.<mode>.accuracy_mean（mode = pixel | single_s1..s6 | multi）
  权重 : D:\\2024_SARatrX_2\\result\\base\\target\\base\\<mode>\\checkpoint-49.pth
         （50 epoch target ablation 预训练，见 visualize/run_pretrain_target_ablation.sh）

【σ=0.15 时 8 个点数值（Multi α = checkpoint-50 / old_2/jiaquan_simple）】
  Spearman ρ ≈ -0.93, exact two-sided p ≈ 0.0022 (n=8, 8! permutations)
  注意：scipy.stats.spearmanr 的渐近 p≈8.6e-4 会误报为 p<0.001；小样本须用精确置换 p。

  Target   drift_l1 (x)     acc±std (y, %)     来源 key
  ------   --------------   ----------------   ----------
  Pixel    1.174e-02        32.9 ± 0.29        pixel / pixel
  S1       3.000e-04        35.2 ± 0.14        S1 / single_s1
  S2       4.468e-04        29.1 ± 0.14        S2 / single_s2
  S3       2.987e-04        33.6 ± 0.21        S3 / single_s3
  S4       1.949e-04        37.2 ± 0.36        S4 / single_s4
  S5       1.515e-04        38.4 ± 0.28        S5 / single_s5
  S6       1.267e-04        38.3 ± 0.12        S6 / single_s6
  Multi    1.204e-04        39.2 ± 0.21        multi / multi  (α@50ep)

  合并表输出 : visualize/stability_transfer_out/stability_transfer_xy.csv

Usage:
  python visualize/visualize_stability_transfer_scatter.py
  python visualize/visualize_stability_transfer_scatter.py --speckle_std 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.stats import spearmanr

# n! enumeration is exact and fast for n<=9 (9!=362880); above that use asymptotic.
_EXACT_SPEARMAN_MAX_N = 9

VIS_ROOT = Path(__file__).resolve().parent
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))

from plot_speckle_stability_figure import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    DPI,
    FIG_SUBPLOTS_ADJUST_LEFT,
    LEGEND_FONTSIZE,
    LINE_WIDTH_PIXEL,
    MARKER_EDGEWIDTH,
    MARKER_EDGEWIDTH_MULTI,
    MARKER_SIZE,
    MARKER_SIZE_MULTI,
    MULTI_LINE_COLOR,
    MULTI_MARKER_EDGE,
    MULTI_MARKER_FACE,
    PIXEL_LINE_COLOR,
    PIXEL_MARKER_EDGE,
    PIXEL_MARKER_FACE,
    TICK_LABELSIZE,
    XLABEL_FONTSIZE,
    YLABEL_FONTSIZE,
    YLABEL_LABELPAD,
    Y_LOG_MINOR_SUBS,
    _blue_gradient_colors,
    _png_pdf_paths,
    apply_style,
    legend_display_name,
)

DEFAULT_SPECKLE_JSON = VIS_ROOT / "speckle_stability_out_50ep" / "speckle_stability_summary.json"
DEFAULT_TRANSFER_JSON = VIS_ROOT / "target_ablation_soc_10shot_summary.json"
DEFAULT_OUT_DIR = VIS_ROOT / "stability_transfer_out"

# 实验结果数据已迁移到仓库 results/visualize/ 目录（见 readme）。
# 优先读取 results/visualize/ 下的数据；若不存在则回退到 visualize/ 本地副本。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPECKLE_JSON = _REPO_ROOT / "results" / "visualize" / "speckle_stability_out_50ep" / "speckle_stability_summary.json"
_DEFAULT_TRANSFER_JSON = _REPO_ROOT / "results" / "visualize" / "target_ablation_soc_10shot_summary.json"
if _DEFAULT_SPECKLE_JSON.exists():
    DEFAULT_SPECKLE_JSON = _DEFAULT_SPECKLE_JSON
if _DEFAULT_TRANSFER_JSON.exists():
    DEFAULT_TRANSFER_JSON = _DEFAULT_TRANSFER_JSON

# ---------------------------------------------------------------------------
# 默认输入路径（相对 visualize/）；见文件头 docstring 中 8 点数值表
# ---------------------------------------------------------------------------
# drift  : speckle_stability_out_50ep/...（α from checkpoint-50, paired with 50-epoch transfer）
# transfer: target_ablation_soc_10shot_summary.json → results[mode].accuracy_mean
# weights: D:/2024_SARatrX_2/result/base/target/base/<mode>/checkpoint-49.pth

# 相对 speckle 参考图字号 +2 pt
FONT_BUMP = 2
FS_ANNOT = ANNOTATION_FONTSIZE + FONT_BUMP
FS_XLABEL = XLABEL_FONTSIZE + FONT_BUMP
FS_YLABEL = YLABEL_FONTSIZE + FONT_BUMP
FS_TICK = TICK_LABELSIZE + FONT_BUMP
FS_LEGEND = LEGEND_FONTSIZE + FONT_BUMP

# 横版 ~1.4:1；顶部留图例区
SCATTER_FIGSIZE = (5.2, 3.8)

# 与 speckle 参考图 y 轴一致
X_LABEL = r"Mean $\ell_1$ difference"
Y_LABEL = "10-shot accuracy (%)\n(ATRNet-STAR, SOC-50)"

# 图例顺序与 speckle 折线图一致
LEGEND_ORDER = ["pixel"] + [f"S{k}" for k in range(1, 7)] + ["multi"]

MODE_ORDER = [
    ("pixel", "pixel", "Pixel"),
    ("S1", "single_s1", "S1"),
    ("S2", "single_s2", "S2"),
    ("S3", "single_s3", "S3"),
    ("S4", "single_s4", "S4"),
    ("S5", "single_s5", "S5"),
    ("S6", "single_s6", "S6"),
    ("multi", "multi", "Fused"),
]

GUIDE_LINE_COLOR = "#B0B0B0"
GRID_MAJOR = {"linestyle": "--", "linewidth": 0.7, "alpha": 0.45, "color": "#B0B0B0"}
GRID_MINOR = {"linestyle": ":", "linewidth": 0.5, "alpha": 0.35, "color": "#C8C8C8"}


def _scatter_s(markersize: float) -> float:
    return (markersize * 2.15) ** 2


def load_drift_at_sigma(speckle_json: Path, sigma: float) -> dict[str, float]:
    data = json.loads(speckle_json.read_text(encoding="utf-8"))
    best = None
    best_diff = float("inf")
    for row in data["by_condition"]:
        if "speckle_std" not in row:
            continue
        diff = abs(float(row["speckle_std"]) - sigma)
        if diff < best_diff:
            best_diff = diff
            best = row["mean_over_images"]
    if best is None:
        raise ValueError(f"No speckle_std≈{sigma} in {speckle_json}")
    return {k: float(v) for k, v in best.items()}


def load_transfer(transfer_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(transfer_json.read_text(encoding="utf-8"))
    return data["results"]


def build_xy_table(
    drift: dict[str, float],
    transfer: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speckle_key, transfer_key, label in MODE_ORDER:
        if speckle_key not in drift:
            raise KeyError(f"Missing drift key: {speckle_key}")
        if transfer_key not in transfer:
            raise KeyError(f"Missing transfer key: {transfer_key}")
        t = transfer[transfer_key]
        rows.append({
            "mode": transfer_key,
            "speckle_key": speckle_key,
            "label": label,
            "drift_l1": drift[speckle_key],
            "transfer_acc": float(t["accuracy_mean"]),
            "transfer_acc_std": float(t.get("accuracy_std", 0.0)),
            "macro_f1": float(t.get("macro_f1_mean", 0.0)),
        })
    return rows


def spearmanr_exact_twosided(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> tuple[float, float, str]:
    """Spearman ρ with exact two-sided p via full permutation when n is small.

    scipy.stats.spearmanr uses a large-n asymptotic approximation. For n=8 that
    approximation understates p (e.g. ~8.6e-4 vs exact ~2.2e-3), which can
    incorrectly trigger a ``p<0.001`` claim.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = int(x_arr.size)
    rho, p_asymp = spearmanr(x_arr, y_arr)
    rho_f = float(rho)
    if n > _EXACT_SPEARMAN_MAX_N:
        return rho_f, float(p_asymp), "asymptotic"
    obs = abs(rho_f)
    n_extreme = 0
    n_total = 0
    for y_perm in permutations(y_arr.tolist()):
        n_total += 1
        r, _ = spearmanr(x_arr, y_perm)
        if abs(float(r)) + 1e-12 >= obs:
            n_extreme += 1
    return rho_f, n_extreme / n_total, "exact_permutation"


def format_spearman_annotation(rho: float, pval: float) -> str:
    """Paper-facing annotation: avoid overstating significance for small n."""
    txt = f"Spearman $\\rho$ = {rho:.2f}"
    if pval < 0.001:
        txt += " ($p<0.001$)"
    elif pval < 0.01:
        # Prefer an explicit value near the exact permutation result (≈0.0022).
        txt += f" ($p={pval:.3f}$)"
    else:
        txt += f" ($p={pval:.3g}$)"
    return txt


def save_xy_table(
    rows: list[dict[str, Any]],
    out_dir: Path,
    sigma: float,
    spearman_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stability_transfer_xy.csv"
    json_path = out_dir / "stability_transfer_xy.json"

    fieldnames = [
        "mode", "speckle_key", "label",
        "drift_l1", "transfer_acc", "transfer_acc_std", "macro_f1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    payload: dict[str, Any] = {"speckle_std": sigma, "points": rows}
    if spearman_meta is not None:
        payload["spearman"] = spearman_meta
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def _apply_axes_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", **GRID_MAJOR)
    ax.grid(True, which="minor", **GRID_MINOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK)


def plot_scatter(
    rows: list[dict[str, Any]],
    out_dir: Path,
    sigma: float,
    spearman_rho: float,
    spearman_p: float,
    dpi: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    rcParams["font.size"] = float(rcParams["font.size"]) + FONT_BUMP

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    scale_colors = _blue_gradient_colors(6)

    x = np.array([r["drift_l1"] for r in rows], dtype=float)
    y = np.array([r["transfer_acc"] for r in rows], dtype=float)

    order = np.argsort(-x)
    ax.plot(
        x[order], y[order],
        color=GUIDE_LINE_COLOR, linestyle="--", linewidth=LINE_WIDTH_PIXEL,
        zorder=1, alpha=0.85,
    )

    scale_idx = 0
    labeled: set[str] = set()

    for r in rows:
        key = r["speckle_key"]
        xv, yv = r["drift_l1"], r["transfer_acc"]
        ye = r["transfer_acc_std"]
        lbl = legend_display_name(key) if key not in labeled else "_nolegend_"
        labeled.add(key)

        if key == "pixel":
            ax.scatter(
                xv, yv,
                s=_scatter_s(MARKER_SIZE),
                color=PIXEL_MARKER_FACE,
                edgecolors=PIXEL_MARKER_EDGE,
                linewidths=MARKER_EDGEWIDTH,
                zorder=4,
                label=lbl,
            )
        elif key == "multi":
            ax.scatter(
                xv, yv,
                s=_scatter_s(MARKER_SIZE_MULTI),
                color=MULTI_MARKER_FACE,
                edgecolors=MULTI_MARKER_EDGE,
                linewidths=MARKER_EDGEWIDTH_MULTI,
                zorder=5,
                label=lbl,
            )
        else:
            color = scale_colors[scale_idx]
            scale_idx += 1
            ax.scatter(
                xv, yv,
                s=_scatter_s(MARKER_SIZE),
                color=color, edgecolors=color,
                linewidths=MARKER_EDGEWIDTH,
                zorder=3,
                label=lbl,
            )

        if ye > 0:
            ax.errorbar(
                xv, yv, yerr=ye, fmt="none", ecolor="#888888",
                elinewidth=0.7, capsize=2.0, capthick=0.7, zorder=2,
            )

        if key == "pixel":
            ax.annotate(
                "Pixel",
                (xv, yv),
                xytext=(-16, 8),
                textcoords="offset points",
                fontsize=FS_ANNOT,
                color=PIXEL_LINE_COLOR,
                ha="right", va="bottom", clip_on=False, zorder=10,
                arrowprops=dict(arrowstyle="-", color=PIXEL_LINE_COLOR, lw=0.55, shrinkA=3, shrinkB=2),
            )
        elif key == "multi":
            ax.annotate(
                "Multi",
                (xv, yv),
                xytext=(0, 13),
                textcoords="offset points",
                fontsize=FS_ANNOT,
                color=MULTI_LINE_COLOR,
                fontweight="bold",
                ha="center", va="bottom", clip_on=False, zorder=10,
                arrowprops=dict(arrowstyle="-", color=MULTI_LINE_COLOR, lw=0.55, shrinkA=3, shrinkB=2),
            )

    ax.set_xscale("log")
    ax.set_xlabel(X_LABEL, fontsize=FS_XLABEL)
    ax.set_ylabel(Y_LABEL, fontsize=FS_YLABEL, labelpad=YLABEL_LABELPAD)
    ax.set_xlim(x.min() * 0.65, x.max() * 1.45)
    ax.set_ylim(max(26.5, y.min() - 2.5), min(43.5, y.max() + 2.5))

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=Y_LOG_MINOR_SUBS))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(NullFormatter())

    _apply_axes_style(ax)

    handles, labels = ax.get_legend_handles_labels()
    order_map = {legend_display_name(k): i for i, k in enumerate(LEGEND_ORDER)}
    paired = sorted(zip(handles, labels), key=lambda t: order_map.get(t[1], 99))
    if paired:
        handles, labels = zip(*paired)
        ax.legend(
            handles,
            labels,
            ncol=4,
            frameon=True,
            fancybox=False,
            edgecolor="#999999",
            facecolor="#FAFAFA",
            fontsize=FS_LEGEND,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            borderaxespad=0.35,
            columnspacing=0.9,
            handletextpad=0.35,
        )

    rho_txt = format_spearman_annotation(spearman_rho, spearman_p)
    ax.text(
        0.98, 0.04, rho_txt,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FS_ANNOT, color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA", edgecolor="#999999", alpha=0.95),
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.subplots_adjust(left=FIG_SUBPLOTS_ADJUST_LEFT)

    stem = str(out_dir / "stability_transfer_scatter.png")
    png_path, pdf_path = _png_pdf_paths(stem)
    save_kw = {"bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(png_path, dpi=dpi, **save_kw)
    fig.savefig(pdf_path, format="pdf", **save_kw)
    plt.close(fig)
    return Path(png_path), Path(pdf_path)


def main():
    p = argparse.ArgumentParser(description="Plot stability–transfer scatter.")
    p.add_argument("--speckle_json", default=str(DEFAULT_SPECKLE_JSON))
    p.add_argument("--transfer_json", default=str(DEFAULT_TRANSFER_JSON))
    p.add_argument("--speckle_std", type=float, default=0.15)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=float, default=float(DPI))
    args = p.parse_args()

    speckle_json = Path(args.speckle_json)
    transfer_json = Path(args.transfer_json)
    out_dir = Path(args.out_dir)

    drift = load_drift_at_sigma(speckle_json, args.speckle_std)
    transfer = load_transfer(transfer_json)
    rows = build_xy_table(drift, transfer)

    rho, pval, p_method = spearmanr_exact_twosided(
        [r["drift_l1"] for r in rows],
        [r["transfer_acc"] for r in rows],
    )
    _, p_asymp = spearmanr(
        [r["drift_l1"] for r in rows],
        [r["transfer_acc"] for r in rows],
    )
    spearman_meta = {
        "rho": rho,
        "p_exact" if p_method == "exact_permutation" else "p": pval,
        "p_method": p_method,
        "p_asymptotic_scipy": float(p_asymp),
        "n": len(rows),
        "note": (
            "For n<=9, report the exact two-sided permutation p-value; "
            "do not use scipy's asymptotic p (can falsely suggest p<0.001)."
        ),
    }

    csv_path, json_path = save_xy_table(
        rows, out_dir, args.speckle_std, spearman_meta=spearman_meta,
    )
    png_path, pdf_path = plot_scatter(
        rows, out_dir, args.speckle_std, float(rho), float(pval), args.dpi,
    )

    print(
        f"Spearman rho = {rho:.3f}  p_{p_method} = {pval:.4g}  "
        f"(scipy asymptotic p = {float(p_asymp):.4g})"
    )
    print("\nXY table:")
    print(f"{'Label':<8} {'drift_l1':>12} {'acc (%)':>10} {'±std':>8}")
    print("-" * 42)
    for r in rows:
        print(
            f"{r['label']:<8} {r['drift_l1']:>12.2e} "
            f"{r['transfer_acc']:>10.1f} {r['transfer_acc_std']:>8.2f}"
        )
    print(f"\nSaved: {csv_path}")
    print(f"       {json_path}")
    print(f"Saved figure: {png_path}")
    print(f"              {pdf_path}")


if __name__ == "__main__":
    main()
