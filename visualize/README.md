<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# visualize — Paper-Figure Reproduction Scripts

This directory collects the visualization scripts directly related to the paper figures. All scripts use **bilingual (Chinese + English) comments**, and hard-coded paths are converted to relative paths or environment variables where possible.

- Generated PNG / PDF / SVG figures are **not version-controlled**; run the scripts to regenerate them in this directory.
- The small experiment data (csv/json) needed by the scripts lives under `results/visualize/` at the repo root.
- Scripts that need pre-trained weights (`hogs.*` / `My_SAR_feature`) resolve them via the env var `SARATRX_PRETRAIN_CKPT` or the default path `weights/base/jiaquan_simple/checkpoint-1200.pth` (see each script's `resolve_pretrain_ckpt()`).

## Scripts and Corresponding Paper Figures

| Script | Purpose | Paper figure |
| --- | --- | --- |
| `visualize_all_result.py` | 4×3 horizontal bar chart of the 12 SAR benchmarks (class./det./seg., sorted by gain) | Fig. main results |
| `visualize_cube_grids.py` | Isometric cube grid illustration (mask patches used in the framework figure) | Fig. framework (mask-patch) |
| `visualize_sar_targets.py` | Raw SAR / masked image / S1–S6 multi-scale features / fused target feature | Method figure (multi-scale structural target) |
| `visualize_fusion_residual.py` | Per-pixel residual between the fused target and each single-scale feature | Fig. fusion residual |
| `visualize_fusion_residual_metrics.py` | Fusion-residual concentration metrics inside salient target regions (R_res / R_area) | Ablation value source |
| `visualize_speckle_target_stability.py` | Stability of pixel / S1–S6 / multi targets under speckle perturbation (mean L1) | Ablation (speckle robustness) |
| `plot_speckle_stability_figure.py` | Speckle-stability line plot (aggregates multiple runs into the paper figure) | Fig. speckle stability |
| `visualize_stability_transfer_scatter.py` | Scatter of target stability vs downstream 10-shot accuracy | Fig. stability–transfer |

## Usage Examples

```bash
# 1) 12-benchmark comparison figure (data embedded; no external input needed)
python visualize/visualize_all_result.py

# 2) Multi-scale structural target visualization (needs pre-trained weights + one SAR image)
python visualize/visualize_sar_targets.py --image_path <path/to/sar.png>
# Weights can be specified via env var:
export SARATRX_PRETRAIN_CKPT=/path/to/checkpoint-1200.pth

# 3) Speckle-stability line plot (reads experiment data under results/visualize/)
python visualize/plot_speckle_stability_figure.py
python visualize/visualize_stability_transfer_scatter.py
```

## Data and Weight Dependencies

| Dependency | Location |
| --- | --- |
| Pre-trained weights (`hogs.*` / `My_SAR_feature`) | `weights/base/jiaquan_simple/checkpoint-1200.pth` or `SARATRX_PRETRAIN_CKPT` |
| Speckle-stability experiment data | `results/visualize/speckle_stability_out(_50ep)/` |
| Target-ablation 10-shot data | `results/visualize/target_ablation_soc_10shot_summary.csv/.json` |
| Stability–transfer scatter data | `results/visualize/stability_transfer_out/stability_transfer_xy.csv/.json` |
| SAR input images (fusion residual / target viz) | Pre-training dataset (e.g. FAIR_CSAR / 500K) |

## Runtime Environment

- Python 3.8+
- Scripts that need the model (`models.masked_autoencoder`) automatically add both the repo root and `pre-training/` to `sys.path`, so they can be run from anywhere.
- Dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`
