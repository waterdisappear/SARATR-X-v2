# visualize — Paper-Figure Reproduction Scripts / 论文图表复现脚本

This directory collects the visualization scripts directly related to the paper figures. All scripts use **bilingual (Chinese + English) comments**, and hard-coded paths are converted to relative paths or environment variables where possible.

本目录集中存放与论文图表直接相关的可视化脚本。所有脚本均使用**中英文双语注释**，并尽量将硬编码路径改为相对路径或环境变量。

- Generated PNG / PDF / SVG figures are **not version-controlled**; run the scripts to regenerate them in this directory.
  生成的 PNG / PDF / SVG 等图片产物**不纳入版本控制**，运行脚本即可在本目录生成。
- The small experiment data (csv/json) needed by the scripts lives under `results/visualize/` at the repo root.
  脚本运行所需的少量实验数据（csv/json）放在仓库根目录的 `results/visualize/` 下。
- Scripts that need pre-trained weights (`hogs.*` / `My_SAR_feature`) resolve them via the env var `SARATRX_PRETRAIN_CKPT` or the default path `weights/base/jiaquan_simple/checkpoint-1200.pth` (see each script's `resolve_pretrain_ckpt()`).
  需要预训练权重的脚本统一通过环境变量 `SARATRX_PRETRAIN_CKPT` 或默认路径 `weights/base/jiaquan_simple/checkpoint-1200.pth` 解析（见各脚本 `resolve_pretrain_ckpt()`）。

## Scripts and Corresponding Paper Figures / 目录结构与论文图对应关系

| Script / 脚本 | Purpose / 用途 | Paper figure / 论文对应 |
| --- | --- | --- |
| `visualize_all_result.py` | 4×3 horizontal bar chart of the 12 SAR benchmarks (class./det./seg., sorted by gain) / 12 个 benchmark 的 4×3 横向条形图 | Fig. main results / 主结果对比图 |
| `visualize_cube_grids.py` | Isometric cube grid illustration (mask patches used in the framework figure) / 等距立方体网格示意图（框架图中的掩码块元素） | Fig. framework (mask-patch) / 框架图（掩码块元素） |
| `visualize_sar_targets.py` | Raw SAR / masked image / S1–S6 multi-scale features / fused target feature / 原始 SAR / 掩码图 / 多尺度特征 / 融合目标特征 | Method figure (multi-scale structural target) / 方法图 |
| `visualize_fusion_residual.py` | Per-pixel residual between the fused target and each single-scale feature / 融合监督目标与单尺度特征的逐像素残差 | Fig. fusion residual / 融合残差图 |
| `visualize_fusion_residual_metrics.py` | Fusion-residual concentration metrics inside salient target regions (R_res / R_area) / 显著目标区域内融合残差集中度指标 | Ablation value source / 消融实验数值来源 |
| `visualize_speckle_target_stability.py` | Stability of pixel / S1–S6 / multi targets under speckle perturbation (mean L1) / 斑点扰动下各目标的稳定性 | Ablation (speckle robustness) / 消融（斑点鲁棒性） |
| `plot_speckle_stability_figure.py` | Speckle-stability line plot (aggregates multiple runs into the paper figure) / 斑点稳定性折线图 | Fig. speckle stability / 斑点稳定性图 |
| `visualize_stability_transfer_scatter.py` | Scatter of target stability vs downstream 10-shot accuracy / 目标稳定性与下游 10-shot 精度的散点图 | Fig. stability–transfer / 稳定性–迁移散点图 |

## Usage Examples / 运行示例

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

## Data and Weight Dependencies / 数据与权重依赖

| Dependency / 依赖 | Location / 位置与获取方式 |
| --- | --- |
| Pre-trained weights (`hogs.*` / `My_SAR_feature`) | `weights/base/jiaquan_simple/checkpoint-1200.pth` or `SARATRX_PRETRAIN_CKPT` |
| Speckle-stability experiment data | `results/visualize/speckle_stability_out(_50ep)/` |
| Target-ablation 10-shot data | `results/visualize/target_ablation_soc_10shot_summary.csv/.json` |
| Stability–transfer scatter data | `results/visualize/stability_transfer_out/stability_transfer_xy.csv/.json` |
| SAR input images (fusion residual / target viz) | Pre-training dataset (e.g. FAIR_CSAR / 500K) |

## Runtime Environment / 运行环境

- Python 3.8+
- Scripts that need the model (`models.masked_autoencoder`) automatically add both the repo root and `pre-training/` to `sys.path`, so they can be run from anywhere.
  需要模型（`models.masked_autoencoder`）的脚本会自动把仓库根目录与 `pre-training/` 加入 `sys.path`，可在任意目录运行。
- Dependencies / 依赖：`torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`
