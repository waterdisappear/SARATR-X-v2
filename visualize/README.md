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
| `visualize_all_result.py` | 4×3 horizontal bar chart of the 12 SAR benchmarks (class./det./seg., sorted by gain) / 12 个 benchmark 的 4×3 横向条形图 | Main results figure / 主结果对比图 |
| `visualize_all_result_3x4.py` | 3×4 transposed-layout variant (reuses data & style) / 3×4 转置布局变体 | Main results figure (alt. layout) / 主结果对比图（备选布局） |
| `visualize_sar_targets.py` | Raw SAR / masked image / S1–S6 multi-scale features / fused target feature / 原始 SAR / 掩码图 / 多尺度特征 / 融合目标特征 | Method figure (multi-scale structural target) / 方法图 |
| `visualize_cube_grids.py` | Isometric cube grid illustration (3×3 / 5×5 / 7×7) / 等距立方体网格示意图 | Method figure (mask-patch concept) / 方法图（掩码块概念示意） |
| `visualize_fusion_residual.py` | Per-pixel residual between the fused target and each single-scale feature / 融合监督目标与单尺度特征的逐像素残差 | Method figure (Fig.2(a)-like) / 方法图 |
| `visualize_fusion_residual_metrics.py` | Fusion-residual concentration metrics inside salient target regions (R_res / R_area) / 显著目标区域内融合残差集中度指标 | Ablation value source / 消融实验数值来源 |
| `visualize_speckle_target_stability.py` | Stability of pixel / S1–S6 / multi targets under speckle perturbation (mean L1) / 斑点扰动下各目标的稳定性 | Ablation (speckle robustness) / 消融（斑点鲁棒性） |
| `plot_speckle_stability_figure.py` | Speckle-stability line plot (aggregates multiple runs into the paper figure) / 斑点稳定性折线图 | Ablation figure (speckle stability) / 消融图 |
| `visualize_stability_transfer_scatter.py` | Scatter of target stability vs downstream 10-shot accuracy / 目标稳定性与下游 10-shot 精度的散点图 | Ablation figure (stability–transfer) / 消融图 |
| `visualize_sar_gain_mstar_detection_lollipop.py` | MSTAR detection gain (lollipop) / MSTAR 检测增量 | Gain figure (appendix) / 增益图（附录） |
| `visualize_sar_gain_single_wide_lollipop.py` | Single-wide gain (lollipop) / 单宽图增益 | Gain figure (appendix) / 增益图（附录） |
| `visualize_sar_gain_vertical_grouped.py` | Vertical grouped gain bar chart / 分组纵向增益柱状图 | Gain figure (appendix) / 增益图（附录） |
| `visualize_4msar_detection.py` | Qualitative detection boxes on 4MSAR / 4MSAR 数据集的检测框定性展示 | Detection qualitative example / 检测定性示例 |
| `visualize_segmatition.py` | Semantic segmentation overlay on AIR-PolarSAR-Seg-2.0 / 语义分割叠加展示 | Segmentation qualitative example / 分割定性示例 |

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
| Detection qualitative visualization | 4MSAR dataset (JPEGImages + VOC XML labels) |
| Segmentation qualitative visualization | AIR-PolarSAR-Seg-2.0 dataset (train hh + gt_color) |

## Runtime Environment / 运行环境

- Python 3.8+
- Scripts that need the model (`models.masked_autoencoder`) automatically add both the repo root and `pre-training/` to `sys.path`, so they can be run from anywhere.
  需要模型（`models.masked_autoencoder`）的脚本会自动把仓库根目录与 `pre-training/` 加入 `sys.path`，可在任意目录运行。
- Dependencies / 依赖：`torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`

> Note / 注：`visualize_4msar_detection.py` and `visualize_segmatition.py` depend on the detection/segmentation datasets
> (data-preprocessing visualizations); pass their paths via command-line arguments.
> `visualize_4msar_detection.py` 与 `visualize_segmatition.py` 依赖检测/分割数据集，属数据预处理可视化，路径通过命令行参数传入。
