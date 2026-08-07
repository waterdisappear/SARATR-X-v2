# visualize — 论文图表复现脚本

本目录集中存放与论文图表直接相关的可视化脚本。所有脚本均使用**中英文双语注释**，并尽量将硬编码路径改为相对路径或环境变量。

- 生成的 PNG / PDF / SVG 等图片产物**不纳入版本控制**，运行脚本即可在本目录生成。
- 脚本运行所需的少量实验数据（csv/json）放在仓库根目录的 `results/visualize/` 下。
- 需要预训练权重（`hogs.*` / `My_SAR_feature`）的脚本，统一通过环境变量 `SARATRX_PRETRAIN_CKPT` 或默认路径 `weights/jiaquan_simple/checkpoint-1200.pth` 解析（见各脚本 `resolve_pretrain_ckpt()`）。

## 目录结构与论文图对应关系

| 脚本 | 用途 | 论文对应 |
| --- | --- | --- |
| `visualize_all_result.py` | 12 个 SAR benchmark 的 4×3 横向条形图（分类/检测/分割，按增量降序） | 主结果对比图 |
| `visualize_all_result_3x4.py` | 3×4 转置布局变体（数据与样式复用前者） | 主结果对比图（备选布局） |
| `visualize_sar_targets.py` | 原始 SAR / 掩码图 / S1–S6 多尺度特征 / 融合目标特征 | 方法图（多尺度结构目标） |
| `visualize_cube_grids.py` | 等距立方体网格示意图（3×3 / 5×5 / 7×7） | 方法图（掩码块概念示意） |
| `visualize_fusion_residual.py` | 融合监督目标与单尺度特征的逐像素残差 | 方法图 Fig.2(a) 类 |
| `visualize_fusion_residual_metrics.py` | 显著目标区域内融合残差集中度指标（R_res / R_area） | 消融实验数值来源 |
| `visualize_speckle_target_stability.py` | 斑点扰动下 pixel / S1–S6 / multi 目标的稳定性（均值 L1） | 消融（斑点鲁棒性） |
| `plot_speckle_stability_figure.py` | 斑点稳定性折线图（多张图汇总为论文图） | 消融图（斑点稳定性） |
| `visualize_stability_transfer_scatter.py` | 目标稳定性与下游 10-shot 精度的散点图 | 消融图（稳定性—迁移关系） |
| `visualize_sar_gain_mstar_detection_lollipop.py` | MSTAR 检测增量（lollipop 图） | 增益图（附录） |
| `visualize_sar_gain_single_wide_lollipop.py` | 单宽图增益（lollipop） | 增益图（附录） |
| `visualize_sar_gain_vertical_grouped.py` | 分组纵向增益柱状图 | 增益图（附录） |
| `visualize_4msar_detection.py` | 4MSAR 数据集的检测框定性展示 | 检测定性示例 |
| `visualize_segmatition.py` | AIR-PolarSAR-Seg-2.0 的语义分割叠加展示 | 分割定性示例 |

## 运行示例

```bash
# 1) 12 benchmark 对比图（数据内嵌，无需外部输入）
python visualize/visualize_all_result.py

# 2) 多尺度结构目标可视化（需要预训练权重 + 一张 SAR 图）
python visualize/visualize_sar_targets.py --image_path <path/to/sar.png>
# 预训练权重可通过环境变量指定：
export SARATRX_PRETRAIN_CKPT=/path/to/checkpoint-1200.pth

# 3) 斑点稳定性折线图（读取 results/visualize/ 下的实验数据）
python visualize/plot_speckle_stability_figure.py
python visualize/visualize_stability_transfer_scatter.py
```

## 数据与权重依赖

| 依赖 | 位置 / 获取方式 |
| --- | --- |
| 预训练权重（`hogs.*` / `My_SAR_feature`） | `weights/jiaquan_simple/checkpoint-1200.pth` 或 `SARATRX_PRETRAIN_CKPT` |
| 斑点稳定性实验数据 | `results/visualize/speckle_stability_out(_50ep)/` |
| 目标消融 10-shot 数据 | `results/visualize/target_ablation_soc_10shot_summary.csv/.json` |
| 稳定性—迁移散点数据 | `results/visualize/stability_transfer_out/stability_transfer_xy.csv/.json` |
| 融合残差 / 目标可视化输入 SAR 图 | 预训练数据集（如 FAIR_CSAR / 500K） |
| 检测定性可视化 | 4MSAR 数据集（JPEGImages + labels VOC XML） |
| 分割定性可视化 | AIR-PolarSAR-Seg-2.0 数据集（train 的 hh + gt_color） |

## 运行环境

- Python 3.8+
- 需要 `pre-training/` 模块（脚本会 import `models.masked_autoencoder`），因此请将仓库根目录加入 `PYTHONPATH` 或从仓库根目录运行。
- 依赖：`torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`

> 注：`visualize_4msar_detection.py` 与 `visualize_segmatition.py` 依赖检测/分割数据集，属数据预处理可视化，路径通过命令行参数传入。
