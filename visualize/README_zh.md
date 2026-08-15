<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

# visualize — 论文图表复现脚本

本目录集中存放与论文图表直接相关的可视化脚本。所有脚本均使用**中英文双语注释**，并尽量将硬编码路径改为相对路径或环境变量。

- 生成的 PNG / PDF / SVG 等图片产物**不纳入版本控制**，运行脚本即可在本目录生成。
- 脚本运行所需的少量实验数据（csv/json）放在仓库根目录的 `results/visualize/` 下。
- 需要预训练权重的脚本统一通过环境变量 `SARATRX_PRETRAIN_CKPT` 或默认路径 `weights/base/jiaquan_simple/checkpoint-1200.pth` 解析（见各脚本 `resolve_pretrain_ckpt()`）。

## 目录结构与论文图对应关系

| 脚本 | 用途 | 论文对应 |
| --- | --- | --- |
| `visualize_all_result.py` | 12 个 benchmark 的 4×3 横向条形图 | 主结果对比图 |
| `visualize_cube_grids.py` | 等距立方体网格示意图（框架图中的掩码块元素） | 框架图（掩码块元素） |
| `visualize_sar_targets.py` | 原始 SAR、掩码图、多尺度特征、融合目标特征 | 方法图 |
| `visualize_fusion_residual.py` | 融合监督目标与单尺度特征的逐像素残差 | 融合残差图 |
| `visualize_fusion_residual_metrics.py` | 显著目标区域内融合残差集中度指标 | 消融实验数值来源 |
| `visualize_speckle_target_stability.py` | 斑点扰动下各目标的稳定性 | 消融（斑点鲁棒性） |
| `plot_speckle_stability_figure.py` | 斑点稳定性折线图 | 斑点稳定性图 |
| `visualize_stability_transfer_scatter.py` | 目标稳定性与下游 10-shot 精度的散点图 | 稳定性–迁移散点图 |

## 运行示例

```bash
# 1) 12 个 benchmark 对比图（数据已内置，无需外部输入）
python visualize/visualize_all_result.py

# 2) 多尺度结构目标可视化（需要预训练权重 + 一张 SAR 图像）
python visualize/visualize_sar_targets.py --image_path <path/to/sar.png>
# 可通过环境变量指定权重：
export SARATRX_PRETRAIN_CKPT=/path/to/checkpoint-1200.pth

# 3) 斑点稳定性折线图（读取 results/visualize/ 下的实验数据）
python visualize/plot_speckle_stability_figure.py
python visualize/visualize_stability_transfer_scatter.py
```

## 数据与权重依赖

| 依赖 | 位置与获取方式 |
| --- | --- |
| 预训练权重（`hogs.*` / `My_SAR_feature`） | `weights/base/jiaquan_simple/checkpoint-1200.pth` 或 `SARATRX_PRETRAIN_CKPT` |
| 斑点稳定性实验数据 | `results/visualize/speckle_stability_out(_50ep)/` |
| 目标消融 10-shot 数据 | `results/visualize/target_ablation_soc_10shot_summary.csv/.json` |
| 稳定性–迁移散点数据 | `results/visualize/stability_transfer_out/stability_transfer_xy.csv/.json` |
| SAR 输入图像（融合残差、目标可视化） | 预训练数据集（如 FAIR_CSAR / 500K） |

## 运行环境

- Python 3.8+
- 需要模型（`models.masked_autoencoder`）的脚本会自动把仓库根目录与 `pre-training/` 加入 `sys.path`，可在任意目录运行。
- 依赖：`torch`、`torchvision`、`numpy`、`opencv-python`、`matplotlib`、`Pillow`
