# results — Experiment Results / 实验结果说明

This directory stores the experimental results from the paper (checkpoints, logs, metric files). Since these files are large, they are **not committed**; download them from the release / 网盘 and place them following the structure below.

本目录存放论文中的实验结果（检查点、日志、指标文件）。由于文件较大，**不随仓库提交**，请从 release 附件 / 网盘下载后按下列结构放置。

## Directory Structure / 目录结构

```
results/
├── classification/               # Classification results / 分类结果
│   ├── soc50/                    # SOC-50 (linear / k-NN / few-shot) / 线性 / k-NN / 少样本
│   ├── sarvasa/                  # SAR-VSA
│   └── fusar/                    # FUSAR-Ship
├── detection/                    # Rotated detection results (mmrotate work_dirs export) / 旋转目标检测结果
│   ├── rsar/                     # RSAR (ReDet + iTPN-B/L)
│   ├── hrsid/                    # HRSID
│   ├── ssdd/                     # SSDD
│   └── sardet/                   # SARDet-100k
├── segmentation/                 # Semantic segmentation results (mmseg work_dirs export) / 语义分割结果
│   ├── air_polarsar2/            # AIR-PolarSAR-Seg-2.0 (UperNet + iTPN-B/L)
│   ├── whu-opt-sar/              # WHU-OPT-SAR
│   └── ddhr-sk/                  # DDHR-SK
├── target_ablation/              # Target-design ablation (10-shot accuracy etc.) / 目标设计消融
│   ├── pixel/                    # Pixel target / 像素级目标
│   ├── multi/                    # Multi-scale fused target (main method) / 多尺度融合目标（论文主方法）
│   └── single_s1 .. single_s6/   # Individual single-scale targets / 各单尺度目标
└── visualize/                    # Experiment data for visualization scripts (csv/json, committed) / 可视化数据
```

## Mapping to Paper Tables / 与论文表格的对应关系

| Paper table / 论文表格 | Location / 数据位置 |
| --- | --- |
| 12-benchmark overview (class./det./seg.) / 12 基准总表 | `classification/`、`detection/`、`segmentation/` |
| Target-design ablation (pixel/S1–S6/multi) / 目标设计消融 | `target_ablation/`（`target_ablation_soc_10shot_summary.json` in `results/visualize/`） |
| Speckle stability / stability-transfer / 斑点稳定性 / 稳定性迁移 | `results/visualize/`（`speckle_stability_*`、`stability_transfer_xy.*`） |

## Usage / 使用方式

- Classification eval scripts output to their `work_dirs`; see each README for metric summaries.
  分类评测脚本默认将结果输出到各自 `work_dirs`；指标汇总见对应 README。
- Visualization scripts (`visualize/`) read the csv/json under `results/visualize/` directly.
  可视化脚本（`visualize/`）读取 `results/visualize/` 下的 csv/json 直接出图。
- Detection/segmentation checkpoints follow the framework READMEs (`latest.pth` / `best_mIoU_*.pth`).
  检测/分割的模型检查点按各框架 README 使用 `latest.pth` / `best_mIoU_*.pth`。

> Note / 说明：The experiment data files (csv/json) under `results/visualize/` are small and **committed**
> for paper-figure reproduction; the remaining large files are uploaded separately.
> `results/visualize/` 下的实验数据文件（csv/json，体积小）已随仓库提交，便于论文图表复现；其余大文件由用户单独上传。
