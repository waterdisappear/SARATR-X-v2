# results — 实验结果说明

本目录存放论文中的实验结果（检查点、日志、指标文件）。由于文件较大，**不随仓库提交**，请从 release 附件 / 网盘下载后按下列结构放置。

## 目录结构

```
results/
├── classification/               # 分类结果
│   ├── soc50/                    # SOC-50（线性 / k-NN / 少样本）
│   ├── sarvasa/                  # SAR-VSA
│   └── fusar/                    # FUSAR-Ship
├── detection/                    # 旋转目标检测结果（mmrotate work_dirs 导出）
│   ├── rsar/                     # RSAR（ReDet + iTPN-B/L）
│   ├── hrsid/                    # HRSID
│   ├── ssdd/                     # SSDD
│   └── sardet/                   # SARDet-100k
├── segmentation/                 # 语义分割结果（mmseg work_dirs 导出）
│   ├── air_polarsar2/            # AIR-PolarSAR-Seg-2.0（UperNet + iTPN-B/L）
│   ├── whu-opt-sar/              # WHU-OPT-SAR
│   └── ddhr-sk/                  # DDHR-SK
├── target_ablation/              # 目标设计消融（10-shot 精度等）
│   ├── pixel/                    # 像素级目标
│   ├── multi/                    # 多尺度融合目标（论文主方法）
│   └── single_s1 .. single_s6/   # 各单尺度目标
└── visualize/                    # 可视化脚本依赖的实验数据（csv/json，随仓库提交）
```

## 与论文表格的对应关系

| 论文表格 | 数据位置 |
| --- | --- |
| 12 基准总表（分类/检测/分割） | `classification/`、`detection/`、`segmentation/` |
| 目标设计消融（pixel/S1–S6/multi） | `target_ablation/`（`target_ablation_soc_10shot_summary.json` 见 `results/visualize/`） |
| 斑点稳定性 / 稳定性迁移 | `results/visualize/`（`speckle_stability_*`、`stability_transfer_xy.*`） |

## 使用方式

- 分类评测脚本默认将结果输出到各自 `work_dirs`；指标汇总见对应 README。
- 可视化脚本（`visualize/`）读取 `results/visualize/` 下的 csv/json 直接出图。
- 检测/分割的模型检查点按各框架 README 使用 `latest.pth` / `best_mIoU_*.pth`。

> 说明：`results/visualize/` 下的实验数据文件（csv/json，体积小）已随仓库提交，
> 便于论文图表复现；其余大文件由用户单独上传。
