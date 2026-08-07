# results — Experiment Results / 实验结果说明

This directory stores the experiment data of SARATR-X-v2. It contains the raw training logs / configs of the **detection and segmentation** downstream experiments (copied from the original run folder `D:\2024_SARatrX_2\result`), plus the small csv/json data needed by the visualization scripts.

本目录存放 SARATR-X-v2 的实验结果数据：包括**检测与分割**下游实验的原始训练日志与配置（从原始实验目录 `D:\2024_SARatrX_2\result` 复制），以及可视化脚本所需的小体积 csv/json 数据。

> Note / 提示：Model checkpoints (`.pth`, 1–4 GB each) are **not committed** — they live under `weights/` and are uploaded separately. The logs here contain the final metrics used in the paper tables.
> 模型检查点（`.pth`，单个 1–4 GB）**不随仓库提交**，统一放在 `weights/` 目录单独上传。这里存放的日志包含论文表格所用的最终指标。

## Directory Structure / 目录结构

```
results/
├── detection/                   # Detection logs & configs / 检测实验日志与配置
│   ├── rsar/                    # RSAR (OBB, ReDet / RoI Trans / Oriented R-CNN + iTPN) / RSAR 旋转框检测
│   ├── sardet/                  # SARDet-100K (HBB, GFL + iTPN) / SARDet-100K 水平框检测
│   ├── ssdd/                    # SSDD (HBB, GFL + iTPN) / SSDD 水平框检测
│   └── hrsid/                   # HRSID (HBB, GFL + iTPN) / HRSID 水平框检测
├── segmentation/                # Segmentation logs & configs / 分割实验日志与配置
│   ├── air_polarsar2/           # AIR-PolSAR-Seg-2.0 (UperNet + iTPN) / 幅度影像分割
│   ├── whu-opt-sar/             # WHU-OPT-SAR (UperNet + iTPN) / 光学-雷达影像分割
│   └── ddhr-sk/                 # DDHR-SK (UperNet + iTPN) / 高分辨率 SAR 分割
└── visualize/                   # Experiment data for visualization scripts (csv/json) / 可视化脚本依赖的实验数据
```

Each downstream folder contains both `base` and `large` (prefix `large_`) log files plus the exact config that produced them. The `*.py` configs here cover the benchmarks that have **no config in `detection/` or `segmentation/`** (SARDet-100K / SSDD / HRSID / WHU-OPT-SAR / DDHR-SK); they follow the MMDetection / MMSegmentation style and can be adapted for reproduction.

每个下游目录包含 `base` 与 `large`（前缀 `large_`）两份日志及对应的实验配置。这里的 `*.py` 配置覆盖了 `detection/`、`segmentation/` 中**未收录配置**的基准（SARDet-100K / SSDD / HRSID / WHU-OPT-SAR / DDHR-SK），为 MMDetection / MMSegmentation 风格，可直接参考复现。

## Main Results / 主要结果

### Detection / 检测

| Dataset / 数据集 | Protocol / 协议 | Method / 方法 | Metric / 指标 | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- | --- |
| RSAR | OBB | ReDet + iTPN | mAP50 | 78.2 | 80.4 |
| SARDet-100K | HBB | GFL + iTPN | bbox mAP | 65.4 | 68.7 |
| SSDD | HBB | GFL + iTPN | bbox mAP | 71.0 | 71.8 |
| HRSID | HBB | GFL + iTPN | bbox mAP | 71.5 | 73.5 |

### Segmentation / 分割

| Dataset / 数据集 | Method / 方法 | Metric / 指标 | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- |
| AIR-PolSAR-Seg-2.0 | UperNet + iTPN | mIoU | 90.77 | 95.32 |
| WHU-OPT-SAR | UperNet + iTPN | mIoU | 45.74 | 47.05 |
| DDHR-SK | UperNet + iTPN | mIoU | 83.39 | 83.51 |

> The exact per-class numbers and the last validation iteration are in the log files (e.g. `grep "mIoU"` / the final `"mode": "val"` line of each `.log.json`).
> 各类别详细精度与最终验证迭代见各日志文件（如 `grep "mIoU"` 或每个 `.log.json` 的末尾 `"mode": "val"` 行）。

## Contents under visualize/ / visualize/ 内容说明

| File(s) / 文件 | Description / 说明 |
| --- | --- |
| `speckle_stability_out/`、`speckle_stability_out_50ep/` | Speckle-stability experiment data (1200-epoch / 50-epoch settings) / 斑点稳定性实验数据（1200 epoch / 50 epoch 设置） |
| `stability_transfer_out/` | Stability–transfer scatter data (x: drift, y: 10-shot acc.) / 稳定性—迁移散点数据 |
| `target_ablation_soc_10shot_summary.csv/.json` | 10-shot target-ablation summary on SOC-50 / SOC-50 上 10-shot 目标消融汇总 |

## Usage / 使用方式

- Visualization scripts (`visualize/`) read the csv/json under `results/visualize/` directly.
  可视化脚本（`visualize/`）读取 `results/visualize/` 下的 csv/json 直接出图。
- Downstream detection/segmentation configs are also available under `detection/` and `segmentation/` (iTPN-related parts); the extra configs here cover the remaining paper benchmarks.
  下游检测/分割的自定义代码见 `detection/` 与 `segmentation/`（iTPN 相关部分）；这里的配置补充了论文中其余基准。
