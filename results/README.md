# results — Experiment Results / 实验结果说明

This directory stores the experiment data used for paper-figure reproduction. Only the small csv/json data needed by the visualization scripts is committed here (under `results/visualize/`). Larger downstream outputs (checkpoints, logs, metric files from classification / detection / segmentation) are **not committed** — see the paper tables and each module's README for the numbers.

本目录存放论文图表复现所需的小体积实验数据。只有可视化脚本依赖的 csv/json 数据随仓库提交（位于 `results/visualize/`）。分类 / 检测 / 分割的大体积下游输出（检查点、日志、指标文件）**不随仓库提交**，具体数值以论文表格与各模块 README 为准。

## Directory Structure / 目录结构

```
results/
└── visualize/                 # Experiment data for visualization scripts (csv/json, committed) / 可视化脚本依赖的实验数据（随仓库提交）
```

## Contents / 内容说明

| File(s) / 文件 | Description / 说明 |
| --- | --- |
| `speckle_stability_out/`、`speckle_stability_out_50ep/` | Speckle-stability experiment data (1200-epoch / 50-epoch settings) / 斑点稳定性实验数据（1200 epoch / 50 epoch 设置） |
| `stability_transfer_out/` | Stability–transfer scatter data (x: drift, y: 10-shot acc.) / 稳定性—迁移散点数据 |
| `target_ablation_soc_10shot_summary.csv/.json` | 10-shot target-ablation summary on SOC-50 / SOC-50 上 10-shot 目标消融汇总 |

## Usage / 使用方式

- Visualization scripts (`visualize/`) read the csv/json under `results/visualize/` directly.
  可视化脚本（`visualize/`）读取 `results/visualize/` 下的 csv/json 直接出图。
- For downstream task results, see `classification/`、`detection/`、`segmentation/` READMEs and the paper's experimental section.
  下游任务结果见 `classification/`、`detection/`、`segmentation/` 各 README 及论文实验章节。
