<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

# results — 实验结果说明

本目录存放 SARATR-X-v2 的实验结果数据：包括**检测与分割**下游实验的原始训练日志与配置（从原始实验目录 `D:\2024_SARatrX_2\result` 复制），以及可视化脚本所需的小体积 csv/json 数据。

> 提示：模型检查点（`.pth`，单个 1–4 GB）**不随仓库提交**，统一放在 `weights/` 目录单独上传。这里存放的日志包含论文表格所用的最终指标。

## 目录结构

```
results/
├── detection/                   # 检测实验日志与配置
│   ├── rsar/                    # RSAR 旋转框检测
│   ├── sardet/                  # SARDet-100K 水平框检测
│   ├── ssdd/                    # SSDD 水平框检测
│   └── hrsid/                   # HRSID 水平框检测
├── segmentation/                # 分割实验日志与配置
│   ├── air_polarsar2/           # 幅度影像分割
│   ├── whu-opt-sar/             # 光学-雷达影像分割
│   └── ddhr-sk/                 # 高分辨率 SAR 分割
└── visualize/                   # 可视化脚本依赖的实验数据
```

每个下游目录包含 `base` 与 `large`（前缀 `large_`）两份日志及对应的实验配置。这里的 `*.py` 配置覆盖了 `detection/`、`segmentation/` 中**未收录配置**的基准（SARDet-100K / SSDD / HRSID / WHU-OPT-SAR / DDHR-SK），为 MMDetection / MMSegmentation 风格，可直接参考复现。

## 主要结果

### 检测

| 数据集 | 协议 | 方法 | 指标 | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- | --- |
| RSAR | OBB | ReDet + iTPN | mAP@50 (test) | 74.2 | 77.3 |
| SARDet-100K | HBB | GFL + iTPN | bbox mAP | 63.4 | 68.7 |
| SSDD | HBB | GFL + iTPN | bbox mAP | 71.0 | 71.8 |
| HRSID | HBB | GFL + iTPN | bbox mAP | 71.5 | 73.5 |

### 分割

| 数据集 | 方法 | 指标 | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- |
| AIR-PolSAR-Seg-2.0 | UperNet + iTPN | mIoU | 90.8 | 95.3 |
| WHU-OPT-SAR | UperNet + iTPN | mIoU | 46.1 | 47.1 |
| DDHR-SK | UperNet + iTPN | mIoU | 83.4 | 83.6 |

> **与论文的对应关系。** 以上数字与论文报告值完全一致（附录 `tab_RSAR`、`tab_SARDet-100K`、`tab_SSDD`、`tab_HRSID`、`tab_airpolsar_amplitude`、`tab_whu_opt_sar`、`tab_ddhr_sk`）。SSDD / HRSID 及全部分割日志与论文完全对应；RSAR 行采用 **测试集** 指标（日志字段 `test_mAP50`）。

## visualize/ 内容说明

| 文件 | 说明 |
| --- | --- |
| `speckle_stability_out/`、`speckle_stability_out_50ep/` | 斑点稳定性实验数据（1200 epoch / 50 epoch 设置） |
| `stability_transfer_out/` | 稳定性—迁移散点数据 |
| `target_ablation_soc_10shot_summary.csv/.json` | SOC-50 上 10-shot 目标消融汇总 |

## 使用方式

- 可视化脚本（`visualize/`）读取 `results/visualize/` 下的 csv/json 直接出图。
- 下游检测/分割的自定义代码见 `detection/` 与 `segmentation/`（iTPN 相关部分）；这里的配置补充了论文中其余基准。
