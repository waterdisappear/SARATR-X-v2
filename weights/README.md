# weights — 预训练权重说明

本目录存放 SARATR-X-v2 的预训练权重。由于模型文件较大（每个约 1 GB），**不随仓库提交**，请从 release 附件 / 网盘下载后按下列结构放置。

## 目录结构

```
weights/
├── base/                          # iTPN-Base 系列（embed_dim=512, depth=24）
│   ├── jiaquan_simple/            # 【论文主模型】加权简单融合目标（多尺度结构目标）
│   ├── jiaquan_complex/           # 加权复杂融合目标（消融）
│   ├── jiaquan_sum/               # 直接求和融合目标（消融）
│   ├── scale_6/                   # 6 尺度目标设置（消融）
│   ├── itpn/                      # 基线 iTPN（官方目标）
│   ├── hivit/                     # 基线 HiViT（对比骨干）
│   ├── hivit_fused/               # HiViT + 融合目标（对比骨干）
│   ├── old/  old_2/               # 历史实验版本
└── large/                         # iTPN-Large 系列（embed_dim=768）
    ├── jiaquan_simple/            # 【论文主模型】iTPN-Large 加权简单融合
    ├── jiaquan_complex/           # 消融
    └── jiaquan_new/  old/  old_2/ # 历史实验版本
```

每个目录内含 `checkpoint-{100,...,1600}.pth` 等按 epoch 保存的检查点，**论文实验默认使用 `checkpoint-1200.pth`**。

## 与下游任务的对应关系

| 权重 | 分类 | 检测 | 分割 |
| --- | --- | --- | --- |
| `base/jiaquan_simple/checkpoint-1200.pth` | 线性评测 / 少样本（iTPN-B） | ReDet/RoI Trans + iTPN-B | UperNet + iTPN-B |
| `large/jiaquan_simple/checkpoint-1200.pth` | 线性评测（iTPN-L） | ReDet + iTPN-L | UperNet + iTPN-L |
| `base/hivit/checkpoint-1200.pth` | 少样本（HiViT） | — | UperNet + HiViT |

## 使用方式

各模块通过环境变量或默认路径引用权重：

- **线性评测**（`classification/linear_eval`）：`SARATRX_PRETRAIN_CKPT`（默认 `weights/jiaquan_simple/checkpoint-1200.pth`）
- **少样本**（`classification/fewshot`）：`SARATRX_PRETRAIN_CKPT`（iTPN）/ `HIVIT_PRETRAIN_CKPT`（HiViT）
- **检测**（`detection`）：`ITPN_CKPT`（iTPN-B）/ `ITPN_LARGE_CKPT`（iTPN-L），默认 `ckpts/iTPN/...`
- **分割**（`segmentation`）：配置内 `model.pretrained` 或 `--cfg-options model.pretrained=...`
- **可视化**（`visualize`）：`SARATRX_PRETRAIN_CKPT`

> 提示：仓库其余代码默认从仓库根目录的 `weights/` 解析权重；检测/分割因为框架安装在
> mmrotate/mmseg 环境中，按各自 README 放置到框架的 `ckpts/` 目录。
