# weights — Pre-trained Weights / 预训练权重说明

This directory stores the pre-trained weights of SARATR-X-v2. Since each model file is large (~1 GB), they are **not committed** to the repository; download them from the release / 网盘 and place them following the structure below.

本目录存放 SARATR-X-v2 的预训练权重。由于模型文件较大（每个约 1 GB），**不随仓库提交**，请从 release 附件 / 网盘下载后按下列结构放置。

## Directory Structure / 目录结构

```
weights/
├── base/                          # iTPN-Base series (embed_dim=512, depth=24) / iTPN-Base 系列
│   ├── jiaquan_simple/            # [Main model] weighted simple fusion target / 【论文主模型】加权简单融合目标
│   ├── jiaquan_complex/           # weighted complex fusion target (ablation) / 加权复杂融合目标（消融）
│   ├── jiaquan_sum/               # direct-sum fusion target (ablation) / 直接求和融合目标（消融）
│   ├── scale_6/                   # 6-scale target setting (ablation) / 6 尺度目标设置（消融）
│   ├── itpn/                      # baseline iTPN (official target) / 基线 iTPN（官方目标）
│   ├── hivit/                     # baseline HiViT (comparison backbone) / 基线 HiViT（对比骨干）
│   └── hivit_fused/               # HiViT + fused target (comparison backbone) / HiViT + 融合目标（对比骨干）
└── large/                         # iTPN-Large series (embed_dim=768) / iTPN-Large 系列
    ├── jiaquan_simple/            # [Main model] iTPN-Large weighted simple fusion / 【论文主模型】iTPN-Large 加权简单融合
    └── jiaquan_complex/           # weighted complex fusion target (ablation) / 加权复杂融合目标（消融）
```

Each directory contains checkpoints saved per epoch, e.g. `checkpoint-{100,...,1600}.pth`. **The paper experiments use `checkpoint-1200.pth` by default.**

每个目录内含 `checkpoint-{100,...,1600}.pth` 等按 epoch 保存的检查点，**论文实验默认使用 `checkpoint-1200.pth`**。

## Mapping to Downstream Tasks / 与下游任务的对应关系

| Weight / 权重 | Classification / 分类 | Detection / 检测 | Segmentation / 分割 |
| --- | --- | --- | --- |
| `base/jiaquan_simple/checkpoint-1200.pth` | Linear eval / few-shot (iTPN-B) / 线性评测 / 少样本（iTPN-B） | ReDet/RoI Trans + iTPN-B | UperNet + iTPN-B |
| `large/jiaquan_simple/checkpoint-1200.pth` | Linear eval (iTPN-L) / 线性评测（iTPN-L） | ReDet + iTPN-L | UperNet + iTPN-L |
| `base/hivit/checkpoint-1200.pth` | Few-shot (HiViT) / 少样本（HiViT） | — | UperNet + HiViT |

## Usage / 使用方式

Each module resolves weights via environment variables or default paths. 各模块通过环境变量或默认路径引用权重：

- **Linear eval / 线性评测**（`classification/linear_eval`）：`SARATRX_PRETRAIN_CKPT`（default / 默认 `weights/jiaquan_simple/checkpoint-1200.pth`）
- **Few-shot / 少样本**（`classification/fewshot`）：`SARATRX_PRETRAIN_CKPT`（iTPN）/ `HIVIT_PRETRAIN_CKPT`（HiViT）
- **Detection / 检测**（`detection`）：`ITPN_CKPT`（iTPN-B）/ `ITPN_LARGE_CKPT`（iTPN-L），default / 默认 `ckpts/iTPN/...`
- **Segmentation / 分割**（`segmentation`）：set `model.pretrained` in config or pass `--cfg-options model.pretrained=...`
- **Visualization / 可视化**（`visualize`）：`SARATRX_PRETRAIN_CKPT`

> Note / 提示：The rest of the repo resolves weights from `weights/` at the repo root by default; detection/segmentation
> run inside the mmrotate/mmseg environments, so place weights under the framework `ckpts/` per their READMEs.
> 仓库其余代码默认从仓库根目录的 `weights/` 解析权重；检测/分割因为框架安装在
> mmrotate/mmseg 环境中，按各自 README 放置到框架的 `ckpts/` 目录。
