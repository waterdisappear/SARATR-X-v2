# Few-shot Classification Evaluation / 少样本分类评测

This folder implements SAR few-shot classification evaluation based on the Dassl framework (protocol in the paper: linear probing with a frozen backbone):

本目录基于 Dassl 框架，实现 SAR 少样本分类评测（论文协议：冻结 backbone 的线性探测）：

- `MIM_linear_itpn`: linear probing with the iTPN backbone (the main protocol used for ATRNet-STAR / MSTAR in the paper) / iTPN 骨干线性探测（论文 ATRNet-STAR / MSTAR 的主评测）
- `MIM_linear`: linear probing with HiViT (reference backbone from v1) / HiViT 骨干线性探测（v1 SARATR-X 的骨干，作对比参考）
- `MIM_finetune`: end-to-end few-shot fine-tuning with HiViT (reference) / HiViT 少样本全量微调（作对比参考）

## Directory Layout / 目录结构

```
fewshot/
├── train.py                    # Dassl evaluation entry / 评测入口
├── trainers/                   # Trainers / 训练器
│   ├── MIM_linear_itpn.py
│   ├── MIM_linear.py
│   └── MIM_finetune.py
├── model/                      # Backbones / 骨干网络
│   ├── models_itpn.py          # iTPN
│   └── hivit.py                # HiViT
├── datasets/                   # SAR dataset definitions / SAR 数据集定义
├── configs/
│   ├── datasets/               # Per-dataset yaml / 数据集配置
│   └── trainers/               # Per-trainer yaml / 训练器配置
└── scripts/
    ├── MIM_linear_itpn/        # iTPN linear-probing scripts + target ablation / iTPN 线性探测脚本 + 目标消融脚本
    ├── MIM_linear/
    └── MIM_finetune/
```

## Environment / 环境

Needs Dassl ([KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)), plus `timm` and `scipy`. After installing Dassl, add it to `PYTHONPATH`.

需要 Dassl（[KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)）以及 `timm`、`scipy`。安装 Dassl 后将其加入 `PYTHONPATH`。

## Usage / 使用

```bash
cd classification/fewshot

# MSTAR 5-shot linear probing (iTPN-B backbone)
bash scripts/MIM_linear_itpn/main.sh MSTAR_SOC vit_b16 5

# SOC-50 10-shot linear probing
bash scripts/MIM_linear_itpn/main.sh SOC_50classes vit_b16 10
```

Pre-trained weights are specified via the `SARATRX_PRETRAIN_CKPT` / `HIVIT_PRETRAIN_CKPT` env vars, or loaded from the repo `weights/` by default (see `resolve_pretrain_ckpt` in `trainers/MIM_linear_itpn.py`).

预训练权重通过环境变量指定，或默认从仓库 `weights/` 目录加载（见 `trainers/MIM_linear_itpn.py` 的 `resolve_pretrain_ckpt`）。

## Target Ablation (reconstruction target) / 消融实验（重建目标）

`scripts/MIM_linear_itpn/run_target_ablation.sh` sequentially runs 10-shot linear probing on SOC-50 with the pre-trained weights of all 8 reconstruction targets (pixel, single_s1~s6, multi). Weights live in `weights/base/<mode>/checkpoint-49.pth`.

`scripts/MIM_linear_itpn/run_target_ablation.sh` 依次用 8 种重建目标（pixel、single_s1~s6、multi）的预训练权重在 SOC-50 上做 10-shot 线性探测，权重放在 `weights/base/<mode>/checkpoint-49.pth`。
