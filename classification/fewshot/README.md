<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# Few-shot Classification Evaluation

This folder implements SAR few-shot classification evaluation based on the Dassl framework (protocol in the paper: linear probing with a frozen backbone):

- `MIM_linear_itpn`: linear probing with the iTPN backbone (the main protocol used for ATRNet-STAR / MSTAR in the paper)
- `MIM_linear`: linear probing with HiViT (reference backbone from v1)
- `MIM_finetune`: end-to-end few-shot fine-tuning with HiViT (reference)

## Directory Layout

```
fewshot/
├── train.py                    # Dassl evaluation entry
├── trainers/                   # Trainers
│   ├── MIM_linear_itpn.py
│   ├── MIM_linear.py
│   └── MIM_finetune.py
├── model/                      # Backbones
│   ├── models_itpn.py          # iTPN
│   └── hivit.py                # HiViT
├── datasets/                   # SAR dataset definitions
├── configs/
│   ├── datasets/               # Per-dataset yaml
│   └── trainers/               # Per-trainer yaml
└── scripts/
    ├── MIM_linear_itpn/        # iTPN linear-probing scripts + target ablation
    ├── MIM_linear/
    └── MIM_finetune/
```

## Environment

Needs Dassl ([KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)), plus `timm` and `scipy`. After installing Dassl, add it to `PYTHONPATH`.

## Usage

```bash
cd classification/fewshot

# MSTAR 5-shot linear probing (iTPN-B backbone)
bash scripts/MIM_linear_itpn/main.sh MSTAR_SOC vit_b16 5

# SOC-50 10-shot linear probing
bash scripts/MIM_linear_itpn/main.sh SOC_50classes vit_b16 10
```

Pre-trained weights are specified via the `SARATRX_PRETRAIN_CKPT` / `SARATRX_HIVIT_CKPT` env vars, or loaded from the repo `weights/` by default (see `resolve_pretrain_ckpt` in `trainers/MIM_linear_itpn.py`).

## Target Ablation (reconstruction target)

`scripts/MIM_linear_itpn/run_target_ablation.sh` sequentially runs 10-shot linear probing on SOC-50 with the pre-trained weights of all 8 reconstruction targets (pixel, single_s1~s6, multi). Weights live in `weights/base/<mode>/checkpoint-49.pth`.
