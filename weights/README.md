<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# weights — Pre-trained Weights

This directory stores the pre-trained weights of SARATR-X-v2. Since each model file is large (~1 GB), they are **not committed** to the repository; download them from the release and place them following the structure below.

## Directory Structure

```
weights/
├── base/                          # iTPN-Base series (embed_dim=512, depth=24)
│   ├── jiaquan_simple/            # [Main model] weighted simple fusion target
│   ├── jiaquan_complex/           # weighted complex fusion target (ablation)
│   ├── jiaquan_sum/               # direct-sum fusion target (ablation)
│   ├── scale_6/                   # 6-scale target setting (ablation)
│   ├── itpn/                      # baseline iTPN (official target)
│   ├── hivit/                     # baseline HiViT (comparison backbone)
│   ├── hivit_fused/               # HiViT + fused target (comparison backbone)
│   ├── pixel/                     # target ablation: pixel reconstruction
│   ├── single_s1 .. single_s6/    # target ablation: single-scale targets
│   └── multi/                     # target ablation: fused target
└── large/                         # iTPN-Large series (embed_dim=768)
    ├── jiaquan_simple/            # [Main model] iTPN-Large weighted simple fusion
    └── jiaquan_complex/           # weighted complex fusion target (ablation)
```

Each directory contains checkpoints saved per epoch, e.g. `checkpoint-{100,...,1600}.pth`. **The paper experiments use `checkpoint-1200.pth` by default.** The target-ablation directories (`pixel` / `single_s1..s6` / `multi`) contain the 50-epoch checkpoints (`checkpoint-49.pth`) produced by `pre-training/run_pretrain_target_ablation.sh`.

## Mapping to Downstream Tasks

| Weight | Classification | Detection | Segmentation |
| --- | --- | --- | --- |
| `base/jiaquan_simple/checkpoint-1200.pth` | Linear eval / few-shot (iTPN-B) | ReDet/RoI Trans + iTPN-B | UperNet + iTPN-B |
| `large/jiaquan_simple/checkpoint-1200.pth` | Linear eval (iTPN-L) | ReDet + iTPN-L | UperNet + iTPN-L |
| `base/hivit/checkpoint-1200.pth` | Few-shot (HiViT) | — | UperNet + HiViT |

## Usage

Each module resolves weights via environment variables or default paths.

- **Linear eval** (`classification/linear_eval`): `SARATRX_PRETRAIN_CKPT` (default `weights/base/jiaquan_simple/checkpoint-1200.pth`)
- **Few-shot** (`classification/fewshot`): `SARATRX_PRETRAIN_CKPT` (iTPN) / `SARATRX_HIVIT_CKPT` (HiViT)
- **Detection** (`detection`): `ITPN_CKPT` (iTPN-B) / `ITPN_LARGE_CKPT` (iTPN-L), default `ckpts/iTPN/...`
- **Segmentation** (`segmentation`): set `model.pretrained` in config or pass `--cfg-options model.pretrained=...`
- **Visualization** (`visualize`): `SARATRX_PRETRAIN_CKPT`

> Note: The rest of the repo resolves weights from `weights/` at the repo root by default; detection/segmentation
> run inside the mmrotate/mmseg environments, so place weights under the framework `ckpts/` per their READMEs.
