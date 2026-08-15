<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# Pre-training

SARATR-X-v2 performs **scale-aware structural masked pre-training** on **500K unlabeled SAR target chips**. This directory is based on [iTPN](https://github.com/sunny2109/iTPN) and [HiViT](https://github.com/zhangxiaosong18/hivit), and its key innovation is replacing the raw-pixel reconstruction target with a **multi-scale structural target** (Method section of the paper).

| Target | Description |
|---|---|
| `pixel` | Raw-pixel reconstruction (baseline) |
| `single_s1` | S1: 3×3 blind-spot neighborhood aggregation (zero center) |
| `single_s2..s6` | S2~S6: log-ratio directional contrast, r = 3/5/9/13/17 |
| `multi` | **Main setting**: softmax-constrained learnable multi-scale fusion |

## Environment

```bash
conda create -n saratrx_v2 python=3.10 -y
conda activate saratrx_v2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.5.4 tensorboard opencv-python numpy==1.26.4
```

## Data Preparation

Organize the 500K unlabeled SAR chips under arbitrary sub-folders (`ImageFolder`-style; labels are not used in training), or use `SARImageDataset` in `util/datasets.py` to load via a file list (the original 500K corpus format).

## Main Pre-training (1200 epochs)

```bash
bash run_pretrain.sh              # 8 GPUs
# or manually:
torchrun --nproc_per_node=8 main_pretrain.py \
    --model itpn_base_dec512d8b \
    --data_path /path/to/500K \
    --output_dir ./output/500K/multi \
    --epochs 1200 --warmup_epochs 5 --batch_size 200 \
    --mask_ratio 0.75 --target_mode multi --norm_pix_loss \
    --init_ckpt ../weights/base/itpn/itpn_base_fpn256.pth   # optional warm-start
```

`--init_ckpt` is an optional warm-start: load the official iTPN ImageNet weights (`itpn_base_fpn256.pth`) for a non-strict initialization, then continue pre-training with the SAR structural target (corresponding to the Implementation Details in the paper).

## Target Ablation (paper Table ablation)

```bash
bash run_pretrain_target_ablation.sh          # run all 8 target modes for 50 epochs
bash run_pretrain_target_ablation.sh 8 pixel  # run only one mode
```

Checkpoints (`checkpoint-49.pth`) are written directly to `<repo>/weights/base/<mode>/` so that the downstream
linear-probe ablation (`classification/fewshot/scripts/MIM_linear_itpn/run_target_ablation.sh`) picks them up.

## Code Structure

```
pre-training/
├── main_pretrain.py                  # Pre-training entry (`--target_mode` selects target)
├── main_finetune.py                  # Fully-supervised finetuning entry (ImageNet-style, optional)
├── engine_pretrain.py                # Pre-training training loop
├── engine_finetune.py                # Finetuning loop + evaluation
├── run_pretrain.sh                   # Main pre-training script
├── run_pretrain_target_ablation.sh   # Target ablation script
├── models/
│   ├── masked_autoencoder.py         # ★ Multi-scale structural target + MAE base class (method core)
│   ├── models_itpn.py                # iTPN hierarchical encoder (with FPN)
│   ├── models_itpn_mim.py            # iTPN masked autoencoder (pre-training model)
│   └── __init__.py
└── util/
    ├── datasets.py                   # Data loading (ImageFolder / file list)
    ├── misc.py / pos_embed.py / lr_sched.py / lr_decay.py / flop_count.py
```
