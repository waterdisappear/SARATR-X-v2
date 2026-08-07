# Pre-training / 预训练

SARATR-X-v2 performs **scale-aware structural masked pre-training** on **500K unlabeled SAR target chips**. This directory is based on [iTPN](https://github.com/sunny2109/iTPN) and [HiViT](https://github.com/zhangxiaosong18/hivit), and its key innovation is replacing the raw-pixel reconstruction target with a **multi-scale structural target** (Method section of the paper).

SARATR-X-v2 在 **500K 张无标注 SAR 目标切片**上执行"尺度感知结构掩码预训练"。本目录代码基于 [iTPN](https://github.com/sunny2109/iTPN) 与 [HiViT](https://github.com/zhangxiaosong18/hivit)，核心创新是把重建目标从原始像素替换为**多尺度结构目标**（论文 Method 章节）。

| Target / 目标 | Description / 说明 |
|---|---|
| `pixel` | Raw-pixel reconstruction (baseline) / 原始像素重建（基线） |
| `single_s1` | S1: 3×3 blind-spot neighborhood aggregation (zero center) / 盲点邻域聚合（中心像素置 0） |
| `single_s2..s6` | S2~S6: log-ratio directional contrast, r = 3/5/9/13/17 / 对数比方向对比 |
| `multi` | **Main setting / 论文主设置**: softmax-constrained learnable multi-scale fusion / 可学习多尺度融合 |

## Environment / 环境要求

```bash
conda create -n saratrx_v2 python=3.10 -y
conda activate saratrx_v2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.5.4 tensorboard opencv-python numpy==1.26.4
```

## Data Preparation / 数据准备

Organize the 500K unlabeled SAR chips under arbitrary sub-folders (`ImageFolder`-style; labels are not used in training), or use `SARImageDataset` in `util/datasets.py` to load via a file list (the original 500K corpus format).

将 500K 无标注 SAR 切片按任意子文件夹组织（`ImageFolder` 风格，标签不参与训练），或用 `util/datasets.py` 中的 `SARImageDataset` 按文件列表加载（旧 500K 语料格式）。

## Main Pre-training (1200 epochs) / 主预训练（1200 epochs）

```bash
bash run_pretrain.sh              # 8 GPUs
# or manually / 或手动执行：
torchrun --nproc_per_node=8 main_pretrain.py \
    --model itpn_base_dec512d8b \
    --data_path /path/to/500K \
    --output_dir ./output/500K/multi \
    --epochs 1200 --warmup_epochs 5 --batch_size 200 \
    --mask_ratio 0.75 --target_mode multi --norm_pix_loss \
    --init_ckpt ../weights/base/itpn/itpn_base_fpn256.pth   # optional warm-start / 可选 warm-start
```

`--init_ckpt` is an optional warm-start: load the official iTPN ImageNet weights (`itpn_base_fpn256.pth`) for a non-strict initialization, then continue pre-training with the SAR structural target (corresponding to the Implementation Details in the paper).

`--init_ckpt` 是可选 warm-start：加载 iTPN 官方 ImageNet 权重（`itpn_base_fpn256.pth`）做非严格初始化，然后继续 SAR 结构目标预训练（对应论文 Implementation Details）。

## Target Ablation (paper Table ablation) / 目标消融（论文 Table 消融）

```bash
bash run_pretrain_target_ablation.sh          # run all 8 target modes for 50 epochs / 8 个 target_mode 各跑 50 epochs
bash run_pretrain_target_ablation.sh 8 pixel  # run only one mode / 只跑某一种
```

Checkpoints (`checkpoint-49.pth`) are written directly to `<repo>/weights/base/<mode>/` so that the downstream
linear-probe ablation (`classification/fewshot/scripts/MIM_linear_itpn/run_target_ablation.sh`) picks them up.

消融权重（`checkpoint-49.pth`）直接输出到 `<repo>/weights/base/<mode>/`，供下游少样本线性探测消融
（`classification/fewshot/scripts/MIM_linear_itpn/run_target_ablation.sh`）直接使用。

## Code Structure / 代码结构

```
pre-training/
├── main_pretrain.py                  # Pre-training entry (`--target_mode` selects target) / 预训练主入口
├── main_finetune.py                  # Fully-supervised finetuning entry (ImageNet-style, optional) / 全监督微调入口
├── engine_pretrain.py                # Pre-training training loop / 预训练单轮训练
├── engine_finetune.py                # Finetuning loop + evaluation / 微调单轮训练 + 评估
├── run_pretrain.sh                   # Main pre-training script / 主预训练启动脚本
├── run_pretrain_target_ablation.sh   # Target ablation script / 目标消融预训练脚本
├── models/
│   ├── masked_autoencoder.py         # ★ Multi-scale structural target + MAE base class (method core) / 多尺度结构目标 + MAE 基类
│   ├── models_itpn.py                # iTPN hierarchical encoder (with FPN) / iTPN 分层编码器
│   ├── models_itpn_mim.py            # iTPN masked autoencoder (pre-training model) / iTPN 掩码自编码器
│   └── __init__.py
└── util/
    ├── datasets.py                   # Data loading (ImageFolder / file list) / 数据加载
    ├── misc.py / pos_embed.py / lr_sched.py / lr_decay.py / flop_count.py
```
