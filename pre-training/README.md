# Pre-training / 预训练

SARATR-X-v2 在 **500K 张无标注 SAR 目标切片**上执行“尺度感知结构掩码预训练”。
本目录代码基于 [iTPN](https://github.com/sunny2109/iTPN) 与 [HiViT](https://github.com/zhangxiaosong18/hivit)，
核心创新是把重建目标从原始像素替换为**多尺度结构目标**（论文 Method 章节）：

| 目标 | 说明 |
|---|---|
| `pixel` | 原始像素重建（基线） |
| `single_s1` | S1：3x3 blind-spot 邻域聚合（中心像素置 0） |
| `single_s2..s6` | S2~S6：log-ratio 方向对比，r = 3/5/9/13/17 |
| `multi` | **论文主设置**：softmax 约束的可学习多尺度融合 |

## 环境要求

```bash
conda create -n saratrx_v2 python=3.10 -y
conda activate saratrx_v2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.5.4 tensorboard opencv-python numpy==1.26.4
```

## 数据准备

将 500K 无标注 SAR 切片按任意子文件夹组织（`ImageFolder` 风格，标签不参与训练），
或用 `util/datasets.py` 中的 `SARImageDataset` 按文件列表加载（旧 500K 语料格式）。

## 主预训练（1200 epochs）

```bash
bash run_pretrain.sh              # 8 GPUs
# 或手动执行：
torchrun --nproc_per_node=8 main_pretrain.py \
    --model itpn_base_dec512d8b \
    --data_path /path/to/500K \
    --output_dir ./output/500K/multi \
    --epochs 1200 --warmup_epochs 5 --batch_size 200 \
    --mask_ratio 0.75 --target_mode multi --norm_pix_loss \
    --init_ckpt ./weights/itpn_base_fpn256.pth   # 可选 warm-start
```

`--init_ckpt` 是可选 warm-start：加载 iTPN 官方 ImageNet 权重（`itpn_base_fpn256.pth`）
做非严格初始化，然后继续 SAR 结构目标预训练（对应论文 Implementation Details）。

## 目标消融（论文 Table 消融）

```bash
bash run_pretrain_target_ablation.sh          # 8 个 target_mode 各跑 50 epochs
bash run_pretrain_target_ablation.sh 8 pixel  # 只跑某一种
```

## 代码结构

```
pre-training/
├── main_pretrain.py                  # 预训练主入口（--target_mode 选择目标）
├── main_finetune.py                  # 全监督微调入口（ImageNet 风格，可选）
├── engine_pretrain.py                # 预训练单轮训练
├── engine_finetune.py                # 微调单轮训练 + 评估
├── run_pretrain.sh                   # 主预训练启动脚本
├── run_pretrain_target_ablation.sh   # 目标消融预训练脚本
├── models/
│   ├── masked_autoencoder.py         # ★ 多尺度结构目标 + MAE 基类（方法核心）
│   ├── models_itpn.py                # iTPN 分层编码器（含 FPN）
│   ├── models_itpn_mim.py            # iTPN 掩码自编码器（预训练模型）
│   └── __init__.py
└── util/
    ├── datasets.py                   # 数据加载（ImageFolder / 文件列表）
    ├── misc.py / pos_embed.py / lr_sched.py / lr_decay.py / flop_count.py
