#!/usr/bin/env bash
# =============================================================================
# SARATR-X-v2 主预训练脚本（500K 无标注 SAR 图像，多尺度结构目标）
#
#   --target_mode multi : 可学习 softmax 多尺度融合（论文主设置）
#   --init_ckpt         : 可选，iTPN ImageNet 权重 warm-start
#                         （如官方 itpn_base_fpn256.pth）
#
# Usage:
#   bash run_pretrain.sh                 # 8 GPUs 启动
#   bash run_pretrain.sh 4               # 4 GPUs 启动
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

NUM_GPUS="${1:-8}"

DATA_PATH="/home/data/500K/"        # 500K 预训练数据根目录
OUTPUT_DIR="${ROOT}/output/500K/multi"
INIT_CKPT="${ROOT}/weights/itpn_base_fpn256.pth"   # 可选 warm-start 权重

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" main_pretrain.py \
  --model itpn_base_dec512d8b \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${OUTPUT_DIR}" \
  --epochs 1200 \
  --warmup_epochs 5 \
  --batch_size 200 \
  --mask_ratio 0.75 \
  --target_mode multi \
  --norm_pix_loss \
  --init_ckpt "${INIT_CKPT}" \
  --seed 0

echo "[done] pretraining finished, checkpoints in: ${OUTPUT_DIR}/"
