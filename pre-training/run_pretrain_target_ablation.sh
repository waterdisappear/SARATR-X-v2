#!/usr/bin/env bash
# =============================================================================
# Target ablation pre-training (50 epochs each)
#   - pixel          : 原始像素重建
#   - single_s1..s6  : 单尺度固定权重 (S1=3x3 blind-spot, S2-S6=log-ratio)
#   - multi          : simple 可学习多尺度融合 (softmax simple_weights)
#
# Model : itpn_base_dec512d8b（init: itpn_base_fpn256.pth，见 main_pretrain.py）
# Data  : /home/data/500K/
# Output: <项目根目录>/pretrain_ablation/base/<target_mode>/
# Weights: checkpoint-49.pth（仅最后一轮，50 epoch）
#
# Usage:
#   cd pre-training
#   bash run_pretrain_target_ablation.sh          # 8 GPUs
#   bash run_pretrain_target_ablation.sh 4        # 4 GPUs
#   bash run_pretrain_target_ablation.sh 8 pixel  # 只跑 pixel
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

NUM_GPUS="${1:-8}"
ONLY_MODE="${2:-}"   # 可选：只跑某一种 target_mode

DATA_PATH="/home/data/500K/"
OUTPUT_ROOT="${ROOT}/output/pretrain_ablation/base"
EPOCHS=50
BATCH_SIZE=200
WARMUP=5
MODEL="itpn_base_dec512d8b"
INIT_CKPT=""   # 可选 warm-start 权重（如 itpn_base_fpn256.pth）

TARGET_MODES=(
  "pixel"
  "single_s1"
  "single_s2"
  "single_s3"
  "single_s4"
  "single_s5"
  "single_s6"
  "multi"
)

run_one() {
  local mode="$1"
  local out_dir="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out_dir}"

  echo ""
  echo "============================================================"
  echo " target_mode = ${mode}"
  echo " output_dir  = ${out_dir}"
  echo " data_path   = ${DATA_PATH}"
  echo " epochs      = ${EPOCHS}"
  echo "============================================================"

  torchrun --nproc_per_node="${NUM_GPUS}" main_pretrain.py \
    --model "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${out_dir}" \
    --log_dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP}" \
    --batch_size "${BATCH_SIZE}" \
    --mask_ratio 0.75 \
    --target_mode "${mode}" \
    --norm_pix_loss \
    --init_ckpt "${INIT_CKPT}" \
    --seed 0

  echo "[done] ${mode} -> ${out_dir}/checkpoint-49.pth"
}

if [[ -n "${ONLY_MODE}" ]]; then
  run_one "${ONLY_MODE}"
  exit 0
fi

for mode in "${TARGET_MODES[@]}"; do
  run_one "${mode}"
done

echo ""
echo "All ablation runs finished. Checkpoints under: ${OUTPUT_ROOT}/"
