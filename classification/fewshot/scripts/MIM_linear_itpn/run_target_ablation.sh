#!/usr/bin/env bash
# =============================================================================
# 不同重建目标（target）预训练权重的 SOC-50 线性探测消融实验
# (Linear probe ablation on SOC_50classes for each target-mode pretrained weight)
#
# 权重放置 (weight layout)：
#   <repo>/weights/target_ablation/<mode>/checkpoint-49.pth
#   mode ∈ {pixel, single_s1, ..., single_s6, multi}
#   可通过环境变量 TARGET_WEIGHT_ROOT 覆盖权重根目录。
#
# 输出 (output)：output/SOC_50classes/MIM_linear_itpn/<mode>_vit_b16_10shots/seed*
#
# 用法（在 classification/fewshot 目录下）(Usage):
#   bash scripts/MIM_linear_itpn/run_target_ablation.sh
#   bash scripts/MIM_linear_itpn/run_target_ablation.sh 10 pixel   # 10-shot, only pixel
#   bash scripts/MIM_linear_itpn/run_target_ablation.sh 10 all 0   # 10-shot, seed 0 only
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

# 数据根目录：指向仓库 dataset/classification
# (data root: points to dataset/classification in the repo)
DATA="${ROOT}/../../dataset/classification/"
TRAINER="MIM_linear_itpn"
DATASET="SOC_50classes"
CFG="vit_b16"
SHOTS="${1:-10}"
ONLY_MODE="${2:-all}"
ONLY_SEED="${3:-all}"

# 消融权重根目录（默认仓库 weights/target_ablation，可用环境变量覆盖）
WEIGHT_ROOT="${TARGET_WEIGHT_ROOT:-${ROOT}/../../weights/target_ablation}"
CKPT_NAME="checkpoint-49.pth"

TARGET_MODES=(
  "pixel"
  "single_s1" "single_s2" "single_s3" "single_s4" "single_s5" "single_s6"
  "multi"
)

run_one_seed() {
  local mode="$1"
  local seed="$2"
  local ckpt="${WEIGHT_ROOT}/${mode}/${CKPT_NAME}"
  local out_dir="output/${DATASET}/${TRAINER}/${mode}_${CFG}_${SHOTS}shots/seed${seed}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] checkpoint not found: ${ckpt}"
    return 1
  fi
  if [[ -d "${out_dir}" ]]; then
    echo "[skip] results exist: ${out_dir}"
    return 0
  fi

  echo "============================================================"
  echo " mode=${mode} seed=${seed} shots=${SHOTS}"
  echo " ckpt=${ckpt}"
  echo " out=${out_dir}"
  echo "============================================================"

  export SARATRX_PRETRAIN_CKPT="${ckpt}"
  python train.py \
    --root "${DATA}" \
    --seed "${seed}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${out_dir}" \
    DATASET.NUM_SHOTS "${SHOTS}"
}

if [[ "${ONLY_MODE}" != "all" ]]; then
  TARGET_MODES=("${ONLY_MODE}")
fi

if [[ "${ONLY_SEED}" != "all" ]]; then
  SEEDS=("${ONLY_SEED}")
else
  SEEDS=($(seq 0 9))
fi

for mode in "${TARGET_MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one_seed "${mode}" "${seed}" || true
  done
done

echo ""
echo "Done. Summarize with:"
echo "  python scripts/MIM_linear_itpn/collect_target_ablation_results.py"
