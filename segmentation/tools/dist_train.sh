#!/usr/bin/env bash
# Pure LF (no CRLF). On server after bad checkout: sed -i 's/\r$//' tools/dist_train.sh
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python -m torch.distributed.launch --nproc_per_node="${GPUS}" --master_port="${PORT}" "${_ROOT}/tools/train.py" "${CONFIG}" --launcher pytorch "${@:3}"
