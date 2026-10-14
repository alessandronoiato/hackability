#!/usr/bin/env bash
# GRPO sweep over arms x conditions x seeds, one job per GPU at a time.
#
#   bash experiments/tie_drift_llm/run_sweep.sh experiments/tie_drift_llm/data/mul2x2.json 0,1,2,3
#
#   CONDS="rule exception" bash experiments/tie_drift_llm/run_sweep.sh experiments/tie_drift_llm/data/cue_s0.json 0
#
# Environment overrides (defaults in brackets):
#   SEEDS ["0 1 2"]   CONDS [$KS or "1 128"]   ARMS ["adam sgd adam_mu2 adam_mu2_clip"]   EXTRA [""]
# CONDS are grader conditions: levels K (K design) or rule / anti / exception (cue design).
# KS is the old name for CONDS and still works.
# Arms:
#   adam           AdamW lr 1e-6, beta2 0.999, on-policy (one step per batch)
#   sgd            SGD, lr auto (first step has the norm of a 1e-6 sign step)
#   sign           sign descent lr 1e-6
#   adam_mu2       AdamW, 2 steps per batch, no ratio clipping
#   adam_mu2_clip  AdamW, 2 steps per batch, PPO clip 0.2   (compare with adam_mu2)
#   adam_frozenemb AdamW with token embeddings / LM head frozen
# Logs: <dataset dir>/runs/<dataset stem>/logs/<arm>_<K<K> | cond>_s<seed>.log
set -euo pipefail
DS=${1:?dataset json}
GPUS=${2:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
SEEDS=${SEEDS:-"0 1 2"}
CONDS=${CONDS:-${KS:-"1 128"}}
ARMS=${ARMS:-"adam sgd adam_mu2 adam_mu2_clip"}
EXTRA=${EXTRA:-""}
LOGDIR="$(dirname "$DS")/runs/$(basename "$DS" .json)/logs"
mkdir -p "$LOGDIR"

arm_args() {
  case "$1" in
    adam)           echo "--optimizer adam --lr 1e-6" ;;
    sgd)            echo "--optimizer sgd" ;;
    sign)           echo "--optimizer sign --lr 1e-6" ;;
    adam_mu2)       echo "--optimizer adam --lr 1e-6 --inner-steps 2" ;;
    adam_mu2_clip)  echo "--optimizer adam --lr 1e-6 --inner-steps 2 --clip-eps 0.2" ;;
    adam_frozenemb) echo "--optimizer adam --lr 1e-6 --freeze-embeddings" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

# Seed-major order: the first finished jobs already form complete K pairs.
JOBS=()
for s in $SEEDS; do for arm in $ARMS; do for c in $CONDS; do JOBS+=("$arm $c $s"); done; done; done
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NG=${#GPU_LIST[@]}
echo "${#JOBS[@]} jobs on GPUs ${GPU_LIST[*]}; logs in $LOGDIR"

worker() {
  local g=$1 idx=$2
  for ((j = idx; j < ${#JOBS[@]}; j += NG)); do
    read -r arm c s <<< "${JOBS[$j]}"
    if [[ $c =~ ^[0-9]+$ ]]; then tag="K$c"; else tag=$c; fi
    log="$LOGDIR/${arm}_${tag}_s${s}.log"
    echo "[gpu $g] $arm cond=$c seed=$s"
    # shellcheck disable=SC2046
    CUDA_VISIBLE_DEVICES=$g python "$HERE/grpo.py" --dataset "$DS" --cond "$c" --seed "$s" \
      $(arm_args "$arm") $EXTRA > "$log" 2>&1 || echo "[gpu $g] FAILED: $log"
  done
}

for ((w = 0; w < NG; w++)); do worker "${GPU_LIST[$w]}" "$w" & done
wait
python "$HERE/analyze.py" --dataset "$DS"
