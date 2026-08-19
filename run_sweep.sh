#!/usr/bin/env bash
# Phase 1: full privacy sweep. 6 epsilon values x 3 seeds = 18 runs.
#
# BOTH GPU flags are required (see the project gotchas):
#   --init-args-num-gpus 1        Ray sees a GPU at all
#   --client-resources-num-gpus   each ClientApp reserves a share
# Without the first, ClientApps run on CPU, NVML measures an idle card, the
# run still looks fine, and every joule recorded is meaningless. The sanity
# check at the bottom of this script exists to catch exactly that.
set -euo pipefail
mkdir -p results logs

EPSILONS=("inf" 8 4 2 1 0.5)
SEEDS=(0 1 2)
GPU_SHARE="${GPU_SHARE:-0.25}"

for EPS in "${EPSILONS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="eps${EPS}-s${SEED}"
    if compgen -G "results/run_eps${EPS//./p}_seed${SEED}_${TAG}.json" > /dev/null; then
      echo "=== $TAG (already present, skipping) ==="
      continue
    fi
    echo "=== $TAG ==="
    flwr run . --stream \
      --init-args-num-gpus 1 \
      --client-resources-num-gpus "$GPU_SHARE" \
      --run-config "seed=$SEED epsilon='$EPS' run-id='$TAG'" \
      2>&1 | tee "logs/${TAG}.log"
  done
done

echo
echo "=== Phase 1 verification ==="
python analyze.py results/ --check
