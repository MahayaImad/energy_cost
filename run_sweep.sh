#!/usr/bin/env bash
# Week 2: full privacy sweep. 6 epsilon values x 3 seeds = 18 runs.
# Run one epsilon first and time it before committing to the whole grid.
set -euo pipefail
mkdir -p results logs

EPSILONS=("inf" 8 4 2 1 0.5)
SEEDS=(0 1 2)

for EPS in "${EPSILONS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="eps${EPS}-s${SEED}"
    echo "=== $TAG ==="
    flwr run . --stream \
      --run-config "seed=$SEED epsilon='$EPS' run-id='$TAG'" \
      2>&1 | tee "logs/${TAG}.log"
  done
done