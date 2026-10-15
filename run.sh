#!/usr/bin/env bash
# Run the experiments of the paper.
#
#   ./run.sh main har            6 budgets x 3 seeds, 100 rounds -> results/har
#   ./run.sh main cifar10        6 budgets x 3 seeds, 30 rounds  -> results/cifar10
#   ./run.sh ablation har        epochs, clipping at eps in {1, inf} -> results/ablation_har
#   ./run.sh ablation cifar10    same on CIFAR-10                -> results/ablation_cifar10
#
# A run whose JSON already exists is skipped.
set -euo pipefail

MODE="$1"
DATASET="$2"
GPU_SHARE="${GPU_SHARE:-0.25}"
SEEDS="0 1 2"

if [ "$DATASET" = "har" ]; then CLIENTS=21; ROUNDS=100; else CLIENTS=20; ROUNDS=30; fi
if [ "$MODE" = "main" ]; then OUT="results/$DATASET"; else OUT="results/ablation_$DATASET"; fi
mkdir -p "$OUT" logs

run() {  # tag eps seed [extra run-config]
  local tag="$1" eps="$2" seed="$3" extra="${4:-}"
  if [ -e "$OUT/run_eps${eps//./p}_seed${seed}_${tag}.json" ]; then return; fi
  echo "=== $DATASET $tag ==="
  # Without --federation-config the simulation defaults to 2 CPU clients and
  # NVML would measure an idle GPU.
  flwr run . --stream \
    --federation-config "num_supernodes=$CLIENTS client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1" \
    --run-config "dataset='$DATASET' epsilon='$eps' seed=$seed run-id='$tag' results-dir='$OUT' num-supernodes=$CLIENTS num-server-rounds=$ROUNDS $extra" \
    2>&1 | tee "logs/$DATASET-$tag.log"
}

if [ "$MODE" = "main" ]; then
  # Randomised (but reproducible) order, so drift over the sweep cannot
  # masquerade as an effect of epsilon.
  python -c "
import random
pairs = [(e, s) for e in ['inf', '8', '4', '2', '1', '0.5'] for s in '012']
random.Random(1234).shuffle(pairs)
print('\n'.join(f'{e} {s}' for e, s in pairs))" |
  while read -r EPS SEED; do
    run "eps$EPS-s$SEED" "$EPS" "$SEED" < /dev/null
  done
else
  # One factor at a time; the others stay at epochs=1, C=1.0.
  for EPS in inf 1; do
    for SEED in $SEEDS; do
      for E in 1 2 5; do
        run "epochs$E-s$SEED-eps$EPS" "$EPS" "$SEED" "local-epochs=$E"
      done
      if [ "$EPS" != "inf" ]; then
        for C in 0.5 1 2; do
          run "clip$C-s$SEED-eps$EPS" "$EPS" "$SEED" "clipping-norm=$C"
        done
      fi
    done
  done
fi
