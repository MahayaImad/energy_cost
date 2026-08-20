#!/usr/bin/env bash
# Phase 2 ablations. One factor at a time, eps in {1, inf}, seed 0 only.
#
# Run this only after `python analyze.py results/ --check` passes on the main
# sweep. These runs deliberately CHANGE the workload (more clients, more local
# epochs, different partition skew), so per-round energy is expected to move
# here -- which is exactly why the main sweep, where workload is held fixed,
# has to be clean first. Otherwise there is no baseline to read them against.
#
# Written to results_ablation/ so they never mix into the main sweep's
# statistics: analyze.py globs every run_*.json in the directory it is given.
set -euo pipefail

OUT="${OUT:-results_ablation}"
mkdir -p "$OUT" logs
GPU_SHARE="${GPU_SHARE:-0.25}"
SEED=0

# Baseline values; each ablation varies exactly one of them.
BASE_ALPHA=0.5
BASE_CLIENTS=20
BASE_EPOCHS=1
BASE_CLIP=1.0

# eps=1 is the tight-privacy regime where the convergence effect lives;
# eps=inf isolates what is workload rather than privacy.
EPSILONS=("inf" 1)

run_one() {  # name alpha clients epochs clip eps
  local name="$1" alpha="$2" clients="$3" epochs="$4" clip="$5" eps="$6"
  local tag="${name}-eps${eps}"
  if compgen -G "$OUT/run_eps${eps//./p}_seed${SEED}_${tag}.json" > /dev/null; then
    echo "=== $tag (already present, skipping) ==="
    return
  fi
  echo "=== $tag  alpha=$alpha clients=$clients epochs=$epochs C=$clip ==="
  flwr run . --stream \
    --federation-config "num_supernodes=$clients client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1" \
    --run-config "seed=$SEED epsilon='$eps' run-id='$tag' results-dir='$OUT' num-supernodes=$clients dirichlet-alpha=$alpha local-epochs=$epochs clipping-norm=$clip" \
    2>&1 | tee "logs/${tag}.log"
}

python prewarm_sigma.py --ablation

for EPS in "${EPSILONS[@]}"; do
  # Partition skew. alpha=0.1 is severely non-IID, 10 is nearly IID.
  for A in 0.1 0.5 10; do
    run_one "alpha$A" "$A" $BASE_CLIENTS $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Federation size. num-supernodes must match the federation's, or the app
  # partitions the data for a different number of clients than exist.
  for N in 10 20 50; do
    run_one "clients$N" $BASE_ALPHA "$N" $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Local epochs. More local work per round: fewer rounds to converge, but
  # more privacy budget spent per round, so sigma rises to compensate.
  for E in 1 2 5; do
    run_one "epochs$E" $BASE_ALPHA $BASE_CLIENTS "$E" $BASE_CLIP "$EPS"
  done
  # Clipping norm. Only meaningful under DP; skipped for the baseline.
  if [ "$EPS" != "inf" ]; then
    for C in 0.5 1 2; do
      run_one "clip$C" $BASE_ALPHA $BASE_CLIENTS $BASE_EPOCHS "$C" "$EPS"
    done
  fi
done

echo
echo "=== ablation summary ==="
python analyze.py "$OUT" --ablation
