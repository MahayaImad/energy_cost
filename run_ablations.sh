#!/usr/bin/env bash
# Phase 2 ablations: one factor at a time, at eps in {inf, 1}.
#
#   ./run_ablations.sh                                        CIFAR-10
#   SEEDS="0 1 2" AXES=epochs DATASET=har ROUNDS=100 ./run_ablations.sh
#
# Run this only after `python analyze.py results/ --check` passes on the main
# sweep. These runs deliberately CHANGE the workload, so per-round energy is
# expected to move here -- which is exactly why the main sweep, where the
# workload is held fixed, has to be clean first. Otherwise there is no
# baseline to read them against.
#
# Written to a separate directory so they never mix into the main sweep's
# statistics: analyze.py globs every run_*.json in the directory it is given.
set -euo pipefail

# These MUST be threaded into the run-config. Without them the runs fall back
# to the pyproject defaults -- CIFAR-10 at 30 rounds -- and a DATASET=har
# invocation that never reaches the app produces a full, plausible-looking
# CIFAR ablation with nothing in the output to say so.
DATASET="${DATASET:-cifar10}"
ROUNDS="${ROUNDS:-30}"
GPU_SHARE="${GPU_SHARE:-0.25}"

# Seeds per cell. One seed reads a direction; any cell quoted as a RESULT
# needs three, because the noise floor is several percent and a single run
# cannot separate a real 10% shift from variance.
SEEDS="${SEEDS:-0}"

# On HAR only epochs and clip exist: a client IS a study participant, so the
# federation size is fixed by the split and there is no Dirichlet alpha to
# vary -- the partition comes from who contributed the data.
if [ "$DATASET" = "har" ]; then
  BASE_CLIENTS=21
  OUT="${OUT:-results_ablation_har}"
  AXES="${AXES:-epochs clip}"
  for a in $AXES; do
    case "$a" in
      alpha|clients)
        echo "error: axis '$a' does not exist on HAR (clients are participants," >&2
        echo "       so the partition and the federation size are both fixed)." >&2
        exit 1 ;;
    esac
  done
else
  BASE_CLIENTS=20
  OUT="${OUT:-results_ablation}"
  AXES="${AXES:-alpha clients epochs clip}"
fi
mkdir -p "$OUT" logs

# Baseline values; each ablation varies exactly one of them.
BASE_ALPHA=0.5
BASE_EPOCHS=1
BASE_CLIP=1.0

# eps=1 is the tight-privacy regime where the convergence effect lives;
# eps=inf isolates what is workload rather than privacy.
EPSILONS=("inf" 1)

has_axis() { [[ " $AXES " == *" $1 "* ]]; }

run_one() {  # name alpha clients epochs clip eps
  local name="$1" alpha="$2" clients="$3" epochs="$4" clip="$5" eps="$6"
  local seed tag
  for seed in $SEEDS; do
    tag="${name}-s${seed}-eps${eps}"
    if compgen -G "$OUT/run_eps${eps//./p}_seed${seed}_${tag}.json" > /dev/null; then
      echo "=== $tag (already present, skipping) ==="
      continue
    fi
    echo "=== $tag  ds=$DATASET rounds=$ROUNDS alpha=$alpha clients=$clients epochs=$epochs C=$clip ==="
    flwr run . --stream \
      --federation-config "num_supernodes=$clients client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1" \
      --run-config "seed=$seed epsilon='$eps' run-id='$tag' results-dir='$OUT' dataset='$DATASET' num-server-rounds=$ROUNDS num-supernodes=$clients dirichlet-alpha=$alpha local-epochs=$epochs clipping-norm=$clip" \
      2>&1 | tee "logs/${tag}.log"
  done
}

# Echo the effective configuration: the CIFAR-instead-of-HAR mistake this
# guards against is invisible in the results, because every table renders
# normally either way.
echo "ablations: dataset=$DATASET rounds=$ROUNDS clients=$BASE_CLIENTS" \
     "axes='$AXES' seeds='$SEEDS' -> $OUT"

python prewarm_sigma.py --ablation --dataset "$DATASET" --rounds "$ROUNDS"

for EPS in "${EPSILONS[@]}"; do
  # Partition skew. alpha=0.1 is severely non-IID, 10 is nearly IID.
  has_axis alpha && for A in 0.1 0.5 10; do
    run_one "alpha$A" "$A" $BASE_CLIENTS $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Federation size.
  has_axis clients && for N in 10 20 50; do
    run_one "clients$N" $BASE_ALPHA "$N" $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Local epochs. More local work per round means fewer rounds to converge,
  # but more privacy budget spent per round, so sigma rises to compensate.
  has_axis epochs && for E in 1 2 5; do
    run_one "epochs$E" $BASE_ALPHA $BASE_CLIENTS "$E" $BASE_CLIP "$EPS"
  done
  # Clipping norm. Only meaningful under DP, so skipped for the baseline.
  if [ "$EPS" != "inf" ] && has_axis clip; then
    for C in 0.5 1 2; do
      run_one "clip$C" $BASE_ALPHA $BASE_CLIENTS $BASE_EPOCHS "$C" "$EPS"
    done
  fi
done

echo
echo "=== ablation summary ==="
python analyze.py "$OUT" --ablation
