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

# Dataset, split and round budget. These MUST be threaded into the run-config:
# without them the runs silently fall back to the pyproject defaults, which is
# CIFAR-10 at 30 rounds. A DATASET=har invocation that never reaches the app
# produces a full, plausible-looking CIFAR ablation instead -- there is nothing
# in the output to suggest the requested dataset was ignored.
DATASET="${DATASET:-cifar10}"
HAR_SPLIT="${HAR_SPLIT:-official}"
ROUNDS="${ROUNDS:-30}"

if [ "$DATASET" = "har" ]; then
  [ "$HAR_SPLIT" = "official" ] && BASE_CLIENTS_DEFAULT=21 || BASE_CLIENTS_DEFAULT=30
  OUT="${OUT:-results_ablation_har}"
else
  BASE_CLIENTS_DEFAULT=20
  OUT="${OUT:-results_ablation}"
fi
mkdir -p "$OUT" logs
GPU_SHARE="${GPU_SHARE:-0.25}"
# Seeds per cell. The brief specifies seed 0 only, which is fine for reading
# a direction, but any cell quoted as a RESULT needs more than one run: the
# noise floor is ~6%, so a single run cannot separate a real 10% shift from
# variance. Re-run a specific axis with three seeds before publishing it, e.g.
#   SEEDS="0 1 2" AXES=epochs ./run_ablations.sh
SEEDS="${SEEDS:-0}"
# Which axes to run: any space-separated subset of "alpha clients epochs clip".
# On HAR only epochs and clip exist: a client IS a study participant, so the
# federation size is fixed by the split and there is no Dirichlet alpha to
# vary -- the partition comes from who contributed the data.
if [ "$DATASET" = "har" ]; then
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
  AXES="${AXES:-alpha clients epochs clip}"
fi

has_axis() { [[ " $AXES " == *" $1 "* ]]; }

# Baseline values; each ablation varies exactly one of them.
BASE_ALPHA=0.5
BASE_CLIENTS=$BASE_CLIENTS_DEFAULT
BASE_EPOCHS=1
BASE_CLIP=1.0

# eps=1 is the tight-privacy regime where the convergence effect lives;
# eps=inf isolates what is workload rather than privacy.
EPSILONS=("inf" 1)

run_one() {  # name alpha clients epochs clip eps
  local name="$1" alpha="$2" clients="$3" epochs="$4" clip="$5" eps="$6"
  local seed tag
  for seed in $SEEDS; do
    tag="${name}-s${seed}-eps${eps}"
    # Also match the pre-multi-seed naming, which carried no seed in the tag,
    # so an existing single-seed grid is not needlessly re-run.
    legacy="$OUT/run_eps${eps//./p}_seed${seed}_${name}-eps${eps}.json"
    if compgen -G "$OUT/run_eps${eps//./p}_seed${seed}_${tag}.json" > /dev/null \
       || compgen -G "$legacy" > /dev/null; then
      echo "=== $tag (already present, skipping) ==="
      continue
    fi
    echo "=== $tag  ds=$DATASET rounds=$ROUNDS alpha=$alpha clients=$clients epochs=$epochs C=$clip ==="
    flwr run . --stream \
      --federation-config "num_supernodes=$clients client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1" \
      --run-config "seed=$seed epsilon='$eps' run-id='$tag' results-dir='$OUT' dataset='$DATASET' har-split='$HAR_SPLIT' num-server-rounds=$ROUNDS num-supernodes=$clients dirichlet-alpha=$alpha local-epochs=$epochs clipping-norm=$clip" \
      2>&1 | tee "logs/${tag}.log"
  done
}

# Echo the effective configuration. The CIFAR-instead-of-HAR mistake this
# guards against is invisible in the results: every table renders normally.
echo "ablations: dataset=$DATASET rounds=$ROUNDS clients=$BASE_CLIENTS "\
     "axes='$AXES' seeds='$SEEDS' -> $OUT"

if [ "$DATASET" = "har" ]; then
  python prewarm_sigma.py --ablation --dataset har --har-split "$HAR_SPLIT" --rounds "$ROUNDS"
else
  python prewarm_sigma.py --ablation --rounds "$ROUNDS"
fi

for EPS in "${EPSILONS[@]}"; do
  # Partition skew. alpha=0.1 is severely non-IID, 10 is nearly IID.
  has_axis alpha && for A in 0.1 0.5 10; do
    run_one "alpha$A" "$A" $BASE_CLIENTS $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Federation size. num-supernodes must match the federation's, or the app
  # partitions the data for a different number of clients than exist.
  has_axis clients && for N in 10 20 50; do
    run_one "clients$N" $BASE_ALPHA "$N" $BASE_EPOCHS $BASE_CLIP "$EPS"
  done
  # Local epochs. More local work per round: fewer rounds to converge, but
  # more privacy budget spent per round, so sigma rises to compensate.
  has_axis epochs && for E in 1 2 5; do
    run_one "epochs$E" $BASE_ALPHA $BASE_CLIENTS "$E" $BASE_CLIP "$EPS"
  done
  # Clipping norm. Only meaningful under DP; skipped for the baseline.
  if [ "$EPS" != "inf" ] && has_axis clip; then
    for C in 0.5 1 2; do
      run_one "clip$C" $BASE_ALPHA $BASE_CLIENTS $BASE_EPOCHS "$C" "$EPS"
    done
  fi
done

echo
echo "=== ablation summary ==="
python analyze.py "$OUT" --ablation
