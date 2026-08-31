#!/usr/bin/env bash
# Phase 1: the privacy sweep. 6 epsilon values x 3 seeds = 18 runs.
#
#   ./run_sweep.sh                          CIFAR-10, 30 rounds -> results/
#   DATASET=har ROUNDS=100 ./run_sweep.sh   UCI-HAR, 100 rounds -> results_har/
#
# Simulation resources come from --federation-config (flwr >= 1.33). Omit it
# and the defaults are num_supernodes=2 with no GPU: the run completes, NVML
# measures an idle card, and every joule recorded is meaningless. The keys are
# flat and underscored here, unlike the hyphenated form used in a Flower
# config file.
set -euo pipefail

DATASET="${DATASET:-cifar10}"
# Round budget. Passed to BOTH the prewarm and the runs: sigma depends on it,
# so a mismatch silently voids the sigma cache.
ROUNDS="${ROUNDS:-30}"
GPU_SHARE="${GPU_SHARE:-0.25}"
SEEDS=(0 1 2)
EPSILONS=("inf" 8 4 2 1 0.5)

# On HAR a client IS a study participant, so the federation size is fixed at
# the 21 participants of the published subject-disjoint split.
if [ "$DATASET" = "har" ]; then
  NUM_SUPERNODES=21
  OUT="${OUT:-results_har}"
else
  NUM_SUPERNODES=20
  OUT="${OUT:-results}"
fi
mkdir -p "$OUT" logs

# A venv rebuild silently resolving flwr down to 1.31 has cost this project
# two sessions: 1.31 has no --federation-config, so the sweep would fall back
# to 2 CPU clients or fail outright.
python - <<'EOF' || exit 1
import sys, flwr
if tuple(int(x) for x in flwr.__version__.split(".")[:2]) < (1, 33):
    sys.exit(f"flwr {flwr.__version__} is too old; >=1.33 required.\n"
             "  pip install --upgrade 'flwr[simulation]>=1.33.0'")
print(f"flwr {flwr.__version__} ok")
EOF

# Randomised run order. Sweeping epsilon in a fixed order makes execution
# order and epsilon almost perfectly rank-correlated, so any drift over the
# sweep is indistinguishable from a cost of privacy -- and the first sweep
# showed exactly that signature. Shuffling lets analyze.py check [5] tell them
# apart. ORDER_SEED keeps it reproducible.
ORDER_SEED="${ORDER_SEED:-1234}"
mapfile -t PAIRS < <(python - "$ORDER_SEED" "${EPSILONS[*]}" "${SEEDS[*]}" <<'EOF'
import random, sys
seed, eps, seeds = int(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
pairs = [f"{e} {s}" for e in eps for s in seeds]
random.Random(seed).shuffle(pairs)
print("\n".join(pairs))
EOF
)

echo "sweep: dataset=$DATASET rounds=$ROUNDS clients=$NUM_SUPERNODES -> $OUT"
echo "run order (${#PAIRS[@]} runs, order seed $ORDER_SEED):"
printf '  %s\n' "${PAIRS[@]}"
echo

# Derive every sigma before any measurement starts, so the accountant's binary
# search never runs inside a measured round. Cheap, and idempotent.
python prewarm_sigma.py --dataset "$DATASET" --rounds "$ROUNDS"

for PAIR in "${PAIRS[@]}"; do
  read -r EPS SEED <<< "$PAIR"
  TAG="eps${EPS}-s${SEED}"
  if compgen -G "$OUT/run_eps${EPS//./p}_seed${SEED}_${TAG}.json" > /dev/null; then
    echo "=== $TAG (already present, skipping) ==="
    continue
  fi
  echo "=== $TAG ==="
  flwr run . --stream \
    --federation-config "num_supernodes=$NUM_SUPERNODES client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1" \
    --run-config "seed=$SEED epsilon='$EPS' run-id='$TAG' dataset='$DATASET' num-supernodes=$NUM_SUPERNODES num-server-rounds=$ROUNDS results-dir='$OUT'" \
    2>&1 | tee "logs/${TAG}.log"
done

echo
echo "=== Phase 1 verification ==="
python analyze.py "$OUT" --check
