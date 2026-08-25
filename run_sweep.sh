#!/usr/bin/env bash
# Phase 1: full privacy sweep. 6 epsilon values x 3 seeds = 18 runs.
#
# Simulation resources come from --federation-config (flwr >= 1.33). The
# defaults if you omit it are num_supernodes=2 and client_resources_num_gpus=0
# -- i.e. two clients on the CPU. The run completes, NVML measures an idle
# card, and every joule recorded is meaningless. Both keys below are load-
# bearing:
#   init_args_num_gpus=1          Ray sees a GPU at all
#   client_resources_num_gpus     each ClientApp reserves a share of it
# Note the keys are flat and underscored here, unlike the hyphenated TOML
# form used in a Flower config file. The check at the bottom of this script
# exists to catch a CPU-only run before it wastes an hour.
set -euo pipefail
mkdir -p results logs

# Preflight: a venv rebuild silently resolving flwr down to 1.31 has cost
# this project two sessions. 1.31 has no --federation-config, so the sweep
# would fall back to 2 CPU clients or fail outright.
python - <<'EOF' || exit 1
import sys, flwr
v = tuple(int(x) for x in flwr.__version__.split(".")[:2])
if v < (1, 33):
    sys.exit(
        f"flwr {flwr.__version__} is too old; >=1.33 required.\n"
        "  pip install --upgrade 'flwr[simulation]>=1.33.0'"
    )
print(f"flwr {flwr.__version__} ok")
EOF

EPSILONS=("inf" 8 4 2 1 0.5)
SEEDS=(0 1 2)
GPU_SHARE="${GPU_SHARE:-0.25}"

# DATASET=har ./run_sweep.sh runs the HAR sweep into results_har/.
DATASET="${DATASET:-cifar10}"
HAR_SPLIT="${HAR_SPLIT:-official}"
# Must match num-supernodes in [tool.flwr.app.config]: one is what the
# federation spawns, the other is what the app partitions the data for.
# On HAR a client IS a study participant, so the count is fixed by the split:
# 21 for the published subject-disjoint split, 30 if all participants are used.
if [ "$DATASET" = "har" ]; then
  [ "$HAR_SPLIT" = "official" ] && NUM_SUPERNODES=21 || NUM_SUPERNODES=30
  OUT="${OUT:-results_har}"
else
  NUM_SUPERNODES=20
  OUT="${OUT:-results}"
fi
mkdir -p "$OUT"
FEDCFG="num_supernodes=$NUM_SUPERNODES client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1"

# Randomised run order. Sweeping epsilon in a fixed descending order makes
# execution order and epsilon almost perfectly rank-correlated, so any drift
# over the sweep (caches warming, clocks settling, anything that changes
# slowly) is indistinguishable from a cost of privacy -- and the first sweep
# showed exactly that signature: wall-clock falling monotonically over the
# run while step counts stayed flat. Shuffling decorrelates the two so
# analyze.py check [5] can actually attribute the trend. ORDER_SEED keeps it
# reproducible; set SHUFFLE=0 to restore the old grouped order.
SHUFFLE="${SHUFFLE:-1}"
ORDER_SEED="${ORDER_SEED:-1234}"
mapfile -t PAIRS < <(python - "$SHUFFLE" "$ORDER_SEED" "${EPSILONS[*]}" "${SEEDS[*]}" <<'EOF'
import random, sys
shuffle, seed, eps, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
pairs = [f"{e} {s}" for e in eps for s in seeds]
if shuffle == "1":
    random.Random(seed).shuffle(pairs)
print("\n".join(pairs))
EOF
)
echo "run order (${#PAIRS[@]} runs, shuffle=$SHUFFLE seed=$ORDER_SEED):"
printf '  %s\n' "${PAIRS[@]}"
echo

# Derive every sigma before any measurement starts, so the accountant's
# binary search never runs inside a measured round. Cheap, and idempotent.
if [ "$DATASET" = "har" ]; then
  python prewarm_sigma.py --dataset har --har-split "$HAR_SPLIT"
else
  python prewarm_sigma.py
fi

for PAIR in "${PAIRS[@]}"; do
  read -r EPS SEED <<< "$PAIR"
  {
    TAG="eps${EPS}-s${SEED}"
    if compgen -G "$OUT/run_eps${EPS//./p}_seed${SEED}_${TAG}.json" > /dev/null; then
      echo "=== $TAG (already present, skipping) ==="
      continue
    fi
    echo "=== $TAG ==="
    flwr run . --stream \
      --federation-config "$FEDCFG" \
      --run-config "seed=$SEED epsilon='$EPS' run-id='$TAG' dataset='$DATASET' har-split='$HAR_SPLIT' num-supernodes=$NUM_SUPERNODES results-dir='$OUT'" \
      2>&1 | tee "logs/${TAG}.log"
  }
done

echo
echo "=== Phase 1 verification ==="
python analyze.py "$OUT" --check
