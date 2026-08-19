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
# Must match num-supernodes in [tool.flwr.app.config]: one is what the
# federation spawns, the other is what the app partitions the data for.
NUM_SUPERNODES=20
FEDCFG="num_supernodes=$NUM_SUPERNODES client_resources_num_gpus=$GPU_SHARE init_args_num_gpus=1"

for EPS in "${EPSILONS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="eps${EPS}-s${SEED}"
    if compgen -G "results/run_eps${EPS//./p}_seed${SEED}_${TAG}.json" > /dev/null; then
      echo "=== $TAG (already present, skipping) ==="
      continue
    fi
    echo "=== $TAG ==="
    flwr run . --stream \
      --federation-config "$FEDCFG" \
      --run-config "seed=$SEED epsilon='$EPS' run-id='$TAG'" \
      2>&1 | tee "logs/${TAG}.log"
  done
done

echo
echo "=== Phase 1 verification ==="
python analyze.py results/ --check
