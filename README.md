# The Energy Cost of Privacy in Federated Learning — source code

Code and measurements for the paper *The Energy Cost of Privacy in Federated
Learning* (I. Mahaya, A. Behloul, K. Hamouid). Federated learning on Flower
(FedAvg), example-level local DP-SGD with Opacus, and GPU energy read from
the NVML hardware counter around each federated round.

## Layout

| file | role |
| --- | --- |
| `energyfl/task.py` | models (HAR 1D CNN, CIFAR-10 CNN), data partitioning, train/test |
| `energyfl/dp.py` | noise calibration over the whole run (Eq. 1) and DP-SGD |
| `energyfl/client_app.py` | one round of local training (private or not) |
| `energyfl/server_app.py` | FedAvg, NVML energy per round, idle baseline, one JSON per run |
| `prepare_har.py` | UCI-HAR archive → `data/har.npz` |
| `run.sh` | runs the main sweeps and the ablations |
| `analyze.py` | every table, figure and in-text number of the paper |
| `realised_epsilon.py` | "Worst ε" column of Table I |
| `results/` | the measurements behind the paper (one JSON per run) |

## Reproducing the paper's numbers from the committed measurements

Needs only `numpy` and `matplotlib`, runs in about 30 s:

```bash
pip install numpy matplotlib
python analyze.py
```

| paper item | printed section |
| --- | --- |
| Table I, per-round overhead (+51% HAR, +34% CIFAR-10), wall-clock / power split, noise floors, permutation test (p = 0.27, 0.79), idle baseline | `Per-round energy` |
| Table II, energy to target accuracy (and CIFAR-10 at 0.20) | `Net energy (J) to target accuracy` |
| Section IV-C, stopping rounds (CIFAR-10 ε = 0.5: round 8, 60–68% of energy after it) | `Energy-optimal stopping round` |
| Table III, carbon per 1000 runs | `gCO2eq per 1000 HAR runs` |
| Section IV-E, ablations | `Ablations` |
| Fig. 1, Fig. 2 | `figures/fig1_convergence.png`, `figures/fig2_energy_per_round.png` |

## Re-running the experiments

Needs an NVIDIA GPU whose driver exposes the total-energy counter (the paper
used an RTX 5080, PyTorch 2.10.0+cu128, CUDA 12.8).

```bash
pip install -e .

# UCI-HAR, downloaded by hand from
# https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
python prepare_har.py --root "/path/to/UCI HAR Dataset"

./run.sh main har            # 18 runs, 100 rounds   -> results/har/
./run.sh main cifar10        # 18 runs, 30 rounds    -> results/cifar10/
./run.sh ablation har        # epochs, clipping norm -> results/ablation_har/
./run.sh ablation cifar10    #                       -> results/ablation_cifar10/

python realised_epsilon.py   # -> results/realised_epsilon_har.json
python analyze.py
```

Move the committed `results/` aside first; a run whose JSON already exists is
skipped. Fresh runs give the same conclusions, though not identical joules:
GPU kernels are not forced to be deterministic, because that changes kernel
selection and therefore the energy measured.

## Setup (Section III)

- FedAvg, 50% of clients sampled for training and 50% for evaluation per round,
  1 local epoch, batch 32, SGD lr 0.01 momentum 0.9, seeds 0, 1, 2,
  ε ∈ {0.5, 1, 2, 4, 8, ∞}, δ = 1e-5, clipping norm C = 1.0.
- UCI-HAR: 21 clients, one per training participant; the 9 test participants
  form the central test set; raw inertial signals (9 × 128); 100 rounds.
- CIFAR-10: 20 clients, Dirichlet(α = 0.5) label partition, 30 rounds.
- Measurement controls: every σ is computed by the server *before* the first
  measured round (the accountant search is slower for a loose budget and would
  otherwise create a fake ε trend); the run order of the main sweep is
  randomised with a fixed seed; the idle baseline is probed after board power
  settles and subtracted over the measured wall-clock only.

## Notes on the committed data

- CIFAR-10 ε = 4 is averaged over two seeds. In the original run of seed 0,
  another job was using the GPU, so its idle baseline read about 300 W and
  that run was discarded. The re-run is kept in `results/excluded/` but not
  used: it was measured a month after the rest of the sweep (2026-10-05
  against 2026-09-01) and is not comparable, with 82 s of measured wall-clock
  against 67–68 s for the other two seeds.
- `results/ablation_cifar10/` also holds single-seed runs varying the
  Dirichlet α and the number of clients. The paper does not report them, and
  `analyze.py` ignores them except where their configuration equals the
  baseline cell.
