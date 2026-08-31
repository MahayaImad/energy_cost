# The Energy Cost of Privacy in Federated Learning

Measures what client-level differential privacy costs in GPU joules when
added to federated learning, using NVML hardware counters differenced around
each round. Two datasets: UCI-HAR, where a client is one study participant,
and CIFAR-10 partitioned with a Dirichlet partitioner.

The paper is in `paper/` (build with `pdflatex main`).

## Running it

Everything below writes into a results directory of `run_*.json` files. Those
JSONs are the unit of analysis: `analyze.py` reads them and nothing else, so
no number in the paper can come from a log line.

```bash
pip install -e .                       # needs flwr >= 1.33, see pyproject.toml

python verify_dp.py                    # Phase 0: does the DP machinery hold up?
                                       # CPU only, seconds, no dataset needed

python prepare_har.py --root "/path/to/UCI HAR Dataset"    # HAR only, once

DATASET=har ROUNDS=100 ./run_sweep.sh  # Phase 1: 6 epsilons x 3 seeds
                                       # -> results_har/, then runs the gate

SEEDS="0 1 2" DATASET=har ROUNDS=100 ./run_ablations.sh    # Phase 2
                                       # -> results_ablation_har/

python analyze.py results_har/         # checks, tables and the three figures
python analyze.py results_har/ --paper # the numbers the write-up quotes
```

For CIFAR-10, drop `DATASET`/`ROUNDS` and the results land in `results/` and
`results_ablation/`.

## The gate

`analyze.py --check` has to pass before any energy number is quoted. It exists
because the energy axis only means something if every epsilon condition ran
the *same workload*: it checks that wall-clock is flat across the private
conditions, that executed steps match the steps the accountant was calibrated
against, that every client trained on the GPU NVML is watching, that the idle
baselines agree, and that wall-clock is not simply tracking execution order.
It also reports the measurement noise floor from the across-seed spread; no
effect smaller than that is claimed anywhere.

## Layout

| file | what it does |
| --- | --- |
| `energyfl/dp.py` | noise calibration and DP-SGD training |
| `energyfl/task.py` | models, partitioning, train/eval loops |
| `energyfl/client_app.py` | one round of local training |
| `energyfl/server_app.py` | FedAvg, energy bracketing, JSON logging |
| `energyfl/energy.py` | NVML counters and the idle baseline probe |
| `prepare_har.py` | UCI-HAR archive -> a validated `.npz` |
| `prewarm_sigma.py` | derives every sigma *before* measurement starts |
| `verify_dp.py` | Phase-0 checks on the privacy accounting |
| `analyze.py` | the gate, the tables, the figures |

## One thing worth knowing

`prewarm_sigma.py` is not an optimisation. Deriving a noise multiplier
binary-searches the RDP accountant, and that search costs far more for a loose
budget than a tight one (0.086 s at epsilon=0.5 against 0.693 s at
epsilon=8). Run inside the federated loop, where the obvious implementation
puts it, that cost lands inside the measurement window and scales with
epsilon — taking exactly the shape of a cost of privacy. In one of our sweeps
it accounted for the entire observed trend. Deriving every sigma up front is
what keeps it out.
