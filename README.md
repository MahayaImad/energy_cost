# The Energy Cost of Privacy in Federated Learning

Measures what client-level differential privacy costs in GPU joules when
added to federated learning, using NVML hardware counters differenced around
each round. Two datasets: UCI-HAR, where a client is one study participant,
and CIFAR-10 partitioned with a Dirichlet partitioner.

The paper is in `paper/` (build with `pdflatex main`).

## Reproduction log

Every command below was actually run, in this order, and the numbers beside
each one are the numbers the paper reports. Fill in the hardware line from
`--paper` output; everything else is as measured.

The run JSONs in this repository are the evidence. `analyze.py` reads them and
nothing else, so no figure in the paper comes from a log line.

### 0. Environment

```bash
pip install -e .                # flwr >= 1.33 is required, not preferred
python -c "import flwr; print(flwr.__version__)"
```

`flwr < 1.33` has no `--federation-config`, so the sweep silently falls back to
two CPU clients and NVML measures an idle card. `run_sweep.sh` refuses to start
in that case.

    GPU / torch / CUDA    <run `python analyze.py results_har/ --paper`>

### 1. Phase 0 — does the privacy machinery hold up?

```bash
python verify_dp.py             # CPU, seconds, no dataset, no Flower
```

    ALL CHECKS PASS

Three claims the paper makes about its own method, checked in code rather than
asserted in prose: sigma decreases in epsilon; the DP branch takes exactly the
same number of optimiser steps as the non-private one at every epsilon; and
the epsilon actually spent never exceeds the epsilon claimed, which is what
licenses the "reported epsilon is an upper bound" statement.

### 2. Prepare UCI-HAR

```bash
python prepare_har.py --root "/path/to/UCI HAR Dataset"
```

Writes `data/har.npz` from the raw inertial signals (9 channels x 128 samples),
not the 561 engineered features. Validates every shape against the published
counts — 7352 train and 2947 test windows over 30 subjects — and then reports
class coverage per client, which is the check that caught the original split
bug: all 21 clients must hold all six activities on both sides.

### 3. Phase 1 — the privacy sweep (primary result)

```bash
DATASET=har ROUNDS=100 ./run_sweep.sh      # -> results_har/
```

Six privacy budgets x three seeds = 18 runs, in randomised order, about 19
minutes end to end. The script derives every sigma first (`prewarm_sigma.py`),
then runs, then runs the gate. The gate output on the committed data:

| check | result |
| --- | --- |
| [0] completeness | 18/18 runs, all six conditions at three seeds |
| [1] wall-clock flatness | 47.8–48.3 s across the private conditions against 37.2 s without privacy; **1.2 % spread**, ordering collapsed into noise |
| [2] compute parity | 89 optimiser steps per round in every condition, 0 steps of disagreement with the accountant |
| [3] GPU residency | every client on the GPU |
| [3b] idle baselines | median **18.21 W**, all runs within tolerance |
| [4] noise floor | **±5.43 W (9.46 %)** — nothing smaller is reported as an effect |
| [5] run-order drift | rho = +0.105 against execution order, −0.142 against epsilon, −0.196 between the two |

`VERDICT: PASS`. Check [5] is the one that matters most: it is what separates
a real epsilon effect from the sweep simply drifting, and the near-zero
confound (−0.196) is what randomising the run order buys.

### 4. Analysis — the published numbers

```bash
python analyze.py results_har/                        # tables + figures
python analyze.py results_har/ --targets 0.40,0.50,0.70,0.85
python analyze.py results_har/ --paper                # config and measurement
```

**Per-round energy is flat in epsilon** (Table I, Fig. 2). Net of idle,
private training costs **21.0 J per round against 14.5 J**, an overhead of
**+45 %**, varying by 8.2 % across the five private conditions — inside the
9.46 % floor. So the per-round cost of privacy is a fixed implementation
overhead, and everything budget-dependent acts through the round count.

**Energy-to-target is not flat at all** (Table II). At 0.40 accuracy every
condition costs 178–218 J: privacy is free. At 0.50 the private conditions
cost 1.17–2.10x the non-private 204 J. At 0.70 the survivors cost **5.8x**
(1753 J at epsilon = 4 against 300 J) and epsilon <= 2 never arrive. At 0.85,
which the non-private model reaches in 57.7 rounds and 878 J, **no budget
reaches it at any expenditure**.

**An energy-optimal stopping round** (Fig. 1). At epsilon = 0.5 accuracy peaks
at 0.4712 at round 17 and ends at 0.4553; the following 83 rounds cost 2477 J,
**83 % of the run**, and buy negative accuracy.

**Carbon by grid** (Table III, Fig. 3). Scaled to 1000 runs, the cost
attributable to privacy is **105–133 gCO2eq on Algeria's grid against
5.6–7.1 g in Norway — a factor of 18.8** for identical computation under an
identical guarantee.

### 5. Phase 2 — ablations

```bash
SEEDS="0 1 2" DATASET=har ROUNDS=100 ./run_ablations.sh   # -> results_ablation_har/
python analyze.py results_ablation_har/ --ablation
```

One factor at a time at epsilon in {1, inf}. On HAR only local epochs and the
clipping norm are meaningful: a client is a participant, so the federation size
is fixed and there is no Dirichlet parameter to vary.

- **The model becomes noise-limited.** At epsilon = 1 final accuracy spans
  2.6 % across one, two and five local epochs and 2.3 % across C in
  {0.5, 1, 2} — both inside the 9.46 % floor. Tuning is not a route to a
  better private model.
- **The standard efficiency lever stops paying.** Without privacy, energy to
  reach 0.70 falls from 304 J at one local epoch to 192 J at five, a 1.6x
  saving. Under epsilon = 1 the saving vanishes while energy per round still
  rises from 29 J to 64 J: local steps enter the budget, so extra local work
  forces a larger sigma.
- **Clipping is free in compute** — 29 J per round at all three values — so it
  acts purely on convergence.

### 6. CIFAR-10 as a secondary check

```bash
./run_sweep.sh                       # -> results/          MISSING, see below
./run_ablations.sh                   # -> results_ablation/
python analyze.py results_ablation/ --ablation
```

CIFAR-10 reproduces the per-round result (+60 % net of idle, flat to within
9.6 %) and shows the stopping effect across more budgets, with peak rounds
ordering 7, 11, 19, 24. Reaching 0.20 accuracy costs 1482 J net at epsilon = 1
against 406 J without privacy, 3.65x. The clipping ablation inverts here:
C = 2 peaks at 0.191 at round 8 and ends at 0.102, giving back 46.5 % of its
peak against 3.1 % at C = 0.5.

> **The `results/` directory is empty.** The CIFAR-10 main sweep behind the
> `+60 %`, the `3.65x` and the peak-round ordering was lost before the data
> was committed; only the CIFAR ablations survive. Re-running `./run_sweep.sh`
> reproduces it in about 35 minutes, but the values will shift by a few
> percent and Section IV-D would need updating. Until then those four figures
> are reported without raw data behind them.

### 7. Figures for the paper

```bash
python analyze.py results_har/       # writes the PNGs into results_har/
cp results_har/fig1_convergence.png results_har/fig3_energy_per_round.png \
   results_har/fig4_carbon.png paper/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## The data in this repository

| directory | runs | what |
| --- | --- | --- |
| `results_har/` | 18 | the primary sweep: 6 budgets x 3 seeds, 100 rounds |
| `results_ablation_har/` | 27 | HAR ablations on local epochs and clipping norm |
| `results_ablation/` | 27 | CIFAR-10 ablations, 30 rounds |
| `results/` | 0 | the CIFAR-10 main sweep, lost (see step 6) |

All committed runs post-date the per-class HAR split fix: mean `eval_acc`
0.699 against `central_acc` 0.670, which is the right way round. Earlier
30-round and 150-round HAR directories were trained under the broken split —
clients holding five of six activities and validating on two — and were
deleted rather than archived, because they read as perfectly valid to the
analysis.

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
