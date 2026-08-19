"""Read run JSONs and produce the Week 1 checkpoint figure.

Usage:  python plot_baseline.py results/
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_runs(d: Path):
    runs = []
    for p in sorted(d.glob("run_*.json")):
        runs.append(json.loads(p.read_text()))
    if not runs:
        sys.exit(f"no run_*.json found in {d}")
    return runs


def main(d: Path):
    runs = load_runs(d)
    by_eps = defaultdict(list)
    for r in runs:
        by_eps[str(r["config"]["epsilon"])].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for eps, group in sorted(by_eps.items()):
        acc = np.array(
            [[rd.get("central_acc", np.nan) for rd in r["rounds"]] for r in group]
        )
        cum_j = np.array(
            [np.cumsum([rd["energy_j"] for rd in r["rounds"]]) for r in group]
        )
        rounds = np.arange(1, acc.shape[1] + 1)
        label = "no DP" if eps == "inf" else f"eps={eps}"

        m, s = np.nanmean(acc, 0), np.nanstd(acc, 0)
        ax1.plot(rounds, m, label=label)
        ax1.fill_between(rounds, m - s, m + s, alpha=0.2)

        ax2.plot(np.mean(cum_j, 0), m, label=label)

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Central test accuracy")
    ax1.set_title("Convergence")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Cumulative GPU energy (J)")
    ax2.set_ylabel("Central test accuracy")
    ax2.set_title("Accuracy per joule")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = d / "baseline.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)

    for eps, group in sorted(by_eps.items()):
        tot = np.mean([r["totals"]["energy_j"] for r in group])
        fin = np.mean([r["rounds"][-1].get("central_acc", np.nan) for r in group])
        print(f"eps={eps:>4}  n={len(group)}  {tot:8.0f} J  final acc {fin:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
