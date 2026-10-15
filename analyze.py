"""Regenerate every table, figure and in-text number of the paper from the
run JSONs in results/.

    python analyze.py                 # prints everything, writes figures/

All energies are net of the idle baseline unless labelled gross.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

GRIDS = {"Norway": 26, "France": 56, "Germany": 344, "USA": 369,
         "Algeria": 488, "India": 631}          # gCO2eq/kWh (Ember, 2024)
CARBON_RUNS = 1000


# ----------------------------------------------------------------- loading


def eps_key(e):
    return math.inf if str(e) == "inf" else float(e)


def load(directory):
    """{epsilon: [runs]}, epsilon sorted with the non-private baseline last."""
    groups = defaultdict(list)
    for p in sorted(Path(directory).glob("run_*.json")):
        r = json.loads(p.read_text())
        groups[str(r["config"]["epsilon"])].append(r)
    return dict(sorted(groups.items(), key=lambda kv: eps_key(kv[0])))


def gross(r):
    return r["totals"]["energy_j"]


def wall(r):
    return r["totals"]["wall_s_measured"]


def net(r):
    return max(0.0, gross(r) - r["totals"]["idle_power_w"] * wall(r))


def net_per_round(r):
    return net(r) / len(r["rounds"])


def acc_curve(r):
    return np.array([rd["central_acc"] for rd in r["rounds"]])


def final_acc(r):
    return r["rounds"][-1]["central_acc"]


def mean(f, group):
    return float(np.mean([f(r) for r in group]))


def worst_rel_sd(f, groups):
    """Noise floor: worst within-condition across-seed SD, % of the mean."""
    return max(100 * np.std(v, ddof=1) / np.mean(v)
               for v in ([f(r) for r in g] for g in groups) if len(v) > 1)


def accuracy_floor(groups):
    return max(np.std([final_acc(r) for r in g], ddof=1)
               for g in groups if len(g) > 1)


def permutation_f_test(groups, draws=200000, seed=0):
    """(F, p) of a one-way F statistic, p from permuting the seed labels."""
    groups = [np.asarray(g, float) for g in groups]
    sizes = np.cumsum([len(g) for g in groups])[:-1]

    def f_stat(gs):
        allv = np.concatenate(gs)
        k, n = len(gs), len(allv)
        between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
        within = sum(((g - g.mean()) ** 2).sum() for g in gs)
        return (between / (k - 1)) / (within / (n - k))

    observed = f_stat(groups)
    pool, rng = np.concatenate(groups), np.random.default_rng(seed)
    hits = sum(f_stat(np.split(rng.permutation(pool), sizes)) >= observed - 1e-12
               for _ in range(draws))
    return observed, (hits + 1) / (draws + 1)


def energy_to_target(group, target):
    """(mean net J, mean round, seeds that reached it) to first reach target."""
    joules, rounds = [], []
    for r in group:
        cum = 0.0
        for rd in r["rounds"]:
            cum += rd["energy_j"]
            if rd["central_acc"] >= target:
                joules.append(cum * net(r) / gross(r))   # gross -> net
                rounds.append(rd["round"])
                break
    if not joules:
        return None, None, 0
    return np.mean(joules), np.mean(rounds), len(joules)


def peak(group):
    """(peak round, peak accuracy, final accuracy) of the mean curve."""
    m = np.mean([acc_curve(r) for r in group], axis=0)
    i = int(np.argmax(m))
    return i + 1, m[i], m[-1]


# ---------------------------------------------------------------- sections


def per_round_energy(name, by_eps, realised=None):
    print(f"\n== Per-round energy, {name} (Table I) ==")
    base = by_eps["inf"]
    private = {e: g for e, g in by_eps.items() if e != "inf"}
    b = mean(net_per_round, base)
    print(f"{'eps':>5} {'Gross W':>8} {'Net W':>7} {'Net J/rd':>9} "
          f"{'Overhead':>9} {'Worst eps':>10} {'sigma range':>14}")
    for e, g in by_eps.items():
        jr = mean(net_per_round, g)
        row = (f"{e:>5} {mean(lambda r: gross(r) / wall(r), g):8.2f} "
               f"{mean(lambda r: net(r) / wall(r), g):7.2f} {jr:9.1f}")
        if e != "inf":
            lo = min(rd["sigma_min"] for r in g for rd in r["rounds"])
            hi = max(rd["sigma_max"] for r in g for rd in r["rounds"])
            worst = f"{realised[e]:.3f}" if realised else "--"
            row += f" {100 * (jr - b) / b:+8.1f}% {worst:>10} {lo:6.2f}-{hi:.2f}"
        print(row)

    jr = [mean(net_per_round, g) for g in private.values()]
    over = [100 * (j - b) / b for j in jr]
    wall_ratio = (np.mean([mean(lambda r: wall(r) / len(r["rounds"]), g)
                           for g in private.values()])
                  / mean(lambda r: wall(r) / len(r["rounds"]), base))
    power_ratio = (np.mean([mean(lambda r: net(r) / wall(r), g)
                            for g in private.values()])
                   / mean(lambda r: net(r) / wall(r), base))
    F, p = permutation_f_test([[net_per_round(r) for r in g]
                               for g in private.values()])
    groups = list(by_eps.values())
    print(f"private {np.mean(jr):.1f} J/round vs {b:.1f} without privacy: "
          f"overhead {np.mean(over):+.0f}% (range {min(over):+.1f}% to "
          f"{max(over):+.1f}%, spread {100 * np.ptp(jr) / np.mean(jr):.1f}% "
          f"of the mean)")
    print(f"wall-clock {100 * (wall_ratio - 1):+.1f}%, "
          f"net power {100 * (power_ratio - 1):+.1f}%")
    print(f"noise floors: energy/round {worst_rel_sd(net_per_round, groups):.1f}%, "
          f"board power {worst_rel_sd(lambda r: gross(r) / wall(r), groups):.1f}%, "
          f"final accuracy +/-{accuracy_floor(groups):.3f}")
    print(f"permutation test, dependence of energy/round on eps: "
          f"F = {F:.2f}, p = {p:.2f}")
    idle = [r["totals"]["idle_power_w"] for g in groups for r in g]
    idle_sd = [r["totals"]["idle_power_w_sd"] for g in groups for r in g]
    print(f"idle baseline {np.mean(idle):.1f} W (sd {np.mean(idle_sd):.2f} W)")


def target_table(name, by_eps, targets):
    print(f"\n== Net energy (J) to target accuracy, {name} (Table II) ==")
    print("mean rounds in parentheses; '-' unreached; '*' not all seeds")
    print(f"{'eps':>5}" + "".join(f"{t:>15.2f}" for t in targets))
    for e, g in by_eps.items():
        cells = []
        for t in targets:
            j, rd, n = energy_to_target(g, t)
            cells.append("-" if n == 0 else
                         f"{j:.0f}{'*' if n < len(g) else ''} ({rd:.1f})")
        print(f"{e:>5}" + "".join(f"{c:>15}" for c in cells))


def stopping_rounds(name, by_eps):
    """A stopping round is reported only when the mean curve's peak-to-final
    drop exceeds the across-seed SD of final accuracy and the mean's peak
    lies within the per-seed peak rounds."""
    print(f"\n== Energy-optimal stopping round, {name} (Section IV-C) ==")
    for e, g in by_eps.items():
        pr, pk, fin = peak(g)
        sd = np.std([final_acc(r) for r in g], ddof=1)
        seeds = []
        for r in sorted(g, key=lambda r: r["config"]["seed"]):
            a = acc_curve(r)
            i = int(np.argmax(a))
            e_round = [rd["energy_j"] for rd in r["rounds"]]
            seeds.append(f"rd {i + 1} drop {a[i] - a[-1]:.3f} "
                         f"{100 * sum(e_round[i + 1:]) / sum(e_round):.0f}% after")
        peaks = [int(np.argmax(acc_curve(r))) + 1 for r in g]
        ok = pk - fin > sd and min(peaks) <= pr <= max(peaks)
        print(f"eps {e:>4}: mean peak {pk:.3f} at rd {pr}, final {fin:.3f}, "
              f"drop {pk - fin:.3f} vs +/-{sd:.3f} -> "
              f"{'RESOLVED' if ok else 'not resolved'}")
        print("          per seed: " + "; ".join(seeds))


def final_accuracy(name, by_eps):
    print(f"\n== Final accuracy, {name} ==")
    for e, g in by_eps.items():
        f = [final_acc(r) for r in g]
        print(f"eps {e:>4}: mean {np.mean(f):.3f} (seeds {min(f):.3f}-{max(f):.3f})")


def carbon_table(by_eps):
    print(f"\n== gCO2eq per {CARBON_RUNS} HAR runs (Table III) ==")
    kwh = lambda g: mean(gross, g) / 3.6e6 * CARBON_RUNS
    base = kwh(by_eps["inf"])
    dp = [kwh(g) for e, g in by_eps.items() if e != "inf"]
    print(f"{'Grid':<8} {'DP':>5} {'No DP':>6} {'Privacy':>12}")
    for grid, c in GRIDS.items():
        lo, hi = (min(dp) - base) * c, (max(dp) - base) * c
        print(f"{grid:<8} {np.mean(dp) * c:5.0f} {base * c:6.0f} "
              f"{lo:6.1f}-{hi:.1f}")


def ablations(name, directory, target):
    """One factor at a time at eps in {1, inf}; the others at epochs=1, C=1."""
    runs = [r for g in load(directory).values() for r in g]
    # The CIFAR-10 directory also holds single-seed alpha / client-count runs
    # that the paper does not report; keep the baseline partition only.
    runs = [r for r in runs if r["config"]["dirichlet_alpha"] == 0.5
            and r["config"]["num_supernodes"] in (20, 21)]
    cells = defaultdict(list)
    for r in runs:
        c = r["config"]
        cells[(c["local_epochs"], c["clipping_norm"], str(c["epsilon"]))].append(r)

    print(f"\n== Ablations, {name} (Section IV-E) ==")
    print(f"floors within a cell: board power "
          f"{worst_rel_sd(lambda r: gross(r) / wall(r), cells.values()):.1f}%, "
          f"final accuracy +/-{accuracy_floor(cells.values()):.3f}")
    print(f"{'factor':>10} {'eps':>4} {'n':>2} {'J/rd':>5} {'peak rd':>8} "
          f"{'peak':>6} {'final':>6} {'lost':>6} {'J after peak':>13} "
          f"{f'J to {target}':>9}")
    axes = (("epochs", 0, lambda k: k[1] == 1.0),    # vary epochs, C = 1
            ("C", 1, lambda k: k[0] == 1))           # vary C, epochs = 1
    for label, field, others_at_base in axes:
        for key in sorted(filter(others_at_base, cells)):
            g = cells[key]
            jr = mean(net_per_round, g)
            pr, pk, fin = peak(g)
            j, _, n = energy_to_target(g, target)
            print(f"{label + '=' + str(key[field]):>10} {key[2]:>4} {len(g):>2} {jr:5.0f} "
                  f"{pr:>8} {pk:6.3f} {fin:6.3f} {100 * (pk - fin) / pk:5.1f}% "
                  f"{jr * (len(g[0]['rounds']) - pr):13.0f} "
                  f"{'-' if n == 0 else f'{j:.0f}':>9}")


def figures(by_eps, out="figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).mkdir(exist_ok=True)
    label = lambda e: "no DP" if e == "inf" else f"$\\epsilon$={e}"

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for e, g in by_eps.items():
        acc = np.array([acc_curve(r) for r in g])
        m, sd = acc.mean(0), acc.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, label=label(e))
        ax.fill_between(x, m - sd, m + sd, alpha=0.18)
    ax.set(xlabel="Communication round", ylabel="Central test accuracy",
           title="Convergence under a fixed privacy budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_convergence.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    vals = [[net_per_round(r) for r in g] for g in by_eps.values()]
    ax.bar(range(len(vals)), [np.mean(v) for v in vals],
           yerr=[np.std(v) for v in vals], capsize=3,
           color=["tab:gray" if e == "inf" else "tab:blue" for e in by_eps])
    ax.set_xticks(range(len(vals)), [label(e) for e in by_eps], fontsize=7)
    ax.set(ylabel="Energy per round (J, net of idle)",
           title="Per-round energy: fixed DP overhead, flat in $\\epsilon$")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_energy_per_round.png", dpi=200)
    print(f"\nwrote {out}/fig1_convergence.png, {out}/fig2_energy_per_round.png")


def main():
    har, cifar = load("results/har"), load("results/cifar10")
    realised = Path("results/realised_epsilon_har.json")
    realised = json.loads(realised.read_text()) if realised.exists() else None

    print("######## UCI-HAR (primary) ########")
    per_round_energy("UCI-HAR", har, realised)
    target_table("UCI-HAR", har, (0.40, 0.50, 0.70, 0.85))
    stopping_rounds("UCI-HAR", har)
    carbon_table(har)
    ablations("UCI-HAR", "results/ablation_har", 0.70)
    figures(har)

    print("\n######## CIFAR-10 (secondary) ########")
    per_round_energy("CIFAR-10", cifar)
    target_table("CIFAR-10", cifar, (0.20,))
    stopping_rounds("CIFAR-10", cifar)
    final_accuracy("CIFAR-10", cifar)
    ablations("CIFAR-10", "results/ablation_cifar10", 0.20)


if __name__ == "__main__":
    main()
