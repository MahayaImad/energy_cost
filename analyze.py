"""Analysis for 'The Energy Cost of Privacy in Federated Learning'.

Reads results/run_*.json and nothing else. The JSONs are the unit of
analysis, so no number in the paper comes from a log line.

    python analyze.py results_har/                    checks, tables, figures
    python analyze.py results_har/ --check            the gate only, exit code
    python analyze.py results_ablation_har/ --ablation
    python analyze.py results_har/ --paper            numbers for the write-up
    python analyze.py results_har/ --targets 0.4,0.5,0.7

THE GATE (--check)
The energy axis only means something if every epsilon condition ran the same
workload. Before any energy number is quoted: wall-clock must be flat across
the DP conditions, executed steps must match the steps the accountant was
calibrated against, and every client must have trained on the GPU that NVML
is watching. Effects smaller than the measured noise floor are not reported
as effects; the floor is computed from the across-seed spread, not assumed.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Grid carbon intensity, gCO2eq/kWh. Algeria's grid is overwhelmingly natural
# gas, which is the regional hook: the same joules bought with the same
# privacy budget emit very differently depending on where they burn.
GRID_GCO2_PER_KWH = {
    "Algeria": 488, "France": 56, "Germany": 344,
    "USA (avg)": 369, "Norway": 26, "India": 631,
}

# Carbon is reported per this many runs. One run emits well under a gram,
# which is too small to reason about. This is the SAME measured workload
# repeated, not an extrapolation to real edge hardware.
CARBON_RUNS = 1000

# What counts as a usable model differs by task. A target nothing reaches is
# reported as unreached, which is itself a result -- but a whole column of
# them tells the reader nothing. Override with --targets.
TARGET_ACCS_BY_DATASET = {
    "cifar10": (0.15, 0.20, 0.25),
    "har": (0.50, 0.70, 0.85),
}

ABLATION_FACTORS = {
    "dirichlet_alpha": "alpha",
    "num_supernodes": "clients",
    "local_epochs": "epochs",
    "clipping_norm": "C",
}


# --------------------------------------------------------------------- loading


def eps_key(e):
    """Sort key for an epsilon label. The non-private baseline sorts last."""
    s = str(e).strip().lower()
    return math.inf if s in ("inf", "infinity", "none", "") else float(s)


def eps_label(e):
    return "no DP" if math.isinf(eps_key(e)) else f"$\\epsilon$={e}"


# Fields written by the current server_app. A run missing any of them predates
# a fix that changed what the numbers mean -- idle subtracted over the wrong
# window, or step counts never recorded -- so it is rejected rather than
# silently averaged in with runs that are not comparable to it.
REQUIRED_TOTALS = ("energy_j", "wall_s_measured", "energy_j_net_idle", "idle_power_w")
REQUIRED_ROUND = ("energy_j", "wall_s", "steps_sum", "expected_steps_sum")


def load(d: Path):
    """Load every run in a directory, newest schema only."""
    paths = sorted(d.glob("run_*.json"))
    if not paths:
        sys.exit(f"no run_*.json found in {d}")

    runs = []
    for p in paths:
        r = json.loads(p.read_text())
        missing = [k for k in REQUIRED_TOTALS if k not in r.get("totals", {})]
        missing += [k for k in REQUIRED_ROUND if k not in (r["rounds"] or [{}])[0]]
        if missing:
            sys.exit(
                f"{p.name} is missing {missing}.\n"
                "It was written by an older server_app whose numbers are not "
                "comparable with the current ones. Move it aside and re-run."
            )
        runs.append(r)

    datasets = {str(r["config"]["dataset"]).lower() for r in runs}
    if len(datasets) > 1:
        sys.exit(
            f"{d} mixes datasets {sorted(datasets)}; energy and accuracy are "
            "not comparable across them. Keep one dataset per directory."
        )

    by_eps = defaultdict(list)
    for r in runs:
        by_eps[str(r["config"]["epsilon"])].append(r)
    by_eps = dict(sorted(by_eps.items(), key=lambda kv: eps_key(kv[0])))
    return runs, by_eps, datasets.pop()


def rounds_matrix(group, field):
    """(n_seeds, n_rounds) array of a per-round field, NaN where absent."""
    n = min(len(r["rounds"]) for r in group)
    return np.array(
        [[rd.get(field, np.nan) for rd in r["rounds"][:n]] for r in group], float
    )


def measured_wall(r):
    """Wall-clock the energy counter actually covers (not the whole process)."""
    return r["totals"]["wall_s_measured"]


def total_energy(r, net=False):
    """Gross, or idle-subtracted, round-bracketed energy in joules."""
    t = r["totals"]
    if not net:
        return t["energy_j"]
    return max(0.0, t["energy_j"] - t["idle_power_w"] * measured_wall(r))


def energy_to_target(group, target, net=False):
    """Mean joules to first reach `target` accuracy. Returns (J, round, n).

    Seeds that never reach the target are excluded and counted: a target no
    configuration reaches is a result, and must be shown as unreached rather
    than quietly dropped.
    """
    joules, at_round = [], []
    for r in group:
        cumulative = 0.0
        for rd in r["rounds"]:
            cumulative += rd["energy_j"]
            if rd.get("central_acc", float("nan")) >= target:
                # Scale gross -> net by the run's own ratio, so this stays
                # consistent with totals.energy_j_net_idle.
                gross = total_energy(r)
                ratio = total_energy(r, True) / gross if net and gross > 0 else 1.0
                joules.append(cumulative * ratio)
                at_round.append(rd["round"])
                break
    if not joules:
        return None, None, 0
    return float(np.mean(joules)), float(np.mean(at_round)), len(joules)


def peak_round(group):
    """(round of peak mean accuracy, that peak, final-round accuracy).

    Under a tight budget the curve peaks and then declines: past that round,
    training spends joules AND loses accuracy.
    """
    acc = rounds_matrix(group, "central_acc")
    mean = np.nanmean(acc, 0)
    if np.isnan(mean).all():
        return None, None, None
    i = int(np.nanargmax(mean))
    return i + 1, float(mean[i]), float(mean[-1])


def noise_floor(by_eps):
    """(watts, percent) of the worst within-condition across-seed spread.

    Nothing smaller than this is claimed as an effect anywhere in the paper.
    """
    worst_w, worst_pct = 0.0, 0.0
    for group in by_eps.values():
        if len(group) < 2:
            continue
        power = np.array([total_energy(r) / measured_wall(r) for r in group])
        worst_w = max(worst_w, float(power.std(ddof=1)))
        worst_pct = max(worst_pct, float(100 * power.std(ddof=1) / power.mean()))
    return worst_w, worst_pct


def spearman(xs, ys):
    """Rank correlation, ties averaged. 0.0 when undefined. (No scipy.)"""
    if len(xs) < 3:
        return 0.0

    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


# ---------------------------------------------------------------------- checks


def check_drift(runs) -> bool:
    """Is wall-clock tracking epsilon, or just tracking execution order?

    Nothing in DP-SGD makes runtime depend on the magnitude of sigma: noise is
    drawn from the same shaped tensor either way. So a wall-clock trend in
    epsilon with flat step counts is more likely to be drift over the sweep,
    and has to be ruled out before any of it is called a cost of privacy.
    run_sweep.sh randomises run order so the two can be told apart at all.
    """
    dated = sorted((r for r in runs if r.get("timestamp_utc")),
                   key=lambda r: r["timestamp_utc"])
    if len(dated) < 4:
        print("\n[5] Run-order drift: too few timestamped runs to assess")
        return True

    print("\n[5] Run-order drift (runs in execution order)")
    print(f"    {'#':>3} {'eps':>6} {'seed':>5} {'wall_s':>9} {'gross W':>9}  timestamp")
    walls, order, eps_vals = [], [], []
    for i, r in enumerate(dated):
        w = measured_wall(r)
        e = eps_key(r["config"]["epsilon"])
        walls.append(w)
        order.append(float(i))
        eps_vals.append(1e9 if math.isinf(e) else e)   # inf must sort last
        print(f"    {i:>3} {str(r['config']['epsilon']):>6} {r['config']['seed']:>5} "
              f"{w:>9.1f} {total_energy(r) / w:>9.2f}  {r['timestamp_utc'][:19]}")

    # Restrict to DP runs, where epsilon actually varies and the non-private
    # baseline cannot drive the correlation on its own.
    dp = [(o, e, w) for o, e, w in zip(order, eps_vals, walls) if e < 1e9]
    o_dp, e_dp, w_dp = ([d[i] for d in dp] for i in (0, 1, 2))
    rho_order, rho_eps = spearman(o_dp, w_dp), spearman(e_dp, w_dp)
    confound = spearman(o_dp, e_dp)

    print(f"\n    wall_s vs execution order : rho = {rho_order:+.3f}")
    print(f"    wall_s vs epsilon         : rho = {rho_eps:+.3f}")
    print(f"    execution order vs epsilon: rho = {confound:+.3f}   <- design check")

    if abs(confound) > 0.8:
        print("\n    FAIL: epsilon and execution order are confounded by design.")
        print("          The sweep visited epsilon in a fixed order, so no")
        print("          correlation computed on this data can separate them.")
        print("          Re-run with run_sweep.sh, which randomises the order.")
        return False
    if abs(rho_order) > 0.6 and abs(rho_order) >= abs(rho_eps):
        print("\n    FAIL: under randomised order, wall-clock still tracks")
        print("          execution order more than epsilon. The drift is real")
        print("          and is not a privacy effect; find its cause first.")
        return False
    if abs(rho_eps) > 0.6:
        print("\n    Wall-clock tracks epsilon under randomised order. That may")
        print("    be real -- but with step counts flat (check [2]) the")
        print("    mechanism is not obvious and needs explaining before it")
        print("    goes in the paper. The accountant artefact looked like this.")
    else:
        print("\n    PASS: no strong drift in either direction.")
    return True


def check(by_eps, runs) -> bool:
    ok = True
    print("=" * 68)
    print("PHASE-1 VERIFICATION")
    print("=" * 68)

    # [0] Completeness. Every later number is a mean over seeds, and a
    # condition short a seed weakens exactly the comparison the paper rests
    # on -- while looking identical to a finished sweep in every other check.
    counts = {eps: len(g) for eps, g in by_eps.items()}
    expected = max(counts.values())
    print(f"\n[0] Sweep completeness: {sum(counts.values())}/"
          f"{expected * len(counts)} runs")
    short = {e: n for e, n in counts.items() if n < expected}
    for e, n in short.items():
        seeds = sorted(r["config"]["seed"] for r in by_eps[e])
        print(f"    eps={e:>5} : {n}/{expected}, seeds {seeds}  <- INCOMPLETE")
    if short:
        print("    FAIL: re-run the sweep to fill these in (run_sweep.sh skips")
        print("          runs whose JSON already exists), then re-check.")
        ok = False
    else:
        print(f"    ok: all {len(counts)} conditions have {expected} seeds")

    # [1] Flat wall-clock. A monotone rise with epsilon means compute is
    # drifting with the budget, and the energy axis is measuring workload size.
    print("\n[1] Wall-clock flatness across epsilon")
    print(f"    {'eps':>6} {'n':>3} {'wall_s':>18} {'energy_J':>18} {'steps/round':>12}")
    dp_walls = {}
    for eps, group in by_eps.items():
        w = np.array([measured_wall(r) for r in group])
        e = np.array([total_energy(r) for r in group])
        steps = np.nanmean(rounds_matrix(group, "steps_sum"))
        print(f"    {eps:>6} {len(group):>3} {w.mean():>10.1f} +/-{w.std():>5.1f} "
              f"{e.mean():>10.0f} +/-{e.std():>5.0f} {steps:>12.0f}")
        if not math.isinf(eps_key(eps)):
            dp_walls[eps] = w.mean()

    if len(dp_walls) >= 2:
        vals = np.array(list(dp_walls.values()))
        spread = 100 * (vals.max() - vals.min()) / vals.mean()
        ordered = [dp_walls[e] for e in sorted(dp_walls, key=eps_key)]
        print(f"\n    spread across DP conditions: {spread:.1f}% of mean")
        if all(a < b for a, b in zip(ordered, ordered[1:])):
            print("    FAIL: wall-clock still rises monotonically with epsilon.")
            print("          Compute is drifting with the privacy budget.")
            ok = False
        elif spread > 5.0:
            print(f"    FAIL: {spread:.1f}% spread exceeds the 5% tolerance.")
            ok = False
        else:
            print("    PASS: flat within tolerance, ordering collapsed into noise.")

    # [2] Compute parity against the privacy claim.
    print("\n[2] Executed steps vs steps the accountant was calibrated for")
    print("    A shortfall of a step or two per round is an Opacus artefact:")
    print("    it sets sample_rate = 1/len(loader), then takes int(1/rate), and")
    print("    for some partition sizes the float reciprocal lands just under")
    print("    the integer and floors down. It runs FEWER steps than the")
    print("    accountant charged for, so epsilon stays an upper bound. Only a")
    print("    shortfall above 0.5% of a round's steps is material.")
    for eps, group in by_eps.items():
        got = rounds_matrix(group, "steps_sum")
        want = rounds_matrix(group, "expected_steps_sum")
        gap = float(np.nanmax(np.abs(got - want)))
        pct = 100 * gap / max(float(np.nanmean(want)), 1.0)
        over_ran = float(np.nanmax(got - want)) > 0
        flag = "MATERIAL" if pct > 0.5 else ("over-run" if over_ran else "ok")
        print(f"    {eps:>6} : max |executed - calibrated| = {gap:.0f} steps "
              f"({pct:.2f}% of round)  [{flag}]")
        if pct > 0.5 or over_ran:
            ok = False

    # [3] GPU residency. Clients on CPU means NVML measured an idle card.
    print("\n[3] Clients trained on the GPU")
    for eps, group in by_eps.items():
        deficit = np.nanmin(rounds_matrix(group, "clients_on_cuda")
                            - rounds_matrix(group, "n_clients"))
        print(f"    {eps:>6} : {'ok' if deficit >= 0 else 'FAIL -- some ran on CPU'}")
        ok = ok and deficit >= 0

    # [3b] Idle baselines are all the same card doing nothing, so they must
    # agree. One that stands out was probed before the GPU wound down from the
    # previous run, and net energy = gross - idle*t, so it under-states that
    # run's net energy without disturbing anything else.
    print("\n[3b] Idle baseline agreement across runs")
    idles = [(str(r["config"]["epsilon"]), r["config"]["seed"],
              r["totals"]["idle_power_w"]) for r in runs]
    median = float(np.median([v for _, _, v in idles]))
    off = [(e, s, v) for e, s, v in idles
           if abs(v - median) > max(2.0, 0.10 * median)]
    unsettled = [(str(r["config"]["epsilon"]), r["config"]["seed"]) for r in runs
                 if r["totals"].get("idle_settled") is False]
    print(f"    median {median:.2f} W over {len(idles)} runs")
    for e, s, v in off:
        print(f"    eps={e:>5} seed={s}: {v:.2f} W  <- {v - median:+.1f} W off median")
    for e, s in unsettled:
        print(f"    eps={e:>5} seed={s}: probe never settled")
    if off or unsettled:
        print("    FAIL: contaminated idle baseline -- the GPU had not wound")
        print("          down when the probe started. Net energy for these runs")
        print("          is wrong; re-run them.")
        ok = False
    else:
        print("    ok: all baselines within tolerance of the median")

    # [3c] Very short rounds sit near the limit of a bracketed counter: GPU
    # power swings over tens of milliseconds, so a sub-second round averages
    # almost none of that away. The energy is still correct, just noisier.
    durations = [rd["wall_s"] for r in runs for rd in r["rounds"]]
    mean_round = float(np.mean(durations))
    print(f"\n[3c] Mean round duration {mean_round:.2f} s")
    if mean_round < 1.0:
        print("    WARNING: rounds under 1 s. Per-round energy is a noisy")
        print("             estimate here; prefer whole-run totals.")

    w, pct = noise_floor(by_eps)
    print(f"\n[4] Measurement noise floor: +/-{w:.2f} W ({pct:.2f}%)")
    print("    No effect smaller than this is resolvable. State it before")
    print("    claiming any energy difference.")

    ok = check_drift(runs) and ok
    print("\n" + ("VERDICT: PASS" if ok else "VERDICT: FAIL"))
    return ok


# ---------------------------------------------------------------------- tables


def tables(by_eps, targets):
    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)

    print("\n-- Per-round energy (separates fixed overhead from slowdown) --")
    print("    The overhead below is per-round ENERGY, net of idle. Power is")
    print("    the wrong basis: DP rounds draw more power AND run longer, so a")
    print("    power ratio drops the second factor and understates the cost.")
    print(f"    {'eps':>6} {'gross W':>9} {'net W':>9} {'gross J/rd':>11} "
          f"{'net J/rd':>9} {'sigma range':>18}")

    def net_per_round(group):
        return float(np.mean([total_energy(r, True) / len(r["rounds"])
                              for r in group]))

    baseline = None
    for eps, group in by_eps.items():
        gross_w = np.mean([total_energy(r) / measured_wall(r) for r in group])
        net_w = np.mean([total_energy(r, True) / measured_wall(r) for r in group])
        gross_jr = np.mean([total_energy(r) / len(r["rounds"]) for r in group])
        lo = rounds_matrix(group, "sigma_min")
        hi = rounds_matrix(group, "sigma_max")
        sigma = ("--" if np.isnan(lo).all()
                 else f"{np.nanmin(lo):.3f} - {np.nanmax(hi):.3f}")
        if math.isinf(eps_key(eps)):
            baseline = net_per_round(group)
        print(f"    {eps:>6} {gross_w:>9.2f} {net_w:>9.2f} {gross_jr:>11.0f} "
              f"{net_per_round(group):>9.1f} {sigma:>18}")

    if baseline:
        print("\n    DP overhead per round vs no-DP, net of idle:")
        private = []
        for eps, group in by_eps.items():
            if math.isinf(eps_key(eps)):
                continue
            jr = net_per_round(group)
            private.append(jr)
            print(f"      eps={eps:>4}: {100 * (jr - baseline) / baseline:+6.1f}%")
        mean = float(np.mean(private))
        spread = 100 * (max(private) - min(private)) / mean
        _, floor_pct = noise_floor(by_eps)
        print(f"\n      mean across DP conditions: "
              f"{100 * (mean - baseline) / baseline:+.0f}%")
        print(f"      spread across DP conditions: {spread:.1f}% of mean, "
              f"against a {floor_pct:.1f}% noise floor")
        if spread <= floor_pct * 1.75:
            print("      -> flat in epsilon within resolution: the per-round")
            print("         cost of DP is a FIXED implementation overhead, and")
            print("         everything budget-dependent lives in the round")
            print("         count, which energy-to-target measures.")
        else:
            print("      -> NOT flat: the per-round cost varies with epsilon by")
            print("         more than the instrument can attribute to noise.")

    print("\n-- Energy to target accuracy (the headline metric) --")
    for t in targets:
        print(f"\n    target acc = {t:.2f}")
        print(f"    {'eps':>6} {'gross J':>12} {'net J':>12} {'round':>7} {'seeds':>12}")
        for eps, group in by_eps.items():
            j, rd, n = energy_to_target(group, t)
            j_net, _, _ = energy_to_target(group, t, net=True)
            if n == 0:
                print(f"    {eps:>6} {'UNREACHED':>12} {'UNREACHED':>12} "
                      f"{'--':>7} {f'0/{len(group)}':>12}")
            else:
                print(f"    {eps:>6} {j:>12.0f} {j_net:>12.0f} {rd:>7.1f} "
                      f"{f'{n}/{len(group)}':>12}")

    print("\n-- Energy-optimal stopping round (peak-then-decline) --")
    print(f"    {'eps':>6} {'peak round':>11} {'peak acc':>10} {'final acc':>10} "
          f"{'wasted J':>10}")
    for eps, group in by_eps.items():
        pr, peak, final = peak_round(group)
        if pr is None:
            continue
        per_round = np.mean([total_energy(r) / len(r["rounds"]) for r in group])
        n_rounds = min(len(r["rounds"]) for r in group)
        note = "  <- declines after peak" if final < peak - 0.005 else ""
        print(f"    {eps:>6} {pr:>11} {peak:>10.4f} {final:>10.4f} "
              f"{per_round * (n_rounds - pr):>10.0f}{note}")

    print(f"\n-- Carbon: gCO2eq per {CARBON_RUNS} training runs, by grid --")
    print("    (the same measured workload repeated, not an extrapolation)")
    header = "".join(f"{k:>13}" for k in GRID_GCO2_PER_KWH)

    def kwh_per_scale(group):
        return np.mean([total_energy(r) for r in group]) / 3.6e6 * CARBON_RUNS

    print(f"    {'eps':>6}{header}")
    for eps, group in by_eps.items():
        kwh = kwh_per_scale(group)
        print(f"    {eps:>6}" + "".join(f"{kwh * g:>13.1f}"
                                        for g in GRID_GCO2_PER_KWH.values()))

    base = [g for e, g in by_eps.items() if math.isinf(eps_key(e))]
    if base:
        base_kwh = kwh_per_scale(base[0])
        print("\n    Carbon cost OF PRIVACY (DP minus no-DP), same scale:")
        print(f"    {'eps':>6}{header}")
        for eps, group in by_eps.items():
            if math.isinf(eps_key(eps)):
                continue
            delta = kwh_per_scale(group) - base_kwh
            print(f"    {eps:>6}" + "".join(f"{delta * g:>13.1f}"
                                            for g in GRID_GCO2_PER_KWH.values()))


# --------------------------------------------------------------------- figures


def figures(by_eps, out: Path):
    """The three figures the paper uses. Written into `out`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(by_eps)

    # Convergence, with a marker on any condition that declines after its peak.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for eps, group in by_eps.items():
        acc = rounds_matrix(group, "central_acc")
        mean, sd = np.nanmean(acc, 0), np.nanstd(acc, 0)
        x = np.arange(1, acc.shape[1] + 1)
        ax.plot(x, mean, label=eps_label(eps))
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.18)
        pr, peak, final = peak_round(group)
        if pr and final < peak - 0.005:
            ax.plot([pr], [peak], "v", color="k", ms=5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Central test accuracy")
    ax.set_title("Convergence under a fixed privacy budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig1_convergence.png", dpi=200)
    plt.close(fig)

    # Per-round energy: a flat DP plateau a fixed step above the baseline is
    # what separates the fixed overhead from the convergence slowdown.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    per_round = [[total_energy(r, True) / len(r["rounds"]) for r in by_eps[e]]
                 for e in labels]
    ax.bar(range(len(labels)), [np.mean(v) for v in per_round],
           yerr=[np.std(v) for v in per_round], capsize=3,
           color=["tab:gray" if math.isinf(eps_key(e)) else "tab:blue"
                  for e in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([eps_label(e) for e in labels], fontsize=7)
    ax.set_ylabel("Energy per round (J, net of idle)")
    ax.set_title("Per-round energy: fixed DP overhead, flat in $\\epsilon$")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig3_energy_per_round.png", dpi=200)
    plt.close(fig)

    # The same workload against different grids.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    regions = list(GRID_GCO2_PER_KWH)
    for eps in labels:
        kwh = np.mean([total_energy(r) for r in by_eps[eps]]) / 3.6e6 * CARBON_RUNS
        ax.plot(regions, [kwh * GRID_GCO2_PER_KWH[g] for g in regions], "o-",
                label=eps_label(eps), ms=4)
    ax.set_ylabel(f"gCO$_2$eq per {CARBON_RUNS} runs")
    ax.set_title("Identical workload, different grid")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig4_carbon.png", dpi=200)
    plt.close(fig)

    print(f"\nwrote fig1, fig3, fig4 to {out}/")


# ------------------------------------------------------------------- ablations


def ablations(runs, targets):
    """Each ablation factor against the baseline configuration.

    These runs deliberately change the workload, so unlike the main sweep a
    per-round energy difference here is expected and is the point. What they
    show is how the cost of privacy responds to a deployment choice.
    """
    # The baseline is READ FROM THE DATA. Each ablation varies one factor and
    # holds the rest at baseline, so the baseline value of every factor is the
    # one appearing in the most runs. Hardcoding it broke HAR silently once:
    # CIFAR's baseline has 20 supernodes and HAR has 21, so the "others at
    # baseline" filter rejected every run and the tables came out empty.
    base = {}
    for field in ABLATION_FACTORS:
        values = [r["config"].get(field) for r in runs
                  if r["config"].get(field) is not None]
        if values:
            base[field] = Counter(values).most_common(1)[0][0]
    print("\n    baseline read from the data: "
          + ", ".join(f"{ABLATION_FACTORS[k]}={v}" for k, v in base.items()))

    n_rounds = min((len(r["rounds"]) for r in runs), default=0)
    target = targets[len(targets) // 2]
    print(f"    'lost' is accuracy given back between the peak and round "
          f"{n_rounds}; 'wasted J' is the energy spent getting there.")
    print("\n    (* = single seed. Below the noise floor nothing under roughly")
    print("     10% is resolvable; re-run an axis with three seeds before")
    print('     quoting it: SEEDS="0 1 2" AXES=epochs ./run_ablations.sh)')

    for field, label in ABLATION_FACTORS.items():
        varied = sorted({r["config"].get(field) for r in runs
                         if r["config"].get(field) is not None})
        if len(varied) < 2:
            continue                      # left at baseline: not ablated
        print(f"\n-- {label} ({field}) --")
        print(f"    {'value':>8} {'eps':>6} {'n':>3} {'J/round':>9} {'peak rd':>8} "
              f"{'peak acc':>9} {'final acc':>10} {'lost':>7} {'wasted J':>9} "
              f"{f'J to {target:.2f}':>11}")
        for value in varied:
            # Hold every other factor at baseline: one thing varies at a time.
            selected = [
                r for r in runs
                if r["config"].get(field) == value
                and all(r["config"].get(k, base[k]) == base[k]
                        for k in base if k != field)
            ]
            for eps in sorted({str(r["config"]["epsilon"]) for r in selected},
                              key=eps_key):
                group = [r for r in selected
                         if str(r["config"]["epsilon"]) == eps]
                jr = np.mean([total_energy(r) / len(r["rounds"]) for r in group])
                j_target, _, n_hit = energy_to_target(group, target, net=True)
                pr, peak, final = peak_round(group)
                rounds = min(len(r["rounds"]) for r in group)
                wasted = jr * (rounds - pr) if pr else 0.0
                lost = 100 * (peak - final) / peak if peak else 0.0
                print(f"    {str(value):>8} {eps:>6} {len(group):>3} {jr:>9.0f} "
                      f"{str(pr):>8} {peak:>9.4f} {final:>10.4f} {lost:>6.1f}% "
                      f"{wasted:>9.0f} "
                      f"{('unreached' if n_hit == 0 else f'{j_target:.0f}'):>11}"
                      + (" *" if len(group) < 2 else ""))


# ----------------------------------------------------------------- paper notes


def paper(runs, by_eps):
    """The configuration and measurement numbers the write-up needs.

    Read off the JSONs, so the paper and the data cannot drift apart. The
    prose that used to live here now lives in paper/methodology.tex.
    """
    one = runs[0]["config"]
    seeds = sorted({r["config"]["seed"] for r in runs})
    hw = runs[0].get("hardware", {})

    print("=" * 68)
    print("PAPER NUMBERS -- transcribe, do not retype from memory")
    print("=" * 68)

    print("\n### Configuration\n")
    for key in ("dataset", "num_rounds", "num_supernodes", "fraction_train",
                "dirichlet_alpha", "local_epochs", "batch_size",
                "learning_rate", "clipping_norm", "delta"):
        print(f"    {key:<18} {one[key]}")
    print(f"    {'seeds':<18} {seeds}")
    print(f"    {'GPU':<18} {hw.get('gpu')}")
    print(f"    {'torch / cuda':<18} {hw.get('torch')} / {hw.get('cuda')}")

    print("\n### Per-client sigma range (varies with partition size)\n")
    for eps, group in by_eps.items():
        if math.isinf(eps_key(eps)):
            continue
        lo, hi = rounds_matrix(group, "sigma_min"), rounds_matrix(group, "sigma_max")
        if not np.isnan(lo).all():
            print(f"    eps={eps:>4}   sigma {np.nanmin(lo):.3f} - {np.nanmax(hi):.3f}")

    idle = np.mean([r["totals"]["idle_power_w"] for r in runs])
    idle_sd = np.mean([r["totals"]["idle_power_w_sd"] for r in runs])
    w, pct = noise_floor(by_eps)
    print("\n### Measurement\n")
    print("    NVML nvmlDeviceGetTotalEnergyConsumption, a hardware counter in")
    print("    accumulated mJ, differenced per round -- measured, not modelled.")
    print(f"    Idle baseline      {idle:.2f} W (sd {idle_sd:.2f})")
    print(f"    Noise floor        +/-{w:.2f} W ({pct:.2f}%)")
    print(f"    -> no effect below ~{pct:.0f}% is resolvable.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="?", default="results", type=Path)
    ap.add_argument("--check", action="store_true", help="the gate only")
    ap.add_argument("--ablation", action="store_true",
                    help="summarise ablation runs by varied factor")
    ap.add_argument("--paper", action="store_true",
                    help="configuration and measurement numbers for the write-up")
    ap.add_argument("--targets", default=None,
                    help="comma-separated target accuracies, e.g. 0.5,0.7,0.85")
    args = ap.parse_args()

    runs, by_eps, dataset = load(args.results)
    targets = (tuple(float(x) for x in args.targets.split(","))
               if args.targets
               else TARGET_ACCS_BY_DATASET.get(dataset, (0.15, 0.20, 0.25)))
    print(f"dataset: {dataset}   targets: {targets}\n")

    if args.paper:
        paper(runs, by_eps)
    elif args.ablation:
        print("=" * 68)
        print("ABLATIONS")
        print("=" * 68)
        ablations(runs, targets)
    elif args.check:
        sys.exit(0 if check(by_eps, runs) else 1)
    else:
        check(by_eps, runs)
        tables(by_eps, targets)
        figures(by_eps, args.results)


if __name__ == "__main__":
    main()
