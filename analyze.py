"""Analysis for 'The Energy Cost of Privacy in Federated Learning'.

Reads results/run_*.json and nothing else -- the JSONs are the unit of
analysis, and no number in the paper should come from a log.

    python analyze.py results/            # checks + tables + figures
    python analyze.py results/ --check    # Phase-1 gate only, exits non-zero

PHASE-1 GATE (--check)
----------------------
The energy axis is only interpretable if every epsilon condition ran the
same workload. Three things must hold before any energy number is quoted:

  1. Wall-clock is FLAT across the five DP conditions, with only the
     epsilon=inf baseline lower. The pre-step-cap sweep scaled monotonically
     110s (eps=0.5) -> 133s (eps=8), which meant compute was drifting with
     epsilon and the energy axis was measuring workload size, not privacy.
  2. Executed steps match the steps the accountant was calibrated against
     (steps_sum vs expected_steps_sum). A gap means the privacy claim and
     the workload disagree.
  3. Every client trained on the GPU NVML is watching.

Effects smaller than the measurement noise floor are not reported as
effects. The floor is computed here from the across-seed spread within each
condition rather than assumed.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Grid carbon intensity, gCO2eq/kWh. Algeria's grid is overwhelmingly
# natural gas, which is the regional hook: the same joules bought with the
# same privacy budget emit very differently depending on where they burn.
GRID_GCO2_PER_KWH = {
    "Algeria": 488,
    "France": 56,
    "Germany": 344,
    "USA (avg)": 369,
    "Norway": 26,
    "India": 631,
}

TARGET_ACCS = (0.15, 0.20, 0.25)


# ------------------------------------------------------------------ loading


def eps_key(e):
    """Sort key: the non-private baseline sorts last."""
    s = str(e).strip().lower()
    return math.inf if s in ("inf", "infinity", "none", "") else float(s)


def eps_label(e):
    return "no DP" if math.isinf(eps_key(e)) else f"$\\epsilon$={e}"


def load(d: Path):
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"))]
    if not runs:
        sys.exit(f"no run_*.json found in {d}")
    by_eps = defaultdict(list)
    for r in runs:
        by_eps[str(r["config"]["epsilon"])].append(r)
    return runs, dict(sorted(by_eps.items(), key=lambda kv: eps_key(kv[0])))


def rounds_matrix(group, field, default=np.nan):
    """(n_seeds, n_rounds) array of a per-round field."""
    n = min(len(r["rounds"]) for r in group)
    return np.array(
        [[rd.get(field, default) for rd in r["rounds"][:n]] for r in group], float
    )


def total_energy(r, net=False):
    """Gross, or idle-subtracted, round-bracketed energy in joules.

    Net is recomputed here rather than read back from the file. Runs written
    before the idle fix subtracted idle over the whole-process wall-clock
    while energy_j only ever covered the round windows (~60% of it), which
    over-subtracted ~20% of gross. Recomputing over measured_wall puts old
    and new runs on the same footing.
    """
    t = r["totals"]
    if not net:
        return t["energy_j"]
    idle = t.get("idle_power_w", r.get("hardware", {}).get("idle_power_w_at_start"))
    if idle is None:
        return t["energy_j_net_idle"]
    return max(0.0, t["energy_j"] - idle * measured_wall(r))


def measured_wall(r):
    """Wall-clock the energy counter actually covers.

    Older JSONs wrote only `wall_s` (whole process, ~40% of which the round
    window never covered). Prefer the round-bracketed sum.
    """
    t = r["totals"]
    if "wall_s_measured" in t:
        return t["wall_s_measured"]
    return sum(rd["wall_s"] for rd in r["rounds"])


# ------------------------------------------------------------------- checks


def noise_floor(by_eps):
    """Measurement noise floor: worst within-condition across-seed spread.

    Reported in watts and as a percentage, and nothing smaller than this is
    claimed as an effect anywhere in the paper.
    """
    worst_w, worst_pct = 0.0, 0.0
    for group in by_eps.values():
        if len(group) < 2:
            continue
        p = np.array([total_energy(r) / measured_wall(r) for r in group])
        worst_w = max(worst_w, float(p.std(ddof=1)))
        worst_pct = max(worst_pct, float(100 * p.std(ddof=1) / p.mean()))
    return worst_w, worst_pct


def _rank(xs):
    """Ranks with ties averaged (for Spearman without pulling in scipy)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation. Returns 0.0 when undefined."""
    if len(xs) < 3:
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return num / den if den else 0.0


def drift_check(runs) -> bool:
    """Is wall-clock tracking epsilon, or just tracking execution order?

    The sweep runs epsilon in a fixed descending order, so "slower at high
    epsilon" and "slower early in the sweep" predict the same table. They are
    completely different findings: one is a cost of privacy, the other is the
    machine warming up. Correlating wall-clock against both separates them.

    Nothing in DP-SGD makes runtime depend on the MAGNITUDE of sigma -- noise
    is drawn from the same shaped tensor either way, and step counts are flat
    across conditions. So a wall-clock trend in epsilon with flat step counts
    is far more likely to be run-order drift, and must be ruled out before
    any of it is attributed to privacy.
    """
    dated = [r for r in runs if r.get("timestamp_utc")]
    if len(dated) < 4:
        print("\n[5] Run-order drift: too few timestamped runs to assess")
        return True
    dated.sort(key=lambda r: r["timestamp_utc"])

    walls, order_idx, eps_rank = [], [], []
    print("\n[5] Run-order drift (runs in execution order)")
    print(f"    {'#':>3} {'eps':>6} {'seed':>5} {'wall_s':>9} {'gross W':>9}  timestamp")
    for i, r in enumerate(dated):
        w = measured_wall(r)
        e = eps_key(r["config"]["epsilon"])
        walls.append(w)
        order_idx.append(float(i))
        # inf sorts last; use a large finite stand-in for ranking
        eps_rank.append(1e9 if math.isinf(e) else e)
        print(
            f"    {i:>3} {str(r['config']['epsilon']):>6} {r['config']['seed']:>5} "
            f"{w:>9.1f} {total_energy(r) / w:>9.2f}  {r['timestamp_utc'][:19]}"
        )

    rho_order = spearman(order_idx, walls)
    rho_eps = spearman(eps_rank, walls)
    # Same question restricted to DP runs, where epsilon actually varies and
    # the non-private baseline cannot drive the correlation on its own.
    dp = [
        (o, e, w)
        for o, e, w in zip(order_idx, eps_rank, walls)
        if e < 1e9
    ]
    rho_order_dp = spearman([d[0] for d in dp], [d[2] for d in dp])
    rho_eps_dp = spearman([d[1] for d in dp], [d[2] for d in dp])

    print(f"\n    wall_s vs execution order : rho = {rho_order:+.3f}   "
          f"(DP only: {rho_order_dp:+.3f})")
    print(f"    wall_s vs epsilon         : rho = {rho_eps:+.3f}   "
          f"(DP only: {rho_eps_dp:+.3f})")

    # Are the two explanatory variables themselves confounded? If epsilon was
    # swept in a fixed order, order and epsilon are near-perfectly
    # rank-correlated and NO correlation computed on this data can separate
    # them. Saying which one "wins" would then be an artefact of noise.
    conf = spearman([d[0] for d in dp], [d[1] for d in dp])
    print(f"    execution order vs epsilon: rho = {conf:+.3f}   <- design check")

    ok = True
    if abs(conf) > 0.8:
        print("\n    FAIL: epsilon and execution order are confounded by design")
        print(f"          (|rho| = {abs(conf):.2f}). The sweep visited epsilon in a")
        print("          fixed order, so 'slower at high epsilon' and 'slower")
        print("          early in the sweep' predict the same table and this")
        print("          data cannot tell them apart. Note nothing in DP-SGD")
        print("          makes runtime depend on the magnitude of sigma, and")
        print("          step counts are flat across conditions, so drift is")
        print("          the more likely of the two. Re-run: run_sweep.sh now")
        print("          randomises run order, which breaks the confound.")
        ok = False
    elif abs(rho_order_dp) > 0.6 and abs(rho_order_dp) >= abs(rho_eps_dp):
        print("\n    FAIL: with run order randomised, wall-clock still tracks")
        print("          execution order more strongly than epsilon. The drift")
        print("          is real and is not a privacy effect; find its cause")
        print("          before quoting any energy number.")
        ok = False
    elif abs(rho_eps_dp) > 0.6:
        print("\n    Wall-clock tracks epsilon under randomised order. This is")
        print("    a real epsilon effect -- but check [1] and [2] first: with")
        print("    step counts flat, the mechanism is not obvious and needs an")
        print("    explanation before it goes in the paper.")
    else:
        print("\n    PASS: no strong drift in either direction.")
    return ok


def check(by_eps, runs=None) -> bool:
    ok = True
    print("=" * 68)
    print("PHASE-1 VERIFICATION")
    print("=" * 68)

    dp_walls = {}
    print("\n[1] Wall-clock flatness across epsilon")
    print(f"    {'eps':>6} {'n':>3} {'wall_s':>18} {'energy_J':>18} {'steps/round':>12}")
    for eps, group in by_eps.items():
        w = np.array([measured_wall(r) for r in group])
        e = np.array([total_energy(r) for r in group])
        st = rounds_matrix(group, "steps_sum")
        sm = "n/a" if np.isnan(st).all() else f"{np.nanmean(st):.0f}"
        print(
            f"    {eps:>6} {len(group):>3} {w.mean():>10.1f} +/-{w.std(ddof=0):>5.1f} "
            f"{e.mean():>10.0f} +/-{e.std(ddof=0):>5.0f} {sm:>12}"
        )
        if not math.isinf(eps_key(eps)):
            dp_walls[eps] = w.mean()

    if len(dp_walls) >= 2:
        vals = np.array(list(dp_walls.values()))
        spread = 100 * (vals.max() - vals.min()) / vals.mean()
        order = [e for e, _ in sorted(dp_walls.items(), key=lambda kv: eps_key(kv[0]))]
        seq = [dp_walls[e] for e in order]
        monotone = all(a < b for a, b in zip(seq, seq[1:]))
        print(f"\n    spread across DP conditions: {spread:.1f}% of mean")
        if monotone:
            print("    FAIL: wall-clock still increases monotonically with epsilon.")
            print("          Compute is drifting with the privacy budget; the")
            print("          energy axis is measuring workload size, not privacy.")
            ok = False
        elif spread > 5.0:
            print(f"    FAIL: {spread:.1f}% spread exceeds the 5% tolerance.")
            ok = False
        else:
            print("    PASS: flat within tolerance, ordering collapsed into noise.")

    print("\n[2] Executed steps vs steps the accountant was calibrated for")
    print("    A shortfall of a step or two per round is an Opacus artefact,")
    print("    not a bug here: it sets sample_rate = 1/len(loader) and then")
    print("    takes int(1/sample_rate), and for some partition sizes the")
    print("    float reciprocal lands just under the integer and floors down")
    print("    (e.g. ceil(n/B)=93 yields 92 batches). It runs FEWER steps than")
    print("    the accountant charged for, so epsilon stays an upper bound.")
    print("    Only a shortfall above 0.5% of the round's steps is material.")
    for eps, group in by_eps.items():
        got = rounds_matrix(group, "steps_sum")
        want = rounds_matrix(group, "expected_steps_sum")
        if np.isnan(got).all() or np.isnan(want).all():
            print(f"    {eps:>6} : n/a (run predates steps_sum logging)")
            continue
        d = float(np.nanmax(np.abs(got - want)))
        pct = 100 * d / max(float(np.nanmean(want)), 1.0)
        over = float(np.nanmax(got - want)) > 0
        flag = "MATERIAL" if pct > 0.5 else ("over-run" if over else "ok")
        print(
            f"    {eps:>6} : max |executed - calibrated| = {d:.0f} steps "
            f"({pct:.2f}% of round)  [{flag}]"
        )
        if pct > 0.5 or over:
            ok = False

    print("\n[3] Clients trained on the GPU")
    for eps, group in by_eps.items():
        cu = rounds_matrix(group, "clients_on_cuda")
        nc = rounds_matrix(group, "n_clients")
        if np.isnan(cu).all():
            print(f"    {eps:>6} : n/a (run predates device logging)")
            continue
        if np.nanmin(cu - nc) < 0:
            print(f"    {eps:>6} : FAIL -- some ClientApps ran on CPU")
            ok = False
        else:
            print(f"    {eps:>6} : ok")

    w, pct = noise_floor(by_eps)
    print(f"\n[4] Measurement noise floor: +/-{w:.2f} W ({pct:.2f}%)")
    print("    No effect smaller than this is resolvable. State it before")
    print("    claiming any energy difference.")

    if runs:
        ok = drift_check(runs) and ok

    print("\n" + ("VERDICT: PASS" if ok else "VERDICT: FAIL"))
    return ok


# ------------------------------------------------------------------- tables


def energy_to_target(group, target, net=False):
    """Mean joules to first reach `target` central accuracy, and the round.

    Returns (joules, round, n_reached). Unreached seeds are excluded and
    counted -- a target no configuration reaches within the round budget is
    itself a result, and must be reported as unreached rather than dropped.
    """
    js, rds = [], []
    for r in group:
        cum = 0.0
        for rd in r["rounds"]:
            cum += rd["energy_j"]
            if rd.get("central_acc", float("nan")) >= target:
                # Scale gross->net by the run's own ratio, so the reported
                # net figure is consistent with totals.energy_j_net_idle.
                ratio = (
                    total_energy(r, True) / total_energy(r)
                    if net and total_energy(r) > 0
                    else 1.0
                )
                js.append(cum * ratio)
                rds.append(rd["round"])
                break
    if not js:
        return None, None, 0
    return float(np.mean(js)), float(np.mean(rds)), len(js)


def peak_round(group):
    """Round of peak mean accuracy, the peak, and the final-round accuracy.

    Under a tight budget the curve peaks and then declines: past that round,
    training spends joules AND loses accuracy. That round is the
    energy-optimal stopping point.
    """
    acc = rounds_matrix(group, "central_acc")
    m = np.nanmean(acc, 0)
    if np.isnan(m).all():
        return None, None, None
    i = int(np.nanargmax(m))
    return i + 1, float(m[i]), float(m[-1])


def tables(by_eps):
    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)

    print("\n-- Per-round energy (separates fixed overhead from slowdown) --")
    print(f"    {'eps':>6} {'gross W':>10} {'net W':>10} {'J/round':>10} {'sigma range':>20}")
    base_net = None
    for eps, group in by_eps.items():
        gw = np.mean([total_energy(r) / measured_wall(r) for r in group])
        nw = np.mean([total_energy(r, True) / measured_wall(r) for r in group])
        jr = np.mean([total_energy(r) / len(r["rounds"]) for r in group])
        lo, hi = rounds_matrix(group, "sigma_min"), rounds_matrix(group, "sigma_max")
        sig = (
            "--"
            if np.isnan(lo).all()
            else f"{np.nanmin(lo):.3f} - {np.nanmax(hi):.3f}"
        )
        if math.isinf(eps_key(eps)):
            base_net = nw
        print(f"    {eps:>6} {gw:>10.2f} {nw:>10.2f} {jr:>10.0f} {sig:>20}")
    if base_net:
        print(f"\n    DP overhead vs no-DP, net of idle (fixed per-round cost):")
        for eps, group in by_eps.items():
            if math.isinf(eps_key(eps)):
                continue
            nw = np.mean([total_energy(r, True) / measured_wall(r) for r in group])
            print(f"      eps={eps:>4}: {100 * (nw - base_net) / base_net:+6.1f}%")

    print("\n-- Energy to target accuracy (the headline metric) --")
    for t in TARGET_ACCS:
        print(f"\n    target acc = {t:.2f}")
        print(f"    {'eps':>6} {'gross J':>12} {'net J':>12} {'round':>7} {'seeds':>12}")
        for eps, group in by_eps.items():
            j, rd, n = energy_to_target(group, t)
            jn, _, _ = energy_to_target(group, t, net=True)
            if n == 0:
                print(f"    {eps:>6} {'UNREACHED':>12} {'UNREACHED':>12} {'--':>7} {f'0/{len(group)}':>12}")
            else:
                print(f"    {eps:>6} {j:>12.0f} {jn:>12.0f} {rd:>7.1f} {f'{n}/{len(group)}':>12}")

    print("\n-- Energy-optimal stopping round (peak-then-decline) --")
    print(f"    {'eps':>6} {'peak round':>11} {'peak acc':>10} {'final acc':>10} {'wasted J':>10}")
    for eps, group in by_eps.items():
        pr, pa, fa = peak_round(group)
        if pr is None:
            continue
        jr = np.mean([total_energy(r) / len(r["rounds"]) for r in group])
        nr = min(len(r["rounds"]) for r in group)
        wasted = jr * (nr - pr)
        note = "  <-- declines after peak" if fa < pa - 0.005 else ""
        print(f"    {eps:>6} {pr:>11} {pa:>10.4f} {fa:>10.4f} {wasted:>10.0f}{note}")

    print("\n-- Carbon: gCO2eq for one full run, by grid --")
    hdr = "".join(f"{k:>13}" for k in GRID_GCO2_PER_KWH)
    print(f"    {'eps':>6}{hdr}")
    for eps, group in by_eps.items():
        kwh = np.mean([total_energy(r) for r in group]) / 3.6e6
        row = "".join(f"{kwh * g:>13.2f}" for g in GRID_GCO2_PER_KWH.values())
        print(f"    {eps:>6}{row}")


# ------------------------------------------------------------------ figures


def figures(by_eps, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1 -- accuracy vs round, SD bands
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for eps, group in by_eps.items():
        acc = rounds_matrix(group, "central_acc")
        m, s = np.nanmean(acc, 0), np.nanstd(acc, 0)
        x = np.arange(1, acc.shape[1] + 1)
        ax.plot(x, m, label=eps_label(eps))
        ax.fill_between(x, m - s, m + s, alpha=0.18)
        pr, pa, fa = peak_round(group)
        if pr and fa < pa - 0.005:
            ax.plot([pr], [pa], "v", color="k", ms=5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Central test accuracy")
    ax.set_title("Convergence under a fixed privacy budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig1_convergence.png", dpi=200)
    plt.close(fig)

    # Fig 2 -- energy to target accuracy
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    width = 0.26
    labels = list(by_eps)
    xs = np.arange(len(labels))
    for i, t in enumerate(TARGET_ACCS):
        vals, hatch = [], []
        for eps in labels:
            j, _, n = energy_to_target(by_eps[eps], t, net=True)
            vals.append(0 if n == 0 else j)
            hatch.append(n == 0)
        b = ax.bar(xs + (i - 1) * width, vals, width, label=f"acc {t:.2f}")
        for rect, un in zip(b, hatch):
            if un:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    0,
                    "unreached",
                    rotation=90,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )
    ax.set_xticks(xs)
    ax.set_xticklabels([eps_label(e) for e in labels], fontsize=7)
    ax.set_ylabel("Energy to target (J, net of idle)")
    ax.set_title("Energy-to-target-accuracy")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2_energy_to_target.png", dpi=200)
    plt.close(fig)

    # Fig 3 -- per-round energy: flat within DP, step up from no-DP.
    # This is the figure that separates fixed overhead from convergence
    # slowdown: a flat DP plateau means the per-round cost of privacy is a
    # constant implementation overhead, and everything else in the
    # energy-to-target metric is the slowdown.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = list(by_eps)
    means = [
        np.mean([total_energy(r) / len(r["rounds"]) for r in by_eps[e]]) for e in labels
    ]
    sds = [
        np.std([total_energy(r) / len(r["rounds"]) for r in by_eps[e]]) for e in labels
    ]
    cols = ["tab:gray" if math.isinf(eps_key(e)) else "tab:blue" for e in labels]
    ax.bar(range(len(labels)), means, yerr=sds, capsize=3, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([eps_label(e) for e in labels], fontsize=7)
    ax.set_ylabel("Energy per round (J)")
    ax.set_title("Per-round energy: fixed DP overhead, flat in $\\epsilon$")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig3_energy_per_round.png", dpi=200)
    plt.close(fig)

    # Fig 4 -- carbon by grid region
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    regions = list(GRID_GCO2_PER_KWH)
    for eps in labels:
        kwh = np.mean([total_energy(r) for r in by_eps[eps]]) / 3.6e6
        ax.plot(regions, [kwh * GRID_GCO2_PER_KWH[g] for g in regions], "o-",
                label=eps_label(eps), ms=4)
    ax.set_ylabel("gCO$_2$eq per run")
    ax.set_title("Same workload, same privacy, different grid")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig4_carbon.png", dpi=200)
    plt.close(fig)

    print(f"\nwrote fig1..fig4 to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results", type=Path)
    ap.add_argument("--check", action="store_true", help="Phase-1 gate only")
    a = ap.parse_args()

    runs, by_eps = load(a.results)
    ok = check(by_eps, runs)
    if a.check:
        sys.exit(0 if ok else 1)
    tables(by_eps)
    figures(by_eps, a.results)


if __name__ == "__main__":
    main()
