"""Per-seed peak and final accuracy, one row per seed.

    python per_seed_peak.py results_har/            every epsilon
    python per_seed_peak.py results/ 0.5            one epsilon

The stopping-round claim is read off the MEAN curve over three seeds. That
mean can be produced two ways: all seeds peaking early, or one seed peaking
early and dragging an otherwise flat mean down. Only the first supports the
claim, so the per-seed rows have to be shown before it is made.

Energies are NET of idle, matching the paper's tables.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def eps_key(e):
    s = str(e).strip().lower()
    return math.inf if s in ("inf", "infinity", "none", "") else float(s)


def net_energy(run):
    """Idle-subtracted energy over the window the counter covers."""
    t = run["totals"]
    return max(0.0, t["energy_j"] - t["idle_power_w"] * t["wall_s_measured"])


def report(target, runs):
    runs = sorted(runs, key=lambda r: r["config"]["seed"])
    n_rounds = min(len(r["rounds"]) for r in runs)
    print(f"\nepsilon = {target}, {len(runs)} seeds, {n_rounds} rounds\n")
    print(f"  {'seed':>4} {'peak rd':>8} {'peak acc':>9} {'final acc':>10} "
          f"{'drop':>8} {'net J after peak':>17} {'% of run':>9}")

    curves = []
    for r in runs:
        acc = [rd["central_acc"] for rd in r["rounds"]]
        gross = [rd["energy_j"] for rd in r["rounds"]]
        # Scale gross per-round energy to net by the run's own ratio, so these
        # numbers are consistent with totals.energy_j_net_idle.
        scale = net_energy(r) / sum(gross) if sum(gross) > 0 else 1.0
        curves.append(acc)
        peak = max(acc)
        pr = acc.index(peak) + 1
        after = sum(gross[pr:]) * scale
        total = sum(gross) * scale
        print(f"  {r['config']['seed']:>4} {pr:>8} {peak:>9.4f} {acc[-1]:>10.4f} "
              f"{peak - acc[-1]:>+8.4f} {after:>17.0f} "
              f"{100 * after / total if total else 0:>8.1f}%")

    n = min(len(c) for c in curves)
    mean = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
    peak = max(mean)
    pr = mean.index(peak) + 1
    drop = peak - mean[-1]
    print(f"\n  mean curve: peak {peak:.4f} at round {pr}, final {mean[-1]:.4f}, "
          f"drop {drop:+.4f}")

    # The across-seed spread of FINAL accuracy is this condition's accuracy
    # resolution. A peak-to-final drop below it is not a decline, and an
    # argmax read off a plateau that noisy is not a stopping round.
    finals = [c[-1] for c in curves]
    m = sum(finals) / len(finals)
    sd = (sum((x - m) ** 2 for x in finals) / (len(finals) - 1)) ** 0.5
    print(f"  across-seed SD of final accuracy: +/-{sd:.4f}")

    peaks = sorted(c.index(max(c)) + 1 for c in curves)
    print(f"  per-seed peak rounds: {peaks}")

    if drop <= sd:
        print(f"  -> NO RESOLVABLE STOPPING ROUND: the drop ({drop:+.4f}) is")
        print(f"     within the across-seed spread (+/-{sd:.4f}). Reporting a")
        print("     peak round here would be reading an argmax off noise.")
        return
    if not (peaks[0] <= pr <= peaks[-1]):
        print(f"  -> the mean peaks at round {pr}, OUTSIDE the per-seed range")
        print(f"     {peaks[0]}..{peaks[-1]}. Averaging noisy plateaus moved the")
        print("     argmax; do not quote the mean's peak round.")
        return
    spread = peaks[-1] - peaks[0]
    if spread > 0.3 * n:
        print("  -> the drop clears the spread, but the per-seed peak rounds")
        print(f"     scatter over {spread} rounds of {n}. Quote the range, not")
        print("     a single round.")
    else:
        print("  -> the drop clears the across-seed spread and the seeds agree")
        print("     on when it happens. The stopping round is real here.")


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    wanted = str(sys.argv[2]) if len(sys.argv) > 2 else None

    runs = [json.loads(p.read_text()) for p in sorted(results.glob("run_*.json"))]
    if not runs:
        sys.exit(f"no run_*.json found in {results}")

    by_eps = defaultdict(list)
    for r in runs:
        by_eps[str(r["config"]["epsilon"])].append(r)

    if wanted is not None:
        if wanted not in by_eps:
            sys.exit(f"no runs at epsilon={wanted}; have "
                     f"{sorted(by_eps, key=eps_key)}")
        report(wanted, by_eps[wanted])
        return
    for eps in sorted(by_eps, key=eps_key):
        report(eps, by_eps[eps])


if __name__ == "__main__":
    main()
