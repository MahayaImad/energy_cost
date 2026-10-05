"""Recompute each client's SPENT epsilon from the run logs.

    python check_realised_epsilon.py results_har/

Why this is not a formality. Sigma is calibrated against the EXPECTED number
of rounds a client joins, num_rounds * fraction_train. The realised number is
a binomial draw around it: with 100 rounds at fraction 0.5 the expectation is
50 but the standard deviation is 5, so a client that happens to be sampled 60
times takes 20% more steps than the accountant was paid for -- and its spent
epsilon exceeds the epsilon the paper reports. "Reported epsilon is an upper
bound" is justified by claiming no subsampling amplification; it is NOT
justified for over-participation. This script checks it instead of assuming.

The logs do not store per-client sigma or n, only the per-round min/max and
the sum of examples. Both are recovered deterministically:

  participation  from `sampled_partitions`, recorded for every round
  n per client   by rebuilding the partition the ClientApps were given
  sigma          as a pure function of (epsilon, n, B, rounds, frac, epochs)

The reconstruction is then PROVEN rather than trusted: for every round, the
sum of n over that round's sampled partitions must equal the recorded
`examples_sum`, and the min and max of the reconstructed sigmas must equal the
recorded `sigma_min` and `sigma_max`. Any mismatch aborts.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


_sizes_cache: dict = {}


def partition_sizes(dataset: str, num_partitions: int, alpha: float, seed: int):
    """Local train-set size per client, exactly as the ClientApp saw it.

    Keyed on the seed, because it matters: the HAR partition is by study
    participant and so is seed-independent, but CIFAR-10's Dirichlet
    partition is redrawn per seed, giving each run of the sweep DIFFERENT
    client sizes. Computing one set of sizes and applying it to every run
    would audit a federation that never existed.
    """
    from energyfl.task import is_har, load_partition

    if is_har(dataset):
        seed = 0
    key = (str(dataset).lower(), int(num_partitions), float(alpha), int(seed))
    if key not in _sizes_cache:
        _sizes_cache[key] = [
            len(load_partition(pid, num_partitions, 32, alpha, seed,
                               dataset=dataset)[0].dataset)
            for pid in range(num_partitions)
        ]
    return _sizes_cache[key]


def spent_epsilon(sigma: float, n: int, batch_size: int, steps: int,
                  delta: float) -> float:
    """Epsilon actually spent after `steps` steps at this sigma."""
    from opacus.accountants import create_accountant

    from energyfl import dp

    accountant = create_accountant(dp.ACCOUNTANT)
    accountant.history = [(sigma, 1.0 / dp.steps_per_epoch(n, batch_size), steps)]
    return accountant.get_epsilon(delta=delta)


def worst_case_epsilon(run, sizes):
    """Epsilon if a client were sampled in EVERY round: the true worst case.

    Sigma is calibrated against R * fraction_train steps, an expectation. With
    participation random and no subsampling amplification claimed, nothing in
    the protocol stops a client from being drawn every round, and then it
    takes R * E * ceil(n/B) steps. That bound, not the expectation, is what a
    guarantee has to hold against -- so the number here is what the reported
    epsilon would have to be to be an upper bound at all. Returns
    (worst_eps, client_id) or (None, None) for the non-private baseline.
    """
    from energyfl import dp

    cfg = run["config"]
    if not dp.is_private(cfg["epsilon"]):
        return None, None
    batch_size, epochs = cfg["batch_size"], cfg["local_epochs"]
    worst, worst_pid = -1.0, None
    for pid, n in enumerate(sizes):
        sigma = dp.noise_multiplier_for(
            epsilon=float(cfg["epsilon"]), n_examples=n,
            batch_size=batch_size, num_rounds=cfg["num_rounds"],
            fraction_train=cfg["fraction_train"], local_epochs=epochs,
        )
        steps = cfg["num_rounds"] * epochs * dp.steps_per_epoch(n, batch_size)
        eps = spent_epsilon(sigma, n, batch_size, steps, cfg["delta"])
        if eps > worst:
            worst, worst_pid = eps, pid
    return worst, worst_pid


def check_run(run, sizes, verbose=False):
    """Returns (target_eps, worst_spent, worst_client, participation counts)."""
    from energyfl import dp

    cfg = run["config"]
    batch_size, epochs = cfg["batch_size"], cfg["local_epochs"]
    target = cfg["epsilon"]

    # Participation, straight from the logs.
    joined = Counter()
    for rd in run["rounds"]:
        if "sampled_partitions" not in rd:
            sys.exit(f"{run['run_id']}: round {rd['round']} has no "
                     "sampled_partitions; this run cannot be audited.")
        joined.update(rd["sampled_partitions"])

    # Prove the reconstructed sizes are the ones that actually ran.
    for rd in run["rounds"]:
        want = rd.get("examples_sum")
        got = sum(sizes[p] for p in rd["sampled_partitions"])
        if want is not None and abs(got - want) > 0.5:
            sys.exit(
                f"{run['run_id']} round {rd['round']}: reconstructed partition "
                f"sizes sum to {got} but the run recorded {want}. The partition "
                "cannot be reproduced, so no epsilon computed here would be "
                "about the run that actually happened."
            )

    if not dp.is_private(target):
        return None, None, None, joined

    sigmas = {
        pid: dp.noise_multiplier_for(
            epsilon=float(target), n_examples=sizes[pid],
            batch_size=batch_size, num_rounds=cfg["num_rounds"],
            fraction_train=cfg["fraction_train"], local_epochs=epochs,
        )
        for pid in joined
    }

    # Cross-check against what the server logged per round.
    for rd in run["rounds"]:
        if "sigma_min" not in rd:
            continue
        here = [sigmas[p] for p in rd["sampled_partitions"]]
        for name, ours, theirs in (("min", min(here), rd["sigma_min"]),
                                   ("max", max(here), rd["sigma_max"])):
            if abs(ours - theirs) > 1e-6:
                sys.exit(
                    f"{run['run_id']} round {rd['round']}: reconstructed sigma "
                    f"{name} {ours:.6f} != logged {theirs:.6f}. The noise "
                    "actually applied is not the noise this script models."
                )

    worst_eps, worst_pid = -1.0, None
    for pid, rounds_joined in joined.items():
        steps = rounds_joined * epochs * dp.steps_per_epoch(sizes[pid], batch_size)
        eps = spent_epsilon(sigmas[pid], sizes[pid], batch_size, steps,
                            cfg["delta"])
        if verbose:
            calibrated = dp.total_local_steps(
                sizes[pid], batch_size, cfg["num_rounds"],
                cfg["fraction_train"], epochs)
            print(f"        client {pid:>2}  n={sizes[pid]:>5}  joined "
                  f"{rounds_joined:>3}  steps {steps:>5} vs {calibrated:>5} "
                  f"calibrated  sigma {sigmas[pid]:.4f}  spent eps {eps:.4f}")
        if eps > worst_eps:
            worst_eps, worst_pid = eps, pid

    return float(target), worst_eps, worst_pid, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--verbose", action="store_true",
                    help="one line per client per run")
    ap.add_argument("--worst-case", action="store_true",
                    help="also report epsilon under participation in EVERY "
                         "round, the bound a guarantee must hold against")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="write {epsilon: worst spent epsilon} for "
                         "analyze.py --latex --realised-eps")
    args = ap.parse_args()

    paths = sorted(args.results.glob("run_*.json"))
    if not paths:
        sys.exit(f"no run_*.json found in {args.results}")
    runs = [json.loads(p.read_text()) for p in paths]

    def sizes_for(cfg):
        return partition_sizes(cfg["dataset"], cfg["num_supernodes"],
                               cfg["dirichlet_alpha"], cfg["seed"])

    one = runs[0]["config"]
    spans = {}
    for run in runs:
        sz = sizes_for(run["config"])
        spans[run["config"]["seed"]] = (min(sz), max(sz))
    print(f"dataset {one['dataset']}, {one['num_supernodes']} clients")
    for seed in sorted(spans):
        lo, hi = spans[seed]
        print(f"  seed {seed}: n = {lo}..{hi}")
    print()

    print("Participation: how many rounds each client was actually sampled in,")
    print("against the expectation the accountant was calibrated for.\n")

    breaches = []
    for run, path in zip(runs, paths):
        cfg = run["config"]
        sizes = sizes_for(cfg)
        target, worst, pid, joined = check_run(run, sizes, args.verbose)
        expected = cfg["num_rounds"] * cfg["fraction_train"]
        counts = sorted(joined.values())
        line = (f"  {path.name:<44} joined {counts[0]}..{counts[-1]} rounds "
                f"(expected {expected:.0f})")
        if target is None:
            print(line + "   no DP")
            continue
        over = worst > target + 1e-9
        print(line + f"\n  {'':<44} worst spent eps = {worst:.4f} vs "
              f"{target} claimed, client {pid}"
              + ("   <-- EXCEEDED" if over else "   ok"))
        if over:
            breaches.append((path.name, target, worst, pid))

    if args.json_out:
        worst_by_eps = {}
        for run in runs:
            target = run["config"]["epsilon"]
            _, worst, _, _ = check_run(run, sizes_for(run["config"]))
            if worst is None:
                continue
            key = str(target)
            worst_by_eps[key] = max(worst_by_eps.get(key, 0.0), worst)
        args.json_out.write_text(json.dumps(worst_by_eps, indent=2, sort_keys=True))
        print(f"wrote worst spent epsilon per condition to {args.json_out}")

    if args.worst_case:
        print("\nWorst case: epsilon if a client joined every round "
              f"({one['num_rounds']} of {one['num_rounds']}), against the "
              f"{one['num_rounds'] * one['fraction_train']:.0f} the "
              "accountant was calibrated for.\n")
        seen = {}
        for run in runs:
            target = run["config"]["epsilon"]
            worst, pid = worst_case_epsilon(run, sizes_for(run["config"]))
            if worst is None:
                continue
            key = str(target)
            if worst > seen.get(key, (0.0, None))[0]:
                seen[key] = (worst, pid)
        for key in sorted(seen, key=float):
            worst, pid = seen[key]
            print(f"  target eps {key:>4}   worst case {worst:.4f}   "
                  f"({100 * (worst / float(key) - 1):+.0f}%, client {pid})")
        print("\nThis is the number a claimed upper bound would have to use.")

    print()
    if breaches:
        print("BREACH: the reported epsilon is not an upper bound for these runs.")
        for name, target, worst, pid in breaches:
            print(f"  {name}: client {pid} spent {worst:.4f} against "
                  f"{target} claimed ({100 * (worst / target - 1):+.1f}%)")
        print("\nThe paper must either report the worst realised epsilon per")
        print("condition instead of the target, or calibrate sigma against a")
        print("participation bound rather than the expectation.")
        raise SystemExit(1)
    print("PASS: every client's spent epsilon stays at or under the claim.")


if __name__ == "__main__":
    main()
