"""Derive every sigma a sweep needs, before any measurement starts.

    python prewarm_sigma.py                  # main sweep
    python prewarm_sigma.py --ablation       # also the ablation grid

Why this exists. get_noise_multiplier binary-searches the RDP accountant, and
the search costs far more for a loose privacy budget than a tight one
(measured 0.086 s at epsilon=0.5 against 0.693 s at epsilon=8). Run inside
the federated loop, that cost lands in the energy-measurement window and
scales with epsilon -- an accountant artefact that looks exactly like a cost
of privacy. It produced the entire wall-clock trend across epsilon in one
sweep, and a 3 s residual in the next.

sigma depends only on (epsilon, n, batch size, rounds, fraction, epochs,
delta), all known in advance, so it can be derived here and looked up as a
dict hit at run time. Writes .sigma_cache.json, which energyfl.dp reads.

Partition sizes are reproduced exactly as the ClientApp sees them: the same
Dirichlet partitioner, the same seed, the same 0.2 train/test split. If any
of those change, delete the cache and rerun -- a stale entry would silently
apply the wrong noise.

Prints the per-client sigma range, which the paper has to report anyway.
"""

import argparse
import json
from pathlib import Path

from energyfl import dp

DATASET = "uoft-cs/cifar10"
EPSILONS = (0.5, 1, 2, 4, 8)
SEEDS = (0, 1, 2)
BATCH, ROUNDS, FRAC = 32, 30, 0.5


def partition_sizes(num_partitions: int, alpha: float, seed: int):
    """Local TRAIN set size per client -- what noise_multiplier_for is given.

    Mirrors task.load_partition: same partitioner arguments, same split.
    """
    from flwr_datasets import FederatedDataset          # CIFAR-only dependency
    from flwr_datasets.partitioner import DirichletPartitioner

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        seed=seed,
        min_partition_size=10,
        self_balancing=False,
    )
    fds = FederatedDataset(dataset=DATASET, partitioners={"train": partitioner})
    sizes = []
    for pid in range(num_partitions):
        part = fds.load_partition(pid)
        sizes.append(len(part.train_test_split(test_size=0.2, seed=seed)["train"]))
    return sizes


def har_partition_sizes(num_partitions: int, split: str):
    """Local train-set size per participant, exactly as the ClientApp sees it.

    Delegates to the same loader the clients use, so the sizes cannot drift
    apart from the ones sigma is calibrated against. The subject partition is
    deterministic, so unlike CIFAR it does not depend on the seed.
    """
    from energyfl.task import load_partition

    sizes = []
    for pid in range(num_partitions):
        tl, _ = load_partition(pid, num_partitions, BATCH, 0.5, 0,
                               dataset="har", har_split=split)
        sizes.append(len(tl.dataset))
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "har"])
    ap.add_argument("--har-split", default="official", choices=["official", "all"])
    ap.add_argument("--ablation", action="store_true",
                    help="also cover the ablation grid (alpha, clients, epochs)")
    ap.add_argument("--out", type=Path, default=dp.SIGMA_CACHE_PATH)
    a = ap.parse_args()

    # (num_partitions, alpha, local_epochs) combinations to cover.
    combos = [(20, 0.5, 1)]
    if a.dataset == "har":
        combos = [(21 if a.har_split == "official" else 30, 0.5, 1)]
    if a.ablation:
        extra = [(20, 0.5, 2), (20, 0.5, 5)]
        if a.dataset == "cifar10":
            extra += [(20, 0.1, 1), (20, 10.0, 1), (10, 0.5, 1), (50, 0.5, 1)]
        combos += [(c[0] if a.dataset == "cifar10" else combos[0][0], c[1], c[2])
                   for c in extra]

    cache = {}
    if a.out.exists():
        try:
            cache = json.loads(a.out.read_text())
            print(f"extending existing cache ({len(cache)} entries)")
        except ValueError:
            pass

    for clients, alpha, epochs in combos:
        # The HAR partition is by participant and therefore seed-independent,
        # so one pass covers every seed.
        seeds = SEEDS if a.dataset == "cifar10" else [0]
        for seed in seeds:
            if a.dataset == "har":
                sizes = har_partition_sizes(clients, a.har_split)
            else:
                sizes = partition_sizes(clients, alpha, seed)
            label = (f"clients={clients} split={a.har_split} epochs={epochs}"
                     if a.dataset == "har"
                     else f"clients={clients} alpha={alpha} epochs={epochs} seed={seed}")
            print(f"\n{label}: n = {min(sizes)}..{max(sizes)}")
            for eps in EPSILONS:
                sig = []
                for n in sizes:
                    key = dp.sigma_cache_key(
                        eps, n, BATCH, ROUNDS, FRAC, epochs, dp.TARGET_DELTA
                    )
                    if key not in cache:
                        cache[key] = dp.noise_multiplier_for(
                            epsilon=eps, n_examples=n, batch_size=BATCH,
                            num_rounds=ROUNDS, fraction_train=FRAC,
                            local_epochs=epochs,
                        )
                    sig.append(cache[key])
                print(f"    eps={eps:<4} sigma {min(sig):.4f} - {max(sig):.4f}")

    a.out.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f"\nwrote {len(cache)} sigmas to {a.out}")
    print("Report the per-epsilon sigma ranges above in the paper's "
          "reproducibility table.")


if __name__ == "__main__":
    main()
