"""Derive every sigma a sweep will need, before any measurement starts.

    python prewarm_sigma.py --dataset har --rounds 100
    python prewarm_sigma.py --rounds 30 --ablation

Why this exists: get_noise_multiplier binary-searches the RDP accountant, and
the search costs far more for a loose budget than a tight one (0.086 s at
epsilon=0.5 against 0.693 s at epsilon=8). Run inside the federated loop that
cost lands in the energy-measurement window and scales with epsilon -- an
artefact shaped exactly like a cost of privacy, which in one sweep accounted
for the entire observed trend.

sigma depends only on (epsilon, n, batch size, rounds, fraction, epochs,
delta), all known in advance, so deriving it here makes every run-time lookup
a dict hit. Writes .sigma_cache.json, which energyfl.dp reads and never
writes.

Partition sizes are reproduced exactly as the ClientApp sees them. If the
partitioning changes, delete the cache and rerun -- a stale entry would
silently apply the wrong noise.
"""

import argparse
import json
from pathlib import Path

from energyfl import dp

EPSILONS = (0.5, 1, 2, 4, 8)
SEEDS = (0, 1, 2)
BATCH_SIZE = 32
FRACTION_TRAIN = 0.5

# One entry per (clients, alpha, local_epochs) the sweep will visit. HAR has
# 21 clients by construction -- one per training participant -- and no
# Dirichlet alpha, because the partition comes from who contributed the data.
BASE_COMBOS = {"cifar10": [(20, 0.5, 1)], "har": [(21, 0.5, 1)]}
ABLATION_COMBOS = {
    "cifar10": [(20, 0.5, 2), (20, 0.5, 5), (20, 0.1, 1), (20, 10.0, 1),
                (10, 0.5, 1), (50, 0.5, 1)],
    "har": [(21, 0.5, 2), (21, 0.5, 5)],
}


def cifar_partition_sizes(num_partitions: int, alpha: float, seed: int):
    """Local TRAIN set size per client, mirroring task.load_partition."""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha,
            seed=seed, min_partition_size=10, self_balancing=False,
        )},
    )
    return [
        len(fds.load_partition(pid).train_test_split(test_size=0.2, seed=seed)["train"])
        for pid in range(num_partitions)
    ]


def har_partition_sizes(num_partitions: int):
    """Same, via the loader the clients use, so the sizes cannot drift apart.

    The subject partition is deterministic, so unlike CIFAR it does not depend
    on the seed and one pass covers every seed.
    """
    from energyfl.task import load_partition

    return [
        len(load_partition(pid, num_partitions, BATCH_SIZE, 0.5, 0,
                           dataset="har")[0].dataset)
        for pid in range(num_partitions)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "har"])
    ap.add_argument("--rounds", type=int, required=True,
                    help="must match num-server-rounds of the sweep: rounds "
                         "enter the cache key, so a mismatch misses every "
                         "lookup and puts the accountant back in the window")
    ap.add_argument("--ablation", action="store_true",
                    help="also cover the ablation grid")
    ap.add_argument("--out", type=Path, default=dp.SIGMA_CACHE_PATH)
    args = ap.parse_args()

    combos = list(BASE_COMBOS[args.dataset])
    if args.ablation:
        combos += ABLATION_COMBOS[args.dataset]

    cache = {}
    if args.out.exists():
        try:
            cache = json.loads(args.out.read_text())
            print(f"extending existing cache ({len(cache)} entries)")
        except ValueError:
            pass

    seeds = SEEDS if args.dataset == "cifar10" else (0,)
    for clients, alpha, epochs in combos:
        for seed in seeds:
            if args.dataset == "har":
                sizes = har_partition_sizes(clients)
                label = f"clients={clients} epochs={epochs}"
            else:
                sizes = cifar_partition_sizes(clients, alpha, seed)
                label = (f"clients={clients} alpha={alpha} epochs={epochs} "
                         f"seed={seed}")
            print(f"\n{label} rounds={args.rounds}: n = {min(sizes)}..{max(sizes)}")

            for eps in EPSILONS:
                sigmas = []
                for n in sizes:
                    key = dp.sigma_cache_key(eps, n, BATCH_SIZE, args.rounds,
                                             FRACTION_TRAIN, epochs,
                                             dp.TARGET_DELTA)
                    if key not in cache:
                        cache[key] = dp.noise_multiplier_for(
                            epsilon=eps, n_examples=n, batch_size=BATCH_SIZE,
                            num_rounds=args.rounds,
                            fraction_train=FRACTION_TRAIN, local_epochs=epochs,
                        )
                    sigmas.append(cache[key])
                print(f"    eps={eps:<4} sigma {min(sigmas):.4f} - {max(sigmas):.4f}")

    args.out.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f"\nwrote {len(cache)} sigmas to {args.out}")
    print("The per-epsilon ranges above go in the paper's reproducibility table.")


if __name__ == "__main__":
    main()
