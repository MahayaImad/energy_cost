"""Worst realised epsilon per budget (Table I, 'Worst eps' column).

    python realised_epsilon.py            # needs data/har.npz

sigma is calibrated on the EXPECTED participation R * q_c, but the number of
rounds a client is actually sampled in is binomial. This replays the
participation log of every run (`sampled_partitions`) and recomputes each
client's spent epsilon; the worst client per budget is written to
results/realised_epsilon_har.json, which analyze.py reads.
"""

import json
from collections import Counter

from analyze import load
from energyfl import dp
from energyfl.task import load_partition


def main():
    worst = {}
    sizes = None
    for eps, group in load("results/har").items():
        if eps == "inf":
            continue
        for run in group:
            c = run["config"]
            if sizes is None:   # HAR partition is by participant: seed-free
                sizes = [len(load_partition(p, c["num_supernodes"],
                                            c["batch_size"], 0.5, 0,
                                            "har")[0].dataset)
                         for p in range(c["num_supernodes"])]
            joined = Counter(p for rd in run["rounds"]
                             for p in rd["sampled_partitions"])
            for pid, k in joined.items():
                n, B = sizes[pid], c["batch_size"]
                sigma = dp.noise_multiplier(eps, n, B, c["num_rounds"],
                                            c["fraction_train"],
                                            c["local_epochs"])
                steps = k * c["local_epochs"] * dp.steps_per_epoch(n, B)
                spent = dp.spent_epsilon(sigma, n, B, steps)
                worst[eps] = max(worst.get(eps, 0.0), round(spent, 4))
        print(f"eps {eps}: worst client spent {worst[eps]:.3f}")
    with open("results/realised_epsilon_har.json", "w") as f:
        json.dump(worst, f)


if __name__ == "__main__":
    main()
