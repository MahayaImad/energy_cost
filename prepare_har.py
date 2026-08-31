"""Convert the UCI HAR Dataset into a single validated .npz.

    python prepare_har.py --root "/path/to/UCI HAR Dataset"

The archive is not downloadable from every environment, so fetch it once by
hand and point this at the extracted directory:

    https://archive.ics.uci.edu/static/public/240/
        human+activity+recognition+using+smartphones.zip

We use the RAW INERTIAL SIGNALS (9 channels x 128 samples per window), not
the 561 engineered features: a 1D CNN over raw sensor channels is the model
an edge deployment would actually run, and the engineered features would make
the workload unrepresentative of the setting the paper is about.

Every shape and count is checked against the published dataset. The loader
fails loudly on a mismatch rather than silently training on the wrong thing,
because a misparsed sensor file still produces plausible-looking arrays.
"""

import argparse
from pathlib import Path

import numpy as np

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
# Published counts. A mismatch means a different or truncated archive.
EXPECT = {"train": 7352, "test": 2947}
WINDOW = 128
N_CLASSES = 6
N_SUBJECTS = 30


def _load_split(root: Path, split: str):
    sig_dir = root / split / "Inertial Signals"
    if not sig_dir.is_dir():
        raise SystemExit(
            f"missing {sig_dir}\n"
            "Point --root at the extracted 'UCI HAR Dataset' directory "
            "(the one containing train/, test/ and activity_labels.txt)."
        )
    chans = []
    for s in SIGNALS:
        f = sig_dir / f"{s}_{split}.txt"
        if not f.is_file():
            raise SystemExit(f"missing signal file {f}")
        a = np.loadtxt(f, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != WINDOW:
            raise SystemExit(
                f"{f.name}: expected (N, {WINDOW}), got {a.shape}"
            )
        chans.append(a)
    X = np.stack(chans, axis=1)                       # (N, 9, 128)

    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subj = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)

    n = EXPECT[split]
    for name, arr, want in (("X", X, (n, 9, WINDOW)), ("y", y, (n,)),
                            ("subject", subj, (n,))):
        if arr.shape != want:
            raise SystemExit(
                f"{split}/{name}: expected {want}, got {arr.shape}. "
                "This is not the published UCI HAR Dataset."
            )
    if y.min() < 0 or y.max() != N_CLASSES - 1:
        raise SystemExit(f"{split}: labels out of range [0, {N_CLASSES-1}]")
    return X, y, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/har.npz"))
    a = ap.parse_args()

    Xtr, ytr, str_ = _load_split(a.root, "train")
    Xte, yte, ste = _load_split(a.root, "test")

    subjects = np.unique(np.concatenate([str_, ste]))
    if len(subjects) != N_SUBJECTS:
        raise SystemExit(f"expected {N_SUBJECTS} subjects, found {len(subjects)}")
    if set(np.unique(str_)) & set(np.unique(ste)):
        raise SystemExit("train and test subjects overlap; archive is not standard")

    # Per-channel standardisation constants from the TRAIN split only, so no
    # test statistic leaks in. Note in the paper that these are global across
    # clients: strictly, a federated deployment would have to agree them
    # without pooling raw data, and we treat them as a fixed preprocessing
    # constant rather than a learned quantity.
    mean = Xtr.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (Xtr.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        X_train=Xtr, y_train=ytr, subject_train=str_,
        X_test=Xte, y_test=yte, subject_test=ste,
        mean=mean, std=std,
    )

    print(f"wrote {a.out}")
    print(f"  train {Xtr.shape}  subjects {sorted(np.unique(str_).tolist())}")
    print(f"  test  {Xte.shape}  subjects {sorted(np.unique(ste).tolist())}")
    print(f"  classes {N_CLASSES}, window {WINDOW}, channels {len(SIGNALS)}")
    print("\n  windows per training subject "
          "(these become client partition sizes):")
    cnt = [int((str_ == s).sum()) for s in np.unique(str_)]
    print(f"    n = {min(cnt)}..{max(cnt)}, mean {sum(cnt)/len(cnt):.0f}")
    print("  Re-derive sigma for these sizes:\n"
          "    python prewarm_sigma.py --dataset har --rounds <num-server-rounds>")

    # Class coverage per client. UCI-HAR records the six activities in blocks,
    # so any partitioning that cuts a participant's recording by position
    # rather than within each class leaves clients training on a subset of the
    # activities. That is silent: training runs, energy looks normal, and only
    # the gap between client-side and central accuracy gives it away.
    try:
        import os
        os.environ.setdefault("FL_HAR_NPZ", str(a.out))
        from energyfl.task import load_partition

        print("\n  class coverage per client (all six expected on both sides):")
        bad = 0
        n_clients = len(np.unique(str_))
        for pid in range(n_clients):
            tl, vl = load_partition(pid, n_clients, 32, 0.5, 0, dataset="har")
            ctr = set(tl.dataset.y.tolist())
            cva = set(vl.dataset.y.tolist())
            if len(ctr) < N_CLASSES or len(cva) < N_CLASSES:
                bad += 1
                print(f"    client {pid}: train {sorted(ctr)} val {sorted(cva)}  <-- INCOMPLETE")
        print(f"    {n_clients - bad}/{n_clients} clients hold all {N_CLASSES} "
              "activities in both train and validation")
    except Exception as exc:
        print(f"\n  [skipped class-coverage check: {exc}]")


if __name__ == "__main__":
    main()
