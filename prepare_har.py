"""Convert the UCI HAR Dataset into data/har.npz.

    python prepare_har.py --root "/path/to/UCI HAR Dataset"

Download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

Uses the raw inertial signals (9 channels x 128 samples per window) and the
published subject-disjoint split (21 training / 9 test participants).
Standardisation constants are per channel, from the training split.
"""

import argparse
from pathlib import Path

import numpy as np

SIGNALS = [f"{kind}_{axis}" for kind in ("body_acc", "body_gyro", "total_acc")
           for axis in "xyz"]


def load_split(root: Path, split: str):
    d = root / split
    X = np.stack([np.loadtxt(d / "Inertial Signals" / f"{s}_{split}.txt",
                             dtype=np.float32) for s in SIGNALS], axis=1)
    y = np.loadtxt(d / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(d / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/har.npz"))
    a = ap.parse_args()

    Xtr, ytr, str_ = load_split(a.root, "train")
    Xte, yte, ste = load_split(a.root, "test")
    assert Xtr.shape == (7352, 9, 128) and Xte.shape == (2947, 9, 128)
    assert len(np.unique(str_)) == 21 and len(np.unique(ste)) == 9

    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X_train=Xtr, y_train=ytr, subject_train=str_,
                        X_test=Xte, y_test=yte, subject_test=ste,
                        mean=mean, std=std)
    print(f"wrote {a.out}: train {Xtr.shape}, test {Xte.shape}")


if __name__ == "__main__":
    main()
