"""Models, data partitioning, and the train/evaluate loops.

Two datasets. CIFAR-10 is partitioned synthetically with a Dirichlet
partitioner; UCI-HAR is partitioned by study participant, so a client is a
real person rather than a slice of a pooled dataset.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import Compose, Normalize, ToTensor

CIFAR_HUB_ID = "uoft-cs/cifar10"
HAR_PATH = Path(os.environ.get("FL_HAR_NPZ", "data/har.npz"))

_HAR_NAMES = ("har", "uci-har", "uci_har")


def is_har(dataset: str) -> bool:
    return str(dataset).strip().lower() in _HAR_NAMES


def set_seed(seed: int) -> None:
    """Seed every RNG that touches the run.

    Deliberately NOT torch.use_deterministic_algorithms: it changes kernel
    selection and therefore changes measured energy, i.e. it would measure a
    configuration nobody deploys. Reproducibility comes from averaging over
    seeds instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------- models


class Net(nn.Module):
    """Small CNN for CIFAR-10, from the PyTorch 60-minute blitz. 62k params."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class HARNet(nn.Module):
    """1D CNN over raw inertial channels: 9 x 128 -> 6 activities. 37k params.

    No BatchNorm, because Opacus rejects it: it mixes information across the
    samples in a batch and breaks the per-sample gradient guarantee. Global
    average pooling keeps the parameter count within the same order as the
    CIFAR model's 62k, so a cross-dataset energy comparison is not dominated
    by model size.
    """

    def __init__(self, in_ch: int = 9, n_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=2)           # global average pool over time
        return self.head(x)


def build_model(dataset: str) -> nn.Module:
    """Model for a dataset name. Every call site goes through this."""
    if is_har(dataset):
        return HARNet()
    if str(dataset).strip().lower() in ("cifar10", "cifar-10", "cifar"):
        return Net()
    raise ValueError(f"unknown dataset {dataset!r}; expected 'cifar10' or 'har'")


# ------------------------------------------------------------------- UCI-HAR

_har = None
_har_testloader = None


class _ArrayDataset(TorchDataset):
    """Tensors served under the keys the CIFAR pipeline uses.

    The key stays "img" although these are sensor windows: train_fn, test_fn
    and train_private all index batches by that name, and renaming it would
    fork the training path per dataset for no benefit.
    """

    def __init__(self, X, y):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"img": self.X[i], "label": self.y[i]}


def _load_har():
    """Load and standardise the prepared HAR arrays once per process."""
    global _har
    if _har is None:
        if not HAR_PATH.is_file():
            raise SystemExit(
                f"{HAR_PATH} not found. Build it once with:\n"
                '    python prepare_har.py --root "/path/to/UCI HAR Dataset"'
            )
        z = np.load(HAR_PATH)
        mean, std = z["mean"], z["std"]
        _har = {
            "X_train": ((z["X_train"] - mean) / std).astype(np.float32),
            "y_train": z["y_train"],
            "s_train": z["subject_train"],
            "X_test": ((z["X_test"] - mean) / std).astype(np.float32),
            "y_test": z["y_test"],
        }
    return _har


def _split_within_classes(idx, labels, frac: float):
    """Split one participant's windows into (train, val), class by class.

    Two constraints pull against each other. HAR windows overlap by 50%, so a
    random split puts the two halves of one recording on both sides of the
    train/val line; splitting in file order and dropping the straddling window
    fixes that. But a participant records the six activities in BLOCKS, so a
    single ordered cut across the whole recording gives the early activities
    to one side and the late ones to the other.

    That is not hypothetical -- it is what the first HAR runs did. Clients
    scored 0.268 on their own held-out data against 0.478 on nine unseen
    participants: local validation doing 21 points WORSE than generalisation
    to strangers, which only happens if the two sides hold different classes.

    Cutting inside each class satisfies both: every side gets all six
    activities, and the cut is still ordered, so overlapping neighbours stay
    together apart from the one window dropped at each class boundary.
    """
    train, val = [], []
    for c in np.unique(labels[idx]):
        own = idx[labels[idx] == c]
        cut = max(1, int(len(own) * frac))
        train.append(own[: cut - 1])
        val.append(own[cut:])
    return np.concatenate(train), np.concatenate(val)


def _har_partition(partition_id: int, num_partitions: int, batch_size: int):
    """One client = one study participant.

    Uses the published subject-disjoint split: the 21 training participants
    become clients, and the 9 test participants are held out entirely, so
    central accuracy measures generalisation to people the federation never
    saw.
    """
    h = _load_har()
    subjects = np.unique(h["s_train"])
    if num_partitions != len(subjects):
        raise SystemExit(
            f"HAR has {len(subjects)} training participants but the federation "
            f"was given {num_partitions} supernodes. Set both num-supernodes "
            f"and num_supernodes to {len(subjects)}."
        )

    own = np.where(h["s_train"] == subjects[partition_id])[0]
    tr, va = _split_within_classes(own, h["y_train"], 0.8)
    X, y = h["X_train"], h["y_train"]

    return (
        DataLoader(_ArrayDataset(X[tr], y[tr]), batch_size=batch_size,
                   shuffle=True, drop_last=False),
        DataLoader(_ArrayDataset(X[va], y[va]), batch_size=batch_size),
    )


# -------------------------------------------------------------- CIFAR-10 data

_fds = None          # cached across clients in the same Ray worker
_testloader = None
_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _apply_transforms(batch):
    batch["img"] = [_TRANSFORMS(img) for img in batch["img"]]
    return batch


# ------------------------------------------------------------------ dispatch


def load_partition(partition_id: int, num_partitions: int, batch_size: int,
                   alpha: float, seed: int, dataset: str = "cifar10"):
    """Local train/val dataloaders for one client.

    HAR partitions by participant. CIFAR-10 partitions by Dirichlet(alpha),
    where lower alpha means more skewed label distributions; alpha=0.5 is the
    common non-IID setting.
    """
    if is_har(dataset):
        return _har_partition(partition_id, num_partitions, batch_size)

    # Imported lazily so a HAR run needs neither flwr-datasets nor its vision
    # extra: the two datasets share no loading machinery.
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    global _fds
    if _fds is None:
        _fds = FederatedDataset(
            dataset=CIFAR_HUB_ID,
            partitioners={"train": DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                seed=seed,
                min_partition_size=10,
                self_balancing=False,
            )},
        )

    split = _fds.load_partition(partition_id)
    split = split.train_test_split(test_size=0.2, seed=seed)
    split = split.with_transform(_apply_transforms)
    return (
        DataLoader(split["train"], batch_size=batch_size, shuffle=True,
                   drop_last=False),
        DataLoader(split["test"], batch_size=batch_size),
    )


def load_centralized_testset(batch_size: int = 256, dataset: str = "cifar10"):
    """Held-out test split for server-side evaluation.

    Cached, because global_evaluate is called every round. Loaded
    independently of the partitioner, since the ServerApp runs in a different
    process from the ClientApps.
    """
    global _har_testloader, _testloader
    if is_har(dataset):
        if _har_testloader is None:
            h = _load_har()
            _har_testloader = DataLoader(
                _ArrayDataset(h["X_test"], h["y_test"]), batch_size=batch_size
            )
        return _har_testloader

    from datasets import load_dataset

    if _testloader is None:
        testset = load_dataset(CIFAR_HUB_ID, split="test")
        _testloader = DataLoader(
            testset.with_transform(_apply_transforms), batch_size=batch_size
        )
    return _testloader


# -------------------------------------------------------- train and evaluate


def train_fn(net, trainloader, epochs: int, lr: float, device) -> float:
    """Plain (non-private) local training. Returns mean loss per batch."""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running, nbatches = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            nbatches += 1
    return running / max(nbatches, 1)


def test_fn(net, loader, device):
    """Returns (mean loss per batch, accuracy)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, loss, n = 0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            n += labels.size(0)
    return loss / max(len(loader), 1), correct / max(n, 1)
