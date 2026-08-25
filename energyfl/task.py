"""Model, data partitioning, and train/eval functions."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import Compose, Normalize, ToTensor

_testloader = None
_har_testloader = None

# ---------------------------------------------------------------- seeding


def set_seed(seed: int) -> None:
    """Seed every RNG that touches the run.

    Note: full bitwise determinism on GPU would require
    torch.use_deterministic_algorithms(True), which changes kernel selection
    and therefore changes measured energy. We deliberately do NOT enable it --
    we want the fast kernels a real deployment would use. Reproducibility
    across runs comes from averaging over seeds instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ model


class Net(nn.Module):
    """Small CNN for CIFAR-10 (from the PyTorch 60-minute blitz)."""

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
    """1D CNN over raw inertial channels for UCI-HAR (9 x 128 -> 6 classes).

    No BatchNorm: Opacus rejects it, since it mixes information across the
    samples in a batch and breaks the per-sample gradient guarantee. Global
    average pooling keeps the parameter count near the CIFAR model's (46k vs
    62k), so per-round energy stays comparable between the two datasets and
    a cross-dataset energy comparison is not dominated by model size.
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
    """Model for a dataset name. Keep every call site going through this."""
    d = str(dataset).strip().lower()
    if d in ("har", "uci-har", "uci_har"):
        return HARNet()
    if d in ("cifar10", "cifar-10", "cifar"):
        return Net()
    raise ValueError(f"unknown dataset {dataset!r}; expected 'cifar10' or 'har'")


# ------------------------------------------------------------------- data

DATASET = "uoft-cs/cifar10"
_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

_fds = None  # cached across clients in the same Ray worker


def _apply_transforms(batch):
    batch["img"] = [_TRANSFORMS(img) for img in batch["img"]]
    return batch


# ------------------------------------------------------------------ UCI-HAR

_har = None
HAR_PATH = Path(os.environ.get("FL_HAR_NPZ", "data/har.npz"))


class _ArrayDataset(TorchDataset):
    """Tensors served under the same keys the CIFAR pipeline uses.

    The key stays "img" although these are sensor windows: train_fn, test_fn
    and train_private all index batches by that name, and renaming it here
    would fork the training path per dataset for no benefit.
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
            "s_test": z["subject_test"],
        }
    return _har


def _time_split(idx, frac: float):
    """Split indices in file order, dropping one window at the boundary.

    HAR windows overlap by 50%, so two adjacent windows share half their
    samples. A random split would put those halves on both sides of the
    train/test line and inflate accuracy. Splitting in order and discarding
    the straddling window removes the shared samples.
    """
    cut = max(1, int(len(idx) * frac))
    return idx[: cut - 1], idx[cut:]


def _har_partition(partition_id: int, num_partitions: int, batch_size: int,
                   split: str):
    """One client = one study participant.

    'official' keeps the published subject-disjoint split: the 21 training
    participants become clients and the 9 test participants are held out
    entirely, so central accuracy measures generalisation to people the
    federation never saw. 'all' makes clients of all 30 participants and
    holds out a time-ordered tail of each one instead, which trades that
    guarantee for more clients.
    """
    h = _load_har()
    if split == "official":
        subs = np.unique(h["s_train"])
        X, y, sub = h["X_train"], h["y_train"], h["s_train"]
    else:
        subs = np.unique(np.concatenate([h["s_train"], h["s_test"]]))
        X = np.concatenate([h["X_train"], h["X_test"]])
        y = np.concatenate([h["y_train"], h["y_test"]])
        sub = np.concatenate([h["s_train"], h["s_test"]])

    if num_partitions != len(subs):
        raise SystemExit(
            f"HAR split '{split}' has {len(subs)} participants but the "
            f"federation was given {num_partitions} supernodes. Set both "
            f"num-supernodes and num_supernodes to {len(subs)}."
        )

    own = np.where(sub == subs[partition_id])[0]
    if split == "all":
        own, _ = _time_split(own, 0.8)      # tail goes to the central test set
    tr, va = _time_split(own, 0.8)

    trainloader = DataLoader(
        _ArrayDataset(X[tr], y[tr]), batch_size=batch_size,
        shuffle=True, drop_last=False,
    )
    valloader = DataLoader(_ArrayDataset(X[va], y[va]), batch_size=batch_size)
    return trainloader, valloader


def _har_testset(batch_size: int, split: str):
    h = _load_har()
    if split == "official":
        return DataLoader(
            _ArrayDataset(h["X_test"], h["y_test"]), batch_size=batch_size
        )
    X = np.concatenate([h["X_train"], h["X_test"]])
    y = np.concatenate([h["y_train"], h["y_test"]])
    sub = np.concatenate([h["s_train"], h["s_test"]])
    keep = []
    for sid in np.unique(sub):
        own = np.where(sub == sid)[0]
        _, tail = _time_split(own, 0.8)
        keep.append(tail)
    keep = np.concatenate(keep)
    return DataLoader(_ArrayDataset(X[keep], y[keep]), batch_size=batch_size)


# --------------------------------------------------------------- dispatch


def load_partition(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float,
    seed: int,
    dataset: str = "cifar10",
    har_split: str = "official",
):
    """Dirichlet (non-IID) partition -> local train/val dataloaders.

    alpha controls heterogeneity: lower = more skewed label distributions.
    Report the value you used; alpha=0.5 is the common non-IID setting.
    """
    if str(dataset).strip().lower() in ("har", "uci-har", "uci_har"):
        return _har_partition(partition_id, num_partitions, batch_size, har_split)

    # Imported lazily so a HAR run needs neither flwr-datasets nor the
    # vision extra: the two datasets share no loading machinery.
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    global _fds
    if _fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            seed=seed,
            min_partition_size=10,
            self_balancing=False,
        )
        _fds = FederatedDataset(
            dataset=DATASET,
            partitioners={"train": partitioner},
        )

    partition = _fds.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=seed)
    split = split.with_transform(_apply_transforms)

    trainloader = DataLoader(
        split["train"], batch_size=batch_size, shuffle=True, drop_last=False
    )
    valloader = DataLoader(split["test"], batch_size=batch_size)
    return trainloader, valloader


def load_centralized_testset(batch_size: int = 256, dataset: str = "cifar10",
                            har_split: str = "official"):
    """Held-out CIFAR-10 test split for server-side evaluation.

    Loaded straight from the hub: the ServerApp runs in a separate process
    from the ClientApps and must not depend on the partitioner state.
    Cached because global_evaluate is called every round.
    """
    if str(dataset).strip().lower() in ("har", "uci-har", "uci_har"):
        global _har_testloader
        if _har_testloader is None:
            _har_testloader = _har_testset(batch_size, har_split)
        return _har_testloader

    from datasets import load_dataset

    global _testloader
    if _testloader is None:
        testset = load_dataset(DATASET, split="test").with_transform(_apply_transforms)
        _testloader = DataLoader(testset, batch_size=batch_size)
    return _testloader


# --------------------------------------------------------- train / evaluate


def train_fn(net, trainloader, epochs: int, lr: float, device) -> float:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
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
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
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
