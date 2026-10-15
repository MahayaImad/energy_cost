"""Models, data partitioning and the plain train/test loops.

UCI-HAR: one client per training participant (21), the 9 test participants
are the central test set. CIFAR-10: Dirichlet(alpha) label partition.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HAR_NPZ = "data/har.npz"
CIFAR_ID = "uoft-cs/cifar10"


def set_seed(seed: int) -> None:
    # No torch.use_deterministic_algorithms: it changes kernel selection and
    # therefore the measured energy. Reproducibility comes from three seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- models
# Neither model uses BatchNorm, which Opacus rejects.


class CifarNet(nn.Module):
    """Five-layer CNN, 62k parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class HARNet(nn.Module):
    """Three-block 1D CNN with global average pooling, 37k parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(9, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 6))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=2))


def build_model(dataset: str) -> nn.Module:
    return HARNet() if dataset == "har" else CifarNet()


# ------------------------------------------------------------------ data
# Every loader yields (x, y) batches.

_har = None
_cifar_fds = {}


def _load_har():
    global _har
    if _har is None:
        z = np.load(HAR_NPZ)
        norm = lambda X: ((X - z["mean"]) / z["std"]).astype(np.float32)
        _har = dict(X_train=norm(z["X_train"]), y_train=z["y_train"],
                    s_train=z["subject_train"],
                    X_test=norm(z["X_test"]), y_test=z["y_test"])
    return _har


def _split_within_classes(idx, labels, frac=0.8):
    """Ordered 80/20 split inside each activity, dropping the straddling
    window: windows overlap by 50% (a random split would leak) and the
    activities are recorded in blocks (one global cut would drop classes)."""
    train, val = [], []
    for c in np.unique(labels[idx]):
        own = idx[labels[idx] == c]
        cut = max(1, int(len(own) * frac))
        train.append(own[: cut - 1])
        val.append(own[cut:])
    return np.concatenate(train), np.concatenate(val)


def _tensors(X, y):
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                         torch.from_numpy(np.ascontiguousarray(y)))


def _cifar_transform(batch):
    from torchvision.transforms import Compose, Normalize, ToTensor
    tf = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])
    batch["img"] = [tf(img) for img in batch["img"]]
    return batch


def _collate(batch):
    return (torch.stack([b["img"] for b in batch]),
            torch.tensor([b["label"] for b in batch]))


def load_partition(pid, num_partitions, batch_size, alpha, seed, dataset):
    """(train_loader, val_loader) of client `pid`."""
    if dataset == "har":
        h = _load_har()
        subject = np.unique(h["s_train"])[pid]
        own = np.where(h["s_train"] == subject)[0]
        tr, va = _split_within_classes(own, h["y_train"])
        X, y = h["X_train"], h["y_train"]
        train, val = _tensors(X[tr], y[tr]), _tensors(X[va], y[va])
        return (DataLoader(train, batch_size=batch_size, shuffle=True),
                DataLoader(val, batch_size=batch_size))

    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    key = (num_partitions, alpha, seed)
    if key not in _cifar_fds:
        _cifar_fds[key] = FederatedDataset(
            dataset=CIFAR_ID,
            partitioners={"train": DirichletPartitioner(
                num_partitions=num_partitions, partition_by="label",
                alpha=alpha, seed=seed, min_partition_size=10,
                self_balancing=False)})
    split = _cifar_fds[key].load_partition(pid)
    split = split.train_test_split(test_size=0.2, seed=seed)
    split = split.with_transform(_cifar_transform)
    return (DataLoader(split["train"], batch_size=batch_size, shuffle=True,
                       collate_fn=_collate),
            DataLoader(split["test"], batch_size=batch_size,
                       collate_fn=_collate))


def load_testset(dataset, batch_size=256):
    """Central test set, evaluated by the server after every round."""
    if dataset == "har":
        h = _load_har()
        return DataLoader(_tensors(h["X_test"], h["y_test"]),
                          batch_size=batch_size)
    from datasets import load_dataset
    test = load_dataset(CIFAR_ID, split="test").with_transform(_cifar_transform)
    return DataLoader(test, batch_size=batch_size, collate_fn=_collate)


# ------------------------------------------------------- train / evaluate


def train(net, loader, epochs, lr, device):
    """Non-private local training (SGD, momentum 0.9)."""
    net.to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x.to(device)), y.to(device)).backward()
            opt.step()


def test(net, loader, device):
    """Returns (mean loss per batch, accuracy)."""
    net.to(device).eval()
    loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss += F.cross_entropy(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss / len(loader), correct / n
