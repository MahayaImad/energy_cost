"""Model, data partitioning, and train/eval functions."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import load_dataset

_testloader = None

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


# ------------------------------------------------------------------- data

DATASET = "uoft-cs/cifar10"
_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

_fds = None  # cached across clients in the same Ray worker


def _apply_transforms(batch):
    batch["img"] = [_TRANSFORMS(img) for img in batch["img"]]
    return batch


def load_partition(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float,
    seed: int,
):
    """Dirichlet (non-IID) partition -> local train/val dataloaders.

    alpha controls heterogeneity: lower = more skewed label distributions.
    Report the value you used; alpha=0.5 is the common non-IID setting.
    """
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


def load_centralized_testset(batch_size: int = 256):
    """Held-out CIFAR-10 test split for server-side evaluation.

    Loaded straight from the hub: the ServerApp runs in a separate process
    from the ClientApps and must not depend on the partitioner state.
    Cached because global_evaluate is called every round.
    """
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
