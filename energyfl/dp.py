"""Differential privacy: noise calibration and DP-SGD training.

ACCOUNTING MODEL (state this verbatim in the paper's methodology)
----------------------------------------------------------------
We enforce *client-level local DP*: each client holds its own (epsilon,
delta) budget over the entire federated run, and the server is untrusted.

The budget is accounted over ALL local SGD steps the client takes across
ALL rounds in which it participates -- not per round, and not per local
epoch. This is the conservative choice and the one reviewers check first.

For a client with n local training examples:

    steps_per_epoch = ceil(n / batch_size)
    expected_rounds = num_rounds * fraction_train     # client sampling
    total_steps     = expected_rounds * local_epochs * steps_per_epoch

We invert the RDP accountant once, before training starts, to find the
noise multiplier sigma that yields the target epsilon at delta after
total_steps. Sigma is then held constant for the whole run.

Why precompute instead of using PrivacyEngine's running accountant: in
simulation each client is reconstructed every round, so an engine-internal
accountant resets each round and would silently report a per-round epsilon
while the true budget compounds across rounds -- wrong by roughly a factor
of the round count. That error is the most common reason these papers are
rejected.

We do NOT claim privacy amplification by client subsampling. Amplification
requires assumptions we do not make (secure aggregation or a shuffler), so
the reported epsilon is an upper bound. Say so explicitly in the paper.

FIXED STEP COUNT
----------------
Poisson sampling makes the number of steps per epoch a random variable,
which lets total compute drift between configurations. Because this study
measures energy, compute must be identical across every epsilon -- otherwise
the energy axis measures workload size rather than the cost of privacy.
train_private therefore caps each local epoch at a deterministic
steps_per_epoch = ceil(n / batch_size), and that same count is what the
accountant is calibrated against. Report the cap.

Clipping norm C is FIXED across all epsilon values. Tuning C per epsilon
would confound the privacy-utility comparison.
"""

import math
from typing import Optional, Tuple

import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

# Fixed across the entire sweep. Report these values.
CLIPPING_NORM = 1.0
TARGET_DELTA = 1e-5
ACCOUNTANT = "rdp"


def is_private(epsilon) -> bool:
    """'inf' (or None) selects the non-private baseline."""
    if epsilon is None:
        return False
    if isinstance(epsilon, str):
        return epsilon.strip().lower() not in ("inf", "infinity", "none", "")
    return math.isfinite(float(epsilon))


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    return max(1, math.ceil(n_examples / batch_size))


def total_local_steps(
    n_examples: int,
    batch_size: int,
    num_rounds: int,
    fraction_train: float,
    local_epochs: int,
) -> int:
    """Expected number of local SGD steps over the whole federated run."""
    expected_rounds = max(1.0, num_rounds * fraction_train)
    spe = steps_per_epoch(n_examples, batch_size)
    return max(1, int(round(expected_rounds * local_epochs * spe)))


def noise_multiplier_for(
    epsilon: float,
    n_examples: int,
    batch_size: int,
    num_rounds: int,
    fraction_train: float,
    local_epochs: int,
    delta: float = TARGET_DELTA,
) -> float:
    """Invert the RDP accountant: target epsilon -> sigma.

    Called once per client at setup. Because Dirichlet partitions differ in
    size, sigma differs per client -- correct (each client has its own budget
    over its own data), but the range must be reported.
    """
    sample_rate = min(1.0, batch_size / max(n_examples, 1))
    steps = total_local_steps(
        n_examples, batch_size, num_rounds, fraction_train, local_epochs
    )
    return get_noise_multiplier(
        target_epsilon=float(epsilon),
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant=ACCOUNTANT,
    )


def validate_model(model: torch.nn.Module) -> torch.nn.Module:
    """Opacus rejects BatchNorm (it mixes information across samples).

    The CNN in task.py is already compatible; this guards future edits.
    """
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
    return model


def train_private(
    model: torch.nn.Module,
    trainloader,
    epochs: int,
    lr: float,
    device: torch.device,
    noise_multiplier: float,
    max_grad_norm: float = CLIPPING_NORM,
) -> Tuple[float, dict, int]:
    """One round of local DP-SGD.

    Returns (avg_loss, clean_state_dict, n_steps).

    The state_dict must be unwrapped: Opacus wraps the model in a
    GradSampleModule, which prefixes every key with '_module.'. Returning the
    wrapped keys to the server silently breaks aggregation -- FedAvg finds no
    matching keys, the global model never updates, and everything appears to
    run fine. This is the most common Opacus + Flower integration bug.
    """
    n_examples = len(trainloader.dataset)
    batch_size = trainloader.batch_size or 32
    cap = steps_per_epoch(n_examples, batch_size)

    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    privacy_engine = PrivacyEngine(accountant=ACCOUNTANT)
    model, optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,  # required for the sample_rate accounting above
    )

    running, nsteps = 0.0, 0
    for _ in range(epochs):
        taken = 0
        for batch in dp_loader:
            if taken >= cap:  # deterministic compute across all epsilon
                break
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            taken += 1
            nsteps += 1

    clean = {k.replace("_module.", ""): v for k, v in model.state_dict().items()}
    return running / max(nsteps, 1), clean, nsteps