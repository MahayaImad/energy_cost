"""Noise calibration and DP-SGD training.

We enforce client-level local DP with an untrusted server: each client holds
its own (epsilon, delta) budget over the whole federated run, accounted over
every local SGD step it takes in every round it joins -- not per round, and
not per epoch. For a client with n local training examples,

    total_steps = num_rounds * fraction_train * local_epochs * ceil(n / B)

The RDP accountant is inverted once, before training, to find the sigma that
spends exactly epsilon after total_steps. Sigma is then constant for the run.

Two things here are easy to get wrong and silent when you do.

Do not use PrivacyEngine's running accountant. In simulation each client is
rebuilt every round, so an engine-internal accountant resets every round and
reports a per-round epsilon while the real budget compounds across rounds --
wrong by roughly the round count.

Do not claim amplification by subsampling. It needs secure aggregation or a
shuffler, neither of which we assume, so the reported epsilon is an upper
bound.

Clipping norm C is fixed across all epsilon; tuning it per budget would
confound the privacy-utility comparison.
"""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

# Fixed across the entire sweep. Report these values.
CLIPPING_NORM = 1.0
TARGET_DELTA = 1e-5
ACCOUNTANT = "rdp"

# Sigmas derived ahead of the sweep by prewarm_sigma.py, read-only here.
SIGMA_CACHE_PATH = Path(os.environ.get("FL_SIGMA_CACHE", ".sigma_cache.json"))
_disk_cache: Optional[dict] = None


def is_private(epsilon) -> bool:
    """'inf' (or None) selects the non-private baseline."""
    if epsilon is None:
        return False
    if isinstance(epsilon, str):
        return epsilon.strip().lower() not in ("inf", "infinity", "none", "")
    return math.isfinite(float(epsilon))


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    return max(1, math.ceil(n_examples / batch_size))


def total_local_steps(n_examples, batch_size, num_rounds, fraction_train,
                      local_epochs) -> int:
    """Expected number of local SGD steps over the whole federated run."""
    expected_rounds = max(1.0, num_rounds * fraction_train)
    return max(1, int(round(
        expected_rounds * local_epochs * steps_per_epoch(n_examples, batch_size)
    )))


def sigma_cache_key(epsilon, n_examples, batch_size, num_rounds,
                    fraction_train, local_epochs, delta) -> str:
    """Stable key. Must match between prewarm and run, exactly."""
    return "|".join(str(x) for x in (
        float(epsilon), int(n_examples), int(batch_size), int(num_rounds),
        float(fraction_train), int(local_epochs), float(delta),
    ))


def _disk_cached(key: str) -> Optional[float]:
    global _disk_cache
    if _disk_cache is None:
        try:
            _disk_cache = json.loads(SIGMA_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _disk_cache = {}
    hit = _disk_cache.get(key)
    return None if hit is None else float(hit)


@lru_cache(maxsize=4096)
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

    Partitions differ in size, so sigma differs per client. That is correct --
    each client budgets over its own data -- but the range has to be reported.

    The caching is load-bearing for the energy measurement, not a speed-up.
    get_noise_multiplier binary-searches the accountant, and the search costs
    far more for a loose budget than a tight one: 0.086 s at epsilon=0.5
    against 0.693 s at epsilon=8. Called once per client per round inside the
    measured window, that alone reproduced an entire wall-clock trend across
    epsilon while step counts stayed flat -- an accountant artefact shaped
    exactly like a cost of privacy. In-process memoisation is not enough,
    because a client's first call still lands inside a measured round; hence
    the disk cache, filled by prewarm_sigma.py before anything is measured.
    """
    key = sigma_cache_key(epsilon, n_examples, batch_size, num_rounds,
                          fraction_train, local_epochs, delta)
    hit = _disk_cached(key)
    if hit is not None:
        return hit

    # Opacus does NOT sample at B/n. DPDataLoader sets sample_rate =
    # 1/len(loader) = 1/ceil(n/B), so calibrating at B/n over-states q by 1-3%
    # and yields an accidentally conservative sigma. Matching the sampler
    # exactly makes the accounting tight rather than merely safe.
    return get_noise_multiplier(
        target_epsilon=float(epsilon),
        target_delta=delta,
        sample_rate=1.0 / steps_per_epoch(n_examples, batch_size),
        steps=total_local_steps(n_examples, batch_size, num_rounds,
                                fraction_train, local_epochs),
        accountant=ACCOUNTANT,
    )


def validate_model(model: torch.nn.Module) -> torch.nn.Module:
    """Opacus rejects BatchNorm, which mixes information across samples.

    The models in task.py are already compatible; this guards future edits.
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
    """One round of local DP-SGD. Returns (avg_loss, clean_state_dict, steps).

    Each epoch is capped at ceil(n/B) steps, the same count the accountant was
    calibrated against, so compute cannot drift between epsilon conditions --
    which would make the energy axis measure workload size instead of privacy.
    In practice the cap never binds: Opacus already yields exactly that many
    batches (verify_dp.py check 0.2). Poisson sampling randomises batch SIZE,
    not step count, and that residual is unbiased in epsilon.

    The returned state_dict must be unwrapped. Opacus wraps the model in a
    GradSampleModule, which prefixes every key with '_module.'; hand those
    keys back to the server and FedAvg matches nothing, the global model never
    updates, and the run still looks completely healthy.
    """
    n_examples = len(trainloader.dataset)
    batch_size = trainloader.batch_size or 32
    cap = steps_per_epoch(n_examples, batch_size)

    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model, optimizer, dp_loader = PrivacyEngine(accountant=ACCOUNTANT).make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,  # required for the sample_rate accounting above
    )

    running, nsteps = 0.0, 0
    for _ in range(epochs):
        for taken, batch in enumerate(dp_loader):
            if taken >= cap:
                break
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            nsteps += 1

    clean = {k.replace("_module.", ""): v for k, v in model.state_dict().items()}
    return running / max(nsteps, 1), clean, nsteps
