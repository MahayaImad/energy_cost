"""Example-level local DP-SGD with whole-run accounting (Section III-B).

Each client clips and noises per-sample gradients (Opacus) and holds its own
(epsilon, delta) budget over the whole run. For a client with n examples:

    s_total = R * q_c * E * ceil(n / B)                         (Eq. 1)

The RDP accountant is inverted once, before training, to get the noise
multiplier sigma that spends epsilon after s_total steps.
"""

import math

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine

DELTA = 1e-5


def is_private(epsilon) -> bool:
    return str(epsilon).lower() != "inf"


def steps_per_epoch(n, batch_size):
    return math.ceil(n / batch_size)


def noise_multiplier(epsilon, n, batch_size, rounds, fraction, epochs):
    """sigma for one client. Opacus samples at rate 1/ceil(n/B), not B/n,
    so the accountant is calibrated at that rate."""
    from opacus.accountants.utils import get_noise_multiplier

    spe = steps_per_epoch(n, batch_size)
    steps = max(1, round(max(1.0, rounds * fraction) * epochs * spe))
    return get_noise_multiplier(target_epsilon=float(epsilon),
                                target_delta=DELTA, sample_rate=1.0 / spe,
                                steps=steps, accountant="rdp")


def spent_epsilon(sigma, n, batch_size, steps):
    """Epsilon actually spent after `steps` steps (for the Worst-eps column)."""
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    acc.history = [(sigma, 1.0 / steps_per_epoch(n, batch_size), steps)]
    return acc.get_epsilon(delta=DELTA)


def train_private(net, loader, epochs, lr, device, sigma, clip):
    """One round of local DP-SGD. Returns a state_dict without the
    '_module.' prefix Opacus adds, so FedAvg can aggregate it."""
    net.to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net, opt, loader = PrivacyEngine(accountant="rdp").make_private(
        module=net, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=clip, poisson_sampling=True)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x.to(device)), y.to(device)).backward()
            opt.step()
    return {k.replace("_module.", ""): v for k, v in net.state_dict().items()}
