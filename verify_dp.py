"""Phase-0 checks on the DP machinery. Run before trusting a sweep.

    python verify_dp.py

These are the claims the paper makes about its own method, so they are
checked in code rather than asserted in prose. Runs on CPU in seconds; no
GPU, no dataset, no Flower.

  0.1  sigma responds to epsilon (tighter budget -> more noise).
  0.2  step counts are deterministic and identical across epsilon, so a
       per-round energy difference cannot be a workload difference.
  0.3  the accountant is calibrated at the rate Opacus actually samples at,
       and the epsilon spent never exceeds the epsilon claimed.
"""

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, Dataset

from energyfl import dp

BATCH, ROUNDS, FRAC, EPOCHS = 32, 30, 0.5, 1
# Spread of Dirichlet(0.5) partition sizes seen with 20 clients on CIFAR-10.
SIZES = (400, 800, 1600, 2000, 2400, 4000)
EPSILONS = (0.5, 1, 2, 4, 8)


class _Toy(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(3, 32, 32), i % 10


def check_sigma_responds():
    print("[0.1] sigma vs epsilon")
    ok = True
    for n in (800, 2000):
        sig = [dp.noise_multiplier_for(e, n, BATCH, ROUNDS, FRAC, EPOCHS) for e in EPSILONS]
        print(f"   n={n:<5} " + "  ".join(f"eps={e}:{s:.3f}" for e, s in zip(EPSILONS, sig)))
        if not all(a > b for a, b in zip(sig, sig[1:])):
            print("      FAIL: sigma is not decreasing in epsilon")
            ok = False
    lo = dp.noise_multiplier_for(0.5, 2000, BATCH, ROUNDS, FRAC, EPOCHS)
    hi = dp.noise_multiplier_for(8, 2000, BATCH, ROUNDS, FRAC, EPOCHS)
    print(f"   sigma(0.5)/sigma(8) = {lo / hi:.2f}x")
    print(f"   sigma range over realistic partitions, eps=0.5: "
          f"{min(dp.noise_multiplier_for(0.5,n,BATCH,ROUNDS,FRAC,EPOCHS) for n in SIZES):.3f}"
          f" - {max(dp.noise_multiplier_for(0.5,n,BATCH,ROUNDS,FRAC,EPOCHS) for n in SIZES):.3f}"
          "   <- report this range in the paper")
    return ok


def check_steps_deterministic():
    print("\n[0.2] step counts: DP vs non-private, and across epsilon")
    ok = True
    print(f"   {'n':>6} {'cap ceil(n/B)':>14} {'non-DP steps':>13} {'DP steps':>9} {'examples':>10}")
    for n in SIZES:
        loader = DataLoader(_Toy(n), batch_size=BATCH, shuffle=True, drop_last=False)
        cap = dp.steps_per_epoch(n, BATCH)
        nonprivate = len(loader)
        taken = None
        for eps in (0.5, 8):  # step count must not depend on epsilon
            torch.manual_seed(0)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            sigma = dp.noise_multiplier_for(eps, n, BATCH, ROUNDS, FRAC, EPOCHS)
            _, _, dpl = PrivacyEngine(accountant="rdp").make_private(
                module=model, optimizer=opt,
                data_loader=DataLoader(_Toy(n), batch_size=BATCH, shuffle=True),
                noise_multiplier=sigma, max_grad_norm=dp.CLIPPING_NORM,
                poisson_sampling=True)
            seen, t = 0, 0
            for batch in dpl:
                if t >= cap:
                    break
                seen += len(batch[1])
                t += 1
            if taken is not None and t != taken:
                print(f"      FAIL: n={n} step count changed with epsilon")
                ok = False
            taken = t
        print(f"   {n:>6} {cap:>14} {nonprivate:>13} {taken:>9} {seen:>10}")
        if taken != nonprivate:
            print(f"      FAIL: DP took {taken} steps, non-private takes {nonprivate}")
            ok = False
    print("   note: the cap never binds -- Opacus already yields exactly")
    print("   ceil(n/B) batches. Poisson randomises batch SIZE (~+/-2% of n),")
    print("   not step count, and that residual is unbiased in epsilon.")
    return ok


def check_accounting_is_tight():
    print("\n[0.3] epsilon actually spent vs epsilon claimed")
    ok = True
    print(f"   {'n':>6} {'eps':>5} {'sigma':>8} {'steps':>7} {'eps spent':>10}")
    for n in (800, 2000):
        q = 1.0 / dp.steps_per_epoch(n, BATCH)          # what DPDataLoader uses
        assert abs(DPDataLoader.from_data_loader(
            DataLoader(_Toy(n), batch_size=BATCH)).sample_rate - q) < 1e-12
        steps = dp.total_local_steps(n, BATCH, ROUNDS, FRAC, EPOCHS)
        for eps in EPSILONS:
            sigma = dp.noise_multiplier_for(eps, n, BATCH, ROUNDS, FRAC, EPOCHS)
            acc = create_accountant("rdp")
            acc.history = [(sigma, q, steps)]
            spent = acc.get_epsilon(delta=dp.TARGET_DELTA)
            print(f"   {n:>6} {eps:>5} {sigma:>8.4f} {steps:>7} {spent:>10.4f}")
            if spent > eps + 1e-9:
                print(f"      FAIL: spent {spent:.4f} > claimed {eps}")
                ok = False
    print("   epsilon spent stays at or under the claim, as required for the")
    print("   'reported epsilon is an upper bound' statement.")
    return ok


if __name__ == "__main__":
    results = [check_sigma_responds(), check_steps_deterministic(),
               check_accounting_is_tight()]
    print("\n" + ("ALL CHECKS PASS" if all(results) else "CHECKS FAILED"))
    raise SystemExit(0 if all(results) else 1)
