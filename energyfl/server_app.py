"""ServerApp: FedAvg with per-round GPU energy measurement and JSON logging.

Every run writes one self-describing JSON file containing the full config
plus a per-round record of accuracy, loss, energy, wall-clock, and the
aggregated client-side training metrics (noise multiplier, local step
count). That file is the unit of analysis for the paper -- the plotting
scripts should read these and nothing else.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from energyfl import energy
from energyfl.task import (
    Net,
    get_device,
    load_centralized_testset,
    set_seed,
    test_fn,
)

app = ServerApp()

RESULTS_DIR = Path(os.environ.get("FL_RESULTS_DIR", "results"))


class EnergyFedAvg(FedAvg):
    """FedAvg that brackets each round with an NVML energy reading.

    configure_train marks the start of a round; aggregate_evaluate marks the
    end. The measured window therefore covers client training, client
    evaluation, and aggregation -- i.e. everything the round costs on this
    GPU.

    aggregate_train is overridden only to capture the client-side metrics
    (noise_multiplier, local_steps, train_loss) so they land in the JSON.
    Without this they are aggregated by the base strategy and then lost.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds: list[dict] = []
        self._e0: Optional[int] = None
        self._t0: Optional[float] = None
        self._round: int = 0
        self._train_metrics: dict = {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._round = server_round
        self._e0 = energy.energy_mj()
        self._t0 = time.perf_counter()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies):
        out = super().aggregate_train(server_round, replies)
        try:
            # FedAvg returns (ArrayRecord, MetricRecord)
            metrics = out[1] if isinstance(out, tuple) else out
            self._train_metrics = {k: float(v) for k, v in dict(metrics).items()}
        except Exception as exc:  # never let logging break a run
            print(f"[warn] could not capture train metrics: {exc}")
            self._train_metrics = {}
        return out

    def aggregate_evaluate(self, server_round: int, replies):
        out = super().aggregate_evaluate(server_round, replies)
        joules = (energy.energy_mj() - self._e0) / 1000.0
        seconds = time.perf_counter() - self._t0

        rec = {
            "round": server_round,
            "energy_j": round(joules, 3),
            "wall_s": round(seconds, 3),
            "mean_power_w": round(joules / seconds, 2) if seconds > 0 else None,
        }
        if out is not None:
            rec.update({k: float(v) for k, v in dict(out).items()})
        rec.update(self._train_metrics)

        self.rounds.append(rec)
        sigma = rec.get("noise_multiplier")
        extra = f" sigma={sigma:.3f}" if sigma is not None else ""
        print(
            f"[energy] round {server_round}: "
            f"{rec['energy_j']:.1f} J over {rec['wall_s']:.1f} s "
            f"({rec['mean_power_w']} W){extra}"
        )
        return out


def make_global_evaluate(seed: int, store: dict):
    """Server-side evaluation on the held-out CIFAR-10 test split.

    Results are stashed in `store` as well as returned: the strategy calls
    this after aggregate_evaluate, so the per-round record has already been
    appended by the time these numbers exist. main() merges them afterwards.
    """

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = get_device()
        loss, acc = test_fn(model, load_centralized_testset(), device)
        store[server_round] = {"central_loss": loss, "central_acc": acc}
        return MetricRecord({"central_loss": loss, "central_acc": acc})

    return global_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    seed = int(cfg["seed"])
    set_seed(seed)
    energy.init()

    if not energy.supports_total_energy():
        raise RuntimeError(
            "NVML total-energy counter unavailable; results would be estimates."
        )

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = EnergyFedAvg(
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
    )

    idle_w = energy.power_w()
    t_start = time.perf_counter()

    central: dict = {}
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": float(cfg["learning-rate"])}),
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=make_global_evaluate(seed, central),
    )

    # Central evaluation happens after the round record is appended, so fold
    # those metrics in now.
    for rec in strategy.rounds:
        rec.update(central.get(rec["round"], {}))

    total_s = time.perf_counter() - t_start
    total_j = sum(r["energy_j"] for r in strategy.rounds)

    run = {
        "run_id": cfg.get("run-id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "epsilon": cfg.get("epsilon", "inf"),  # "inf" = no-DP baseline
            "delta": 1e-5,
            "seed": seed,
            "dataset": "cifar10",
            "num_rounds": int(cfg["num-server-rounds"]),
            "num_supernodes": int(cfg["num-supernodes"]),
            "fraction_train": float(cfg["fraction-train"]),
            "fraction_evaluate": float(cfg["fraction-evaluate"]),
            "dirichlet_alpha": float(cfg["dirichlet-alpha"]),
            "local_epochs": int(cfg["local-epochs"]),
            "batch_size": int(cfg["batch-size"]),
            "learning_rate": float(cfg["learning-rate"]),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "idle_power_w_at_start": round(idle_w, 2),
        },
        "totals": {
            "energy_j": round(total_j, 3),
            "wall_s": round(total_s, 3),
            # Idle-subtracted compute energy. Report this alongside the gross
            # figure: at ~42 W measured against ~18 W idle, the net number is
            # roughly 43% of gross and is the honest measure of compute cost.
            "energy_j_net_idle": round(max(0.0, total_j - idle_w * total_s), 3),
        },
        "rounds": strategy.rounds,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eps = str(run["config"]["epsilon"]).replace(".", "p")
    path = RESULTS_DIR / f"run_eps{eps}_seed{seed}_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2))
    print(
        f"\n[energy] wrote {path}  "
        f"({total_j:.0f} J gross, {run['totals']['energy_j_net_idle']:.0f} J net, "
        f"{total_s:.0f} s)"
    )

    if bool(cfg.get("save-model", False)):
        torch.save(result.arrays.to_torch_state_dict(), RESULTS_DIR / f"model_seed{seed}.pt")