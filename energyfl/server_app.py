"""ServerApp: FedAvg with per-round GPU energy measurement and JSON logging.

Each run writes one self-describing JSON: the full config, then per round the
accuracy, loss, energy, wall-clock and the client-side training metrics. That
file is the unit of analysis -- analyze.py reads these and nothing else, so
no number in the paper can come from a log line.
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
    build_model,
    get_device,
    load_centralized_testset,
    set_seed,
    test_fn,
)

app = ServerApp()

# Where run_*.json goes when the run config does not say. Note this is a run
# config key, NOT an environment variable: the ServerApp is a child of the
# detached SuperLink daemon, which keeps the environment of whichever `flwr
# run` first started it, so env vars set on later invocations never arrive.
DEFAULT_RESULTS_DIR = Path(os.environ.get("FL_RESULTS_DIR", "results"))

IDLE_PROBE_S = 5.0


class EnergyFedAvg(FedAvg):
    """FedAvg that brackets each round with an NVML energy reading.

    configure_train opens the window and aggregate_evaluate closes it, so the
    measurement covers client training, client evaluation and aggregation --
    everything the round costs on this GPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds: list[dict] = []
        self._e0: Optional[int] = None
        self._t0: Optional[float] = None
        self._train_metrics: dict = {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._e0 = energy.energy_mj()
        self._t0 = time.perf_counter()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies):
        # Materialise first: `replies` may be a one-shot iterator, and reading
        # it here would hand super() an exhausted one.
        replies = list(replies)
        per_client = self._client_metrics(replies)
        out = super().aggregate_train(server_round, replies)

        # FedAvg's own metrics are num-examples-weighted MEANS, which cannot
        # show whether two epsilon conditions did the same amount of compute;
        # the sums in per_client can. Averaging an id or a boolean flag is
        # meaningless, so those two are dropped rather than kept as noise.
        try:
            metrics = out[1] if isinstance(out, tuple) else out
            self._train_metrics = {
                k: float(v) for k, v in dict(metrics or {}).items()
                if k not in ("partition_id", "device_cuda")
            }
        except Exception as exc:  # logging must never kill a run
            print(f"[warn] could not read aggregated train metrics: {exc}")
            self._train_metrics = {}
        self._train_metrics.update(per_client)
        return out

    @staticmethod
    def _client_metrics(replies) -> dict:
        """Un-aggregated per-client training metrics for this round.

        The energy claim rests on every epsilon condition executing the same
        workload, and verifying that needs summed steps and summed examples.
        sigma is recorded as a range because it varies with partition size,
        and sampled_partitions lets the analysis tell a real epsilon effect
        apart from two rounds simply drawing different clients.
        """
        fields = {"local_steps": [], "expected_steps": [], "num-examples": [],
                  "noise_multiplier": [], "partition_id": [], "device_cuda": []}
        for reply in replies:
            try:
                if reply.has_error():
                    continue
                m = dict(reply.content["metrics"])
            except Exception:      # a malformed reply is skipped, not fatal
                continue
            for name, bucket in fields.items():
                if name in m:
                    bucket.append(float(m[name]))

        rec = {
            "n_clients": len(fields["num-examples"]) or len(fields["local_steps"]),
            "steps_sum": sum(fields["local_steps"]),
            "expected_steps_sum": sum(fields["expected_steps"]),
            "examples_sum": sum(fields["num-examples"]),
        }
        if fields["noise_multiplier"]:
            rec["sigma_min"] = min(fields["noise_multiplier"])
            rec["sigma_max"] = max(fields["noise_multiplier"])
        if fields["partition_id"]:
            rec["sampled_partitions"] = sorted(int(p) for p in fields["partition_id"])

        on_cuda = fields["device_cuda"]
        if on_cuda:
            rec["clients_on_cuda"] = int(sum(on_cuda))
            if sum(on_cuda) < len(on_cuda):
                print(
                    f"[FATAL] {len(on_cuda) - int(sum(on_cuda))}/{len(on_cuda)} "
                    "ClientApps trained on CPU. NVML is measuring an idle GPU, "
                    "so this run's energy numbers are meaningless. Pass the GPU "
                    "resources through --federation-config, e.g.\n"
                    '  --federation-config "init_args_num_gpus=1 '
                    'client_resources_num_gpus=0.25"'
                )
        return rec

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
        print(f"[energy] round {server_round}: {rec['energy_j']:.1f} J over "
              f"{rec['wall_s']:.1f} s ({rec['mean_power_w']} W)"
              + (f" sigma={sigma:.3f}" if sigma is not None else ""))
        return out


def make_global_evaluate(store: dict, dataset: str):
    """Server-side evaluation on the held-out test split.

    Results are stashed in `store` as well as returned, because the strategy
    calls this after aggregate_evaluate has already appended the round record.
    main() folds them in afterwards.
    """

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(dataset)
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test_fn(
            model, load_centralized_testset(dataset=dataset), get_device()
        )
        store[server_round] = {"central_loss": loss, "central_acc": acc}
        return MetricRecord({"central_loss": loss, "central_acc": acc})

    return global_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    seed = int(cfg["seed"])
    dataset = str(cfg["dataset"])
    set_seed(seed)

    energy.init()
    if not energy.supports_total_energy():
        raise RuntimeError(
            "NVML total-energy counter unavailable; results would be estimates."
        )

    # Idle baseline, measured after the card has wound down from whatever ran
    # before. Net-of-idle energy subtracts it and is a headline number, so it
    # is worth a short averaged window rather than one instantaneous sample.
    idle_w, idle_sd, settled, waited = energy.measure_idle_power(IDLE_PROBE_S)
    print(f"[energy] idle baseline {idle_w:.2f} W (sd {idle_sd:.2f}), "
          f"settled after {waited:.1f} s"
          + ("" if settled else "  [WARN: never settled]"))

    strategy = EnergyFedAvg(
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
    )

    t_start = time.perf_counter()
    central: dict = {}
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(build_model(dataset).state_dict()),
        train_config=ConfigRecord({"lr": float(cfg["learning-rate"])}),
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=make_global_evaluate(central, dataset),
    )
    for rec in strategy.rounds:
        rec.update(central.get(rec["round"], {}))

    total_s = time.perf_counter() - t_start
    total_j = sum(r["energy_j"] for r in strategy.rounds)
    # The wall-clock the energy counter actually covers, which is NOT total_s:
    # the round window is configure_train -> aggregate_evaluate, so start-up,
    # central evaluation between rounds and teardown fall outside it (~40% of
    # total_s in practice). Subtracting idle over total_s while total_j covers
    # only measured_s would over-subtract and understate net compute energy.
    measured_s = sum(r["wall_s"] for r in strategy.rounds)

    run = {
        "run_id": cfg.get("run-id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "epsilon": cfg["epsilon"],          # "inf" = the no-DP baseline
            "delta": 1e-5,
            "seed": seed,
            "dataset": dataset,
            "num_rounds": int(cfg["num-server-rounds"]),
            "num_supernodes": int(cfg["num-supernodes"]),
            "fraction_train": float(cfg["fraction-train"]),
            "fraction_evaluate": float(cfg["fraction-evaluate"]),
            "dirichlet_alpha": float(cfg["dirichlet-alpha"]),
            "local_epochs": int(cfg["local-epochs"]),
            "batch_size": int(cfg["batch-size"]),
            "learning_rate": float(cfg["learning-rate"]),
            "clipping_norm": float(cfg["clipping-norm"]),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "totals": {
            "energy_j": round(total_j, 3),
            "wall_s_measured": round(measured_s, 3),
            # Whole-process wall-clock, for context only. Never divide
            # energy_j by this.
            "wall_s_total": round(total_s, 3),
            "mean_power_w": round(total_j / measured_s, 2) if measured_s > 0 else None,
            # Idle-subtracted compute energy: what the workload actually costs,
            # and the figure the DP comparison uses. Subtracted over
            # measured_s, matching the window energy_j covers.
            "energy_j_net_idle": round(max(0.0, total_j - idle_w * measured_s), 3),
            "idle_power_w": round(idle_w, 3),
            "idle_power_w_sd": round(idle_sd, 3),
            "idle_settled": bool(settled),
            "idle_settle_wait_s": round(waited, 2),
        },
        "rounds": strategy.rounds,
    }

    results_dir = Path(str(cfg.get("results-dir", "")) or DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    eps = str(run["config"]["epsilon"]).replace(".", "p")
    path = results_dir / f"run_eps{eps}_seed{seed}_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2))
    print(f"\n[energy] wrote {path}  ({total_j:.0f} J gross, "
          f"{run['totals']['energy_j_net_idle']:.0f} J net, {total_s:.0f} s)")
