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

# Default output directory. Overridable per run via the `results-dir` run
# config -- NOT via an environment variable: the ServerApp runs as a child of
# the detached local SuperLink daemon, which was started with the environment
# of whichever `flwr run` first launched it, so env vars set on later
# invocations never reach it. Run config travels with the run itself.
DEFAULT_RESULTS_DIR = Path(os.environ.get("FL_RESULTS_DIR", "results"))


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
        # Materialise first: `replies` may be a one-shot iterator, and reading
        # it here would hand super() an exhausted one.
        replies = list(replies)
        per_client = self._client_metrics(replies)
        out = super().aggregate_train(server_round, replies)
        try:
            # FedAvg returns (ArrayRecord, MetricRecord). Its metrics are
            # num-examples-WEIGHTED MEANS, which is the wrong summary for a
            # workload quantity: a mean cannot show whether two epsilon
            # conditions did the same amount of compute. Sums come from
            # per_client below; the means are kept for continuity.
            metrics = out[1] if isinstance(out, tuple) else out
            # Averaging an id or a boolean flag yields a number that means
            # nothing; per_client records both properly.
            drop = {"partition_id", "device_cuda"}
            self._train_metrics = {
                k: float(v) for k, v in dict(metrics or {}).items() if k not in drop
            }
        except Exception as exc:  # never let logging break a run
            print(f"[warn] could not capture train metrics: {exc}")
            self._train_metrics = {}
        self._train_metrics.update(per_client)
        return out

    @staticmethod
    def _client_metrics(replies) -> dict:
        """Un-aggregated per-client training metrics for this round.

        The energy claim rests on every epsilon condition executing the same
        workload. Verifying that needs summed steps and summed examples --
        the weighted means FedAvg returns cannot distinguish "same compute"
        from "different compute, same average". Sigma min/max is recorded
        because it varies with Dirichlet partition size, and the paper has to
        report that range. sampled_partitions lets the analysis tell a real
        epsilon effect apart from two conditions simply drawing different
        clients in the same round.
        """
        steps, expected, examples, sigmas, parts = [], [], [], [], []
        on_cuda = []
        for reply in replies:
            try:
                if reply.has_error():
                    continue
                m = dict(reply.content["metrics"])
            except Exception:
                continue
            if "local_steps" in m:
                steps.append(float(m["local_steps"]))
            if "expected_steps" in m:
                expected.append(float(m["expected_steps"]))
            if "num-examples" in m:
                examples.append(float(m["num-examples"]))
            if "noise_multiplier" in m:
                sigmas.append(float(m["noise_multiplier"]))
            if "partition_id" in m:
                parts.append(int(m["partition_id"]))
            if "device_cuda" in m:
                on_cuda.append(float(m["device_cuda"]))

        rec: dict = {
            "n_clients": len(examples) or len(steps),
            "steps_sum": sum(steps),
            "expected_steps_sum": sum(expected),
            "examples_sum": sum(examples),
        }
        if sigmas:
            rec["sigma_min"] = min(sigmas)
            rec["sigma_max"] = max(sigmas)
        if parts:
            rec["sampled_partitions"] = sorted(parts)
        if on_cuda:
            rec["clients_on_cuda"] = int(sum(on_cuda))
            if sum(on_cuda) < len(on_cuda):
                print(
                    f"[FATAL] {len(on_cuda) - int(sum(on_cuda))}/{len(on_cuda)} "
                    "ClientApps trained on CPU. NVML is measuring an idle GPU "
                    "and this run's energy numbers are meaningless. Pass BOTH "
                    "--init-args-num-gpus 1 and --client-resources-num-gpus."
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

    # Idle baseline. Net-of-idle energy is a headline number (it roughly
    # doubles the apparent DP effect), so it is worth more than the single
    # instantaneous sample this used to take: average over a short quiet
    # window and keep the spread, so the paper can state how firm the
    # baseline is. Set idle-probe-s to 0 to skip the probe.
    probe_s = float(cfg.get("idle-probe-s", 5.0))
    if probe_s > 0:
        idle_w, idle_sd = energy.measure_idle_power_stats(probe_s)
    else:
        idle_w, idle_sd = energy.power_w(), 0.0
    print(f"[energy] idle baseline {idle_w:.2f} W (sd {idle_sd:.2f}) over {probe_s:.0f}s")

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
    # Wall-clock actually covered by the energy counter. This is NOT total_s:
    # the round window runs configure_train -> aggregate_evaluate, so run
    # start-up, the central evaluation between rounds, and teardown fall
    # outside it (~40% of total_s in practice). Subtracting idle over
    # total_s while total_j only covers measured_s over-subtracts the idle
    # draw of that uncovered time and understates net compute energy.
    measured_s = sum(r["wall_s"] for r in strategy.rounds)

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
            "clipping_norm": float(cfg.get("clipping-norm", 1.0)),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "idle_power_w_at_start": round(idle_w, 2),  # see totals.idle_power_w
        },
        "totals": {
            # Gross round-bracketed energy, and the wall-clock it covers.
            "energy_j": round(total_j, 3),
            "wall_s_measured": round(measured_s, 3),
            # Whole-process wall-clock, including start-up, central
            # evaluation and teardown. Reported for context only -- never
            # divide energy_j by this.
            "wall_s_total": round(total_s, 3),
            "mean_power_w": round(total_j / measured_s, 2) if measured_s > 0 else None,
            # Idle-subtracted compute energy: the honest measure of what the
            # workload costs, and the figure the DP comparison should use.
            # Subtracted over measured_s, matching the window energy_j covers.
            "energy_j_net_idle": round(max(0.0, total_j - idle_w * measured_s), 3),
            "idle_power_w": round(idle_w, 3),
            "idle_power_w_sd": round(idle_sd, 3),
        },
        "rounds": strategy.rounds,
    }

    results_dir = Path(str(cfg.get("results-dir", "")) or DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    eps = str(run["config"]["epsilon"]).replace(".", "p")
    path = results_dir / f"run_eps{eps}_seed{seed}_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2))
    print(
        f"\n[energy] wrote {path}  "
        f"({total_j:.0f} J gross, {run['totals']['energy_j_net_idle']:.0f} J net, "
        f"{total_s:.0f} s)"
    )

    if bool(cfg.get("save-model", False)):
        torch.save(result.arrays.to_torch_state_dict(), results_dir / f"model_seed{seed}.pt")