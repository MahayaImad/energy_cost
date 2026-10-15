"""ServerApp: FedAvg, NVML energy per round, one JSON per run.

The NVML counter (accumulated mJ) is read when a round starts
(configure_train) and when it ends (aggregate_evaluate); the difference is the
round's GPU energy. Start-up, central evaluation and teardown fall outside
the window. The idle baseline is probed before training and subtracted over
the measured wall-clock to give net energy.
"""

import json
import time
from pathlib import Path

import pynvml
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from energyfl import dp
from energyfl.task import (build_model, get_device, load_partition,
                           load_testset, set_seed, test)

app = ServerApp()


def gpu_energy_j(handle):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0


def idle_power_w(handle, seconds=5.0):
    """Wait until board power is stable (spread <= 1 W over 4 readings),
    then average it for `seconds`. A card still winding down from the
    previous run would read a busy baseline."""
    power = lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    recent, t0 = [], time.time()
    while time.time() - t0 < 60:
        recent = (recent + [power()])[-4:]
        if len(recent) == 4 and max(recent) - min(recent) <= 1.0:
            break
        time.sleep(0.5)
    readings = []
    for _ in range(int(seconds / 0.25)):
        readings.append(power())
        time.sleep(0.25)
    mean = sum(readings) / len(readings)
    sd = (sum((x - mean) ** 2 for x in readings) / (len(readings) - 1)) ** 0.5
    return mean, sd


class EnergyFedAvg(FedAvg):
    def __init__(self, handle, **kw):
        super().__init__(**kw)
        self.handle, self.rounds = handle, []

    def configure_train(self, server_round, arrays, config, grid):
        self._e0, self._t0 = gpu_energy_j(self.handle), time.perf_counter()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        m = [dict(r.content["metrics"]) for r in replies if not r.has_error()]
        self._clients = {"sampled_partitions": sorted(int(x["partition_id"]) for x in m)}
        sigmas = [x["noise_multiplier"] for x in m if "noise_multiplier" in x]
        if sigmas:
            self._clients.update(sigma_min=min(sigmas), sigma_max=max(sigmas))
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round, replies):
        out = super().aggregate_evaluate(server_round, replies)
        joules = gpu_energy_j(self.handle) - self._e0
        seconds = time.perf_counter() - self._t0
        self.rounds.append({"round": server_round, "energy_j": round(joules, 3),
                            "wall_s": round(seconds, 3), **self._clients})
        print(f"[energy] round {server_round}: {joules:.1f} J in {seconds:.2f} s")
        return out


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    dataset, seed = cfg["dataset"], int(cfg["seed"])
    rounds, n_clients = int(cfg["num-server-rounds"]), int(cfg["num-supernodes"])
    set_seed(seed)

    # Every sigma is derived here, before measurement. The accountant search
    # is slower for a loose budget than a tight one; run inside the measured
    # rounds it produces an energy trend in epsilon that is not a cost of
    # privacy (Section V-D).
    train_cfg = {}
    if dp.is_private(cfg["epsilon"]):
        sizes = [len(load_partition(pid, n_clients, int(cfg["batch-size"]),
                                    float(cfg["dirichlet-alpha"]), seed,
                                    dataset)[0].dataset)
                 for pid in range(n_clients)]
        train_cfg["sigmas"] = [
            float(dp.noise_multiplier(cfg["epsilon"], n, int(cfg["batch-size"]),
                                      rounds, float(cfg["fraction-train"]),
                                      int(cfg["local-epochs"])))
            for n in sizes]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    idle_w, idle_sd = idle_power_w(handle)
    print(f"[energy] idle baseline {idle_w:.2f} W (sd {idle_sd:.2f})")

    testloader = load_testset(dataset)
    central = {}

    def central_eval(server_round, arrays):
        model = build_model(dataset)
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, testloader, get_device())
        central[server_round] = {"central_loss": loss, "central_acc": acc}
        return MetricRecord(central[server_round])

    strategy = EnergyFedAvg(handle,
                            fraction_train=float(cfg["fraction-train"]),
                            fraction_evaluate=float(cfg["fraction-evaluate"]))
    strategy.start(grid=grid,
                   initial_arrays=ArrayRecord(build_model(dataset).state_dict()),
                   train_config=ConfigRecord(train_cfg), num_rounds=rounds,
                   evaluate_fn=central_eval)

    for r in strategy.rounds:
        r.update(central.get(r["round"], {}))
    energy = sum(r["energy_j"] for r in strategy.rounds)
    wall = sum(r["wall_s"] for r in strategy.rounds)
    run = {
        "run_id": cfg["run-id"],
        "config": {
            "epsilon": cfg["epsilon"], "delta": dp.DELTA, "seed": seed,
            "dataset": dataset, "num_rounds": rounds,
            "num_supernodes": n_clients,
            "fraction_train": float(cfg["fraction-train"]),
            "fraction_evaluate": float(cfg["fraction-evaluate"]),
            "dirichlet_alpha": float(cfg["dirichlet-alpha"]),
            "local_epochs": int(cfg["local-epochs"]),
            "batch_size": int(cfg["batch-size"]),
            "learning_rate": float(cfg["learning-rate"]),
            "clipping_norm": float(cfg["clipping-norm"]),
        },
        "hardware": {"gpu": torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None,
                     "torch": torch.__version__, "cuda": torch.version.cuda},
        "totals": {"energy_j": round(energy, 3),
                   "wall_s_measured": round(wall, 3),
                   "idle_power_w": round(idle_w, 3),
                   "idle_power_w_sd": round(idle_sd, 3),
                   "energy_j_net_idle": round(max(0.0, energy - idle_w * wall), 3)},
        "rounds": strategy.rounds,
    }
    out = Path(cfg["results-dir"])
    out.mkdir(parents=True, exist_ok=True)
    eps = str(cfg["epsilon"]).replace(".", "p")
    path = out / f"run_eps{eps}_seed{seed}_{cfg['run-id']}.json"
    path.write_text(json.dumps(run, indent=1))
    print(f"[energy] wrote {path}")
