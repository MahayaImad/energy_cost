"""GPU energy measurement via NVML hardware counters.

nvmlDeviceGetTotalEnergyConsumption returns accumulated millijoules since
driver load. Differencing it around a block gives the block's energy with no
sampling or integration error, which is the whole reason the paper can report
measured joules rather than a TDP estimate.
"""

import time

import pynvml

_handle = None
_supports_total = None


def init(gpu_index: int = 0) -> None:
    global _handle, _supports_total
    if _handle is not None:
        return
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(_handle)
        _supports_total = True
    except pynvml.NVMLError:
        _supports_total = False


def supports_total_energy() -> bool:
    init()
    return bool(_supports_total)


def energy_mj() -> int:
    """Accumulated GPU energy in millijoules since driver load."""
    init()
    if not _supports_total:
        raise RuntimeError(
            "nvmlDeviceGetTotalEnergyConsumption is unsupported on this GPU. "
            "Every energy number would be a power-sampling estimate, so the "
            "run refuses to start rather than quietly changing what it means."
        )
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_handle)


def power_w() -> float:
    """Instantaneous board power draw in watts."""
    init()
    return pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000.0


def wait_until_settled(tol_w=1.0, timeout_s=60.0, interval=0.5, window=4):
    """Block until board power stops moving. Returns (settled, waited_s).

    A GPU does not drop to its floor the moment a run ends; clocks and fans
    wind down over seconds. Net energy subtracts idle * time, so a baseline
    probed too early is inflated and silently UNDER-states that run's net
    compute energy.

    Sweeps make this easy to hit. Back-to-back HAR runs 35 s apart put the
    probe inside the previous run's tail and read 29.5 W where neighbouring
    conditions read 18.6-19.9 W -- on its own enough to fail the per-round
    flatness check.

    Settling is judged by stability rather than by an absolute floor, which
    is hardware-specific and unknown here. A steady but genuinely busy card
    therefore still looks settled; analyze.py catches that case afterwards by
    comparing baselines across runs.
    """
    init()
    t0 = time.perf_counter()
    recent: list[float] = []
    while time.perf_counter() - t0 < timeout_s:
        recent.append(power_w())
        recent = recent[-window:]
        if len(recent) == window and max(recent) - min(recent) <= tol_w:
            return True, time.perf_counter() - t0
        time.sleep(interval)
    return False, time.perf_counter() - t0


def measure_idle_power(seconds=5.0, interval=0.25):
    """Probe the idle baseline. Returns (mean_w, sd_w, settled, waited_s).

    Waits for the card to settle first. The spread comes back alongside the
    mean because every run's net energy rests on this one number, and the
    paper states how firm it is.
    """
    init()
    settled, waited = wait_until_settled()
    readings = []
    for i in range(max(2, int(seconds / interval))):
        if i:
            time.sleep(interval)
        readings.append(power_w())
    mean = sum(readings) / len(readings)
    var = sum((x - mean) ** 2 for x in readings) / (len(readings) - 1)
    return mean, var ** 0.5, settled, waited


if __name__ == "__main__":
    init()
    print("total-energy counter supported:", supports_total_energy())
    print("current power:", f"{power_w():.1f} W")
    print("measuring idle power for 20 s, keep the GPU quiet...")
    mean, sd, settled, waited = measure_idle_power(20.0)
    print(f"idle power: {mean:.2f} +/- {sd:.2f} W "
          f"({'settled' if settled else 'NEVER SETTLED'} after {waited:.1f} s)")
