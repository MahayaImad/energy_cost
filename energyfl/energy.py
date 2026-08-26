"""GPU energy measurement via NVML hardware counters.

Uses nvmlDeviceGetTotalEnergyConsumption (accumulated millijoules) when
available -- this is a hardware counter, not an estimate, and differencing
it avoids all sampling/integration error.
"""

import time

import pynvml

_HANDLE = None
_SUPPORTS_TOTAL = None


def init(gpu_index: int = 0) -> None:
    global _HANDLE, _SUPPORTS_TOTAL
    if _HANDLE is not None:
        return
    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE)
        _SUPPORTS_TOTAL = True
    except pynvml.NVMLError:
        _SUPPORTS_TOTAL = False


def supports_total_energy() -> bool:
    init()
    return bool(_SUPPORTS_TOTAL)


def energy_mj() -> int:
    """Accumulated GPU energy in millijoules since driver load."""
    init()
    if not _SUPPORTS_TOTAL:
        raise RuntimeError(
            "nvmlDeviceGetTotalEnergyConsumption unsupported on this GPU; "
            "fall back to power sampling."
        )
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE)


def power_w() -> float:
    """Instantaneous board power draw in watts."""
    init()
    return pynvml.nvmlDeviceGetPowerUsage(_HANDLE) / 1000.0


def measure_idle_power(seconds: float = 20.0, samples: int = 40) -> float:
    """Mean idle board power. Run once on an otherwise-quiet GPU.

    Report energy both gross and net of this baseline in the paper.
    """
    init()
    interval = seconds / samples
    readings = []
    for _ in range(samples):
        readings.append(power_w())
        time.sleep(interval)
    return sum(readings) / len(readings)


def wait_until_settled(
    tol_w: float = 1.0,
    timeout_s: float = 60.0,
    interval: float = 0.5,
    window: int = 4,
):
    """Block until board power stops changing. Returns (settled, waited_s).

    A GPU does not drop to its floor the instant a run ends; clocks and fans
    wind down over seconds. Probing the idle baseline before that happens
    reads a still-busy card, and since net energy subtracts idle*t, an
    inflated baseline silently UNDER-states net compute energy for that run.

    Short runs make this easy to hit: back-to-back HAR runs 35 s apart put
    the probe inside the previous run's tail and produced a 29.5 W baseline
    where neighbouring conditions read 18.6-19.9 W, which was enough on its
    own to fail the per-round flatness check.

    Settling is judged by stability, not by an absolute floor, since the
    floor is hardware-specific and unknown here. A steady but genuinely busy
    card therefore still looks settled -- that case is caught afterwards by
    comparing baselines across runs.
    """
    init()
    t0 = time.perf_counter()
    recent: list[float] = []
    while time.perf_counter() - t0 < timeout_s:
        recent.append(power_w())
        if len(recent) > window:
            recent.pop(0)
        if len(recent) == window and max(recent) - min(recent) <= tol_w:
            return True, time.perf_counter() - t0
        time.sleep(interval)
    return False, time.perf_counter() - t0


def measure_idle_power_stats(
    seconds: float = 5.0, interval: float = 0.25, settle: bool = True
):
    """(mean, sample sd) of idle board power over a quiet window.

    Net-of-idle energy subtracts this baseline from every run, so a single
    instantaneous reading is too thin a basis: report the spread alongside
    the mean and let the paper state how firm the baseline is.

    Waits for the card to settle first; see wait_until_settled.
    """
    init()
    if settle:
        wait_until_settled()
    n = max(2, int(seconds / interval))
    readings = []
    for _ in range(n):
        readings.append(power_w())
        time.sleep(interval)
    mean = sum(readings) / len(readings)
    var = sum((x - mean) ** 2 for x in readings) / (len(readings) - 1)
    return mean, var ** 0.5


class RoundEnergy:
    """Context manager measuring GPU joules and wall-clock over a block."""

    def __init__(self) -> None:
        self.joules = 0.0
        self.seconds = 0.0
        self._e0 = 0
        self._t0 = 0.0

    def __enter__(self) -> "RoundEnergy":
        init()
        self._e0 = energy_mj()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        self.joules = (energy_mj() - self._e0) / 1000.0


if __name__ == "__main__":
    init()
    print("total-energy counter supported:", supports_total_energy())
    print("current power (W):", power_w())
    print("measuring idle power for 20s, keep the GPU quiet...")
    m, sd = measure_idle_power_stats(20.0)
    print(f"idle power (W): {m:.2f} +/- {sd:.2f}")
