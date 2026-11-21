"""How long the steps of estimating a linearised model take, by model size: python benchmarks/model_speed.py

Models: the New Keynesian template (3 equations), and chains of blocks (a forward-looking gap, a backward trend and a
policy rule with max(), 90 to 300 equations), with or without identities (static and led-only variables). For each:
the steady state, linearisation, the first-order solution, the Kalman-filter likelihood, and one trial value of an
estimation (rebuild the model at new parameter values, then all of the above), the step Metropolis-Hastings repeats
for every draw: cold, and warm (the next step of a random walk, starting from the last solution). The best of three
runs is shown, in seconds. Add --mcmc for 2 x 1000 draws on the template.

The machine, Python and linear-algebra library are printed first: timings depend on all three, so compare numbers
from different machines with that in mind.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.metadata import Role
from pymetrix.models import MeasurementMap
from pymetrix.models.estimation import estimate_model


def chain(blocks: int, identities: bool = False) -> Model:
    eqs, params, roles = [], {}, {}
    for i in range(blocks):
        prev = f"x{i - 1}" if i else "x0"
        eqs += [Equation(f"x{i} ~ a{i}*x{i}[-1] + b{i}*x{i}[+1] + 0.1*{prev} - 0.2*(exp(0.5*r{i}) - 1) + ex{i}", params=[f"a{i}", f"b{i}"]),
                Equation(f"g{i} ~ rho{i}*g{i}[-1] + (1-rho{i})*2 + 0.05*x{i} + eg{i}", params=[f"rho{i}"]),
                Equation(f"r{i} ~ max(c{i}*r{i}[-1] + (1-c{i})*(1.5*x{i}[+1] + 0.5*x{i}) + er{i}, -5)", params=[f"c{i}"])]
        if identities:
            eqs += [Equation(f"y{i} = x{i} + 0.5*g{i}"), Equation(f"w{i} = y{i} - 0.3*r{i} + 0.1*y{i}[+1]"), Equation(f"q{i} = 2*w{i} - y{i}")]
        params.update({f"a{i}": 0.5, f"b{i}": 0.3, f"rho{i}": 0.8, f"c{i}": 0.7})
        roles.update({f"ex{i}": Role.SHOCK, f"eg{i}": Role.SHOCK, f"er{i}": Role.SHOCK})
    return Model(eqs, params=params, roles=roles)


def _cpu() -> str:
    """The processor's name, where the operating system says it."""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine() -> list[str]:
    """What the timings depend on: processor, operating system, Python, NumPy and SciPy, and the BLAS they use."""
    import scipy

    import pymetrix
    import pymetrix.statespace as ss

    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas = f"{blas.get('name', '?')} {blas.get('version', '')}".strip()
    except Exception:
        blas = "unknown"
    try:
        import statsmodels

        filter_engine = f"{ss.ENGINE} (statsmodels {statsmodels.__version__} installed)"
    except ImportError:
        filter_engine = f"{ss.ENGINE} (statsmodels not installed: the native filter)"
    from pymetrix.threads import get_blas_threads

    return [f"Machine: {_cpu()}, {os.cpu_count()} logical processors; {platform.platform()}",
            f"Python {platform.python_version()}; NumPy {np.__version__}, SciPy {scipy.__version__}, BLAS {blas}",
            f"pymetrix {pymetrix.__version__}; Kalman filter {filter_engine}; "
            f"BLAS threads during estimation {get_blas_threads() or 'library default'} (each step below is timed so)"]


def best(f, n: int = 3) -> float:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return min(times)


def row(label: str, model: Model, shocks, sd, measurement, free) -> list:
    ss = model.steady_state()
    lin = model.linearize(ss, shocks=shocks)
    sol = lin.solve()
    paths = sol.simulate(120, sd, seed=1)
    paths.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    data = measurement.apply(Dataset.from_frame(paths))
    system = sol.system(measurement, sd)
    fit = estimate_model(model, data, measurement, shocks=shocks, shock_sd=sd, estimate=free, max_iter=1)
    mode = np.asarray(fit.params.values, dtype=float)
    owner = fit.log_posterior.__self__

    def cold() -> None:  # a trial value from scratch
        owner.forget()
        fit.log_posterior(mode * 1.01)

    walk = mode * (1 + 0.01 * np.random.default_rng(0).standard_normal((6, len(mode))))

    def warm() -> None:  # five steps of a random walk, as Metropolis-Hastings takes them: each starts from the last
        fit.log_posterior(walk[0])
        start = time.perf_counter()
        for x in walk[1:]:
            fit.log_posterior(x)
        steps.append((time.perf_counter() - start) / 5)

    steps: list[float] = []
    best(warm)
    return [label, len(model.equations), len(system.states), len(system.needed().states), best(lambda: model.steady_state()),
            best(lambda: model.linearize(ss, shocks=shocks)), best(lambda: lin.solve()), best(lambda: system.loglike(data)),
            best(cold), min(steps)]


def main() -> None:
    print("\n".join(machine()), end="\n\n")
    from pymetrix.threads import blas_threads

    with blas_threads():  # as estimation runs them
        rows = _rows()
    _show(rows)
    if "--mcmc" in sys.argv:
        _mcmc()


def _rows() -> list[list]:
    rows = []
    nk = templates.new_keynesian()
    rows.append(row("New Keynesian template", nk.model, nk.shocks, nk.shock_sd, nk.measurement, ["kappa", "rho"]))
    for blocks, identities in ((30, False), (60, False), (100, False), (30, True), (50, True)):
        m = chain(blocks, identities)
        shocks = [s for i in range(blocks) for s in (f"ex{i}", f"eg{i}", f"er{i}")]
        observed = [f"x{i}" for i in range(0, blocks, blocks // 5)][:5]
        measurement = MeasurementMap([f"o{v} = {v}" for v in observed])
        rows.append(row(f"chain{' with identities' if identities else ''}", m, shocks, {s: 0.1 for s in shocks}, measurement, ["a0", "c0"]))
    return rows


def _show(rows: list[list]) -> None:
    head = ["model", "equations", "states", "filtered", "steady", "linearise", "solve", "likelihood", "estimation step",
            "next step (warm)"]
    print(" | ".join(head))
    for r in rows:
        print(" | ".join([r[0], *map(str, r[1:4]), *(f"{v:.3f}" for v in r[4:])]))


def _mcmc() -> None:
    from pymetrix.estimation.priors import Beta, Gamma, InvGamma1

    nk = templates.new_keynesian()
    truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
    truth.index = pd.period_range("1995Q1", periods=120, freq="Q", name="time")
    fit = nk.estimate(nk.measurement.apply(Dataset.from_frame(truth)), estimate=["kappa", "rho", "sd_ei"],
                      priors={"kappa": Gamma(0.15, 0.05), "rho": Beta(0.7, 0.1), "sd_ei": InvGamma1(0.2, float("inf"))})
    for workers in (1, None):
        start = time.perf_counter()
        fit.mcmc(draws=1000, chains=2, seed=1, workers=workers)
        how = "one after another" if workers == 1 else "in parallel when it pays (the default)"
        print(f"Metropolis-Hastings on the template, 2 x 1000 draws after as many burn-in, {how}: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
