"""Many small regressions, by formula and by ``pymetrix.ols``: python benchmarks/ols_arrays.py [N]

N regressions (10,000 by default) of 40 quarterly observations on 3 to 6 regressors with a constant: by
``Equation("y ~ x0 + x1 + ...").fit(as_dataset(frame))``, by ``pymetrix.ols(y, X)`` on the same Series and DataFrame,
by ``pymetrix.ols`` on numpy arrays, and by ``pymetrix.ols`` with HAC standard errors. Every path computes the same
result (a ``FittedEquation`` with its provenance); ``tests/test_ols_arrays.py`` checks the numbers agree.

The machine, Python and library versions are printed first: timings depend on them, and on how busy the machine is.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np
import pandas as pd

from pymetrix import Equation, as_dataset, ols


def problems(n: int, seed: int = 0) -> list[tuple[pd.Series, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    out = []
    for i in range(n):
        k = 3 + i % 4
        X = pd.DataFrame(rng.normal(size=(40, k)), columns=[f"x{j}" for j in range(k)], index=index)
        out.append(((X.sum(axis=1) + rng.normal(size=40)).rename("y"), X))
    return out


def main(n: int) -> None:
    print(f"{platform.platform()}; Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}")
    data = problems(n)
    runs = {
        "formula and Dataset": lambda y, X: Equation("y ~ " + " + ".join(X.columns)).fit(as_dataset(pd.concat([y, X], axis=1))),
        "ols(Series, DataFrame)": lambda y, X: ols(y, X),
        "ols(numpy arrays)": lambda y, X: ols(y.to_numpy(), X.to_numpy()),
        "ols(..., cov='hac')": lambda y, X: ols(y, X, cov="hac"),
    }
    first = None
    for name, run in runs.items():
        start = time.perf_counter()
        for y, X in data:
            run(y, X)
        elapsed = time.perf_counter() - start
        first = first or elapsed
        print(f"{name:24s} {elapsed:8.2f} s   {elapsed / n * 1e3:6.2f} ms each   {first / elapsed:5.1f}x")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
