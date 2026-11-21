"""How the period-by-period solver's options change the work of solving a large backward-looking model:
python benchmarks/solve_options.py [--equations 200] [--quarters 40] [--repeats 7]

The model is generated: a chain of identities, each needing the one before (recursive), fed by two small nonlinear
simultaneous blocks (three equations with an ``exp`` and a ``log``, and two with a ratio), about 200 equations over
40 quarters. It is solved four ways: the default (``Newton()`` on all the equations at once); ``ordering="blocks"``;
``Newton(refresh="period")``; and the combination docs/solving-large-models.md recommends (``RECOMMENDED``). For each:
the median wall time of ``Model.solve`` over the repeats (after one solve not timed, which compiles the equations),
and of the time in the periods' solves alone (the sum of ``Solution.convergence.attrs["seconds"]``: the rest is
setting the solve up), the equations evaluated (each residual of each equation counts one; a Jacobian counts its rows), the Jacobians
evaluated and their rows in all, the iterations, and the largest difference from the default's solution.

Time your own model (any model ``Model.from_file`` reads, and data ``Dataset.read`` reads, with values before the
span for the lags; ``--time`` and ``--freq`` name a CSV or Excel file's column of periods and their frequency)::

    python benchmarks/solve_options.py --model mymodel.xml --data mydata.csv --time date --freq Q --span 2025Q1 2034Q4

The machine, Python and library versions are printed first: timings depend on them, and on how busy the machine is.
The counts do not: they are the same on any machine.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from pymetrix import Dataset, Equation, Fallback, GaussSeidel, Model, Newton

# the starting configuration docs/solving-large-models.md recommends for a model of mostly identities with a few
# simultaneous blocks (its test checks the page gives the same)
RECOMMENDED: dict[str, Any] = {
    "ordering": "blocks",
    "solver": Fallback(Newton(convergence="step", refresh="period"),
                       Newton(convergence="step", line_search=True),
                       GaussSeidel(convergence="step", damping=0.5, max_iter=1000)),
}


def generate(equations: int = 200, quarters: int = 40) -> tuple[Model, Dataset, tuple[str, str]]:
    """A backward-looking model of ``equations`` equations (at least 6), its data and the span to solve: two small
    nonlinear simultaneous blocks, (c, inv, y) and (p, u), and a recursive chain of identities v0, v1, ... fed by
    them. Four quarters of history before the span give the lags; the endogenous variables have no values in the
    span, as in a projection."""
    if equations < 6:
        raise ValueError("the generated model has at least 6 equations")
    eqs = [Equation("c = 0.55*y + 0.25*c[-1] + 0.1*log(w)", name="cons"),
           Equation("inv = 0.12*y + 0.05*exp(0.2*(y - y[-1])) + 0.4*inv[-1]", name="invest"),
           Equation("y = c + inv + g", name="output"),
           Equation("p = 0.6*p[-1] + 0.3*log(1 + u) + 0.01*y", name="prices"),
           Equation("u = 0.5*u[-1] + 0.2*p/(1 + p) + 0.1*z", name="slack"),
           Equation("v0 = 0.1*y + 0.3*v0[-1]", name="link0")]
    for k in range(1, equations - 5):
        eqs.append(Equation(f"v{k} = 0.5*v{k - 1} + 0.1*p + 0.3*v{k}[-1] + 0.01*z", name=f"link{k}"))
    endogenous = [e.dependent for e in eqs]
    index = pd.period_range("2000Q1", periods=quarters + 4, freq="Q", name="time")
    t = np.arange(len(index))
    frame = pd.DataFrame(1.0, index=index, columns=endogenous)
    frame["g"] = 1.0 + 0.05 * np.sin(t / 3.0)
    frame["w"] = 1.5 + 0.01 * t
    frame["z"] = 0.5 + 0.1 * np.cos(t / 5.0)
    frame.iloc[4:, : len(endogenous)] = np.nan
    return Model(eqs), Dataset.from_frame(frame), (str(index[4]), str(index[-1]))


def configurations() -> list[tuple[str, dict[str, Any]]]:
    return [("default: Newton(), joint", {}),
            ('ordering="blocks"', {"ordering": "blocks"}),
            ('Newton(refresh="period")', {"solver": Newton(refresh="period")}),
            ("recommended (blocks + Fallback)", RECOMMENDED)]


@contextmanager
def counting():
    """Count the work at the period system every solve builds (``pymetrix.models.model._PeriodSystem``, internal):
    ``equations``, each equation evaluated (all of a system's for its residuals, one for an equation on its own, a
    Jacobian's rows); ``jacobians`` and ``rows``, the Jacobians evaluated and their rows."""
    from pymetrix.models import model as model_module

    cls = model_module._PeriodSystem
    residuals, jacobian, own = cls.residuals, cls.jacobian, cls.own
    count = {"equations": 0, "jacobians": 0, "rows": 0}

    def counted_residuals(self, x):
        count["equations"] += len(self.eqs)
        return residuals(self, x)

    def counted_jacobian(self, x):
        count["equations"] += len(self.eqs)
        count["jacobians"] += 1
        count["rows"] += len(self.eqs)
        return jacobian(self, x)

    def counted_own(self, i, x):
        count["equations"] += 1
        return own(self, i, x)

    cls.residuals, cls.jacobian, cls.own = counted_residuals, counted_jacobian, counted_own
    try:
        yield count
    finally:
        cls.residuals, cls.jacobian, cls.own = residuals, jacobian, own


def measure(model: Model, data: Dataset, span: tuple[str, str], repeats: int = 7) -> list[dict[str, Any]]:
    """Each configuration's median seconds, counts, iterations and largest difference from the default's solution."""
    rows = []
    reference = None
    for label, options in configurations():
        model.solve(data, span, **options)  # not timed: compiles the equations (and the blocks' systems)
        times, solving = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            timed = model.solve(data, span, **options)
            times.append(time.perf_counter() - start)
            solving.append(sum(timed.convergence.attrs.get("seconds", [])))
        with counting() as count:
            solution = model.solve(data, span, **options)
        values = solution.solved().frame[list(model.endogenous)].to_numpy(float)
        reference = values if reference is None else reference
        rows.append({"configuration": label, "seconds": statistics.median(times), "solving": statistics.median(solving),
                     **count,
                     "iterations": int(solution.convergence["iterations"].sum()),
                     "difference": float(np.max(np.abs(values - reference) / (1 + np.abs(reference)), initial=0.0))})
    return rows


def show(rows: list[dict[str, Any]], out=None) -> None:
    out = out or sys.stdout
    head = ["configuration", "seconds", "in periods", "equations evaluated", "Jacobians", "Jacobian rows", "iterations",
            "largest difference"]
    table = [[r["configuration"], f"{r['seconds']:.4f}", f"{r['solving']:.4f}", str(r["equations"]), str(r["jacobians"]), str(r["rows"]),
              str(r["iterations"]), f"{r['difference']:.1e}"] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(head)]
    print(" | ".join(h.ljust(w) for h, w in zip(head, widths)), file=out)
    print("-|-".join("-" * w for w in widths), file=out)
    for t in table:
        print(" | ".join([t[0].ljust(widths[0]), *(v.rjust(w) for v, w in zip(t[1:], widths[1:]))]), file=out)


def machine() -> str:
    import scipy

    import pymetrix

    return (f"{platform.processor() or platform.machine()}; {platform.platform()}; Python {platform.python_version()}; "
            f"NumPy {np.__version__}, SciPy {scipy.__version__}, pandas {pd.__version__}; pymetrix {pymetrix.__version__}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--equations", type=int, default=200, help="equations in the generated model (default 200)")
    parser.add_argument("--quarters", type=int, default=40, help="quarters solved (default 40)")
    parser.add_argument("--repeats", type=int, default=7, help="timed solves of each configuration (default 7)")
    parser.add_argument("--model", help="a model file to time instead (any Model.from_file reads)")
    parser.add_argument("--data", help="its data (any file Dataset.read reads)")
    parser.add_argument("--time", help="the data file's column of periods, for a CSV or Excel file (e.g. date)")
    parser.add_argument("--freq", help="their frequency, with --time (e.g. Q)")
    parser.add_argument("--span", nargs=2, metavar=("START", "END"), help="the periods to solve, e.g. 2025Q1 2034Q4")
    args = parser.parse_args(argv)
    print(machine(), end="\n\n")
    if args.model:
        if not (args.data and args.span):
            parser.error("--model needs --data and --span")
        data = Dataset.read(args.data, **{k: v for k, v in (("time", args.time), ("freq", args.freq)) if v})
        model, span = Model.from_file(args.model), tuple(args.span)
        print(f"{args.model}: {len(model.equations)} equations, {span[0]} to {span[1]}")
    else:
        model, data, span = generate(args.equations, args.quarters)
        print(f"Generated model: {len(model.equations)} equations, {span[0]} to {span[1]} ({args.quarters} quarters)")
    print(f"Median of {args.repeats} solves, in seconds; differences scaled as |x - x_default| / (1 + |x_default|)\n")
    show(measure(model, data, span, args.repeats))


if __name__ == "__main__":
    main()
