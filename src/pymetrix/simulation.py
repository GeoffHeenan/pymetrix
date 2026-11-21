"""Monte Carlo experiments, and bootstrap settings.

    periods = pd.period_range("2000Q1", periods=100, freq="Q")
    def dgp(rng):                                    # one simulated dataset from a numpy Generator
        x = rng.normal(size=100)
        return Dataset.from_frame(pd.DataFrame({"y": 1 + 0.5 * x + rng.normal(size=100), "x": x}, index=periods))

    mc = MonteCarlo(dgp, lambda d: Equation("y ~ x").fit(d), reps=1000, truth={"x": 0.5}).run()
    mc.summary()                                     # mean, s.d., bias, RMSE of each quantity; failures
    mc.draws                                         # one row per replication
    MonteCarlo(dgp, lambda d: Equation("y ~ x").fit(d).test("wald", "x = 0.5"), reps=1000).run().rejection()

Each replication draws from its own random stream (``SeedSequence(seed).spawn(reps)``), so the results are the same
whatever ``workers`` is. The estimator may return a number, a sequence, a mapping of names to numbers, a test result
(its statistic and p-value are kept) or a fitted result (its parameters). A replication that fails is counted and its
error kept; it does not stop the others. With ``workers`` above 1 the replications run in other processes when the
functions can be sent there (defined at the top level of a module, not lambdas); otherwise here, with a note.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .errors import SpecificationError
from .output import Table
from ._random import generator as make_rng, get_random_generator, seed_sequence

__all__ = ["Bootstrap", "MonteCarlo", "MonteCarloResult"]


@dataclass(frozen=True)
class Bootstrap:
    """Bootstrap settings for the methods that take them: residual, wild or block resampling."""

    kind: Literal["residual", "wild", "block"] = "residual"
    reps: int = 999
    block_length: int | None = None
    seed: int | None = None


def _values(out: Any) -> dict[str, float]:
    """The quantities an estimator returned, by name."""
    from .stats.base import TestResult

    if isinstance(out, TestResult):
        return {"statistic": float(out.statistic), **({"p_value": float(out.p_value)} if out.p_value is not None else {})}
    params = getattr(out, "params", None)
    if params is not None and hasattr(params, "names") and hasattr(params, "values"):
        return {str(n): float(v) for n, v in zip(params.names, params.values)}
    if isinstance(out, pd.Series):
        return {str(k): float(v) for k, v in out.items()}
    if isinstance(out, Mapping):
        return {str(k): float(v) for k, v in out.items()}
    if np.ndim(out) == 0:
        return {"value": float(out)}
    return {f"value{i}": float(v) for i, v in enumerate(np.ravel(out))}


def _replicate(dgp: Callable[[np.random.Generator], Any], estimator: Callable[[Any], Any],
               seeds: Sequence[np.random.SeedSequence]) -> list[tuple[dict[str, float] | None, str | None]]:
    out = []
    for s in seeds:
        try:
            out.append((_values(estimator(dgp(make_rng(s)))), None))
        except Exception as exc:  # noqa: BLE001 - one failed replication must not stop the others
            out.append((None, f"{type(exc).__name__}: {exc}"))
    return out


@dataclass(frozen=True, eq=False)
class MonteCarlo:
    """A Monte Carlo experiment: ``reps`` datasets from ``dgp`` (a function of a numpy Generator), each given to
    ``estimator``; ``truth`` (a mapping of quantity to true value) adds bias and RMSE."""

    dgp: Any  # a function of a numpy Generator returning a dataset (or anything the estimator takes)
    estimator: Callable[[Any], Any]
    reps: int = 1000
    seed: int = 0
    workers: int = 1
    truth: Mapping[str, float] = field(default_factory=dict)

    def run(self) -> MonteCarloResult:
        if not callable(self.dgp):
            raise SpecificationError("dgp is a function of a numpy Generator returning one simulated dataset, e.g. "
                                     "lambda rng: model.simulate(100, seed=rng.integers(2**31))")
        reps, workers = int(self.reps), max(1, int(self.workers))
        if reps < 1:
            raise SpecificationError("reps must be at least 1")
        seeds = seed_sequence(int(self.seed)).spawn(reps)
        notes = []
        results = None
        if workers > 1 and reps > 1:
            from ._processes import run_in_processes

            chunks = [seeds[i::workers] for i in range(min(workers, reps))]
            parts = run_in_processes([(_replicate, (self.dgp, self.estimator, c)) for c in chunks], workers, what="replications")
            if parts is None:
                notes.append("The functions could not be sent to other processes (lambdas, or defined in a script): run here")
            else:
                results = [None] * reps
                for i, part in enumerate(parts):
                    for j, r in enumerate(part):
                        results[i + j * len(chunks)] = r
        if results is None:
            from .progress import report

            results = []
            for start in range(0, reps, 50):
                report(start / reps, f"replication {start + 1} of {reps}")
                results += _replicate(self.dgp, self.estimator, seeds[start:start + 50])
        rows = [r for r, _ in results]
        failures = {i: e for i, (_, e) in enumerate(results) if e is not None}
        if len(failures) == reps:
            raise SpecificationError(f"every replication failed; the first: {next(iter(failures.values()))}")
        draws = pd.DataFrame([r if r is not None else {} for r in rows])
        draws.index.name = "replication"
        return MonteCarloResult(self, draws, failures, tuple(notes), get_random_generator())


@dataclass(frozen=True, eq=False)
class MonteCarloResult:
    """The outcome of a Monte Carlo experiment: each replication's quantities (``draws``; failed ones are missing)."""

    spec: MonteCarlo
    draws: pd.DataFrame
    failures: Mapping[int, str]
    notes: tuple[str, ...] = ()
    random_generator: str = "pcg64"  # the bit generator the replications drew with (pymetrix.set_random_generator)

    def moments(self) -> pd.DataFrame:
        """Mean, standard deviation, quantiles, and with a true value bias and RMSE, of each quantity."""
        d = self.draws
        out = pd.DataFrame({"mean": d.mean(), "s.d.": d.std(ddof=1), "5%": d.quantile(0.05), "median": d.median(),
                            "95%": d.quantile(0.95), "replications": d.notna().sum()})
        truth = pd.Series({k: float(v) for k, v in self.spec.truth.items()}, dtype=float).reindex(d.columns)
        if truth.notna().any():
            out["true"] = truth
            out["bias"] = out["mean"] - truth
            out["RMSE"] = np.sqrt(((d - truth) ** 2).mean())
        return out

    def rejection(self, levels: Sequence[float] = (0.01, 0.05, 0.10), p_value: str = "p_value") -> pd.Series:
        """The share of replications whose ``p_value`` is below each level, with its binomial standard error."""
        if p_value not in self.draws:
            raise SpecificationError(f"no {p_value!r} among the quantities; the estimator returns a test result or a mapping with it")
        p = self.draws[p_value].dropna()
        rates = {f"{level:.0%}": float((p < level).mean()) for level in levels}
        se = {f"{level:.0%} s.e.": float(np.sqrt(level * (1 - level) / len(p))) for level in levels}
        return pd.Series({**rates, **se}, name="rejection rate")

    def summary(self) -> Table:
        notes = [f"{len(self.draws)} replications, seed {self.spec.seed}; {len(self.failures)} failed"
                 + (f" (the first: {next(iter(self.failures.values()))})" if self.failures else ""), *self.notes]
        if "p_value" in self.draws:
            r = self.rejection()
            notes.append("Rejection rates: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items() if "s.e." not in k))
        return Table.from_frame(self.moments(), title="Monte Carlo experiment", notes=notes)

    def __repr__(self) -> str:
        return f"<MonteCarloResult: {len(self.draws)} replications of {len(self.draws.columns)} quantities, {len(self.failures)} failed>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
