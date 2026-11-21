"""Parameter stability: Brown-Durbin-Evans recursive residuals, CUSUM and CUSUM of squares."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .._random import generator as make_rng
from ..errors import SpecificationError
from ..estimation import FittedEquation
from .base import Test, TestResult, register
from .equation_tests import _ols_parts

if TYPE_CHECKING:
    from ..data import Dataset

_CUSUM_5PCT = 0.948  # Brown, Durbin and Evans (1975)


def recursive_residuals(fit: FittedEquation, test: str = "recursive residuals") -> tuple[np.ndarray, pd.Index, int]:
    """Standardised one-step prediction errors w[t] for t = k .. n-1, each from a fit to the first t observations."""
    y, X, _ = _ols_parts(fit, test, linear=True)
    if not isinstance(fit.resid.index, pd.PeriodIndex):  # type: ignore[union-attr]
        raise SpecificationError(f"{test} needs a single time series")
    n, k = X.shape
    if n - k < 3:
        raise SpecificationError(f"{test} needs at least {k + 3} observations")
    w = np.empty(n - k)
    for i, t in enumerate(range(k, n)):
        Xt = X[:t]
        b = np.linalg.lstsq(Xt, y[:t], rcond=None)[0]
        inv = np.linalg.pinv(Xt.T @ Xt)
        x = X[t]
        w[i] = (y[t] - x @ b) / np.sqrt(1 + x @ inv @ x)
    return w, fit.resid.index[k:], k  # type: ignore[union-attr]


def _cusum(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(w)
    path = np.cumsum(w) / np.std(w, ddof=1)
    scale = np.sqrt(m) + 2 * np.arange(1, m + 1) / np.sqrt(m)  # boundary per unit of a
    return path, scale


def _cusumsq(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(w)
    return np.cumsum(w**2) / np.sum(w**2), np.arange(1, m + 1) / m


@lru_cache(maxsize=64)
def _cusumsq_null(m: int, reps: int = 20000, seed: int = 20260101) -> np.ndarray:
    """Simulated null distribution of max |S_r - r/m| (sorted), for m recursive residuals."""
    rng = make_rng(seed)
    out = np.empty(reps)
    expected = np.arange(1, m + 1) / m
    for start in range(0, reps, 1000):
        w2 = rng.standard_normal((min(1000, reps - start), m)) ** 2
        S = np.cumsum(w2, axis=1) / w2.sum(axis=1, keepdims=True)
        out[start : start + len(S)] = np.abs(S - expected).max(axis=1)
    return np.sort(out)


def recursive_table(fit: FittedEquation) -> Dataset:
    from ..data import Dataset
    from ..metadata import DatasetMeta

    w, index, k = recursive_residuals(fit)
    path, scale = _cusum(w)
    sq, expected = _cusumsq(w)
    c = float(np.quantile(_cusumsq_null(len(w)), 0.95))
    frame = pd.DataFrame(
        {
            "recursive_resid": w,
            "cusum": path,
            "cusum_lower": -_CUSUM_5PCT * scale,
            "cusum_upper": _CUSUM_5PCT * scale,
            "cusumsq": sq,
            "cusumsq_expected": expected,
            "cusumsq_lower": expected - c,
            "cusumsq_upper": expected + c,
        },
        index=index,
    )
    meta = DatasetMeta(title=f"Recursive residuals: {fit.equation.text}", description="Bands are 5% boundaries; CUSUMSQ bands are simulated.")
    return Dataset.from_frame(frame, meta=meta)


@register
@dataclass(frozen=True)
class CUSUM(Test):
    """Brown-Durbin-Evans CUSUM test on recursive residuals."""

    maturity: ClassVar[str] = "stable"
    name = "cusum"
    topics = ("diagnostic-tests",)
    references = (
        ("Brown, R. L., Durbin, J. and Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. Journal of the Royal Statistical Society, Series B (Methodological) 37(2), 149-192. doi:10.1111/j.2517-6161.1975.tb01532.x"),
    )
    null = "the coefficients are stable (no systematic drift in recursive residuals)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        w, _, _ = recursive_residuals(fit, self.name)
        path, scale = _cusum(w)
        a = float(np.max(np.abs(path) / scale))
        p = min(1.0, 2 * (1 - sps.norm.cdf(3 * a) + np.exp(-4 * a * a) * sps.norm.cdf(a)))
        return TestResult("CUSUM", self.null, a, None, float(p), "Brown-Durbin-Evans boundary crossing",
                          critical_values={"10%": 0.850, "5%": _CUSUM_5PCT, "1%": 1.143})


@register
@dataclass(frozen=True)
class CUSUMSQ(Test):
    """CUSUM of squares on recursive residuals; critical values by simulation of the exact null."""

    maturity: ClassVar[str] = "stable"
    name = "cusumsq"
    topics = ("diagnostic-tests",)
    references = (
        ("Brown, R. L., Durbin, J. and Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. Journal of the Royal Statistical Society, Series B (Methodological) 37(2), 149-192. doi:10.1111/j.2517-6161.1975.tb01532.x"),
    )
    null = "the error variance is stable"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        w, _, _ = recursive_residuals(fit, self.name)
        sq, expected = _cusumsq(w)
        d = float(np.max(np.abs(sq - expected)))
        null = _cusumsq_null(len(w))
        p = float(1 - np.searchsorted(null, d, side="left") / len(null))
        crit = {f"{int(100 * a)}%": float(np.quantile(null, 1 - a)) for a in (0.10, 0.05, 0.01)}
        return TestResult("CUSUM of squares", self.null, d, len(w), p, "simulated", critical_values=crit)
