"""More unit-root tests: Ng and Perron's M tests and Elliott, Rothenberg and Stock's point-optimal test (both on
GLS-detrended data), and Perron's test with a break at a known date.

    data.test("ng_perron", "y", trend="c")        # MZa, MZt, MSB and MPT; the statistic reported is MZa
    data.test("ers_point_optimal", "y", trend="ct")
    data.test("perron_break", "y", date="1973Q1", model="C")

GLS detrending (ERS 1996) quasi-differences the series and the deterministic terms with a = 1 + cbar / T (cbar = -7
with a constant, -13.5 with a trend) and regresses one on the other; the detrended series is y - z'delta. The long-run
variance is Perron and Ng's autoregressive spectral estimate s2 = sigma2_k / (1 - b(1))^2, from the ADF regression of the
detrended series, with k chosen by the modified AIC (Ng and Perron 2001) or given. Then, with kappa = sum_{t=2}^T
y_{t-1}^2 / T^2,

    MZa = (y_T^2 / T - s2) / (2 kappa),  MSB = sqrt(kappa / s2),  MZt = MZa MSB,
    MPT = (cbar^2 kappa - cbar y_T^2 / T) / s2  (constant),  (cbar^2 kappa + (1 - cbar) y_T^2 / T) / s2  (trend),

and ERS's PT = (S(a) - a S(1)) / s2 with S the sums of squared residuals of the quasi-differenced regressions.
Critical values are Ng and Perron's (2001, Table 1) and ERS's asymptotic ones; small values reject the unit root,
except MSB, which rejects when small too.

Perron's (1989) test adds break dummies at the known date T_B to the Dickey-Fuller regression: model "A" a level
shift (DU and the one-time dummy D(TB)), "B" a change of slope (DT*), "C" both. Its critical values depend on the
break fraction lambda; the p-value here is simulated under the null (a random walk with the same T and break date),
with a fixed seed, so it repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..data import Dataset
from ..errors import SpecificationError
from ..forms import Widget, advanced
from .base import Test, TestResult, register
from .data_tests import _int_or, _series
from .._random import generator as make_rng

CBAR = {"c": -7.0, "ct": -13.5}
# Ng and Perron (2001), Table 1: 1%, 5%, 10% asymptotic critical values
NG_PERRON_CRITICAL = {
    "c": {"MZa": (-13.8, -8.1, -5.7), "MZt": (-2.58, -1.98, -1.62), "MSB": (0.174, 0.233, 0.275), "MPT": (1.78, 3.17, 4.45)},
    "ct": {"MZa": (-23.8, -17.3, -14.2), "MZt": (-3.42, -2.91, -2.62), "MSB": (0.143, 0.168, 0.185), "MPT": (4.03, 5.48, 6.67)},
}
# Elliott, Rothenberg and Stock (1996), Table 1, T = infinity
ERS_CRITICAL = {"c": (1.99, 3.26, 4.48), "ct": (3.96, 5.62, 6.89)}


def _deterministic(T: int, trend: str) -> np.ndarray:
    return np.ones((T, 1)) if trend == "c" else np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float)])


def _quasi_difference(M: np.ndarray, a: float) -> np.ndarray:
    out = M.copy()
    out[1:] = M[1:] - a * M[:-1]
    return out


def gls_detrend(y: np.ndarray, trend: str) -> tuple[np.ndarray, float]:
    """(the GLS-detrended series, S(a): the quasi-differenced regression's sum of squared residuals)."""
    T = len(y)
    a = 1 + CBAR[trend] / T
    z = _deterministic(T, trend)
    ya, za = _quasi_difference(y[:, None], a)[:, 0], _quasi_difference(z, a)
    delta = np.linalg.lstsq(za, ya, rcond=None)[0]
    e = ya - za @ delta
    return y - z @ delta, float(e @ e)


def _ssr_at(y: np.ndarray, trend: str, a: float) -> float:
    z = _deterministic(len(y), trend)
    ya, za = _quasi_difference(y[:, None], a)[:, 0], _quasi_difference(z, a)
    e = ya - za @ np.linalg.lstsq(za, ya, rcond=None)[0]
    return float(e @ e)


def _adf_design(x: np.ndarray, k: int, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Delta x_t on x_{t-1} and k lagged differences, over the common sample t = kmax + 1, ..., T - 1."""
    dx = np.diff(x)
    X = np.column_stack([x[kmax:-1]] + [dx[kmax - j:len(dx) - j] for j in range(1, k + 1)]) if k else x[kmax:-1, None]
    return X, dx[kmax:]


def ar_spectral_variance(x: np.ndarray, lags: Any = None, max_lags: int | None = None) -> tuple[float, int]:
    """Perron and Ng's autoregressive long-run variance of the detrended ``x``, with the lag chosen by the modified
    AIC (Ng and Perron 2001) over a common sample unless given."""
    T = len(x)
    kmax = int(12 * (T / 100) ** 0.25) if max_lags is None else int(max_lags)
    if isinstance(lags, int):
        k, kmax_used = lags, lags
    else:
        best, k = np.inf, 0
        for kk in range(kmax + 1):
            X, dy = _adf_design(x, kk, kmax)
            b = np.linalg.lstsq(X, dy, rcond=None)[0]
            e = dy - X @ b
            s2 = e @ e / len(e)
            tau = b[0] ** 2 * np.sum(X[:, 0] ** 2) / s2
            maic = np.log(s2) + 2 * (tau + kk) / (T - 1 - kmax)
            if maic < best:
                best, k = maic, kk
        kmax_used = k
    X, dy = _adf_design(x, k, kmax_used)
    b = np.linalg.lstsq(X, dy, rcond=None)[0]
    e = dy - X @ b
    s2 = e @ e / len(e)
    return float(s2 / (1 - b[1:].sum()) ** 2), k


def m_statistics(y: np.ndarray, trend: str, lags: Any = None, max_lags: int | None = None) -> dict[str, float]:
    """Ng and Perron's MZa, MZt, MSB and MPT, with the long-run variance and the lag used."""
    if trend not in CBAR:
        raise SpecificationError("trend is 'c' (a constant) or 'ct' (a constant and a linear trend)")
    T = len(y)
    yd, _ = gls_detrend(y, trend)
    s2, k = ar_spectral_variance(yd, lags, max_lags)
    kappa = np.sum(yd[:-1] ** 2) / T**2
    end = yd[-1] ** 2 / T
    cbar = CBAR[trend]
    MZa = (end - s2) / (2 * kappa)
    MSB = np.sqrt(kappa / s2)
    MPT = (cbar**2 * kappa - cbar * end) / s2 if trend == "c" else (cbar**2 * kappa + (1 - cbar) * end) / s2
    return {"MZa": float(MZa), "MZt": float(MZa * MSB), "MSB": float(MSB), "MPT": float(MPT), "s2": s2, "lags": k}


@register
@dataclass(frozen=True)
class NgPerron(Test):
    """Ng-Perron M unit-root tests (MZa, MZt, MSB, MPT) on GLS-detrended data, with good size and power even when
    the errors have a large negative moving-average root.

    The null is a unit root; the alternative is stationarity around a level (c) or a linear trend (ct). Detrending
    by GLS gives more power than ADF against roots near one, and the modified statistics keep their size where
    Phillips-Perron tests reject far too often. All four reject when small. The statistic shown is MZa; MZt, MSB and
    MPT and their critical values are in the details. There is no p-value: compare each statistic with Ng and
    Perron's asymptotic critical values. The lag of the autoregression for the long-run variance is chosen by the
    modified AIC unless given; Ng and Perron recommend it because a short lag chosen by AIC or BIC can distort the
    size badly.
    """

    maturity: ClassVar[str] = "beta"
    variable: str
    trend: Annotated[Literal["c", "ct"], Widget("", help="terms under the alternative: c a constant, ct a constant and linear trend (trending series)")] = "c"
    lags: Annotated[str | int, advanced(help="lags of the AR regression for the long-run variance, or maic (modified AIC, recommended)")] = "maic"
    max_lags: Annotated[str | int, advanced(help="largest lag the modified AIC tries; empty = 12*(T/100)^(1/4)")] = ""

    name = "ng_perron"
    topics = ("unit-roots",)
    references = (
        ("Ng, S. and Perron, P. (2001). Lag length selection and the construction of unit root tests with good size and power. Econometrica 69(6), 1519-1554. doi:10.1111/1468-0262.00256"),
        ("Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996). Efficient tests for an autoregressive unit root. Econometrica 64(4), 813-836. doi:10.2307/2171846"),
    )
    null = "the series has a unit root"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        y = _series(data, self.variable, "Ng-Perron")
        stats = m_statistics(y, self.trend, _int_or(self.lags, ("maic",)), _int_or(self.max_lags, ()))
        crit = NG_PERRON_CRITICAL[self.trend]
        details = {**{k: stats[k] for k in ("MZt", "MSB", "MPT")}, "long-run variance": stats["s2"], "lags": stats["lags"],
                   "observations": len(y), **{f"{name} {level}": v for name, vals in crit.items()
                                              for level, v in zip(("1%", "5%", "10%"), vals) if name != "MZa"}}
        return TestResult(f"Ng-Perron MZa ({self.variable})", self.null, stats["MZa"], None, None, "Ng-Perron (2001)",
                          critical_values=dict(zip(("1%", "5%", "10%"), crit["MZa"])), details=details)


@register
@dataclass(frozen=True)
class ERSPointOptimal(Test):
    """Elliott-Rothenberg-Stock point-optimal unit-root test (PT): the likelihood ratio of a unit root against the
    local alternative a = 1 + cbar / T, scaled by the long-run variance.

    The null is a unit root; the alternative is stationarity around a level (c) or a linear trend (ct). The test is
    designed to have nearly the best possible power against roots close to one, where ADF has little. PT rejects
    when small. There is no p-value: compare PT with ERS's asymptotic critical values, which are only approximate in
    short samples. The long-run variance comes from an autoregression whose lag is chosen by the modified AIC
    unless given; as for the Ng-Perron tests, too short a lag can distort the size.
    """

    maturity: ClassVar[str] = "beta"
    variable: str
    trend: Annotated[Literal["c", "ct"], Widget("", help="terms under the alternative: c a constant, ct a constant and linear trend (trending series)")] = "c"
    lags: Annotated[str | int, advanced(help="lags of the AR regression for the long-run variance, or maic (modified AIC, recommended)")] = "maic"
    max_lags: Annotated[str | int, advanced(help="largest lag the modified AIC tries; empty = 12*(T/100)^(1/4)")] = ""

    name = "ers_point_optimal"
    topics = ("unit-roots",)
    references = (
        ("Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996). Efficient tests for an autoregressive unit root. Econometrica 64(4), 813-836. doi:10.2307/2171846"),
    )
    null = "the series has a unit root"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        y = _series(data, self.variable, "ERS point-optimal")
        if self.trend not in CBAR:
            raise SpecificationError("trend is 'c' or 'ct'")
        T = len(y)
        a = 1 + CBAR[self.trend] / T
        yd, S_a = gls_detrend(y, self.trend)
        s2, k = ar_spectral_variance(yd, _int_or(self.lags, ("maic",)), _int_or(self.max_lags, ()))
        PT = (S_a - a * _ssr_at(y, self.trend, 1.0)) / s2
        return TestResult(f"ERS point-optimal ({self.variable})", self.null, float(PT), None, None, "ERS (1996)",
                          critical_values=dict(zip(("1%", "5%", "10%"), ERS_CRITICAL[self.trend])),
                          details={"long-run variance": s2, "lags": k, "observations": T})


def _perron_regression(y: np.ndarray, tb: int, model: str, k: int) -> float:
    """The t statistic on y_{t-1} - 1... as the coefficient on y_{t-1} in Delta y_t on the break terms, a trend, y_{t-1}
    and k lagged differences (tb: the index of the last period before the break)."""
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)
    DU = (t > tb + 1).astype(float)
    DT = np.where(t > tb + 1, t - (tb + 1), 0.0)
    DTB = (t == tb + 2).astype(float)
    cols = [np.ones(T), t]
    if model in ("A", "C"):
        cols += [DU, DTB]
    if model in ("B", "C"):
        cols.append(DT)
    Z = np.column_stack(cols)
    dy = np.diff(y)
    start = k
    X = np.column_stack([Z[1:][start:], y[:-1][start:]] + [dy[start - j:len(dy) - j] for j in range(1, k + 1)])
    target = dy[start:]
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ target
    e = target - X @ b
    s2 = e @ e / (len(e) - X.shape[1])
    j = Z.shape[1]
    return float(b[j] / np.sqrt(s2 * XtX[j, j]))


def perron_simulated(T: int, tb: int, model: str, k: int, reps: int = 2000, seed: int = 0) -> np.ndarray:
    rng = make_rng(seed)
    return np.array([_perron_regression(np.cumsum(rng.standard_normal(T)), tb, model, k) for _ in range(reps)])


@register
@dataclass(frozen=True)
class PerronBreak(Test):
    """Perron's (1989) unit-root test with a structural break at a known date: a level shift (A), a change of slope
    (B) or both (C) under the alternative of trend stationarity.

    The null is a unit root with the break; the alternative is stationarity around a broken trend. An unmodelled
    break makes ADF fail to reject, which this test corrects. The date must be known from outside the data (an oil
    shock, a policy change): picking it by looking at the series makes the test reject too often, and for an unknown
    date a test that searches over dates, such as Zivot-Andrews, is the right one. The statistic is the t-ratio on
    the lagged level and rejects when large and negative; the p-value and critical values are simulated for this
    sample size, break date, model and lag. The lag is fixed: too few leave serial correlation that distorts the
    size.
    """

    maturity: ClassVar[str] = "beta"
    variable: str
    date: Annotated[str, Widget("period", help="the first period of the new regime, known from outside the data")] = ""
    model: Annotated[Literal["A", "B", "C"], Widget("choice", help="A: a level shift; B: a change of slope; C: both")] = "C"
    lags: Annotated[int, advanced(help="lagged differences absorbing serial correlation; too few distort the size")] = 0
    reps: Annotated[int, advanced(help="replications for the simulated p-value; more, less simulation error")] = 2000
    seed: Annotated[int, advanced(help="random seed for the simulated p-value")] = 0

    name = "perron_break"
    topics = ("unit-roots",)
    references = (
        ("Perron, P. (1989). The great crash, the oil price shock, and the unit root hypothesis. Econometrica 57(6), 1361-1401. doi:10.2307/1913712"),
    )
    null = "the series has a unit root (with the break)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        if data.kind != "time_series":
            raise SpecificationError("the Perron test needs a single time series")
        s = data[self.variable].dropna()
        if not self.date:
            raise SpecificationError("give the break date, the first period of the new regime: date='1973Q1'")
        try:
            first = s.index.get_loc(pd.Period(self.date, freq=s.index.freq))
        except KeyError:
            raise SpecificationError(f"{self.date} is not in the sample of {self.variable}") from None
        y = s.to_numpy(float)
        T, tb = len(y), int(first) - 1
        if tb < 2 + self.lags or tb > T - 4:
            raise SpecificationError("the break must leave a few periods on each side")
        stat = _perron_regression(y, tb, self.model, self.lags)
        sims = perron_simulated(T, tb, self.model, self.lags, self.reps, self.seed)
        p = float((np.sum(sims <= stat) + 1) / (len(sims) + 1))
        crit = {f"{q:.0%}": float(np.quantile(sims, q)) for q in (0.01, 0.05, 0.10)}
        return TestResult(f"Perron break test, model {self.model} ({self.variable})", self.null, stat, None, p,
                          "simulated", critical_values=crit,
                          details={"break fraction": (tb + 1) / T, "break date": self.date, "lags": self.lags, "replications": self.reps})
