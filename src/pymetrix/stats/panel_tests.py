"""Panel unit-root and stationarity tests: Fisher-type (Maddala-Wu, Choi), Im-Pesaran-Shin, Levin-Lin-Chu, Breitung,
Pesaran's CIPS and Hadri.

Fisher-type tests combine the p-values of ADF tests run entity by entity
(computed by statsmodels): Maddala and Wu's P = -2 sum log p_i is chi-squared with 2N degrees
of freedom, and Choi's Z = sum Phi^-1(p_i) / sqrt(N) is standard normal, under the null that
every series has a unit root; they suit unbalanced panels. Hadri's LM test reverses the null
(every series is stationary): the average of the entities' KPSS statistics, standardised with
its known mean and variance, is standard normal under the null and large when some series have
unit roots. It needs a balanced panel. With autocorrelated errors the original test (lags=0)
rejects far too often; give Bartlett lags for a long-run variance.

Im, Pesaran and Shin average the entities' ADF t-statistics and standardise the average with the
mean and variance of each t-statistic under the null (W t-bar, standard normal as N grows). Their
paper tabulates those moments for a grid of T and lags; here they are simulated for each entity's
own T and lags (seeded, 100,000 random walks, cached), so unbalanced panels and any T are exact up
to simulation error, which is reported. Levin, Lin and Chu pool the entities' orthogonalised ADF
regressions into one with a common coefficient, and adjust its t-statistic with the mean and
standard-deviation corrections of their Table 2; it needs a balanced panel. Both follow gretl's
conventions (Levin-Lin-Chu agrees with gretl's levinlin to the printed precision). Choi's tests
(inverse normal and logit) are reported with the Fisher-type test.

All of these but CIPS assume the entities are independent. A common shock (a world business cycle,
global interest rates) makes them over-reject. Pesaran's CIPS test augments each entity's ADF
regression with the cross-section mean's lagged level and differences, which absorb a single
common factor, and averages the resulting t-statistics. Its distribution is non-standard; Pesaran
tabulates critical values for a grid of N and T, and here they are simulated for the panel's own N,
T and lags (seeded, cached), which also gives a p-value. The simulated values reproduce Pesaran's
Table II to within about 0.02. Unbalanced panels are taken as plm's cipstest takes them: each period's
cross-section means average the entities whose regression row is complete then, and the null is simulated
with the same periods missing.

Breitung's test (Breitung 2000) pools like Levin, Lin and Chu but transforms the data so that the pooled
t-statistic needs no bias adjustment: the levels are taken from each series' first value (and, with a trend,
detrended by the first and last), and with a trend the differences are forward orthogonalised. Lags are
removed first by each series' own autoregression (prewhitening). Breitung and Das's robust version divides by
a sandwich of the residuals' cross-section covariance, for panels with a common factor. No other package at
hand computes it, so it is checked by hand and by its null distribution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated, ClassVar, Literal

import numpy as np

from pymetrix._lazy import stats as sps

from .._random import generator as make_rng
from ..adapters import require
from ..adapters import statsmodels as statsmodels_adapter
from ..adapters.base import AdapterSpec, BackendOptions, check_backend_options
from ..data import Dataset
from ..errors import SpecificationError
from ..forms import Widget, advanced
from . import adf_arrays
from .base import Test, TestResult, register
from .data_tests import _int_or


def _by_entity(data: Dataset, name: str, test: str) -> list[tuple[str, np.ndarray]]:
    if data.kind != "panel" or len(data.entity_axes) != 1:
        raise SpecificationError(f"{test} needs a panel with one entity axis; for one series use ADF or KPSS")
    entity = data.entity_axes[0].name
    out = []
    for key, s in data[name].groupby(level=entity):
        values = s.dropna().to_numpy(float)
        if len(values):
            out.append((str(key), values))
    if len(out) < 2:
        raise SpecificationError(f"{test} needs at least two entities with data")
    return out


@register
@dataclass(frozen=True)
class FisherADF(Test):
    """Fisher-type panel unit-root test: ADF tests entity by entity, combined (Maddala-Wu chi-squared; Choi Z).

    Each entity's ADF test is run on its own (MacKinnon p-values) and the p-values are combined. The null is that
    every series has a unit root; the alternative is that at least one is stationary, so a rejection does not say how
    many are: the details count the entities rejecting at 5%. Entities may have their own lags and sample lengths,
    which suits unbalanced panels. The combination assumes the entities' tests are independent; a common shock makes
    it reject too often, and then CIPS or Breitung's robust statistic is the better choice. Choi's inverse-normal and
    logit statistics are in the details.
    """

    maturity: ClassVar[str] = "beta"
    wraps: ClassVar[str] = "statsmodels.tsa.stattools.adfuller"

    variable: str
    lags: Annotated[str | int, advanced(help="lagged differences in each ADF regression, or aic / bic to choose entity by entity")] = "aic"
    trend: Annotated[Literal["c", "ct", "n"], Widget("", help="each entity's own terms: c a constant, ct a constant and trend (trending series), n none")] = "c"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other arguments of each statsmodels adfuller")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.FISHER_ADF

    name = "fisher_adf"
    topics = ("unit-roots", "panel-data")
    references = (
        ("Maddala, G. S. and Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. Oxford Bulletin of Economics and Statistics 61(S1), 631-652. doi:10.1111/1468-0084.0610s1631"),
        "Choi, I. (2001). Unit root tests for panel data. Journal of International Money and Finance 20(2), 249-272. doi:10.1016/s0261-5606(00)00048-6",
    )
    null = "every series in the panel has a unit root"
    applies_to = (Dataset,)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def run(self, data: Dataset) -> TestResult:
        stattools = require("statsmodels.tsa.stattools")
        lags = _int_or(self.lags, ("aic", "bic"))
        p = []
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:  # one ADF per entity, the same arguments
            for key, x in _by_entity(data, self.variable, "the Fisher-type test"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    if isinstance(lags, int):
                        result = run.call("test", stattools.adfuller, x, maxlag=_entity_lag(lags, x, self.trend),
                                          regression=self.trend, autolag=None)
                    else:
                        result = run.call("test", stattools.adfuller, x, maxlag=_entity_lag(None, x, self.trend),
                                          regression=self.trend, autolag={"aic": "AIC", "bic": "BIC"}[lags or "aic"])
                adf_arrays.degenerate_statistic(float(result[0]), f"the Fisher-type test, entity {key}")
                p.append(float(result[1]))
        p_arr = np.clip(np.asarray(p), 1e-300, 1 - 1e-16)
        n = len(p_arr)
        stat = float(-2 * np.sum(np.log(p_arr)))
        z = float(np.sum(sps.norm.ppf(p_arr)) / np.sqrt(n))
        logit = float(np.sum(np.log(p_arr / (1 - p_arr))) * np.sqrt(3 * (5 * n + 4) / (np.pi**2 * n * (5 * n + 2))))
        return TestResult(
            f"Fisher-type ADF, Maddala-Wu ({self.variable})", self.null, stat, 2 * n, float(sps.chi2.sf(stat, 2 * n)), "chi2",
            details={"Choi Z": z, "Choi p-value": float(sps.norm.cdf(z)),
                     "Choi logit": logit, "Choi logit p-value": float(sps.t.cdf(logit, 5 * n + 4)), "entities": n,
                     "entities rejecting at 5%": int(np.sum(p_arr < 0.05))},
            backend_call=run.record,
        )


@register
@dataclass(frozen=True)
class Hadri(Test):
    """Hadri LM test: the null is that every series is stationary (the panel counterpart of KPSS).

    The null is that every series is stationary around its own level (c) or trend (ct); the alternative is that some
    have a unit root. The statistic averages the entities' KPSS-type LM statistics, standardised to N(0, 1), and
    rejects when large. With Lags 0 (Hadri's original) each series' variance is its residual variance, valid only for
    serially uncorrelated errors: with persistent but stationary data that version rejects far too often, so give
    Bartlett lags or auto. It needs a balanced panel and assumes independent entities; cross-sectional dependence
    also makes it reject too often. Read it with a test whose null is a unit root (IPS, CIPS).
    """

    maturity: ClassVar[str] = "beta"
    variable: str
    trend: Annotated[Literal["c", "ct"], Widget("", help="the null: each series stationary around its own level (c) or linear trend (ct)")] = "c"
    heteroskedastic: Annotated[bool, advanced(help="each series its own error variance (default), or one pooled variance")] = True
    lags: Annotated[str | int, advanced(help="Bartlett lags for a long-run variance when errors are autocorrelated; 0 = none (Hadri's original), auto = 4(T/100)^(2/9)")] = 0

    name = "hadri"
    topics = ("unit-roots", "panel-data")
    references = (
        "Hadri, K. (2000). Testing for stationarity in heterogeneous panel data. Econometrics Journal 3(2), 148-161. doi:10.1111/1368-423x.00043",
    )
    null = "every series in the panel is stationary (around a level, or a trend with trend='ct')"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        series = _by_entity(data, self.variable, "the Hadri test")
        lengths = {len(x) for _, x in series}
        if len(lengths) != 1:
            raise SpecificationError("the Hadri test needs a balanced panel (every entity with the same periods); use balance()")
        T = lengths.pop()
        t = np.arange(1, T + 1)
        X = np.ones((T, 1)) if self.trend == "c" else np.column_stack([np.ones(T), t])
        residuals = [x - X @ np.linalg.lstsq(X, x, rcond=None)[0] for _, x in series]
        lags = _int_or(self.lags, ("auto",))
        q = int(4 * (T / 100) ** (2 / 9)) if lags == "auto" else int(lags or 0)

        def long_run(e: np.ndarray) -> float:
            v = e @ e / T
            for j in range(1, q + 1):
                v += 2 * (1 - j / (q + 1)) * (e[j:] @ e[:-j]) / T
            return float(v)

        pooled = np.mean([long_run(e) for e in residuals])
        lm = []
        for e in residuals:
            sigma2 = long_run(e) if self.heteroskedastic else pooled
            S = np.cumsum(e)
            lm.append(float(S @ S / (T**2 * sigma2)))
        n = len(lm)
        mean, var = (1 / 6, 1 / 45) if self.trend == "c" else (1 / 15, 11 / 6300)
        LM = float(np.mean(lm))
        z = float(np.sqrt(n) * (LM - mean) / np.sqrt(var))
        return TestResult(
            f"Hadri LM ({self.variable})", self.null, z, None, float(sps.norm.sf(z)), "normal",
            details={"LM": LM, "entities": n, "periods": T, "variance": "heteroskedastic" if self.heteroskedastic else "pooled",
                     "Bartlett lags": q},
        )


def _entity_lag(lags: int | None, x: np.ndarray, trend: str) -> int:
    """A fixed lag, checked to leave the entity's ADF regression a residual degree of freedom, or the largest lag a
    search tries (``adf_arrays.default_max_lag``: statsmodels' rule, capped so short series keep 4)."""
    ntrend = 0 if trend == "n" else len(trend)
    return adf_arrays.default_max_lag(len(x), ntrend) if lags is None else adf_arrays.check_lag(lags, len(x), ntrend)


def _entity_adf(data: Dataset, variable: str, lags: str | int, trend: str, test: str) -> list[tuple[str, np.ndarray, float, int]]:
    """Each entity's series, ADF t-statistic and lag order (fixed, or chosen by aic / bic), by statsmodels."""
    stattools = require("statsmodels.tsa.stattools")
    chosen = _int_or(lags, ("aic", "bic"))
    out = []
    for key, x in _by_entity(data, variable, test):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if isinstance(chosen, int):
                r = stattools.adfuller(x, maxlag=_entity_lag(chosen, x, trend), regression=trend, autolag=None)
            else:
                r = stattools.adfuller(x, maxlag=_entity_lag(None, x, trend), regression=trend,
                                       autolag={"aic": "AIC", "bic": "BIC"}[chosen or "aic"])
        adf_arrays.degenerate_statistic(float(r[0]), f"{test}, entity {key}")
        out.append((key, x, float(r[0]), int(r[2])))
    return out


def _deterministic(n: int, trend: str) -> np.ndarray:
    cols = ([np.ones(n)] if trend != "n" else []) + ([np.arange(1, n + 1.0)] if trend == "ct" else [])
    return np.column_stack(cols) if cols else np.empty((n, 0))


def _residual(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0] if Z.shape[1] else Y


def _adf_t(Y: np.ndarray, p: int, trend: str) -> np.ndarray:
    """ADF t-statistics for each row of Y: the regression of statsmodels' adfuller(maxlag=p, autolag=None)."""
    dy = np.diff(Y, axis=1)
    R, m = dy.shape
    n = m - p
    cols = [Y[:, p:-1]] + [dy[:, p - j:m - j] for j in range(1, p + 1)]
    if trend != "n":
        cols.append(np.ones((R, n)))
    if trend == "ct":
        cols.append(np.broadcast_to(np.arange(1, n + 1.0), (R, n)))
    X = np.stack(cols, axis=2)
    inv = np.linalg.inv(np.einsum("rti,rtj->rij", X, X))
    b = np.einsum("rij,rj->ri", inv, np.einsum("rti,rt->ri", X, dy[:, p:]))
    u = dy[:, p:] - np.einsum("rti,ri->rt", X, b)
    s2 = np.einsum("rt,rt->r", u, u) / (n - X.shape[2])
    return b[:, 0] / np.sqrt(s2 * inv[:, 0, 0])


@lru_cache(maxsize=256)
def _ips_moments(T: int, p: int, trend: str, replications: int, seed: int) -> tuple[float, float]:
    """Mean and variance of the ADF t-statistic for a Gaussian random walk of length T, by simulation."""
    rng = make_rng(seed)
    chunk = max(200, 2_000_000 // (T * (p + 3)))
    draws = [_adf_t(np.cumsum(rng.normal(size=(min(chunk, replications - k), T)), axis=1), p, trend)
             for k in range(0, replications, chunk)]
    t = np.concatenate(draws)
    return float(t.mean()), float(t.var(ddof=1))


@register
@dataclass(frozen=True)
class ImPesaranShin(Test):
    """Im-Pesaran-Shin test: the average of the entities' ADF t-statistics (t-bar), standardised (W t-bar).

    The null is that every series has a unit root. The alternative lets the autoregressive coefficient differ across
    entities and needs only some of the series to be stationary, so a rejection does not mean that all are. Each
    entity keeps its own lags and sample length; W t-bar is standard normal under the null and rejects when large
    and negative. It assumes independent entities: a common factor makes it reject too often, and CIPS, which
    allows one, is then the better choice. Power is low when T is short, and lower still with trends.
    """

    maturity: ClassVar[str] = "stable"
    variable: str
    lags: Annotated[str | int, advanced(help="lagged differences in each ADF regression, or aic / bic to choose entity by entity")] = "aic"
    trend: Annotated[Literal["c", "ct"], Widget("", help="each entity's own terms: c a constant, ct a constant and trend (trending series)")] = "c"
    replications: Annotated[int, advanced(help="random walks simulated for the null mean and variance of each t-statistic; more, less simulation error")] = 100_000
    seed: Annotated[int, advanced()] = 0

    name = "im_pesaran_shin"
    topics = ("unit-roots", "panel-data")
    references = (
        ("Im, K. S., Pesaran, M. H. and Shin, Y. (2003). Testing for unit roots in heterogeneous panels. Journal of Econometrics 115(1), 53-74. doi:10.1016/s0304-4076(03)00092-7"),
    )
    null = "every series in the panel has a unit root"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        entities = _entity_adf(data, self.variable, self.lags, self.trend, "the Im-Pesaran-Shin test")
        for key, x, _, p in entities:
            if len(x) - 2 * p - 1 - len(self.trend) - 1 < 3:
                raise SpecificationError(f"entity {key} has too few periods ({len(x)}) for {p} lags")
        reps, seed = int(self.replications), int(self.seed)
        moments = [_ips_moments(len(x), p, self.trend, reps, seed) for _, x, _, p in entities]
        n = len(entities)
        tbar = float(np.mean([t for _, _, t, _ in entities]))
        mean = float(np.mean([m for m, _ in moments]))
        var = float(np.mean([v for _, v in moments]))
        w = float(np.sqrt(n) * (tbar - mean) / np.sqrt(var))
        lags = sorted({p for *_, p in entities})
        return TestResult(
            f"Im-Pesaran-Shin W t-bar ({self.variable})", self.null, w, None, float(sps.norm.cdf(w)), "normal",
            details={"t-bar": tbar, "E[t] under the null": mean, "Var[t] under the null": var, "entities": n,
                     "lags": lags[0] if len(lags) == 1 else f"{lags[0]} to {lags[-1]}",
                     "simulation s.e. of W": float(np.sqrt(n / reps))},
        )


# Levin, Lin and Chu (2002), Table 2: mean and standard-deviation adjustments by T~ = T - p - 1, with
# no deterministic terms, a constant, and a constant and trend; the last row is the limit.
_LLC_TABLE = (
    (25, (0.004, 1.049), (-0.554, 0.919), (-0.703, 1.003)),
    (30, (0.003, 1.035), (-0.546, 0.889), (-0.674, 0.949)),
    (35, (0.002, 1.027), (-0.541, 0.867), (-0.653, 0.906)),
    (40, (0.002, 1.021), (-0.537, 0.850), (-0.637, 0.871)),
    (45, (0.001, 1.017), (-0.533, 0.837), (-0.624, 0.842)),
    (50, (0.001, 1.014), (-0.531, 0.826), (-0.614, 0.818)),
    (60, (0.001, 1.011), (-0.527, 0.810), (-0.598, 0.780)),
    (70, (0.000, 1.008), (-0.524, 0.798), (-0.587, 0.751)),
    (80, (0.000, 1.007), (-0.521, 0.789), (-0.578, 0.728)),
    (90, (0.000, 1.006), (-0.520, 0.782), (-0.571, 0.710)),
    (100, (0.000, 1.005), (-0.518, 0.776), (-0.566, 0.695)),
    (250, (0.000, 1.001), (-0.509, 0.742), (-0.533, 0.603)),
    (np.inf, (0.000, 1.000), (-0.500, 0.707), (-0.500, 0.500)),
)


def _llc_adjustment(T_tilde: float, trend: str) -> tuple[float, float]:
    """The first row whose T~ is at least the panel's, as gretl does."""
    col = {"n": 1, "c": 2, "ct": 3}[trend]
    return next(row[col] for row in _LLC_TABLE if row[0] >= T_tilde)


@register
@dataclass(frozen=True)
class LevinLinChu(Test):
    """Levin-Lin-Chu test: a pooled ADF regression with a common autoregressive coefficient, adjusted to N(0, 1).

    The null is that every series has a unit root; the alternative is that all are stationary with the same
    autoregressive coefficient. That is a strong restriction: when only some series may be stationary, IPS or the
    Fisher-type tests suit better. Lags and deterministic terms are each entity's own; the adjusted t-statistic is
    standard normal under the null and rejects when large and negative. It needs a balanced panel and assumes
    independent entities (a common factor makes it reject too often). The adjustments are tabulated from 25 periods
    (after lags) up: shorter panels get those for 25, and the test is best suited to moderate panels.
    """

    maturity: ClassVar[str] = "stable"
    variable: str
    lags: Annotated[str | int, advanced(help="lagged differences in each ADF regression, or aic / bic to choose entity by entity")] = "aic"
    trend: Annotated[Literal["c", "ct", "n"], Widget("", help="each entity's own terms: c a constant, ct a constant and trend (trending series), n none")] = "c"

    name = "levin_lin_chu"
    topics = ("unit-roots", "panel-data")
    references = (
        ("Levin, A., Lin, C.-F. and Chu, C.-S. J. (2002). Unit root tests in panel data: asymptotic and finite-sample properties. Journal of Econometrics 108(1), 1-24. doi:10.1016/s0304-4076(01)00098-7"),
    )
    null = "every series in the panel has a unit root (alternative: every series is stationary, with a common coefficient)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        entities = _entity_adf(data, self.variable, self.lags, self.trend, "the Levin-Lin-Chu test")
        lengths = {len(x) for _, x, _, _ in entities}
        if len(lengths) != 1:
            raise SpecificationError("the Levin-Lin-Chu test needs a balanced panel (every entity with the same periods); use balance()")
        T = lengths.pop()
        mean_p = float(np.mean([p for *_, p in entities]))
        T_tilde = T - mean_p - 1
        K = int(3.21 * T_tilde ** (1 / 3))  # Bartlett truncation for the long-run variance
        E, V, ratios = [], [], []
        for _, y, _, p in entities:
            dy = np.diff(y)
            n = len(dy) - p
            lagged = [dy[p - j:len(dy) - j] for j in range(1, p + 1)]
            Z = np.column_stack([_deterministic(n, self.trend), *lagged]) if lagged else _deterministic(n, self.trend)
            e, v = _residual(dy[p:], Z), _residual(y[p:-1], Z)  # step 1: orthogonalised differences and lagged levels
            delta = (v @ e) / (v @ v)
            sigma_e = np.sqrt(np.sum((e - delta * v) ** 2) / n)
            x = _residual(dy, _deterministic(T - 1, self.trend))  # step 2: long-run variance of the differences
            gamma = [(x[j:] @ x[:len(x) - j]) / (T - 1) for j in range(K + 1)]
            long_run = gamma[0] + 2 * sum((1 - j / (K + 1)) * gamma[j] for j in range(1, K + 1))
            E.append(e / sigma_e)
            V.append(v / sigma_e)
            ratios.append(np.sqrt(long_run) / sigma_e)
        e, v = np.concatenate(E), np.concatenate(V)  # step 3: the pooled regression, adjusted
        N = len(entities)
        delta = float((v @ e) / (v @ v))
        s2 = float(np.sum((e - delta * v) ** 2) / (N * T_tilde))
        se = float(np.sqrt(s2 / (v @ v)))
        t = delta / se
        S_N = float(np.mean(ratios))
        mu, sigma = _llc_adjustment(T_tilde, self.trend)
        z = float((t - N * T_tilde * S_N * se * mu / s2) / sigma)
        return TestResult(
            f"Levin-Lin-Chu adjusted t ({self.variable})", self.null, z, None, float(sps.norm.cdf(z)), "normal",
            details={"pooled coefficient": delta, "unadjusted t": t, "S_N": S_N, "mean adjustment": mu,
                     "s.d. adjustment": sigma, "entities": N, "periods": T, "average lags": mean_p, "Bartlett lags": K},
        )


@register
@dataclass(frozen=True)
class Breitung(Test):
    """Breitung's unbiased panel unit-root test, optionally robust to cross-sectional dependence.

    Breitung (2000); the robust version is Breitung and Das's (2005), standard normal as both N and T grow: with ten
    entities it rejects a true null about 8-11% of the time at 5%, against 40% or more for the standard statistic under
    a strong common factor.

    The null is that every series has a unit root; the alternative, as for Levin, Lin and Chu, is that all are
    stationary with a common coefficient. The statistic rejects when large and negative. Because it needs no bias
    adjustment it keeps more power than Levin-Lin-Chu when each series has its own trend. It needs a balanced panel,
    and the robust statistic needs more periods than entities; use it when a common shock is likely."""

    maturity: ClassVar[str] = "beta"
    variable: str
    lags: Annotated[int, advanced(help="lagged differences removed from each series first (prewhitening), the same for all; 0 = none")] = 0
    trend: Annotated[Literal["c", "ct"], Widget("", help="each entity's own terms: c a constant, ct a constant and trend (trending series)")] = "c"
    robust: Annotated[bool, advanced(help="Breitung and Das's statistic, robust to cross-sectional dependence")] = False

    name = "breitung"
    topics = ("unit-roots", "panel-data")
    references = (
        ("Breitung, J. (2000). The local power of some unit root tests for panel data. Advances in Econometrics 15, 161-178. doi:10.1016/s0731-9053(00)15006-6"),
        ("Breitung, J. and Das, S. (2005). Panel unit root tests under cross-sectional dependence. Statistica Neerlandica 59(4), 414-433. doi:10.1111/j.1467-9574.2005.00299.x"),
    )
    null = "every series in the panel has a unit root (alternative: every series is stationary, with a common coefficient)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        series = _by_entity(data, self.variable, "the Breitung test")
        lengths = {len(x) for _, x in series}
        if len(lengths) != 1:
            raise SpecificationError("the Breitung test needs a balanced panel (every entity with the same periods); use balance()")
        p = int(self.lags)
        if lengths.pop() - p < 5 + (self.trend == "ct"):
            raise SpecificationError(f"too few periods for {p} lags")
        E, X = _breitung_terms(np.array([x for _, x in series]), p, self.trend)
        N, n = E.shape
        if self.robust:  # Breitung and Das (2005): the pooled coefficient over a sandwich of the residual covariances
            if n <= N:
                raise SpecificationError("the robust statistic needs more periods than entities")
            a = float(np.sum(E * X) / np.sum(X * X))
            U = E - a * X
            omega = U @ U.T / n
            stat = float(np.sum(E * X) / np.sqrt(np.einsum("it,ij,jt->", X, omega, X)))
        else:
            a = float(np.sum(E * X) / np.sum(X * X))
            s2 = float(np.sum((E - a * X) ** 2) / (N * n - 1))
            stat = a / np.sqrt(s2 / float(np.sum(X * X)))
        return TestResult(
            f"Breitung{' (robust)' if self.robust else ''} lambda ({self.variable})", self.null, stat, None,
            float(sps.norm.cdf(stat)), "normal",
            details={"pooled coefficient": a, "entities": N, "periods used": n, "lags": p},
        )


def _breitung_terms(Y: np.ndarray, p: int, trend: str) -> tuple[np.ndarray, np.ndarray]:
    """Breitung's transformed differences and levels, (N, n) each, from levels Y (N, T).

    Each series is prewhitened by its own regression of dy_t on dy_t-1..t-p (and a constant, for the drift, when there
    is a trend; Breitung and Das 2005), z_t = y_t - b_1 y_t-1 - ... - b_p y_t-p, and divided by that regression's residual standard error.
    With a constant, the levels are taken from the first: x_t = z_t-1 - z_0, and the differences are left alone; with a
    trend, the differences are forward orthogonalised (s_t (dz_t - the mean of dz_t+1..T), s_t^2 = (T - t) / (T - t + 1))
    and the levels detrended with the first and last, x_t = z_t-1 - z_0 - (t - 1) / T (z_T - z_0)."""
    N = Y.shape[0]
    Z, scale = [], []
    for y in Y:
        dy = np.diff(y)
        if p:
            L = np.column_stack([dy[p - j:len(dy) - j] for j in range(1, p + 1)] + ([np.ones(len(dy) - p)] if trend == "ct" else []))
            b = np.linalg.lstsq(L, dy[p:], rcond=None)[0]
            u = dy[p:] - L @ b
            z = y[p:] - sum(b[j - 1] * y[p - j:len(y) - j] for j in range(1, p + 1))  # the drift stays: x_t removes it
        else:
            u, z = (dy - dy.mean() if trend == "ct" else dy), y
        Z.append(z)
        scale.append(np.sqrt(u @ u / (len(u) - p - (trend == "ct"))))  # the drift is a parameter too
    Z = np.array(Z) / np.array(scale)[:, None]  # (N, T') levels z_0..z_T
    dz = np.diff(Z, axis=1)  # dz_1..dz_T
    T = dz.shape[1]
    if trend == "c":
        return dz, Z[:, :-1] - Z[:, :1]
    t = np.arange(1, T)  # the last difference has no future mean
    future = (np.cumsum(dz[:, ::-1], axis=1)[:, ::-1][:, 1:]) / (T - t)  # mean of dz_t+1..T, t = 1..T-1
    E = np.sqrt((T - t) / (T - t + 1)) * (dz[:, :-1] - future)
    X = Z[:, :T - 1] - Z[:, :1] - (t - 1) / T * (Z[:, -1:] - Z[:, :1])
    return E, X


def _cadf_t(Y: np.ndarray, p: int, trend: str) -> np.ndarray:
    """Pesaran's CADF t-statistics. Y holds levels, (replications, N, T); returns (replications, N).

    Each entity's ADF regression is augmented with the cross-section mean's lagged level and its
    current and p lagged differences: dy_it on y_i,t-1, ybar_t-1, dybar_t..t-p, dy_i,t-1..t-p and
    the deterministic terms."""
    R, N, T = Y.shape
    dy = np.diff(Y, axis=2)
    ybar = Y.mean(axis=1)
    dbar = np.diff(ybar, axis=1)
    m = T - 1
    n = m - p
    cols = [Y[:, :, p:-1], np.broadcast_to(ybar[:, None, p:-1], (R, N, n))]
    cols += [np.broadcast_to(dbar[:, None, p - j:m - j], (R, N, n)) for j in range(p + 1)]
    cols += [dy[:, :, p - j:m - j] for j in range(1, p + 1)]
    if trend != "n":
        cols.append(np.ones((R, N, n)))
    if trend == "ct":
        cols.append(np.broadcast_to(np.arange(1, n + 1.0), (R, N, n)))
    X = np.stack(cols, axis=3)
    D = dy[:, :, p:]
    inv = np.linalg.inv(np.einsum("rnti,rntj->rnij", X, X))
    b = np.einsum("rnij,rnj->rni", inv, np.einsum("rnti,rnt->rni", X, D))
    u = D - np.einsum("rnti,rni->rnt", X, b)
    s2 = np.einsum("rnt,rnt->rn", u, u) / (n - X.shape[3])
    return b[..., 0] / np.sqrt(s2 * inv[..., 0, 0])


def _cadf_t_unbalanced(Y: np.ndarray, p: int, trend: str) -> np.ndarray:
    """Pesaran's CADF t-statistics in an unbalanced panel, as plm's cipstest computes them. Y holds levels,
    (replications, N, T), missing where an entity is not observed. An entity's row at t is complete when dy_t,
    y_t-1 and dy_t-1..t-p are all observed; the cross-section means at t (of dy_t, y_t-1 and the lagged differences)
    average the entities whose row at t is complete, so in a balanced panel they are the usual means."""
    R, N, T = Y.shape
    dy = np.diff(Y, axis=2)
    m = T - 1
    n = m - p
    terms = [dy[:, :, p:], Y[:, :, p:-1]] + [dy[:, :, p - j:m - j] for j in range(1, p + 1)]  # each (R, N, n)
    complete = ~np.isnan(np.stack([t[0] for t in terms])).any(axis=0)  # (N, n): the same in every replication
    count = complete.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = [np.where(complete, t, 0.0).sum(axis=1) / count for t in terms]  # each (R, n)
    out = np.empty((R, N))
    for i in range(N):
        rows = complete[i]
        k = 2 + (p + 1) + p + {"n": 0, "c": 1, "ct": 2}[trend]
        if rows.sum() - k < 1:
            raise SpecificationError(f"entity {i + 1} has too few periods for {p} lags")
        cols = [terms[1][:, i], means[1], means[0]] + means[2:] + [t[:, i] for t in terms[2:]]
        if trend != "n":
            cols.append(np.ones((R, n)))
        if trend == "ct":
            cols.append(np.broadcast_to(np.arange(1, n + 1.0), (R, n)))
        X = np.stack(cols, axis=2)[:, rows]
        d = terms[0][:, i][:, rows]
        inv = np.linalg.inv(np.einsum("rti,rtj->rij", X, X))
        b = np.einsum("rij,rj->ri", inv, np.einsum("rti,rt->ri", X, d))
        u = d - np.einsum("rti,ri->rt", X, b)
        s2 = np.einsum("rt,rt->r", u, u) / (d.shape[1] - X.shape[2])
        out[:, i] = b[:, 0] / np.sqrt(s2 * inv[:, 0, 0])
    return out


# Pesaran (2007), equation (34): CADF statistics truncated to [-K1, K2] for CIPS*.
_CIPS_BOUNDS = {"n": (-6.12, 4.16), "c": (-6.19, 2.61), "ct": (-6.42, 1.70)}


@lru_cache(maxsize=64)
def _cips_null(N: int, T: int, p: int, trend: str, truncated: bool, replications: int, seed: int,
               missing: bytes | None = None) -> np.ndarray:
    """CIPS under the null (independent Gaussian random walks), sorted, by simulation. ``missing`` is an unbalanced
    panel's (N, T) mask of unobserved periods, as bytes: the walks are cut to the same periods."""
    rng = make_rng(seed)
    mask = None if missing is None else np.frombuffer(missing, dtype=bool).reshape(N, T)
    chunk = max(20, 3_000_000 // (N * T * (p + 5)))
    out = []
    for k in range(0, replications, chunk):
        Y = np.cumsum(rng.normal(size=(min(chunk, replications - k), N, T)), axis=2)
        if mask is None:
            t = _cadf_t(Y, p, trend)
        else:
            Y[:, mask] = np.nan
            t = _cadf_t_unbalanced(Y, p, trend)
        out.append((np.clip(t, *_CIPS_BOUNDS[trend]) if truncated else t).mean(axis=1))
    return np.sort(np.concatenate(out))


@register
@dataclass(frozen=True)
class CIPS(Test):
    """Pesaran's CIPS test: cross-sectionally augmented ADF regressions, for panels with a common factor.

    The null is that every series has a unit root; the alternative is that some are stationary. Each entity's ADF
    regression adds the cross-section means' lagged level and differences, which absorb a single unobserved common
    factor, so the test stays valid where IPS, Levin-Lin-Chu and Fisher-type tests reject too often; several
    factors are not fully absorbed. The statistic is the average CADF t-ratio and rejects when large and negative;
    critical values and the p-value are simulated for the panel's N, T and lags. The lag is fixed and the same for
    every entity: too few leave serial correlation that distorts the size, so check that the result survives a few
    values. The entities' own CADF statistics are in the details.
    """

    maturity: ClassVar[str] = "stable"
    variable: str
    lags: Annotated[int, advanced(help="lagged differences in each CADF regression, the same for every entity; too few distort the size")] = 0
    trend: Annotated[Literal["c", "ct", "n"], Widget("", help="each entity's own terms: c a constant, ct a constant and trend (trending series), n none")] = "c"
    truncated: Annotated[bool, advanced(help="truncate each CADF statistic (CIPS*), which matters only in very short panels")] = False
    replications: Annotated[int, advanced(help="simulations of the null distribution for this panel's N, T and lags")] = 10_000
    seed: Annotated[int, advanced()] = 0

    name = "cips"
    topics = ("unit-roots", "panel-data")
    references = (
        ("Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. Journal of Applied Econometrics 22(2), 265-312. doi:10.1002/jae.951"),
    )
    null = "every series in the panel has a unit root (allowing a common factor across entities)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        series = _by_entity(data, self.variable, "the CIPS test")
        frame = data[self.variable].unstack(data.entity_axes[0].name)
        frame = frame.loc[:, frame.notna().any(axis=0)]  # the entities with data, as _by_entity keeps them
        frame = frame.loc[frame.notna().any(axis=1)]
        T, N, p = len(frame), len(series), int(self.lags)
        k = 2 + (p + 1) + p + {"n": 0, "c": 1, "ct": 2}[self.trend]
        Y = frame.to_numpy(float).T[None]  # (1, N, T), entities in the panel's order
        missing = np.isnan(Y[0])
        if missing.any():
            cadf = _cadf_t_unbalanced(Y, p, self.trend)[0]
        else:
            if T - 1 - p - k < 3:
                raise SpecificationError(f"{T} periods are too few for {p} lags")
            cadf = _cadf_t(Y, p, self.trend)[0]
        used = np.clip(cadf, *_CIPS_BOUNDS[self.trend]) if self.truncated else cadf
        stat = float(used.mean())
        null = _cips_null(N, T, p, self.trend, bool(self.truncated), int(self.replications), int(self.seed),
                          missing.tobytes() if missing.any() else None)
        p_value = float((np.searchsorted(null, stat, side="right") + 1) / (len(null) + 1))
        crit = {f"{q}%": float(np.quantile(null, q / 100)) for q in (1, 5, 10)}
        return TestResult(
            f"Pesaran CIPS{'*' if self.truncated else ''} ({self.variable})", self.null, stat, None, p_value, "simulated",
            critical_values=crit,
            details={"entities": N, "periods": T if not missing.any() else f"{int((~missing).sum(axis=1).min())} to {T}", "lags": p,
                     "entities' CADF statistics": ", ".join(f"{key} {t:.2f}" for (key, _), t in zip(series, cadf)),
                     "simulation replications": len(null)},
        )
