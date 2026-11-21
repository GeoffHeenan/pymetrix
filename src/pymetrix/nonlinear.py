"""Threshold autoregressions and tests of linearity (Elliott and Timmermann 2016, chapter 8).

    tar = ThresholdAR("dgdp", lags=2).fit(data)        # SETAR: two AR(2) regimes, split by dgdp[-d] at a threshold
    tar.summary(); tar.linearity_test()                # Hansen's sup-F (bootstrap p-value)
    terasvirta(data, "dgdp", lags=2, delay=1)          # LM test of linearity against smooth transition
    tar.forecast(data, ("2025Q1", "2025Q4"))           # by simulation: multi-step forecasts of a nonlinear model
    backtest(tar, data, ("2015Q1", "2022Q4"))          # densities scored from the simulated draws

The two-regime self-exciting threshold autoregression is
y_t = (c1 + a1' y_lags) 1{y_(t-d) <= gamma} + (c2 + a2' y_lags) 1{y_(t-d) > gamma} + e_t, with one error variance.
It is estimated by conditional least squares: for each delay d (1 to the lag order, or the one given) and each
candidate threshold (the values of y_(t-d) between the ``trim`` and 1 - ``trim`` quantiles, so each regime has at
least that share of the observations), the regimes are fitted by least squares, and the pair with the smallest sum
of squared residuals is kept (Tong 1990; Hansen 1997).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .forms import Widget
from .output import Table
from .plugins import register
from .results import deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedThresholdAR", "ThresholdAR", "terasvirta"]


def _lagged(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """y_t and [1, y_(t-1), ..., y_(t-p)] for t = p, ..., n - 1."""
    X = np.column_stack([np.ones(len(y) - p)] + [y[p - j:len(y) - j] for j in range(1, p + 1)])
    return y[p:], X


def _ssr(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(e @ e), b


def _search(y: np.ndarray, X: np.ndarray, q: dict[int, np.ndarray], trim: float) -> tuple[float, int, float]:
    """The delay and threshold with the smallest sum of squared residuals: (ssr, delay, threshold)."""
    best = (np.inf, 0, np.nan)
    n = len(y)
    least = max(int(np.ceil(trim * n)), X.shape[1] + 1)
    for d, qd in q.items():
        values = np.sort(qd)
        candidates = np.unique(values[least - 1:n - least])
        for gamma in candidates:
            low = qd <= gamma
            if low.sum() < least or (~low).sum() < least:
                continue
            ssr = _ssr(y[low], X[low])[0] + _ssr(y[~low], X[~low])[0]
            if ssr < best[0]:
                best = (ssr, d, float(gamma))
    if not np.isfinite(best[0]):
        raise SpecificationError("too few observations for two regimes of this size: lower the lags or the trimming")
    return best


@dataclass(frozen=True)
class ThresholdAR:
    """A two-regime self-exciting threshold autoregression (SETAR), by conditional least squares over the threshold
    and the delay.

    The model lets the dynamics differ between two regimes, e.g. recessions and expansions, split by whether an
    earlier value of the series is above an estimated threshold. Before reading the regimes, test linearity with
    ``linearity_test()``: under the null of one regime the threshold is not identified, so ordinary F
    distributions do not apply and the p-value is bootstrapped. The coefficients' standard errors treat the
    threshold and delay as known. Trim sets the smallest regime: lower values let the search find a rare regime
    but estimate it from few observations. Multi-step forecasts are simulated, since a nonlinear model's forecast
    is not its one-step rule iterated.
    """

    label: ClassVar[str] = "Threshold autoregression (SETAR)"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    series: Annotated[str, Widget("text", help="the series, stationary (e.g. GDP growth, not its level); its own lag switches the regime")]
    lags: Annotated[int, Widget("number", min=1, max=12, help="the autoregressive order in each regime")] = 1
    delay: Annotated[int, Widget("number", min=0, max=12, help="the lag of the series that switches regimes; 0 = choose (1 to lags)")] = 0
    trim: Annotated[float, Widget("number", min=0.05, max=0.4, step=0.05, advanced=True, help="the least share of observations in each regime")] = 0.15

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise SpecificationError("lags must be 1 or more")
        if self.delay < 0 or self.delay > max(self.lags, 12):
            raise SpecificationError("the delay must be 0 (choose) or a lag up to 12")
        if not 0 < self.trim < 0.5:
            raise SpecificationError("trim must be between 0 and 0.5")

    def _delays(self) -> list[int]:
        return [self.delay] if self.delay else list(range(1, self.lags + 1))

    def _arrays(self, data: Dataset, span: Any) -> tuple[pd.Index, np.ndarray, np.ndarray, dict[int, np.ndarray], int]:
        """The estimation periods, y, [1, lags] and each delay's switching variable. As for equations, the sample is the
        periods of y; their lags may come from earlier data."""
        if self.series not in data.names:
            raise SpecificationError(f"no series {self.series!r} in the dataset")
        if data.kind != "time_series":
            raise SpecificationError("threshold autoregressions need a single time series")
        s = data[self.series]
        m = max(self.lags, max(self._delays()))
        y_all = s.to_numpy(float)
        rows = np.arange(m, len(y_all))
        y = y_all[rows]
        X = np.column_stack([np.ones(len(rows))] + [y_all[rows - j] for j in range(1, self.lags + 1)])
        q = {d: y_all[rows - d] for d in self._delays()}
        ok = np.asarray(data.sample_mask(span))[rows] & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        for qd in q.values():
            ok &= np.isfinite(qd)
        if not ok.any():
            raise SpecificationError(f"{self.series} has no observations (with their lags) in the sample")
        return s.index[rows[ok]], y[ok], X[ok], {d: qd[ok] for d, qd in q.items()}, m

    @api("pymetrix:threshold_ar.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedThresholdAR:
        """Estimate on the periods of ``span`` (first, last): least squares in each regime at every candidate threshold
        (and delay, when it is 0), keeping the one with the smallest sum of squared residuals."""
        from ._capture import capture, provenance_for

        index, y, X, q, m = self._arrays(data, span)
        s = data[self.series]
        if len(y) < 2 * (X.shape[1] + 5):
            raise SpecificationError(f"{len(y)} observations: too few for two AR({self.lags}) regimes")
        with capture() as caught:
            ssr, d, gamma = _search(y, X, q, self.trim)
        low = q[d] <= gamma
        ssr_low, b_low = _ssr(y[low], X[low])
        ssr_high, b_high = _ssr(y[~low], X[~low])
        k = 2 * X.shape[1]
        sigma2 = ssr / (len(y) - k)
        cov = []
        for mask in (low, ~low):
            Xm = X[mask]
            cov.append(sigma2 * np.linalg.inv(Xm.T @ Xm))
        ssr0, _ = _ssr(y, X)
        prov = provenance_for(self, data, [self.series], index, s.loc[index].to_frame(), caught,
                              {"type": "threshold autoregression", "estimation": "conditional least squares", "delay": d,
                               "threshold": gamma}, span=span, resolved={"delay": d, "threshold": gamma})
        return FittedThresholdAR(self, s, d, gamma, b_low, b_high, cov[0], cov[1], float(sigma2), float(ssr), float(ssr0),
                                 int(low.sum()), int((~low).sum()), prov, span)


@dataclass(frozen=True, eq=False)
class FittedThresholdAR:
    """An estimated two-regime threshold autoregression."""

    spec: ThresholdAR
    data: pd.Series = field(repr=False)  # the whole series
    delay: int = 1
    threshold: float = 0.0
    low: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # coefficients when y_(t-d) <= threshold: constant, lags
    high: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    cov_low: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    cov_high: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    sigma2: float = 0.0
    ssr: float = 0.0
    ssr_linear: float = 0.0  # of the AR on the same observations
    n_low: int = 0
    n_high: int = 0
    provenance: Any = None
    span: Any = None  # the sample asked for (the series is kept whole: lags may come from before it)

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def variables(self) -> list[str]:
        return [self.spec.series]

    def summary(self) -> Table:
        """The coefficients and standard errors in each regime, with the threshold, the delay and the fit."""
        names = ["constant"] + [f"{self.dependent}[-{j}]" for j in range(1, self.spec.lags + 1)]
        rows = []
        for j, name in enumerate(names):
            rows.append([float(self.low[j]), float(np.sqrt(self.cov_low[j, j])), float(self.high[j]), float(np.sqrt(self.cov_high[j, j]))])
        n = self.n_low + self.n_high
        notes = (f"Regimes split by {self.dependent}[-{self.delay}] at {self.threshold:.6g}: {self.n_low} observations at or below, "
                 f"{self.n_high} above ({n} in all).",
                 f"Error sd {np.sqrt(self.sigma2):.6g}; sum of squared residuals {self.ssr:.6g} against {self.ssr_linear:.6g} for the linear AR.",
                 "Standard errors treat the threshold and delay as known (they converge faster than the coefficients).")
        return Table(["low regime", "std err", "high regime", "std err "], rows, index=names, index_name="coefficient",
                     title=f"Threshold autoregression for {self.dependent}", notes=notes)

    def linearity_test(self, reps: int = 500, seed: int = 0) -> Any:
        """Hansen's (1996, 1997) test of the linear AR against the threshold model: sup-F over the thresholds (and
        delays, when chosen), F = n (SSR_linear - SSR_threshold) / SSR_threshold, with a fixed-regressor bootstrap
        p-value (the threshold is not identified under the null, so the usual F distribution does not apply)."""
        from .stats.base import TestResult

        _, y, X, q, _ = self.spec._arrays(_series_data(self.data), self.span)
        n = len(y)
        stat = n * (self.ssr_linear - self.ssr) / self.ssr
        rng = make_rng(seed)
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        exceed = 0
        for _ in range(int(reps)):
            ystar = e * rng.standard_normal(n)  # Hansen's fixed-regressor bootstrap
            s0, _ = _ssr(ystar, X)
            s1, _, _ = _search(ystar, X, q, self.spec.trim)
            exceed += n * (s0 - s1) / s1 >= stat
        p = (exceed + 1) / (reps + 1)
        return TestResult("Hansen threshold test (sup-F)", "the series is a linear AR (no threshold)", float(stat), None, float(p), "bootstrap",
                          details={"threshold": self.threshold, "delay": self.delay, "observations": n, "replications": int(reps)})

    def predictive_draws(self, data: Dataset, span: tuple[Any, Any], draws: int | None = None, seed: int = 0) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Simulated forecast paths (draws, periods, 1) from the end of ``data`` before ``span``: each draw follows the
        regimes with its own normal shocks, so multi-step forecasts reflect the nonlinearity."""
        freq = self.data.index.freq
        from .data.sample import as_span
        span = as_span(span)  # '1990Q1:2019Q4' or a pair
        start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
        periods = pd.period_range(start, end, freq=freq, name=self.data.index.name or "time")
        p, d = self.spec.lags, self.delay
        m = max(p, d)
        series = data[self.dependent]
        history = series.reindex(pd.period_range(start - m, start - 1, freq=freq)).to_numpy(float)
        if not np.all(np.isfinite(history)):
            raise SpecificationError(f"data for the {m} periods before {start} are needed to start the forecast")
        n = int(draws or 2000)
        rng = make_rng(seed)
        paths = np.empty((n, len(periods)))
        hist = np.tile(history, (n, 1))
        sd = np.sqrt(self.sigma2)
        for i in range(len(periods)):
            x = np.column_stack([np.ones(n)] + [hist[:, -j] for j in range(1, p + 1)])
            regime_low = hist[:, -d] <= self.threshold
            mean = np.where(regime_low, x @ self.low, x @ self.high)
            new = mean + sd * rng.standard_normal(n)
            paths[:, i] = new
            hist = np.column_stack([hist, new])
        return periods, paths[:, :, None]

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True, draws: int = 2000, seed: int = 0) -> Any:
        """Forecasts by simulation (``draws`` paths): mean, sd and the 1 - alpha interval from the simulated quantiles.
        The one-step forecast is the regime's prediction, exactly."""
        from .data import Dataset as _Dataset

        if not dynamic:
            raise SpecificationError("threshold AR forecasts are dynamic")
        periods, paths = self.predictive_draws(data, span, draws, seed)
        paths = paths[:, :, 0]
        lo, hi = np.quantile(paths, [alpha / 2, 1 - alpha / 2], axis=0)
        frame = pd.DataFrame({"forecast": paths.mean(axis=0), "se": paths.std(axis=0, ddof=1), "lower": lo, "upper": hi}, index=periods)
        frame["actual"] = data[self.dependent].reindex(periods).to_numpy(float)
        return _Dataset(frame, data.axes)

    def plot(self) -> Any:
        """The series, with the threshold."""
        from .figures import Figure, Line, Panel, Rule

        return Figure((Panel((Line(self.data, self.dependent, color="#222222"), Rule(self.threshold, "--", "#d55e00")),
                             title=f"{self.dependent}: regimes split at {self.threshold:.4g} (by {self.dependent}[-{self.delay}])"),))

    def __repr__(self) -> str:
        return f"<FittedThresholdAR {self.dependent}: AR({self.spec.lags}) regimes split by lag {self.delay} at {self.threshold:.4g}>"


def _series_data(s: pd.Series) -> Any:
    from .data import Dataset

    return Dataset.from_frame(s.to_frame())


def terasvirta(data: Dataset, series: str, lags: int = 1, delay: int = 1, span: Any = None) -> Any:
    """Teräsvirta's (1994) LM test of linearity against a smooth-transition AR: regress the AR(lags) residuals on the
    lags, and the lags times y_(t-d), y_(t-d)^2 and y_(t-d)^3 (a third-order expansion of the transition); F test of
    the added terms."""
    from .stats.base import TestResult

    spec = ThresholdAR(series, lags, delay)
    _, y, X, q, _ = spec._arrays(data, span)
    qd = q[delay]
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    lagged = X[:, 1:]
    extra = np.column_stack([lagged * qd[:, None], lagged * (qd**2)[:, None], lagged * (qd**3)[:, None]])
    Z = np.column_stack([X, extra])
    ssr0 = float(e @ e)
    ssr1, _ = _ssr(e, Z)
    m, n = extra.shape[1], len(y)
    df2 = n - Z.shape[1]
    stat = ((ssr0 - ssr1) / m) / (ssr1 / df2)
    return TestResult("Teräsvirta (LM3)", "the series is a linear AR (no smooth transition)", float(stat), (m, df2), float(sps.f.sf(stat, m, df2)), "F",
                      details={"lags": lags, "delay": delay, "observations": n})


register("models", ThresholdAR)
