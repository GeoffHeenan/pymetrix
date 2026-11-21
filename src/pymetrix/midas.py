"""MIDAS: regressions of a low-frequency series on high-frequency indicators (Ghysels, Santa-Clara and Valkanov).

    m = MIDAS("gdp ~ ip", lags=12, weights="exp_almon").fit(quarterly, monthly)
    m.weights["ip"]                      # the estimated lag weights, month by month
    m.nowcast(quarterly, monthly)        # GDP for every quarter the indicator covers, the current one included
    MIDAS("gdp ~ ip", lags=12, high_lag=1)   # a nowcast made one month before the quarter's data are complete

Each indicator enters through ``lags`` of its own high-frequency values, ending at the last
high-frequency period of the low-frequency period (less ``high_lag``), combined by a lag
polynomial with few parameters: exponential Almon (two parameters) or normalised Beta (two),
so that twelve monthly lags cost three coefficients, not twelve. ``unrestricted`` (U-MIDAS)
estimates every lag by least squares instead, which works when the frequency gap is small.
``ar`` adds lags of the dependent variable. The parametric forms are estimated by nonlinear
least squares.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from .errors import NotFoundError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["MIDAS", "FittedMIDAS"]

Weights = Literal["exp_almon", "beta", "unrestricted"]


def lag_weights(kind: str, theta: Sequence[float], K: int) -> np.ndarray:
    """Normalised lag weights (summing to one), lag 0 first."""
    j = np.arange(K, dtype=float)
    if kind == "exp_almon":
        z = theta[0] * j + theta[1] * j**2
        z = z - z.max()  # numerically safe
        w = np.exp(z)
    elif kind == "beta":
        eps = 1e-8
        x = np.clip(j / max(K - 1, 1), eps, 1 - eps)
        a, b = np.exp(theta[0]), np.exp(theta[1])  # positive shapes, estimated on the log scale
        w = x ** (a - 1) * (1 - x) ** (b - 1)
    else:
        raise SpecificationError(f"unknown weights {kind!r}")
    total = w.sum()
    return w / total if total > 0 and np.isfinite(total) else np.full(K, 1.0 / K)


@dataclass(frozen=True)
class MIDAS:
    """``formula``: the low-frequency series on one or more high-frequency indicators (``gdp ~ ip + sales``)."""

    formula: str
    lags: int = 12
    weights: Weights = "exp_almon"
    ar: int = 0
    high_lag: int = 0  # high-frequency periods missing at the end of each low-frequency period (for nowcasts)

    def __post_init__(self) -> None:
        if self.lags < 2:
            raise SpecificationError("lags must be 2 or more")
        if self.ar < 0 or self.high_lag < 0:
            raise SpecificationError("ar and high_lag must be 0 or more")

    def _parts(self) -> tuple[str, list[str], bool]:
        from .estimation import Equation

        eq = Equation(self.formula)
        names = [t.label for t in eq.terms]
        if not names or any(not t.label.isidentifier() for t in eq.terms):
            raise SpecificationError("write MIDAS indicators as series names: gdp ~ ip + sales")
        return eq.require_dependent(), names, eq.intercept

    def design(self, low: Dataset, high: Dataset, periods: pd.PeriodIndex | None = None) -> tuple[pd.PeriodIndex, dict[str, np.ndarray]]:
        """For each low-frequency period: each indicator's ``lags`` high-frequency values, latest first."""
        target, names, _ = self._parts()
        if low.kind != "time_series" or high.kind != "time_series":
            raise SpecificationError("MIDAS needs two time series datasets")
        missing = [n for n in names if n not in high.names]
        if missing:
            raise NotFoundError(f"no indicator {missing} in the high-frequency data")
        hi_index = pd.PeriodIndex(high.frame.index)
        lo = pd.PeriodIndex(low.frame.index) if periods is None else periods
        out = {}
        for name in names:
            x = high[name].astype(float)
            rows = []
            for p in lo:
                anchor = p.asfreq(hi_index.freq, how="end") - self.high_lag
                rows.append([x.get(anchor - j, np.nan) for j in range(self.lags)])
            out[name] = np.asarray(rows, dtype=float)
        return lo, out

    @api("pymetrix:midas.fit")
    def fit(self, low: Dataset, high: Dataset, span: tuple[Any, Any] | None = None) -> FittedMIDAS:
        from dataclasses import replace

        from ._capture import capture, provenance_for

        with capture() as caught:
            fitted = self._fit(low, high, span)
        target, names, _ = self._parts()
        used = pd.concat([fitted.resid + fitted.fitted, *[high[n] for n in names]], axis=1)
        converged = fitted.provenance.converged if fitted.provenance is not None else None
        prov = provenance_for(self, low, [target], fitted.resid.index, used, caught, {"type": "MIDAS", "weights": self.weights}, converged,
                              span=span)
        vintages = {**prov.data_vintages, **{n: high.variables[n].vintage for n in names if n in high.variables}}  # indicators too
        return replace(fitted, provenance=replace(prov, data_vintages=vintages))

    def _fit(self, low: Dataset, high: Dataset, span: tuple[Any, Any] | None = None) -> FittedMIDAS:
        target, names, intercept = self._parts()
        lo, X = self.design(low, high)
        y = low[target].astype(float).reindex(lo).to_numpy()
        ylags = [pd.Series(y, index=lo).shift(p).to_numpy() for p in range(1, self.ar + 1)]
        ok = np.isfinite(y) & np.all([np.all(np.isfinite(X[n]), axis=1) for n in names], axis=0)
        for yl in ylags:
            ok &= np.isfinite(yl)
        ok &= np.asarray(low.sample_mask(span))  # the span, or else the low-frequency data's active sample
        if ok.sum() < 10:
            raise SpecificationError("fewer than ten complete low-frequency periods to estimate on")
        yv = y[ok]
        Xs = {n: X[n][ok] for n in names}
        L = np.column_stack([yl[ok] for yl in ylags]) if ylags else np.zeros((ok.sum(), 0))
        const = np.ones((ok.sum(), 1)) if intercept else np.zeros((ok.sum(), 0))
        if self.weights == "unrestricted":
            Z = np.column_stack([const] + [Xs[n] for n in names] + [L])
            beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
            u = yv - Z @ beta
            k = Z.shape[1]
            s2 = u @ u / (len(yv) - k)
            cov = s2 * np.linalg.inv(Z.T @ Z)
            pnames = (["const"] if intercept else []) + [f"{n}[{-j}]" for n in names for j in range(self.lags)] + [f"{target}[-{p}]" for p in range(1, self.ar + 1)]
            w = {n: pd.Series(beta[(1 if intercept else 0) + i * self.lags:(1 if intercept else 0) + (i + 1) * self.lags], index=pd.RangeIndex(self.lags, name="lag")) for i, n in enumerate(names)}
            return self._result(lo[ok], yv, u, beta, cov, pnames, w, k)
        # parametric: [const] + per indicator (slope, theta1, theta2) + ar
        def unpack(b: np.ndarray) -> tuple[float, list[tuple[float, np.ndarray]], np.ndarray]:
            c = b[0] if intercept else 0.0
            pos = 1 if intercept else 0
            parts = []
            for _ in names:
                parts.append((b[pos], b[pos + 1 : pos + 3]))
                pos += 3
            return c, parts, b[pos:]

        def fitted(b: np.ndarray) -> np.ndarray:
            c, parts, g = unpack(b)
            f = np.full(len(yv), c)
            for n, (slope, theta) in zip(names, parts):
                f = f + slope * (Xs[n] @ lag_weights(self.weights, theta, self.lags))
            return f + (L @ g if len(g) else 0.0)

        # start: flat weights (theta 0), slopes and constant by least squares on the flat aggregates
        flat = np.column_stack([const] + [Xs[n].mean(axis=1, keepdims=True) for n in names] + [L])
        b0, *_ = np.linalg.lstsq(flat, yv, rcond=None)
        start = list(b0[: 1 if intercept else 0])
        for i, _ in enumerate(names):
            start += [b0[(1 if intercept else 0) + i], 0.0, 0.0]
        start += list(b0[(1 if intercept else 0) + len(names):])
        best = None
        for theta_start in ((0.0, 0.0), (0.1, -0.01), (-0.1, -0.01)) if self.weights == "exp_almon" else ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)):
            s0 = np.array(start, dtype=float)
            pos = 1 if intercept else 0
            for _ in names:
                s0[pos + 1 : pos + 3] = theta_start
                pos += 3
            res = optimize.least_squares(lambda b: yv - fitted(b), s0, method="lm" if len(yv) > len(s0) else "trf", xtol=1e-12, ftol=1e-12, max_nfev=20000)
            if best is None or res.cost < best.cost:
                best = res
        b = best.x
        u = yv - fitted(b)
        k = len(b)
        s2 = u @ u / (len(yv) - k)
        J = best.jac
        cov = s2 * np.linalg.pinv(J.T @ J)
        pnames = (["const"] if intercept else [])
        for n in names:
            pnames += [n, f"{n}: theta1", f"{n}: theta2"]
        pnames += [f"{target}[-{p}]" for p in range(1, self.ar + 1)]
        _, parts, _ = unpack(b)
        w = {n: pd.Series(lag_weights(self.weights, theta, self.lags), index=pd.RangeIndex(self.lags, name="lag")) for n, (_, theta) in zip(names, parts)}
        return self._result(lo[ok], yv, u, b, cov, pnames, w, k, converged=bool(best.success))

    def _result(self, index: pd.PeriodIndex, y: np.ndarray, u: np.ndarray, beta: np.ndarray, cov: np.ndarray, names: list[str],
                weights: dict[str, pd.Series], k: int, converged: bool | None = None) -> FittedMIDAS:
        from .estimation.params import point

        n = len(y)
        ssr = float(u @ u)
        tss = float(((y - y.mean()) ** 2).sum())
        loglike = -n / 2 * (np.log(2 * np.pi * ssr / n) + 1)
        stats = {"nobs": float(n), "k": float(k), "r2": 1 - ssr / tss, "se_regression": float(np.sqrt(ssr / (n - k))),
                 "loglike": float(loglike), "aic": float(-2 * loglike + 2 * k), "bic": float(-2 * loglike + k * np.log(n)), "df_resid": float(n - k)}
        from .metadata import Provenance

        carrier = Provenance({}, {}, {}, None, n, 0, converged=converged)  # replaced by fit(), which keeps converged
        return FittedMIDAS(self, point(names, beta, cov), weights, pd.Series(u, index=index, name="resid"),
                           pd.Series(y - u, index=index, name="fitted"), stats, carrier)


@dataclass(frozen=True, eq=False)
class FittedMIDAS:
    spec: MIDAS
    params: Any
    weights: dict[str, pd.Series]
    resid: pd.Series
    fitted: pd.Series
    stats: dict[str, float] = field(default_factory=dict)
    provenance: Any = None

    def nowcast(self, low: Dataset, high: Dataset, through: Any = None) -> pd.Series:
        """Predictions for every low-frequency period whose indicator lags are available (to ``through``, default the
        last period the indicators reach), including periods after the low-frequency data end."""
        target, names, intercept = self.spec._parts()
        hi = pd.PeriodIndex(high.frame.index)
        lo_freq = pd.PeriodIndex(low.frame.index).freq
        last = pd.Period(str(through), freq=lo_freq) if through is not None else (hi[-1] + self.spec.high_lag).asfreq(lo_freq)
        periods = pd.period_range(pd.PeriodIndex(low.frame.index)[0], last, freq=lo_freq)
        _, X = self.spec.design(low, high, periods)
        p = self.params.as_dict()
        out = np.full(len(periods), p.get("const", 0.0))
        for n in names:
            if self.spec.weights == "unrestricted":
                coefs = np.array([p[f"{n}[{-j}]"] for j in range(self.spec.lags)])
            else:
                coefs = p[n] * self.weights[n].to_numpy()
            out = out + X[n] @ coefs
        if self.spec.ar:
            y = low[target].astype(float).reindex(periods)
            history = y.copy()
            for i, period in enumerate(periods):  # recursive: use the data where there are data, else the nowcast
                if not np.isfinite(out[i]):
                    continue
                for lag in range(1, self.spec.ar + 1):
                    prev = history.iloc[i - lag] if i - lag >= 0 else np.nan
                    out[i] += p[f"{target}[-{lag}]"] * prev
                if not np.isfinite(history.iloc[i]):
                    history.iloc[i] = out[i]
        return pd.Series(out, index=periods, name=f"{target} (MIDAS)").dropna()

    def summary(self) -> Table:
        p = self.params
        z = p.values / p.se
        pv = 2 * sps.t.sf(np.abs(z), self.stats["df_resid"])
        rows = [[v, s, t, q] for v, s, t, q in zip(p.values, p.se, z, pv)]
        s = self.spec
        notes = [f"MIDAS, {s.weights} weights, {s.lags} high-frequency lags" + (f", data {s.high_lag} periods short" if s.high_lag else "")
                 + (f", {s.ar} autoregressive lags" if s.ar else ""),
                 f"Sample: {self.resid.index[0]} to {self.resid.index[-1]}, {len(self.resid)} observations",
                 f"R-squared: {self.stats['r2']:.4f}", f"S.E. of regression: {self.stats['se_regression']:.4f}",
                 f"Akaike info criterion: {self.stats['aic']:.3f}"]
        return Table(["coef", "std err", "t", "p>|t|"], rows, index=list(p.names), title=f"MIDAS: {s.formula}", notes=notes)

    def plot(self) -> Any:
        """The estimated lag weights of each indicator."""
        from .figures import Bars, Figure, Panel

        panels = [Panel((Bars(w, "weight"),), title=f"{n}: lag weights", xlabel="high-frequency lag") for n, w in self.weights.items()]
        return Figure(tuple(panels), columns=2 if len(panels) > 1 else 1)

    def __repr__(self) -> str:
        return f"<FittedMIDAS {self.spec.formula}: {self.spec.weights}, {self.spec.lags} lags>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
