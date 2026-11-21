"""Value at risk and expected shortfall, forecast one period ahead and backtested.

    risk = value_at_risk(data, "r", level=0.99, method="garch", window=500)
    risk.summary()        # the violations, Kupiec, Christoffersen, the dynamic quantile test, the shortfall test, the zone
    risk.plot()           # returns against -VaR and -ES, violations marked
    risk.frame            # return, var, es, violation for each period forecast

Returns are in the units of the series (log or simple returns, as given); VaR and ES are reported as positive
losses: at ``level`` 0.99, the period's return falls below ``-var`` with probability 1%, and ``-es`` is its
expected value when it does. Each period's forecast uses only the returns before it: the last ``window`` for
historical simulation and the normal, Student-t and Cornish-Fisher methods; for GARCH and filtered historical
simulation, a GARCH(1,1) estimated on everything before (re-estimated every ``refit`` periods, filtered forward
between).

The methods:

- ``historical``: the empirical quantile of the window (numpy's linear interpolation); ES, the mean beyond it.
- ``normal``: the window's mean and standard deviation.
- ``t``: a Student-t fitted to the window by maximum likelihood (re-fitted every ``refit`` periods).
- ``cornish_fisher``: the normal quantile adjusted for the window's skewness and excess kurtosis; ES by averaging the
  adjusted quantile over the tail.
- ``garch``: the GARCH forecast of the next period's volatility with normal errors.
- ``fhs``: filtered historical simulation, the GARCH volatility forecast times the empirical quantile of the
  standardised residuals.

Backtests: Kupiec's (1995) proportion of failures; Christoffersen's (1998) independence and conditional coverage;
Engle and Manganelli's (2004) dynamic quantile test (hits on four lags and VaR); McNeil and Frey's (2000) test that
losses beyond VaR average ES; and the Basel traffic-light zone of the violation count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

METHODS = ("historical", "normal", "t", "cornish_fisher", "garch", "fhs")


# -- one period's VaR and ES from a sample, or from a volatility forecast ---------------------------------------------

def historical(x: np.ndarray, p: float) -> tuple[float, float]:
    q = float(np.quantile(x, p))
    tail = x[x <= q]
    return -q, -float(tail.mean()) if len(tail) else -q


def normal(mu: float, sigma: float, p: float) -> tuple[float, float]:
    z = sps.norm.ppf(p)
    return -(mu + sigma * z), -(mu - sigma * sps.norm.pdf(z) / p)


def student_t(nu: float, loc: float, scale: float, p: float) -> tuple[float, float]:
    """VaR and ES of loc + scale * T(nu): the standard t's tail mean is -pdf(q) (nu + q^2) / ((nu - 1) p)."""
    if nu <= 1:
        raise SpecificationError("the fitted t has no mean (degrees of freedom of 1 or less): use another method")
    q = sps.t.ppf(p, nu)
    tail_mean = -sps.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * p)
    return -(loc + scale * q), -(loc + scale * tail_mean)


def _cf_quantile(z: np.ndarray, skew: float, kurt: float) -> np.ndarray:
    return z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36


def cornish_fisher(mu: float, sigma: float, skew: float, kurt: float, p: float) -> tuple[float, float]:
    """The Cornish-Fisher quantile (``kurt`` is excess kurtosis); ES as the adjusted quantile's mean over (0, p)."""
    q = _cf_quantile(np.array([sps.norm.ppf(p)]), skew, kurt)[0]
    grid = (np.arange(1000) + 0.5) / 1000 * p  # the tail, by the midpoint rule
    tail = _cf_quantile(sps.norm.ppf(grid), skew, kurt).mean()
    return -(mu + sigma * q), -(mu + sigma * tail)


# -- GARCH(1,1) by pymetrix, filtered forward --------------------------------------------------------------------------

@dataclass
class _Garch:
    mu: float
    omega: float
    alpha: float
    beta: float
    var_next: float  # the variance forecast for the period after the last one filtered
    z: np.ndarray  # standardised residuals of the estimation sample
    sigma: np.ndarray | None = None  # the conditional standard deviations of the estimation sample
    gamma: float = 0.0  # GJR: the extra weight on negative surprises
    nu: float | None = None  # t errors: the degrees of freedom of the standardised (unit-variance) t

    def step(self, r: float) -> None:
        """Filter one more period: the variance forecast for the period after ``r``."""
        e = r - self.mu
        self.z = np.append(self.z, e / np.sqrt(self.var_next))
        self.var_next = self.omega + (self.alpha + self.gamma * (e < 0)) * e**2 + self.beta * self.var_next

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    def var_es(self, p: float) -> tuple[float, float]:
        """Next period's VaR and ES under the model's own errors (normal, or the standardised t)."""
        sigma = float(np.sqrt(self.var_next))
        if self.nu is None:
            return normal(self.mu, sigma, p)
        return student_t(self.nu, self.mu, sigma * np.sqrt((self.nu - 2) / self.nu), p)


def _fit_garch(r: pd.Series, dist: str = "normal", asymmetric: bool = False) -> _Garch:
    from pymetrix.data import Dataset
    from pymetrix.estimation import ML, Equation
    from pymetrix import GARCH

    scale = 1.0 / max(float(r.std()), 1e-12)  # estimated on returns of standard deviation 1 (well conditioned), scaled back
    data = Dataset.from_frame((r * scale).rename("r").to_frame())
    if dist not in ("normal", "t"):
        raise SpecificationError("the GARCH errors are 'normal' or 't'")
    fit = Equation("r ~ 1", variance=GARCH(asym=int(asymmetric), dist=dist)).fit(data, method=ML())
    v = fit.params.as_dict()
    mu, omega, alpha, beta = v["const"] / scale, v["omega"] / scale**2, v["alpha[1]"], v["beta[1]"]
    gamma = float(v.get("gamma[1]", 0.0))
    vol = np.asarray(fit.extra["volatility"], float) / scale
    e = r.to_numpy(float) - mu
    var_next = omega + (alpha + gamma * (e[-1] < 0)) * e[-1] ** 2 + beta * vol[-1] ** 2
    return _Garch(mu, omega, alpha, beta, var_next, e / vol, vol, gamma, float(v["nu"]) if dist == "t" else None)


# -- the forecasts and their backtests ---------------------------------------------------------------------------------

def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec's likelihood ratio for the share of violations: chi-squared(1)."""
    n, x = len(hits), int(hits.sum())

    def ll(q: float) -> float:  # the binomial log likelihood of x violations in n, with 0 log 0 = 0
        return _xlogy(n - x, 1 - q) + _xlogy(x, q)

    lr = max(-2 * (ll(p) - ll(x / n)), 0.0)
    return float(lr), float(sps.chi2.sf(lr, 1))


def _xlogy(a: float, b: float) -> float:
    return float(a * np.log(b)) if a > 0 else 0.0


def christoffersen(hits: np.ndarray, p: float) -> dict[str, float]:
    """Christoffersen's independence test (violations not clustered) and conditional coverage (both together)."""
    h = hits.astype(int)
    prev, cur = h[:-1], h[1:]
    n00, n01 = int(np.sum((prev == 0) & (cur == 0))), int(np.sum((prev == 0) & (cur == 1)))
    n10, n11 = int(np.sum((prev == 1) & (cur == 0))), int(np.sum((prev == 1) & (cur == 1)))

    xlogy = _xlogy
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    unrestricted = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = float(-2 * (restricted - unrestricted))
    lr_pof, _ = kupiec(h[1:], p)  # on the same periods as the transitions
    lr_cc = lr_pof + lr_ind
    return {"LR_ind": lr_ind, "p_ind": float(sps.chi2.sf(lr_ind, 1)), "LR_cc": float(lr_cc), "p_cc": float(sps.chi2.sf(lr_cc, 2)),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def dynamic_quantile(hits: np.ndarray, var: np.ndarray, p: float, lags: int = 4) -> tuple[float, float, int]:
    """Engle and Manganelli's DQ test: hit - p regressed on a constant, its own lags and VaR, all zero under a
    correct model; DQ = b'X'X b / (p (1 - p)), chi-squared with as many degrees of freedom as regressors."""
    hit = hits.astype(float) - p
    rows = np.arange(lags, len(hit))
    X = np.column_stack([np.ones(len(rows))] + [hit[rows - j] for j in range(1, lags + 1)] + [var[rows]])
    y = hit[rows]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    dq = float(b @ X.T @ X @ b / (p * (1 - p)))
    k = X.shape[1]
    return dq, float(sps.chi2.sf(dq, k)), k


def shortfall_test(losses: np.ndarray, es: np.ndarray, hits: np.ndarray) -> tuple[float, float, int]:
    """McNeil and Frey's test: on violation days the loss beyond ES averages zero; a one-sided t-test against losses
    that exceed ES (shortfall underestimated)."""
    d = (losses - es)[hits.astype(bool)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n))) if d.std(ddof=1) > 0 else 0.0
    return t, float(sps.t.sf(t, n - 1)), n


def traffic_light(x: int, n: int, p: float) -> str:
    """The Basel zone for x violations in n periods: green while P(X <= x) < 95%, yellow below 99.99%, red beyond."""
    cdf = float(sps.binom.cdf(x, n, p))
    return "green" if cdf < 0.95 else "yellow" if cdf < 0.9999 else "red"


@dataclass(frozen=True, eq=False)
class RiskForecast:
    """Value at risk and expected shortfall forecast one period ahead, with the returns that followed."""

    series: str
    level: float
    method: str
    window: int
    frame: pd.DataFrame = field(repr=False)  # return, var, es, violation; by period

    stage = "finance"  # its badges and next steps belong to the Finance workspace (extension API 1.1)

    @property
    def p(self) -> float:
        return 1 - self.level

    @property
    def violations(self) -> int:
        return int(self.frame["violation"].sum())

    def backtests(self) -> Table:
        f = self.frame
        hits = f["violation"].to_numpy(bool)
        n, x = len(f), int(hits.sum())
        lr, p_pof = kupiec(hits, self.p)
        ch = christoffersen(hits, self.p)
        dq, p_dq, k = dynamic_quantile(hits, f["var"].to_numpy(float), self.p)
        t, p_es, m = shortfall_test(-f["return"].to_numpy(float), f["es"].to_numpy(float), hits)
        rows = [["violations", float(x), None, f"{x} of {n} ({x / n:.2%}; expected {self.p:.2%}, {self.p * n:.1f})"],
                ["Kupiec: proportion of failures", lr, p_pof, "chi-squared(1)"],
                ["Christoffersen: independence", ch["LR_ind"], ch["p_ind"], "chi-squared(1): violations not clustered"],
                ["Christoffersen: conditional coverage", ch["LR_cc"], ch["p_cc"], "chi-squared(2): both"],
                [f"Engle-Manganelli: dynamic quantile", dq, p_dq, f"chi-squared({k}): hits unpredictable from 4 lags and VaR"],
                ["McNeil-Frey: shortfall", t, p_es, f"one-sided t on {m} violations: losses beyond VaR average ES"],
                ["Basel zone", None, None, traffic_light(x, n, self.p)]]
        return Table(["statistic", "p-value", "note"], [r[1:] for r in rows], index=[r[0] for r in rows], index_name="backtest",
                     title=f"Backtests of {self.level:.1%} VaR and ES ({self.method}) for {self.series}",
                     notes=("Small p-values reject the forecasts. The Basel zone counts violations as for 250 days at 99%, "
                            "here on the periods forecast.",))

    def summary(self) -> Table:
        return self.backtests()

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel, Scatter

        f = self.frame
        hit = f[f["violation"]]
        layers = (Line(f["return"], self.series, color="#888888", width=0.8), Line(-f["var"], f"-VaR {self.level:.0%}", color="#c92a2a"),
                  Line(-f["es"], "-ES", color="#7b2cbf", style="--"))
        if len(hit):
            layers = layers + (Scatter(pd.Series(hit.index, index=hit.index), hit["return"], "violations"),)
        return Figure((Panel(layers, title=f"{self.series}: {self.level:.1%} VaR and ES ({self.method}), {self.violations} violations"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.frame

    def __repr__(self) -> str:
        return (f"<RiskForecast {self.level:.1%} VaR and ES of {self.series} by {self.method}: {len(self.frame)} periods, "
                f"{self.violations} violations>")


with plugin_context(PLUGIN):
    @api("finance:value_at_risk")
    def value_at_risk(data: Dataset, series: str, level: float = 0.99, method: Literal["historical", "normal", "t", "cornish_fisher",
                      "garch", "fhs"] = "historical", window: int = 250, refit: int = 20, start: Any = None,
                      dist: Literal["normal", "t"] = "normal", asymmetric: bool = False) -> RiskForecast:
        """One-period-ahead value at risk and expected shortfall of the returns in ``series``, for each period from
        ``start`` (default: once ``window`` returns are available), each from the returns before it only. See the
        module notes for the methods. For GARCH and filtered historical simulation, ``dist="t"`` gives the GARCH
        Student-t errors and ``asymmetric=True`` the GJR form (more response to falls than to rises)."""
        from pymetrix.extension import report_progress as report

        if dist not in ("normal", "t"):
            raise SpecificationError("dist is 'normal' or 't'")
        if (dist != "normal" or asymmetric) and method not in ("garch", "fhs"):
            raise SpecificationError("dist and asymmetric are for the GARCH methods (garch, fhs)")

        if method not in METHODS:
            raise SpecificationError(f"method is one of {', '.join(METHODS)}")
        if not 0.5 < level < 1:
            raise SpecificationError("level is the confidence, between 0.5 and 1 (0.99 for 99%)")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("value at risk needs one series of returns over time")
        if series not in data.names:
            raise SpecificationError(f"no series {series!r} in the dataset")
        r = data[series].dropna()
        if len(r) <= window + 10:
            raise SpecificationError(f"{len(r)} returns: more than the window ({window}) are needed, to have periods to forecast")
        first = window if start is None else max(window, int(r.index.get_indexer([pd.Period(str(start), freq=r.index.freq)])[0]))
        if first < 0:
            raise SpecificationError(f"{start} is not a period of {series}")
        p = 1 - level
        values = r.to_numpy(float)
        out_var, out_es = np.empty(len(r) - first), np.empty(len(r) - first)
        garch: _Garch | None = None
        t_fit: tuple[float, float, float] | None = None
        for k, i in enumerate(range(first, len(r))):
            if k % 50 == 0:
                report(k / (len(r) - first), f"value at risk: period {k + 1} of {len(r) - first}")
            win = values[i - window:i]
            if method == "historical":
                v, e = historical(win, p)
            elif method == "normal":
                v, e = normal(float(win.mean()), float(win.std(ddof=1)), p)
            elif method == "t":
                if t_fit is None or k % refit == 0:
                    t_fit = sps.t.fit(win)
                v, e = student_t(*t_fit, p)
            elif method == "cornish_fisher":
                v, e = cornish_fisher(float(win.mean()), float(win.std(ddof=1)), float(sps.skew(win, bias=False)),
                                      float(sps.kurtosis(win, bias=False)), p)
            else:
                if garch is None or k % refit == 0:
                    garch = _fit_garch(r.iloc[:i], dist, asymmetric)
                sigma = float(np.sqrt(garch.var_next))
                if method == "garch":
                    v, e = garch.var_es(p)
                else:
                    zq, ze = historical(garch.z, p)  # in standardised units, as losses
                    v, e = -garch.mu + sigma * zq, -garch.mu + sigma * ze
                garch.step(values[i])
            out_var[k], out_es[k] = v, e
        index = r.index[first:]
        ret = values[first:]
        frame = pd.DataFrame({"return": ret, "var": out_var, "es": out_es, "violation": ret < -out_var}, index=index)
        label = method if dist == "normal" and not asymmetric else f"{method} ({'GJR, ' if asymmetric else ''}{dist} errors)"
        return RiskForecast(series, float(level), label, int(window), frame)
