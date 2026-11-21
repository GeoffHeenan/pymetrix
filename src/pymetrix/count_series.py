"""Time series of counts and durations: INAR(1), Poisson autoregression with feedback, and autoregressive conditional
duration models, all by maximum likelihood.

    inar(data, "claims")                        # y_t = alpha o y_{t-1} + e_t: binomial thinning, Poisson innovations
    poisson_autoregression(data, "claims")      # y_t | past ~ Poisson(lambda_t), lambda_t = omega + a y_{t-1} + b lambda_{t-1}
    acd(data, "duration")                       # x_t = psi_t e_t, psi_t = omega + a x_{t-1} + b psi_{t-1}, e exponential
    acd(data, "duration", distribution="weibull")
    ach(data, "event")                          # P(event at t) = 1 / psi, psi an ACD of the durations between events

A dynamic probit (or logit, or Poisson regression on lagged counts) needs nothing new: put the lagged dependent
variable in the equation, ``Equation("d ~ d[-1] + x", family=Probit()).fit(data, method=ML())``.

INAR(1) (McKenzie 1985; Al-Osh and Alzaid 1987): each of last period's y_{t-1} units survives with probability alpha,
and Poisson(lambda) new ones arrive, so P(y_t | y_{t-1}) is the convolution of a binomial and a Poisson; the mean is
lambda / (1 - alpha) and the autocorrelation at lag k is alpha^k. The conditional least-squares estimates (y_t on
y_{t-1}) start the likelihood. The Poisson autoregression (Ferland, Latour and Oraichi 2006; Fokianos, Rahbek and
Tjostheim 2009) starts lambda at the sample mean. The ACD (Engle and Russell 1998) starts psi at the mean duration;
with exponential errors its estimates are quasi-maximum likelihood, consistent whatever the errors' distribution, with
sandwich standard errors; Weibull errors (shape k) are maximum likelihood with Hessian standard errors. Standard
errors of the Poisson and INAR models are from the Hessian.

The autoregressive conditional hazard (Hamilton and Jorda 2002) models a 0/1 event series in discrete time (a policy
rate change, a default): the probability of an event at t is h_t = 1 / psi, with psi = omega + a u + b psi_prev
updated at each event, u the duration just ended (in periods), and held between events; so it is an ACD for the
expected duration, estimated by the Bernoulli likelihood of every period. psi starts at the mean duration and is
kept above 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize, special

from pymetrix._lazy import stats as sps

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["SeriesModel", "acd", "ach", "inar", "poisson_autoregression"]


@dataclass(frozen=True, eq=False)
class SeriesModel:
    """A count or duration time-series model estimated by maximum likelihood."""

    model: str
    variable: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    loglike: float = 0.0
    nobs: int = 0
    fitted: pd.Series = field(default=None, repr=False)  # the conditional mean E[y_t | past]
    notes: tuple[str, ...] = ()

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.values)))

    @property
    def aic(self) -> float:
        return float(-2 * self.loglike + 2 * len(self.values))

    def summary(self) -> Table:
        z = self.values / self.se
        rows = [[float(v), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for v, s, t in zip(self.values, self.se, z)]
        notes = (f"Observations: {self.nobs}", f"Log likelihood: {self.loglike:.4f}", f"Akaike info criterion: {self.aic:.4f}",
                 *self.notes)
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"{self.model}: {self.variable}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.values, "se": self.se}, index=list(self.names))

    def plot(self) -> Any:
        """The series and its conditional mean."""
        from .figures import Figure, Line, Panel

        return Figure((Panel((Line(self.fitted.rename("conditional mean"), "conditional mean"),), title=f"{self.model}: {self.variable}"),))

    def __repr__(self) -> str:
        return f"<SeriesModel {self.model} {self.variable}: {dict(zip(self.names, np.round(self.values, 4)))}>"


def _series(data: Dataset, variable: str, span: Any) -> pd.Series:
    if data.kind != "time_series":
        raise SpecificationError("count and duration models need a single time series")
    if variable not in data:
        raise SpecificationError(f"{variable!r} is not a series of the data")
    s = data[variable].astype(float)
    if span is not None:
        s = s[data.sample_mask(span)]
    s = s.dropna()
    if len(s) < 30:
        raise SpecificationError("these models need at least 30 observations")
    return s


def _hessian(f: Any, x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.empty((n, n))
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _scores(per_period: Any, x: np.ndarray) -> np.ndarray:
    G = np.empty((len(per_period(x)), len(x)))
    for j in range(len(x)):
        h = 1e-6 * max(abs(x[j]), 1e-2)
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        G[:, j] = (per_period(up) - per_period(dn)) / (2 * h)
    return G


def _maximise(per_period: Any, x0: np.ndarray, bounds: Any) -> np.ndarray:
    f = lambda x: -float(np.sum(per_period(x)))  # noqa: E731
    found = optimize.minimize(f, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000})
    polish = optimize.minimize(f, found.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000})
    x = polish.x if polish.fun <= found.fun and _inside(polish.x, bounds) else found.x
    return x


def _inside(x: np.ndarray, bounds: Any) -> bool:
    return all((lo is None or v >= lo) and (hi is None or v <= hi) for v, (lo, hi) in zip(x, bounds))


# INAR(1)


def _inar_logpmf(y: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """log P(y_t | y_{t-1}) for t = 1, ..., T - 1: sum over the survivors k of Binomial(k; y_{t-1}, alpha) Poisson(y_t - k)."""
    prev, cur = y[:-1].astype(int), y[1:].astype(int)
    m = int(min(prev.max(), cur.max()))
    k = np.arange(m + 1)
    terms = (sps.binom.logpmf(k[None, :], prev[:, None], alpha) + sps.poisson.logpmf(cur[:, None] - k[None, :], lam))
    terms = np.where((k[None, :] <= np.minimum(prev, cur)[:, None]), terms, -np.inf)
    return special.logsumexp(terms, axis=1)


@api("pymetrix:inar")
def inar(data: Dataset, variable: str, span: Any = None) -> SeriesModel:
    """INAR(1) with Poisson innovations by maximum likelihood, from conditional least squares."""
    s = _series(data, variable, span)
    y = s.to_numpy()
    if np.any(y < 0) or np.any(y != np.round(y)):
        raise SpecificationError(f"{variable} must hold non-negative whole numbers (counts)")
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    lam0, a0 = np.linalg.lstsq(X, y[1:], rcond=None)[0]  # conditional least squares
    a0 = float(np.clip(a0, 0.01, 0.99))
    lam0 = float(max(lam0, 0.1 * y.mean() + 1e-3))
    per = lambda th: _inar_logpmf(y, th[0], th[1])  # noqa: E731
    theta = _maximise(per, np.array([a0, lam0]), [(1e-6, 1 - 1e-6), (1e-8, None)])
    cov = np.linalg.pinv(-_hessian(lambda th: float(np.sum(per(th))), theta))
    alpha, lam = theta
    fitted = pd.Series(np.r_[np.nan, alpha * y[:-1] + lam], index=s.index, name="conditional mean")
    notes = (f"Conditional least squares started it: alpha {a0:.4f}, lambda {lam0:.4f}. Mean lambda / (1 - alpha) = "
             f"{lam / (1 - alpha):.4f} (sample mean {y.mean():.4f}); the likelihood conditions on the first observation.",)
    return SeriesModel("INAR(1), Poisson innovations", variable, ("alpha", "lambda"), theta, cov, float(np.sum(per(theta))),
                       len(y) - 1, fitted, notes)


# Poisson autoregression


def _intensity(y: np.ndarray, omega: float, a: float, b: float) -> np.ndarray:
    from scipy.signal import lfilter

    x = np.empty(len(y))
    x[0] = y.mean()
    x[1:] = omega + a * y[:-1]
    return lfilter([1.0], [1.0, -b], x)


@api("pymetrix:poisson_autoregression")
def poisson_autoregression(data: Dataset, variable: str, feedback: bool = True, span: Any = None) -> SeriesModel:
    """The Poisson autoregression y_t | past ~ Poisson(lambda_t), lambda_t = omega + a y_{t-1} + b lambda_{t-1}
    (``feedback=False``: b = 0), by maximum likelihood; a + b < 1 for stationarity."""
    s = _series(data, variable, span)
    y = s.to_numpy()
    if np.any(y < 0) or np.any(y != np.round(y)):
        raise SpecificationError(f"{variable} must hold non-negative whole numbers (counts)")

    def per(th: np.ndarray) -> np.ndarray:
        lam = _intensity(y, th[0], th[1], th[2] if feedback else 0.0)
        return sps.poisson.logpmf(y[1:], np.maximum(lam[1:], 1e-12))

    m = y.mean()
    x0 = np.array([0.3 * m, 0.3, 0.4]) if feedback else np.array([0.7 * m, 0.3])
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)] if feedback else [(1e-8, None), (0.0, 0.999)]
    theta = _maximise(per, x0, bounds)
    if feedback and theta[1] + theta[2] >= 1:
        raise SpecificationError("the estimates are nonstationary (a + b >= 1)")
    cov = np.linalg.pinv(-_hessian(lambda th: float(np.sum(per(th))), theta))
    lam = _intensity(y, theta[0], theta[1], theta[2] if feedback else 0.0)
    names = ("omega", "a", "b") if feedback else ("omega", "a")
    mean = theta[0] / (1 - theta[1] - (theta[2] if feedback else 0))
    return SeriesModel("Poisson autoregression" + (" with feedback" if feedback else ""), variable, names, theta, cov,
                       float(np.sum(per(theta))), len(y) - 1, pd.Series(lam, index=s.index, name="conditional mean"),
                       (f"Implied mean {mean:.4f} (sample mean {m:.4f}); lambda starts at the sample mean.",))


# ACD


def _psi(x: np.ndarray, omega: float, a: float, b: float) -> np.ndarray:
    from scipy.signal import lfilter

    z = np.empty(len(x))
    z[0] = x.mean()
    z[1:] = omega + a * x[:-1]
    return lfilter([1.0], [1.0, -b], z)


@api("pymetrix:acd")
def acd(data: Dataset, variable: str, distribution: Literal["exponential", "weibull"] = "exponential", span: Any = None) -> SeriesModel:
    """The ACD(1, 1) of Engle and Russell (1998): x_t = psi_t e_t, psi_t = omega + a x_{t-1} + b psi_{t-1}, with
    exponential (quasi-maximum likelihood) or Weibull errors of unit mean."""
    s = _series(data, variable, span)
    x = s.to_numpy()
    if np.any(x <= 0):
        raise SpecificationError(f"{variable} must hold positive durations")
    if distribution not in ("exponential", "weibull"):
        raise SpecificationError("distribution is 'exponential' or 'weibull'")
    weibull = distribution == "weibull"

    def per(th: np.ndarray) -> np.ndarray:
        psi = np.maximum(_psi(x, th[0], th[1], th[2]), 1e-12)
        if not weibull:
            return -np.log(psi) - x / psi
        k = th[3]
        g = special.gamma(1 + 1 / k)  # unit mean: the scale is psi / Gamma(1 + 1/k)
        u = g * x / psi
        return np.log(k) - np.log(x) + k * np.log(u) - u**k

    m = x.mean()
    x0 = np.array([0.1 * m, 0.1, 0.8] + ([1.0] if weibull else []))
    bounds = [(1e-10, None), (0.0, 0.999), (0.0, 0.999)] + ([(0.05, 20.0)] if weibull else [])
    theta = _maximise(per, x0, bounds)
    if theta[1] + theta[2] >= 1:
        raise SpecificationError("the estimates are nonstationary (a + b >= 1)")
    H = _hessian(lambda th: float(np.sum(per(th))), theta)
    Hinv = np.linalg.pinv(-H)
    if weibull:
        cov = Hinv
        note = "Weibull errors with unit mean: maximum likelihood, Hessian standard errors."
    else:
        G = _scores(per, theta)
        cov = Hinv @ (G.T @ G) @ Hinv
        note = "Exponential errors: quasi-maximum likelihood, consistent for any unit-mean errors; sandwich standard errors."
    names = ("omega", "a", "b") + (("shape",) if weibull else ())
    psi = _psi(x, *theta[:3])
    return SeriesModel(f"ACD(1, 1), {distribution} errors", variable, names, theta, cov, float(np.sum(per(theta))), len(x),
                       pd.Series(psi, index=s.index, name="conditional mean"), (note, "psi starts at the mean duration."))


# ACH


def _hazard(x: np.ndarray, omega: float, a: float, b: float, start: float) -> np.ndarray:
    """h_t = 1 / psi_{N(t-1)}: psi updated after each event with the duration just ended, held between events."""
    h = np.empty(len(x))
    psi, last = start, -1
    for t in range(len(x)):
        h[t] = 1 / max(psi, 1.0 + 1e-9)
        if x[t]:
            u = t - last
            psi = omega + a * u + b * psi
            last = t
    return h


@api("pymetrix:ach")
def ach(data: Dataset, variable: str, span: Any = None) -> SeriesModel:
    """The autoregressive conditional hazard ACH(1, 1) of the 0/1 event series ``variable`` by maximum likelihood."""
    s = _series(data, variable, span)
    x = s.to_numpy()
    if not np.isin(x, (0.0, 1.0)).all():
        raise SpecificationError(f"{variable} must be 0 or 1 (did the event happen in the period?)")
    events = int(x.sum())
    if events < 10:
        raise SpecificationError("the ACH needs at least 10 events")
    mean_duration = len(x) / events

    def per(th: np.ndarray) -> np.ndarray:
        h = _hazard(x, th[0], th[1], th[2], mean_duration)
        h = np.clip(h, 1e-12, 1 - 1e-12)
        return x * np.log(h) + (1 - x) * np.log(1 - h)

    x0 = np.array([0.3 * mean_duration, 0.1, 0.6])
    theta = _maximise(per, x0, [(1e-6, None), (0.0, 0.999), (0.0, 0.999)])
    cov = np.linalg.pinv(-_hessian(lambda th: float(np.sum(per(th))), theta))
    h = _hazard(x, *theta, mean_duration)
    return SeriesModel("ACH(1, 1)", variable, ("omega", "a", "b"), theta, cov, float(np.sum(per(theta))), len(x),
                       pd.Series(h, index=s.index, name="hazard"),
                       (f"{events} events in {len(x)} periods (mean duration {mean_duration:.4g}); the hazard is 1 / psi, "
                        f"psi held above 1; implied mean duration {theta[0] / max(1 - theta[1] - theta[2], 1e-12):.4g}.",))
