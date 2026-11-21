"""GARCH-in-mean (Engle, Lilien and Robins 1987): the conditional variance enters the mean, as a risk premium.

    garch_in_mean(data, "r")                     # r_t = mu + delta sigma_t^2 + e_t, sigma_t^2 = omega + a e_{t-1}^2 + b sigma_{t-1}^2
    garch_in_mean(data, "r", form="sd")          # delta sigma_t
    garch_in_mean(data, "r", form="log")         # delta log sigma_t^2

Normal errors, by maximum likelihood, with Bollerslev-Wooldridge sandwich standard errors. As arch does, the
recursion starts from a backcast (an exponentially weighted mean of the first 75 squared demeaned returns, weights
0.94^j), so ``form=None`` (no in-mean term) reproduces arch's constant-mean GARCH(1, 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from ._api import api
from .count_series import SeriesModel, _hessian, _scores, _series
from .errors import SpecificationError

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["garch_in_mean"]


def _backcast(e: np.ndarray) -> float:
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    return float(np.sum(e[:tau] ** 2 * w / w.sum()))


def _path(r: np.ndarray, theta: np.ndarray, form: str | None, backcast: float) -> tuple[np.ndarray, np.ndarray]:
    """(residuals, conditional variances), the mean depending on the current variance."""
    mu, omega, a, b = theta[:4]
    delta = theta[4] if form else 0.0
    T = len(r)
    e, h = np.empty(T), np.empty(T)
    prev_e2, prev_h = backcast, backcast
    for t in range(T):
        h[t] = omega + a * prev_e2 + b * prev_h
        g = 0.0 if not form else h[t] if form == "variance" else np.sqrt(h[t]) if form == "sd" else np.log(h[t])
        e[t] = r[t] - mu - delta * g
        prev_e2, prev_h = e[t] ** 2, h[t]
    return e, h


@api("pymetrix:garch_in_mean")
def garch_in_mean(data: Dataset, variable: str, form: Literal["variance", "sd", "log"] | None = "variance", span: Any = None) -> SeriesModel:
    """A GARCH(1, 1)-in-mean model of ``variable`` by maximum likelihood with normal errors (``form=None``: an ordinary
    constant-mean GARCH)."""
    if form not in ("variance", "sd", "log", None):
        raise SpecificationError("form is 'variance', 'sd', 'log' or None")
    s = _series(data, variable, span)
    r = s.to_numpy()
    if len(r) < 100:
        raise SpecificationError("a GARCH-in-mean needs at least 100 observations")
    backcast = _backcast(r - r.mean())

    def per(th: np.ndarray) -> np.ndarray:
        if th[1] <= 0 or th[2] < 0 or th[3] < 0 or th[2] + th[3] >= 1:
            return np.full(len(r), -1e10)
        e, h = _path(r, th, form, backcast)
        return -0.5 * (np.log(2 * np.pi) + np.log(h) + e**2 / h)

    v = r.var()
    x0 = np.array([r.mean(), 0.05 * v, 0.05, 0.9] + ([0.0] if form else []))
    f = lambda th: -float(np.sum(per(th)))  # noqa: E731
    best = optimize.minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 40_000})
    best = optimize.minimize(f, best.x, method="BFGS", options={"gtol": 1e-8}) if np.isfinite(f(best.x)) else best
    theta = best.x
    H = _hessian(lambda th: float(np.sum(per(th))), theta)
    G = _scores(per, theta)
    Hinv = np.linalg.pinv(-H)
    cov = Hinv @ (G.T @ G) @ Hinv
    e, h = _path(r, theta, form, backcast)
    names = ("mu", "omega", "alpha", "beta") + (("delta",) if form else ())
    label = {None: "GARCH(1, 1)", "variance": "GARCH(1, 1)-in-mean (variance)", "sd": "GARCH(1, 1)-in-mean (standard deviation)",
             "log": "GARCH(1, 1)-in-mean (log variance)"}[form]
    fitted = pd.Series(r - e, index=s.index, name="conditional mean")
    return SeriesModel(label, variable, names, theta, cov, float(np.sum(per(theta))), len(r), fitted,
                       ("Normal errors; Bollerslev-Wooldridge (sandwich) standard errors; the recursion starts from arch's backcast.",
                        f"Conditional volatility: mean {np.sqrt(h).mean():.4f}."))
