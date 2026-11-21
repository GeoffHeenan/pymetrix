"""Continuous-time diffusions by exact maximum likelihood, from observations dt apart.

    diffusion(data, "r", model="vasicek", dt=1/12)    # dr = kappa (theta - r) dt + sigma dW (Ornstein-Uhlenbeck)
    diffusion(data, "r", model="cir", dt=1/12)        # dr = kappa (theta - r) dt + sigma sqrt(r) dW (Cox-Ingersoll-Ross)
    diffusion(data, "p", model="gbm", dt=1/252)       # dp = mu p dt + sigma p dW (geometric Brownian motion)

Each uses the exact transition density, so there is no discretisation bias however wide dt is: Vasicek's is normal,
r_{t+dt} | r_t ~ N(theta + (r_t - theta) e^{-kappa dt}, sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa)); CIR's is a scaled
noncentral chi-squared, 2c r_{t+dt} ~ chi2'(4 kappa theta / sigma^2, 2c r_t e^{-kappa dt}), c = 2 kappa / (sigma^2
(1 - e^{-kappa dt})); GBM's log returns are N((mu - sigma^2 / 2) dt, sigma^2 dt). The likelihood conditions on the first
observation; standard errors from the Hessian. For models with no closed-form density, see
``pymetrix.simulation_estimators`` (indirect inference, EMM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from .count_series import SeriesModel, _hessian, _series
from .errors import SpecificationError

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["diffusion"]


def _vasicek_logpdf(x: np.ndarray, kappa: float, theta: float, sigma: float, dt: float) -> np.ndarray:
    a = np.exp(-kappa * dt)
    mean = theta + (x[:-1] - theta) * a
    var = sigma**2 * (1 - a**2) / (2 * kappa)
    return sps.norm.logpdf(x[1:], mean, np.sqrt(var))


def _cir_logpdf(x: np.ndarray, kappa: float, theta: float, sigma: float, dt: float) -> np.ndarray:
    a = np.exp(-kappa * dt)
    c = 2 * kappa / (sigma**2 * (1 - a))
    df = 4 * kappa * theta / sigma**2
    nc = 2 * c * x[:-1] * a
    return sps.ncx2.logpdf(2 * c * x[1:], df, nc) + np.log(2 * c)


@api("pymetrix:diffusion")
def diffusion(data: Dataset, variable: str, model: Literal["vasicek", "cir", "gbm"] = "vasicek", dt: float = 1.0,
              span: Any = None) -> SeriesModel:
    """A diffusion of ``variable`` observed every ``dt`` (in the time units of the parameters: 1/12 for monthly
    observations of annual rates) by exact maximum likelihood."""
    if dt <= 0:
        raise SpecificationError("dt, the time between observations, must be positive")
    s = _series(data, variable, span)
    x = s.to_numpy()
    if model == "gbm":
        if np.any(x <= 0):
            raise SpecificationError("geometric Brownian motion needs positive values")
        r = np.diff(np.log(x))
        n = len(r)
        sigma = float(np.sqrt(np.mean((r - r.mean()) ** 2) / dt))  # the ML estimates, in closed form
        mu = float(r.mean() / dt + sigma**2 / 2)
        theta = np.array([mu, sigma])
        # mu = mean(r) / dt + sigma^2 / 2, the mean independent of sigma, var(sigma) = sigma^2 / 2n: the delta method
        cov = np.array([[sigma**2 / (n * dt) + sigma**4 / (2 * n), sigma**3 / (2 * n)], [sigma**3 / (2 * n), sigma**2 / (2 * n)]])
        ll = float(np.sum(sps.norm.logpdf(r, (mu - sigma**2 / 2) * dt, sigma * np.sqrt(dt))) - np.sum(np.log(x[1:])))
        names = ("mu", "sigma")
        mean = pd.Series(np.r_[np.nan, x[:-1] * np.exp(mu * dt)], index=s.index, name="conditional mean")
        return SeriesModel("Geometric Brownian motion", variable, names, theta, cov, ll, n, mean,
                           (f"Exact maximum likelihood, dt = {dt:g}; the likelihood is of the levels.",))
    if model not in ("vasicek", "cir"):
        raise SpecificationError("model is 'vasicek', 'cir' or 'gbm'")
    if model == "cir" and np.any(x <= 0):
        raise SpecificationError("the CIR model needs positive values")
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    (b0, b1), *_ = np.linalg.lstsq(X, x[1:], rcond=None)  # the AR(1) the Vasicek model implies: a start
    b1 = float(np.clip(b1, 1e-4, 1 - 1e-4))
    kappa0 = -np.log(b1) / dt
    theta0 = b0 / (1 - b1)
    resid = x[1:] - X @ np.array([b0, b1])
    sigma0 = float(np.sqrt(resid.var() * 2 * kappa0 / (1 - b1**2)))
    if model == "cir":
        sigma0 = sigma0 / np.sqrt(max(np.mean(x), 1e-8))
    logpdf = _vasicek_logpdf if model == "vasicek" else _cir_logpdf

    def per(th: np.ndarray) -> np.ndarray:
        kappa, theta, sigma = th
        if kappa <= 0 or sigma <= 0 or (model == "cir" and theta <= 0):
            return np.full(len(x) - 1, -1e10)
        return logpdf(x, kappa, theta, sigma, dt)

    f = lambda th: -float(np.sum(per(th)))  # noqa: E731
    found = optimize.minimize(f, np.array([kappa0, theta0, sigma0]), method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000})
    theta = found.x
    cov = np.linalg.pinv(-_hessian(lambda th: float(np.sum(per(th))), theta))
    kappa, mean_level, sigma = theta
    cmean = pd.Series(np.r_[np.nan, mean_level + (x[:-1] - mean_level) * np.exp(-kappa * dt)], index=s.index, name="conditional mean")
    notes = [f"Exact maximum likelihood with dt = {dt:g}; half-life of a deviation {np.log(2) / kappa:.4g} time units."]
    if model == "cir":
        notes.append(f"Feller condition 2 kappa theta >= sigma^2: {2 * kappa * mean_level:.4g} against {sigma**2:.4g}.")
    return SeriesModel("Vasicek (Ornstein-Uhlenbeck)" if model == "vasicek" else "Cox-Ingersoll-Ross", variable,
                       ("kappa", "theta", "sigma"), theta, cov, float(-found.fun), len(x) - 1, cmean, tuple(notes))
