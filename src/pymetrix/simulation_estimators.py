"""Estimation by simulation: indirect inference, the simulated method of moments and the efficient method of moments
(Martin, Hurn and Harris 2013, chapter 12; Gourieroux, Monfort and Renault 1993; Duffie and Singleton 1993; Gallant
and Tauchen 1996).

    def simulate(theta, shocks):                # the model: a path from parameters and fixed random numbers
        return shocks[1:] + theta[0] * shocks[:-1]                  # an MA(1)
    def ar(y):                                   # an auxiliary model that is easy to fit: an AR(3) by least squares
        return ar_coefficients(y, 3)
    ii = indirect_inference(y, simulate, ar, theta0=[0.2], paths=20)
    ii.summary()                                 # the estimate, its standard error and the test of the auxiliary fit

All three take the data as an array, a pandas Series (a dataset's column), a DataFrame or a ``Dataset`` (one series
gives a vector, as a 1-d array would), and choose the model's parameters theta so that simulated data look like the
data through a summary:

- **Indirect inference**: the auxiliary model's estimates beta, on the data and averaged over ``paths`` simulated
  paths, as close as possible: min (b_data - b_sim(theta))' W (b_data - b_sim(theta)), with W the inverse of b's
  covariance (estimated by simulating at a first-step estimate) for efficiency, or the identity.
- **Simulated method of moments**: moments (the mean of per-period functions ``moments(y)``) on the data and on the
  simulated paths, weighted by the inverse of the data moments' long-run (Newey-West) covariance.
- **Efficient method of moments**: the auxiliary model's score (per-period derivatives of its log likelihood,
  ``score(y, beta)``), evaluated at the data's beta on one long simulated path, as close to zero as possible, weighted
  by the inverse of the data scores' outer product.

The random numbers are drawn once (``seed``) and held fixed, so the objective is a smooth function of theta (common
random numbers), minimised by Nelder-Mead. Standard errors are the sandwich for minimum distance, inflated by
(1 + 1/S) for the simulation, S the number of paths (or the long path's length over the data's). With an efficient
weight and more summaries than parameters, J = T/(1 + 1/S) times the minimised distance is chi-squared with their
difference as degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

__all__ = ["SimulationEstimate", "ar_coefficients", "ar_score", "efficient_method_of_moments", "indirect_inference", "newey_west", "simulated_moments"]


def ar_coefficients(y: Any, p: int) -> np.ndarray:
    """An AR(p) with a constant by least squares: (constant, lags, residual variance), a convenient auxiliary model."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y) - p)] + [y[p - j:len(y) - j] for j in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
    e = y[p:] - X @ b
    return np.concatenate([b, [e @ e / len(e)]])


def ar_score(p: int) -> Callable[[Any, Any], np.ndarray]:
    """The per-period score of a Gaussian AR(p) with a constant, at (constant, lags, variance) as ``ar_coefficients``
    gives them: the auxiliary score for the efficient method of moments."""

    def score(y: Any, beta: Any) -> np.ndarray:
        y = np.asarray(y, float)
        b, s2 = np.asarray(beta[:-1], float), float(beta[-1])
        X = np.column_stack([np.ones(len(y) - p)] + [y[p - j:len(y) - j] for j in range(1, p + 1)])
        e = y[p:] - X @ b
        return np.column_stack([X * e[:, None] / s2, (e**2 / s2 - 1) / (2 * s2)])

    return score


def newey_west(G: np.ndarray, lags: int | None = None) -> np.ndarray:
    """The long-run covariance of the rows of G (demeaned), Bartlett weights."""
    G = G - G.mean(axis=0)
    T = len(G)
    L = int(np.floor(4 * (T / 100) ** (2 / 9))) if lags is None else int(lags)
    S = G.T @ G / T
    for lag in range(1, L + 1):
        Gl = G[lag:].T @ G[:-lag] / T
        S += (1 - lag / (L + 1)) * (Gl + Gl.T)
    return S


def _jacobian(f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray) -> np.ndarray:
    f0 = f(theta)
    J = np.empty((len(f0), len(theta)))
    for j in range(len(theta)):
        h = 1e-5 * max(abs(theta[j]), 1e-2)
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (f(up) - f(dn)) / (2 * h)
    return J


@dataclass(frozen=True, eq=False)
class SimulationEstimate:
    """Parameters estimated by simulation, with sandwich standard errors and the over-identification test."""

    method: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    distance: float = 0.0
    J: float | None = None
    J_df: int = 0
    paths: int = 1
    nobs: int = 0
    converged: bool = True
    summaries: int = 0

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.values / self.se
        rows = [[float(v), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for v, s, t in zip(self.values, self.se, z)]
        notes = [f"{self.method}: {self.summaries} summaries for {len(self.values)} parameters; {self.nobs} observations; "
                 f"standard errors inflated by (1 + 1/S), S = {self.paths:g}."]
        if self.J is not None:
            notes.append(f"J = {self.J:.4f}, chi-squared({self.J_df}), p = {sps.chi2.sf(self.J, self.J_df):.4f}: small p-values reject the model.")
        if not self.converged:
            notes.append("The optimiser did not converge.")
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"Estimation by simulation: {self.method}", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.values, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<SimulationEstimate {self.method}: {dict(zip(self.names, np.round(self.values, 4)))}>"


def _minimise(objective: Callable[[np.ndarray], float], theta0: np.ndarray, maxiter: int) -> optimize.OptimizeResult:
    best = None
    start = theta0
    for _ in range(3):  # Nelder-Mead restarted from its own answer: it can stall on a ridge
        found = optimize.minimize(objective, start, method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": maxiter, "adaptive": len(theta0) > 2})
        if best is not None and abs(best.fun - found.fun) < 1e-12:
            best = found
            break
        best, start = found, found.x
    return best


def _observations(data: Any) -> np.ndarray:
    """The data as the float array ``simulate``'s paths are compared with: an array as given; a pandas Series (a
    dataset's column), a one-column DataFrame or a one-series ``Dataset`` as a vector; a wider frame or dataset as an
    array of periods by series, in its column order."""
    from .data import Dataset

    if isinstance(data, Dataset):
        data = data.frame
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy(float)
        return values[:, 0] if values.shape[1] == 1 else values
    if isinstance(data, pd.Series):
        return data.to_numpy(float)
    return np.asarray(data, float)


def _names(names: Sequence[str] | None, k: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"theta{i}" for i in range(k))
    if len(names) != k:
        raise SpecificationError(f"give a name for each of the {k} parameters")
    return tuple(names)


@api("pymetrix:indirect_inference", as_dataset=False)
def indirect_inference(data: Any, simulate: Callable[[np.ndarray, np.ndarray], Any], auxiliary: Callable[[Any], Any], theta0: Sequence[float],
                       paths: int = 10, shocks: Callable[[np.random.Generator, int], np.ndarray] | None = None, names: Sequence[str] | None = None,
                       weight: str = "efficient", seed: int = 0, maxiter: int = 4000) -> SimulationEstimate:
    """Indirect inference: theta such that the auxiliary estimates on ``paths`` simulated paths average to those on the
    data. ``simulate(theta, shocks)`` gives a path of the data's length from the random numbers ``shocks`` (by default
    standard normals, one more than the data's length; give ``shocks(rng, n)`` to draw others). ``weight``: "efficient"
    (the inverse covariance of the auxiliary estimates, by simulation at a first-step estimate) or "identity"."""
    y = _observations(data)
    T = len(y)
    rng = make_rng(seed)
    draw = shocks or (lambda r, n: r.standard_normal(n + 1))
    fixed = [draw(rng, T) for _ in range(paths)]
    b_data = np.asarray(auxiliary(y), float)
    theta0 = np.asarray(theta0, float)
    k, q = len(theta0), len(b_data)
    if q < k:
        raise SpecificationError(f"the auxiliary model has {q} estimates for {k} parameters: it cannot identify them")

    def b_sim(theta: np.ndarray) -> np.ndarray:
        return np.mean([np.asarray(auxiliary(np.asarray(simulate(theta, e), float)), float) for e in fixed], axis=0)

    def distance(theta: np.ndarray, W: np.ndarray) -> float:
        try:
            d = b_data - b_sim(theta)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return 1e12
        v = float(d @ W @ d)
        return v if np.isfinite(v) else 1e12

    found = _minimise(lambda th: distance(th, np.eye(q)), theta0, maxiter)
    theta = found.x
    # the auxiliary estimates' covariance, from fresh paths at the estimate (each of the data's length)
    check = make_rng(seed + 1)
    reps = np.array([auxiliary(np.asarray(simulate(theta, draw(check, T)), float)) for _ in range(max(200, 10 * q))], float)
    Vb = np.cov(reps.T) if q > 1 else np.atleast_2d(np.var(reps, ddof=1))
    W = np.eye(q)
    if weight == "efficient":
        W = np.linalg.pinv(Vb)
        found = _minimise(lambda th: distance(th, W), theta, maxiter)
        theta = found.x
    elif weight != "identity":
        raise SpecificationError("weight is 'efficient' or 'identity'")
    Jb = -_jacobian(b_sim, theta)  # d(b_data - b_sim)/dtheta
    bread = np.linalg.pinv(Jb.T @ W @ Jb)
    cov = (1 + 1 / paths) * bread @ Jb.T @ W @ Vb @ W @ Jb @ bread
    dist = distance(theta, W)
    J = dist / (1 + 1 / paths) if weight == "efficient" and q > k else None
    return SimulationEstimate("indirect inference", _names(names, k), theta, cov, dist, J, q - k, paths, T, bool(found.success), q)


@api("pymetrix:simulated_moments", as_dataset=False)
def simulated_moments(data: Any, simulate: Callable[[np.ndarray, np.ndarray], Any], moments: Callable[[Any], Any], theta0: Sequence[float],
                      paths: int = 10, shocks: Callable[[np.random.Generator, int], np.ndarray] | None = None, names: Sequence[str] | None = None,
                      lags: int | None = None, seed: int = 0, maxiter: int = 4000) -> SimulationEstimate:
    """The simulated method of moments: theta such that the means of the per-period ``moments(y)`` (an array, periods
    by moments) on simulated paths match the data's, weighted by the inverse of their Newey-West long-run covariance."""
    y = _observations(data)
    T = len(y)
    rng = make_rng(seed)
    draw = shocks or (lambda r, n: r.standard_normal(n + 1))
    fixed = [draw(rng, T) for _ in range(paths)]
    M = np.atleast_2d(np.asarray(moments(y), float).T).T
    m_data = M.mean(axis=0)
    theta0 = np.asarray(theta0, float)
    k, q = len(theta0), len(m_data)
    if q < k:
        raise SpecificationError(f"{q} moments cannot identify {k} parameters")
    Omega = newey_west(M, lags)
    W = np.linalg.pinv(Omega)

    def m_sim(theta: np.ndarray) -> np.ndarray:
        return np.mean([np.atleast_2d(np.asarray(moments(np.asarray(simulate(theta, e), float)), float).T).T.mean(axis=0) for e in fixed], axis=0)

    def distance(theta: np.ndarray) -> float:
        try:
            d = m_data - m_sim(theta)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return 1e12
        v = float(d @ W @ d)
        return v if np.isfinite(v) else 1e12

    found = _minimise(distance, theta0, maxiter)
    theta = found.x
    G = -_jacobian(m_sim, theta)
    cov = (1 + 1 / paths) * np.linalg.pinv(G.T @ W @ G) / T
    dist = distance(theta)
    J = T * dist / (1 + 1 / paths) if q > k else None
    return SimulationEstimate("simulated method of moments", _names(names, k), theta, cov, dist, J, q - k, paths, T, bool(found.success), q)


@api("pymetrix:efficient_method_of_moments", as_dataset=False)
def efficient_method_of_moments(data: Any, simulate: Callable[[np.ndarray, np.ndarray], Any], fit_auxiliary: Callable[[Any], Any],
                                score: Callable[[Any, Any], Any], theta0: Sequence[float], length: int = 10,
                                shocks: Callable[[np.random.Generator, int], np.ndarray] | None = None, names: Sequence[str] | None = None,
                                lags: int | None = None, seed: int = 0, maxiter: int = 4000) -> SimulationEstimate:
    """The efficient method of moments: the auxiliary model is fitted to the data (``fit_auxiliary(y)`` gives beta), and
    theta makes the mean of its per-period score (``score(y, beta)``, periods by parameters) on one simulated path
    ``length`` times the data's as near zero as possible, weighted by the inverse long-run covariance of the data's
    scores."""
    y = _observations(data)
    T = len(y)
    rng = make_rng(seed)
    draw = shocks or (lambda r, n: r.standard_normal(n + 1))
    long = draw(rng, T * int(length))
    beta = fit_auxiliary(y)
    S_data = np.atleast_2d(np.asarray(score(y, beta), float).T).T
    theta0 = np.asarray(theta0, float)
    k, q = len(theta0), S_data.shape[1]
    if q < k:
        raise SpecificationError(f"the auxiliary score has {q} elements for {k} parameters")
    W = np.linalg.pinv(newey_west(S_data, lags))

    def g(theta: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(score(np.asarray(simulate(theta, long), float), beta), float).T).T.mean(axis=0)

    def distance(theta: np.ndarray) -> float:
        try:
            d = g(theta)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return 1e12
        v = float(d @ W @ d)
        return v if np.isfinite(v) else 1e12

    found = _minimise(distance, theta0, maxiter)
    theta = found.x
    G = _jacobian(g, theta)
    cov = (1 + 1 / length) * np.linalg.pinv(G.T @ W @ G) / T
    dist = distance(theta)
    J = T * dist / (1 + 1 / length) if q > k else None
    return SimulationEstimate("efficient method of moments", _names(names, k), theta, cov, dist, J, q - k, float(length), T, bool(found.success), q)
