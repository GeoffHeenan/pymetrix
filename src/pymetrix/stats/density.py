"""Scoring forecast densities: log score, CRPS and the probability integral transform (PIT).

A forecast density is scored against what happened, ``y``:

- **log score**: the log of the predictive density at ``y`` (higher is better);
- **CRPS** (continuous ranked probability score): the integral of the squared distance between the
  predictive distribution function and the step at ``y``, in the units of ``y`` (lower is better; for a
  point forecast it is the absolute error);
- **PIT**: the predictive probability of a value at or below ``y``. Over many forecasts from a
  well-calibrated model PITs are uniform on (0, 1), and one-step PITs are independent
  (``berkowitz`` tests both).

Each comes for a normal forecast (mean and standard deviation, as forecasts with standard errors give),
for a sample of draws (as simulated forecasts give: Bayesian VARs), and for a linear pool of normals
(a weighted mixture, as combined forecasts give). Normal formulas follow Gneiting and Raftery (2007);
the mixture CRPS is Grimit, Gneiting, Berrocal and Johnson (2006); the sample CRPS is the energy form
E|X - y| - E|X - X'| / 2; the sample log score uses a Gaussian kernel density (Silverman's bandwidth).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymetrix._lazy import stats as sps

from ..errors import SpecificationError

__all__ = ["berkowitz", "crps_mixture", "crps_normal", "crps_sample", "log_score_mixture", "log_score_normal", "log_score_sample",
           "pit_mixture", "pit_normal", "pit_sample"]

_SQRT_PI = np.sqrt(np.pi)


def _check_sd(sd: Any) -> np.ndarray:
    sd = np.asarray(sd, float)
    if np.any(sd <= 0):
        raise SpecificationError("standard deviations must be positive")
    return sd


def log_score_normal(y: Any, mean: Any, sd: Any) -> np.ndarray:
    """log N(y; mean, sd^2)."""
    sd = _check_sd(sd)
    z = (np.asarray(y, float) - np.asarray(mean, float)) / sd
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * z**2


def crps_normal(y: Any, mean: Any, sd: Any) -> np.ndarray:
    """CRPS of N(mean, sd^2) at y: sd [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)], z = (y - mean) / sd."""
    sd = _check_sd(sd)
    z = (np.asarray(y, float) - np.asarray(mean, float)) / sd
    return sd * (z * (2 * sps.norm.cdf(z) - 1) + 2 * sps.norm.pdf(z) - 1 / _SQRT_PI)


def pit_normal(y: Any, mean: Any, sd: Any) -> np.ndarray:
    """P(X <= y) for X ~ N(mean, sd^2)."""
    sd = _check_sd(sd)
    return sps.norm.cdf((np.asarray(y, float) - np.asarray(mean, float)) / sd)


def _draws(draws: Any) -> np.ndarray:
    x = np.asarray(draws, float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise SpecificationError("a sample forecast needs at least two draws")
    return np.sort(x)


def crps_sample(y: float, draws: Any) -> float:
    """CRPS of the empirical distribution of ``draws`` at y: mean |x - y| - mean |x - x'| / 2 (over all pairs)."""
    x = _draws(draws)
    n = len(x)
    pairs = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n**2  # mean |x_i - x_j| over all n^2 pairs, from the sorted draws
    return float(np.mean(np.abs(x - y)) - 0.5 * pairs)


def log_score_sample(y: float, draws: Any) -> float:
    """log of a Gaussian kernel density of ``draws`` at y (Silverman's rule of thumb for the bandwidth)."""
    x = _draws(draws)
    return float(np.log(sps.gaussian_kde(x, bw_method="silverman")(y)[0]))


def pit_sample(y: float, draws: Any) -> float:
    """The share of draws below y, counting ties as half."""
    x = _draws(draws)
    return float((np.sum(x < y) + 0.5 * np.sum(x == y)) / len(x))


def _mixture(weights: Any, means: Any, sds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, m, s = (np.asarray(a, float) for a in (weights, means, sds))
    if w.shape != m.shape or w.shape != s.shape or w.ndim != 1:
        raise SpecificationError("give one weight, mean and standard deviation for each component")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise SpecificationError("mixture weights must be non-negative and sum to one")
    return w, m, _check_sd(s)


def log_score_mixture(y: float, weights: Any, means: Any, sds: Any) -> float:
    """log of sum_i w_i N(y; m_i, s_i^2)."""
    w, m, s = _mixture(weights, means, sds)
    return float(np.log(np.sum(w * sps.norm.pdf(y, m, s))))


def pit_mixture(y: float, weights: Any, means: Any, sds: Any) -> float:
    """sum_i w_i Phi((y - m_i) / s_i)."""
    w, m, s = _mixture(weights, means, sds)
    return float(np.sum(w * sps.norm.cdf(y, m, s)))


def _a(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """E|X| for X ~ N(mu, var)."""
    sd = np.sqrt(var)
    return 2 * sd * sps.norm.pdf(mu / sd) + mu * (2 * sps.norm.cdf(mu / sd) - 1)


def crps_mixture(y: float, weights: Any, means: Any, sds: Any) -> float:
    """CRPS of sum_i w_i N(m_i, s_i^2) at y: sum_i w_i A(y - m_i, s_i^2) - 1/2 sum_ij w_i w_j A(m_i - m_j, s_i^2 + s_j^2),
    with A(mu, var) = E|N(mu, var)|."""
    w, m, s = _mixture(weights, means, sds)
    first = np.sum(w * _a(y - m, s**2))
    second = np.sum(np.outer(w, w) * _a(m[:, None] - m[None, :], s[:, None] ** 2 + s[None, :] ** 2))
    return float(first - 0.5 * second)


def berkowitz(pit: Any) -> Any:
    """Berkowitz (2001) likelihood-ratio test that PITs are independent and uniform.

    The PITs are mapped to normal quantiles z = Phi^-1(PIT) and an AR(1) with mean and variance,
    z_t = c + rho z_(t-1) + e_t, e_t ~ N(0, s^2), is fitted by conditional maximum likelihood; the test
    is of c = 0, rho = 0 and s = 1 (chi-squared, 3 degrees of freedom). It is for one-step forecasts:
    PITs of h-step forecasts are correlated even when the model is right."""
    from .base import TestResult

    u = np.asarray(pit, float)
    u = u[np.isfinite(u)]
    if len(u) < 6:
        raise SpecificationError("the Berkowitz test needs at least six PITs")
    z = sps.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    y, x = z[1:], z[:-1]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = float(resid @ resid / len(y))
    n = len(y)
    ll1 = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    ll0 = -0.5 * n * np.log(2 * np.pi) - 0.5 * float(y @ y)
    stat = float(max(2 * (ll1 - ll0), 0.0))
    p = float(sps.chi2.sf(stat, 3))
    return TestResult("Berkowitz", "the PITs are independent and uniform (the densities are well calibrated)", stat, 3, p, "chi2",
                      details={"mean of z": float(coef[0]), "AR(1) of z": float(coef[1]), "sd of z": float(np.sqrt(s2)), "observations": n})
