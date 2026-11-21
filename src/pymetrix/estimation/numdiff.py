"""Numerical derivatives for likelihoods without analytic derivatives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..errors import SpecificationError

_EPS = np.finfo(float).eps


def _steps(x: np.ndarray, power: float) -> np.ndarray:
    return _EPS**power * np.maximum(np.abs(x), 0.1)


def hessian(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    x = np.asarray(x, dtype=float)
    k = len(x)
    h = _steps(x, 1 / 4)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian of a vector function: rows are observations."""
    x = np.asarray(x, dtype=float)
    h = _steps(x, 1 / 3)
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h[i]
        cols.append((f(x + e) - f(x - e)) / (2 * h[i]))
    return np.column_stack(cols)


def ml_covariance(kind: str, loglike: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """``classic``: inverse observed information. ``white``: sandwich (quasi-ML)."""
    H = hessian(lambda p: float(np.sum(loglike(p))), x)
    inv_info = np.linalg.inv(-H)
    if kind == "classic":
        return inv_info
    if kind == "white":
        S = jacobian(loglike, x)
        return inv_info @ (S.T @ S) @ inv_info
    raise SpecificationError(f"cov must be 'classic' or 'white' for this model, not {kind!r}")
