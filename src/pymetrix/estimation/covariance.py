"""Covariance estimators shared by least-squares methods.

``X`` is the regressor matrix (or the Jacobian of the residuals for NLS),
``u`` the residuals, and ``df_resid`` the residual degrees of freedom after
any absorbed fixed effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import SpecificationError


@dataclass(frozen=True)
class Cluster:
    """Cluster-robust covariance by one or two grouping variables or axes."""

    levels: tuple[str, ...]

    def __init__(self, *levels: str):
        if not 1 <= len(levels) <= 2:
            raise SpecificationError("Cluster takes one or two grouping names")
        object.__setattr__(self, "levels", tuple(levels))


def default_hac_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def covariance(
    kind: str | Cluster,
    X: np.ndarray,
    u: np.ndarray,
    df_resid: int,
    *,
    groups: Sequence[np.ndarray] = (),
    hac_lags: int | None = None,
    hac_correction: bool = True,
) -> np.ndarray:
    n = len(u)
    bread = np.linalg.inv(X.T @ X)
    if isinstance(kind, Cluster):
        return _cluster(X, u, bread, groups, df_resid)
    if kind == "classic":
        return (u @ u / df_resid) * bread
    scores = X * u[:, None]
    if kind == "white":  # HC1
        return n / df_resid * bread @ (scores.T @ scores) @ bread
    if kind == "hac":  # Newey-West, Bartlett kernel; times n / (n - k) unless hac_correction is False (statsmodels' default)
        lags = default_hac_lags(n) if hac_lags is None else hac_lags
        S = scores.T @ scores
        for lag in range(1, lags + 1):
            gamma = scores[lag:].T @ scores[:-lag]
            S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
        return (n / df_resid if hac_correction else 1.0) * bread @ S @ bread
    raise SpecificationError(f"unknown covariance {kind!r}; use 'classic', 'white', 'hac' or Cluster(...)")


def _cluster(X: np.ndarray, u: np.ndarray, bread: np.ndarray, groups: Sequence[np.ndarray], df_resid: int) -> np.ndarray:
    n = len(u)
    k_eff = n - df_resid

    def one(values: np.ndarray) -> np.ndarray:
        codes = pd.factorize(values)[0]
        G = codes.max() + 1
        sums = np.zeros((G, X.shape[1]))
        np.add.at(sums, codes, X * u[:, None])
        c = G / (G - 1) * (n - 1) / (n - k_eff)
        return c * bread @ (sums.T @ sums) @ bread

    if len(groups) == 1:
        return one(groups[0])
    a, b = (pd.Series(g).astype(str).to_numpy() for g in groups)
    both = np.char.add(np.char.add(a.astype(str), "|"), b.astype(str))
    return one(a) + one(b) - one(both)
