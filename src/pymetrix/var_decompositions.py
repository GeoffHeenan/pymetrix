"""Variance and historical decompositions for VARs whose parameters change over time: the shared arithmetic behind
``FittedBVARSV.fevd`` / ``historical_decomposition``, ``FittedMSVAR.fevd`` / ``historical_decomposition`` and
``FittedQuantileVAR.fevd``.

A historical decomposition of y_t = c_t + sum_l A_l,t y_t-l + P_t e_t (the intercepts, lag matrices and impact
matrix those in effect in period t) splits each y_t into the baseline b_t = c_t + sum_l A_l,t b_t-l (the observed
values before the sample for t - l < 0) and each shock's contribution x_j,t = sum_l A_l,t x_j,t-l + P_t[:, j] e_j,t
(zero before the sample), with e_t = P_t^-1 u_t from the residuals. The recursion is linear, so b_t + sum_j x_j,t =
y_t exactly, period by period, for any path of parameters; with constant parameters it is the usual VAR
decomposition. It runs over a stack of draws at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["fevd_shares", "tv_contributions"]


def tv_contributions(lagm: np.ndarray, impact: np.ndarray, U: np.ndarray, const: np.ndarray, history: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contributions (N, T, variable, shock), baselines (N, T, K) and shocks (N, T, K) of N draws.

    ``lagm`` (N, T, p, K, K): the lag matrices in effect each period (broadcast views are fine); ``impact`` (N, T, K, K);
    ``U`` (N, T, K) the residuals; ``const`` (N, T, K) the intercepts in effect; ``history`` (N, p, K) the p
    observations before the sample, oldest first."""
    N, T, K = U.shape
    p = lagm.shape[2]
    e = np.linalg.solve(impact, U[..., None])[..., 0]
    contrib = np.zeros((N, T, K, K))
    base = np.zeros((N, T, K))
    for t in range(T):
        x = impact[:, t] * e[:, t, None, :]
        b = const[:, t].copy()
        for l in range(1, p + 1):
            Al = lagm[:, t, l - 1]
            if t - l >= 0:
                x = x + Al @ contrib[:, t - l]
                b = b + np.einsum("nij,nj->ni", Al, base[:, t - l])
            else:
                b = b + np.einsum("nij,nj->ni", Al, history[:, p + t - l])
        contrib[:, t] = x
        base[:, t] = b
    return contrib, base, e


def fevd_shares(theta: np.ndarray) -> np.ndarray:
    """Shares (h, variable, shock) of the forecast-error variance from the structural moving-average weights theta
    (h, variable, shock): the cumulated squares, each row divided by its total."""
    c = np.cumsum(theta**2, axis=-3)
    return c / c.sum(axis=-1, keepdims=True)


def decomposition(contributions: np.ndarray, baselines: np.ndarray, shocks: np.ndarray, Y: np.ndarray, index: pd.Index,
                  variables: Sequence[str], label: str, measure: str, coverage: float | None = None, source: str = "") -> Any:
    """A ``Decomposition`` from the draws' contributions (N, T, K, K), baselines (N, T, K) and shocks (N, T, K): the
    point the mean over draws (the parts still add up to the data, as the mean is linear), bands the pointwise
    quantiles when there is more than one draw."""
    from .models.decompose import Decomposition

    names = list(variables)

    def frames(x: np.ndarray) -> dict[str, pd.DataFrame]:
        return {v: pd.DataFrame(x[:, i, :], index=index, columns=names) for i, v in enumerate(names)}

    point, base = contributions.mean(axis=0), baselines.mean(axis=0)
    baseline = pd.DataFrame(base, index=index, columns=names)
    bands: dict[str, Any] = {}
    if coverage is not None and len(contributions) > 1:
        lo, med, hi = np.quantile(contributions, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        bands = {"lower": frames(lo), "median": frames(med), "upper": frames(hi), "coverage": coverage, "band_source": source}
    return Decomposition(frames(point), pd.DataFrame(Y, index=index, columns=names) - baseline,
                         pd.DataFrame(shocks.mean(axis=0), index=index, columns=names), title=f"Historical decomposition ({label})",
                         measure=measure, baseline=baseline, **bands)
