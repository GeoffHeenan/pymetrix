"""Correlated shocks: the covariance of a model's shocks from their standard deviations and correlations.

    sol.system(measurement, shock_sd={"ey": 0.5, "ei": 0.2}, shock_corr={("ey", "ei"): 0.3})
    model.estimate(data, measurement, shock_sd=..., shock_corr={("ey", "ei"): 0.3})   # corr_ey_ei estimated too

``shock_corr`` maps pairs of shocks to their correlation, ``{("ey", "ei"): 0.3}`` (either order), or names
``{"corr_ey_ei": 0.3}``; pairs not given are uncorrelated. The covariance is ``D C D``, ``D`` the diagonal of
standard deviations and ``C`` the correlations, which must be positive definite. As an estimated parameter a
correlation is ``corr_<first>_<second>``, the two shocks in the model's order (as Dynare's ``corr`` in
``estimated_params``), within (-1, 1).

What uses it, for first-order solutions: the state-space form (``Q`` of the Kalman filter, the likelihood and the
smoother), stochastic simulation (normal draws with this covariance), theoretical moments, and impulse responses
and variance decompositions, which with correlated shocks are orthogonalised by the Cholesky factor of the
covariance, the shocks in the model's order, as Dynare's ``stoch_simul``: the response to the j-th shock is the
j-th column of ``Q chol(D C D)``, so the first shock moves the others it is correlated with, and the decomposition
assigns their common part to the earlier shock. Without correlations everything is as before: unit or ``shock_sd``
sized, independent shocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .solvers import ModelError

__all__ = ["corr_pairs", "correlation_matrix", "shock_covariance", "shock_factor"]


def corr_pairs(shocks: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``{"corr_<a>_<b>": (a, b)}`` for each pair of shocks, ``a`` before ``b`` in ``shocks``. A name two pairs would
    share (shocks with underscores: ``e_a`` and ``b_c`` against ``e`` and ``a_b_c``) is left out."""
    out: dict[str, tuple[str, str]] = {}
    clash: set[str] = set()
    for i, a in enumerate(shocks):
        for b in shocks[i + 1:]:
            name = f"corr_{a}_{b}"
            if name in out:
                clash.add(name)
            out[name] = (a, b)
    return {n: p for n, p in out.items() if n not in clash}


def _pair(key: Any, shocks: Sequence[str]) -> tuple[str, str]:
    if isinstance(key, str):
        pairs = corr_pairs(shocks)
        if key not in pairs:
            raise ModelError(f"shock_corr names {key!r}, which is not corr_<shock>_<shock> for two shocks of the model "
                             f"({', '.join(shocks)}); give the pair as a tuple, e.g. ({shocks[0]!r}, {shocks[-1]!r})")
        return pairs[key]
    if not isinstance(key, (tuple, list)) or len(key) != 2 or not all(isinstance(k, str) for k in key):
        raise ModelError(f"shock_corr is {{(shock, shock): correlation}}, not a key {key!r}")
    a, b = key
    unknown = [s for s in (a, b) if s not in shocks]
    if unknown:
        raise ModelError(f"shock_corr names {', '.join(unknown)}, which {'is not a shock' if len(unknown) == 1 else 'are not shocks'} "
                         f"of the model ({', '.join(shocks)})")
    if a == b:
        raise ModelError(f"shock_corr ({a!r}, {b!r}): a correlation is between two different shocks")
    return (a, b) if list(shocks).index(a) < list(shocks).index(b) else (b, a)


def normalise(shocks: Sequence[str], shock_corr: Mapping[Any, float] | None) -> dict[tuple[str, str], float]:
    """The correlations as ``{(a, b): r}``, ``a`` before ``b`` in ``shocks``; checked: shocks of the model, values in
    (-1, 1), each pair once."""
    out: dict[tuple[str, str], float] = {}
    for key, value in dict(shock_corr or {}).items():
        pair = _pair(key, list(shocks))
        value = float(value)
        if not -1 < value < 1:
            raise ModelError(f"the correlation of {pair[0]} and {pair[1]} is {value:g}, outside (-1, 1)")
        if pair in out and out[pair] != value:
            raise ModelError(f"the correlation of {pair[0]} and {pair[1]} is given twice ({out[pair]:g} and {value:g})")
        out[pair] = value
    return out


def correlation_matrix(shocks: Sequence[str], shock_corr: Mapping[Any, float] | None) -> np.ndarray | None:
    """The shocks' correlation matrix, or None when no correlation is other than zero (so that uncorrelated shocks take
    exactly the arithmetic they always did). Refused when not positive definite."""
    pairs = {k: v for k, v in normalise(shocks, shock_corr).items() if v != 0}
    if not pairs:
        return None
    shocks = list(shocks)
    C = np.eye(len(shocks))
    for (a, b), r in pairs.items():
        i, j = shocks.index(a), shocks.index(b)
        C[i, j] = C[j, i] = r
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ModelError("the shocks' correlations are not a valid correlation matrix (not positive definite): "
                         + ", ".join(f"{a}-{b} {r:g}" for (a, b), r in pairs.items())) from None
    return C


def shock_factor(shocks: Sequence[str], sd: np.ndarray, shock_corr: Mapping[Any, float] | None) -> np.ndarray | None:
    """``F`` with ``F F'`` the shocks' covariance, lower triangular (the Cholesky factor, ``diag(sd) chol(C)``), or None
    when the shocks are uncorrelated."""
    C = correlation_matrix(shocks, shock_corr)
    if C is None:
        return None
    return np.asarray(sd, dtype=float)[:, None] * np.linalg.cholesky(C)


def shock_covariance(shocks: Sequence[str], sd: np.ndarray, shock_corr: Mapping[Any, float] | None) -> np.ndarray:
    """The covariance ``D C D`` (diagonal without correlations)."""
    sd = np.asarray(sd, dtype=float)
    C = correlation_matrix(shocks, shock_corr)
    if C is None:
        return np.diag(sd ** 2)
    return sd[:, None] * C * sd[None, :]
