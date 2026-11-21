"""Theoretical moments of a first-order solution: variances, autocorrelations and variance decompositions.

    m = solution.moments(shock_sd={"ey": 0.5, "epi": 0.3, "ei": 0.2}, lags=4)
    m.std, m.covariance, m.autocorrelation     # of the model's variables, implied by the model (no simulation)
    m.shares                                   # the share of each variable's variance due to each shock
    solution.fevd(20, shock_sd)                # the same by forecast horizon (an FEVD, as for VARs)

For ``x(t) = P x(t-1) + Q e(t)`` with independent shocks of standard deviations ``s``, the variance of the state
solves the discrete Lyapunov equation ``S = P S P' + Q diag(s^2) Q'``, and the autocovariance at lag k is ``P^k S``.
Each shock's part of the variance solves the same equation with only that shock. The h-step forecast-error
variance adds up the first h impulse responses. As Dynare's ``stoch_simul`` reports. Models with unit roots have
no unconditional variance; their forecast-error variances, by horizon, still exist.

With correlated shocks (``shock_corr``) the variance takes the full covariance, ``S = P S P' + Q Σ Q'``, and the
decompositions, as Dynare's, are of the shocks orthogonalised by the Cholesky factor of ``Σ`` in the model's order of
the shocks: they still add up to one, but the part two correlated shocks share goes to the one that comes first
(reorder the shocks to see the other attribution). The decomposition's notes say so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import linalg

from ..output import Table
from .solvers import ModelError

if TYPE_CHECKING:
    from .linear import LinearSolution

__all__ = ["Moments", "moments", "fevd"]


@dataclass(frozen=True, eq=False)
class Moments:
    """Unconditional moments of a model's variables implied by its first-order solution."""

    covariance: pd.DataFrame
    autocorrelation: pd.DataFrame  # variable x lag
    shares: pd.DataFrame  # variable x shock: share of the variance (each row sums to one)
    shock_sd: Mapping[str, float]
    shock_corr: Mapping[tuple[str, str], float] = field(default_factory=dict)

    @property
    def std(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.covariance.to_numpy())), index=self.covariance.index, name="std. dev.")

    @property
    def correlation(self) -> pd.DataFrame:
        sd = self.std.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            values = self.covariance.to_numpy() / np.outer(sd, sd)
        return pd.DataFrame(values, index=self.covariance.index, columns=self.covariance.columns)

    def summary(self) -> Table:
        frame = pd.concat([self.std, self.autocorrelation.add_prefix("autocorr. ")], axis=1)
        notes = ["Implied by the first-order solution, in deviations from the steady state; shock standard deviations "
                 + ", ".join(f"{k} {v:g}" for k, v in self.shock_sd.items())]
        if self.shock_corr:
            notes.append("Correlations " + ", ".join(f"{a}-{b} {r:g}" for (a, b), r in self.shock_corr.items()))
        return Table.from_frame(frame, title="Theoretical moments", notes=notes)

    def table(self) -> Table:
        notes = ["Correlated shocks orthogonalised by Cholesky, in the order " + ", ".join(self.shares.columns)
                 + ": a shared part goes to the earlier shock"] if self.shock_corr else []
        return Table.from_frame(100 * self.shares, title="Variance decomposition (%), unconditional", notes=notes)

    def __repr__(self) -> str:
        return f"<Moments of {len(self.covariance)} variables>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _impact(sol: LinearSolution, shock_sd: Mapping[str, float] | None, shock_corr: Mapping[Any, float] | None = None
            ) -> tuple[np.ndarray, dict[str, float]]:
    """The impact of each (orthogonalised) shock: ``Q`` times the standard deviations, or times the Cholesky factor of
    the covariance with correlations."""
    from .covariance import shock_factor

    sd = {s: float((shock_sd or {}).get(s, 1.0)) for s in sol.shocks}
    sizes = np.array([sd[s] for s in sol.shocks])
    factor = shock_factor(sol.shocks, sizes, shock_corr)
    return (sol.Q * sizes if factor is None else sol.Q @ factor), sd


def moments(sol: LinearSolution, shock_sd: Mapping[str, float] | None = None, lags: int = 4,
            shock_corr: Mapping[Any, float] | None = None) -> Moments:
    roots = np.abs(np.linalg.eigvals(sol.P)) if sol.P.size else np.zeros(0)
    if roots.size and roots.max() >= 1 - 1e-9:
        raise ModelError("the model has a unit root, so its variables have no unconditional variance; use fevd() for "
                         "forecast-error variances by horizon, or moments of growth rates")
    from .covariance import normalise

    impact, sd = _impact(sol, shock_sd, shock_corr)
    observed = list(sol.model.observed)
    k = len(observed)
    S = linalg.solve_discrete_lyapunov(sol.P, impact @ impact.T)
    covariance = pd.DataFrame(S[:k, :k], index=observed, columns=observed)
    variances = np.diag(S)[:k]
    auto = {}
    Pk = np.eye(len(S))
    for lag in range(1, lags + 1):
        Pk = Pk @ sol.P
        with np.errstate(divide="ignore", invalid="ignore"):
            auto[lag] = np.diag(Pk @ S)[:k] / variances
    autocorrelation = pd.DataFrame(auto, index=observed)
    parts = []
    for j in range(impact.shape[1]):
        Sj = linalg.solve_discrete_lyapunov(sol.P, np.outer(impact[:, j], impact[:, j]))
        parts.append(np.diag(Sj)[:k])
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = pd.DataFrame(np.column_stack(parts) / variances[:, None], index=observed, columns=list(sol.shocks))
    return Moments(covariance, autocorrelation, shares, sd, {k: v for k, v in normalise(sol.shocks, shock_corr).items() if v != 0})


def fevd(sol: LinearSolution, periods: int = 20, shock_sd: Mapping[str, float] | None = None,
         shock_corr: Mapping[Any, float] | None = None) -> Any:
    """Shares of the h-step forecast-error variance due to each shock, h = 1, ..., ``periods`` (an ``FEVD``)."""
    from ..var import FEVD

    impact, _ = _impact(sol, shock_sd, shock_corr)
    k = len(sol.model.observed)
    responses = np.empty((periods, k, impact.shape[1]))
    x = impact
    for h in range(periods):
        responses[h] = x[:k]
        x = sol.P @ x
    contributions = np.cumsum(responses**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = contributions / contributions.sum(axis=2, keepdims=True)
    return FEVD(values, list(sol.model.observed), list(sol.shocks), identification="rational-expectations solution")
