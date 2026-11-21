"""Proxy SVARs: one structural shock identified by an external instrument (Stock and Watson 2012, 2018; Mertens and
Ravn 2013), with moving-block bootstrap bands (Jentsch and Lunsford 2019).

    var = VAR(["ff", "ip", "cpi"], lags=12).fit(data)
    psvar = var.proxy_svar(data, "ff4", normalise_on="ff", periods=36)   # the high-frequency surprise instruments ff's shock
    psvar.impact; psvar.irf.plot(); psvar.first_stage

A proxy z_t correlated with the target shock and uncorrelated with the others identifies that shock's impact column
up to scale: b is proportional to Cov(u_t, z_t), u the VAR's residuals. Scaled so the target variable moves by one on
impact (``scale="unit"``), b_j = Cov(u_j, z) / Cov(u_target, z); scaled to a one-standard-deviation shock (the
default), b is that times sigma, sigma^2 = 1 / (b' Sigma^-1 b). The responses are the VAR's moving-average
coefficients times b. The first-stage F of u_target on z (with a constant) says whether the proxy is strong.

Bands: the residuals and the proxy are resampled together in moving blocks, the data rebuilt from the VAR, and the
VAR and the proxy's impact estimated again (Jentsch and Lunsford's moving-block bootstrap, which is valid for the
proxy's weak and conditionally heteroskedastic relations, as the wild bootstrap is not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api, renamed_arguments
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table
from .results import deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset
    from .var import IRF, FittedVAR


@renamed_arguments(target="normalise_on")
def proxy_impact(U: np.ndarray, z: np.ndarray, normalise_on: int, sigma: np.ndarray, scale: str = "sd") -> np.ndarray:
    """The impact column identified by the proxy z from residuals U (T, K)."""
    zc = z - z.mean()
    cov = (U - U.mean(axis=0)).T @ zc / len(z)
    if abs(cov[normalise_on]) < 1e-14:
        raise SpecificationError("the proxy is uncorrelated with the residual of the variable it normalises on: it identifies nothing")
    b = cov / cov[normalise_on]
    if scale == "unit":
        return b
    return b / np.sqrt(b @ np.linalg.solve(sigma, b))


def first_stage(u: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """The F statistic (and p-value) of the target's residual on the proxy, with a constant."""
    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, u, rcond=None)
    e = u - X @ beta
    r = u - u.mean()
    F = ((r @ r - e @ e) / 1) / (e @ e / (len(u) - 2))
    return float(F), float(sps.f.sf(F, 1, len(u) - 2))


@dataclass(frozen=True, eq=False)
class ProxySVAR:
    """A VAR's shock identified by an external instrument: its impact, responses with bands, and the first stage."""

    var: Any = field(repr=False)
    instrument: str = ""
    normalise_on: str = ""  # was target
    scale: str = "sd"
    impact: pd.Series = field(default=None)  # by variable
    irf: Any = field(default=None, repr=False)
    first_stage: tuple[float, float] = (np.nan, np.nan)
    nobs: int = 0
    z: Any = field(default=None, repr=False)  # the proxy over the VAR's sample (NaN where missing): for the bootstrap
    block: int | None = None  # the bootstrap's block length (None: Jentsch and Lunsford's rule)

    stage = "estimate"

    def summary(self) -> Table:
        F, p = self.first_stage
        rows = [[float(v)] for v in self.impact]
        unit = "a one-standard-deviation shock" if self.scale == "sd" else f"a shock moving {self.normalise_on} by one on impact"
        return Table(["impact"], rows, index=list(self.impact.index), index_name="variable",
                     title=f"Proxy SVAR: the {self.normalise_on} shock, instrumented by {self.instrument}",
                     notes=(f"Impact of {unit}; {self.nobs} periods with the proxy.",
                            f"First stage: F = {F:.2f} (p = {p:.4f}) for {self.normalise_on}'s residual on {self.instrument}"
                            + ("; below 10, the proxy is weak." if F < 10 else "."),
                            "Bands: moving-block bootstrap of the residuals and the proxy together (Jentsch and Lunsford)."))

    def plot(self) -> Any:
        return self.irf.plot()

    @api()
    def historical_decomposition(self, draws: int = 0, alpha: float = 0.05, seed: int = 0) -> Any:
        """Each variable over the VAR's sample, split into the identified shock's contribution, the other shocks'
        contribution combined, and the baseline (the path from the first observations and the deterministic terms,
        with no shocks). The parts add up to the data exactly.

        The identified shock is ``e(t) = b' Sigma^-1 u(t) / (b' Sigma^-1 b)``, the part of the residuals along the
        impact column b, whatever its scale; its contribution propagates ``b e(t)`` through the VAR, the other shocks'
        propagates ``u(t) - b e(t)``. The other shocks are not identified one by one, so they are shown together.

        With ``draws``, ``1 - alpha`` bands: quantiles over the decompositions of the data under Jentsch and
        Lunsford's moving-block bootstrap draws (the residuals and the proxy resampled together, the VAR and the
        impact column estimated again), the draws the impulse-response bands use for the same ``seed``. Returns a
        ``Decomposition``."""
        from .historical_extra import proxy_historical

        return proxy_historical(self, draws, alpha, seed)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.irf.to_frame()

    def __repr__(self) -> str:
        return f"<ProxySVAR: {self.normalise_on} shock by {self.instrument}, first-stage F {self.first_stage[0]:.1f}>"

    def __setstate__(self, state: dict[str, Any]) -> None:  # results cached in projects saved before the rename
        if "target" in state:
            state["normalise_on"] = state.pop("target")
        self.__dict__.update(state)

    target = deprecated_alias("normalise_on", "target", why="the variable the shock moves by one; target is its old name")


def proxy_svar(var: FittedVAR, data: Dataset, instrument: str, normalise_on: str | None = None, periods: int = 20, scale: Literal["sd", "unit"] = "sd",
               draws: int = 500, block: int | None = None, alpha: float = 0.05, seed: int = 0) -> ProxySVAR:
    from .var import IRF, _ma

    variables = list(var.variables)
    target = normalise_on or variables[0]
    if target not in variables:
        raise SpecificationError(f"{target!r} is not one of the VAR's variables ({', '.join(variables)})")
    if scale not in ("sd", "unit"):
        raise SpecificationError("scale is 'sd' (a one-standard-deviation shock) or 'unit' (normalise_on moves by one)")
    if var.arrays.D.shape[1] != (1 if var.spec.trend == "c" else 2 if var.spec.trend == "ct" else 0) or var.spec.exog:
        raise SpecificationError("proxy SVARs here take VARs without exogenous variables")
    z_all = data.eval(instrument).reindex(var.arrays.index).to_numpy(float)
    ok = np.isfinite(z_all)
    if ok.sum() < 20:
        raise SpecificationError(f"{instrument} has {int(ok.sum())} periods within the VAR's sample: too few")
    i = variables.index(target)
    U = var.estimate.U
    sigma = var.estimate.sigma
    b = proxy_impact(U[ok], z_all[ok], i, sigma, scale)
    values = _ma(var.coefs, periods) @ b[:, None]
    lower = upper = None
    if draws:
        sims = [_ma(A, periods) @ bs[:, None] for A, _, _, bs in bootstrap_draws(var, z_all, i, scale, draws, block, seed)]
        if sims:
            lower, upper = np.quantile(np.stack(sims), [alpha / 2, 1 - alpha / 2], axis=0)
    shock = f"{target} shock ({instrument})"
    irf = IRF(values, tuple(variables), (shock,), lower, upper, False, f"proxy SVAR ({instrument})", 1 - alpha)
    return ProxySVAR(var, instrument, target, scale, pd.Series(b, index=variables, name="impact"), irf, first_stage(U[ok, i], z_all[ok]),
                     int(ok.sum()), z_all, block)


def bootstrap_draws(var: FittedVAR, z_all: np.ndarray, i: int, scale: str, draws: int, block: int | None, seed: int) -> Any:
    """Jentsch and Lunsford's moving-block bootstrap: for each draw, the residuals and the proxy resampled together in
    blocks, the data rebuilt from the VAR, the VAR and the proxy's impact estimated again. Yields (A, Bd, sigma, b):
    the lag matrices, the deterministic coefficients, the residual covariance and the impact column of each draw
    (draws whose proxy identifies nothing are skipped)."""
    from .progress import report
    from .var import _lag_matrices, _reestimate

    a = var.arrays
    T, K = a.Y.shape
    p, d = var.lags, a.D.shape[1]
    A = var.coefs
    Bd = var.estimate.B[:d]
    length = int(block or max(2, round(5.03 * T ** 0.25)))  # Jentsch and Lunsford's rule
    starts = np.arange(T - length + 1)
    U = var.estimate.U
    Uc = U - U.mean(axis=0)
    zc = np.where(np.isfinite(z_all), z_all, np.nan)
    rng = make_rng(seed)
    start_lags = a.L[0, : K * p].reshape(p, K)
    for k in range(draws):
        report(k / draws, f"bootstrap draw {k + 1} of {draws}")
        picks = np.concatenate([np.arange(s, s + length) for s in rng.choice(starts, int(np.ceil(T / length)))])[:T]
        Ustar, zstar = Uc[picks], zc[picks]
        history = list(start_lags[::-1])
        Y = np.empty((T, K))
        for t in range(T):
            y = a.D[t] @ Bd + Ustar[t]
            for l in range(1, p + 1):
                y = y + A[l - 1] @ history[-l]
            Y[t] = y
            history.append(y)
        hist = np.vstack([start_lags[::-1], Y])
        L = np.hstack([hist[p - l: p - l + T] for l in range(1, p + 1)])
        est = _reestimate(var, a, p, Y, L)
        keep = np.isfinite(zstar)
        try:
            bs = proxy_impact(est.U[keep], zstar[keep], i, est.sigma, scale)
        except SpecificationError:
            continue
        yield _lag_matrices(est.B, d, K, p), est.B[:d], est.sigma, bs
