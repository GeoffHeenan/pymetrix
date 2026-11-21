"""More ways to identify a VAR's shocks: the maximum forecast-error-variance share (Uhlig 2004; Francis, Owyang, Roush
and DiCecio 2014; Barsky and Sims 2011 for news shocks), narrative restrictions on sign-identified shocks
(Antolin-Diaz and Rubio-Ramirez 2018), and identification through heteroskedasticity (Rigobon 2003).

    var.identify(MaxShare("tfp", horizon=40))                         # the shock explaining most of tfp's FEV to 40
    var.identify(MaxShare("tfp", horizon=40, zero_impact=True))       # a news shock: no effect on tfp on impact
    var.identify(SignRestrictions({"monetary": {"ff": "+", "cpi": "-"}},
                                  narrative_signs={"monetary": {"1979Q4": "+"}},
                                  narrative_major={"monetary": {"1979Q4": "ff"}}))
    hsvar = var.heteroskedasticity_svar(["1984Q1"])                   # variance regimes split at the dates given

Max share: with P the Cholesky factor of Sigma and Phi_h the moving-average coefficients, the share of variable i's
forecast-error variance from h0 to H due to the shock P g (|g| = 1) is g' S g / sum_h e_i'Phi_h Sigma Phi_h'e_i with
S = sum_{h=h0}^{H} P'Phi_h'e_i e_i'Phi_h P, so g is S's leading eigenvector (with zero_impact, the leading eigenvector of
S restricted to g_1 = 0 with i ordered first, so that the shock does not move i on impact). The other shocks complete
an orthonormal basis and are not identified. The shock is signed to raise the target at horizon H.

Narrative restrictions keep only the sign-restriction rotations under which the structural shocks e_t = B^-1 u_t
have the given signs at the given dates, or under which a shock is the largest contributor |B_ij e_jt| to variable
i's surprise u_it at a date. Rotations are then resampled with weights proportional to the inverse of the probability
that random shocks would satisfy the restrictions, as Antolin-Diaz and Rubio-Ramirez's posterior requires. They need
the residuals, so bands come from the identified set at the estimates.

Heteroskedasticity: with the residual covariance Sigma_1 before the first break and Sigma_r in regime r, Sigma_1 = B B'
and Sigma_r = B Lambda_r B' with Lambda_r diagonal; with two regimes B comes from the eigendecomposition of
Sigma_1^-1/2 Sigma_2 Sigma_1^-1/2 and is identified if the variance ratios lambda differ. Shocks are ordered by
decreasing ratio; each has unit standard deviation in the first regime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ._api import api, renamed_arguments
from .errors import SpecificationError
from .output import Table
from ._random import generator as make_rng

__all__ = ["HeteroskedasticitySVAR", "MaxShare", "heteroskedasticity_svar", "max_share_impact"]


@dataclass(frozen=True)
class MaxShare:
    """One shock: the one that explains the largest share of ``variable``'s forecast-error variance between ``start``
    and ``horizon`` periods ahead (``zero_impact``: none of it on impact, as for news shocks)."""

    code_positional: ClassVar[int] = 1
    variable: str = ""  # was target
    horizon: int = 40
    start: int = 0
    zero_impact: bool = False

    def __post_init__(self) -> None:
        if not self.variable:
            raise SpecificationError("name the variable: MaxShare('tfp')")
        if self.horizon < self.start or self.start < 0:
            raise SpecificationError("0 <= start <= horizon")

    def shock_names(self, variables: Sequence[str]) -> list[str]:
        return [f"max share: {self.variable}"] + [f"unidentified {i}" for i in range(1, len(variables))]


MaxShare.__init__ = renamed_arguments(target="variable")(MaxShare.__init__)  # type: ignore[method-assign]


def _ma(A: np.ndarray, horizons: int) -> np.ndarray:
    from .var import _ma as ma

    return ma(A, horizons)


def max_share_impact(scheme: MaxShare, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str]) -> np.ndarray:
    """The impact matrix: the max-share shock first, then an orthonormal completion."""
    variables = list(variables)
    if scheme.variable not in variables:
        raise SpecificationError(f"{scheme.variable!r} is not a variable of the VAR ({', '.join(variables)})")
    K = len(variables)
    i = variables.index(scheme.variable)
    order = [i] + [j for j in range(K) if j != i]  # the target first: Cholesky's first column is its own surprise
    perm = np.eye(K)[order]
    P = perm.T @ np.linalg.cholesky(perm @ sigma @ perm.T)  # rows in the VAR's order, columns ordered target first
    phi = _ma(A, scheme.horizon)
    rows = phi[scheme.start:, i, :] @ P  # (horizons, K): e_i'Phi_h P
    S = rows.T @ rows
    if scheme.zero_impact:
        Sr = S[1:, 1:]  # g_1 = 0: no Cholesky-first (impact) component
        w, V = np.linalg.eigh(Sr)
        g = np.r_[0.0, V[:, -1]]
    else:
        w, V = np.linalg.eigh(S)
        g = V[:, -1]
    if (phi[scheme.horizon, i, :] @ P @ g) < 0:
        g = -g
    Q = np.linalg.qr(np.column_stack([g, np.eye(K)]))[0][:, :K]
    if Q[:, 0] @ g < 0:
        Q[:, 0] = -Q[:, 0]
    return P @ Q


# narrative restrictions


def narrative_check(scheme: Any, resid: pd.DataFrame, variables: Sequence[str]) -> Any:
    """A function of an impact matrix B (columns: the named shocks first): whether the narrative restrictions hold,
    and the probability that random shocks would satisfy them (for the importance weights)."""
    names = list(getattr(scheme, "named", None) or scheme.signs)
    index = {v: i for i, v in enumerate(variables)}
    periods = resid.index
    freq = getattr(periods, "freq", None)

    def where(date: Any) -> int:
        try:
            return int(periods.get_loc(pd.Period(str(date), freq=freq) if freq is not None else date))
        except KeyError:
            raise SpecificationError(f"{date} is not in the VAR's estimation sample") from None

    signs = [(names.index(k), where(d), {"+": 1, "-": -1}[str(s).strip()]) for k, dates in (scheme.narrative_signs or {}).items()
             for d, s in dates.items()]
    major = [(names.index(k), where(d), index[v]) for k, dates in (scheme.narrative_major or {}).items() for d, v in dates.items()]
    U = resid.to_numpy(float)
    rng = make_rng(12345)
    draws = 2000

    def check(B: np.ndarray) -> tuple[bool, float]:
        E = np.linalg.solve(B, U.T).T  # structural shocks by period
        for j, t, s in signs:
            if s * E[t, j] <= 0:
                return False, 0.0
        for j, t, i in major:
            contrib = np.abs(B[i] * E[t])
            if np.argmax(contrib) != j:
                return False, 0.0
        if not major:  # with sign restrictions only the probability is 0.5^n whatever B: equal weights
            return True, 1.0
        Z = rng.standard_normal((draws, B.shape[1]))
        ok = np.ones(draws, bool)
        for j, _, s in signs:
            ok &= s * Z[:, j] > 0
        for j, _, i in major:
            ok &= np.argmax(np.abs(B[i] * Z), axis=1) == j
        return True, max(float(ok.mean()), 1.0 / draws)

    for k in list(scheme.narrative_signs or {}) + list(scheme.narrative_major or {}):
        if k not in names:
            raise SpecificationError(f"the narrative restriction names {k!r}, which is not one of the sign-restricted shocks")
    for dates in (scheme.narrative_major or {}).values():
        for v in dates.values():
            if v not in index:
                raise SpecificationError(f"{v!r} is not a variable of the VAR")
    return check


def apply_narrative(draws: list[np.ndarray], check: Any, rng: np.random.Generator) -> list[np.ndarray]:
    """The rotations satisfying the narrative, resampled with weights 1 / P(the narrative holds by chance)."""
    kept, weights = [], []
    for B in draws:
        ok, p = check(B)
        if ok:
            kept.append(B)
            weights.append(1.0 / p)
    if not kept:
        raise SpecificationError("no accepted rotation satisfies the narrative restrictions: draw more rotations or relax them")
    w = np.array(weights) / np.sum(weights)
    pick = rng.choice(len(kept), size=len(kept), replace=True, p=w)
    return [kept[i] for i in pick]


# heteroskedasticity


@dataclass(frozen=True, eq=False)
class HeteroskedasticitySVAR:
    """A VAR identified through shifts in its shocks' variances (Rigobon 2003)."""

    variables: tuple[str, ...]
    breaks: tuple[str, ...]
    impact: np.ndarray  # B: unit-variance shocks in the first regime
    ratios: np.ndarray = field(repr=False)  # (regimes - 1, K): each shock's variance relative to the first regime
    regime_sigmas: tuple[np.ndarray, ...] = field(repr=False)
    regime_nobs: tuple[int, ...] = ()
    coefs: np.ndarray = field(default=None, repr=False)

    stage = "estimate"

    @property
    def shocks(self) -> list[str]:
        return [f"shock {j + 1}" for j in range(len(self.variables))]

    @api()
    def irf(self, periods: int = 20, cumulative: bool = False) -> Any:
        from .var import IRF, _ma

        values = _ma(self.coefs, periods) @ self.impact
        if cumulative:
            values = np.cumsum(values, axis=0)
        return IRF(values, self.variables, self.shocks, None, None, cumulative, "heteroskedasticity (Rigobon)")

    @api()
    def fevd(self, periods: int = 20) -> Any:
        """Forecast-error variance decomposition in the first variance regime, where each shock has unit variance."""
        from .var import FEVD, _ma

        theta = _ma(self.coefs, periods - 1) @ self.impact
        contributions = np.cumsum(theta**2, axis=0)
        return FEVD(contributions / contributions.sum(axis=2, keepdims=True), self.variables, self.shocks,
                    "heteroskedasticity (Rigobon), first variance regime")

    def summary(self) -> Table:
        rows = [[v, *map(float, self.impact[i])] for i, v in enumerate(self.variables)]
        rows += [[f"variance ratio, regime {r + 2}", *map(float, self.ratios[r])] for r in range(len(self.ratios))]
        gap = np.min(np.abs(np.diff(np.sort(self.ratios[0])))) if len(self.variables) > 1 else np.nan
        notes = (f"Regimes split at {', '.join(self.breaks)} ({', '.join(map(str, self.regime_nobs))} observations); the impact "
                 "matrix gives one-standard-deviation shocks of the first regime.",
                 f"Identified if the variance ratios differ: the closest two differ by {gap:.4f}.")
        return Table(["", *self.shocks], rows, title="SVAR identified through heteroskedasticity", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame(self.impact, index=list(self.variables), columns=self.shocks)

    def __repr__(self) -> str:
        return f"<HeteroskedasticitySVAR {', '.join(self.variables)}: breaks {', '.join(self.breaks)}>"


def heteroskedasticity_svar(var: Any, breaks: Sequence[Any]) -> HeteroskedasticitySVAR:
    """Identify a fitted VAR's shocks by the change in their variances at ``breaks`` (the first period of each new
    regime): exactly with two regimes, by Gaussian maximum likelihood with more."""
    from scipy import linalg, optimize

    resid = var.resid.frame[list(var.variables)]
    idx = resid.index
    freq = getattr(idx, "freq", None)
    cuts = sorted(idx.get_indexer([pd.Period(str(b), freq=freq) for b in breaks]))
    if not breaks or min(cuts) < 0:
        raise SpecificationError("give the break dates, each in the VAR's estimation sample")
    edges = [0, *cuts, len(idx)]
    U = resid.to_numpy(float)
    K = U.shape[1]
    parts = [U[a:b] for a, b in zip(edges[:-1], edges[1:])]
    if min(len(p) for p in parts) <= 2 * K:
        raise SpecificationError("each variance regime needs more than twice as many observations as variables")
    sigmas = tuple(p.T @ p / len(p) for p in parts)
    if len(parts) == 2:
        w, V = linalg.eigh(sigmas[1], sigmas[0])  # Sigma_2 v = lambda Sigma_1 v, V' Sigma_1 V = I
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        B = np.linalg.inv(V.T)  # Sigma_1 = B B', Sigma_2 = B diag(w) B'
        ratios = w[None, :]
    else:  # more regimes: maximise sum_r -T_r/2 (log|B Lambda_r B'| + tr((B Lambda_r B')^-1 Sigma_r))
        w, V = linalg.eigh(sigmas[1], sigmas[0])
        B0 = np.linalg.inv(V.T)
        R = len(parts)

        def unpack(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return th[:K * K].reshape(K, K), np.exp(th[K * K:].reshape(R - 1, K))

        def nll(th: np.ndarray) -> float:
            Bm, L = unpack(th)
            total = 0.0
            for r, (p, S) in enumerate(zip(parts, sigmas)):
                lam = np.ones(K) if r == 0 else L[r - 1]
                O = (Bm * lam) @ Bm.T
                sign, ld = np.linalg.slogdet(O)
                if sign <= 0:
                    return 1e300
                total += len(p) / 2 * (ld + np.trace(np.linalg.solve(O, S)))
            return float(total)

        start = np.r_[B0.ravel(), np.log(np.vstack([np.diag(np.linalg.solve(B0, S) @ np.linalg.inv(B0).T) for S in sigmas[1:]])).ravel()]
        th = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-9, "maxiter": 20000}).x
        B, ratios = unpack(th)
        order = np.argsort(ratios[0])[::-1]
        B, ratios = B[:, order], ratios[:, order]
    for j in range(K):  # signed so that each shock's largest impact is positive
        if B[np.argmax(np.abs(B[:, j])), j] < 0:
            B[:, j] = -B[:, j]
    return HeteroskedasticitySVAR(tuple(var.variables), tuple(str(b) for b in breaks), B, ratios, sigmas,
                                  tuple(len(p) for p in parts), var.coefs)
