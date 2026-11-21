"""Panel regressions with interactive fixed effects (Bai 2009).

    ife = interactive_fe(panel, "y", ["x1", "x2"], factors=2)              # Y_it = X_it' beta + lambda_i' F_t + e_it
    interactive_fe(panel, "y", ["x1", "x2"], factors="auto")               # r by Bai and Ng's criteria
    interactive_fe(panel, "y", ["x1"], factors=1, effects="twoway", bias_correction=True)

Unobserved common factors F_t (T x r) with unit-specific loadings lambda_i may be correlated with the regressors;
two-way fixed effects remove only additive effects (r = 0 after the two-way transform), so they are biased then. The
estimator iterates between beta = (sum_i X_i' M_F X_i)^-1 sum_i X_i' M_F Y_i and the factors, sqrt(T) times the r
leading eigenvectors of the residual panel's T x T second moment (F'F/T = I), from two starting values (pooled least
squares, and the principal components of Y), keeping the smaller sum of squares. ``effects`` removes additive unit,
time or both effects first (the balanced-panel within transform, which keeps the factor structure's rank); "none"
estimates a constant with the other coefficients.

Standard errors (Bai 2009, Theorem 3): with Z_i = M_F X_i - (1/N) sum_k a_ik M_F X_k, a_ik = lambda_i' (L'L/N)^-1
lambda_k and D = (1/NT) sum_i Z_i' Z_i, "homoskedastic" sigma^2 D^-1 / NT; "robust" D^-1 (1/NT sum_it e_it^2 z_it z_it')
D^-1 / NT; "cluster" sums z_it e_it over t within each unit first (serial correlation). sigma^2 is SSR over
NT - (N + T) r - p - the additive effects' parameters. ``bias_correction`` subtracts B/N + C/T for errors
heteroskedastic across units (B) and over time (C) (Bai 2009, section 7):

    B = -D^-1 (1/N) sum_i (X_i - V_i)' F/T (L'L/N)^-1 lambda_i sigma_i^2,     V_i = (1/N) sum_k a_ik X_k,
    C = -D^-1 (1/NT) sum_i X_i' M_F Omega F (L'L/N)^-1 lambda_i,              Omega = diag(sigma_t^2).

Number of factors: Bai and Ng's (2002) ICp1-3 and PCp1-3 for k = 0 .. max_factors, from the residual Y - X beta of the
fit with max_factors factors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg

from . import _micro
from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedInteractiveFE", "bai_ng_criteria", "interactive_fe"]

CRITERIA = ("icp1", "icp2", "icp3", "pcp1", "pcp2", "pcp3")


def _within(A: np.ndarray, effects: str) -> np.ndarray:
    """The balanced-panel transform of an N x T (x p) array removing the additive effects."""
    if effects == "none":
        return A
    if effects == "unit":
        return A - A.mean(axis=1, keepdims=True)
    if effects == "time":
        return A - A.mean(axis=0, keepdims=True)
    return A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean(axis=(0, 1), keepdims=True)


def _factors(W: np.ndarray, r: int) -> np.ndarray:
    """sqrt(T) times the r leading eigenvectors of W'W (T x T), signed so that each sums to a non-negative number."""
    T = W.shape[1]
    if r == 0:
        return np.zeros((T, 0))
    vals, vecs = linalg.eigh(W.T @ W)
    F = np.sqrt(T) * vecs[:, ::-1][:, :r]
    signs = np.sign(F.sum(axis=0))
    return F * np.where(signs == 0, 1, signs)


def _annihilate(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """M_F X_i for each unit, M_F = I - F F' / T."""
    T = X.shape[1]
    if F.shape[1] == 0:
        return X
    return X - np.einsum("tr,irp->itp", F, np.einsum("tr,itp->irp", F, X, optimize=True), optimize=True) / T


def _beta(Y: np.ndarray, X: np.ndarray, F: np.ndarray) -> np.ndarray:
    MX = _annihilate(X, F)
    A = np.einsum("itp,itq->pq", MX, X)
    b = np.einsum("itp,it->p", MX, Y)
    return np.linalg.solve(A, b)


def _iterate(Y: np.ndarray, X: np.ndarray, r: int, beta: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    for it in range(1, max_iter + 1):
        F = _factors(Y - np.einsum("itp,p->it", X, beta), r)
        new = _beta(Y, X, F)
        if np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(beta))):
            return new, _factors(Y - np.einsum("itp,p->it", X, new), r), it, True
        beta = new
    return beta, _factors(Y - np.einsum("itp,p->it", X, beta), r), max_iter, False


def bai_ng_criteria(W: np.ndarray, max_factors: int) -> pd.DataFrame:
    """Bai and Ng's (2002) ICp1-3 and PCp1-3 for 0 .. ``max_factors`` factors of the N x T panel ``W``; V(k) is the mean
    squared residual after k principal components. The smallest value of each criterion picks the number."""
    N, T = W.shape
    s2 = np.linalg.svd(W, compute_uv=False) ** 2
    total = float((W**2).sum())
    V = np.array([(total - s2[:k].sum()) / (N * T) for k in range(max_factors + 1)])
    c2 = min(N, T)
    p1, p2, p3 = (N + T) / (N * T) * np.log(N * T / (N + T)), (N + T) / (N * T) * np.log(c2), np.log(c2) / c2
    k = np.arange(max_factors + 1)
    sig = V[-1]
    frame = pd.DataFrame({"V": V, "icp1": np.log(V) + k * p1, "icp2": np.log(V) + k * p2, "icp3": np.log(V) + k * p3,
                          "pcp1": V + k * sig * p1, "pcp2": V + k * sig * p2, "pcp3": V + k * sig * p3}, index=pd.Index(k, name="factors"))
    return frame


@dataclass(frozen=True, eq=False)
class FittedInteractiveFE:
    """A panel regression with interactive fixed effects: beta, the estimated factors and loadings."""

    outcome: str
    regressors: tuple[str, ...]
    factors: int
    effects: str
    cov_type: str
    beta: np.ndarray
    cov: np.ndarray = field(repr=False)
    beta_uncorrected: np.ndarray = field(repr=False)
    F: pd.DataFrame = field(repr=False)  # T x r factors, by period
    loadings: pd.DataFrame = field(repr=False)  # N x r, by unit
    resid: pd.Series = field(repr=False)  # e_it on the panel's index
    fitted: pd.Series = field(repr=False)
    criteria: pd.DataFrame | None = field(default=None, repr=False)  # Bai-Ng criteria, when the number of factors was chosen
    criterion: str | None = None
    bias_correction: bool = False
    sigma2: float = 0.0
    units: int = 0
    periods: int = 0
    iterations: int = 0
    converged: bool = True
    provenance: Any = field(default=None, repr=False)

    stage = "estimate"

    @property
    def params(self) -> Any:
        from .estimation.params import PointEstimate

        return PointEstimate(self.regressors, np.asarray(self.beta, float), np.asarray(self.cov, float))

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def stats(self) -> dict[str, float]:
        return {"nobs": float(self.units * self.periods), "units": float(self.units), "periods": float(self.periods), "factors": float(self.factors),
                "sigma2": self.sigma2, "ssr": float((self.resid**2).sum()), "iterations": float(self.iterations)}

    def summary(self) -> Table:
        effects = {"none": "a constant", "unit": "unit effects", "time": "period effects", "twoway": "unit and period effects"}[self.effects]
        notes = [f"Bai (2009) interactive fixed effects with {self.factors} factor(s) and {effects}; {self.units} units, {self.periods} periods.",
                 {"homoskedastic": "Homoskedastic standard errors.", "robust": "Standard errors robust to heteroskedasticity.",
                  "cluster": "Standard errors clustered by unit (serial correlation within units)."}[self.cov_type]]
        if self.criterion:
            notes.append(f"Number of factors chosen by Bai and Ng's {self.criterion.upper()}.")
        if self.bias_correction:
            notes.append("Bias-corrected for heteroskedasticity across units and over time (Bai 2009, section 7).")
        if not self.converged:
            notes.append(f"The iterations did not converge in {self.iterations} steps.")
        return _micro.coefficient_table(self.regressors, self.beta, self.se, f"Interactive fixed effects: {self.outcome}", notes)

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        if self.factors == 0:
            return Figure((Panel((Line(self.resid.groupby(level=-1).mean(), "mean residual"),), title="No factors: the mean residual by period"),))
        lines = tuple(Line(self.F[c], c) for c in self.F.columns)
        return Figure((Panel(lines, title=f"Estimated factors ({self.outcome})"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedInteractiveFE {self.outcome}, {self.factors} factor(s): {dict(zip(self.regressors, np.round(self.beta, 4)))}>"


@api("pymetrix:interactive_fe")
def interactive_fe(data: Dataset, outcome: str, regressors: Sequence[str], factors: int | Literal["auto"] = 1,
                   effects: Literal["none", "unit", "time", "twoway"] = "none", cov: Literal["homoskedastic", "robust", "cluster"] = "robust",
                   bias_correction: bool = False, max_factors: int = 8, criterion: str = "icp1", tol: float = 1e-9,
                   max_iter: int = 10_000) -> FittedInteractiveFE:
    """Bai's (2009) interactive fixed effects: ``outcome`` on ``regressors`` with ``factors`` unobserved common factors
    with unit-specific loadings (``"auto"``: chosen by Bai and Ng's ``criterion`` up to ``max_factors``), after
    removing additive ``effects``; a balanced panel."""
    from ._capture import capture

    regressors = _micro.names(regressors)
    if not regressors:
        raise SpecificationError("interactive fixed effects need at least one regressor")
    if data.kind != "panel" or data.time_axis is None or not data.entity_axes:
        raise SpecificationError("interactive fixed effects need a panel of units and periods")
    if effects not in ("none", "unit", "time", "twoway"):
        raise SpecificationError(f"effects is 'none', 'unit', 'time' or 'twoway', not {effects!r}")
    if cov not in ("homoskedastic", "robust", "cluster"):
        raise SpecificationError(f"cov is 'homoskedastic', 'robust' or 'cluster', not {cov!r}")
    criterion = str(criterion).lower()
    if criterion not in CRITERIA:
        raise SpecificationError(f"criterion is one of {', '.join(CRITERIA)}, not {criterion!r}")
    unit, time = data.entity_axes[0].name, data.time_axis.name
    frame = data.frame[[outcome, *regressors]] if all(n in data.frame.columns for n in [outcome, *regressors]) else None
    if frame is None:
        missing = [n for n in [outcome, *regressors] if n not in data.frame.columns]
        raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
    frame = frame.astype(float)
    wide = frame.unstack(time)
    if wide.isna().to_numpy().any():
        raise SpecificationError("interactive fixed effects need a balanced panel with no missing values")
    units, periods = wide.index, wide[outcome].columns
    N, T = len(units), len(periods)
    Y = _within(wide[outcome].to_numpy(), effects)
    X = _within(np.stack([wide[x].to_numpy() for x in regressors], axis=2), effects)
    names = list(regressors)
    if effects == "none":  # a constant, estimated with the other coefficients (demeaning would add a factor)
        X = np.concatenate([np.ones((N, T, 1)), X], axis=2)
        names = ["const", *names]
    p = len(names)
    if factors == "auto":
        if not 1 <= max_factors < min(N, T):
            raise SpecificationError(f"max_factors must be between 1 and {min(N, T) - 1}")
    elif not isinstance(factors, (int, np.integer)) or factors < 0 or factors >= min(N, T):
        raise SpecificationError(f"factors is 'auto' or a whole number from 0 to {min(N, T) - 1}")
    with capture() as caught:
        def fit(r: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
            b0 = _beta(Y, X, np.zeros((T, 0)))
            if r == 0:
                return b0, np.zeros((T, 0)), 0, True
            b1 = _beta(Y, X, _factors(Y, r))
            best = None
            for start in (b0, b1):
                b, F, it, ok = _iterate(Y, X, r, start, tol, max_iter)
                W = Y - np.einsum("itp,p->it", X, b)
                ssr = float(((W - W @ F @ F.T / T) ** 2).sum())
                if best is None or ssr < best[0] - 1e-10 * (1 + abs(ssr)):
                    best = (ssr, b, F, it, ok)
            assert best is not None
            return best[1], best[2], best[3], best[4]

        table, chosen = None, None
        if factors == "auto":
            b_max, _, _, _ = fit(max_factors)
            table = bai_ng_criteria(Y - np.einsum("itp,p->it", X, b_max), max_factors)
            r = int(table[criterion].idxmin())
            chosen = criterion
        else:
            r = int(factors)
        beta, F, iterations, converged = fit(r)
        W = Y - np.einsum("itp,p->it", X, beta)
        L = W @ F / T  # N x r loadings
        E = W - L @ F.T
        effects_df = {"none": 0, "unit": N, "time": T, "twoway": N + T - 1}[effects]
        df = N * T - (N + T) * r - p - effects_df
        if df <= 0:
            raise SpecificationError(f"{N} units and {T} periods: too few for {r} factors and {p} regressors")
        sigma2 = float((E**2).sum() / df)
        MX = _annihilate(X, F)
        if r:
            G = np.linalg.pinv(L.T @ L / N)
            A = L @ G @ L.T  # a_ik
            Z = MX - np.einsum("ik,ktp->itp", A, MX, optimize=True) / N
        else:
            A = np.zeros((N, N))
            Z = MX
        D0 = np.einsum("itp,itq->pq", Z, Z) / (N * T)
        Dinv = np.linalg.inv(D0)
        if cov == "homoskedastic":
            V = sigma2 * Dinv / (N * T)
        else:
            scale = (N * T) / df
            if cov == "robust":
                meat = np.einsum("it,itp,itq->pq", E**2, Z, Z, optimize=True) / (N * T)
            else:
                s = np.einsum("itp,it->ip", Z, E)
                meat = s.T @ s / (N * T)
            V = Dinv @ meat @ Dinv / (N * T) * scale
        beta_raw = beta.copy()
        if bias_correction and r:
            sig_i = (E**2).mean(axis=1)
            sig_t = (E**2).mean(axis=0)
            Vi = np.einsum("ik,ktp->itp", A, X, optimize=True) / N
            Bterm = np.einsum("itp,tr,rs,is,i->p", X - Vi, F / T, G, L, sig_i, optimize=True) / N
            Cterm = np.einsum("itp,t,tr,rs,is->p", MX, sig_t, F, G, L, optimize=True) / (N * T)
            B, C = -Dinv @ Bterm, -Dinv @ Cterm
            beta = beta - B / N - C / T
    raw_y = wide[outcome].to_numpy()
    idx = pd.MultiIndex.from_product([units, periods], names=[unit, time])
    resid = pd.Series(E.ravel(), index=idx, name="resid")
    fitted_s = pd.Series((raw_y - E).ravel(), index=idx, name="fitted")  # actual = fitted + resid
    Fframe = pd.DataFrame(F, index=periods, columns=[f"factor {j + 1}" for j in range(r)])
    Lframe = pd.DataFrame(L, index=units, columns=[f"factor {j + 1}" for j in range(r)])
    spec = {"outcome": outcome, "regressors": regressors, "factors": factors, "effects": effects, "cov": cov, "bias_correction": bias_correction,
            "max_factors": max_factors, "criterion": criterion}
    used = frame.reorder_levels([unit, time]) if list(frame.index.names) != [unit, time] else frame
    prov = _micro.provenance(spec, data, used, [outcome, *regressors], caught,
                             {"type": "interactive fixed effects", "estimation": "Bai (2009) iterated least squares and principal components"},
                             converged=converged, resolved={"factors": r} if factors == "auto" else None)
    return FittedInteractiveFE(outcome, tuple(names), r, effects, cov, beta, V, beta_raw, Fframe, Lframe, resid, fitted_s, table, chosen,
                               bool(bias_correction and r), sigma2, N, T, iterations, converged, prov)
