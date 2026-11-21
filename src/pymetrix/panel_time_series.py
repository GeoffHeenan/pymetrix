"""Heterogeneous panels with long time series: mean group (Pesaran and Smith 1995), common correlated effects (Pesaran
2006) and pooled mean group (Pesaran, Shin and Smith 1999).

    mean_group(data, "y", ["x"])                       # each unit's least squares, averaged
    cce(data, "y", ["x"])                              # with the cross-section averages of y and x: CCE mean group
    cce(data, "y", ["x"], pooled=True)                 # CCE pooled
    pmg = pooled_mean_group(data, "y", ["x"], lags=1)  # a common long run, unit-specific adjustment and short run
    pmg.summary(); pmg.hausman()                       # against the mean group of the units' own long runs

Mean-group standard errors are nonparametric: the spread of the unit estimates over sqrt(N). CCE augments each unit's
regression with the cross-section averages of the dependent variable and the regressors, which absorb unobserved
common factors whatever their loadings; CCE pooled weights the units' within moments together, its variance by
Pesaran's (2006) nonparametric formula. The pooled mean group estimates the error-correction ARDL

    Delta y_it = phi_i (y_i,t-1 - x_i,t-1' theta) + sum_j lambda_ij Delta y_i,t-j + sum_j delta_ij' Delta x_i,t-j + mu_i + e_it

by maximum likelihood, theta common, the rest by unit: for given theta each unit is least squares, so theta maximises
the concentrated likelihood -sum_i T_i / 2 log sigma_i^2(theta); its standard errors are from that likelihood's Hessian.
The adjustment phi_i and the short run are averaged across units, mean-group style. A Hausman test compares theta with
the mean group of the units' own long runs (-beta_i / phi_i from each unit's ARDL).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["PanelMeanGroup", "PooledMeanGroup", "cce", "mean_group", "pooled_mean_group"]


def _units(data: Dataset, names: Sequence[str]) -> list[tuple[Any, pd.DataFrame]]:
    if data.kind != "panel" or data.time_axis is None or not data.entity_axes:
        raise SpecificationError("these estimators need a panel of units observed over time")
    missing = [n for n in names if n not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    unit = data.entity_axes[0].name
    frame = data.frame[list(names)].astype(float)
    return [(u, g.droplevel(unit)) for u, g in frame.groupby(level=unit, sort=True)]


@dataclass(frozen=True, eq=False)
class PanelMeanGroup:
    """A mean-group or CCE estimate: average coefficients with their standard errors, and each unit's coefficients."""

    method: str
    y_name: str
    names: tuple[str, ...]
    params: np.ndarray
    cov: np.ndarray = field(repr=False)
    units: pd.DataFrame = field(repr=False)  # each unit's coefficients (mean group and CCEMG)
    nobs: int = 0

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.params / self.se
        rows = [[float(b), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for b, s, t in zip(self.params, self.se, z)]
        label = {"mg": "Mean group (Pesaran-Smith)", "ccemg": "Common correlated effects, mean group (Pesaran)",
                 "ccep": "Common correlated effects, pooled (Pesaran)"}[self.method]
        notes = (f"{label}; {len(self.units)} units, {self.nobs} observations; nonparametric standard errors (the spread across units).",)
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"{label}: {self.y_name}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<PanelMeanGroup {self.method} {self.y_name}: {dict(zip(self.names, np.round(self.params, 4)))}>"


def _mg_summary(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = len(B)
    mean = B.mean(axis=0)
    if N < 2:
        return mean, np.full((B.shape[1], B.shape[1]), np.nan)
    D = B - mean
    return mean, D.T @ D / (N * (N - 1))


@api("pymetrix:mean_group")
def mean_group(data: Dataset, y: str, x: Sequence[str], trend: bool = False) -> PanelMeanGroup:
    """Pesaran and Smith's mean group: least squares of ``y`` on ``x`` (a constant, and a trend with ``trend``) unit by
    unit, the coefficients averaged."""
    x = [x] if isinstance(x, str) else list(x)
    rows, index, n = [], [], 0
    for u, g in _units(data, [y, *x]):
        g = g.dropna()
        X = np.column_stack([np.ones(len(g))] + ([np.arange(len(g), dtype=float)] if trend else []) + [g[x].to_numpy()])
        if len(g) <= X.shape[1]:
            raise SpecificationError(f"unit {u} has too few observations")
        rows.append(np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0])
        index.append(u)
        n += len(g)
    names = ("const", *(["trend"] if trend else []), *x)
    B = np.array(rows)
    mean, cov = _mg_summary(B)
    return PanelMeanGroup("mg", y, names, mean, cov, pd.DataFrame(B, index=index, columns=list(names)), n)


@api("pymetrix:cce")
def cce(data: Dataset, y: str, x: Sequence[str], pooled: bool = False) -> PanelMeanGroup:
    """Pesaran's common correlated effects: each unit's regression with the cross-section averages of ``y`` and ``x``
    added; mean group (default) or pooled. The averages' coefficients are not reported."""
    x = [x] if isinstance(x, str) else list(x)
    unit = data.entity_axes[0].name if data.entity_axes else None
    groups = _units(data, [y, *x])
    frame = data.frame[[y, *x]].astype(float)
    if frame.isna().to_numpy().any():
        raise SpecificationError("CCE here needs a balanced panel with no missing values")
    averages = frame.groupby(level=data.time_axis.name).mean()
    H = np.column_stack([np.ones(len(averages)), averages.to_numpy()])  # the constant and the cross-section averages
    k = len(x)
    if not pooled:
        rows, index, n = [], [], 0
        for u, g in groups:
            Hu = H[averages.index.get_indexer(g.index)]
            X = np.column_stack([g[x].to_numpy(), Hu])
            rows.append(np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0][:k])
            index.append(u)
            n += len(g)
        B = np.array(rows)
        mean, cov = _mg_summary(B)
        return PanelMeanGroup("ccemg", y, tuple(x), mean, cov, pd.DataFrame(B, index=index, columns=x), n)
    # pooled: sum of X_i' M X_i and X_i' M y_i, M annihilating the averages
    XMX, XMy, parts, n = np.zeros((k, k)), np.zeros(k), [], 0
    for u, g in groups:
        Hu = H[averages.index.get_indexer(g.index)]
        M = np.eye(len(Hu)) - Hu @ np.linalg.pinv(Hu)
        Xi, yi = g[x].to_numpy(), g[y].to_numpy()
        XMX += Xi.T @ M @ Xi
        XMy += Xi.T @ M @ yi
        parts.append((Xi, yi, M))
        n += len(g)
    b = np.linalg.solve(XMX, XMy)
    N = len(parts)
    psi = XMX / N
    # Pesaran (2006, eq. 69): R = sum (X_i'MX_i / T)(b_i - b_MG)(b_i - b_MG)'(X_i'MX_i / T) / (N - 1)
    bi = np.array([np.linalg.lstsq(M @ Xi, M @ yi, rcond=None)[0] for Xi, yi, M in parts])
    bmg = bi.mean(axis=0)
    R = np.zeros((k, k))
    for (Xi, yi, M), bu in zip(parts, bi):
        A = Xi.T @ M @ Xi
        d = (bu - bmg)[:, None]
        R += A @ d @ d.T @ A
    R /= N - 1
    psi_inv = np.linalg.inv(psi)
    cov = psi_inv @ R @ psi_inv / N  # the T's of Psi and R cancel
    return PanelMeanGroup("ccep", y, tuple(x), b, cov, pd.DataFrame(bi, index=[u for u, _ in groups], columns=x), n)


# pooled mean group


@dataclass(frozen=True, eq=False)
class _Unit:
    u: Any
    dy: np.ndarray
    ylag: np.ndarray
    xlag: np.ndarray
    W: np.ndarray  # constant and the short-run terms


def _pmg_units(data: Dataset, y: str, x: list[str], lags: int) -> list[_Unit]:
    out = []
    for u, g in _units(data, [y, *x]):
        g = g.dropna()
        Y, X = g[y].to_numpy(), g[x].to_numpy()
        dY, dX = np.diff(Y), np.diff(X, axis=0)
        T = len(dY)
        start = lags  # Delta y_t for t = lags, ..., T - 1 (in the differenced index), so that lagged differences exist
        if T - start <= 3 + len(x) * (lags + 1) + lags:
            raise SpecificationError(f"unit {u} has too few observations for {lags} lags")
        W = [np.ones(T - start)]
        W += [dY[start - j:T - j] for j in range(1, lags)]  # lagged differences of y, 1..lags-1
        W += [dX[start - j:T - j] for j in range(0, lags)]  # current and lagged differences of x, 0..lags-1
        out.append(_Unit(u, dY[start:], Y[start:T], X[start:T], np.column_stack(W)))
    return out


def _concentrated(theta: np.ndarray, units: list[_Unit]) -> tuple[float, list[np.ndarray]]:
    ll, coefs = 0.0, []
    for d in units:
        ec = d.ylag - d.xlag @ theta
        Z = np.column_stack([ec, d.W])
        b = np.linalg.lstsq(Z, d.dy, rcond=None)[0]
        e = d.dy - Z @ b
        T = len(e)
        ll += -T / 2 * (np.log(2 * np.pi) + np.log(e @ e / T) + 1)
        coefs.append(b)
    return ll, coefs


@dataclass(frozen=True, eq=False)
class PooledMeanGroup:
    """The pooled mean group: a common long run, unit-specific adjustment and short run."""

    y_name: str
    x_names: tuple[str, ...]
    theta: np.ndarray
    theta_cov: np.ndarray = field(repr=False)
    phi: pd.Series = field(repr=False)  # each unit's adjustment
    short_run: pd.DataFrame = field(repr=False)  # each unit's constant and short-run coefficients
    mg_theta: np.ndarray = field(repr=False)  # the mean group of the units' own long runs
    mg_theta_cov: np.ndarray = field(repr=False)
    loglike: float = 0.0
    lags: int = 1
    nobs: int = 0

    stage = "estimate"

    def summary(self) -> Table:
        se = np.sqrt(np.diag(self.theta_cov))
        rows = [[f"long run: {n}", float(b), float(s), float(b / s)] for n, b, s in zip(self.x_names, self.theta, se)]
        mean_phi = float(self.phi.mean())
        se_phi = float(self.phi.std(ddof=1) / np.sqrt(len(self.phi)))
        rows.append(["adjustment (mean of phi_i)", mean_phi, se_phi, mean_phi / se_phi])
        for c in self.short_run.columns:
            col = self.short_run[c]
            m, s = float(col.mean()), float(col.std(ddof=1) / np.sqrt(len(col)))
            rows.append([f"short run (mean): {c}", m, s, m / s if s > 0 else np.nan])
        notes = (f"Method: pooled mean group (Pesaran, Shin and Smith 1999), ARDL with {self.lags} lag(s)",
                 f"Units: {len(self.phi)}", f"Observations: {self.nobs}", f"Log likelihood: {self.loglike:.4f}",
                 "Long-run errors from the concentrated likelihood's Hessian; the means' errors are the spread across units.")
        return Table(["", "estimate", "std err", "z"], rows, title=f"Pooled mean group: {self.y_name}", notes=notes)

    def hausman(self) -> Any:
        """Hausman's test of a common long run: the PMG theta (efficient under it) against the mean group of the units'
        own long runs (consistent either way)."""
        from .stats.base import TestResult

        d = self.mg_theta - self.theta
        V = self.mg_theta_cov - self.theta_cov
        try:
            H = float(d @ np.linalg.solve(V, d))
        except np.linalg.LinAlgError:
            H = float(d @ np.linalg.pinv(V) @ d)
        k = len(d)
        return TestResult("Hausman (PMG against MG)", "the long-run coefficients are the same for every unit", H, k,
                          float(sps.chi2.sf(H, k)), "chi2",
                          details={"MG long run": ", ".join(f"{n} {v:.6g}" for n, v in zip(self.x_names, self.mg_theta))})

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.concat([self.phi, self.short_run], axis=1)

    def __repr__(self) -> str:
        return f"<PooledMeanGroup {self.y_name}: long run {dict(zip(self.x_names, np.round(self.theta, 4)))}>"


@api("pymetrix:pooled_mean_group")
def pooled_mean_group(data: Dataset, y: str, x: Sequence[str], lags: int = 1) -> PooledMeanGroup:
    """The pooled mean group estimator of an ARDL(lags, lags) in error-correction form, by concentrated maximum likelihood."""
    x = [x] if isinstance(x, str) else list(x)
    if lags < 1:
        raise SpecificationError("lags must be at least 1")
    units = _pmg_units(data, y, x, lags)
    k = len(x)
    own = []  # each unit's own long run, from its unrestricted error-correction regression
    for d in units:
        Z = np.column_stack([d.ylag, d.xlag, d.W])
        b = np.linalg.lstsq(Z, d.dy, rcond=None)[0]
        own.append(-b[1:1 + k] / b[0])
    own = np.array(own)
    start = np.median(own, axis=0)
    f = lambda th: -_concentrated(th, units)[0]  # noqa: E731
    found = optimize.minimize(f, start, method="BFGS", options={"gtol": 1e-9})
    theta = optimize.minimize(f, found.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}).x
    h = 1e-4 * np.maximum(np.abs(theta), 0.1)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej) + f(theta - ei - ej)) / (4 * h[i] * h[j])
    cov = np.linalg.inv(H)
    ll, coefs = _concentrated(theta, units)
    C = np.array(coefs)
    names = ["const"] + [f"d{y}[-{j}]" for j in range(1, lags)] + [f"d{v}" + (f"[-{j}]" if j else "") for j in range(lags) for v in x]
    index = [d.u for d in units]
    mg, mg_cov = _mg_summary(own)
    return PooledMeanGroup(y, tuple(x), theta, cov, pd.Series(C[:, 0], index=index, name="phi"),
                           pd.DataFrame(C[:, 1:], index=index, columns=names), mg, mg_cov, float(ll), lags,
                           int(sum(len(d.dy) for d in units)))
