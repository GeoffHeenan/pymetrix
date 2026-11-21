"""Factor-augmented VARs (Bernanke, Boivin and Eliasz 2005), by their two-step principal-components method.

    fv = favar(data, panel=[...many series...], observed=["ff"], factors=3, lags=7, slow=[...slow-moving series...])
    fv = favar(data, panel, ["ff"], factors="icp2", lags="bic")   # Bai and Ng's criterion; the VAR's lags by BIC
    fv.var.irf(48)                       # the VAR in the factors and the observed variables
    fv.irf(["ip", "cpi", "m2"], 48, draws=500)   # responses of panel series, bands from the whole two-step procedure
    fv.fevd(["ip", "cpi"], 60)           # the shares of each series' common component, shock by shock

The panel is standardised and its first ``factors`` principal components C (normalised C'C / T = I) estimate the
space of the factors and the observed variables. With ``slow`` (series that do not respond within the period to the
observed variables, such as output and prices to the policy rate) the factors are purged of the observed variables
as Bernanke, Boivin and Eliasz do: C is regressed on the slow series' own components and the observed variables, and
the observed variables' part removed. A VAR in the factors (first) and the observed variables (last, so that a
recursive ordering puts policy last) is then fitted, and each panel series' loadings on the factors and the observed
variables by least squares; a series' response is its loadings times the VAR's responses, in the series' own units.

The number of factors can be chosen by Bai and Ng's (2002) ICp1, ICp2 or ICp3 (``factor_criteria``), and the VAR's
lags by AIC, BIC, HQ or FPE. Bands bootstrap the whole procedure, as Bernanke, Boivin and Eliasz do (Kilian 1998): the
factors and observed variables are simulated from the VAR with resampled residuals, the panel rebuilt from its
loadings with resampled idiosyncratic rows, and everything estimated again. Shocks to the factors depend on how the
factors are rotated, so bands are meaningful for shocks to the observed variables (ordered last).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._api import api
from .errors import SpecificationError
from .output import Table
from ._random import generator as make_rng

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedFAVAR", "factor_criteria", "favar", "principal_components"]

FactorCriterion = Literal["icp1", "icp2", "icp3"]


def principal_components(Z: np.ndarray, r: int) -> np.ndarray:
    """The first r principal components of the (standardised) columns of Z, normalised so that F'F / T = I: the
    leading eigenvectors of the smaller cross-product (Z'Z or ZZ'), as the singular value decomposition gives them."""
    T, N = Z.shape
    if N <= T:
        values, V = np.linalg.eigh(Z.T @ Z)
        V = V[:, ::-1][:, :r]
        U = Z @ V / np.sqrt(np.maximum(values[::-1][:r], np.finfo(float).tiny))
    else:
        values, U = np.linalg.eigh(Z @ Z.T)
        U = U[:, ::-1][:, :r]
        V = Z.T @ U / np.sqrt(np.maximum(values[::-1][:r], np.finfo(float).tiny))
    signs = np.sign(V.sum(axis=0))  # a sign convention: loadings mostly positive
    return np.sqrt(T) * U * np.where(signs == 0, 1, signs)


def factor_criteria(Z: np.ndarray, max_factors: int = 8) -> pd.DataFrame:
    """Bai and Ng's (2002) ICp1, ICp2 and ICp3 for 1 ... ``max_factors`` principal components of the standardised
    panel ``Z`` (T x N): ln V(k) plus k times a penalty, V(k) the mean squared idiosyncratic residual; the smallest wins."""
    T, N = Z.shape
    if not 1 <= max_factors < min(T, N):
        raise SpecificationError(f"max_factors must be between 1 and {min(T, N) - 1}")
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    total = float((Z**2).sum())
    rows = []
    for k in range(1, max_factors + 1):
        V = (total - float((s[:k] ** 2).sum())) / (N * T)
        c2 = min(N, T)
        rows.append([np.log(V) + k * (N + T) / (N * T) * np.log(N * T / (N + T)),
                     np.log(V) + k * (N + T) / (N * T) * np.log(c2),
                     np.log(V) + k * np.log(c2) / c2, V])
    return pd.DataFrame(rows, index=pd.Index(range(1, max_factors + 1), name="factors"), columns=["ICp1", "ICp2", "ICp3", "V"])


def _factors(Zp: np.ndarray, Y: np.ndarray, factors: int, slow: Sequence[int]) -> np.ndarray:
    """The principal-component factors, purged of ``Y`` with the ``slow`` columns' own components (as many as the
    factors, or as the slow series allow when they are fewer)."""
    C = principal_components(Zp, factors)
    if len(slow):
        Zs = Zp[:, list(slow)]
        Zs = (Zs - Zs.mean(axis=0)) / Zs.std(axis=0)
        Fs = principal_components(Zs, min(factors, Zs.shape[1]))
        coef = np.linalg.lstsq(np.column_stack([Fs, Y]), C, rcond=None)[0]
        return C - Y @ coef[Fs.shape[1]:]
    return C


def _loadings(Zp: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Each series' loadings on the factors and observed variables, by least squares with an intercept."""
    return np.linalg.lstsq(np.column_stack([np.ones(len(R)), R]), Zp, rcond=None)[0][1:].T  # (series, factors + observed)


def _var_ols(R: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """A VAR(p) with a constant by least squares, as ``VAR.fit``: the lag matrices (p, m, m) and U'U / (T - k)."""
    T, m = R.shape
    Z = np.column_stack([np.ones(T - p), *[R[p - 1 - l : T - 1 - l] for l in range(p)]])
    B = np.linalg.lstsq(Z, R[p:], rcond=None)[0]
    U = R[p:] - Z @ B
    return np.stack([B[1 + l * m : 1 + (l + 1) * m].T for l in range(p)]), U.T @ U / (len(U) - Z.shape[1])


def _two_step(Zp: np.ndarray, Y: np.ndarray, factors: int, slow: Sequence[int], lags: Any, max_lags: int, index: pd.PeriodIndex,
              names: Sequence[str]) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """The factors (purged with ``slow`` columns), the VAR in them and ``Y``, the loadings and each series' R^2."""
    from .data import Dataset
    from .var import VAR

    F = _factors(Zp, Y, factors, slow)
    R = np.column_stack([F, Y])
    block = pd.DataFrame(R, index=index, columns=list(names))
    var = VAR(list(names), lags=lags, max_lags=max_lags).fit(Dataset.from_frame(block))
    Lam = _loadings(Zp, R)
    fitted = R @ Lam.T
    resid = Zp - (fitted - fitted.mean(axis=0))
    r2 = 1 - (resid**2).sum(axis=0) / (Zp**2).sum(axis=0)
    return F, var, Lam, r2


@dataclass(frozen=True, eq=False)
class FittedFAVAR:
    """A factor-augmented VAR: the VAR in the factors and observed variables, and the panel's loadings."""

    var: Any  # FittedVAR
    factor_names: tuple[str, ...]
    observed: tuple[str, ...]
    loadings: pd.DataFrame = field(repr=False)  # panel series by (factors, observed)
    scale: pd.Series = field(repr=False)  # each panel series' mean and sd, to return to its units
    centre: pd.Series = field(repr=False)
    r2: pd.Series = field(repr=False)  # each series' fit by the factors and observed variables
    factors: pd.DataFrame = field(repr=False)
    panel: pd.DataFrame | None = field(default=None, repr=False)  # the standardised panel, for the bootstrap
    observed_data: pd.DataFrame | None = field(default=None, repr=False)
    slow: tuple[str, ...] = ()
    factor_selection: Table | None = field(default=None, repr=False)  # Bai-Ng criteria, when they chose the factors

    stage = "estimate"

    def _series(self, series: Sequence[str] | str | None) -> list[str]:
        series = list(self.loadings.index) if series is None else ([series] if isinstance(series, str) else list(series))
        unknown = [s for s in series if s not in self.loadings.index]
        if unknown:
            raise SpecificationError(f"{', '.join(unknown)} are not in the FAVAR's panel")
        return series

    def irf(self, series: Sequence[str] | None = None, periods: int = 20, draws: int = 0, bootstrap: Literal["favar", "var"] = "favar",
            alpha: float = 0.05, seed: int = 0) -> Any:
        """Responses of panel ``series`` (default: all) to the VAR's (recursively identified) shocks, in their units,
        with (1 - alpha) percentile bands from ``draws`` bootstrap replications: of the whole two-step procedure
        (``bootstrap="favar"``, Bernanke-Boivin-Eliasz, Kilian 1998), or of the VAR alone, factors and loadings held
        fixed (``"var"``)."""
        from .var import IRF

        if bootstrap not in ("favar", "var"):
            raise SpecificationError("bootstrap is 'favar' (the whole procedure) or 'var' (the VAR alone)")
        series = self._series(series)
        base = self.var.irf(periods, draws=0)
        L = self.loadings.loc[series].to_numpy() * self.scale[series].to_numpy()[:, None]  # (series, factors + observed)

        def through(v: Any) -> Any:
            return np.einsum("sk,hkj->hsj", L, v)

        lower = upper = None
        label = f"FAVAR: {base.identification}"
        if draws and bootstrap == "var":  # each draw's responses through the loadings, then the quantiles
            sims = np.stack([through(self.var._responses(A, S, periods, True, False)) for A, S in self.var._bootstrap(draws, seed)])
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
            label += f"; bands from {draws} bootstraps of the VAR (factors and loadings held fixed)"
        elif draws:
            if self.panel is None:
                raise SpecificationError("this FAVAR was saved without its panel; fit it again for bands from the whole procedure")
            sims = np.stack(list(self._bootstrap(series, periods, draws, seed)))
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
            label += f"; bands from {draws} bootstraps of the whole procedure"
        return IRF(through(base.values), series, list(base.shocks), lower, upper, False, label, 1 - alpha)

    def _bootstrap(self, series: Sequence[str], periods: int, draws: int, seed: int):
        """Responses of ``series`` from the whole procedure on simulated data (see the module docstring)."""
        from .progress import report
        from .var import _impact, _ma

        Zp = self.panel.to_numpy()  # type: ignore[union-attr]
        Y = self.observed_data.to_numpy()  # type: ignore[union-attr]
        k = len(self.factor_names)
        R = np.column_stack([self.factors.to_numpy(), Y])
        T, m = R.shape
        p = self.var.lags
        A, c = self.var.coefs, self.var.estimate.B[0]  # the VAR has a constant first
        U = self.var.resid.frame[list(self.var.variables)].to_numpy()
        U = U - U.mean(axis=0)
        Lam = self.loadings.to_numpy()
        common = R @ Lam.T
        E = Zp - common
        E = E - E.mean(axis=0)  # the idiosyncratic part, intercepts aside
        intercept = (Zp - common).mean(axis=0)
        slow = [list(self.loadings.index).index(s) for s in self.slow]
        rows = [list(self.loadings.index).index(s) for s in series]
        rng = make_rng(seed)
        for d in range(draws):
            report(d / draws, f"bootstrap draw {d + 1} of {draws}")
            Rs = R.copy()
            shocks = U[rng.integers(0, len(U), T - p)]
            for t in range(p, T):
                Rs[t] = c + sum(A[l] @ Rs[t - 1 - l] for l in range(p)) + shocks[t - p]
            Zs = intercept + Rs @ Lam.T + E[rng.integers(0, T, T)]
            sd = Zs.std(axis=0)
            Zs = (Zs - Zs.mean(axis=0)) / sd
            Y_s = Rs[:, k:]
            R_s = np.column_stack([_factors(Zs, Y_s, k, slow), Y_s])
            A_s, sigma_s = _var_ols(R_s, p)
            L = _loadings(Zs, R_s)[rows] * (sd[rows] * self.scale.iloc[rows].to_numpy())[:, None]
            yield np.einsum("sk,hkj->hsj", L, _ma(A_s, periods) @ _impact(self.var.identification, sigma_s, A_s, self.var.variables))

    def fevd(self, series: Sequence[str] | None = None, periods: int = 20) -> Any:
        """The share of each series' common component's forecast-error variance due to each shock, by horizon (the
        idiosyncratic part left out, as Bernanke, Boivin and Eliasz report it; ``r2`` gives the common part's share
        of the series)."""
        from .var import FEVD, _ma

        series = self._series(series)
        L = self.loadings.loc[series].to_numpy()
        theta = np.einsum("sk,hkj->hsj", L, _ma(self.var.coefs, periods - 1) @ self.var.impact())
        contributions = np.cumsum(theta**2, axis=0)
        shares = contributions / contributions.sum(axis=2, keepdims=True)
        return FEVD(shares, series, list(self.var.shocks), f"FAVAR common components, {self.var.irf(1, draws=0).identification}")

    def summary(self) -> Table:
        rows = [[s, float(self.r2[s])] for s in self.r2.sort_values(ascending=False).index[:15]]
        notes = [f"{len(self.factor_names)} factors from {len(self.loadings)} series; observed {', '.join(self.observed)}; "
                 f"VAR with {self.var.lags} lags on {len(self.factors)} periods.",
                 f"Median R^2 of the panel on the factors and observed variables: {self.r2.median():.3f}."]
        if self.slow:
            notes.append(f"Factors purged of the observed variables with {len(self.slow)} slow-moving series.")
        if self.factor_selection is not None:
            notes += [n for n in self.factor_selection.notes if n.startswith("Chosen")]
        return Table(["series", "R^2"], rows, title="Factor-augmented VAR", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.loadings

    def __repr__(self) -> str:
        return f"<FittedFAVAR {len(self.factor_names)} factors, observed {', '.join(self.observed)}, {len(self.loadings)} series>"


@api("pymetrix:favar")
def favar(data: Dataset, panel: Sequence[str], observed: Sequence[str], factors: int | FactorCriterion = 3, lags: int | str = 2,
          slow: Sequence[str] | None = None, span: Any = None, max_factors: int = 8, max_lags: int = 8) -> FittedFAVAR:
    """A FAVAR: ``factors`` principal components of the ``panel`` (or as many as Bai and Ng's ``"icp1"``, ``"icp2"``
    or ``"icp3"`` choose, up to ``max_factors``) with the ``observed`` variables in a VAR(``lags``; or chosen by
    ``"aic"``, ``"bic"``, ``"hq"`` or ``"fpe"`` up to ``max_lags``)."""
    panel, observed = list(panel), [observed] if isinstance(observed, str) else list(observed)
    if data.kind != "time_series":
        raise SpecificationError("a FAVAR needs a single time series panel of series")
    missing = [v for v in [*panel, *observed, *(slow or [])] if v not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    not_in_panel = [s for s in (slow or []) if s not in panel]
    if not_in_panel:
        raise SpecificationError(f"the slow-moving series must be in the panel: {', '.join(not_in_panel)}")
    names = list(dict.fromkeys([*panel, *observed]))
    frame = data.frame[names].astype(float)
    if span is not None:
        frame = frame[data.sample_mask(span)]
    frame = frame.dropna()
    centre, scale = frame[panel].mean(), frame[panel].std(ddof=0)
    if (scale == 0).any():
        raise SpecificationError("a panel series is constant")
    Zp = ((frame[panel] - centre) / scale).to_numpy()
    selection = None
    if isinstance(factors, str):
        if factors not in ("icp1", "icp2", "icp3"):
            raise SpecificationError("factors is a number or one of 'icp1', 'icp2', 'icp3'")
        crit = factor_criteria(Zp, min(max_factors, min(Zp.shape) - 1))  # no more than the panel allows
        label = {"icp1": "ICp1", "icp2": "ICp2", "icp3": "ICp3"}[factors]
        chosen = int(crit[label].idxmin())
        notes = [f"Bai and Ng (2002) on {Zp.shape[1]} standardised series and {Zp.shape[0]} periods; smaller is better",
                 f"Chosen by {label}: {chosen} factor{'s' * (chosen != 1)}"]
        selection = Table.from_frame(crit, title="Number of factors", notes=notes)
        factors = chosen
    if factors < 1 or factors >= len(panel):
        raise SpecificationError("factors must be at least 1 and fewer than the panel's series")
    fnames = [f"factor{i + 1}" for i in range(factors)]
    slow_idx = [panel.index(s) for s in (slow or [])]
    F, var, Lam, r2 = _two_step(Zp, frame[observed].to_numpy(), factors, slow_idx, lags, max_lags, frame.index, [*fnames, *observed])
    return FittedFAVAR(var, tuple(fnames), tuple(observed), pd.DataFrame(Lam, index=panel, columns=[*fnames, *observed]),
                       scale, centre, pd.Series(r2, index=panel, name="R2"), pd.DataFrame(F, index=frame.index, columns=fnames),
                       pd.DataFrame(Zp, index=frame.index, columns=panel), frame[observed], tuple(slow or ()), selection)
