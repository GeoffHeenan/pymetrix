"""Correlograms and ARMA order selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ..errors import SpecificationError
from ..output import Figure, Table

if TYPE_CHECKING:
    from ..data import Dataset


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations at lags 0..nlags (demeaned, divided by n)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = x @ x
    return np.r_[1.0, [x[k:] @ x[:-k] / denom for k in range(1, nlags + 1)]]


def pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelations at lags 0..nlags (Levinson-Durbin on the sample ACF)."""
    r = acf(x, nlags)
    out = [1.0]
    phi = np.zeros(0)
    for k in range(1, nlags + 1):
        kk = (r[k] - phi @ r[k - 1 : 0 : -1]) / (1 - phi @ r[1:k]) if k > 1 else r[1]
        phi = np.r_[phi - kk * phi[::-1], kk]
        out.append(kk)
    return np.array(out)


def correlogram_table(x: np.ndarray, lags: int, title: str, model_df: int = 0) -> Table:
    n = len(x)
    if lags >= n:
        raise SpecificationError(f"{lags} lags need more than {lags} observations")
    ac, pac = acf(x, lags)[1:], pacf(x, lags)[1:]
    k = np.arange(1, lags + 1)
    q = n * (n + 2) * np.cumsum(ac**2 / (n - k))
    df = k - model_df
    p = np.where(df > 0, sps.chi2.sf(q, np.maximum(df, 1)), np.nan)
    rows = [[int(i), a, b, qq, pp] for i, a, b, qq, pp in zip(k, ac, pac, q, p)]
    notes = [f"{n} observations; approximate 95% band for AC and PAC: ±{1.96 / np.sqrt(n):.3f}"]
    if model_df:
        notes.append(f"Q-stat p-values use lags minus {model_df} ARMA terms as degrees of freedom")
    return Table(["lag", "AC", "PAC", "Q-stat", "p-value"], rows, title=title, notes=notes)


def correlogram_figure(table: Table) -> Figure:
    from ..figures import correlogram_figure as draw

    return draw(table)


@dataclass(frozen=True, eq=False)
class OrderSelection:
    """Information criteria over a grid of ARMA orders, all on the same sample."""

    table: Table
    best: dict[str, tuple[int, int]]

    def __str__(self) -> str:
        return str(self.table)

    def _repr_html_(self) -> str:
        return self.table._repr_html_()


def select_order(
    data: Dataset,
    variable: str,
    max_ar: int = 4,
    max_ma: int = 2,
    exog: Sequence[str] = (),
    intercept: bool = True,
    span: tuple | None = None,
) -> OrderSelection:
    """Fit regressions with ARMA(p, q) errors for every p <= max_ar, q <= max_ma
    by exact maximum likelihood and compare AIC, BIC and Hannan-Quinn."""
    from ..estimation import ARMA, Equation

    terms = list(exog) + ([] if intercept else ["0"])
    rhs = " + ".join(terms) if terms else "1"
    rows, names, crit = [], [], {}
    n_used = None
    for p in range(max_ar + 1):
        for q in range(max_ma + 1):
            label = f"ARMA({p},{q})"
            try:
                errors = ARMA(ar=p, ma=q) if p or q else None
                fit = Equation(f"{variable} ~ {rhs}", errors=errors).fit(data, span=span)
                ll, n = fit.stats["loglike"], fit.stats["nobs"]
                n_used = n
                k = len(fit.params) + (0 if errors else 1)  # count sigma2 for the plain regression too
                values = [ll, -2 * ll + 2 * k, -2 * ll + k * np.log(n), -2 * ll + 2 * k * np.log(np.log(n))]
            except Exception:  # an order that fails to converge is reported, not fatal
                values = [np.nan] * 4
            rows.append(values)
            names.append(label)
            crit[label] = (p, q)
    frame = pd.DataFrame(rows, index=names, columns=["loglike", "AIC", "BIC", "HQ"])
    best = {c: crit[frame[c].idxmin()] for c in ("AIC", "BIC", "HQ") if frame[c].notna().any()}
    notes = [f"{variable} ~ {rhs}; exact Gaussian likelihood on the same {int(n_used or 0)} observations"]
    notes += [f"Best by {c}: ARMA({p},{q})" for c, (p, q) in best.items()]
    return OrderSelection(Table.from_frame(frame, title=f"ARMA order selection for {variable}", notes=notes), best)
