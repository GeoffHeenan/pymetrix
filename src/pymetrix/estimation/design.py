"""Building estimation samples: shared by least squares, IV and maximum likelihood.

A design evaluates an equation's terms (and any instruments) over the whole
dataset, so that lags at the start of the span use earlier data, then applies
the sample selection, drops or rejects missing rows, and absorbs fixed effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from .._serial import to_jsonable
from .._version import __version__
from ..expr import Expr, evaluate
from ..metadata import Provenance
from . import criteria
from .covariance import Cluster

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation


@dataclass(frozen=True, eq=False)
class LinearDesign:
    """A linear-form equation's sample: y, regressors X, optional instruments Z."""

    y: np.ndarray
    X: np.ndarray
    names: list[str]
    index: pd.Index
    w: np.ndarray | None
    Z: np.ndarray | None
    z_names: list[str]
    dropped: int
    absorbed: int  # degrees of freedom used by fixed effects
    y_observed: np.ndarray  # before fixed-effect demeaning


@dataclass(frozen=True, eq=False)
class ExplicitDesign:
    """An explicit-form equation's sample: arrays for each variable-at-lag, optional instruments."""

    arrays: dict[tuple[str, int], np.ndarray]
    index: pd.Index
    w: np.ndarray | None
    Z: np.ndarray | None
    z_names: list[str]
    dropped: int

    def resolve(self, name: str, lag: int) -> np.ndarray:
        return self.arrays[(name, lag)]

    @property
    def n(self) -> int:
        return len(self.index)


def sample(frame: pd.DataFrame, mask: np.ndarray, missing: str) -> tuple[pd.DataFrame, int]:
    if missing not in ("drop", "raise"):
        raise SpecificationError("missing must be 'drop' or 'raise'")
    frame = frame[mask]
    bad = ~np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
    if bad.any() and missing == "raise":
        raise SpecificationError(f"{int(bad.sum())} observations in the sample have missing values; use missing='drop'")
    used = frame[~bad]
    if used.empty:
        raise SpecificationError("no observations left in the sample")
    return used, int(bad.sum())


def as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(np.broadcast_to(np.asarray(value, dtype=float), (len(index),)), index=index)


def dummies(values: pd.Series, name: str, drop_first: bool) -> dict[str, pd.Series]:
    """One 0/1 column per category (the first left out when ``drop_first``), named ``C(name)[category]``; the
    seasons of ``seasonal()`` are named ``seasonal[Q2]`` or ``seasonal[M2]`` and ordered by period of the year."""
    if name == "seasonal()":
        cats = sorted(values.dropna().unique())
        prefix = "Q" if values.name == "seasonal(Q)" else "M"
        label = lambda c: f"seasonal[{prefix}{int(c)}]"  # noqa: E731
    else:
        cats = sorted(values.dropna().unique(), key=None if values.attrs.get("calendar") else str)
        label = lambda c: f"C({name})[{c}]"  # noqa: E731
    if drop_first:
        cats = cats[1:]
    return {label(c): (values == c).astype(float).where(values.notna()) for c in cats}


def absorb(M: np.ndarray, codes: Sequence[np.ndarray], tol: float = 1e-11, max_iter: int = 10_000) -> np.ndarray:
    """Remove group means for each fixed effect (alternating projections)."""
    M = M.copy()
    scale = 1.0 + np.abs(M).max()
    for _ in range(max_iter):
        largest = 0.0
        for c in codes:
            counts = np.bincount(c)
            means = np.column_stack([np.bincount(c, weights=M[:, j]) for j in range(M.shape[1])]) / counts[:, None]
            M -= means[c]
            largest = max(largest, float(np.abs(means).max()))
        if len(codes) == 1 or largest < tol * scale:
            return M
    raise RuntimeError("fixed-effect demeaning did not converge")


def _prefixed(series: Mapping[str, pd.Series], prefix: str) -> dict[str, pd.Series]:
    return {f"{prefix}{k}": v for k, v in series.items()}


def linear_design(
    eq: Equation,
    data: Dataset,
    span: Any,
    where: str | None,
    weights: str | None,
    missing: str,
    instruments: Mapping[str, pd.Series] | None = None,
    allow_empty: bool = False,
) -> LinearDesign:
    index = data.frame.index
    cols: dict[str, pd.Series] = {"__y__": as_series(evaluate(eq.lhs, data.lagged), index)}
    if eq.intercept:
        cols["const"] = pd.Series(1.0, index=index)
    for term in eq.terms:
        cols[term.label] = as_series(evaluate(term.expr, data.lagged), index)
    for name in eq.categoricals:
        cols.update(dummies(data.values_of(name), name, drop_first=eq.intercept or bool(eq.fixed_effects)))
    if weights:
        cols["__w__"] = data[weights].astype(float)
    cols.update(_prefixed(instruments or {}, "__z__"))
    used, dropped = sample(pd.DataFrame(cols), data.sample_mask(span, where), missing)
    names = [c for c in used.columns if not c.startswith("__")]
    z_names = [c[5:] for c in used.columns if c.startswith("__z__")]
    if not names and not allow_empty:
        raise SpecificationError("the equation has no regressors")
    y = used["__y__"].to_numpy(float)
    X = used[names].to_numpy(float)
    Z = used[[f"__z__{z}" for z in z_names]].to_numpy(float) if z_names else None
    w = used["__w__"].to_numpy(float) if weights else None
    absorbed = 0
    if eq.fixed_effects:
        if w is not None:
            raise NotImplementedError("weights with fixed effects are not supported yet")
        codes = [pd.factorize(data.values_of(f).reindex(used.index))[0] for f in eq.fixed_effects]
        blocks = [y[:, None], X] + ([Z] if Z is not None else [])
        M = absorb(np.column_stack(blocks), codes)
        y, X = M[:, 0], M[:, 1 : 1 + X.shape[1]]
        Z = M[:, 1 + X.shape[1] :] if Z is not None else None
        absorbed = sum(int(c.max()) + 1 for c in codes) - (len(codes) - 1)
    return LinearDesign(y, X, names, used.index, w, Z, z_names, dropped, absorbed, used["__y__"].to_numpy(float))


def explicit_design(
    eq: Equation,
    data: Dataset,
    span: Any,
    where: str | None,
    weights: str | None,
    missing: str,
    instruments: Mapping[str, pd.Series] | None = None,
) -> ExplicitDesign:
    refs = sorted(eq.lhs.refs() | eq.rhs.refs(), key=str)
    cols = {str(r): data.lagged(r.id, r.lag).astype(float) for r in refs}
    if weights:
        cols["__w__"] = data[weights].astype(float)
    cols.update(_prefixed(instruments or {}, "__z__"))
    used, dropped = sample(pd.DataFrame(cols, index=data.frame.index), data.sample_mask(span, where), missing)
    z_names = [c[5:] for c in used.columns if c.startswith("__z__")]
    return ExplicitDesign(
        arrays={(r.id, r.lag): used[str(r)].to_numpy(float) for r in refs},
        index=used.index,
        w=used["__w__"].to_numpy(float) if weights else None,
        Z=used[[f"__z__{z}" for z in z_names]].to_numpy(float) if z_names else None,
        z_names=z_names,
        dropped=dropped,
    )


def provenance(eq: Equation, method: Any, data: Dataset, used: pd.Index, dropped: int, extra_variables: Sequence[str] = ()) -> Provenance:
    t = data.time_axis
    span = None
    if t is not None:
        periods = used.get_level_values(t.name)
        span = (str(periods.min()), str(periods.max()))
    names = sorted(set(eq.variables()) | {v for v in extra_variables if v in data.variables})
    from .._capture import fingerprint
    from ..adapters import native

    used_frame = data.frame.loc[used, [n for n in names if n in data.frame.columns]]
    return Provenance(
        data_vintages={v: data.variables[v].vintage for v in names},
        spec=to_jsonable(eq),
        method=to_jsonable(method),
        sample=span,
        nobs=len(used),
        dropped=dropped,
        package_version=__version__,
        seed=getattr(method, "seed", None),
        data_fingerprint=fingerprint(used_frame),
        backend=native.backend_record(f"pymetrix.{type(method).__name__}"),  # a library adapter replaces it (with_backend)
    )


def ls_stats(y: np.ndarray, u: np.ndarray, w: np.ndarray | None, k: int, df_resid: int, centered: bool, time_series: bool, within: bool) -> dict[str, float]:
    n = len(u)
    w = np.ones(n) if w is None else w
    ssr = float(np.sum(w * u**2))
    if within:
        tss = float(np.sum(y**2))  # y is already demeaned
    elif centered:
        ybar = np.sum(w * y) / np.sum(w)
        tss = float(np.sum(w * (y - ybar) ** 2))
    else:
        tss = float(np.sum(w * y**2))
    r2 = criteria.r2(ssr, tss)
    loglike = criteria.gaussian_loglike(ssr, n)
    with np.errstate(divide="ignore", invalid="ignore"):  # an exact fit has ssr = 0
        dw = np.sum(np.diff(u) ** 2) / np.sum(u**2)
    out = {  # the definitions are in estimation.criteria (and docs/design/definitions.md)
        "nobs": float(n),
        "k": float(k),
        "df_resid": float(df_resid),
        "ssr": ssr,
        "se_regression": criteria.se_regression(ssr, df_resid),
        "r2_within" if within else "r2": float(r2),
        "r2_adj": criteria.r2_adjusted(r2, n, df_resid, centered),
        "loglike": loglike,
        "aic": criteria.aic(loglike, k),
        "bic": criteria.bic(loglike, k, n),
        "hq": criteria.hq(loglike, k, n),
    }
    if time_series:
        out["durbin_watson"] = float(dw)
    return out


def cov_groups(cov: Any, data: Dataset, used: pd.Index) -> list[np.ndarray]:
    if not isinstance(cov, Cluster):
        return []
    return [data.values_of(level).reindex(used).to_numpy() for level in cov.levels]


def check_hac(cov: Any, data: Dataset) -> None:
    if cov == "hac" and data.kind != "time_series":
        raise SpecificationError("HAC covariance needs a single time series; use Cluster(...) for panels")


def instrument_series(texts: Sequence[str], data: Dataset) -> dict[str, pd.Series]:
    """Evaluate instrument expressions such as ``"g"`` or ``"dlog(y)[-2]"``."""
    from ..expr import parse_expression

    out = {}
    for text in texts:
        node: Expr = parse_expression(text)
        unknown = sorted(node.variables() - set(data.names))
        if unknown:
            raise SpecificationError(f"instrument {text!r} uses names that are not series: {unknown}")
        out[str(node)] = as_series(evaluate(node, data.lagged), data.frame.index)
    return out
