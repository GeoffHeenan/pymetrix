"""Two-way reconciliation: components that add up to a high-frequency total and meet their own low-frequency totals.

    rec = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", low=annual, fixed=["gov"])
    quarterly = quarterly.with_series(**{c: rec.series[c] for c in rec.components})

Each component (often a disaggregated series) is moved as little as possible so that, every high-frequency period,
the components add up to the total, and each component with a series in ``low`` still aggregates to it (sums,
averages, first or last values). Components in ``fixed`` are not moved. The criterion is Di Fonzo and Marini's
(2011) multivariate Denton: the first differences of each component's proportional (``denton``, the default) or
additive (``denton-additive``) adjustment, least-squares balancing (``least-squares``), where each value's
adjustment has a variance proportional to its size, as in Stone, Champernowne and Meade (1942), or weighted least
squares (``weighted``), where each component's adjustments have a constant variance, its weight, whatever its level
and sign. ``weights`` scales how much each component moves (larger moves more).

The problem is a quadratic criterion under linear constraints, solved exactly: a particular solution of the
constraints, then the criterion's minimum over their null space; the constraints then hold to rounding error.
Constraints that contradict each other (components' low-frequency totals that do not add up to the total's) are
reported, not spread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg

from ._api import api
from .disaggregation import conversion_matrix
from .errors import DataError, NotFoundError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["Reconciliation", "reconcile"]

ReconcileMethod = Literal["denton", "denton-additive", "least-squares", "weighted"]
RECONCILE_METHODS = ("denton", "denton-additive", "least-squares", "weighted")


def _names(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [v for v in value.replace(",", " ").split() if v]
    return [str(v) for v in value]


@api("pymetrix:reconcile")
def reconcile(data: Dataset, components: str | Sequence[str], total: str, low: Dataset | None = None, *,
              fixed: str | Sequence[str] = (), method: ReconcileMethod = "denton", conversion: str = "sum",
              weights: Mapping[str, float] | None = None) -> Reconciliation:
    """Reconcile ``components`` (series of ``data``) with their high-frequency ``total`` (a series of ``data``) and,
    where ``low`` has a series of the same name, with that component's low-frequency values.

    ``fixed``: components held as they are. ``method``: ``denton`` (proportional first differences, Di Fonzo and
    Marini's multivariate Denton), ``denton-additive``, ``least-squares`` (each adjustment's variance proportional
    to the value), or ``weighted`` (minimises sum_c sum_t e_ct^2 / w_c: each component's adjustments have a constant
    variance w_c, whatever its level and sign, so with equal weights it is the minimum-norm adjustment).
    ``conversion``: how the low-frequency values aggregate (sum, average, first, last). ``weights``:
    {component: relative weight}, larger moving more (default 1 each; positive: hold a component with ``fixed``)."""
    names = _names(components)
    held = _names(fixed)
    if method not in RECONCILE_METHODS:
        raise SpecificationError(f"method is one of {', '.join(RECONCILE_METHODS)}, not {method!r}")
    if len(names) < 2 or len(set(names)) != len(names):
        raise SpecificationError("give two or more different components")
    if total in names:
        raise SpecificationError(f"the total {total!r} cannot also be a component")
    for n in [*names, total]:
        if n not in data.names:
            raise NotFoundError(f"no series {n!r} in the high-frequency data")
    for n in held:
        if n not in names:
            raise NotFoundError(f"the fixed series {n!r} is not one of the components")
    free = [n for n in names if n not in held]
    if not free:
        raise SpecificationError("every component is fixed: nothing can move to meet the total")
    if data.kind != "time_series" or data.time_axis is None:
        raise SpecificationError("reconciliation needs a time series dataset")
    if low is not None and (low.kind != "time_series" or low.time_axis is None):
        raise SpecificationError("the low-frequency data must be a time series dataset")
    w = {n: 1.0 for n in names}
    for k, v in (weights or {}).items():
        if k not in names:
            raise NotFoundError(f"weights names {k!r}, which is not a component")
        if not float(v) > 0:
            raise SpecificationError(f"weights are positive ({k!r} has {v!r}); to hold a component as it is, list it in fixed")
        w[k] = float(v)

    frame = data.frame[[*names, total]].astype(float)
    ok = np.isfinite(frame.to_numpy()).all(axis=1)
    if not ok.any():
        raise DataError("the components and the total have no period in common")
    first, last = np.flatnonzero(ok)[[0, -1]]
    if not ok[first : last + 1].all():
        raise DataError("the components or the total have gaps; reconciliation needs an unbroken sample")
    frame = frame.iloc[first : last + 1]
    hi = pd.PeriodIndex(frame.index)
    n = len(hi)
    P = frame[names]
    z = frame[total].to_numpy()

    low_used: dict[str, pd.Series] = {}
    C_of: dict[str, np.ndarray] = {}
    if low is not None:
        for name in names:
            if name in low.names:
                values = low[name].astype(float).dropna()
                covered = [p for p in values.index if p.start_time >= hi[0].start_time and p.end_time <= hi[-1].end_time]
                if covered:
                    lo = pd.PeriodIndex(covered)
                    C_of[name] = conversion_matrix(lo, hi, conversion)
                    low_used[name] = values.loc[lo]

    # the free components stacked: y = (y_1', ..., y_k')'; criterion (y - p)' A (y - p); constraints G y = g
    k = len(free)
    p = np.concatenate([P[c].to_numpy() for c in free])
    blocks, D = [], (np.eye(n) - np.eye(n, k=-1))[1:]
    for c in free:
        x = P[c].to_numpy()
        if method == "denton":
            if np.any(x == 0):
                raise SpecificationError(f"proportional Denton needs components without zeros ({c!r} has some); use denton-additive")
            Wc = D @ np.diag(1.0 / x)
            blocks.append(Wc.T @ Wc / w[c])
        elif method == "denton-additive":
            blocks.append(D.T @ D / w[c])
        elif method == "weighted":  # e_c' e_c / w_c: a constant variance w_c, whatever the level and sign
            blocks.append(np.eye(n) / w[c])
        else:
            scale = np.maximum(np.abs(x), 1e-12 * max(float(np.abs(P.to_numpy()).max()), 1.0))
            blocks.append(np.diag(1.0 / (w[c] * scale)))
    A = linalg.block_diag(*blocks)
    G_rows, g_parts = [np.hstack([np.eye(n)] * k)], [z - P[held].to_numpy().sum(axis=1) if held else z.copy()]
    for j, c in enumerate(free):
        if c in C_of:
            row = np.zeros((C_of[c].shape[0], k * n))
            row[:, j * n : (j + 1) * n] = C_of[c]
            G_rows.append(row)
            g_parts.append(low_used[c].to_numpy())
    G, g = np.vstack(G_rows), np.concatenate(g_parts)

    r = g - G @ p
    delta0 = linalg.lstsq(G, r)[0]  # the smallest change that meets the constraints
    miss = float(np.max(np.abs(G @ delta0 - r))) if len(r) else 0.0
    if miss > 1e-8 * max(1.0, float(np.max(np.abs(g)))):
        raise SpecificationError(f"the constraints contradict each other (by up to {miss:.4g}): the components' low-frequency totals "
                                 "do not add up to the total's, or a fixed component's do not fit; correct the data or fix fewer components")
    N = linalg.null_space(G)
    if N.size:
        H = N.T @ A @ N
        v = linalg.lstsq(H, -N.T @ (A @ delta0))[0]
        delta = delta0 + N @ v
    else:
        delta = delta0
    y = (p + delta).reshape(k, n)
    out = P.copy()
    for j, c in enumerate(free):
        out[c] = y[j]
    out.columns.name = None

    from ._capture import data_fingerprint
    from ._version import __version__
    from .metadata import Provenance

    spec: dict[str, Any] = {"components": names, "total": total, "method": method}
    if held:
        spec["fixed"] = held
    if conversion != "sum":
        spec["conversion"] = conversion
    if weights:
        spec["weights"] = dict(w)
    fingerprint = data_fingerprint(data, [*names, total], hi)
    if low is not None and low_used:
        fingerprint = f"{fingerprint}:{data_fingerprint(low, sorted(low_used))}"
    vintages = {c: data.variables[c].vintage for c in [*names, total] if c in data.variables}
    prov = Provenance(vintages, spec, {"type": "reconciliation", "method": method}, (str(hi[0]), str(hi[-1])), n,
                      0, package_version=__version__, data_fingerprint=fingerprint)
    return Reconciliation(out, P.copy(), pd.Series(z, index=hi, name=total), tuple(names), tuple(held), method, conversion,
                          low_used, C_of, dict(w), prov)


@dataclass(frozen=True, eq=False)
class Reconciliation:
    """Reconciled components: ``series`` (the components, fixed ones included), ``preliminary`` (as given),
    ``adjustments`` (series minus preliminary), ``total``, ``check`` (the largest gap from any constraint),
    ``diagnostics()`` (the size of each component's adjustment and its gaps before and after)."""

    series: pd.DataFrame
    preliminary: pd.DataFrame
    total: pd.Series
    components: tuple[str, ...]
    fixed: tuple[str, ...]
    method: str
    conversion: str
    low_totals: dict[str, pd.Series] = field(default_factory=dict)  # each component's low-frequency values used
    C: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    weights: dict[str, float] = field(default_factory=dict)
    provenance: Any = None

    @property
    def adjustments(self) -> pd.DataFrame:
        return self.series - self.preliminary

    @property
    def frame(self) -> pd.DataFrame:
        """The reconciled components and the total, by period."""
        return pd.concat([self.series, self.total], axis=1)

    def total_gaps(self, reconciled: bool = True) -> pd.Series:
        """The total minus the components' sum, by period (zero once reconciled)."""
        source = self.series if reconciled else self.preliminary
        return (self.total - source.sum(axis=1)).rename("gap")

    def low_gaps(self, name: str, reconciled: bool = True) -> pd.Series:
        """A component's low-frequency values minus its high-frequency values aggregated."""
        if name not in self.low_totals:
            raise NotFoundError(f"{name!r} has no low-frequency values")
        source = self.series if reconciled else self.preliminary
        return self.low_totals[name] - self.C[name] @ source[name].to_numpy()

    @property
    def check(self) -> float:
        """The largest gap from the constraints the reconciled series must meet (zero up to rounding)."""
        gaps = [float(np.max(np.abs(self.total_gaps())))]
        gaps += [float(np.max(np.abs(self.low_gaps(c)))) for c in self.low_totals if c not in self.fixed]
        return max(gaps)

    def diagnostics(self) -> Table:
        """By component: held fixed or not, low-frequency values met, the largest and the mean absolute adjustment (and
        in percent), and the largest gap from its low-frequency values before and after; the total's gap last."""
        rows, index = [], []
        adj = self.adjustments
        for c in self.components:
            a = adj[c].to_numpy()
            base = self.preliminary[c].to_numpy()
            pct = 100 * np.abs(a) / np.where(base != 0, np.abs(base), np.nan)
            has_low = c in self.low_totals
            rows.append(["yes" if c in self.fixed else "no", len(self.low_totals[c]) if has_low else 0,
                         float(np.max(np.abs(a))), float(np.mean(np.abs(a))), float(np.nanmean(pct)) if np.isfinite(pct).any() else np.nan,
                         float(np.max(np.abs(self.low_gaps(c, False)))) if has_low else np.nan,
                         float(np.max(np.abs(self.low_gaps(c)))) if has_low else np.nan])
            index.append(c)
        before, after = self.total_gaps(False).abs().max(), self.total_gaps().abs().max()
        rows.append(["", "", np.nan, np.nan, np.nan, float(before), float(after)])
        index.append(f"{self.total.name} (sum of components)")
        return Table(["fixed", "low-frequency values", "largest adjustment", "mean absolute adjustment", "mean absolute adjustment (%)",
                      "largest gap before", "largest gap after"], rows, index=index, title="Reconciliation: adjustments and gaps")

    def summary(self) -> Table:
        t = self.diagnostics()
        labels = {"denton": "proportional Denton (Di Fonzo-Marini)", "denton-additive": "additive Denton (Di Fonzo-Marini)",
                  "least-squares": "least-squares balancing", "weighted": "weighted least squares (constant variance per component)"}
        notes = [(f"Method: {labels[self.method]}; {len(self.components)} components ({len(self.fixed)} fixed), "
                  f"{len(self.series)} periods ({self.series.index[0]} to {self.series.index[-1]})"),
                 f"The components add up to {self.total.name} and meet their low-frequency values to within {self.check:.2e}"]
        return Table(t.columns, t.rows, index=t.index, title=f"Reconciliation of {', '.join(self.components)} with {self.total.name}", notes=notes)

    def plot(self) -> Any:
        """Each component's adjustment, in percent of its preliminary value."""
        from .figures import Figure, Line, Panel

        pct = 100 * self.adjustments / self.preliminary.where(self.preliminary != 0)
        layers = tuple(Line(pct[c], c) for c in self.components if c not in self.fixed)
        return Figure((Panel(layers, title="Adjustments (% of the preliminary values)"),))

    def __repr__(self) -> str:
        return f"<Reconciliation {', '.join(self.components)} with {self.total.name}, {self.method}: {len(self.series)} periods>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
