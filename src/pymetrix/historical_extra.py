"""Historical decompositions of panel VARs and proxy SVARs, in the VAR's ``Decomposition`` (pymetrix.var).

    pvar = VAR(["inv", "sales"], lags=1, by="firm").fit(panel)
    hd = pvar.historical_decomposition(draws=200)        # one decomposition per entity
    hd["firm 3"].plot(); hd.select("firm 3")              # an entity's decomposition; the views of another entity
    psvar = var.proxy_svar(data, "ff4", normalise_on="ff")
    psvar.historical_decomposition(draws=200).plot()      # the identified shock, the other shocks together, the baseline

A panel VAR y_it = A_1 y_i,t-1 + ... + A_p y_i,t-p + mu_i + u_it splits each entity's data, as a VAR does, into the
baseline (the path from the entity's first p observations with its fixed effect mu_i and no shocks) and the
contributions of the structural shocks P^-1 u_it. The fixed effect is the entity's mean of y_it - sum A_l y_i,t-l:
the constant within least squares estimates, and the effect forward orthogonal deviations removed.

A proxy SVAR identifies one impact column b. Its shock is the part of the residuals along it, e(t) = b' Sigma^-1 u(t)
/ (b' Sigma^-1 b) (the first row of B^-1 for any B = [b, ...] with B B' = Sigma); its contribution propagates b e(t),
and the other shocks' contribution, combined, propagates u(t) - b e(t).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .errors import NotFoundError, SpecificationError
from .output import Table
from ._random import generator as make_rng

if TYPE_CHECKING:
    from .models.decompose import Decomposition
    from .panelvar import FittedPanelVAR
    from .proxy import ProxySVAR

__all__ = ["PanelDecomposition", "panel_historical", "propagate", "proxy_historical"]

OTHER = "other shocks"


def propagate(A: np.ndarray, impulses: np.ndarray) -> np.ndarray:
    """Impulses (periods, variables, parts) through the VAR with lag matrices A (lags, K, K), from zero:
    x(t) = impulses(t) + sum_l A_l x(t - l), each part separately."""
    p = A.shape[0]
    out = np.zeros_like(impulses, dtype=float)
    for t in range(len(impulses)):
        x = impulses[t].astype(float)
        for l in range(1, min(p, t) + 1):
            x = x + A[l - 1] @ out[t - l]
        out[t] = x
    return out


def _baseline(A: np.ndarray, constant: np.ndarray | Sequence[np.ndarray], start: np.ndarray, T: int) -> np.ndarray:
    """The path with no shocks: from ``start`` (the first period's lags, lag-major: y(t0-1), ..., y(t0-p)) with the
    constant (one vector, or one per period)."""
    p, K = A.shape[0], A.shape[1]
    history = list(np.asarray(start, float).reshape(p, K)[::-1])  # y(t0-p), ..., y(t0-1)
    c = np.asarray(constant, float)
    out = np.empty((T, K))
    for t in range(T):
        y = (c[t] if c.ndim == 2 else c).copy()
        for l in range(1, p + 1):
            y = y + A[l - 1] @ history[-l]
        out[t] = y
        history.append(y)
    return out


def _bands(samples: np.ndarray, alpha: float, index: pd.Index, variables: Sequence[str], parts: Sequence[str]) -> tuple[dict, dict, dict]:
    """Pointwise lower, median and upper quantiles of contributions (draws, periods, variable, part), as frames."""
    lower, median, upper = np.quantile(samples, [alpha / 2, 0.5, 1 - alpha / 2], axis=0)

    def frames(x: np.ndarray) -> dict[str, pd.DataFrame]:
        return {v: pd.DataFrame(x[:, i, :], index=index, columns=list(parts)) for i, v in enumerate(variables)}

    return frames(lower), frames(median), frames(upper)


def _decomposition(contributions: np.ndarray, Y: np.ndarray, baseline: np.ndarray, shocks: np.ndarray, index: pd.Index,
                   variables: Sequence[str], parts: Sequence[str], shock_names: Sequence[str], title: str, measure: str) -> Decomposition:
    from .models.decompose import Decomposition

    frames = {v: pd.DataFrame(contributions[:, i, :], index=index, columns=list(parts)) for i, v in enumerate(variables)}
    base = pd.DataFrame(baseline, index=index, columns=list(variables))
    return Decomposition(frames, pd.DataFrame(Y, index=index, columns=list(variables)) - base,
                         pd.DataFrame(shocks, index=index, columns=list(shock_names)), title=title, measure=measure, baseline=base)


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise SpecificationError(f"alpha is between 0 and 1, e.g. 0.05; got {alpha}")


# panel VARs


@dataclass(frozen=True, eq=False)
class PanelDecomposition:
    """A panel VAR's historical decomposition: one ``Decomposition`` per entity, each adding up to that entity's data.
    ``hd[entity]`` is one entity's; ``select(entity)`` the same result with its views on another entity."""

    decompositions: dict[Any, Decomposition]
    entity_axis: str
    entity: Any  # the entity the views show
    fixed_effects: pd.DataFrame  # entities x variables: each entity's mu_i
    title: str = "Historical decomposition"
    coverage: float | None = None
    band_source: str = ""

    @property
    def entities(self) -> list[Any]:
        return list(self.decompositions)

    @property
    def has_bands(self) -> bool:
        return self.coverage is not None

    def _key(self, entity: Any) -> Any:
        if entity is None:
            return self.entity
        if entity in self.decompositions:
            return entity
        found = [k for k in self.decompositions if str(k) == str(entity)]
        if not found:
            shown = ", ".join(map(str, self.entities[:8])) + (", ..." if len(self.entities) > 8 else "")
            raise NotFoundError(f"no entity {entity!r} in this decomposition; entities: {shown}")
        return found[0]

    def __getitem__(self, entity: Any) -> Decomposition:
        return self.decompositions[self._key(entity)]

    def select(self, entity: Any) -> PanelDecomposition:
        """The same decomposition with its views (graph, tables, bands) on ``entity``."""
        return replace(self, entity=self._key(entity))

    @property
    def selected(self) -> Decomposition:
        return self.decompositions[self.entity]

    @property
    def contributions(self) -> dict[str, pd.DataFrame]:
        """The selected entity's contributions, by variable."""
        return self.selected.contributions

    def table(self, variable: str, entity: Any = None) -> Table:
        return self[self._key(entity)].table(variable)

    def plot(self, variables: Sequence[str] | None = None, entity: Any = None) -> Any:
        return self[self._key(entity)].plot(variables)

    def bands_table(self, variable: str, entity: Any = None) -> Table:
        return self[self._key(entity)].bands_table(variable)

    def plot_bands(self, variables: Sequence[str] | None = None, entity: Any = None) -> Any:
        return self[self._key(entity)].plot_bands(variables)

    def summary(self) -> Table:
        """Each entity: its sample, its fixed effects, and the largest absolute contribution of any shock."""
        rows, index = [], []
        variables = list(self.fixed_effects.columns)
        for key, d in self.decompositions.items():
            periods = d.deviations.index
            biggest = max(float(np.abs(frame.to_numpy()).max()) for frame in d.contributions.values())
            rows.append([str(periods[0]), str(periods[-1]), len(periods), *[float(self.fixed_effects.loc[key, v]) for v in variables], biggest])
            index.append(str(key))
        notes = [f"Views show entity {self.entity}; choose another with select(entity), or hd[entity] for its decomposition.",
                 "Fixed effect: the entity's mean of y(t) less the lags' effect; the baseline starts from its first lags with it.",
                 "Each entity's contributions and baseline add up to its data."]
        if self.has_bands:
            notes.append(f"{self.coverage:.0%} bands: pointwise quantiles over {self.band_source}")
        return Table(["first", "last", "periods", *[f"mu {v}" for v in variables], "max |contribution|"], rows, index=index,
                     index_name=self.entity_axis, title=f"{self.title}: {len(self.decompositions)} entities", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        """Every entity's contributions, long: entity, period, variable, part, value (``view``: one of its views)."""
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        parts = []
        for key, d in self.decompositions.items():
            for v, frame in d.contributions.items():
                long = frame.rename_axis("period").reset_index().melt(id_vars="period", var_name="part")
                long.insert(0, "variable", v)
                long.insert(0, self.entity_axis, key)
                parts.append(long)
        return pd.concat(parts, ignore_index=True)

    def __str__(self) -> str:
        return str(self.summary())

    def __repr__(self) -> str:
        bands = f", bands from {self.band_source}" if self.has_bands else ""
        return f"<PanelDecomposition ({self.title}): {len(self.decompositions)} entities, showing {self.entity}{bands}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _panel_arrays(fit: FittedPanelVAR) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Each entity's (key, Y, X): the variables and their lags as arrays, by entity in sorted order."""
    p = fit.lags
    lag_cols = [f"{v}[-{lag}]" for lag in range(1, p + 1) for v in fit.variables]
    return [(key, g[list(fit.variables)].to_numpy(float), g[lag_cols].to_numpy(float))
            for key, g in fit.levels.groupby(level=fit.spec.by, sort=True)]


def _panel_parts(fit: FittedPanelVAR, B: np.ndarray, P: np.ndarray,
                 arrays: list[tuple[Any, np.ndarray, np.ndarray]] | None = None) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """For each entity under coefficients B (K p, K) and impact P: (contributions, baseline, shocks, fixed effect).
    ``arrays`` (``_panel_arrays``) saves splitting the data by entity again at each draw of the coefficients."""
    from .var import _lag_matrices

    K, p = len(fit.variables), fit.lags
    A = _lag_matrices(B, 0, K, p)
    out = {}
    for key, Y, X in (arrays if arrays is not None else _panel_arrays(fit)):
        mu = (Y - X @ B).mean(axis=0)
        U = Y - mu - X @ B
        e = np.linalg.solve(P, U.T).T
        contributions = propagate(A, P[None, :, :] * e[:, None, :])
        out[key] = (contributions, _baseline(A, mu, X[0], len(Y)), e, mu)
    return out


def panel_historical(fit: FittedPanelVAR, draws: int = 0, alpha: float = 0.05, seed: int = 0, entity: Any = None) -> PanelDecomposition:
    from .var import Cholesky, _impact, _lag_matrices, _scheme_label

    if fit.levels is None:
        raise SpecificationError("this panel VAR was fitted before historical decompositions kept its data: fit it again")
    K, p = len(fit.variables), fit.lags
    scheme = fit.identification or Cholesky()
    P = fit.impact()
    if P.shape != (K, K):
        raise SpecificationError(f"a historical decomposition needs as many shocks as variables ({K}); the identification gives {P.shape[1]}")
    names = getattr(scheme, "shock_names", None)  # as a VAR names them: the variables', or the sign restrictions' names
    shocks = list(names(fit.variables)) if callable(names) else list(fit.variables)
    label = _scheme_label(scheme)
    arrays = _panel_arrays(fit)
    point = _panel_parts(fit, fit.B, P, arrays)
    time = [n for n in fit.levels.index.names if n != fit.spec.by][0]
    axis = fit.spec.by
    decompositions = {}
    for key, (c, base, e, _) in point.items():
        rows = fit.levels.xs(key, level=axis)
        index = pd.PeriodIndex(rows.index.get_level_values(time) if isinstance(rows.index, pd.MultiIndex) else rows.index)
        decompositions[key] = _decomposition(c, rows[list(fit.variables)].to_numpy(float), base, e, index, fit.variables, shocks, shocks,
                                             f"Historical decomposition ({label}): {axis} {key}",
                                             "relative to the baseline: the entity's fixed effect and initial conditions")
    effects = pd.DataFrame([v[3] for v in point.values()], index=pd.Index(list(point), name=axis), columns=list(fit.variables))
    chosen = entity if entity is not None else next(iter(decompositions))
    result = PanelDecomposition(decompositions, axis, None, effects, f"Historical decomposition ({label})")
    result = result.select(chosen)
    if not draws:
        return result
    _check_alpha(alpha)
    rng = make_rng(seed)  # the coefficient draws of FittedPanelVAR.irf, in the same order
    chol = np.linalg.cholesky(fit.cov + 1e-14 * np.eye(len(fit.cov)))
    b = fit.B.T.reshape(-1)
    samples: dict[Any, list[np.ndarray]] = {key: [] for key in point}
    for _ in range(int(draws)):
        Bn = (b + chol @ rng.standard_normal(len(b))).reshape(K, -1).T
        Pn = _impact(scheme, fit.sigma, _lag_matrices(Bn, 0, K, p), fit.variables)
        for key, (c, *_rest) in _panel_parts(fit, Bn, Pn, arrays).items():
            samples[key].append(c)
    decompositions = {}
    for key, d in result.decompositions.items():
        lower, median, upper = _bands(np.stack(samples[key]), alpha, d.deviations.index, fit.variables, shocks)
        decompositions[key] = replace(d, lower=lower, median=median, upper=upper, coverage=1 - alpha,
                                      band_source=f"{int(draws)} draws of the coefficients")
    return replace(result, decompositions=decompositions, coverage=1 - alpha, band_source=f"{int(draws)} draws of the coefficients")


# proxy SVARs


def _proxy_parts(var: Any, A: np.ndarray, Bd: np.ndarray, sigma: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Under (A, Bd) and impact column b: contributions (periods, variable, [identified, other]), the baseline and
    the identified shock."""
    U, baseline = var._hd_parts(A, Bd)
    w = np.linalg.solve(sigma, b)
    e = U @ w / (b @ w)
    own = np.outer(e, b)
    return propagate(A, np.stack([own, U - own], axis=2)), baseline, e


def proxy_historical(psvar: ProxySVAR, draws: int = 0, alpha: float = 0.05, seed: int = 0) -> Decomposition:
    from .proxy import bootstrap_draws

    var = psvar.var
    variables = list(var.variables)
    a = var.arrays
    d = a.D.shape[1]
    b = psvar.impact.to_numpy(float)
    shock = f"{psvar.normalise_on} shock ({psvar.instrument})"
    parts = [shock, OTHER]
    c, baseline, e = _proxy_parts(var, var.coefs, var.estimate.B[:d], var.estimate.sigma, b)
    result = _decomposition(c, a.Y, baseline, e[:, None], a.index, variables, parts, [shock],
                            f"Historical decomposition (proxy SVAR, {psvar.instrument})",
                            "relative to the baseline: deterministic terms and initial conditions")
    if not draws:
        return result
    _check_alpha(alpha)
    if psvar.z is None:
        raise SpecificationError("this proxy SVAR was made before it kept its proxy for the bootstrap: run proxy_svar again")
    i = variables.index(psvar.normalise_on)
    samples = [_proxy_parts(var, A, Bd, sigma, bs)[0] for A, Bd, sigma, bs in bootstrap_draws(var, psvar.z, i, psvar.scale, int(draws), psvar.block, seed)]
    if not samples:
        raise SpecificationError("no bootstrap draw identified the shock: the proxy is too weak for bands")
    lower, median, upper = _bands(np.stack(samples), alpha, a.index, variables, parts)
    return replace(result, lower=lower, median=median, upper=upper, coverage=1 - alpha,
                   band_source=f"{len(samples)} moving-block bootstrap draws (Jentsch and Lunsford)")
