"""Dynamic panel GMM: Arellano-Bond difference GMM and Blundell-Bond system GMM.

    Equation("n ~ n[-1] + w + k").fit(panel, method=ArellanoBond())                 # difference GMM, two-step
    Equation("n ~ n[-1] + w + k").fit(panel, method=ArellanoBond(system=True))      # system GMM
    ArellanoBond(gmm="n, w", lags="2:4", collapse=True)                            # w endogenous; fewer instruments

With a lagged dependent variable and entity effects, least squares is biased up and the
within (fixed-effects) estimator biased down when T is short. Differencing removes the
effects; lags two and earlier of the dependent variable (and of other endogenous variables,
listed in ``gmm``) are then valid instruments for the differenced equation (Arellano and
Bond 1991). System GMM adds the equation in levels, instrumented by lagged differences
(Blundell and Bond 1998), which helps when the series are persistent. Regressors that do not
involve a ``gmm`` variable are treated as strictly exogenous and instrument themselves.

Conventions follow xtabond2 (Roodman 2009): instruments are one column per period and lag
unless ``collapse``; missing instrument values count as zero; the one-step weight matrix
uses the MA(1) structure of differenced errors (and the identity for the levels rows); the
two-step standard errors carry the Windmeijer (2005) correction. Reported: Hansen's J test
of the overidentifying restrictions, and Arellano-Bond tests for first- and second-order
serial correlation in the differenced residuals (AR(1) is expected; AR(2) should not be).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ..errors import SpecificationError
from ..forms import Widget

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation

__all__ = ["ArellanoBond"]


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _lag_range(text: str) -> tuple[int, int | None]:
    first, _, last = text.partition(":")
    lo = int(first) if first.strip() else 2
    hi = int(last) if last.strip() else None
    if lo < 1 or (hi is not None and hi < lo):
        raise SpecificationError(f"lags {text!r}: write first:last with 1 <= first <= last, e.g. 2:4, or 2: for all")
    return lo, hi


@dataclass(frozen=True)
class ArellanoBond:
    """Dynamic panel GMM: difference GMM (Arellano-Bond); ``system=True`` for system GMM (Blundell-Bond).

    The lags are valid instruments only if the errors are serially uncorrelated in levels. Then the differenced
    residuals show first-order correlation but not second-order: a small AR(2) p-value invalidates lag 2 of the
    gmm variables, so start the lags at 3. System GMM also assumes that the differences of the gmm variables are
    uncorrelated with the entity effects (a mean-stationary start). Without ``collapse`` the instrument count grows
    with the square of the number of periods; too many instruments overfit the endogenous variables, pull the
    estimates towards least squares and weaken the Hansen test, whose p-value near 1 is then a warning rather than
    reassurance. Keep the instruments fewer than the groups by collapsing or capping the lags. As a check, the
    coefficient on the lagged dependent variable should lie between its within estimate (biased down) and its
    pooled least-squares one (biased up).
    """

    maturity: ClassVar[str] = "experimental"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation
        from pymetrix.estimation import ArellanoBond

        panel = Dataset.example("wage_panel")                    # 545 men, 1980-1987
        eq = Equation("lwage ~ lwage[-1] + exper + union")
        print(eq.fit(panel, method=ArellanoBond(collapse=True)).summary())  # the notes give AR(1), AR(2) and Hansen J
        print(eq.fit(panel, method=ArellanoBond(lags="3:", collapse=True)).summary())  # lags from 3: what to use when AR(2) rejects
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, z", ("two-step estimates with Windmeijer-corrected standard errors (one-step: robust); the "
                              "Standard errors line says which")),
        ("period[...]", "the period dummies of time_effects, absorbing shocks common to all entities"),
        ("groups, instruments", ("in the note: keep the instrument count below the groups, or the Hansen test loses "
                                 "its power")),
        ("Hansen J", ("df = instruments - parameters; a small p-value rejects the instruments, a p-value near 1 with "
                      "many instruments is a warning, not reassurance")),
        ("AR(1), AR(2)", ("tests on the differenced residuals: AR(1) is expected to reject; a small AR(2) p-value "
                          "invalidates lag 2, so start the lags at 3")),
        ("fit.stats", "hansen_j, hansen_j_pvalue, ar1, ar1_pvalue, ar2, ar2_pvalue, groups and instruments"),
    )

    label: ClassVar[str] = "Arellano-Bond GMM"
    kind: ClassVar[str] = "panel"
    topics: ClassVar[tuple[str, ...]] = ("dynamic-panels", "panel-data")
    references: ClassVar[tuple[str, ...]] = (
        ("Arellano, M. and Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. Review of Economic Studies 58(2), 277-297. doi:10.2307/2297968"),
        ("Blundell, R. and Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. Journal of Econometrics 87(1), 115-143. doi:10.1016/s0304-4076(98)00009-8"),
        ("Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. Journal of Econometrics 126(1), 25-51. doi:10.1016/j.jeconom.2004.02.005"),
    )

    gmm: Annotated[str, Widget("text", help="variables instrumented by their own lags (endogenous); empty = the dependent variable")] = ""
    lags: Annotated[str, Widget("text", help="lags used as instruments, first:last, e.g. 2:4; 2: = all from the second")] = "2:"
    system: Annotated[bool, Widget("checkbox", help="add the levels equation (Blundell-Bond system GMM)")] = False
    two_step: Annotated[bool, Widget("checkbox", help="two-step GMM with Windmeijer-corrected standard errors")] = True
    collapse: Annotated[bool, Widget("checkbox", help="one instrument column per lag, not per period and lag (fewer instruments)")] = False
    time_effects: Annotated[bool, Widget("checkbox", help="period dummies, instrumenting themselves: absorb common shocks that would correlate the errors across entities")] = True

    def __post_init__(self) -> None:
        _lag_range(self.lags)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: str | None = None, weights: str | None = None,
                 missing: str = "drop") -> FittedEquation:
        if weights:
            raise SpecificationError("weights are not supported with Arellano-Bond GMM")
        return _estimate(self, eq, data, span, where)


@dataclass
class _Rows:
    entity: Any
    y: list[float]
    X: list[np.ndarray]
    Z: list[dict[Any, float]]
    t: list[int]
    level: list[bool]


def _estimate(m: ArellanoBond, eq: Equation, data: Dataset, span: Any, where: str | None) -> FittedEquation:
    from ..expr import evaluate
    from .design import as_series, provenance
    from .fitted import FittedEquation
    from .params import point

    if data.kind != "panel" or len(data.entity_axes) != 1 or data.time_axis is None:
        raise SpecificationError("Arellano-Bond GMM needs a panel with one entity axis and a time axis")
    if eq.form != "linear" or eq.fixed_effects or eq.categoricals:
        raise SpecificationError("Arellano-Bond GMM needs a linear-form equation without fe() or C() (differencing removes the effects)")
    target = eq.require_dependent()
    gmm_vars = _names(m.gmm) or [target]
    unknown = [g for g in gmm_vars if g not in data.names]
    if unknown:
        raise SpecificationError(f"gmm names {unknown}, which are not series")
    first, last = _lag_range(m.lags)
    ent, tname = data.entity_axes[0].name, data.time_axis.name
    full = data.complete()
    index = full.frame.index
    entities = index.get_level_values(ent).unique()
    periods = pd.PeriodIndex(index.get_level_values(tname).unique()).sort_values()
    N, T = len(entities), len(periods)

    def grid(values: Any) -> np.ndarray:
        return as_series(values, index).to_numpy(float).reshape(N, T)

    Y = grid(evaluate(eq.lhs, full.lagged))
    terms = list(eq.terms)
    names = [t.label for t in terms]
    X = np.stack([grid(evaluate(t.expr, full.lagged)) for t in terms], axis=2) if terms else np.zeros((N, T, 0))
    W = np.stack([grid(full[g]) for g in gmm_vars], axis=2)
    exo = [j for j, t in enumerate(terms) if not (t.expr.variables() & set(gmm_vars))]
    used = full.sample_mask(span, where).reshape(N, T)
    dummies = list(range(1, T)) if m.time_effects else []  # one per period after the first
    k = len(names) + (1 if m.system else 0) + len(dummies)
    reg_names = names + (["const"] if m.system else []) + [f"period[{periods[s]}]" for s in dummies]

    per_entity: list[_Rows] = []
    for i in range(N):
        rows = _Rows(entities[i], [], [], [], [], [])
        for t in range(1, T):  # the differenced equation
            if not used[i, t]:
                continue
            dy = Y[i, t] - Y[i, t - 1]
            dx = X[i, t] - X[i, t - 1]
            if not (np.isfinite(dy) and np.all(np.isfinite(dx))):
                continue
            regs = np.r_[dx, [0.0] if m.system else [], [float(s == t) - float(s == t - 1) for s in dummies]]
            z: dict[Any, float] = {}
            top = t if last is None else min(t, last)
            for g in range(len(gmm_vars)):
                for lag in range(first, top + 1):
                    s = t - lag
                    if s < 0:
                        break
                    value = W[i, s, g]
                    if np.isfinite(value):
                        z[("d", g, lag) if m.collapse else ("d", g, t, lag)] = value
            for j in exo:
                z[("dx", j)] = dx[j]
            for s in dummies:
                z[("dt", s)] = float(s == t) - float(s == t - 1)
            rows.y.append(dy), rows.X.append(regs), rows.Z.append(z), rows.t.append(t), rows.level.append(False)
        if m.system:
            for t in range(1, T):  # the levels equation
                if not used[i, t] or not (np.isfinite(Y[i, t]) and np.all(np.isfinite(X[i, t]))):
                    continue
                regs = np.r_[X[i, t], [1.0], [float(s == t) for s in dummies]]
                z = {("const",): 1.0}
                s = t - (first - 1)
                for g in range(len(gmm_vars)):
                    if s >= 1 and np.isfinite(W[i, s, g]) and np.isfinite(W[i, s - 1, g]):
                        z[("l", g) if m.collapse else ("l", g, t)] = W[i, s, g] - W[i, s - 1, g]
                for j in exo:
                    z[("x", j)] = X[i, t, j]
                for d in dummies:
                    z[("t", d)] = float(d == t)
                rows.y.append(Y[i, t]), rows.X.append(regs), rows.Z.append(z), rows.t.append(t), rows.level.append(True)
        if rows.y:
            per_entity.append(rows)
    if not per_entity:
        raise SpecificationError("no observations: each entity needs at least two consecutive periods")
    keys = sorted({key for r in per_entity for z in r.Z for key in z}, key=str)
    col = {key: c for c, key in enumerate(keys)}
    blocks = []
    for r in per_entity:
        Zi = np.zeros((len(r.y), len(keys)))
        for row, z in enumerate(r.Z):
            for key, value in z.items():
                Zi[row, col[key]] = value
        blocks.append((np.asarray(r.y), np.vstack(r.X), Zi, np.asarray(r.t), np.asarray(r.level)))
    # drop regressors that never vary (e.g. a dummy for a period with no rows) and collinear dummies
    Xall = np.vstack([b[1] for b in blocks])
    keep = [j for j in range(k) if np.any(Xall[:, j] != 0)]
    q, rmat = np.linalg.qr(Xall[:, keep])
    independent = np.abs(np.diag(rmat)) > 1e-9 * max(1.0, np.abs(np.diag(rmat)).max())
    keep = [j for j, ok in zip(keep, independent) if ok or j < len(names)]
    blocks = [(y, Xi[:, keep], Zi, t, lv) for y, Xi, Zi, t, lv in blocks]
    reg_names = [reg_names[j] for j in keep]
    zkeep = np.flatnonzero(np.any(np.vstack([b[2] for b in blocks]) != 0, axis=0))
    blocks = [(y, Xi, Zi[:, zkeep], t, lv) for y, Xi, Zi, t, lv in blocks]
    k, L = len(reg_names), len(zkeep)
    if L < k:
        raise SpecificationError(f"the equation is underidentified: {L} instruments for {k} coefficients")

    def H(t: np.ndarray, level: np.ndarray) -> np.ndarray:
        h = np.zeros((len(t), len(t)))
        d = ~level
        td = t[d]
        idx = np.flatnonzero(d)
        h[np.ix_(idx, idx)] = 2 * (td[:, None] == td[None, :]) - 1.0 * (np.abs(td[:, None] - td[None, :]) == 1)
        lv = np.flatnonzero(level)
        h[lv, lv] = 1.0
        return h

    ZX = sum(Zi.T @ Xi for _, Xi, Zi, _, _ in blocks)
    Zy = sum(Zi.T @ y for y, _, Zi, _, _ in blocks)
    A1 = np.linalg.pinv(sum(Zi.T @ H(t, lv) @ Zi for _, _, Zi, t, lv in blocks))

    def solve(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = np.linalg.inv(ZX.T @ A @ ZX)
        return M @ ZX.T @ A @ Zy, M

    b1, M1 = solve(A1)
    u1 = [y - Xi @ b1 for y, Xi, _, _, _ in blocks]
    S1 = sum(Zi.T @ np.outer(u, u) @ Zi for (_, _, Zi, _, _), u in zip(blocks, u1))
    V1 = M1 @ ZX.T @ A1 @ S1 @ A1 @ ZX @ M1  # robust one-step
    if m.two_step:
        A2 = np.linalg.pinv(S1)
        b, M2 = solve(A2)
        u = [y - Xi @ b for y, Xi, _, _, _ in blocks]
        g2 = sum(Zi.T @ ui for (_, _, Zi, _, _), ui in zip(blocks, u))
        D = np.zeros((k, k))
        for j in range(k):  # Windmeijer (2005): how the weight matrix moves with the one-step estimates
            dS = -sum(Zi.T @ (np.outer(Xi[:, j], u1i) + np.outer(u1i, Xi[:, j])) @ Zi for (_, Xi, Zi, _, _), u1i in zip(blocks, u1))
            D[:, j] = -M2 @ ZX.T @ A2 @ dS @ A2 @ g2
        V = M2 + D @ M2 + M2 @ D.T + D @ V1 @ D.T
        A, M_final = A2, M2
    else:
        b, u, V, A, M_final = b1, u1, V1, A1, M1
    S = sum(Zi.T @ np.outer(ui, ui) @ Zi for (_, _, Zi, _, _), ui in zip(blocks, u))
    g = sum(Zi.T @ ui for (_, _, Zi, _, _), ui in zip(blocks, u))
    J = float(g @ np.linalg.pinv(S) @ g)
    df_j = L - k

    def ar_test(order: int) -> tuple[float, float]:
        num = 0.0
        term1 = 0.0
        wX = np.zeros(k)
        c = np.zeros(L)
        for (_, Xi, Zi, t, lv), ui in zip(blocks, u):
            w = np.zeros(len(ui))
            diff_rows = {tt: r for r, (tt, level) in enumerate(zip(t, lv)) if not level}
            for tt, r in diff_rows.items():
                if tt - order in diff_rows:
                    w[r] = ui[diff_rows[tt - order]]
            num += w @ ui
            term1 += (w @ ui) ** 2
            wX += w @ Xi
            c += Zi.T @ ui * (ui @ w)
        var = term1 - 2 * wX @ M_final @ ZX.T @ A @ c + wX @ V @ wX
        if var <= 0:
            return np.nan, np.nan
        z = num / np.sqrt(var)
        return float(z), float(2 * sps.norm.sf(abs(z)))

    ar1, ar1_p = ar_test(1)
    ar2, ar2_p = ar_test(2)
    # residuals of the differenced equation, by entity and period
    resid_rows, resid_idx = [], []
    for (_, _, _, t, lv), ui, rows in zip(blocks, u, per_entity):
        for tt, level, value in zip(t, lv, ui):
            if not level:
                resid_rows.append(value)
                resid_idx.append((rows.entity, periods[tt]))
    resid = pd.Series(resid_rows, index=pd.MultiIndex.from_tuples(resid_idx, names=[ent, tname]), name="resid")
    n_diff = int(sum((~lv).sum() for _, _, _, _, lv in blocks))
    stats = {"nobs": float(n_diff), "groups": float(len(blocks)), "instruments": float(L), "k": float(k),
             "hansen_j": J, "hansen_j_pvalue": float(sps.chi2.sf(J, df_j)) if df_j > 0 else np.nan,
             "ar1": ar1, "ar1_pvalue": ar1_p, "ar2": ar2, "ar2_pvalue": ar2_p}
    kind = "system" if m.system else "difference"
    notes = [f"{'Two' if m.two_step else 'One'}-step {kind} GMM; {len(blocks)} groups, {L} instruments"
             + (" (collapsed)" if m.collapse else "") + f"; GMM-style: {', '.join(gmm_vars)} lags {m.lags}",
             "Standard errors: " + ("Windmeijer-corrected" if m.two_step else "robust"),
             f"Hansen J = {J:.3f} (df {df_j}, p = {stats['hansen_j_pvalue']:.4f})" if df_j > 0 else "Exactly identified: no Hansen test",
             f"Arellano-Bond AR(1) z = {ar1:.3f} (p = {ar1_p:.4f}); AR(2) z = {ar2:.3f} (p = {ar2_p:.4f})"]
    if L > len(blocks):
        notes.append(f"More instruments ({L}) than groups ({len(blocks)}): the Hansen test is weak; consider collapse=True or fewer lags")
    prov = provenance(eq, m, data, resid.index, 0, extra_variables=gmm_vars)
    return FittedEquation(eq, point(reg_names, b, V), m, resid=resid, stats=stats, provenance=prov, extra={"notes": notes})


def _register() -> None:
    from ..plugins import register

    register("methods", ArellanoBond, name="arellano_bond")
    register("methods", ArellanoBond, name="blundell_bond", label="Blundell-Bond system GMM", preset={"system": True})


_register()
