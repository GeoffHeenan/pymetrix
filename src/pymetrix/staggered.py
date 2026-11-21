"""Difference-in-differences with staggered adoption: Callaway and Sant'Anna's group-time effects, Sun and Abraham's
interaction-weighted event study, and Goodman-Bacon's decomposition of the two-way fixed effects estimate.

    cs = staggered_did(panel, "y", "d")                              # ATT(g, t), and by event time, cohort and overall
    cs.event_study(); cs.overall(); cs.plot()
    staggered_did(panel, "y", "d", method="sun_abraham")             # interaction-weighted event study
    bacon_decomposition(panel, "y", "d")                             # what two-way fixed effects averages, and with what weights

Units adopt the treatment at different times and keep it (``treatment`` is 0 before adoption, 1 from it; a cohort
is the units that adopt in the same period). With effects that change over time or differ by cohort, the two-way
fixed effects coefficient averages comparisons that use already-treated units as controls, with weights that can be
negative (Goodman-Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020). The estimators here compare each cohort only
with units not (yet) treated.

**Callaway and Sant'Anna (2021),** without covariates: ATT(g, t) is the change in the outcome from g - 1 (the period
before adoption) to t for cohort g, less the same change for the control units, the never-treated
(``control="never"``) or those not treated by t (``"notyet"``). Pre-adoption ATT(g, t), t < g, test parallel trends
(with the same base period g - 1). They are aggregated by event time e = t - g (weights: cohort sizes), by cohort, and
into an overall ATT (the post-adoption ATT(g, t), weighted by cohort size). Standard errors and bands come from a
bootstrap over units (``draws``).

With ``covariates`` (each unit's values in the first period), parallel trends need hold only among units alike in
them: ``estimator="or"`` subtracts the controls' change predicted from the covariates (outcome regression), ``"ipw"``
reweights the controls by the odds of their propensity to be in the cohort, and ``"dr"`` (the default) does both,
which is consistent if either the outcome regression or the logit propensity is right (Sant'Anna and Zhao 2020).

**Sun and Abraham (2021):** a regression of the outcome on unit and period fixed effects and the cohort-by-event-time
indicators (e = -1 left out), with never-treated (or, with none, the last-treated) units as controls; each event
time's estimate is the average of the cohorts' coefficients weighted by their shares. Errors are clustered by unit.
In a balanced panel without covariates it equals Callaway and Sant'Anna's with never-treated controls.

**Goodman-Bacon (2021):** the two-way fixed effects coefficient is an exact weighted average of 2x2 comparisons:
each cohort against the never-treated, each earlier cohort against a later one before the later adopts, and each
later cohort against an earlier one after the earlier has adopted (the comparison that misleads when effects grow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure

__all__ = ["BaconDecomposition", "StaggeredDiD", "bacon_decomposition", "staggered_did"]


def _wide(data: Dataset, outcome: str, treatment: str) -> tuple[pd.DataFrame, pd.Series, list[Any], str, str]:
    """The outcome as units by periods, each unit's cohort (its first treated period's position; inf if never), the
    periods, and the axis names. The panel must be balanced and the treatment absorbing."""
    if data.kind != "panel":
        raise SpecificationError("staggered difference-in-differences needs a panel (units by periods)")
    unit, time = data.entity_axes[0].name, data.time_axis.name
    for name in (outcome, treatment):
        if name not in data.names:
            raise SpecificationError(f"no series {name!r} in the dataset")
    frame = data.frame[[outcome, treatment]].dropna()
    Y = frame[outcome].unstack(time)
    D = frame[treatment].unstack(time)
    if Y.isna().any().any():
        raise SpecificationError("the panel must be balanced: every unit observed in every period")
    Dv = D.to_numpy(float)
    if not np.isin(Dv, (0.0, 1.0)).all():
        raise SpecificationError(f"{treatment} must be 0 or 1")
    if (np.diff(Dv, axis=1) < 0).any():
        raise SpecificationError(f"{treatment} must stay 1 once a unit is treated (staggered adoption)")
    periods = list(Y.columns)
    first = np.where(Dv.any(axis=1), Dv.argmax(axis=1), np.inf)
    if (first == 0).any():
        raise SpecificationError("some units are treated in the first period: they have no pre-treatment period")
    return Y, pd.Series(first, index=Y.index), periods, unit, time


def _logit(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Fitted probabilities of a logit of d on X (Newton's method; a tiny ridge keeps separated data finite)."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (d - p) - 1e-8 * b
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(Hm, g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def _conditional(dy_t: np.ndarray, dy_c: np.ndarray, X_t: np.ndarray, X_c: np.ndarray, estimator: str) -> float:
    """ATT from the treated's and controls' changes and covariates (with a constant): outcome regression, inverse
    probability weighting, or doubly robust (Sant'Anna and Zhao 2020, panel data)."""
    m, *_ = np.linalg.lstsq(X_c, dy_c, rcond=None)
    if estimator == "or":
        return float(np.mean(dy_t - X_t @ m))
    X = np.vstack([X_t, X_c])
    d = np.concatenate([np.ones(len(X_t)), np.zeros(len(X_c))])
    dy = np.concatenate([dy_t, dy_c])
    p = _logit(X, d)
    w1 = d / d.mean()
    w0 = p * (1 - d) / (1 - p)
    w0 = w0 / w0.mean()
    if estimator == "ipw":
        return float(np.mean(w1 * dy) - np.mean(w0 * dy))
    resid = dy - X @ m
    return float(np.mean(w1 * resid) - np.mean(w0 * resid))


def _att_gt(Y: np.ndarray, G: np.ndarray, control: str, Xc: np.ndarray | None = None,
            estimator: str = "dr") -> dict[tuple[int, int], tuple[float, float]]:
    """ATT(g, t) and the cohort's size, by (g, t), with base period g - 1; with covariates ``Xc`` (units by
    covariates, a constant added), conditional on them."""
    out = {}
    T = Y.shape[1]
    for g in np.unique(G[np.isfinite(G)]).astype(int):
        treated = G == g
        for t in range(T):
            if t == g - 1:
                continue
            ctrl = ~np.isfinite(G) if control == "never" else (G > max(t, g - 1)) & (G != g)
            if not ctrl.any():
                continue
            d_t = Y[treated, t] - Y[treated, g - 1]
            d_c = Y[ctrl, t] - Y[ctrl, g - 1]
            if Xc is None:
                value = float(d_t.mean() - d_c.mean())
            else:
                X_t = np.column_stack([np.ones(treated.sum()), Xc[treated]])
                X_c = np.column_stack([np.ones(ctrl.sum()), Xc[ctrl]])
                value = _conditional(d_t, d_c, X_t, X_c, estimator)
            out[(g, t)] = (value, float(treated.sum()))
    return out


def _aggregate(att: dict[tuple[int, int], tuple[float, float]]) -> tuple[dict[int, float], dict[int, float], float]:
    by_e: dict[int, list[tuple[float, float]]] = {}
    by_g: dict[int, list[float]] = {}
    post = []
    for (g, t), (v, n) in att.items():
        by_e.setdefault(t - g, []).append((v, n))
        if t >= g:
            by_g.setdefault(g, []).append(v)
            post.append((v, n))
    event = {e: float(np.average([v for v, _ in vs], weights=[n for _, n in vs])) for e, vs in sorted(by_e.items())}
    cohort = {g: float(np.mean(vs)) for g, vs in sorted(by_g.items())}
    overall = float(np.average([v for v, _ in post], weights=[n for _, n in post])) if post else np.nan
    return event, cohort, overall


@dataclass(frozen=True, eq=False)
class StaggeredDiD:
    """Group-time or interaction-weighted effects of a staggered treatment, by event time, cohort and overall."""

    outcome: str
    treatment: str
    method: str
    control: str
    periods: list[Any] = field(repr=False)
    event: pd.DataFrame = field(repr=False)  # by event time: estimate, se, lower, upper
    cohorts: pd.DataFrame = field(repr=False)  # by cohort (its first treated period)
    att: pd.DataFrame = field(repr=False)  # ATT(g, t) or the cohort-by-event coefficients
    total: tuple[float, float] = (np.nan, np.nan)
    units: int = 0

    stage = "estimate"

    def event_study(self) -> pd.DataFrame:
        return self.event

    def overall(self) -> dict[str, float]:
        est, se = self.total
        return {"ATT": est, "se": se, "lower": est - 1.96 * se, "upper": est + 1.96 * se,
                "p-value": float(2 * sps.norm.sf(abs(est / se))) if se > 0 else np.nan}

    def summary(self) -> Table:
        e = self.event
        rows = [[float(r.estimate), float(r.se), float(r.lower), float(r.upper)] for r in e.itertuples()]
        o = self.overall()
        rows.append([o["ATT"], o["se"], o["lower"], o["upper"]])
        how = ("Callaway and Sant'Anna, base period g - 1, " + ("never-treated" if self.control == "never" else "not-yet-treated")
               + " controls; bootstrap over units" if self.method == "callaway_santanna"
               else "Sun and Abraham, interaction-weighted, errors clustered by unit")
        return Table(["estimate", "std err", "lower 95%", "upper 95%"], rows, index=[f"e = {i}" for i in e.index] + ["overall (post)"],
                     index_name="event time", title=f"Staggered DiD: {self.treatment} on {self.outcome}",
                     notes=(f"{how}; {self.units} units, {len(self.cohorts)} cohorts.",
                            "Event time 0 is the period of adoption; negative event times test parallel trends."))

    def plot(self) -> Figure:
        from .figures import Band, Figure, Line, Panel, Rule

        e = self.event
        return Figure((Panel((Band(e["lower"], e["upper"], "95% band"), Line(e["estimate"], "effect"), Rule(0.0)),
                             title=f"Effect of {self.treatment} on {self.outcome} by event time", xlabel="periods since adoption"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.event

    def __repr__(self) -> str:
        return f"<StaggeredDiD {self.method}: overall ATT {self.total[0]:.4g}>"


def _sun_abraham(Y: pd.DataFrame, G: np.ndarray, window: tuple[int, int] | None) -> tuple[dict[int, tuple[float, float]], pd.DataFrame, tuple[float, float], dict[int, float]]:
    N, T = Y.shape
    finite = np.isfinite(G)
    if (~finite).any():
        ctrl_mask = ~finite
    else:  # no never-treated: the last-treated cohort is the control, and its treated periods are dropped
        ctrl_mask = G == G.max()
    cohorts = sorted(np.unique(G[finite & ~ctrl_mask]).astype(int))
    rows, cols = [], []
    keep_t = range(T) if (~finite).any() else range(int(G.max()))
    names = []
    for g in cohorts:
        for e in range(-g, T - g):
            if e == -1 or (window and not window[0] <= e <= window[1]) or g + e not in keep_t:
                continue
            names.append((g, e))
    y, Xd, unit_ids, time_ids = [], [], [], []
    for i in range(N):
        for t in keep_t:
            if not (ctrl_mask[i] or G[i] in cohorts):
                continue
            y.append(Y.iat[i, t])
            Xd.append([1.0 if (G[i] == g and t - g == e) else 0.0 for g, e in names])
            unit_ids.append(i)
            time_ids.append(t)
    y, X = np.asarray(y), np.asarray(Xd)
    u, tt = np.asarray(unit_ids), np.asarray(time_ids)

    def demean(M: np.ndarray) -> np.ndarray:  # two-way within transform (balanced: exact in one pass of alternating means)
        M = M.astype(float).copy()
        for _ in range(200):
            prev = M.copy()
            M -= pd.DataFrame(M).groupby(u).transform("mean").to_numpy()
            M -= pd.DataFrame(M).groupby(tt).transform("mean").to_numpy()
            if np.max(np.abs(M - prev)) < 1e-12:
                break
        return M

    Xt, yt = demean(X), demean(y[:, None])[:, 0]
    beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    e_res = yt - Xt @ beta
    XtXi = np.linalg.pinv(Xt.T @ Xt)
    meat = np.zeros((len(names), len(names)))
    for i in np.unique(u):
        s = Xt[u == i].T @ e_res[u == i]
        meat += np.outer(s, s)
    cov = XtXi @ meat @ XtXi
    sizes = {g: float((G == g).sum()) for g in cohorts}
    coefs = pd.DataFrame([{"cohort": g, "event": e, "estimate": b, "se": float(np.sqrt(cov[k, k]))} for k, ((g, e), b) in enumerate(zip(names, beta))])
    event = {}
    for e in sorted({e for _, e in names}):
        idx = [k for k, (g, ee) in enumerate(names) if ee == e]
        w = np.array([sizes[names[k][0]] for k in idx])
        w = w / w.sum()
        est = float(w @ beta[idx])
        event[e] = (est, float(np.sqrt(w @ cov[np.ix_(idx, idx)] @ w)))
    post = [k for k, (g, e) in enumerate(names) if e >= 0]
    wp = np.array([sizes[names[k][0]] for k in post])
    wp = wp / wp.sum()
    total = (float(wp @ beta[post]), float(np.sqrt(wp @ cov[np.ix_(post, post)] @ wp)))
    cohort = {g: float(np.mean([beta[k] for k in post if names[k][0] == g])) for g in cohorts}
    return event, coefs, total, cohort


@api("pymetrix:staggered_did")
def staggered_did(data: Dataset, outcome: str, treatment: str, method: Literal["callaway_santanna", "sun_abraham"] = "callaway_santanna",
                  control: Literal["never", "notyet"] = "never", window: tuple[int, int] | None = None, draws: int = 499,
                  seed: int = 0, covariates: Any = (), estimator: Literal["dr", "or", "ipw"] = "dr") -> StaggeredDiD:
    """Effects of a staggered treatment by event time, cohort and overall: Callaway and Sant'Anna's group-time
    effects, or Sun and Abraham's interaction-weighted event study (see the module notes). ``covariates`` (each unit's
    values in the first period) make parallel trends conditional on them, by ``estimator``: doubly robust, outcome
    regression or inverse probability weighting."""
    Y, first, periods, unit, time = _wide(data, outcome, treatment)
    G = first.to_numpy()
    Yv = Y.to_numpy(float)
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    Xc = None
    if covariates:
        if method != "callaway_santanna":
            raise SpecificationError("covariates are for Callaway and Sant'Anna's estimator")
        if estimator not in ("dr", "or", "ipw"):
            raise SpecificationError("estimator is 'dr' (doubly robust), 'or' (outcome regression) or 'ipw'")
        missing = [c for c in covariates if c not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        Xc = np.column_stack([data.frame[c].unstack(time).reindex(Y.index).iloc[:, 0].to_numpy(float) for c in covariates])
        if not np.isfinite(Xc).all():
            raise SpecificationError("the covariates must be observed in the first period for every unit")
    if method not in ("callaway_santanna", "sun_abraham"):
        raise SpecificationError("method is 'callaway_santanna' or 'sun_abraham'")
    if control not in ("never", "notyet"):
        raise SpecificationError("control is 'never' or 'notyet'")
    if method == "callaway_santanna":
        if control == "never" and np.isfinite(G).all():
            raise SpecificationError("no unit is never treated: use control='notyet'")
        att = _att_gt(Yv, G, control, Xc, estimator)
        if not att:
            raise SpecificationError("no cohort has a control group")
        event, cohort, overall = _aggregate(att)
        rng = make_rng(seed)
        boot_e: dict[int, list[float]] = {e: [] for e in event}
        boot_o = []
        for _ in range(int(draws)):
            pick = rng.integers(0, len(G), len(G))
            a = _att_gt(Yv[pick], G[pick], control, None if Xc is None else Xc[pick], estimator)
            if not a:
                continue
            be, _, bo = _aggregate(a)
            for e in event:
                if e in be:
                    boot_e[e].append(be[e])
            boot_o.append(bo)
        se_e = {e: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan for e, v in boot_e.items()}
        total = (overall, float(np.std(boot_o, ddof=1)) if len(boot_o) > 1 else np.nan)
        rows = [{"event time": e, "estimate": v, "se": se_e[e]} for e, v in event.items()]
        att_frame = pd.DataFrame([{"cohort": periods[g], "period": periods[t], "event": t - g, "ATT": v, "cohort size": n}
                                  for (g, t), (v, n) in sorted(att.items())])
    else:
        ev, att_frame, total, cohort = _sun_abraham(Y, G, window)
        rows = [{"event time": e, "estimate": v, "se": s} for e, (v, s) in ev.items()]
        att_frame["cohort"] = [periods[int(g)] for g in att_frame["cohort"]]
    event_frame = pd.DataFrame(rows).set_index("event time")
    if window and method == "callaway_santanna":
        event_frame = event_frame.loc[window[0]:window[1]]
    event_frame["lower"] = event_frame["estimate"] - 1.96 * event_frame["se"]
    event_frame["upper"] = event_frame["estimate"] + 1.96 * event_frame["se"]
    cohorts = pd.DataFrame({"ATT": cohort}).rename(index=lambda g: periods[int(g)])
    cohorts.index.name = "cohort"
    return StaggeredDiD(outcome, treatment, method, control, periods, event_frame, cohorts, att_frame, total, len(G))


@dataclass(frozen=True, eq=False)
class BaconDecomposition:
    """The two-way fixed effects estimate as a weighted average of its 2x2 comparisons (Goodman-Bacon 2021)."""

    outcome: str
    treatment: str
    twfe: float
    comparisons: pd.DataFrame  # type, treated, control, estimate, weight

    stage = "estimate"

    def by_type(self) -> pd.DataFrame:
        c = self.comparisons
        g = c.assign(wb=c["weight"] * c["estimate"]).groupby("type")
        return pd.DataFrame({"weight": g["weight"].sum(), "average estimate": g["wb"].sum() / g["weight"].sum()})

    def summary(self) -> Table:
        t = self.by_type()
        rows = [[float(r.weight), float(r._2)] for r in t.itertuples()]
        return Table(["weight", "average estimate"], rows, index=list(t.index), index_name="comparison",
                     title=f"Goodman-Bacon decomposition: {self.treatment} on {self.outcome}",
                     notes=(f"Two-way fixed effects estimate {self.twfe:.6g} = the weighted sum of the {len(self.comparisons)} 2x2 comparisons.",
                            "'Later vs earlier treated' uses already-treated units as controls: biased when effects change over time."))

    def plot(self) -> Figure:
        from .figures import Figure, Panel, Rule, Scatter

        c = self.comparisons
        layers = tuple(Scatter(c.loc[c["type"] == k, "weight"].reset_index(drop=True), c.loc[c["type"] == k, "estimate"].reset_index(drop=True), k)
                       for k in c["type"].unique())
        return Figure((Panel(layers + (Rule(self.twfe, "--"),), title="Each 2x2 comparison: its weight and estimate", xlabel="weight",
                             ylabel="estimate"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.comparisons

    def __repr__(self) -> str:
        return f"<BaconDecomposition: TWFE {self.twfe:.4g} from {len(self.comparisons)} comparisons>"


@api("pymetrix:bacon_decomposition")
def bacon_decomposition(data: Dataset, outcome: str, treatment: str) -> BaconDecomposition:
    """Goodman-Bacon's decomposition of the two-way fixed effects DiD estimate into 2x2 comparisons and their weights."""
    Y, first, periods, _, _ = _wide(data, outcome, treatment)
    G = first.to_numpy()
    Yv = Y.to_numpy(float)
    N, T = Yv.shape
    D = (np.arange(T)[None, :] >= G[:, None]).astype(float)
    Dt = D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()
    twfe = float((Dt * Yv).sum() / (Dt**2).sum())
    VD = float((Dt**2).mean())
    groups = sorted(np.unique(G[np.isfinite(G)]).astype(int))
    n = {g: float((G == g).sum()) / N for g in groups}
    n_u = float((~np.isfinite(G)).sum()) / N
    share = {g: (T - g) / T for g in groups}  # D-bar: the share of periods treated

    def mean(mask: np.ndarray, cols: range) -> float:
        return float(Yv[np.ix_(mask, list(cols))].mean())

    rows = []
    for k in groups:
        mk = G == k
        if n_u > 0:
            mu = ~np.isfinite(G)
            b = (mean(mk, range(k, T)) - mean(mk, range(0, k))) - (mean(mu, range(k, T)) - mean(mu, range(0, k)))
            nku = n[k] / (n[k] + n_u)
            s = (n[k] + n_u) ** 2 * nku * (1 - nku) * share[k] * (1 - share[k]) / VD
            rows.append({"type": "treated vs never treated", "treated": periods[k], "control": "never", "estimate": b, "weight": s})
        for l in groups:
            if l <= k:
                continue
            ml = G == l
            nkl = n[k] / (n[k] + n[l])
            # k (earlier) against l before l adopts: periods [0, l)
            b_k = (mean(mk, range(k, l)) - mean(mk, range(0, k))) - (mean(ml, range(k, l)) - mean(ml, range(0, k)))
            s_k = ((n[k] + n[l]) * (1 - share[l])) ** 2 * nkl * (1 - nkl) * ((share[k] - share[l]) / (1 - share[l])) * ((1 - share[k]) / (1 - share[l])) / VD
            rows.append({"type": "earlier vs later treated", "treated": periods[k], "control": periods[l], "estimate": b_k, "weight": s_k})
            # l (later) against k after k has adopted: periods [k, T)
            b_l = (mean(ml, range(l, T)) - mean(ml, range(k, l))) - (mean(mk, range(l, T)) - mean(mk, range(k, l)))
            s_l = ((n[k] + n[l]) * share[k]) ** 2 * nkl * (1 - nkl) * (share[l] / share[k]) * ((share[k] - share[l]) / share[k]) / VD
            rows.append({"type": "later vs earlier treated", "treated": periods[l], "control": periods[k], "estimate": b_l, "weight": s_l})
    return BaconDecomposition(outcome, treatment, twfe, pd.DataFrame(rows))
