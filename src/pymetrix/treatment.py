"""Treatment effects and policy evaluation: difference-in-differences, event studies, regression discontinuity,
weighting on the propensity score, and synthetic control.

    did(panel, "y", "d")                                      # two-way fixed effects, errors clustered by unit
    event_study(panel, "y", "d", window=(-5, 5))              # effects by period relative to adoption
    rdd(data, "y", "score", cutoff=0)                         # local linear at the cutoff, bandwidth by cross-validation
    treatment_effect(data, "y", "treated", ["age", "educ"])   # inverse probability weighting (or method="aipw")
    synthetic_control(panel, "y", treated=1, start=2005)      # a weighted combination of untreated units

Each gives a result with a summary, a figure and its tables, in a window in the app.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure


def _names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.replace("+", ",").split(",") if s.strip()]
    return list(value)


def _panel_axes(data: Dataset, what: str) -> tuple[str, str]:
    if data.kind != "panel" or data.time_axis is None or not data.entity_axes:
        raise SpecificationError(f"{what} needs a panel: units and periods")
    return data.entity_axes[0].name, data.time_axis.name


def _check(data: Dataset, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in data.names]
    if missing:
        raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")


def _estimate_row(value: float, se: float, alpha: float = 0.05) -> list[float]:
    z = value / se if se > 0 else np.nan
    q = sps.norm.ppf(1 - alpha / 2)
    return [value, se, z, float(2 * sps.norm.sf(abs(z))), value - q * se, value + q * se]


_EST_COLUMNS = ["estimate", "std err", "z", "p-value", "[0.025", "0.975]"]


# -- difference-in-differences ---------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DiD:
    """A difference-in-differences estimate by two-way fixed effects."""

    fit: Any = field(repr=False)  # the FittedEquation
    outcome: str
    treatment: str
    means: pd.DataFrame | None  # ever-treated and never-treated, before and after (one adoption date only)
    by_time: pd.DataFrame | None = field(default=None, repr=False)  # the two groups' mean outcome in each period

    @property
    def estimate(self) -> float:
        return float(self.fit.params[self.treatment])

    @property
    def se(self) -> float:
        return float(self.fit.params.se[list(self.fit.params.names).index(self.treatment)])

    def summary(self) -> Table:
        notes = [f"{self.outcome} on {self.treatment} with unit and period fixed effects; standard errors clustered by unit.",
                 f"{int(self.fit.stats['nobs'])} observations."]
        if self.means is not None:
            m = self.means
            notes.append(f"With one adoption date, the estimate is the difference in differences of the means: "
                         f"({m.loc['treated', 'after']:.4g} - {m.loc['treated', 'before']:.4g}) - "
                         f"({m.loc['control', 'after']:.4g} - {m.loc['control', 'before']:.4g}) in a balanced panel without controls.")
        return Table(_EST_COLUMNS, [_estimate_row(self.estimate, self.se)], index=[self.treatment], index_name="effect",
                     title="Difference-in-differences", notes=tuple(notes))

    def plot(self) -> Figure:
        from .figures import Figure, Line, Panel

        frame = self.by_time
        if frame is None:
            return Figure((Panel((), title="Difference-in-differences"),))
        lines = tuple(Line(frame[c], c) for c in frame.columns)
        return Figure((Panel(lines, title=f"Mean {self.outcome}: ever treated and never treated"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<DiD {self.treatment} on {self.outcome}: {self.estimate:.4g} (s.e. {self.se:.3g})>"


@api("pymetrix:did")
def did(data: Dataset, outcome: str, treatment: str, controls: Sequence[str] = ()) -> DiD:
    """Difference-in-differences: ``outcome`` on the 0/1 ``treatment`` (1 where the policy is in force) with unit and
    period fixed effects, errors clustered by unit."""
    from .estimation import OLS, Cluster, Equation

    unit, time = _panel_axes(data, "difference-in-differences")
    controls = _names(controls)
    _check(data, [outcome, treatment, *controls])
    d = data[treatment]
    if not np.isin(d.dropna().to_numpy(), (0.0, 1.0)).all():
        raise SpecificationError(f"{treatment} must be 0 or 1 (1 where the policy is in force)")
    formula = f"{outcome} ~ {' + '.join([treatment, *controls])} + fe({unit}) + fe({time})"
    fit = Equation(formula).fit(data, method=OLS(cov=Cluster(unit)))
    frame = data.frame[[outcome, treatment]].dropna()
    times = frame.index.get_level_values(time)
    ever = frame[treatment].groupby(level=unit).transform("max").to_numpy() == 1
    by_time = frame[outcome].groupby([times, np.where(ever, "ever treated", "never treated")]).mean().unstack(1)
    starts = frame[frame[treatment] == 1].reset_index().groupby(unit)[time].min()
    means = None
    if starts.nunique() == 1 and (~ever).any():  # one adoption date: the two-by-two table of means
        after = np.asarray(times >= starts.iloc[0])
        means = frame[outcome].groupby([np.where(ever, "treated", "control"), np.where(after, "after", "before")]).mean().unstack(1)
        means = means[["before", "after"]]
    return DiD(fit, outcome, treatment, means, by_time)


# -- event study ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class EventStudy:
    """Effects by period relative to adoption, from a two-way fixed effects regression on leads and lags."""

    fit: Any = field(repr=False)
    outcome: str
    relative: tuple[int, ...]  # the relative periods estimated
    reference: int

    def coefficients(self) -> pd.DataFrame:
        names = list(self.fit.params.names)
        rows = {r: (0.0, 0.0) for r in [self.reference]}
        for r in self.relative:
            j = names.index(_rel_name(r))
            rows[r] = (float(self.fit.params.values[j]), float(self.fit.params.se[j]))
        frame = pd.DataFrame(rows, index=["estimate", "std err"]).T.sort_index()
        frame.index.name = "relative period"
        q = sps.norm.ppf(0.975)
        frame["lower"], frame["upper"] = frame["estimate"] - q * frame["std err"], frame["estimate"] + q * frame["std err"]
        return frame

    def summary(self) -> Table:
        c = self.coefficients()
        rows = [_estimate_row(e, s) if s > 0 else [0.0, np.nan, np.nan, np.nan, 0.0, 0.0] for e, s in zip(c["estimate"], c["std err"])]
        return Table(_EST_COLUMNS, rows, index=[str(r) for r in c.index], index_name="relative period", title="Event study",
                     notes=(f"Effects relative to period {self.reference} (set to zero); the end periods gather everything beyond them. "
                            "Unit and period fixed effects; errors clustered by unit. With adoption at different dates and "
                            "effects that change over time, two-way fixed effects can mix comparisons (Sun and Abraham 2021).",))

    def plot(self) -> Figure:
        from .figures import Band, Figure, Line, Panel, Rule

        c = self.coefficients()
        x = pd.Index(c.index, name="relative period")
        est = pd.Series(c["estimate"].to_numpy(), index=x, name="effect")
        layers = (Band(pd.Series(c["lower"].to_numpy(), index=x), pd.Series(c["upper"].to_numpy(), index=x), "95% interval"),
                  Line(est, "effect"), Rule(0.0), Rule(y=self.reference + 0.5, vertical=True, style="--"))
        return Figure((Panel(layers, title=f"Event study: {self.outcome}", xlabel="periods since adoption"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.coefficients()

    def __repr__(self) -> str:
        return f"<EventStudy of {self.outcome}, periods {min(self.relative + (self.reference,))} to {max(self.relative)}>"


def _rel_name(r: int) -> str:
    return f"rel_m{-r}" if r < 0 else f"rel_{r}"


@api("pymetrix:event_study")
def event_study(data: Dataset, outcome: str, treatment: str, window: tuple[int, int] = (-5, 5), reference: int = -1,
                controls: Sequence[str] = ()) -> EventStudy:
    """Effects by period relative to each unit's first treated period, ``window`` (first, last) periods around it,
    relative to ``reference``; periods beyond the window are gathered at its ends. Never-treated units are the
    comparison."""
    from .estimation import OLS, Cluster, Equation

    unit, time = _panel_axes(data, "an event study")
    lo, hi = int(window[0]), int(window[1])
    if not lo <= reference <= hi or lo >= 0 or hi < 0:
        raise SpecificationError("the window runs from a negative period to a non-negative one, and includes the reference")
    controls = _names(controls)
    _check(data, [outcome, treatment, *controls])
    frame = data.frame
    periods = pd.PeriodIndex(frame.index.get_level_values(time))
    position = pd.Series(periods.asi8 if hasattr(periods, "asi8") else np.arange(len(periods)), index=frame.index)
    first = position.where(frame[treatment] == 1).groupby(level=unit).transform("min")
    rel = (position - first).clip(lo, hi)
    extra = {}
    relative = tuple(r for r in range(lo, hi + 1) if r != reference)
    for r in relative:
        extra[_rel_name(r)] = ((rel == r) & first.notna()).astype(float)
    from .data import Dataset as _D

    new = frame.copy()
    for k, v in extra.items():
        new[k] = v.to_numpy()
    augmented = _D(new, data.axes, data.variables, data.flags, data.meta)
    formula = f"{outcome} ~ {' + '.join([*extra, *controls])} + fe({unit}) + fe({time})"
    fit = Equation(formula).fit(augmented, method=OLS(cov=Cluster(unit)))
    return EventStudy(fit, outcome, relative, reference)


# -- regression discontinuity ----------------------------------------------------------------------------------------

def _local_linear(x: np.ndarray, y: np.ndarray, c: float, h: float) -> tuple[float, float, int, int, np.ndarray]:
    """The jump at c by local linear regression with a triangular kernel: WLS of y on 1, x - c, above, above (x - c)
    within h of c. Returns the jump, its robust (HC1) standard error, the observations each side and the coefficients."""
    s = x - c
    w = np.clip(1 - np.abs(s) / h, 0, None)
    keep = w > 0
    s, yk, wk = s[keep], y[keep], w[keep]
    above = (s >= 0).astype(float)
    n_left, n_right = int((above == 0).sum()), int((above == 1).sum())
    if min(n_left, n_right) < 3:
        raise SpecificationError(f"too few observations within the bandwidth {h:.4g} on one side of the cutoff")
    X = np.column_stack([np.ones(len(s)), s, above, above * s])
    sw = np.sqrt(wk)
    Xw, yw = X * sw[:, None], yk * sw
    coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    u = yw - Xw @ coef
    bread = np.linalg.inv(Xw.T @ Xw)
    n, k = Xw.shape
    V = bread @ ((Xw * u[:, None]).T @ (Xw * u[:, None])) @ bread * n / (n - k)
    return float(coef[2]), float(np.sqrt(V[2, 2])), n_left, n_right, coef


def cv_bandwidth(x: np.ndarray, y: np.ndarray, c: float, grid: np.ndarray, share: float = 0.5) -> tuple[float, np.ndarray]:
    """Imbens and Lemieux's (2008) cross-validation: each observation among the ``share`` nearest the cutoff on its
    side is predicted by a linear fit to the observations within h on its far side from the cutoff (as the boundary
    is estimated); the h with the least mean squared error."""
    left, right = x < c, x >= c
    near_l = left & (x >= np.quantile(x[left], 1 - share))
    near_r = right & (x <= np.quantile(x[right], share))
    scores = np.full(len(grid), np.inf)
    for g, h in enumerate(grid):
        errors = []
        for i in np.flatnonzero(near_l | near_r):
            if left[i]:
                pick = left & (x < x[i]) & (x >= x[i] - h)
            else:
                pick = right & (x > x[i]) & (x <= x[i] + h)
            if pick.sum() < 3:
                continue
            X = np.column_stack([np.ones(pick.sum()), x[pick] - x[i]])
            errors.append(y[i] - np.linalg.lstsq(X, y[pick], rcond=None)[0][0])
        if errors:
            scores[g] = float(np.mean(np.square(errors)))
    return float(grid[int(np.argmin(scores))]), scores


@dataclass(frozen=True, eq=False)
class RDD:
    """A sharp regression discontinuity: the jump in the outcome at the cutoff, by local linear regression."""

    outcome: str
    running: str
    cutoff: float
    bandwidth: float
    estimate: float
    se: float
    n_left: int
    n_right: int
    sensitivity: pd.DataFrame
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    chosen_by: str = ""

    def summary(self) -> Table:
        return Table(_EST_COLUMNS, [_estimate_row(self.estimate, self.se)], index=["jump at the cutoff"], index_name="effect",
                     title=f"Regression discontinuity: {self.outcome} at {self.running} = {self.cutoff:g}",
                     notes=(f"Local linear with a triangular kernel, bandwidth {self.bandwidth:.4g} ({self.chosen_by}); "
                            f"{self.n_left} observations below and {self.n_right} above within it. Robust (HC1) standard error; "
                            "conventional, not bias-corrected.",))

    def sensitivity_table(self) -> Table:
        s = self.sensitivity
        return Table(["bandwidth", "estimate", "std err", "observations"], s.to_numpy().tolist(), index=[str(i) for i in s.index],
                     index_name="multiple of the bandwidth", title="The estimate at other bandwidths")

    def plot(self) -> Figure:
        from .figures import Figure, Line, Panel, Rule, Scatter

        bins = 20
        order = np.argsort(self.x)
        x, y = self.x[order], self.y[order]
        layers: list[Any] = []
        for side in (x < self.cutoff, x >= self.cutoff):
            xs, ys = x[side], y[side]
            edges = np.linspace(xs.min(), xs.max(), bins + 1)
            which = np.clip(np.digitize(xs, edges) - 1, 0, bins - 1)
            mx = pd.Series([xs[which == b].mean() for b in range(bins) if (which == b).any()])
            my = pd.Series([ys[which == b].mean() for b in range(bins) if (which == b).any()])
            layers.append(Scatter(mx, my, "bin means" if not layers else ""))
            near = np.abs(xs - self.cutoff) <= self.bandwidth
            if near.sum() > 2:
                X = np.column_stack([np.ones(near.sum()), xs[near] - self.cutoff])
                w = np.clip(1 - np.abs(xs[near] - self.cutoff) / self.bandwidth, 0, None)
                b = np.linalg.lstsq(X * np.sqrt(w)[:, None], ys[near] * np.sqrt(w), rcond=None)[0]
                grid = np.linspace(xs[near].min(), xs[near].max(), 30)
                layers.append(Line(pd.Series(b[0] + b[1] * (grid - self.cutoff), index=pd.Index(grid, name=self.running)), "local linear fit"
                                   if len(layers) < 3 else ""))
        layers.append(Rule(y=self.cutoff, vertical=True, style="--"))
        return Figure((Panel(tuple(layers), title=f"{self.outcome} by {self.running}: a jump of {self.estimate:.4g} at {self.cutoff:g}",
                             xlabel=self.running),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.sensitivity

    def __repr__(self) -> str:
        return f"<RDD jump {self.estimate:.4g} (s.e. {self.se:.3g}) at {self.running} = {self.cutoff:g}, bandwidth {self.bandwidth:.4g}>"


@api("pymetrix:rdd")
def rdd(data: Dataset, outcome: str, running: str, cutoff: float = 0.0, bandwidth: float | None = None) -> RDD:
    """A sharp regression discontinuity: the jump in ``outcome`` where ``running`` reaches ``cutoff``, by local linear
    regression with a triangular kernel; ``bandwidth`` by cross-validation unless given."""
    _check(data, [outcome, running])
    frame = data.frame[[running, outcome]].dropna()
    x, y = frame[running].to_numpy(float), frame[outcome].to_numpy(float)
    if (x < cutoff).sum() < 10 or (x >= cutoff).sum() < 10:
        raise SpecificationError(f"too few observations on one side of {running} = {cutoff:g}")
    if bandwidth is None:
        span = min(cutoff - x.min(), x.max() - cutoff)
        grid = np.linspace(span / 20, span, 20)
        h, _ = cv_bandwidth(x, y, cutoff, grid)
        how = "chosen by cross-validation"
    else:
        h, how = float(bandwidth), "given"
    est, se, nl, nr, _ = _local_linear(x, y, cutoff, h)
    rows = {}
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        try:
            e, s, a, b, _ = _local_linear(x, y, cutoff, h * mult)
            rows[mult] = [h * mult, e, s, a + b]
        except SpecificationError:
            continue
    sens = pd.DataFrame(rows, index=["bandwidth", "estimate", "std err", "observations"]).T
    sens.index.name = "multiple"
    return RDD(outcome, running, float(cutoff), h, est, se, nl, nr, sens, x, y, how)


# -- weighting on the propensity score --------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class TreatmentEffect:
    """An average treatment effect under selection on observables, by inverse probability weighting or doubly
    robust (augmented) weighting."""

    outcome: str
    treatment: str
    covariates: tuple[str, ...]
    estimand: str
    method: str
    estimate: float
    se: float
    propensity: pd.Series = field(repr=False)
    reps: int = 0
    treated: pd.Series | None = field(default=None, repr=False)  # the 0/1 treatment, on the propensity score's rows

    def summary(self) -> Table:
        p = self.propensity
        return Table(_EST_COLUMNS, [_estimate_row(self.estimate, self.se)], index=[self.estimand], index_name="effect",
                     title=f"{'Doubly robust (AIPW)' if self.method == 'aipw' else 'Inverse probability weighting'}: {self.treatment} on {self.outcome}",
                     notes=(f"Propensity score by logit on {', '.join(self.covariates)}; it ranges from {p.min():.3f} to {p.max():.3f}"
                            + (" (values near 0 or 1 make the weights unstable)" if p.min() < 0.02 or p.max() > 0.98 else "") + ".",
                            f"Standard error by the bootstrap ({self.reps} replications, the propensity score re-estimated in each)."))

    def plot(self) -> Figure:
        from .figures import Figure, Histogram, Panel

        d = self.treated
        p = self.propensity
        treated = p[d == 1] if d is not None else p
        control = p[d == 0] if d is not None else p.iloc[:0]
        return Figure((Panel((Histogram(treated.rename("treated"), bins=25), Histogram(control.rename("control"), bins=25)),
                             title="Propensity scores: the overlap of treated and controls"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<TreatmentEffect {self.estimand} {self.estimate:.4g} (s.e. {self.se:.3g}) by {self.method}>"


def _ipw_point(y: np.ndarray, d: np.ndarray, X: np.ndarray, estimand: str, method: str) -> tuple[float, np.ndarray]:
    from .estimation.families import Logit
    from .estimation.ml import _newton

    gamma = _newton(Logit(), d, X, np.ones(len(d)), 100, 1e-10)[0]
    p = 1 / (1 + np.exp(-(X @ gamma)))
    if method == "aipw":
        b1 = np.linalg.lstsq(X[d == 1], y[d == 1], rcond=None)[0]
        b0 = np.linalg.lstsq(X[d == 0], y[d == 0], rcond=None)[0]
        m1, m0 = X @ b1, X @ b0
        if estimand == "ATE":
            return float(np.mean(m1 - m0 + d * (y - m1) / p - (1 - d) * (y - m0) / (1 - p))), p
        odds = p / (1 - p)  # ATT, doubly robust: treated outcomes against controls reweighted to the treated
        return float((np.sum(d * (y - m0)) - np.sum((1 - d) * odds * (y - m0))) / np.sum(d)), p
    if estimand == "ATE":
        w1, w0 = d / p, (1 - d) / (1 - p)
    else:
        w1, w0 = d.astype(float), (1 - d) * p / (1 - p)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)), p


@api("pymetrix:treatment_effect")
def treatment_effect(data: Dataset, outcome: str, treatment: str, covariates: Sequence[str], estimand: Literal["ATE", "ATT"] = "ATE",
                     method: Literal["ipw", "aipw"] = "ipw", reps: int = 200, seed: int = 0) -> TreatmentEffect:
    """The average effect of the 0/1 ``treatment`` on ``outcome`` (ATE, or on the treated, ATT), assuming selection
    on the ``covariates``: inverse probability weighting with normalised weights, or ``method="aipw"`` (doubly
    robust: also a regression of the outcome on the covariates in each group). Bootstrap standard errors."""
    from .progress import report

    covariates = _names(covariates)
    if not covariates:
        raise SpecificationError("give the covariates the treatment depends on")
    if estimand not in ("ATE", "ATT") or method not in ("ipw", "aipw"):
        raise SpecificationError("estimand 'ATE' or 'ATT'; method 'ipw' or 'aipw'")
    _check(data, [outcome, treatment, *covariates])
    frame = data.frame[[outcome, treatment, *covariates]].dropna()
    y, d = frame[outcome].to_numpy(float), frame[treatment].to_numpy(float)
    if not np.isin(d, (0.0, 1.0)).all() or d.min() == d.max():
        raise SpecificationError(f"{treatment} must be 0 or 1, with some of each")
    X = np.column_stack([np.ones(len(d)), frame[covariates].to_numpy(float)])
    est, p = _ipw_point(y, d, X, estimand, method)
    rng = make_rng(seed)
    draws = []
    for r in range(reps):
        report(r / max(reps, 1), f"bootstrap {r + 1} of {reps}")
        i = rng.integers(0, len(d), len(d))
        try:
            draws.append(_ipw_point(y[i], d[i], X[i], estimand, method)[0])
        except Exception:  # noqa: BLE001 - a resample with perfect prediction is skipped
            continue
    se = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
    pscore = pd.Series(p, index=frame.index, name="propensity score")
    return TreatmentEffect(outcome, treatment, tuple(covariates), estimand, method, est, se, pscore, len(draws),
                           pd.Series(d, index=frame.index, name=treatment))


# -- synthetic control ------------------------------------------------------------------------------------------------

def _weights(Y1: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """Non-negative donor weights summing to one that best reproduce the treated unit's pre-treatment path."""
    J = Y0.shape[1]

    def loss(w: np.ndarray) -> float:
        r = Y1 - Y0 @ w
        return float(r @ r)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2 * Y0.T @ (Y1 - Y0 @ w)

    found = optimize.minimize(loss, np.full(J, 1 / J), jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * J,
                              constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(J)}],
                              options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(found.x, 0, None)
    return w / w.sum()


@dataclass(frozen=True, eq=False)
class SyntheticControl:
    """A synthetic control: the treated unit against a weighted combination of untreated units (Abadie, Diamond and
    Hainmueller 2010), with in-space placebos."""

    outcome: str
    treated: Any
    start: Any
    weights: pd.Series
    actual: pd.Series
    synthetic: pd.Series
    placebo_gaps: pd.DataFrame = field(repr=False)
    ratio: float
    p_value: float

    @property
    def gap(self) -> pd.Series:
        return (self.actual - self.synthetic).rename("gap")

    @property
    def effect(self) -> float:
        """The average gap after the start."""
        return float(self.gap[self.gap.index >= self.start].mean())

    def summary(self) -> Table:
        pre = self.gap[self.gap.index < self.start]
        rows = [["average gap after the start", self.effect], ["RMSPE before", float(np.sqrt(np.mean(pre**2)))],
                ["post/pre RMSPE ratio", self.ratio], ["placebo p-value", self.p_value]]
        return Table(["value"], [r[1:] for r in rows], index=[r[0] for r in rows], index_name="",
                     title=f"Synthetic control: unit {self.treated}, from {self.start}",
                     notes=(f"Weights fitted to {self.outcome} before {self.start}. The p-value ranks the treated unit's post/pre "
                            f"RMSPE ratio among it and the {self.placebo_gaps.shape[1]} donors, each treated in turn (in-space placebos).",))

    def weights_table(self) -> Table:
        w = self.weights[self.weights > 1e-6].sort_values(ascending=False)
        return Table(["weight"], [[float(v)] for v in w], index=[str(i) for i in w.index], index_name="donor", title="Donor weights")

    def plot(self) -> Figure:
        from .figures import Figure, Line, Panel, Rule

        top = Panel((Line(self.actual, f"unit {self.treated}", color="#222222"), Line(self.synthetic, "synthetic", style="--"),
                     Rule(y=self.start, vertical=True, style=":")), title=f"{self.outcome}: actual and synthetic")
        placebo = tuple(Line(self.placebo_gaps[c], "", color="#bbbbbb", width=0.8) for c in self.placebo_gaps.columns)
        bottom = Panel(placebo + (Line(self.gap, f"unit {self.treated}", color="#c92a2a"), Rule(0.0), Rule(y=self.start, vertical=True, style=":")),
                       title="Gaps: the treated unit and the placebos")
        return Figure((top, bottom))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"actual": self.actual, "synthetic": self.synthetic, "gap": self.gap})

    def __repr__(self) -> str:
        return f"<SyntheticControl unit {self.treated}: average gap {self.effect:.4g}, placebo p = {self.p_value:.3f}>"


@api("pymetrix:synthetic_control")
def synthetic_control(data: Dataset, outcome: str, treated: Any, start: Any, donors: Sequence[Any] | None = None) -> SyntheticControl:
    """The effect on unit ``treated`` from period ``start``: its ``outcome`` against a weighted average of the
    ``donors`` (default: every other unit), weighted to match its path before ``start``."""
    unit, time = _panel_axes(data, "synthetic control")
    _check(data, [outcome])
    wide = data.frame[outcome].unstack(unit)
    wide.index = pd.PeriodIndex(wide.index) if not isinstance(wide.index, pd.PeriodIndex) else wide.index
    keys = {str(k): k for k in wide.columns}
    if str(treated) not in keys:
        raise SpecificationError(f"no unit {treated!r}")
    treated_key = keys[str(treated)]
    start_p = pd.Period(str(start), freq=wide.index.freq)
    pool = [keys[str(k)] for k in donors] if donors else [k for k in wide.columns if k != treated_key]
    wide = wide[[treated_key, *pool]].dropna()
    pre = wide.index < start_p
    if pre.sum() < 3 or (~pre).sum() < 1:
        raise SpecificationError(f"need periods before and after {start}")

    def fit(target: Any, others: list[Any]) -> tuple[pd.Series, pd.Series]:
        w = _weights(wide.loc[pre, target].to_numpy(float), wide.loc[pre, others].to_numpy(float))
        return pd.Series(w, index=others), pd.Series(wide[others].to_numpy(float) @ w, index=wide.index)

    def ratio(gap: pd.Series) -> float:
        before, after = gap[pre], gap[~pre]
        return float(np.sqrt(np.mean(after**2)) / max(np.sqrt(np.mean(before**2)), 1e-12))

    from .progress import report

    weights, synth = fit(treated_key, pool)
    gap = wide[treated_key] - synth
    placebos = {}
    for j, donor in enumerate(pool):
        report(j / len(pool), f"placebo {j + 1} of {len(pool)}")
        _, s = fit(donor, [k for k in [treated_key, *pool] if k != donor and k != treated_key])
        placebos[donor] = wide[donor] - s
    gaps = pd.DataFrame(placebos)
    r = ratio(gap)
    ranks = [ratio(gaps[c]) for c in gaps.columns]
    p = (1 + sum(x >= r for x in ranks)) / (1 + len(ranks))
    return SyntheticControl(outcome, treated, start_p, weights.rename("weight"), wide[treated_key].rename("actual"),
                            synth.rename("synthetic"), gaps, r, float(p))
