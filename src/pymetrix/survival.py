"""Survival analysis: the Kaplan-Meier estimator with the log-rank test across groups, and Cox's proportional hazards
regression, through ``statsmodels.duration`` (``SurvfuncRight``, ``survdiff`` and ``PHReg``).

    km = kaplan_meier(data, "weeks", "arrest", group="fin")         # curves by group, log-rank test
    km.survival_table(); km.plot()
    cox = cox_ph(data, "weeks", "arrest", ["fin", "age", "prio"], ties="efron")
    cox.hazard_ratios(); cox.ph_test()                             # Grambsch-Therneau test of proportional hazards

Data are one row per subject: a duration (time to the event or to censoring), an event indicator (1: the event
happened, 0: censored), and optionally an entry time (left truncation: the subject joins the risk set only after
it, as in delayed entry) and strata.

Kaplan-Meier's product limit S(t) = prod_{t_i <= t} (1 - d_i / n_i) over the event times t_i, with d_i events among
the n_i at risk; Greenwood's standard error; pointwise intervals on the log(-log) scale (Kalbfleisch and Prentice,
as Stata), the log scale (as R's survfit) or the plain scale; and Hall-Wellner simultaneous bands (statsmodels'; 95%
only). The log-rank test compares observed with expected events across groups (optionally within strata), with
Gehan-Breslow (Wilcoxon) and Tarone-Ware weights as alternatives.

Cox's model h(t | x) = h0(t) exp(x'beta) by partial likelihood, with Breslow's or Efron's handling of tied event
times; standard errors from the inverse Hessian, or robust (Lin and Wei's sandwich, each subject its own cluster or
clustered). The baseline cumulative hazard and survival (at every regressor zero, per stratum) are statsmodels'
Breslow-Nelson-Aalen estimates. The proportional-hazards check is Grambsch and Therneau's test from the scaled
Schoenfeld residuals: for each regressor and globally, against a transform of time ("km": 1 - the Kaplan-Meier
estimate, as R's cox.zph did before version 3; "rank", "identity" or "log"), in the approximation R's survival
package used until 2019 and lifelines uses. The Schoenfeld residuals are statsmodels' (Breslow's risk-set averages,
whichever ties).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from . import _micro
from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedCoxPH", "FittedKaplanMeier", "cox_ph", "kaplan_meier"]

_TESTS = {"logrank": None, "wilcoxon": "gb", "tarone-ware": "tw"}
_TEST_NAMES = {"logrank": "Log-rank test", "wilcoxon": "Gehan-Breslow (Wilcoxon) test", "tarone-ware": "Tarone-Ware test"}


def _backend() -> dict[str, str]:
    import statsmodels

    return {"name": "statsmodels", "version": statsmodels.__version__, "wraps": "statsmodels.duration"}


def _read(data: Dataset, duration: str, event: str, others: Sequence[str], what: str) -> pd.DataFrame:
    frame = _micro.columns(data, [duration, event, *[o for o in others if o]], what)
    t = _micro.numeric(frame, [duration], what)[:, 0]
    d = _micro.numeric(frame, [event], what)[:, 0]
    if not np.isin(d, (0.0, 1.0)).all():
        raise SpecificationError(f"{event} must be 1 (the event happened) or 0 (censored)")
    if (t < 0).any():
        raise SpecificationError(f"{duration} must not be negative")
    if not len(frame):
        raise SpecificationError(f"{what}: no rows with {duration} and {event}")
    return frame


def pointwise_interval(S: np.ndarray, se: np.ndarray, alpha: float, interval: str) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise limits for S from its standard error: "log-log" (Kalbfleisch-Prentice), "log" or "plain"."""
    z = float(sps.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        if interval == "log-log":
            w = z * se / (S * np.abs(np.log(S)))
            lo, hi = S ** np.exp(w), S ** np.exp(-w)
        elif interval == "log":
            lo, hi = S * np.exp(-z * se / S), S * np.exp(z * se / S)
        else:
            lo, hi = S - z * se, S + z * se
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    undefined = (S >= 1) | (S <= 0) | ~np.isfinite(se)
    return np.where(undefined & (interval == "log-log"), np.nan, lo), np.where(undefined & (interval == "log-log"), np.nan, hi)


def _step(times: np.ndarray, values: np.ndarray, start: float = 1.0) -> pd.Series:
    """A step function as points for a line: the value before each time, then after it."""
    x = np.r_[0.0, np.repeat(times, 2)]
    y = np.r_[start, start, np.repeat(values, 2)[:-1]] if len(values) else np.array([start])
    return pd.Series(y, index=pd.Index(x, name="time"))


# -- Kaplan-Meier -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedKaplanMeier:
    """Kaplan-Meier estimates, one curve per group, with the test of equal survival across groups (``test``)."""

    duration: str
    event: str
    table: pd.DataFrame                  # group, time: at risk, events, survival, std err, lower, upper (and band lower, upper)
    groups: pd.DataFrame                 # per group: subjects, events, median and its interval
    group: str | None = None
    test: Any = None                     # TestResult, or None with one group
    alpha: float = 0.05
    interval: str = "log-log"
    bands: bool = False
    entry: str | None = None
    provenance: Any = field(default=None, repr=False)

    stage = "estimate"

    def survival_table(self, group: Any = None) -> pd.DataFrame:
        """The estimates at each event time (of one group, or all)."""
        if group is None:
            return self.table
        return self.table.xs(group, level="group")

    def survival_at(self, times: Sequence[float]) -> pd.DataFrame:
        """S(t) and its interval at chosen times (the step function: the last event time at or before t), by group."""
        out = []
        for g, part in self.table.groupby(level="group", sort=False):
            t = part.index.get_level_values("time").to_numpy(float)
            for x in times:
                k = np.searchsorted(t, float(x), side="right") - 1
                row = part.iloc[k] if k >= 0 else None
                out.append({"group": g, "time": float(x), "survival": 1.0 if row is None else row["survival"],
                            "lower": 1.0 if row is None else row["lower"], "upper": 1.0 if row is None else row["upper"]})
        return pd.DataFrame(out).set_index(["group", "time"])

    def summary(self) -> Table:
        g = self.groups
        level = f"{100 * (1 - self.alpha):g}%"
        notes = [f"Kaplan-Meier estimates of survival in {self.duration} (event: {self.event})"
                 + (f", by {self.group}" if self.group else "") + (f"; entry at {self.entry} (left truncation)" if self.entry else "") + ".",
                 f"Median survival with its {level} interval (statsmodels' quantile_ci, complementary log-log); empty where the curve does not reach 0.5.",
                 f"Pointwise {level} intervals on the {self.interval} scale, Greenwood's standard errors (the survival table)."]
        if self.test is not None:
            t = self.test
            notes.append(f"{t.name}: chi-squared {t.statistic:.4f} on {t.df} df, p = {t.p_value:.4g} (null: {t.null}).")
        rows = g[["subjects", "events", "median", "median lower", "median upper"]].to_numpy(float).tolist()
        return Table(["subjects", "events", "median", "median lower", "median upper"], rows, index=[str(i) for i in g.index],
                     index_name=self.group or "group", title=f"Kaplan-Meier: {self.duration}", notes=tuple(notes))

    def plot(self) -> Any:
        from .figures import PALETTE, Band, Figure, Line, Panel

        layers = []
        for i, (gname, part) in enumerate(self.table.groupby(level="group", sort=False)):
            t = part.index.get_level_values("time").to_numpy(float)
            color = PALETTE[i % len(PALETTE)]
            label = str(gname) if self.group else "survival"
            lo, hi = ("band lower", "band upper") if self.bands else ("lower", "upper")
            layers.append(Band(_step(t, part[lo].to_numpy()), _step(t, part[hi].to_numpy()), color=color))
            layers.append(Line(_step(t, part["survival"].to_numpy()), label=label, color=color))
        what = "Hall-Wellner 95% bands" if self.bands else f"pointwise {100 * (1 - self.alpha):g}% intervals"
        return Figure((Panel(tuple(layers), title=f"Kaplan-Meier survival ({what})", ylabel="survival", xlabel=self.duration),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.table

    def __repr__(self) -> str:
        return f"<FittedKaplanMeier {self.duration}: {len(self.groups)} group(s)>"


@api("pymetrix:kaplan_meier")
def kaplan_meier(data: Dataset, duration: str, event: str, group: str | None = None, entry: str | None = None, strata: str | None = None,
                 alpha: float = 0.05, interval: Literal["log-log", "log", "plain"] = "log-log", bands: bool = False,
                 test: Literal["logrank", "wilcoxon", "tarone-ware"] = "logrank") -> FittedKaplanMeier:
    """Kaplan-Meier survival curves of ``duration`` (``event`` 1 for the event, 0 censored), one per value of
    ``group``, with the log-rank test of equal survival across groups (``test``: "logrank", "wilcoxon"
    (Gehan-Breslow) or "tarone-ware"; within ``strata`` if given). ``entry``: entry times (left truncation).
    Pointwise intervals on the ``interval`` scale; ``bands``: Hall-Wellner simultaneous bands too (95% only)."""
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff

    from ._capture import capture

    if interval not in ("log-log", "log", "plain"):
        raise SpecificationError(f"interval is 'log-log', 'log' or 'plain', not {interval!r}")
    if test not in _TESTS:
        raise SpecificationError(f"test is 'logrank', 'wilcoxon' or 'tarone-ware', not {test!r}")
    if not 0 < alpha < 1:
        raise SpecificationError("alpha is between 0 and 1")
    if bands and alpha != 0.05:
        raise SpecificationError("Hall-Wellner bands are 95% only (alpha=0.05), as statsmodels gives them")
    if strata and not group:
        raise SpecificationError("strata are for the test across groups: give group= too")
    if strata and entry:
        # statsmodels' survdiff takes each stratum's rows but every subject's entry times, so the two cannot be combined
        raise SpecificationError("the stratified test with entry times is not available: give strata or entry, not both")
    frame = _read(data, duration, event, [group or "", entry or "", strata or ""], "Kaplan-Meier")
    t = frame[duration].to_numpy(float)
    d = frame[event].to_numpy(float)
    e = frame[entry].to_numpy(float) if entry else None
    if e is not None and (e > t).any():
        raise SpecificationError(f"{entry} is after {duration} for some subjects: entry must be at or before the end")
    gv = frame[group].to_numpy(object) if group else np.zeros(len(t), object)
    labels = sorted(pd.unique(gv), key=str) if group else ["all"]
    tables, rows = [], []
    with capture() as caught:
        for lab in labels:
            sel = gv == lab if group else np.ones(len(t), bool)
            sf = SurvfuncRight(t[sel], d[sel], entry=None if e is None else e[sel])
            S, se = np.asarray(sf.surv_prob, float), np.asarray(sf.surv_prob_se, float)
            lo, hi = pointwise_interval(S, se, alpha, interval)
            part = pd.DataFrame({"at risk": sf.n_risk, "events": sf.n_events, "survival": S, "std err": se, "lower": lo, "upper": hi},
                                index=pd.MultiIndex.from_arrays([[lab] * len(S), sf.surv_times], names=["group", "time"]))
            if bands:
                with np.errstate(divide="ignore", invalid="ignore"):
                    blo, bhi = sf.simultaneous_cb(alpha=0.05)
                part["band lower"], part["band upper"] = blo, bhi
            tables.append(part)
            med = float(sf.quantile(0.5)) if (S <= 0.5).any() else np.nan
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    mlo, mhi = sf.quantile_ci(0.5, alpha=alpha) if len(S) else (np.nan, np.nan)
            except Exception:  # noqa: BLE001 - no interval where statsmodels finds none
                mlo, mhi = np.nan, np.nan

            def finite(v: Any) -> float:
                return float(v) if v is not None and np.isfinite(v) else np.nan  # inf: the curve stays above 0.5

            rows.append({"group": lab, "subjects": int(sel.sum()), "events": int(d[sel].sum()), "median": med,
                         "median lower": finite(mlo), "median upper": finite(mhi)})
        result_test = None
        if group and len(labels) > 1:
            kw = {} if _TESTS[test] is None else {"weight_type": _TESTS[test]}
            stat, p = survdiff(t, d, gv, strata=frame[strata].to_numpy(object) if strata else None, entry=e, **kw)
            from .stats.base import TestResult

            df_ = len(labels) - 1
            result_test = TestResult(_TEST_NAMES[test] + (f" (stratified by {strata})" if strata else ""),
                                     f"the survival functions are the same in every {group}", float(stat), df_, float(p), f"chi-squared({df_})")
    table = pd.concat(tables)
    groups = pd.DataFrame(rows).set_index("group")
    spec = {"duration": duration, "event": event, "group": group, "entry": entry, "strata": strata, "alpha": alpha, "interval": interval,
            "bands": bands, "test": test}
    prov = _micro.provenance(spec, data, frame, [duration, event], caught, {"type": "Kaplan-Meier", "estimation": "product limit"})
    import dataclasses

    prov = dataclasses.replace(prov, backend=_backend())
    return FittedKaplanMeier(duration, event, table, groups, group, result_test, alpha, interval, bands, entry, prov)


# -- Cox proportional hazards -----------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedCoxPH:
    """Cox's proportional hazards regression by partial likelihood (statsmodels' PHReg)."""

    duration: str
    event: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    loglike: float = 0.0
    loglike_null: float = 0.0
    nobs: int = 0
    events: int = 0
    ties: str = "breslow"
    cov_type: str = "hessian"
    cluster: str | None = None
    strata: str | None = None
    entry: str | None = None
    baseline: pd.DataFrame | None = field(default=None, repr=False)        # stratum, time: cumulative hazard, survival
    schoenfeld: pd.DataFrame | None = field(default=None, repr=False)      # rows with events: time and the residuals
    naive_cov: np.ndarray | None = field(default=None, repr=False)         # the inverse Hessian (for the PH test)
    km: pd.Series | None = field(default=None, repr=False)                 # the whole sample's Kaplan-Meier estimate (the "km" transform)
    provenance: Any = field(default=None, repr=False)

    stage = "estimate"

    @property
    def params(self) -> Any:
        from .estimation.params import PointEstimate

        return PointEstimate(self.names, np.asarray(self.values, float), np.asarray(self.cov, float))

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def stats(self) -> dict[str, float]:
        k = len(self.values)
        lr = 2 * (self.loglike - self.loglike_null)
        return {"nobs": float(self.nobs), "events": float(self.events), "loglike": self.loglike, "loglike_null": self.loglike_null,
                "lr": lr, "lr_pvalue": float(sps.chi2.sf(lr, k)), "aic": -2 * self.loglike + 2 * k}

    def hazard_ratios(self, alpha: float = 0.05) -> pd.DataFrame:
        b, s = np.asarray(self.values), self.se
        q = float(sps.norm.ppf(1 - alpha / 2))
        return pd.DataFrame({"hazard ratio": np.exp(b), "lower": np.exp(b - q * s), "upper": np.exp(b + q * s)},
                            index=pd.Index(self.names, name="parameter"))

    def ph_test(self, transform: Literal["km", "rank", "identity", "log"] = "km") -> Table:
        """Grambsch and Therneau's test of proportional hazards for each regressor and globally: the scaled Schoenfeld
        residuals against ``transform`` of the event times (the approximation of R's cox.zph before version 3, and
        lifelines'). A small p-value says the regressor's effect changes over time."""
        if transform not in ("km", "rank", "identity", "log"):
            raise SpecificationError(f"transform is 'km', 'rank', 'identity' or 'log', not {transform!r}")
        r = self.schoenfeld[list(self.names)].to_numpy(float)
        tt = self.schoenfeld["time"].to_numpy(float)
        ok = np.all(np.isfinite(r), axis=1)
        r, tt = r[ok], tt[ok]
        order = np.argsort(tt, kind="stable")
        r, tt = r[order], tt[order]
        if transform == "identity":
            g = tt
        elif transform == "log":
            g = np.log(tt)
        elif transform == "rank":
            g = np.arange(1, len(tt) + 1, dtype=float)
        else:
            g = 1 - self._km_at(tt)
        V = np.asarray(self.naive_cov if self.naive_cov is not None else self.cov, float)
        d = len(tt)
        gc = g - g.mean()
        scaled = d * r @ V + np.asarray(self.values)
        test = gc @ scaled
        stat = test**2 / (d * np.diag(V) * (gc @ gc))
        u = gc @ r
        glob = float(u @ V @ u) * d / (gc @ gc)
        k = len(self.names)
        rows = [[float(s), 1.0, float(sps.chi2.sf(s, 1))] for s in stat] + [[glob, float(k), float(sps.chi2.sf(glob, k))]]
        notes = (f"Grambsch-Therneau tests of proportional hazards: scaled Schoenfeld residuals against {transform} of time "
                 f"({d} events); the approximation R's survival package used before version 3 and lifelines use.",
                 "A small p-value: the regressor's effect on the hazard changes over time.")
        return Table(["chi-squared", "df", "p-value"], rows, index=[*self.names, "global"], index_name="regressor",
                     title=f"Proportional hazards: {self.duration}", notes=notes)

    def _km_at(self, times: np.ndarray) -> np.ndarray:
        """The Kaplan-Meier estimate of the whole sample at ``times`` (right-continuous), for the "km" transform."""
        km = self.km
        k = np.searchsorted(km.index.to_numpy(float), times, side="right") - 1
        return np.where(k >= 0, km.to_numpy(float)[np.maximum(k, 0)], 1.0)

    def summary(self) -> Table:
        st = self.stats
        cov_text = {"hessian": "Standard errors from the inverse Hessian.", "robust": "Robust (sandwich) standard errors, each subject its own cluster.",
                    "cluster": f"Standard errors clustered by {self.cluster}."}[self.cov_type]
        notes = [f"Cox proportional hazards of {self.duration} (event: {self.event}); {self.nobs} subjects, {self.events} events; "
                 f"{'Efron' if self.ties == 'efron' else 'Breslow'}'s handling of tied event times"
                 + (f"; strata {self.strata}" if self.strata else "") + (f"; entry at {self.entry} (left truncation)" if self.entry else "") + ".",
                 f"Log partial likelihood: {self.loglike:.4f} (all zero: {self.loglike_null:.4f}); LR test {st['lr']:.4f} on {len(self.values)} df, "
                 f"p = {st['lr_pvalue']:.4g}.", "exp(coefficient) is the hazard ratio.", cov_text]
        return _micro.coefficient_table(self.names, self.values, self.se, f"Cox proportional hazards: {self.duration}", notes)

    def plot(self) -> Any:
        from .figures import PALETTE, Figure, Line, Panel

        layers = []
        for i, (s, part) in enumerate(self.baseline.groupby(level="stratum", sort=False)):
            t = part.index.get_level_values("time").to_numpy(float)
            layers.append(Line(_step(t, part["survival"].to_numpy()), label=str(s) if self.strata else "baseline survival",
                               color=PALETTE[i % len(PALETTE)]))
        return Figure((Panel(tuple(layers), title="Baseline survival (every regressor 0)", ylabel="survival", xlabel=self.duration),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedCoxPH {self.duration}: {dict(zip(self.names, np.round(self.values, 4)))}>"


@api("pymetrix:cox_ph")
def cox_ph(data: Dataset, duration: str, event: str, regressors: Sequence[str], entry: str | None = None, strata: str | None = None,
           ties: Literal["breslow", "efron"] = "breslow", cov: Literal["hessian", "robust", "cluster"] = "hessian",
           cluster: str | None = None) -> FittedCoxPH:
    """Cox's proportional hazards regression of ``duration`` (``event`` 1 for the event, 0 censored) on
    ``regressors``, by partial likelihood with ``ties`` "breslow" or "efron"; ``entry`` for left truncation and
    ``strata`` for a baseline hazard per stratum. ``cov``: the inverse Hessian, "robust" (each subject its own
    cluster) or "cluster" (by ``cluster``)."""
    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.duration.survfunc import SurvfuncRight

    from ._capture import capture

    regressors = _micro.names(regressors)
    if not regressors:
        raise SpecificationError("the Cox model needs at least one regressor")
    if ties not in ("breslow", "efron"):
        raise SpecificationError(f"ties is 'breslow' or 'efron', not {ties!r}")
    if cov not in ("hessian", "robust", "cluster"):
        raise SpecificationError(f"cov is 'hessian', 'robust' or 'cluster', not {cov!r}")
    if cov == "cluster" and not cluster:
        raise SpecificationError("cov='cluster' needs cluster= (the series whose values are the clusters)")
    frame = _read(data, duration, event, [*regressors, entry or "", strata or "", cluster or ""], "Cox model")
    t = frame[duration].to_numpy(float)
    d = frame[event].to_numpy(float)
    X = _micro.numeric(frame, regressors, "Cox model")
    e = frame[entry].to_numpy(float) if entry else None
    if e is not None and (e > t).any():
        raise SpecificationError(f"{entry} is after {duration} for some subjects: entry must be at or before the end")
    if d.sum() == 0:
        raise SpecificationError(f"no events: {event} is 0 for every subject")
    if np.linalg.matrix_rank(X - X.mean(axis=0)) < X.shape[1]:
        raise SpecificationError("the regressors are collinear, or one is constant (the baseline hazard absorbs a constant)")
    st = frame[strata].to_numpy(object) if strata else None
    with capture() as caught:
        model = PHReg(t, X, status=d, entry=e, strata=None if st is None else pd.factorize(st)[0], ties=ties)
        groups = None if cov == "hessian" else (np.arange(len(t)) if cov == "robust" else pd.factorize(frame[cluster])[0])
        res = model.fit(groups=groups)
        naive = model.fit() if groups is not None else res
        beta = np.asarray(res.params, float)
        V = np.asarray(res.cov_params(), float)
        ll, ll0 = float(model.loglike(beta)), float(model.loglike(np.zeros(len(beta))))
        names_s = [str(s) for s in pd.factorize(st)[1]] if st is not None else ["all"]
        parts = []
        for s, (tt, H, S) in zip(names_s, res.baseline_cumulative_hazard):
            parts.append(pd.DataFrame({"cumulative hazard": H, "survival": S},
                                      index=pd.MultiIndex.from_arrays([[s] * len(tt), np.asarray(tt, float)], names=["stratum", "time"])))
        baseline = pd.concat(parts)
        sf = SurvfuncRight(t, d, entry=e)
        km = pd.Series(np.asarray(sf.surv_prob, float), index=np.asarray(sf.surv_times, float))
        sch = np.asarray(res.schoenfeld_residuals, float)
        ev = d == 1
        schoenfeld = pd.DataFrame(sch[ev], columns=regressors, index=frame.index[ev])
        schoenfeld.insert(0, "time", t[ev])
    spec = {"duration": duration, "event": event, "regressors": regressors, "entry": entry, "strata": strata, "ties": ties, "cov": cov,
            "cluster": cluster}
    prov = _micro.provenance(spec, data, frame, [duration, event, *regressors], caught,
                             {"type": "Cox proportional hazards", "estimation": "partial likelihood", "ties": ties})
    import dataclasses

    prov = dataclasses.replace(prov, backend=_backend())
    return FittedCoxPH(duration, event, tuple(regressors), beta, V, ll, ll0, len(t), int(d.sum()), ties, cov, cluster, strata, entry,
                       baseline, schoenfeld, np.asarray(naive.cov_params(), float), km, prov)
