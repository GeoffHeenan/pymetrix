"""The forecast workflow: rounds, judgement, fan charts, a store of past rounds, and backtests.

    rnd = ForecastRound.from_forecast("2026Q3 baseline", var.forecast(data, ("2026Q4", "2028Q4")), data=data)
    rnd = rnd.adjust("gdp", {"2026Q4": -0.3}, reason="strike in the ports")   # judgement, logged
    rnd.plot(data)                                                            # fan chart
    store = ForecastStore("forecasts"); store.save(rnd)
    store.compare("2026Q2 baseline", "2026Q3 baseline")                       # revisions
    store.accuracy("gdp", data)                                               # real-time track record

    bt = backtest(fit, data, ("2015Q1", "2022Q4"), horizon=8)                 # rolling-origin evaluation
    bt.accuracy(); bt.compare(other_bt, "gdp", horizon=4)                     # Diebold-Mariano
    bt.density_accuracy(); bt.evaluation()                                    # log score, CRPS, PIT; efficiency and calibration tests
    combine_forecasts({"var": bt, "ols": other_bt}, method="inverse_mse")    # a combination, evaluated as a backtest

A forecast round is the set of paths published at one time, with when it was made, the last
period of data it used, notes, its source, and every judgemental adjustment (with the path
before them). A store is a folder of rounds, so a forecasting team's history can be
compared, revised and evaluated. Backtests re-estimate a specification at each forecast
origin and forecast from there, optionally from the data as they were then (``vintages``),
so the evaluation is real-time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api, call_migration, renamed_arguments
from .errors import NotFoundError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .stats import TestResult

__all__ = ["Backtest", "ForecastRound", "ForecastStore", "backtest", "diebold_mariano", "fan_figure"]

_SUFFIXES = ("_se", "_lower", "_upper", "_median")


def _periods(index: Any) -> pd.PeriodIndex:
    if isinstance(index, pd.MultiIndex):
        raise SpecificationError("forecast rounds hold time series; take one entity first (data.entity(...))")
    return pd.PeriodIndex(index)


def _variables(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not c.endswith(_SUFFIXES) and not c.startswith("__")]


# ---------------------------------------------------------------------------------------------
# rounds


@dataclass(frozen=True, eq=False)
class ForecastRound:
    """The forecast paths of one round, with their record: when, from which data, notes, source, judgement and the
    conditions it was made under (a conditional forecast's paths and ranges, or a model plan's holds)."""

    name: str
    paths: pd.DataFrame  # PeriodIndex x columns: each variable, and optionally {variable}_se
    made: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    data_end: str | None = None  # the last period of actual data the forecast used
    notes: str = ""
    adjustments: tuple[dict[str, Any], ...] = ()  # {variable, period, base, value, reason}
    source: Mapping[str, Any] = field(default_factory=dict)
    base: pd.DataFrame | None = field(default=None, repr=False)  # the paths before judgement
    history: pd.DataFrame | None = field(default=None, repr=False)  # recent actual values, for charts (from_forecast's data)
    conditions: tuple[dict[str, Any], ...] = ()  # {variable or shock, period, value or lower/upper[, shocks]}

    def __post_init__(self) -> None:
        paths = self.paths.copy()
        paths.index = _periods(paths.index)
        object.__setattr__(self, "paths", paths)
        if self.base is None:
            object.__setattr__(self, "base", paths[self.variables].copy())

    @classmethod
    @api()
    def from_forecast(cls, name: str, forecast: Any, variables: Sequence[str] | None = None, data: Dataset | None = None,
                      notes: str = "", source: Mapping[str, Any] | None = None, as_name: str | None = None,
                      conditions: Sequence[Mapping[str, Any]] | None = None) -> ForecastRound:
        """A round from a forecast: an equation's (``as_name`` names its variable), a VAR's or Bayesian VAR's, or a
        model ``Solution`` (give ``variables`` and keep only the forecast periods). ``data`` sets ``data_end``.

        The round records the conditions the forecast was made under: a VAR's conditional forecast's (exact values and
        ranges, and the shocks that met them) or a model plan's holds and ranges, read from the forecast; or give
        ``conditions`` as rows such as ``{"variable": "r", "period": "2026Q1", "value": 2.0}`` or ``{"variable": "pi",
        "period": "2026Q1", "lower": 3.0, "upper": 4.0}``."""
        frame = forecast.frame if hasattr(forecast, "frame") else pd.DataFrame(forecast)
        if "forecast" in frame.columns:  # a single equation's forecast
            target = as_name or (forecast.meta.extra.get("target") if hasattr(forecast, "meta") else None)
            if not target:
                raise SpecificationError("give as_name= for an equation's forecast (the name of the variable it forecasts)")
            frame = pd.DataFrame({target: frame["forecast"], f"{target}_se": frame.get("se")}, index=frame.index)
        keep = list(variables) if variables else _variables(frame)
        missing = [v for v in keep if v not in frame.columns]
        if missing:
            raise NotFoundError(f"the forecast has no {', '.join(missing)}")
        cols = [c for v in keep for c in (v, f"{v}_se") if c in frame.columns]
        paths = frame[cols].dropna(how="all", subset=keep)
        data_end = None
        if data is not None and data.time_axis is not None:
            observed = data.frame[[v for v in keep if v in data.names]].dropna(how="all")
            before = observed[observed.index < _periods(paths.index)[0]]
            data_end = str(before.index[-1]) if len(before) else None
        info = dict(source or {})
        if hasattr(forecast, "meta") and forecast.meta.title:
            info.setdefault("forecast", forecast.meta.title)
        history = None
        if data is not None and data.time_axis is not None:
            shown = [v for v in keep if v in data.names]
            if shown:
                recent = data.frame[shown]
                recent = recent[recent.index < _periods(paths.index)[0]].dropna(how="all").iloc[-16:]
                history = recent if len(recent) else None
        rows = tuple(dict(r) for r in conditions) if conditions is not None else _recorded_conditions(forecast, paths.index)
        return cls(name, paths, data_end=data_end, notes=notes, source=info, history=history, conditions=rows)

    @property
    def variables(self) -> list[str]:
        return _variables(self.paths)

    def path(self, variable: str) -> pd.Series:
        return self.paths[variable].rename(variable)

    @api()
    def adjust(self, variable: str, values: Mapping[Any, float] | float, reason: str, how: str = "add") -> ForecastRound:
        """Judgement: add to (``how="add"``) or replace (``"set"``) the path of ``variable`` in some periods (a
        mapping) or all of them (a number). Each change is logged with ``reason``; the path before judgement is kept."""
        if variable not in self.variables:
            raise NotFoundError(f"no {variable!r} in the round; it has {self.variables}")
        if how not in ("add", "set"):
            raise SpecificationError("how must be 'add' or 'set'")
        if not reason.strip():
            raise SpecificationError("give a reason for the adjustment: it is part of the forecast's record")
        freq = self.paths.index.freq
        cells = {p: values for p in self.paths.index} if np.isscalar(values) else {pd.Period(str(k), freq=freq): float(v) for k, v in values.items()}
        paths = self.paths.copy()
        log = list(self.adjustments)
        for period, value in cells.items():
            if period not in paths.index:
                raise NotFoundError(f"{period} is not a forecast period of this round ({paths.index[0]} to {paths.index[-1]})")
            before = float(paths.at[period, variable])
            after = before + float(value) if how == "add" else float(value)
            paths.at[period, variable] = after
            log.append({"variable": variable, "period": str(period), "base": before, "value": after, "how": how,
                        "change": float(value), "reason": reason})
        return replace(self, paths=paths, adjustments=tuple(log))

    def judgement(self) -> Table:
        """Every adjustment: period, the path before, after, and why."""
        rows = [[a["variable"], a["period"], a["base"], a["value"], a["value"] - a["base"], a["reason"]] for a in self.adjustments]
        return Table(["variable", "period", "before", "after", "change", "reason"], rows, title=f"Judgement in {self.name}")

    def conditions_table(self) -> Table:
        """The conditions the round was made under: what, when, the value or range, and the shocks that met them."""
        rows = []
        for c in self.conditions:
            what = c.get("variable") or f"shock {c.get('shock')}"
            if "value" in c:
                cond = f"= {c['value']:g}"
            else:
                lo, hi = c.get("lower"), c.get("upper")
                cond = f"{lo:g} to {hi:g}" if lo is not None and hi is not None else (f">= {lo:g}" if lo is not None else f"<= {hi:g}")
            rows.append([what, c.get("period", "every period"), cond, ", ".join(c.get("shocks") or []) or "all"])
        return Table(["variable", "period", "condition", "met by shocks"], rows, title=f"Conditions of {self.name}")

    def table(self, variables: Sequence[str] | None = None) -> Table:
        cols = list(variables) if variables else self.variables
        frame = self.paths[cols].copy()
        frame.index = frame.index.astype(str)
        notes = [f"Made {self.made.replace('T', ' ')[:16]}" + (f"; data to {self.data_end}" if self.data_end else ""),
                 *([self.notes] if self.notes else []), *([f"{len(self.adjustments)} judgemental adjustments"] if self.adjustments else []),
                 *([f"Conditional on {len(self.conditions)} conditions (see conditions_table)"] if self.conditions else [])]
        return Table.from_frame(frame, title=self.name, notes=notes)

    def plot(self, data: Dataset | None = None, variables: Sequence[str] | None = None, history: int = 16,
             coverages: Sequence[float] = (0.3, 0.6, 0.9)) -> Any:
        """A fan chart of each variable, after its recent history: from ``data`` when given, else the history the
        round kept when it was made."""
        if data is None and self.history is not None:
            from .data import Dataset

            frame = self.history.copy()
            frame.index = _periods(frame.index)
            frame.index.name = "time"
            data = Dataset.from_frame(frame)
        return fan_figure(self.paths, data, variables or self.variables, coverages, history, title=self.name)

    def __repr__(self) -> str:
        span = f"{self.paths.index[0]} to {self.paths.index[-1]}" if len(self.paths) else "empty"
        return f"<ForecastRound {self.name!r}: {', '.join(self.variables)}, {span}, {len(self.adjustments)} adjustments>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


def _recorded_conditions(forecast: Any, periods: pd.Index) -> tuple[dict[str, Any], ...]:
    """The conditions a forecast was made under: a conditional forecast's, from its metadata, or a model plan's holds
    and ranges, from its scenario (the forecast periods only)."""
    meta = getattr(forecast, "meta", None)
    extra = getattr(meta, "extra", None) or {}
    if extra.get("conditions"):
        shocks = extra.get("shocks")
        return tuple({**dict(r), **({"shocks": list(shocks)} if shocks else {})} for r in extra["conditions"])
    sc = getattr(forecast, "scenario", None)
    if sc is None or not (getattr(sc, "hold", None) or getattr(sc, "ranges", False)):
        return ()
    span = _periods(periods)

    def cells(path: Any) -> dict[str, float]:
        if path is None:
            return {}
        if isinstance(path, (Mapping, pd.Series)):
            return {str(pd.Period(str(q), freq=span.freq)): float(x) for q, x in dict(path).items() if np.isfinite(float(x))}
        return {str(q): float(path) for q in span}

    rows: list[dict[str, Any]] = []
    free = {"shocks": list(sc.free)} if getattr(sc, "free", ()) else {}
    for v, path in sc.hold.items():
        rows += [{"variable": v, "period": q, "value": x, **free} for q, x in cells(path).items()]
    for v in dict.fromkeys([*sc.hold_lower, *sc.hold_upper]):
        lo, hi = cells(sc.hold_lower.get(v)), cells(sc.hold_upper.get(v))
        rows += [{"variable": v, "period": q, "lower": lo.get(q), "upper": hi.get(q), **free} for q in sorted({*lo, *hi})]
    return tuple(rows)


def fan_figure(paths: pd.DataFrame, data: Dataset | None = None, variables: Sequence[str] | None = None,
               coverages: Sequence[float] = (0.3, 0.6, 0.9), history: int = 16, title: str = "") -> Any:
    """Fan charts: central path with bands at each coverage, from ``{variable}_se`` (normal) or, if given,
    ``{variable}_lo{pct}`` and ``{variable}_hi{pct}`` columns; a ``Fan`` layer, after the recent history."""
    from .figures import PALETTE, Fan, Figure, Line, Panel

    frame = paths.copy()
    frame.index = _periods(frame.index)
    variables = list(variables) if variables else _variables(frame)
    panels = []
    for v in variables:
        layers: list[Any] = []
        own = frame
        if data is not None and v in data.names:
            past = data[v]
            past = past[past.index < frame.index[0]].dropna().iloc[-history:]
            if len(past):
                layers.append(Line(past, "actual", color="#444444"))
                own = _joined(frame, v, past)
        try:
            fan = Fan.from_frame(own, v, coverages, label="forecast", color=PALETTE[0], start=False)
            layers.insert(0, fan)  # the bands under the lines
        except SpecificationError:  # no bands: the central path alone
            layers.append(Line(own[v], "forecast", color=PALETTE[0]))
        # one legend, above the first panel clear of the data and the dates: the bands are the same in every panel
        panels.append(Panel(tuple(layers), title=v, legend=not panels, legend_loc="above"))
    if not panels:
        raise SpecificationError("no variables to plot")
    return Figure(tuple(panels), title=title, columns=2 if len(panels) > 3 else 1)


def _joined(frame: pd.DataFrame, name: str, past: pd.Series) -> pd.DataFrame:
    """``name``'s forecast columns with the last actual value put before them, so the forecast line and its fan start
    where the history ends (the bands from zero width there: the last actual is known)."""
    columns = [c for c in frame.columns if c == name or (isinstance(c, str) and re.fullmatch(re.escape(name) + r"_(se|lo\d+|hi\d+)", c))]
    own = frame[columns]
    last = past.index[-1]
    if last in own.index:
        return own
    row = {c: (0.0 if c.endswith("_se") and c != name else float(past.iloc[-1])) for c in columns}
    joined = pd.concat([pd.DataFrame([row], index=pd.PeriodIndex([last], freq=own.index.freq) if isinstance(own.index, pd.PeriodIndex)
                                     else [last]), own])
    joined.index.name = own.index.name
    return joined


# ---------------------------------------------------------------------------------------------
# a store of rounds


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not slug:
        raise SpecificationError("a round needs a name with letters or digits")
    return slug


class ForecastStore:
    """A folder of forecast rounds: one Parquet file of paths and one JSON record per round."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def _files(self, name: str) -> tuple[Path, Path]:
        slug = _slug(name)
        return self.folder / f"{slug}.parquet", self.folder / f"{slug}.json"

    @api()
    def save(self, rnd: ForecastRound, overwrite: bool = False) -> Path:
        data_file, record_file = self._files(rnd.name)
        if record_file.exists() and not overwrite:
            if self._same(self.load(rnd.name), rnd):  # saving the same round again (e.g. replaying a project) is harmless
                return record_file
            raise FileExistsError(f"a different round named {rnd.name!r} is already in {self.folder}; use overwrite=True, or a new name")
        self.folder.mkdir(parents=True, exist_ok=True)
        frame = pd.concat([rnd.paths, rnd.base.add_prefix("__base__")], axis=1)  # type: ignore[union-attr]
        frame.index = frame.index.astype(str)
        frame.to_parquet(data_file)
        record = {"name": rnd.name, "made": rnd.made, "data_end": rnd.data_end, "notes": rnd.notes, "freq": rnd.paths.index.freqstr,
                  "adjustments": list(rnd.adjustments), "source": dict(rnd.source), "conditions": list(rnd.conditions),
                  "history": None if rnd.history is None else {str(p): {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                                                               for p, row in rnd.history.iterrows()}}
        record_file.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        return record_file

    @staticmethod
    def _same(a: ForecastRound, b: ForecastRound) -> bool:
        """The same forecast: paths, judgement and data (when it was made aside)."""
        return (a.paths.shape == b.paths.shape and list(a.paths.columns) == list(b.paths.columns)
                and np.allclose(a.paths.to_numpy(float), b.paths.to_numpy(float), equal_nan=True)
                and list(a.adjustments) == [dict(x) for x in b.adjustments] and a.data_end == b.data_end)

    def names(self) -> list[str]:
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.folder.glob("*.json"))]
        return [r["name"] for r in sorted(records, key=lambda r: r["made"])]

    def load(self, name: str) -> ForecastRound:
        data_file, record_file = self._files(name)
        if not record_file.exists():
            raise NotFoundError(f"no round {name!r} in {self.folder}; rounds: {self.names()}")
        record = json.loads(record_file.read_text(encoding="utf-8"))
        frame = pd.read_parquet(data_file)
        frame.index = pd.PeriodIndex(frame.index, freq=record["freq"])
        base = frame[[c for c in frame.columns if c.startswith("__base__")]].rename(columns=lambda c: c[len("__base__"):])
        paths = frame[[c for c in frame.columns if not c.startswith("__base__")]]
        history = None
        if record.get("history"):
            history = pd.DataFrame.from_dict(record["history"], orient="index").astype(float)
            history.index = pd.PeriodIndex(history.index, freq=record["freq"])
        return ForecastRound(record["name"], paths, record["made"], record["data_end"], record["notes"],
                             tuple(record["adjustments"]), record["source"], base, history,
                             tuple(record.get("conditions", ())))  # rounds saved before conditions were recorded have none

    def rounds(self) -> Table:
        rows = []
        for name in self.names():
            r = self.load(name)
            rows.append([r.name, r.made, r.data_end or "", ", ".join(r.variables), len(r.adjustments), r.notes])
        return Table(["round", "made", "data to", "variables", "adjustments", "notes"], rows, title=f"Forecast rounds in {self.folder}")

    def revisions(self, old: str, new: str, variables: Sequence[str] | None = None) -> pd.DataFrame:
        """Two rounds side by side, with the revision (new less old), period by period: columns ``<v> (<round>)`` and
        ``<v> revision``, indexed by period."""
        a, b = self.load(old), self.load(new)
        names = list(variables) if variables else [v for v in b.variables if v in a.variables]
        frame = pd.DataFrame(index=a.paths.index.union(b.paths.index))
        for v in names:
            frame[f"{v} ({old})"] = a.paths[v]
            frame[f"{v} ({new})"] = b.paths[v]
            frame[f"{v} revision"] = b.paths[v] - a.paths[v]
        return frame

    @api()
    def compare(self, old: str, new: str, variables: Sequence[str] | None = None) -> Table:
        """``revisions`` as a table."""
        frame = self.revisions(old, new, variables)
        frame.index = frame.index.astype(str)
        return Table.from_frame(frame, title=f"Revisions from {old} to {new}")

    def history(self, variable: str) -> pd.DataFrame:
        """The path of ``variable`` in every round, one column per round."""
        return pd.DataFrame({name: self.load(name).paths.get(variable) for name in self.names() if variable in self.load(name).variables})

    def plot_history(self, variable: str, data: Dataset | None = None) -> Any:
        """Every round's path of ``variable`` against what happened (a "hedgehog" chart)."""
        from .figures import PALETTE, Figure, Line, Panel

        paths = self.history(variable)
        layers: list[Any] = []
        if data is not None and variable in data.names:
            layers.append(Line(data[variable].loc[paths.index.min() - 8 : paths.index.max()], "actual", color="#222222", width=2.2))
        for i, col in enumerate(paths.columns):
            layers.append(Line(paths[col].dropna(), col, color=PALETTE[(i % (len(PALETTE) - 1)) + 1], width=1.1))
        return Figure((Panel(tuple(layers), title=f"{variable}: forecasts by round"),))

    def accuracy(self, variable: str, data: Dataset, horizons: int | None = None) -> Table:
        """``scores`` as a table."""
        frame = self.scores(variable, data, horizons)
        rows = [[h, int(r["n"]), r["RMSE"], r["MAE"], r["mean error"]] for h, r in frame.iterrows()]
        return Table(["horizon", "n", "RMSE", "MAE", "mean error"], rows, title=f"Real-time accuracy of the rounds: {variable}")

    def scores(self, variable: str, data: Dataset, horizons: int | None = None) -> pd.DataFrame:
        """Real-time accuracy of the rounds by horizon (periods after each round's last data): n, RMSE, MAE, mean error."""
        rows: dict[int, list[float]] = {}
        actual = data[variable]
        for name in self.names():
            r = self.load(name)
            if variable not in r.variables or r.data_end is None:
                continue
            end = pd.Period(r.data_end, freq=r.paths.index.freq)
            for period, value in r.paths[variable].items():
                h = (period - end).n
                if (horizons is None or h <= horizons) and period in actual.index and np.isfinite(actual[period]):
                    rows.setdefault(h, []).append(float(actual[period] - value))
        out = {h: {"n": len(e), "RMSE": float(np.sqrt(np.mean(np.square(e)))), "MAE": float(np.mean(np.abs(e))), "mean error": float(np.mean(e))}
               for h, e in sorted(rows.items())}
        return pd.DataFrame.from_dict(out, orient="index", columns=["n", "RMSE", "MAE", "mean error"]).rename_axis("horizon")

    def __repr__(self) -> str:
        return f"<ForecastStore {self.folder}: {len(self.names()) if self.folder.exists() else 0} rounds>"


def _store_codec() -> None:
    from .codec import register_codec

    # a store is its folder: saved as text, written in code as ForecastStore('rounds')
    register_codec(ForecastStore, "pymetrix:ForecastStore", lambda store: str(store.folder), ForecastStore,
                   code=lambda store: f"ForecastStore({str(store.folder)!r})")


_store_codec()


# ---------------------------------------------------------------------------------------------
# backtests


def _refit(fitted: Any, window: int | None = None) -> Callable[[Any, Any], Any]:
    """How to re-estimate a fitted object's specification on data up to an origin: from its own start (an
    expanding window), or on the last ``window`` periods (a rolling window)."""
    from .estimation import FittedEquation
    from .var import FittedVAR

    def first(origin: Any, default: Any) -> Any:
        return str(origin - int(window) + 1) if window else default

    if isinstance(fitted, FittedEquation) or (hasattr(fitted, "equation") and hasattr(fitted, "method")):  # and machine-learning fits
        start = fitted.provenance.sample[0] if fitted.provenance and fitted.provenance.sample else None
        return lambda d, origin: fitted.equation.fit(d, span=(first(origin, start), str(origin)), method=fitted.method)
    if isinstance(fitted, FittedVAR):
        method = getattr(fitted, "method", None)  # a Bayesian VAR keeps its prior
        return lambda d, origin: fitted.spec.fit(d, span=(first(origin, None), str(origin)), method=method)
    if hasattr(getattr(fitted, "spec", None), "fit"):  # a fitted model that keeps its specification (exponential smoothing...)
        start = fitted.provenance.sample[0] if getattr(fitted, "provenance", None) and fitted.provenance.sample else None
        return lambda d, origin: fitted.spec.fit(d, span=(first(origin, start), str(origin)))
    if hasattr(fitted, "fit"):
        return lambda d, origin: fitted.fit(d, span=(first(origin, None), str(origin)))
    if callable(fitted):
        if window:
            raise SpecificationError("a rolling window needs a fitted equation or VAR, or a specification: a function chooses its own sample")
        return fitted
    raise TypeError("backtest needs a fitted equation or VAR, a specification with fit(data, span=...), or a function (data, origin) -> fit")


def _forecast_parts(fit: Any, data: Dataset, span: tuple[str, str], variable: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame] | None]:
    """A fit's forecasts over ``span``: the point forecasts (a column per variable), their predictive standard
    deviations (missing where the forecast gives none), and, for fits that simulate (``predictive_draws``),
    each variable's draws (periods x draws)."""
    if hasattr(fit, "predictive_draws"):
        periods, paths = fit.predictive_draws(data, span)
        names = list(fit.variables)
        means = pd.DataFrame(paths.mean(axis=0), index=periods, columns=names)
        sds = pd.DataFrame(paths.std(axis=0, ddof=1), index=periods, columns=names)
        return means, sds, {v: pd.DataFrame(paths[:, :, j].T, index=periods) for j, v in enumerate(names)}
    try:
        fc = fit.forecast(data, span, dynamic=True)
    except TypeError:
        fc = fit.forecast(data, span)
    frame = fc.frame
    if "forecast" in frame.columns:
        # target: what a plugin's fitted model named it before pymetrix's own became dependent
        name = variable or getattr(fit, "dependent", None) or getattr(fit, "target", None) or "forecast"
        means = frame[["forecast"]].rename(columns={"forecast": name})
        se = frame["se"] if "se" in frame.columns else pd.Series(np.nan, index=frame.index)
        if se.isna().all():  # dynamic equation forecasts have no standard errors yet (stochastic simulation is planned)
            try:
                static = fit.forecast(data, span, dynamic=False).frame
            except (TypeError, ValueError):
                static = None
            if (static is not None and "se" in static.columns and static.index.equals(frame.index)  # empty past the data's end
                    and np.allclose(static["forecast"], frame["forecast"], rtol=0, atol=1e-10)):
                se = static["se"]  # no lag of the target: the static forecast is the dynamic one, with its standard errors
        return means, pd.DataFrame({name: se}, index=frame.index), None
    names = _variables(frame)
    sds = pd.DataFrame({v: frame[f"{v}_se"] if f"{v}_se" in frame.columns else np.nan for v in names}, index=frame.index)
    return frame[names], sds, None


def _outcomes(data: Dataset, name: str, cache: dict[str, pd.Series]) -> pd.Series:
    """What happened to ``name`` in the latest data: a series, or an expression of series such as dlog(gdp)
    (the target of a model averaging or a transformed equation). Missing when neither."""
    if name not in cache:
        if name in data.names:
            cache[name] = data.frame[name]
        else:
            try:
                cache[name] = data.eval(name)
            except Exception:  # noqa: BLE001 - not an expression of the data's series: no outcomes to compare with
                cache[name] = pd.Series(dtype=float)
    return cache[name]


def _density_row(actual: float, mean: float, sd: float, draws: np.ndarray | None) -> dict[str, Any]:
    """How a forecast's density scored against the outcome: from its draws, or as a normal with its standard deviation."""
    from .stats.density import (
        crps_normal,
        crps_sample,
        log_score_normal,
        log_score_sample,
        pit_normal,
        pit_sample,
    )

    out: dict[str, Any] = {"density": "", "log score": np.nan, "crps": np.nan, "pit": np.nan}
    if draws is not None and np.sum(np.isfinite(draws)) >= 2:
        out["density"] = "draws"
        if np.isfinite(actual):
            out.update({"log score": log_score_sample(actual, draws), "crps": crps_sample(actual, draws), "pit": pit_sample(actual, draws)})
    elif np.isfinite(sd) and sd > 0:
        out["density"] = "normal"
        if np.isfinite(actual):
            out.update({"log score": float(log_score_normal(actual, mean, sd)), "crps": float(crps_normal(actual, mean, sd)),
                        "pit": float(pit_normal(actual, mean, sd))})
    return out


_LOSSES = ("squared", "absolute", "lin-lin", "linex", "crps", "log score")
_DENSITY_BASIS = {"normal": "normal densities, from the forecasts' standard errors",
                  "draws": "the simulated draws (a kernel density for the log score)",
                  "pool": "the linear pool (weighted mixture) of the members' normal densities"}


@dataclass(frozen=True, eq=False)
class Backtest:
    """Forecasts from each origin, with what happened: one row per origin, horizon and variable.

    ``errors`` has the origin, target, horizon, variable, forecast, actual and error of each forecast, and, where
    the forecast has a density, its standard deviation (``sd``), how the density was formed (``density``) and its
    scores (``log score``, ``crps``, ``pit``). A combination (``combine_forecasts``) also has its ``weights``."""

    errors: pd.DataFrame
    label: str = ""
    provenance: Any = None  # data vintages, origins, method, fingerprint of the data errors are measured against
    weights: Any = None  # a combination's weights: origin, variable, horizon, member, weight, history

    def _rows(self, variable: str, horizon: int) -> pd.DataFrame:
        e = self.errors[(self.errors["variable"] == variable) & (self.errors["horizon"] == int(horizon))]
        if e.empty:
            raise NotFoundError(f"no forecasts of {variable} at horizon {horizon} in {self.label or 'this backtest'}")
        return e.set_index("target")

    def _has_density(self) -> bool:
        return "crps" in self.errors.columns and bool(self.errors["crps"].notna().any())

    def scores(self, variables: Sequence[str] | None = None) -> pd.DataFrame:
        """n, RMSE, MAE and mean error by variable and horizon (a two-level index)."""
        e = self.errors.dropna(subset=["error"])
        if variables:
            e = e[e["variable"].isin(variables)]
        g = e.groupby(["variable", "horizon"])["error"]
        return pd.DataFrame({"n": g.size(), "RMSE": g.apply(lambda x: float(np.sqrt(np.mean(x**2)))),
                             "MAE": g.apply(lambda x: float(np.mean(np.abs(x)))), "mean error": g.mean()})

    def accuracy(self, variables: Sequence[str] | None = None) -> Table:
        """``scores`` as a table."""
        frame = self.scores(variables)
        rows = [[int(r["n"]), r["RMSE"], r["MAE"], r["mean error"]] for _, r in frame.iterrows()]
        index = [f"{v}, h={h}" for v, h in frame.index]
        return Table(["n", "RMSE", "MAE", "mean error"], rows, index=index, title=f"Backtest accuracy{': ' + self.label if self.label else ''}")

    def losses(self, variable: str, horizon: int, loss: str = "squared", tau: float = 0.5, a: float = 1.0) -> pd.Series:
        """The loss of each forecast of ``variable`` at ``horizon``, by target period: the ``"squared"`` or
        ``"absolute"`` error, the asymmetric ``"lin-lin"`` (with ``tau``) or ``"linex"`` (with ``a``) loss of the error,
        the ``"crps"``, or the negative ``"log score"`` (so that lower is better for all of them)."""
        if loss not in _LOSSES:
            raise SpecificationError(f"loss must be one of {', '.join(repr(x) for x in _LOSSES)}")
        e = self._rows(variable, horizon)
        if loss == "squared":
            out = e["error"] ** 2
        elif loss == "absolute":
            out = e["error"].abs()
        elif loss in ("lin-lin", "linex"):
            from .stats.evaluation import linex_loss, linlin_loss

            out = pd.Series(linlin_loss(e["error"], tau) if loss == "lin-lin" else linex_loss(e["error"], a), index=e.index)
        else:
            if not self._has_density():
                raise SpecificationError(f"{self.label or 'these forecasts'} have no densities to score ({loss}): no standard errors or draws")
            out = e["crps"] if loss == "crps" else -e["log score"]
        return out.dropna().rename(self.label or "loss")

    @api()
    def compare(self, other: Backtest, variable: str, horizon: int, loss: str = "squared", test: str = "diebold-mariano",
                tau: float = 0.5, a: float = 1.0) -> TestResult:
        """Test whether these forecasts and ``other``'s are equally accurate at one horizon, on the targets both forecast.

        ``test``: ``"diebold-mariano"`` (with the Harvey-Leybourne-Newbold correction; a negative statistic favours
        these forecasts), ``"giacomini-white"`` (conditional predictive ability), or ``"clark-west"`` when these
        forecasts come from a model that ``other``'s model nests (does the larger model forecast better?).
        ``loss`` (not for Clark-West, which uses squared errors): ``"squared"``, ``"absolute"``, ``"lin-lin"`` (``tau``),
        ``"linex"`` (``a``), ``"crps"`` or ``"log score"``; the last two compare density forecasts. See ``losses``."""
        if test == "clark-west":
            from .stats.evaluation import clark_west

            a, b = self._rows(variable, horizon), other._rows(variable, horizon)
            both = pd.concat([a["actual"].rename("y"), a["forecast"].rename("small"), b["forecast"].rename("large")], axis=1, join="inner").dropna()
            if len(both) < 5:
                raise SpecificationError("fewer than five common forecast targets to compare")
            return clark_west(both["y"], both["small"], both["large"], horizon)
        both = pd.concat([self.losses(variable, horizon, loss, tau, a).rename("a"), other.losses(variable, horizon, loss, tau, a).rename("b")],
                         axis=1).dropna()
        if len(both) < 3:
            raise SpecificationError("fewer than three common forecast targets to compare")
        if test == "giacomini-white":
            from .stats.evaluation import giacomini_white

            return giacomini_white(both["a"], both["b"], horizon)
        if test != "diebold-mariano":
            raise SpecificationError("test must be 'diebold-mariano', 'giacomini-white' or 'clark-west'")
        return _dm_result((both["a"] - both["b"]).to_numpy(), horizon, loss)

    @api()
    def encompasses(self, other: Backtest, variable: str, horizon: int = 1) -> TestResult:
        """Do these forecasts encompass ``other``'s (Harvey, Leybourne and Newbold), on the targets both forecast?"""
        from .stats.evaluation import encompassing

        a, b = self._rows(variable, horizon), other._rows(variable, horizon)
        both = pd.concat([a["actual"].rename("y"), a["forecast"].rename("f1"), b["forecast"].rename("f2")], axis=1, join="inner").dropna()
        return encompassing(both["y"], both["f1"], both["f2"], horizon)

    @api()
    def spa(self, models: Mapping[str, Backtest], variable: str, horizon: int = 1, loss: str = "squared", tau: float = 0.5,
            a: float = 1.0, reps: int = 1000, block: int | None = None, seed: int = 0) -> TestResult:
        """Hansen's (2005) test for superior predictive ability, these forecasts the benchmark: does any of ``models``
        (backtests, by name) forecast ``variable`` at ``horizon`` with lower expected loss? Each backtest's ``losses``
        (``loss``, ``tau``, ``a`` as there) go to ``pymetrix.spa``, with its stationary bootstrap (``reps``, mean
        ``block`` length, ``seed``). Needs the arch package."""
        from .stats.evaluation import spa

        if not models:
            raise SpecificationError("give at least one backtest to compare with the benchmark")
        return spa(self.losses(variable, horizon, loss, tau, a),
                   {str(k): m.losses(variable, horizon, loss, tau, a) for k, m in dict(models).items()}, horizon, reps, block, seed)

    def efficiency(self, variable: str, horizon: int = 1) -> TestResult:
        """Mincer-Zarnowitz test that the forecasts are efficient: outcome = a + b forecast, with a = 0 and b = 1."""
        from .stats.evaluation import mincer_zarnowitz

        e = self._rows(variable, horizon).dropna(subset=["actual"])
        return mincer_zarnowitz(e["actual"], e["forecast"], horizon)

    def density_scores(self, variables: Sequence[str] | None = None) -> pd.DataFrame:
        """By variable and horizon: n, the average log score (higher is better), the average CRPS (lower is better,
        in the variable's units), and the shares of outcomes inside the central 50% and 90% predictive intervals
        (0.5 and 0.9 when the densities are well calibrated). Empty when the forecasts have no densities."""
        columns = ["n", "log score", "CRPS", "in 50%", "in 90%"]
        if not self._has_density():
            return pd.DataFrame(columns=columns)
        e = self.errors.dropna(subset=["crps"])
        if variables:
            e = e[e["variable"].isin(variables)]
        g = e.groupby(["variable", "horizon"])

        def inside(c: float) -> Callable[[pd.Series], float]:
            return lambda p: float(np.mean((p > (1 - c) / 2) & (p < 1 - (1 - c) / 2)))

        return pd.DataFrame({"n": g.size(), "log score": g["log score"].mean(), "CRPS": g["crps"].mean(),
                             "in 50%": g["pit"].apply(inside(0.5)), "in 90%": g["pit"].apply(inside(0.9))})[columns]

    def density_accuracy(self, variables: Sequence[str] | None = None) -> Table:
        """``density_scores`` as a table, saying how the densities were formed."""
        frame = self.density_scores(variables)
        rows = [[int(r["n"]), r["log score"], r["CRPS"], r["in 50%"], r["in 90%"]] for _, r in frame.iterrows()]
        kinds = sorted(set(self.errors["density"].dropna()) - {""}) if "density" in self.errors.columns else []
        notes = ["Log score: higher is better. CRPS: lower is better, in the variable's units. In 50% and 90%: the shares of "
                 "outcomes inside the central predictive intervals."]
        if kinds:
            notes.append("Densities: " + "; ".join(_DENSITY_BASIS.get(k, k) for k in kinds) + ".")
        return Table(["n", "log score", "CRPS", "in 50%", "in 90%"], rows, index=[f"{v}, h={h}" for v, h in frame.index],
                     title=f"Density forecast scores{': ' + self.label if self.label else ''}", notes=notes)

    def pit(self, variable: str, horizon: int = 1) -> pd.Series:
        """The probability integral transforms of the forecasts of ``variable`` at ``horizon``, by target period."""
        if not self._has_density():
            raise SpecificationError(f"{self.label or 'these forecasts'} have no densities: no standard errors or draws")
        return self._rows(variable, horizon)["pit"].dropna().rename("PIT")

    def pit_test(self, variable: str, horizon: int = 1) -> TestResult:
        """Berkowitz test that the PITs are independent and uniform (for one-step forecasts: h-step PITs are correlated)."""
        from .stats.density import berkowitz

        result = berkowitz(self.pit(variable, horizon))
        return replace(result, details={**dict(result.details), "variable": variable, "horizon": int(horizon)})

    def interval_test(self, variable: str, horizon: int = 1, coverage: float = 0.9) -> TestResult:
        """Christoffersen test of the central ``coverage`` predictive intervals: the right share of outcomes falls
        outside them, and the misses do not cluster."""
        from .stats.evaluation import christoffersen

        u = self.pit(variable, horizon)
        hits = ((u < (1 - coverage) / 2) | (u > 1 - (1 - coverage) / 2)).astype(int)
        result = christoffersen(hits, coverage)
        return replace(result, details={**dict(result.details), "variable": variable, "horizon": int(horizon)})

    def evaluation(self, variables: Sequence[str] | None = None, coverage: float = 0.9) -> Table:
        """The tests of each variable and horizon at a glance (p-values): Mincer-Zarnowitz efficiency, and for
        density forecasts the Berkowitz test of the PITs and the Christoffersen test of the ``coverage`` intervals."""
        e = self.errors.dropna(subset=["error"])
        names = list(variables) if variables else list(dict.fromkeys(e["variable"]))
        dens = self._has_density()

        def p(make: Callable[[], Any]) -> float:
            try:
                return float(make().p_value)
            except (SpecificationError, NotFoundError, np.linalg.LinAlgError):
                return float("nan")  # too few forecasts for the test

        rows, index = [], []
        for v in names:
            for h in sorted(e.loc[e["variable"] == v, "horizon"].unique()):
                row = [int(((e["variable"] == v) & (e["horizon"] == h)).sum()), p(lambda: self.efficiency(v, h))]
                if dens:
                    row += [p(lambda: self.pit_test(v, h)), p(lambda: self.interval_test(v, h, coverage))]
                rows.append(row)
                index.append(f"{v}, h={h}")
        columns = ["n", "efficiency (MZ)"] + (["PIT (Berkowitz)", f"{coverage:.0%} intervals (Christoffersen)"] if dens else [])
        notes = ["p-values: small values reject efficiency, calibrated densities, or correct interval coverage. Blank: too few forecasts."]
        if dens:
            notes.append("The Berkowitz and Christoffersen tests assume independent PITs and misses: read them with care beyond one step ahead.")
        return Table(columns, rows, index=index, title=f"Forecast evaluation tests{': ' + self.label if self.label else ''}", notes=notes)

    def weights_table(self) -> Table:
        """A combination's average weight on each member, by variable and horizon."""
        if self.weights is None or len(self.weights) == 0:
            raise SpecificationError("this backtest is not a combination: it has no weights")
        members = list(dict.fromkeys(self.weights["member"]))
        frame = self.weights.groupby(["variable", "horizon", "member"], sort=False)["weight"].mean().unstack("member")[members]
        rows = [[float(r[m]) for m in members] for _, r in frame.iterrows()]
        return Table(members, rows, index=[f"{v}, h={h}" for v, h in frame.index], title="Average combination weights",
                     notes=("Weights are estimated at each origin from the outcomes known then; equal until there is enough history.",))

    def plot_pit(self, variables: Sequence[str] | None = None, horizon: int = 1, bins: int = 10) -> Any:
        """Histograms of the PITs at ``horizon``, as densities (1 throughout for well-calibrated forecasts): a hump
        means the predictive densities are too wide, a U too narrow, a slope a biased forecast."""
        from .figures import Bars, Figure, Panel, Rule

        if not self._has_density():
            raise SpecificationError(f"{self.label or 'these forecasts'} have no densities: no standard errors or draws")
        e = self.errors[self.errors["horizon"] == int(horizon)].dropna(subset=["pit"])
        names = list(variables) if variables else list(dict.fromkeys(e["variable"]))
        edges = np.linspace(0, 1, bins + 1)
        panels = []
        for v in names:
            u = e.loc[e["variable"] == v, "pit"].to_numpy(float)
            heights = pd.Series(np.histogram(u, bins=edges)[0] * bins / max(len(u), 1), index=np.round((edges[:-1] + edges[1:]) / 2, 3))
            panels.append(Panel((Bars(heights, "PIT"), Rule(1.0, "--")), title=f"{v}, h={horizon} ({len(u)} forecasts)", xlabel="PIT", legend=False))
        return Figure(tuple(panels), title="PIT histograms", columns=2 if len(panels) > 3 else 1)

    def plot(self, variables: Sequence[str] | None = None) -> Any:
        """RMSE by horizon for each variable."""
        from .figures import Bars, Figure, Panel

        e = self.errors.dropna(subset=["error"])
        names = list(variables) if variables else list(dict.fromkeys(e["variable"]))
        panels = []
        for v in names:
            rmse = e[e["variable"] == v].groupby("horizon")["error"].apply(lambda x: float(np.sqrt(np.mean(x**2))))
            panels.append(Panel((Bars(rmse, "RMSE"),), title=v, xlabel="horizon"))
        return Figure(tuple(panels), columns=2 if len(panels) > 3 else 1)

    def __repr__(self) -> str:
        origins = self.errors["origin"].nunique() if len(self.errors) else 0
        return f"<Backtest {self.label}: {origins} origins, horizons 1-{self.errors['horizon'].max() if len(self.errors) else 0}>"


@renamed_arguments(target="variable")
@api("pymetrix:backtest", version=2)
def backtest(fitted: Any, data: Dataset, origins: tuple[Any, Any], horizon: int = 8, vintages: Mapping[Any, Dataset] | None = None,
             variable: str | None = None, label: str = "", window: int | None = None) -> Backtest:
    """Re-estimate at each origin from ``origins[0]`` to ``origins[1]`` and forecast ``horizon`` periods ahead.

    ``fitted`` is a fitted equation or VAR (its specification and method are re-estimated on data up to each
    origin), another fitted model that keeps its specification (``spec``, e.g. exponential smoothing), a
    specification with ``fit(data, span=...)``, or a function ``(data, origin) -> fit``. With
    ``vintages`` (origin -> the dataset as it was then) each origin uses the data available at the time;
    errors are always measured against ``data`` (the latest release). ``window``: estimate on the last
    ``window`` periods to each origin (a rolling window) instead of all of them (an expanding window). ``variable``
    names a single-equation forecast's column (default: the fit's dependent variable); it was called ``target``.

    Each forecast keeps its predictive standard deviation (``sd``) and its density is scored against the outcome
    (``log score``, ``crps``, ``pit``): as a normal, or from the simulated draws of fits that make them (Bayesian
    VARs). See ``density_scores`` and ``evaluation``."""
    if data.time_axis is None:
        raise SpecificationError("backtests need time-series data")
    if horizon < 1:
        raise SpecificationError("horizon must be 1 or more")
    if window is not None and int(window) < 2:
        raise SpecificationError("a rolling window needs at least two periods")
    refit = _refit(fitted, window)
    freq = data.frame.index.freq
    start, end = pd.Period(str(origins[0]), freq=freq), pd.Period(str(origins[1]), freq=freq)
    vint = {pd.Period(str(k), freq=freq): v for k, v in (vintages or {}).items()}
    rows = []
    outcomes: dict[str, pd.Series] = {}
    from .progress import report

    origins = pd.period_range(start, end, freq=freq)
    fits: dict[str, Any] = {}  # origin -> the fit made there, for the backtest's provenance
    for k, origin in enumerate(origins):
        report(k / max(len(origins), 1), f"forecast origin {origin} ({k + 1} of {len(origins)})")  # Cancel stops here
        d = vint.get(origin, data)
        if vintages is not None and origin not in vint:
            raise NotFoundError(f"no vintage for origin {origin}")
        fit = refit(d, origin)
        fits[str(origin)] = fit
        span = (str(origin + 1), str(origin + horizon))
        means, sds, draws = _forecast_parts(fit, d, span, variable)
        for v in _variables(means):
            for period, value in means[v].items():
                h = (period - origin).n
                actual = float(_outcomes(data, v, outcomes).get(period, np.nan))
                sd = float(sds[v].get(period, np.nan)) if v in sds.columns else np.nan
                row = {"origin": str(origin), "target": period, "horizon": h, "variable": v, "forecast": float(value),
                       "actual": actual, "error": float(actual - value), "sd": sd}
                row.update(_density_row(actual, float(value), sd, None if draws is None else draws[v].loc[period].to_numpy(float)))
                rows.append(row)
    from ._capture import data_fingerprint
    from ._version import __version__
    from .metadata import Provenance

    errors = pd.DataFrame(rows)
    measured = sorted(set(errors["variable"])) if len(errors) else []
    refits = _refit_record(fits)
    waiting = errors[errors["actual"].isna()] if len(errors) else errors
    spec = {"what": type(fitted).__name__, "origins": [str(o) for o in origins], "horizon": horizon,
            "real time": vintages is not None, "window": f"rolling, {int(window)} periods" if window else "expanding"}
    if refits["spec"]:
        spec["model"] = refits["spec"]
    spec["re-estimates"] = _reestimates("origins", fits)
    prov = Provenance(data_vintages={n: data.variables[n].vintage for n in measured if n in data.variables},
                      spec=spec,
                      method={"type": "pseudo out-of-sample backtest", "vintages": len(vintages or {}), **refits["method"]},
                      sample=(str(origins[0]), str(origins[-1])) if len(origins) else None, nobs=len(errors) - len(waiting), dropped=len(waiting),
                      package_version=__version__, data_fingerprint=data_fingerprint(data, measured),
                      warnings=refits["warnings"], messages=refits["messages"], converged=refits["converged"],
                      backend=refits["backend"], options=refits["options"],
                      rows=[[str(origins[0]), str(origins[-1])]] if len(origins) else None,
                      dropped_rows=_waiting_rows(waiting),
                      vintage_request=_vintage_requests(data, vint))
    return Backtest(errors, label, prov)


@call_migration("pymetrix:backtest", 1)
def _backtest_variable(arguments: dict[str, Any]) -> dict[str, Any]:  # version 2: target became variable
    arguments["variable"] = arguments.pop("target", None)
    return arguments


def _reestimates(over: str, fits: Mapping[Any, Any], **more: Any) -> dict[str, Any]:
    """A multi-fit result's re-estimates, for its provenance (``views.sample_text`` reads it): what they run over
    (``"origins"``, ``"windows ending"``), how many, and the estimation samples of the first and the last."""
    samples = [tuple(p.sample) for f in fits.values() if (p := getattr(f, "provenance", None)) is not None and p.sample]
    out: dict[str, Any] = {"over": over, "count": len(fits), **{k: v for k, v in more.items() if v}}
    if samples:
        out["first"], out["last"] = [str(x) for x in samples[0]], [str(x) for x in samples[-1]]
    return out


def _refit_record(fits: Mapping[str, Any]) -> dict[str, Any]:
    """What the fits made at each origin say about themselves, for a backtest's provenance: the model and method
    re-estimated, the library and its options (one entry when every origin used the same), and the warnings and
    messages, each with the origins where it arose."""
    provs = {o: getattr(f, "provenance", None) for o, f in fits.items()}
    provs = {o: p for o, p in provs.items() if p is not None}

    def same(field: str) -> Any:
        values = [getattr(p, field, None) for p in provs.values()]
        values = [v for v in values if v]
        if not values:
            return None
        first = values[0]
        return first if all(v == first for v in values) else {"by origin": {o: getattr(p, field, None) for o, p in provs.items()}}

    def gathered(field: str) -> tuple[str, ...]:
        where: dict[str, list[str]] = {}
        for o, p in provs.items():
            for text in getattr(p, field, ()) or ():
                where.setdefault(text, []).append(o)
        return tuple(f"{text} (origin{'s' if len(os) > 1 else ''} {', '.join(os)})" for text, os in where.items())

    model = next(iter(provs.values()), None)
    method = {"refit": dict(model.method)} if model is not None and model.method else {}
    converged = [p.converged for p in provs.values() if p.converged is not None]
    return {"spec": dict(model.spec) if model is not None and model.spec else {}, "method": method,
            "backend": same("backend"), "options": same("options"), "warnings": gathered("warnings"),
            "messages": gathered("messages"), "converged": all(converged) if converged else None}


def _waiting_rows(waiting: pd.DataFrame) -> dict[str, Any] | None:
    """The forecasts that have no outcome yet: not scored, listed as dropped rows (the first 50)."""
    if not len(waiting):
        return None
    rows = [[f"{r.variable} at {r.target} from {r.origin}", "no outcome yet"] for r in waiting.itertuples()]
    out: dict[str, Any] = {"by_reason": {"no outcome yet": len(rows)}, "rows": rows[:50]}
    if len(rows) > 50:
        out["more"] = len(rows) - 50
    return out


def _vintage_requests(data: Dataset, vintages: Mapping[Any, Dataset]) -> dict[str, Any]:
    """Which data were asked for: the outcomes' (the latest release, as asked), and each origin's when they differ."""
    out: dict[str, Any] = {"outcomes": dict(data.meta.vintage or {"policy": "unknown"})}
    if vintages:
        out["origins"] = {str(o): dict(d.meta.vintage or {"policy": "unknown"}) for o, d in vintages.items()}
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, horizon: int = 1, loss: str = "squared", small_sample: bool = True) -> TestResult:
    """Diebold-Mariano (1995) test that two forecasts are equally accurate, from their errors at one horizon.

    The loss differential's variance allows for the MA(h-1) correlation of h-step errors; ``small_sample``
    applies the Harvey, Leybourne and Newbold (1997) correction and Student-t p-values. A negative statistic
    means the first forecast is more accurate."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if e1.shape != e2.shape or e1.ndim != 1:
        raise SpecificationError("give two error series of the same length")
    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise SpecificationError("loss must be 'squared' or 'absolute'")
    return _dm_result(d, horizon, loss, small_sample)


def _dm_result(d: np.ndarray, horizon: int, loss: str, small_sample: bool = True) -> TestResult:
    """Diebold-Mariano on a loss differential (first less second): a negative statistic favours the first."""
    from .stats import TestResult
    from .stats.evaluation import mean_differential_test

    stat, df, p = mean_differential_test(d, horizon, small_sample)
    return TestResult("Diebold-Mariano" + (" (HLN)" if small_sample else ""), "the two forecasts are equally accurate",
                      stat, df, p, "t" if small_sample else "normal",
                      details={"mean loss differential": float(np.mean(d)), "observations": len(d), "horizon": int(horizon), "loss": loss})


# ---------------------------------------------------------------------------------------------
# forecast combination


_COMBINATIONS = ("equal", "inverse_mse", "regression")


@api("pymetrix:combine_forecasts")
def combine_forecasts(backtests: Mapping[str, Backtest] | Sequence[Backtest], method: str = "equal", min_history: int = 4,
                      label: str = "") -> Backtest:
    """Combine the forecasts of several backtests into one, and evaluate it as a backtest of its own.

    ``method``: ``"equal"`` (the simple average, hard to beat in practice), ``"inverse_mse"`` (weights in
    proportion to 1 / MSE: Bates and Granger without correlations) or ``"regression"`` (least squares weights
    that sum to one: Granger and Ramanathan; they can be negative). Weights are estimated in real time, for each
    variable and horizon: at each origin only from forecasts whose outcomes were known by then (targets up to the
    origin), and equal until ``min_history`` such forecasts exist (and more than the number of members, for
    regression weights). Forecasts are matched on origin, target, horizon and variable.

    Where every member has a predictive standard deviation and no weight is negative, the combination's density is
    the linear pool of the members' normal densities (their weighted mixture), and it is scored. ``weights`` holds
    every weight used."""
    from functools import reduce

    from ._version import __version__
    from .metadata import Provenance
    from .stats.density import crps_mixture, log_score_mixture, pit_mixture

    if method not in _COMBINATIONS:
        raise SpecificationError(f"method must be one of {', '.join(_COMBINATIONS)}")
    if isinstance(backtests, Mapping):
        members = {str(k): v for k, v in backtests.items()}
    else:
        members = {}
        for i, bt in enumerate(backtests):
            name = bt.label or f"forecast {i + 1}"
            members[name if name not in members else f"{name} ({i + 1})"] = bt
    if len(members) < 2:
        raise SpecificationError("give at least two backtests to combine")
    keys = ["origin", "target", "horizon", "variable"]
    frames = []
    for i, bt in enumerate(members.values()):
        e = bt.errors
        f = e[keys + ["forecast", "actual"]].copy()
        f["sd"] = e["sd"] if "sd" in e.columns else np.nan
        frames.append(f.rename(columns={"forecast": f"f{i}", "actual": f"a{i}", "sd": f"s{i}"}))
    merged = reduce(lambda a, b: a.merge(b, on=keys, how="inner"), frames)
    if merged.empty:
        raise SpecificationError("the backtests have no forecasts in common (the same origins, targets, horizons and variables)")
    k, names = len(members), list(members)
    rows, weight_rows = [], []
    for (variable, horizon), g in merged.groupby(["variable", "horizon"], sort=False):
        g = g.assign(_o=[t - int(horizon) for t in g["target"]]).sort_values("_o")
        F = g[[f"f{i}" for i in range(k)]].to_numpy(float)
        S = g[[f"s{i}" for i in range(k)]].to_numpy(float)
        A = g["a0"].to_numpy(float)
        targets, origins = list(g["target"]), list(g["_o"])
        for r in range(len(g)):
            known = np.array([t <= origins[r] for t in targets]) & np.isfinite(A) & np.all(np.isfinite(F), axis=1)
            w, used = _combination_weights(method, F[known], A[known], min_history)
            mean, actual = float(F[r] @ w), float(A[r])
            row = {"origin": str(g["origin"].iloc[r]), "target": targets[r], "horizon": int(horizon), "variable": variable,
                   "forecast": mean, "actual": actual, "error": actual - mean, "sd": np.nan, "density": "",
                   "log score": np.nan, "crps": np.nan, "pit": np.nan}
            if np.all(w >= 0) and np.all(np.isfinite(S[r])) and np.all(S[r] > 0):
                row["sd"] = float(np.sqrt(max(np.sum(w * (S[r] ** 2 + F[r] ** 2)) - mean**2, 0.0)))
                row["density"] = "pool"
                if np.isfinite(actual):
                    row.update({"log score": log_score_mixture(actual, w, F[r], S[r]), "crps": crps_mixture(actual, w, F[r], S[r]),
                                "pit": pit_mixture(actual, w, F[r], S[r])})
            rows.append(row)
            weight_rows += [{"origin": row["origin"], "variable": variable, "horizon": int(horizon), "member": names[i],
                             "weight": float(w[i]), "history": used} for i in range(k)]
    errors = pd.DataFrame(rows)
    order = [t - h for t, h in zip(errors["target"], errors["horizon"])]
    errors = errors.assign(_o=order).sort_values(["_o", "variable", "horizon"], kind="stable").drop(columns="_o").reset_index(drop=True)
    first = next(iter(members.values())).provenance
    prov = Provenance(data_vintages=dict(getattr(first, "data_vintages", None) or {}),
                      spec={"what": "forecast combination", "members": names, "method": method, "min history": min_history,
                            "re-estimates": {"over": "origins", "note": f"{len(names)} backtests combined"}},
                      method={"type": "forecast combination", "weights": method, "estimated": "in real time, from the outcomes known at each origin"},
                      sample=(str(errors["origin"].iloc[0]), str(errors["origin"].iloc[-1])), nobs=len(errors), dropped=0,
                      package_version=__version__, data_fingerprint=getattr(first, "data_fingerprint", None),
                      backend=_by_member(members, "backend"), options=_by_member(members, "options"),
                      warnings=tuple(f"{name}: {w}" for name, bt in members.items() for w in (getattr(bt.provenance, "warnings", ()) or ())),
                      messages=tuple(f"{name}: {m}" for name, bt in members.items() for m in (getattr(bt.provenance, "messages", ()) or ())),
                      vintage_request=getattr(first, "vintage_request", None))
    return Backtest(errors, label or f"{method.replace('_', ' ')} combination", prov, weights=pd.DataFrame(weight_rows))


def _by_member(members: Mapping[str, Backtest], field: str) -> dict[str, Any] | None:
    """A combination's members' ``backend`` or ``options``, by member (None when no member says)."""
    found = {name: getattr(bt.provenance, field, None) for name, bt in members.items()}
    return {"by member": found} if any(found.values()) else None


def _combination_weights(method: str, F: np.ndarray, a: np.ndarray, min_history: int) -> tuple[np.ndarray, int]:
    """Combination weights from past forecasts F (n x k) and their outcomes a, and how many past forecasts they used."""
    n, k = F.shape
    equal = np.full(k, 1.0 / k)
    if method == "equal" or n < max(int(min_history), 1):
        return equal, n
    if method == "inverse_mse":
        mse = np.mean((a[:, None] - F) ** 2, axis=0)
        w = (mse == 0).astype(float) if np.any(mse == 0) else 1 / mse
        return w / w.sum(), n
    if n <= k:  # regression weights need more forecasts than members
        return equal, n
    # least squares with weights summing to one: the KKT system of min |a - F w|^2 subject to 1'w = 1
    kkt = np.block([[2 * F.T @ F, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
    rhs = np.concatenate([2 * F.T @ a, [1.0]])
    return np.linalg.lstsq(kkt, rhs, rcond=None)[0][:k], n
