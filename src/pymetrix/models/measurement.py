"""Measurement maps: observed data series as functions of a model's variables.

    m = MeasurementMap(["infl = pi_ss + 4*pi",              # annualised inflation from quarterly model inflation
                        "rate = r_ss + pi_ss + 4*i",
                        "growth = g + 4*(y - y[-1])"],      # GDP growth from the output gap: a lag is fine
                       params={"pi_ss": 2, "r_ss": 1, "g": 2}, errors={"growth": 0.5})
    m.apply(solution)                                       # model paths in data units (exact, nonlinear)
    solution.system(m, shock_sd)                            # the state-space form, measurement linearised

Professional models are rarely observed directly: the data are annualised growth rates, year-on-year
inflation, interest rates in percent, while the model works in quarterly deviations or logs. A measurement
map states that link once, in the model's own equation language, so it is checked, shown in algebraic
notation, used by the Kalman filter, and used again to report model results in the units of the data.

Each equation has an observed series on the left (its name; ``sources`` gives the data series if it is
called something else) and an expression in model variables, their lags and the map's parameters on the
right. Leads are not allowed: an observation depends on the present and the past. ``errors`` gives the
standard deviation of each series' measurement error (none by default). A map may have no observed series and only
report some (``MeasurementMap([], reported=[...])``): it reports, but gives no state-space form and cannot be estimated.

**From raw data and back** (``transforms``). An observed series is often made from a raw one: annualised growth
from the level of GDP, ``transforms={"growth": "400*dlog(gdp)"}``. The filter, the likelihood and estimation then take
the raw data (``observed(data)`` makes the observed series), and forecasts report the raw series too, rebuilt from
the observed ones (``reverse``): each period the transform is solved for the raw series' value, given its own past
and any other series it uses, so no inverse has to be written, and any transform that moves with the raw series
works (``100*(x/x[-4] - 1)``, ``log(x/pop)``). The series rebuilt is the one the transform uses, or with several,
the one named in ``sources``.

**Reported, not observed** (``reported``, DSGE.jl's pseudo-observables). Quantities to report alongside the
observed series, in the same equation language, with labels and units, but never fitted to data: the output gap,
the natural rate, annualised real rates. ``reported=["gap = 100*(y - y_pot)", "r_star = 4*rn + 2"]``. They appear with
the Kalman-smoothed history, forecasts and every draw of a stochastic forecast.

    m = MeasurementMap(["growth = g + 4*(y - y[-1])", "infl = pi_ss + 4*pi"], params={"g": 2, "pi_ss": 2},
                       transforms={"growth": "400*dlog(gdp)", "infl": "400*dlog(cpi)"},
                       reported=["gap = 100*y"], labels={"gap": "output gap, percent"})
    m.observed(raw)                 # growth and infl, made from gdp and cpi
    m.report(forecast, raw)         # observed series, reported ones, and gdp and cpi rebuilt
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..expr import Expr, Name, diff, evaluate, parse_equation, parse_expression
from ..expr.nodes import Num
from ..output import Table
from .solvers import ModelError

__all__ = ["MeasurementMap", "Measurement"]

_REVERSE_TOL = 1e-12


@dataclass(frozen=True)
class Measurement:
    """One measurement equation: ``observable = expression``."""

    observable: str
    rhs: Expr
    text: str


@dataclass(frozen=True, eq=False)
class MeasurementMap:
    """Observed series as functions of model variables: see the module docstring."""

    code_positional: ClassVar[int] = 1  # generated code gives the equations by position: MeasurementMap([...])
    equations: Sequence[str]
    params: Mapping[str, float] = field(default_factory=dict)
    errors: Mapping[str, float] = field(default_factory=dict)  # measurement-error standard deviations
    sources: Mapping[str, str] = field(default_factory=dict)  # data series names, where not the observable's name
    units: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    trends: Mapping[str, str] = field(default_factory=dict)  # deterministic terms added to observed series (IRIS !dtrends)
    trend_base: str | None = None  # the period where ttrend is 0, for trends that use it
    transforms: Mapping[str, str] = field(default_factory=dict)  # observed series made from raw data: {"growth": "400*dlog(gdp)"}
    reported: Sequence[str] = ()  # reported, not observed: "gap = 100*(y - y_pot)" (pseudo-observables)

    measurements: tuple[Measurement, ...] = field(init=False, repr=False)
    trend_exprs: Mapping[str, Any] = field(init=False, repr=False)
    reports: tuple[Measurement, ...] = field(init=False, repr=False)
    transform_exprs: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.equations, str):
            object.__setattr__(self, "equations", tuple(line.strip() for line in self.equations.splitlines() if line.strip()))
        built = []
        for text in self.equations:
            lhs, op, rhs = parse_equation(text, list(self.params))
            if op != "=" or not isinstance(lhs, Name) or lhs.lag != 0:
                raise ModelError(f"write a measurement as 'observed = expression in model variables', not {text!r}")
            leads = sorted({str(r) for r in rhs.refs() if r.lag > 0})
            if leads:
                raise ModelError(f"{text!r}: an observation cannot depend on the future ({', '.join(leads)})")
            built.append(Measurement(lhs.id, rhs, text))
        if isinstance(self.reported, str):
            object.__setattr__(self, "reported", tuple(line.strip() for line in self.reported.splitlines() if line.strip()))
        reports = []
        for text in self.reported:
            lhs, op, rhs = parse_equation(text, list(self.params))
            if op != "=" or not isinstance(lhs, Name) or lhs.lag != 0:
                raise ModelError(f"write a reported series as 'name = expression in model variables', not {text!r}")
            leads = sorted({str(r) for r in rhs.refs() if r.lag > 0})
            if leads:
                raise ModelError(f"{text!r}: a reported series is worked out from the present and the past, not {', '.join(leads)}")
            reports.append(Measurement(lhs.id, rhs, text))
        names = [m.observable for m in built]
        everything = names + [r.observable for r in reports]
        dups = sorted({n for n in everything if everything.count(n) > 1})
        if dups:
            raise ModelError(f"more than one equation for {dups} (observed and reported series need names of their own)")
        for what, mapping, allowed in (("errors", self.errors, names), ("sources", self.sources, names), ("units", self.units, everything),
                                       ("labels", self.labels, everything), ("trends", self.trends, names),
                                       ("transforms", self.transforms, names)):
            extra = sorted(set(mapping) - set(allowed))
            if extra:
                raise ModelError(f"{what} given for {extra}, which are not {'observed' if allowed is names else 'observed or reported'} "
                                 f"series: {allowed}")
        transforms = {}
        for name, text in self.transforms.items():
            expr = parse_expression(str(text), list(self.params))
            series = sorted(expr.variables())
            leads = sorted({str(r) for r in expr.refs() if r.lag > 0})
            if leads or not series:
                raise ModelError(f"the transform of {name}, {text!r}, is made from raw series, their present and past"
                                 + (f", not {', '.join(leads)}" if leads else ""))
            raw = self.sources.get(name) or (series[0] if len(series) == 1 else None)
            if raw is None or raw not in series:
                raise ModelError(f"the transform of {name} uses {', '.join(series)}: name the one it is rebuilt into, "
                                 f"sources={{{name!r}: ...}}")
            if diff(expr, Name(raw, 0)) == Num(0.0):
                raise ModelError(f"the transform of {name} does not move with {raw} in the same period, so {raw} cannot be rebuilt from it")
            transforms[name] = (expr, raw)
        object.__setattr__(self, "equations", tuple(self.equations))
        object.__setattr__(self, "reported", tuple(self.reported))
        object.__setattr__(self, "measurements", tuple(built))
        object.__setattr__(self, "reports", tuple(reports))
        object.__setattr__(self, "transform_exprs", transforms)
        exprs = {}
        for name, text in self.trends.items():
            expr = parse_expression(str(text), list(self.params))
            other = sorted(expr.variables() - {"ttrend"})
            if other:
                raise ModelError(f"the trend of {name} may use parameters and ttrend only, not {other}")
            if "ttrend" in expr.variables() and self.trend_base is None:
                raise ModelError(f"the trend of {name} uses ttrend: give trend_base, the period where ttrend is 0 (IRIS's baseYear)")
            exprs[name] = expr
        object.__setattr__(self, "trend_exprs", exprs)

    @classmethod
    def direct(cls, observables: Sequence[str], errors: Mapping[str, float] | None = None) -> MeasurementMap:
        """Model variables observed as they are, in levels: ``x = x`` for each."""
        return cls([f"{o} = {o}" for o in observables], errors=dict(errors or {}))

    @property
    def observables(self) -> tuple[str, ...]:
        return tuple(m.observable for m in self.measurements)

    def variables(self) -> frozenset[str]:
        """The model variables the measurements use."""
        return frozenset().union(*(m.rhs.variables() for m in self.measurements)) if self.measurements else frozenset()

    @property
    def max_lag(self) -> int:
        return max((-r.lag for m in self.measurements for r in m.rhs.refs()), default=0)

    def source(self, observable: str) -> str:
        """The data series behind an observed series: its own name, the one in ``sources``, or with a transform, the
        raw series the transform rebuilds."""
        if observable in self.transform_exprs:
            return self.transform_exprs[observable][1]
        return self.sources.get(observable, observable)

    def data_series(self) -> tuple[str, ...]:
        """Every data series the observed series are made from."""
        out: list[str] = []
        for o in self.observables:
            names = sorted(self.transform_exprs[o][0].variables()) if o in self.transform_exprs else [self.source(o)]
            out += [n for n in names if n not in out]
        return tuple(out)

    def observed(self, data: Any, span: Any = None) -> pd.DataFrame:
        """The observed series in the model's units, made from ``data``: each its data series (``sources``), or its
        transform of raw series. A frame on the data's periods (within ``span``)."""
        from ..errors import SpecificationError

        missing = [n for n in self.data_series() if n not in data.frame.columns]
        if missing:
            raise SpecificationError(f"the data have no {', '.join(missing)} (observed series of the measurement equations)")
        frame = data.frame
        out = pd.DataFrame(index=frame.index)
        for o in self.observables:
            if o in self.transform_exprs:
                out[o] = np.asarray(evaluate(self.transform_exprs[o][0], data.lagged, dict(self.params)), float)
            else:
                out[o] = frame[self.source(o)].to_numpy(float)
        return out[data.sample_mask(span)] if span is not None else out

    def reported_values(self, data: Any) -> pd.DataFrame:
        """The reported series (``reported``) from model paths (a ``Solution`` or a dataset of the model's variables)."""
        missing = sorted(set().union(*(r.rhs.variables() for r in self.reports)) - set(data.names)) if self.reports else []
        if missing:
            raise ModelError(f"the reported series need {missing}, which are not in the paths")
        return pd.DataFrame({r.observable: np.asarray(evaluate(r.rhs, data.lagged, dict(self.params)), float) * np.ones(len(data.frame))
                             for r in self.reports}, index=data.frame.index)

    def with_reported(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` (model variables in levels, by period) with the reported series added: for smoothed histories."""
        if not self.reports:
            return frame
        from ..data import Dataset

        return frame.join(self.reported_values(Dataset.from_frame(frame)).set_axis(frame.index))

    def reverse(self, observed: pd.DataFrame, data: Any) -> pd.DataFrame:
        """The raw series rebuilt from observed ones (``transforms``): where ``observed`` has a value and the data do
        not (a forecast, beyond the data), each period the transform is solved for the raw series, given its past and
        the other series it uses. Columns: the raw series; rows: ``observed``'s periods. See the module docstring."""
        return self._reverse(list(observed.columns), observed.index, observed.to_numpy(float)[None], data)[0]

    def _reverse(self, names: Sequence[str], index: pd.Index, values: np.ndarray, data: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """``reverse`` for draws: ``values`` is draws x periods x ``names``; returns the first draw as a frame and every
        draw's raw series (draws x periods x raw series)."""
        D = values.shape[0]
        todo = [o for o in self.observables if o in self.transform_exprs and o in names]
        periods = pd.PeriodIndex(data.frame.index).union(pd.PeriodIndex(index))
        position = {p: i for i, p in enumerate(periods)}
        rows = np.array([position[p] for p in index])
        needed = sorted(set().union(*(self.transform_exprs[o][0].variables() for o in todo))) if todo else []
        series = {n: np.tile(data.frame[n].reindex(periods).to_numpy(float) if n in data.frame else np.full(len(periods), np.nan), (D, 1))
                  for n in needed}
        params = dict(self.params)
        for t in np.argsort(rows):
            pos = rows[t]
            for o in todo:
                expr, raw = self.transform_exprs[o]
                target = values[:, t, list(names).index(o)]
                current = series[raw][:, pos]
                unknown = np.isnan(current) & np.isfinite(target)
                if not unknown.any():
                    continue
                slope = diff(expr, Name(raw, 0))

                def resolve(name: str, lag: int, x: np.ndarray = current) -> np.ndarray:
                    if name == raw and lag == 0:
                        return x
                    src = pos + lag
                    return series[name][:, src] if 0 <= src < len(periods) else np.full(D, np.nan)

                previous = series[raw][:, pos - 1] if pos > 0 else np.full(D, np.nan)
                x = np.where(np.isfinite(previous), previous, 1.0)
                for _ in range(100):  # Newton's method on the transform, draw by draw at once
                    f = np.asarray(evaluate(expr, lambda n, lag: resolve(n, lag, x), params), float) - target
                    g = np.asarray(evaluate(slope, lambda n, lag: resolve(n, lag, x), params), float) * np.ones(D)
                    step = np.where(unknown & (g != 0), f / np.where(g != 0, g, 1.0), 0.0)
                    x = x - step
                    if np.all(np.abs(step[unknown]) <= _REVERSE_TOL * (1 + np.abs(x[unknown]))):
                        break
                series[raw][:, pos] = np.where(unknown, x, current)
        raws = [self.transform_exprs[o][1] for o in todo]
        stacked = np.stack([series[r][:, rows] for r in raws], axis=2) if raws else np.zeros((D, len(index), 0))
        return pd.DataFrame(stacked[0], index=index, columns=raws), stacked

    def report(self, paths: Any, data: Any = None) -> Any:
        """Model paths in the units of the data: the observed series (``apply``), the reported ones, and with ``data``
        and ``transforms``, the raw series rebuilt beyond the data (``reverse``). A ``Dataset``."""
        from ..data import Dataset

        frame = self.apply(paths).frame
        if self.reports:
            frame = frame.join(self.reported_values(paths))
        if data is not None and self.transform_exprs:
            frame = frame.join(self.reverse(frame[list(self.observables)], data))
        return Dataset.from_frame(frame)

    def with_errors(self, errors: Mapping[str, float]) -> MeasurementMap:
        """The same map with these measurement-error sizes (the equations are not parsed again: estimation calls this
        at every trial value)."""
        extra = sorted(set(errors) - set(self.observables))
        if extra:
            raise ModelError(f"errors given for {extra}, which are not observed series: {list(self.observables)}")
        new = copy.copy(self)
        object.__setattr__(new, "errors", {**self.errors, **errors})
        return new

    def trend_values(self, index: Any) -> np.ndarray:
        """The deterministic trends of the observed series in these periods (periods x observables; zero for series
        without one): ``ttrend`` counts periods from ``trend_base``."""
        out = np.zeros((len(index), len(self.measurements)))
        if not self.trend_exprs:
            return out
        periods = pd.PeriodIndex(index)
        ttrend = periods.asi8 - pd.Period(str(self.trend_base), freq=periods.freq).ordinal if self.trend_base is not None else np.zeros(len(index))
        names = [m.observable for m in self.measurements]
        for name, expr in self.trend_exprs.items():
            value = evaluate(expr, lambda n, lag: ttrend + lag, dict(self.params))
            out[:, names.index(name)] = np.broadcast_to(np.asarray(value, dtype=float), (len(index),))
        return out

    def apply(self, data: Any) -> Any:
        """The observed series implied by model paths (a ``Solution`` or any dataset of the model's variables),
        in the units of the data: evaluated exactly, not linearised."""
        from ..data import Dataset

        missing = sorted(self.variables() - set(data.names))
        if missing:
            raise ModelError(f"the measurement equations need {missing}, which are not in the data")
        frame = pd.DataFrame({m.observable: np.asarray(evaluate(m.rhs, data.lagged, dict(self.params)), float) for m in self.measurements},
                             index=data.frame.index)
        if self.trend_exprs:
            frame = frame + self.trend_values(frame.index)
        return Dataset.from_frame(frame)

    def linearize(self, states: Sequence[tuple[str, int]], steady: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """``Z`` and ``d`` such that observed ≈ d + Z (state deviations), around the steady state: each state is a
        model variable at a lag, ``(name, lag)``. Exact when the measurement equations are linear."""
        values = {**{k: float(v) for k, v in steady.items()}}
        resolve = lambda name, lag: values[name]  # noqa: E731 - the steady state: every lag at the same value
        index = {s: j for j, s in enumerate(states)}
        Z = np.zeros((len(self.measurements), len(states)))
        d = np.zeros(len(self.measurements))
        for i, m in enumerate(self.measurements):
            rhs = m.rhs.substitute(self.params)
            d[i] = float(evaluate(rhs, resolve))
            for ref in rhs.refs():
                key = (ref.id, ref.lag)
                if key not in index:
                    raise ModelError(f"{m.text!r}: {ref} is not a state of the system")
                Z[i, index[key]] += float(evaluate(diff(rhs, ref), resolve))
        return Z, d

    def table(self) -> Table:
        from ..notation import expression

        rows = []
        for m in (*self.measurements, *self.reports):
            try:
                algebraic = f"{expression(Name(m.observable, 0))} = {expression(m.rhs)}"
            except Exception:
                algebraic = ""
            observed = m in self.measurements
            data = (self.transforms.get(m.observable) or self.source(m.observable)) if observed else "(reported)"
            rows.append([m.text, algebraic, self.errors.get(m.observable, 0.0) if observed else None, data,
                         self.units.get(m.observable, ""), self.labels.get(m.observable, "")])
        notes = [f"Parameters: {', '.join(f'{k} = {v:g}' for k, v in self.params.items())}"] if self.params else []
        return Table(["equation", "algebraic", "error s.d.", "data series", "units", "label"], rows,
                     index=[*self.observables, *(r.observable for r in self.reports)],
                     title="Measurement equations", notes=notes)

    def __repr__(self) -> str:
        reported = f"; reported {', '.join(r.observable for r in self.reports)}" if self.reports else ""
        if not self.observables:
            return f"<MeasurementMap: reported {', '.join(r.observable for r in self.reports)}>"
        return f"<MeasurementMap: {', '.join(self.observables)}{reported}>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()

