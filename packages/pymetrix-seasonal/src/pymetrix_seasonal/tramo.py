"""TRAMO-SEATS (Gómez and Maravall, Bank of Spain), as Eurostat's JDemetra+ 3 computes it, in Java
(``pymetrix_seasonal.jdemetra``): one of its predefined specifications (RSA0 to RSAfull), changed by the settings
given. Its series record the model, outliers and calendar effects TRAMO found (``tramo_model``, ``tramo_log``,
``tramo_outliers``, ``tramo_calendar``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix.commands import Widget
from pymetrix.errors import DataError, SpecificationError

from .x13 import _use_log

__all__ = ["TramoSeats"]


_TS_SPECS = {"rsafull": "RSAfull", "rsa5": "RSA5", "rsa4": "RSA4", "rsa3": "RSA3", "rsa2": "RSA2", "rsa1": "RSA1", "rsa0": "RSA0"}
_COMPONENTS = {"calendar": "CalendarEffect", "irregular": "Irregular", "trend": "Trend", "seasonal": "Seasonal"}
_SEATS_DEFAULTS = {"approximation": "legacy", "seats_method": "burman", "trend_boundary": 0.5, "seasonal_boundary": 0.8,
                   "seasonal_boundary_pi": 0.8, "seasonal_tolerance": 2.0, "xl": 0.95}  # JDemetra+'s own (DecompositionSpec.DEFAULT)


@dataclass(frozen=True)
class TramoSeats:
    """TRAMO-SEATS (Gómez and Maravall, Bank of Spain), as Eurostat's JDemetra+ 3 computes it, in Java (see
    ``pymetrix.jdemetra``): TRAMO fits a regARIMA model (logs or levels, the ARIMA model, outliers, trading-day and
    Easter effects), and SEATS splits the linearised series into trend, seasonal and irregular components by signal
    extraction from that model.

    ``spec`` is one of JDemetra+'s predefined specifications, from RSA0 (levels, the airline model, nothing else) to
    RSAfull (the log test, an automatic model, outliers, trading days and Easter, each pretested); the other settings
    change it, and each left empty or at its default keeps the specification's own choice."""

    label: ClassVar[str] = "TRAMO-SEATS"
    needs_periods: ClassVar[bool] = True  # decompose() is given the series' periods (pymetrix.seasonal)
    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment",)

    spec: Annotated[Literal["rsafull", "rsa5", "rsa4", "rsa3", "rsa2", "rsa1", "rsa0"], Widget(
        "choice", help="JDemetra+'s specification: RSAfull, RSA5 and RSA4 choose the model automatically; RSA0 to RSA3 the airline model")] = "rsafull"
    log: Annotated[bool | None, Widget("choice", help="multiplicative (logs) or additive; empty = the specification's (TRAMO's test; levels in RSA0)")] = None
    outlier: Annotated[bool | None, Widget("choice", help="detect outliers; empty = the specification's (all but RSA0)")] = None
    trading: Annotated[Literal["spec", "none", "working", "trading", "auto"], Widget(
        "choice", help="trading-day effects, with leap year, each pretested: working days, six trading-day contrasts, or the better by "
                       "an F-test; spec = the specification's")] = "spec"
    easter: Annotated[bool | None, Widget("choice", help="an Easter effect, pretested; empty = the specification's (RSA2, RSA4, RSA5, RSAfull)")] = None
    libraries: Annotated[str | None, Widget("text", advanced=True,
                                            help="folder of the JDemetra+ jars; empty = JDPLUS_PATH or where pymetrix_seasonal.jdemetra.install() put them")] = None
    arima: Annotated[str, Widget("text", advanced=True, help="an ARIMA model such as (0 1 1)(0 1 1); empty = the specification's")] = ""
    outlier_types: Annotated[str, Widget("text", advanced=True, help="outlier types: ao ls tc so; empty = the specification's (ao ls tc)")] = ""
    critical: Annotated[float | None, Widget("number", min=2, max=10, step=0.1, advanced=True,
                                             help="outlier critical t-value; empty = set from the series' length")] = None
    easter_days: Annotated[int, Widget("number", min=1, max=15, advanced=True, help="days the Easter effect lasts")] = 6
    approximation: Annotated[Literal["legacy", "none", "noisy"], Widget(
        "choice", advanced=True, group="SEATS", help="when the model does not decompose: approximate it as TRAMO-SEATS always has, "
                                                     "not at all, or with a noisy trend")] = "legacy"
    seats_method: Annotated[Literal["burman", "kalman"], Widget("choice", advanced=True, group="SEATS",
                                                                help="components by Burman's algorithm or the Kalman smoother")] = "burman"
    trend_boundary: Annotated[float, Widget("number", min=0, max=1, step=0.05, advanced=True, group="SEATS",
                                            help="AR roots above this modulus go to the trend (rmod)")] = 0.5
    seasonal_boundary: Annotated[float, Widget("number", min=0, max=1, step=0.05, advanced=True, group="SEATS",
                                               help="seasonal AR roots above this modulus go to the seasonal (smod)")] = 0.8
    seasonal_boundary_pi: Annotated[float, Widget("number", min=0, max=1, step=0.05, advanced=True, group="SEATS",
                                                  help="the same, for the root at the frequency pi")] = 0.8
    seasonal_tolerance: Annotated[float, Widget("number", min=0, max=10, step=0.5, advanced=True, group="SEATS",
                                                help="degrees from a seasonal frequency that still count as seasonal (epsphi)")] = 2.0
    xl: Annotated[float, Widget("number", min=0.9, max=1, step=0.01, advanced=True, group="SEATS",
                                help="MA roots above this modulus are made unit roots (xl)")] = 0.95
    interventions: Annotated[str, Widget("text", advanced=True, group="Regression",
                                         help="outliers known in advance, kept without testing: ao, ls, tc or so and the period, "
                                              "e.g. ls 2020Q2, ao 2008Q4")] = ""
    regressors: Annotated[str, Widget("text", advanced=True, group="Regression",
                                      help="series of the dataset as regressors (holidays, strikes, policy changes), each with the "
                                           "component its effect goes to: calendar (the default: removed from the adjusted "
                                           "series), irregular, trend or seasonal; e.g. holidays, strike=irregular")] = ""

    def __post_init__(self) -> None:
        if self.spec not in _TS_SPECS:
            raise SpecificationError(f"spec is one of {', '.join(_TS_SPECS)}")
        if self.trading not in ("spec", "none", "working", "trading", "auto"):
            raise SpecificationError("trading is spec, none, working, trading or auto")
        if self.arima and not re.fullmatch(r"\s*\(\s*\d+\s+\d+\s+\d+\s*\)\s*\(\s*\d+\s+\d+\s+\d+\s*\)\s*", self.arima):
            raise SpecificationError("arima is a model such as (0 1 1)(0 1 1)")
        if any(k not in ("ao", "ls", "tc", "so") for k in self.outlier_types.lower().split()):
            raise SpecificationError("outlier_types are among ao, ls, tc and so")
        if self.outlier is False and (self.outlier_types.strip() or self.critical is not None):
            raise SpecificationError("outlier types and the critical value are for outlier detection: tick outliers, or leave them empty")
        if self.approximation not in ("legacy", "none", "noisy") or self.seats_method not in ("burman", "kalman"):
            raise SpecificationError("approximation is legacy, none or noisy; seats_method burman or kalman")
        self.fixed_outliers()
        self.user_regressors()

    def fixed_outliers(self) -> list[tuple[str, str]]:
        """``interventions`` as (type, period) pairs: ``"ls 2020Q2, ao 2008Q4"`` gives [("ls", "2020Q2"), ("ao", "2008Q4")]."""
        out = []
        for part in (p.strip() for p in self.interventions.split(",") if p.strip()):
            m = re.fullmatch(r"(ao|ls|tc|so)\s+(\S+)", part, re.IGNORECASE)
            if not m:
                raise SpecificationError(f"interventions: {part!r} is not a type (ao, ls, tc or so) and a period, as ls 2020Q2")
            out.append((m.group(1).lower(), m.group(2)))
        return out

    def user_regressors(self) -> list[tuple[str, str]]:
        """``regressors`` as (series, component) pairs: ``"holidays, strike=irregular"`` gives [("holidays", "calendar"),
        ("strike", "irregular")]."""
        out = []
        for part in (p.strip() for p in self.regressors.split(",") if p.strip()):
            name, _, component = (s.strip() for s in part.partition("="))
            component = component.lower() or "calendar"
            if not name.isidentifier() or component not in _COMPONENTS:
                raise SpecificationError(f"regressors: {part!r} is not a series name, optionally with a component "
                                         f"({', '.join(_COMPONENTS)}), as strike=irregular")
            out.append((name, component))
        return out

    def inputs(self) -> tuple[str, ...]:
        """The dataset's series the adjustment uses besides the series adjusted: the regressors."""
        return tuple(name for name, _ in self.user_regressors())

    def changes(self) -> dict[str, Any]:
        """The settings that change the specification: those not at their defaults."""
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ("spec", "libraries") and getattr(self, f.name) != f.default}

    def specification(self, J: Any, period: int, index: pd.PeriodIndex | None = None) -> Any:
        """JDemetra+'s TramoSeatsSpec: the predefined ``spec`` with this object's changes. User regressors refer to the
        context ``decompose`` fills (group ``r``); ``index``, the series' periods, places the interventions."""
        T = "jdplus.tramoseats.base.api."
        base = getattr(J(T + "tramoseats.TramoSeatsSpec"), _TS_SPECS[self.spec])
        tramo = base.getTramo()
        build = tramo.toBuilder()
        if self.log is not None:
            kind = J("jdplus.toolkit.base.api.modelling.TransformationType")
            build.transform(tramo.getTransform().toBuilder().function(kind.Log if self.log else kind.valueOf("None")).build())
        if self.arima.strip():
            p, d, q, bp, bd, bq = (int(v) for v in re.findall(r"\d+", self.arima))
            sarima = J("jdplus.toolkit.base.api.arima.SarimaSpec").builder().period(period).p(p).d(d).q(q).bp(bp).bd(bd).bq(bq).build()
            build.arima(sarima).autoModel(J(T + "tramo.AutoModelSpec").DEFAULT_DISABLED)
        outliers = tramo.getOutliers()
        if self.outlier is not None:
            spec = J(T + "tramo.OutlierSpec")
            outliers = spec.DEFAULT_ENABLED if self.outlier else spec.DEFAULT_DISABLED
        if outliers.isUsed() and (self.outlier_types.strip() or self.critical is not None):
            kinds = set(self.outlier_types.lower().split()) or {k for k in ("ao", "ls", "tc", "so") if getattr(outliers, f"is{k.title()}")()}
            ob = outliers.toBuilder().ao("ao" in kinds).ls("ls" in kinds).tc("tc" in kinds).so("so" in kinds)
            outliers = (ob.criticalValue(float(self.critical)) if self.critical is not None else ob).build()
        elif self.outlier_types.strip() or self.critical is not None:
            raise SpecificationError(f"{self.spec.upper()} detects no outliers: tick outliers to give their types or critical value")
        build.outliers(outliers)
        regression = tramo.getRegression()
        calendar = regression.getCalendar()
        cal = calendar.toBuilder()
        if self.trading != "spec":
            td = J(T + "tramo.TradingDaysSpec")
            kind, length = J("jdplus.toolkit.base.api.timeseries.calendars.TradingDaysType"), J(
                "jdplus.toolkit.base.api.timeseries.calendars.LengthOfPeriodType").LeapYear
            test = J(T + "tramo.RegressionTestType").Separate_T
            cal.tradingDays({"none": lambda: td.none(), "working": lambda: td.td(kind.TD2, length, test, False),
                             "trading": lambda: td.td(kind.TD7, length, test, False),
                             "auto": lambda: td.automatic(length, J(T + "tramo.TradingDaysSpec$AutoMethod").FTEST, 0.01, False)}[self.trading]())
        easter = calendar.getEaster()
        if self.easter is not None:
            es = J(T + "tramo.EasterSpec")
            easter = es.builder().type(J(T + "tramo.EasterSpec$Type").IncludeEaster).duration(self.easter_days).test(True).build() \
                if self.easter else es.DEFAULT_UNUSED
        elif self.easter_days != 6:
            if not easter.isUsed():
                raise SpecificationError(f"{self.spec.upper()} has no Easter effect: tick Easter to set its days")
            easter = easter.toBuilder().duration(self.easter_days).build()
        cal.easter(easter)
        regression_build = regression.toBuilder().calendar(cal.build())
        R = "jdplus.toolkit.base.api.timeseries.regression."
        variable, component = J(R + "Variable"), J("jdplus.sa.base.api.ComponentType")
        for kind, when in self.fixed_outliers():
            freq = index.freq if index is not None else ("Q" if period == 4 else "M")
            try:
                p = pd.Period(when, freq=freq)
            except (ValueError, TypeError):
                raise SpecificationError(f"interventions: {when!r} is not a period of this series") from None
            if index is not None and not (index[0] <= p <= index[-1]):
                raise SpecificationError(f"interventions: {kind} {when} is outside the series ({index[0]} to {index[-1]})")
            at = J("java.time.LocalDateTime").of(p.start_time.year, p.start_time.month, 1, 0, 0)
            core = {"ao": lambda: J(R + "AdditiveOutlier")(at), "ls": lambda: J(R + "LevelShift")(at, True),
                    "tc": lambda: J(R + "TransitoryChange")(at, 0.7), "so": lambda: J(R + "PeriodicOutlier")(at, period, True)}[kind]()
            label = f"{kind.upper()} ({p.start_time.year}-{p.start_time.month:02d})"
            regression_build.outlier(variable.variable(label, core))
        for name, effect in self.user_regressors():
            user = variable.variable(name, J(R + "TsContextVariable")(f"r.{name}"))
            regression_build.userDefinedVariable(J("jdplus.sa.base.api.SaVariable").setComponentType(user, getattr(component, _COMPONENTS[effect])))
        build.regression(regression_build.build())
        seats = base.getSeats().toBuilder()
        if self.approximation != "legacy":
            seats.approximationMode(getattr(J(T + "seats.DecompositionSpec$ModelApproximationMode"), self.approximation.title()))
        if self.seats_method != "burman":
            seats.method(J(T + "seats.DecompositionSpec$ComponentsEstimationMethod").KalmanSmoother)
        for name, setter in (("trend_boundary", "trendBoundary"), ("seasonal_boundary", "seasBoundary"),
                             ("seasonal_boundary_pi", "seasBoundaryAtPi"), ("seasonal_tolerance", "seasTolerance"), ("xl", "xlBoundary")):
            if getattr(self, name) != _SEATS_DEFAULTS[name]:
                getattr(seats, setter)(float(getattr(self, name)))
        return base.toBuilder().tramo(build.build()).seats(seats.build()).build()

    def decompose(self, y: np.ndarray, period: int, name: str, index: pd.PeriodIndex | None = None,
                  notes: dict[str, str] | None = None, regressors: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, bool]:
        """The seasonal factor (or component), from JDemetra+'s adjusted series; ``notes``, if given, gets the model
        TRAMO chose, the outliers it found, the regression effects, the specification and the JDemetra+ version.
        ``regressors`` gives the values of the series named in ``regressors``, over the same periods as ``y``."""
        from . import jdemetra

        if self.log:
            _use_log(True, y, name)
        if index is None:
            raise SpecificationError("TRAMO-SEATS needs the series' periods")
        J = jdemetra.classes(self.libraries)
        import jpype

        start = index[0]
        ts = J("jdplus.toolkit.base.r.timeseries.TsUtility").of(period, start.year, start.month if period == 12 else start.quarter,
                                                                 jpype.JArray(jpype.JDouble)(np.asarray(y, dtype=float)))
        spec = self.specification(J, period, index)
        R = "jdplus.toolkit.base.api.timeseries.regression."
        context = J(R + "ModellingContext").getActiveContext()
        if self.user_regressors():
            context = J(R + "ModellingContext")()  # a context of its own: the regressors of this series only
            suppliers = J(R + "TsDataSuppliers")()
            for regressor, _ in self.user_regressors():
                values = np.asarray((regressors or {}).get(regressor, ()), dtype=float)
                if len(values) != len(y):
                    raise SpecificationError(f"TRAMO-SEATS on {name}: the regressor {regressor} was not given for its periods")
                if not np.all(np.isfinite(values)):
                    raise DataError(f"TRAMO-SEATS on {name}: the regressor {regressor} has missing values in {name}'s sample")
                data = J("jdplus.toolkit.base.r.timeseries.TsUtility").of(
                    period, start.year, start.month if period == 12 else start.quarter, jpype.JArray(jpype.JDouble)(values))
                suppliers.set(regressor, J("jdplus.toolkit.base.api.timeseries.StaticTsDataSupplier")(data))
            context.getTsVariableManagers().set("r", suppliers)
        try:
            result = J("jdplus.tramoseats.base.r.TramoSeats").process(ts, spec, context)
        except Exception as exc:  # noqa: BLE001 - a Java exception: its message
            raise DataError(f"TRAMO-SEATS could not adjust {name}: {exc}") from None
        if result is None or not result.isValid():
            raise DataError(f"TRAMO-SEATS could not adjust {name}: JDemetra+ returned no result, and says no more "
                            "(it does so when TRAMO cannot estimate a model, as for a constant series)")

        def get(key: str) -> Any:
            return result.getData(key, J("java.lang.Object")) if result.contains(key) else None

        sa = np.array(list(result.getData("sa", J("jdplus.toolkit.base.api.timeseries.TsData")).getValues().toArray()), dtype=float)
        if len(sa) != len(y):
            raise DataError(f"TRAMO-SEATS returned {len(sa)} adjusted values for the {len(y)} of {name}")
        multiplicative = str(get("mode")) != "Additive"
        if notes is not None:
            orders = [int(get(f"arima.{k}")) for k in ("p", "d", "q", "bp", "bd", "bq")]
            described = [str(v) for v in (get("regression.details.description") or [])]
            fixed = {f"{k.upper()} ({pd.Period(w, freq=index.freq).start_time.year}-{pd.Period(w, freq=index.freq).start_time.month:02d})"
                     for k, w in self.fixed_outliers()}
            found = [v for v in described if re.match(r"(AO|LS|TC|SO) \(", v) and v not in fixed]
            notes.update({
                "backend": f"TRAMO-SEATS (JDemetra+ {jdemetra.VERSION}, run by pymetrix)",
                "backend_options": f"spec={self.spec}" + "".join(f", {k}={v!r}" for k, v in self.changes().items()),
                "jdemetra_libraries": str(jdemetra.jar_folder(self.libraries)),
                "tramo_model": "({} {} {})({} {} {})".format(*orders),
                "tramo_log": "logs" if int(get("log") or 0) else "levels",
                "tramo_outliers": ", ".join(found) or ("none" if spec.getTramo().getOutliers().isUsed() else "not looked for"),
                "tramo_calendar": f"trading-day regressors: {int(get('regression.ntd') or 0)}; Easter: {'yes' if int(get('regression.leaster') or 0) else 'no'}",
            })
            coefficients = list(get("regression.details.coefficients") or [])
            given = fixed | {r for r, _ in self.user_regressors()}
            effects = [f"{d} {float(c):.4g}" for d, c in zip(described, coefficients) if d in given]
            if effects:
                notes["tramo_regression"] = "; ".join(effects)
        return (y / sa, True) if multiplicative else (y - sa, False)

    def describe(self) -> str:
        changed = self.changes()
        return f"TRAMO-SEATS ({_TS_SPECS[self.spec]}" + "".join(f", {k}={v}" for k, v in changed.items()) + ")"
