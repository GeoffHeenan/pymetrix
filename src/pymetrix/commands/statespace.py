"""Dialogs for state-space models: unobserved components, and any model written as equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command, migration

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _span(start: str, end: str) -> tuple[str | None, str | None] | None:
    return (start or None, end or None) if start or end else None


@command(menu="Estimate/Unobserved components...")
@dataclass(frozen=True)
class FitUnobservedComponents(Command):
    """Split a series into trend, cycle and irregular, estimated with the Kalman filter.

    Choose how the trend moves and the order of the autoregressive cycle. The smoothed trend and cycle use
    all the data (two-sided); the filtered ones use only the data up to each period (real-time), which is
    what a policymaker would have seen. Add either to the dataset with Data → Add results to dataset.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, UnobservedComponents

        data = Dataset.example("sample_macro").derive(ly="100*log(y)")
        uc = UnobservedComponents("ly", trend="local linear trend", cycle=2).fit(data)
        print(uc.summary())        # the shocks' standard deviations and the AR(2) cycle's coefficients
        uc.smoothed                # trend, slope and cycle from all the data (two-sided), with standard errors
        uc.filtered                # the same from the data up to each period (real time)
        UnobservedComponents("ly", trend="smooth trend", cycle=0).fit(data)   # an HP-like trend, no cycle
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("the shocks' standard deviations (sd_trend, sd_cycle...) and the cycle's AR coefficients phi; the "
                     "notes give the log likelihood, AIC and BIC")),
        ("Smoothed states, Filtered states", ("trend, slope and cycle with standard errors, two-sided and real-time; "
                                              "the difference is what later data revise")),
        ("Graph", "each smoothed state with its 95% band"),
        ("Innovations", "standardised one-step-ahead prediction errors, which should look like white noise"),
        ("Responses", "each state's and the series' responses to one-standard-deviation shocks"),
    )

    topics: ClassVar[tuple[str, ...]] = ("state-space", "trend-cycle-filters")

    series: Annotated[str, Widget("series", depends_on="data", help="the series to decompose, e.g. 100*log of real GDP")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    trend: Annotated[Literal["local level", "random walk with drift", "local linear trend", "smooth trend"],
                     Widget("choice", help="how the trend moves: smooth trend gives an HP-like trend")] = "local linear trend"
    cycle: Annotated[int, Widget("number", min=0, max=4, help="order of the autoregressive cycle; 0 = no cycle")] = 2
    irregular: Annotated[bool, Widget("checkbox", help="include a white-noise irregular component")] = True
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "uc"
    diffuse_terms: Annotated[Literal["include", "exclude"], Widget("choice", advanced=True,
                             help="the log likelihood's diffuse periods (the first ones, until the trend is pinned down): "
                                  "included (the exact diffuse likelihood, Durbin and Koopman) or left out (as statsmodels "
                                  "reports it); only the level of the log likelihood, AIC and BIC changes")] = "include"

    def _spec(self) -> Any:
        from ..statespace import UnobservedComponents

        return UnobservedComponents(self.series, self.trend, self.cycle, self.irregular)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:unobserved_components.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end),
                         **({"diffuse_terms": self.diffuse_terms} if self.diffuse_terms != "include" else {}))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/State space model...")
@dataclass(frozen=True)
class FitStateSpace(Command):
    """Estimate a state-space model written as equations, with the Kalman filter.

    One equation per line. Transition equations give each state from lagged states and shocks
    (`cycle = phi1*cycle[-1] + phi2*cycle[-2] + e_cycle`); measurement equations link observed series to the
    states (`y = trend + cycle + u_y`). Each shock has a standard deviation, a parameter or a number
    (`e_cycle = sd_cycle, u_y = 0.1`). Declare every parameter; states listed as stationary (cycles, gaps)
    start from their stationary distribution. Fixed values calibrate parameters instead of estimating them.
    """

    topics: ClassVar[tuple[str, ...]] = ("state-space",)

    states: Annotated[str, Widget("text", help="the unobserved states, e.g. trend, cycle")]
    transition: Annotated[str, Widget("multiline", help="one equation per state: state = ... lagged states and shocks")]
    measurement: Annotated[str, Widget("multiline", help="one per observed series: series = ... current or lagged states")]
    shocks: Annotated[str, Widget("text", help="each shock's standard deviation: e_trend = sd_trend, u_y = 0.1")]
    params: Annotated[str, Widget("text", help="parameters to declare, e.g. phi1, phi2, sd_trend")] = ""
    stationary: Annotated[str, Widget("text", help="states that are stationary (cycles, gaps)")] = ""
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    fixed: Annotated[str, Widget("text", advanced=True, help="calibrated values instead of estimates, e.g. sd_slope = 0")] = ""
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "ssm"
    cov: Annotated[Literal["classic", "none"], Widget("choice", advanced=True,
                                                      help="standard errors from the Hessian (classic), or none (faster)")] = "classic"
    max_iter: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, help="most iterations of the likelihood optimiser; raise it if the fit does not converge")] = 2000
    diffuse_terms: Annotated[Literal["include", "exclude"], Widget("choice", advanced=True,
                             help="the log likelihood's diffuse periods (the first ones, until the non-stationary states are pinned down): "
                                  "included (the exact diffuse likelihood, Durbin and Koopman) or left out (as statsmodels "
                                  "reports it); only the level of the log likelihood, AIC and BIC changes")] = "include"

    def _shocks(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for part in _names(self.shocks):
            shock, sep, sd = part.partition("=")
            if not sep:
                raise SpecificationError(f"give each shock's standard deviation, as {part.strip()} = sd_name or a number")
            sd = sd.strip()
            try:
                out[shock.strip()] = float(sd)
            except ValueError:
                out[shock.strip()] = sd
        return out

    def _fixed(self) -> dict[str, float]:
        out = {}
        for part in _names(self.fixed):
            name, _, value = part.partition("=")
            out[name.strip()] = float(value)
        return out

    def _spec(self) -> Any:
        from ..statespace import StateSpace

        return StateSpace(_names(self.states), _lines(self.transition), _lines(self.measurement), self._shocks(),
                          _names(self.params), stationary=_names(self.stationary))

    def validate(self, session: Session) -> list[str]:
        try:
            self._spec()
            self._fixed()
        except (ValueError, KeyError) as exc:
            return [str(exc.args[0]) if exc.args else str(exc)]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:state_space.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end), fixed=self._fixed() or None,
                         **{k: getattr(self, k) for k, d in (("cov", "classic"), ("max_iter", 2000), ("diffuse_terms", "include"))
                            if getattr(self, k) != d})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Exponential smoothing...")
@dataclass(frozen=True)
class FitExponentialSmoothing(Command):
    """Exponential smoothing (ETS): forecast a series from a level, an optional trend (damped or not) and optional
    seasonal factors, each updated by a share of the latest forecast error.

    No trend and no season is simple exponential smoothing; an additive trend is Holt's method; adding seasons gives
    Holt-Winters. Parameters and initial states are estimated by maximum likelihood (statsmodels). Forecast from the
    result with Forecast, or evaluate it with Forecast → Backtest.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, ExponentialSmoothing

        data = Dataset.example("sample_monthly")
        ets = ExponentialSmoothing("ip", trend="add", damped=True).fit(data)   # Holt's method, damped trend
        print(ets.summary())           # smoothing parameters, damping and initial states (maximum likelihood)
        ets.forecast(data, ("2025-04", "2026-03"))          # forecast, se, lower and upper for each month
        ExponentialSmoothing("ip").fit(data)                # simple exponential smoothing: a level, no trend
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("the model, e.g. ETS(A,Ad,N); alpha, beta, gamma: the share of each error passed to level, trend "
                     "and season; phi, the damping; initial states")),
        ("States", "the level, trend and seasonal states at the end of each period"),
        ("Residuals, Graph", "the series, its one-step-ahead fit and the residual; the graph adds the level"),
        ("ets.forecast(data, span)", "forecast, se, lower and upper for each period (as the Forecast dialog gives)"),
    )

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    series: Annotated[str, Widget("series", depends_on="data", help="the series to forecast, in levels; multiplicative options need it positive")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    trend: Annotated[Literal["none", "add", "mul"], Widget("choice", help="none (simple smoothing), additive (Holt) or multiplicative")] = "none"
    damped: Annotated[bool, Widget("checkbox", help="damp the trend towards flat (needs a trend)")] = False
    seasonal: Annotated[Literal["none", "add", "mul"], Widget("choice", help="seasonal factors: none, additive or multiplicative (Holt-Winters)")] = "none"
    error: Annotated[Literal["add", "mul"], Widget("choice", advanced=True, help="additive or multiplicative errors (positive series)")] = "add"
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "ets"

    def _spec(self) -> Any:
        from ..ets import ExponentialSmoothing

        return ExponentialSmoothing(self.series, self.trend, self.damped, self.seasonal, self.error)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:exponential_smoothing.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Bayesian model averaging...")
@dataclass(frozen=True)
class FitBMA(Command):
    """Bayesian model averaging: weigh every regression on a subset of the candidate regressors by its posterior
    probability (Zellner's g-prior, equal prior probabilities).

    Reports each candidate's posterior inclusion probability, the model-averaged coefficients and the most probable
    models; forecasts average the models' forecasts. Up to 15 candidates (32,768 models).
    """

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    schema_version: ClassVar[int] = 2  # 2: target became outcome

    outcome: Annotated[str, Widget("text", help="the series to explain or forecast, e.g. dlog(gdp)")]
    candidates: Annotated[str, Widget("text", help="candidate regressors, separated by commas, e.g. spread[-1], pmi[-1]")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    always: Annotated[str, Widget("text", advanced=True, help="regressors in every model besides the intercept, separated by commas")] = ""
    g: Annotated[float, Widget("number", min=0, max=1e9, advanced=True, help="prior scale: larger g shrinks coefficients less but penalises larger models more; 0 = max(n, K^2)")] = 0.0
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "bma"

    def _spec(self) -> Any:
        from ..bma import BMA

        return BMA(self.outcome, tuple(_names(self.candidates)), tuple(_names(self.always)), self.g)

    def validate(self, session: Session) -> list[str]:
        try:
            self._spec()
        except (ValueError, SpecificationError) as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:bma.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@migration("pymetrix:fit_bma", 1)
def _fit_bma_outcome(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became outcome
    if "target" in fields:
        fields["outcome"] = fields.pop("target")
    return fields


@command(menu="Estimate/Threshold autoregression...")
@dataclass(frozen=True)
class FitThresholdAR(Command):
    """A two-regime threshold autoregression (SETAR): the series follows one AR when its own lagged value is at or below
    a threshold, another above it.

    The threshold and the delay (which lag switches regimes) are chosen by least squares. The result's Tests tab has
    Hansen's threshold test and Teräsvirta's test against smooth transition; forecasts are simulated, so multi-step
    forecasts reflect the regimes.
    """

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    series: Annotated[str, Widget("series", depends_on="data", help="the series, e.g. GDP growth")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="the autoregressive order in each regime")] = 1
    delay: Annotated[int, Widget("number", min=0, max=12, help="the lag that switches regimes; 0 = choose")] = 0
    trim: Annotated[float, Widget("number", min=0.05, max=0.4, step=0.05, advanced=True, help="least share of observations in each regime; keeps thresholds that leave too little data from being chosen")] = 0.15
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "tar"

    def _spec(self) -> Any:
        from ..nonlinear import ThresholdAR

        return ThresholdAR(self.series, self.lags, self.delay, self.trim)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:threshold_ar.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Smooth-transition autoregression...")
@dataclass(frozen=True)
class FitSmoothTransitionAR(Command):
    """A smooth-transition autoregression (LSTAR or ESTAR): two AR regimes blended by a smooth function of a lag.

    Logistic moves between a low and a high regime as the lagged value rises; exponential separates values near a
    centre from those far from it. The speed and centre of the transition are found by least squares; forecasts are
    simulated. Teräsvirta's test (in the result's Tests tab) asks whether the transition is needed at all.
    """

    topics: ClassVar[tuple[str, ...]] = ("smooth-transition",)

    series: Annotated[str, Widget("series", depends_on="data", help="the series, e.g. GDP growth")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="the autoregressive order in each regime")] = 1
    delay: Annotated[int, Widget("number", min=0, max=12, help="the lag that drives the transition; 0 = choose")] = 1
    transition: Annotated[Literal["logistic", "exponential"], Widget("choice", help="logistic (LSTAR): low versus high values; exponential (ESTAR): near the centre versus far from it")] = "logistic"
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "star"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..smooth_transition import SmoothTransitionAR

        spec = SmoothTransitionAR(self.series, self.lags, self.delay, self.transition)
        return make_call("pymetrix:smooth_transition_ar.fit", spec, output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
