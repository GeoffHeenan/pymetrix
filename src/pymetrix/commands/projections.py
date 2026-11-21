"""The local projections dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@command(menu="Estimate/Local projections...")
@dataclass(frozen=True)
class FitLocalProjection(Command):
    """Impulse responses by local projection: one regression of the response at each horizon on the shock.

    The response and shock can be series or expressions (`100*log(gdp)`). Lags of the response, the shock and
    the controls are included; controls dated at the time of the shock put them before it in a recursive
    ordering. Tick Long difference for variables in log levels. Standard errors allow for the overlap between
    horizons (Newey-West with h+1 lags), or use the lag-augmented robust form. A state indicator (0 to 1)
    gives separate responses in each state, with a test of their difference. External instruments for the shock make each
    horizon's regression two-stage least squares (LP-IV), with each horizon's first-stage F.

    The responses are causal only if the shock is: an identified shock series, or a variable made exogenous by the
    controls and the recursive ordering. Local projections are robust to misspecified dynamics, as each horizon has
    its own regression, but they are less efficient than a VAR, so the responses are noisier and become erratic at
    long horizons where the sample per regression also shrinks. Smoothing trades a little bias for much less noise.
    Read the bands (68% and the wider level) horizon by horizon; they are not joint bands for the whole path. With
    LP-IV, a first-stage F well above 10 at each horizon guards against weak instruments.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, LocalProjection

        data = Dataset.example("sample_macro")
        lp = LocalProjection("100*log(y)", "100*dlog(g)", horizons=8, lags=4).fit(data)
        lp.irf          # by horizon: observations, the response, its se, 68% bands and 90% bands (lower, upper)
        LocalProjection("100*log(y)", "100*dlog(g)", horizons=8, lags=4, long_difference=True, cov="white").fit(data)
        # the response as y[t+h] - y[t-1], with lag-augmented robust standard errors
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Responses", ("by horizon: the response, se, 68% bands (lower68, upper68), the wider bands (lower, upper) and "
                       "nobs, which falls as h grows")),
        ("Graph", "the response with its 68% and wider bands; they are pointwise, not joint bands for the whole path"),
        ("Regressions, Statistics", ("each horizon's regression (h=0, h=1, ...): coefficients with standard errors in "
                                     "brackets; nobs, R-squared and criteria")),
        ("first-stage F", ("with instruments (LP-IV): their strength at each horizon; well above 10 guards against weak "
                           "instruments")),
        ("state response, other response, difference p-value", ("with a state: the response in each state, and a test "
                                                                "that the two are equal")),
    )

    topics: ClassVar[tuple[str, ...]] = ("local-projections", "var")

    response: Annotated[str, Widget("text", help="the response, e.g. 100*log(gdp) or infl")]
    shock: Annotated[str, Widget("text", help="the shock, e.g. an identified monetary surprise or the policy rate")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    horizons: Annotated[int, Widget("number", min=0, max=60, help="longest horizon traced; a few years of the data's frequency, as long horizons lose observations and precision")] = 12
    lags: Annotated[int, Widget("number", min=0, max=24, help="lags of the response, the shock and the controls")] = 4
    controls: Annotated[str, Widget("series", depends_on="data", multiple=True, help="variables whose lags are controlled for, to absorb predictable movements in the shock and response")] = ""
    contemporaneous: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                           help="controls dated at the time of the shock (ordered before it)")] = ""
    long_difference: Annotated[bool, Widget("checkbox", help="response y[t+h] - y[t-1], for variables in log levels")] = False
    cov: Annotated[Literal["hac", "white"], Widget("choice", show_if=("smooth", (False,)),
                                                   help="hac: Newey-West with h+1 lags; white: lag-augmented robust")] = "hac"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "lp"
    state: Annotated[str, Widget("text", advanced=True, help="state indicator series (0 to 1), dated t-1; empty = none")] = ""
    scale: Annotated[float, Widget("number", min=1e-6, max=1e6, advanced=True, help="size of the shock the responses are scaled to")] = 1.0
    instruments: Annotated[str, Widget("series", depends_on="data", multiple=True, advanced=True,
                                       help="external instruments for the shock (LP-IV); empty = none")] = ""
    smooth: Annotated[bool, Widget("checkbox", advanced=True,
                                   help="a smooth response: a penalised spline in the horizon, the penalty by cross-validation")] = False
    alpha: Annotated[float, Widget("number", min=0.01, max=0.5, step=0.01, advanced=True,
                                   help="1 - the wide bands' coverage, e.g. 0.1 for 90% (68% bands are drawn too)")] = 0.1
    penalty: Annotated[float | None, Widget("number", min=0.0, max=1e8, step=1, advanced=True, show_if=("smooth", (True,)),
                                            help="the smoothing penalty; empty = by cross-validation")] = None
    order: Annotated[int, Widget("number", min=1, max=4, advanced=True, show_if=("smooth", (True,)),
                                 help="the penalty's order: 2 shrinks towards a straight line, 1 towards a constant")] = 2
    folds: Annotated[int, Widget("number", min=2, max=20, advanced=True, show_if=("smooth", (True,)), help="blocks of consecutive periods for the cross-validation that sets the smoothing penalty")] = 5

    def _spec(self) -> Any:
        from ..projections import LocalProjection

        return LocalProjection(self.response, self.shock, self.horizons, self.lags, _names(self.controls), _names(self.contemporaneous),
                               self.long_difference, self.cov, self.state or None, self.scale, _names(self.instruments))

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def validate(self, session: Session) -> list[str]:
        try:
            self._spec()
        except ValueError as exc:
            return [str(exc)]
        if self.smooth and (self.state or self.instruments.strip()):
            return ["a smooth response here is for time series, without a state or instruments"]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        call = "pymetrix:local_projection.fit_smooth" if self.smooth else "pymetrix:local_projection.fit"
        changed = (("alpha", 0.1), ("penalty", None), ("order", 2), ("folds", 5)) if self.smooth else (("alpha", 0.1),)
        extra = {k: getattr(self, k) for k, d in changed if getattr(self, k) != d}
        return make_call(call, self._spec(), output=self.name, recorded_by=self.command_id, data=Ref(self.data), span=self._span(), **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Growth at risk...")
@dataclass(frozen=True)
class FitGrowthAtRisk(Command):
    """Growth at risk: quantile projections of future growth on today's conditions, and the skewed t they imply.

    Each horizon's growth (averaged over the next h periods) is regressed on the predictors at its quantiles; each
    period's predicted 5%, 25%, 75% and 95% quantiles are matched by a skewed t, whose 5% quantile is growth at risk
    and whose mean below it is expected shortfall (Adrian, Boyarchenko and Giannone 2019).

    The point is the asymmetry: in the original study, tight financial conditions move the lower quantiles of future
    growth much more than the median or the upper ones, so downside risk varies more than the central forecast. Read
    the quantile regression coefficients across quantiles: a predictor whose coefficients differ across them changes
    the shape of the distribution, not just its location. With few recessions in a sample the extreme quantiles rest
    on few observations, so treat the 5% quantile and expected shortfall as imprecise, and use predictors known at t
    so the projections are genuinely ahead.
    """

    topics: ClassVar[tuple[str, ...]] = ("growth-at-risk",)

    growth: Annotated[str, Widget("text", help="the growth rate, e.g. dlog(gdp)*400")]
    predictors: Annotated[str, Widget("series", depends_on="data", multiple=True, help="conditions at t: financial conditions, growth")]
    data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset holding the growth series and predictors")] = "data"
    horizons: Annotated[str, Widget("text", help="horizons in periods, e.g. 1, 4: growth is averaged over the next h periods for each")] = "1, 4"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "gar"
    quantiles: Annotated[str, Widget("text", advanced=True, help="the quantiles estimated, e.g. 0.05, 0.25, 0.5, 0.75, 0.95")] = "0.05, 0.25, 0.5, 0.75, 0.95"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..errors import SpecificationError

        try:
            horizons = [int(h) for h in self.horizons.replace(";", ",").split(",") if h.strip()]
        except ValueError:
            raise SpecificationError(f"the horizons are whole numbers separated by commas, not {self.horizons!r}") from None
        try:
            quantiles = [float(q) for q in self.quantiles.replace(";", ",").split(",") if q.strip()]
        except ValueError:
            raise SpecificationError(f"the quantiles are numbers between 0 and 1 separated by commas, not {self.quantiles!r}") from None
        span = (self.start or None, self.end or None) if self.start or self.end else None
        extra = {"quantiles": quantiles} if quantiles != [0.05, 0.25, 0.5, 0.75, 0.95] else {}
        return make_call("pymetrix:growth_at_risk", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         growth=self.growth, predictors=_names(self.predictors), horizons=horizons, span=span, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
