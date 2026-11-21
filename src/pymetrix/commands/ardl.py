"""The ARDL dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Estimate/ARDL and bounds test...")
@dataclass(frozen=True)
class FitARDL(Command):
    """An autoregressive distributed lag model, with the Pesaran-Shin-Smith bounds test for a level relationship.

    Write the dependent variable on the regressors (lc ~ ly + lw). Give the lags of each, or let an information
    criterion choose them up to the maximums. The result shows the long-run multipliers, the bounds test and the
    error-correction form, whether the series are I(0) or I(1).

    The bounds test asks whether there is a level (long-run) relationship without first deciding whether each series
    is I(0) or I(1): a statistic above the upper bound rejects "no level relationship" whatever the mix, one below
    the lower bound does not reject, and one between them is inconclusive. It is not valid if any series is I(2).
    The long-run multipliers are the regressors' summed coefficients divided by one minus the sum of the dependent
    variable's lag coefficients; they are meaningful only if a level relationship exists. The error-correction
    coefficient should be negative and between -1 and 0: it is the share of a deviation from the long run corrected
    each period. Enough lags to leave white-noise residuals matter for the test's validity.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import ARDL, Dataset

        data = Dataset.example("sample_monthly").derive(lret="log(retail)", lip="log(ip)")
        ardl = ARDL("lret ~ lip", select="bic").fit(data)   # the lags of each chosen by BIC, up to 4
        print(ardl.summary())          # the chosen orders and the coefficients
        print(ardl.bounds_test())      # F against the I(0) and I(1) bounds: above I(1), a level relationship
        print(ardl.long_run())         # the long-run multiplier of lip, by the delta method
        ardl.ecm()                     # error-correction form: lret[-1]'s coefficient is the speed of adjustment
        ARDL("lret ~ lip", lags=2, order=2).fit(data)       # fixed lags: 2 of lret, 2 of lip
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("the coefficients at the chosen lags; the notes give the orders (and the criterion that chose "
                     "them), R-squared, AIC and BIC")),
        ("Bounds test", ("F against the I(0) and I(1) critical values at 90% to 99.9%: above I(1), a level "
                         "relationship; below I(0), none; 'at 5%' gives the verdict")),
        ("Long run", ("the long-run multipliers with delta-method standard errors; read them only if the bounds test "
                      "finds a level relationship")),
        ("Error correction", ("the error-correction form: the lagged level of the dependent variable's coefficient is "
                              "the speed of adjustment, between -1 and 0 if stable")),
        ("Graph", "the dependent variable and its fitted values"),
    )

    topics: ClassVar[tuple[str, ...]] = ("ardl", "cointegration")

    formula: Annotated[str, Widget("formula", help="dependent variable on the regressors, in levels (often logs), e.g. lc ~ ly + lw")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    select: Annotated[Literal["", "aic", "bic", "hqic"], Widget("choice", help="choose the lags by a criterion; empty = use the lags below")] = ""
    lags: Annotated[int, Widget("number", min=1, max=24, help="lags of the dependent variable; enough to leave serially uncorrelated residuals")] = 1
    order: Annotated[int, Widget("number", min=0, max=24, help="lags of each regressor; 0 = the current value only, which still allows a long-run effect")] = 1
    trend: Annotated[Literal["n", "c", "ct"], Widget("choice", help="deterministic terms: none, constant, or constant and trend; the bounds test's critical values depend on it")] = "c"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "ardl"
    max_lags: Annotated[int, Widget("number", min=1, max=24, advanced=True, help="longest lag of the dependent variable and of each regressor the criterion may choose")] = 4

    def _spec(self) -> Any:
        from ..ardl import ARDL

        return ARDL(self.formula, self.lags, self.order, self.trend, self.select or None, self.max_lags, self.max_lags)

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def validate(self, session: Session) -> list[str]:
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:ardl.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=self._span())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
