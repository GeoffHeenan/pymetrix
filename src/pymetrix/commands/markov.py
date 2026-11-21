"""The Markov-switching dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Estimate/Markov switching...")
@dataclass(frozen=True)
class FitMarkovSwitching(Command):
    """A regression whose intercept and coefficients switch between unobserved regimes (Hamilton 1989).

    Write the dependent variable on the regressors (growth ~ 1 for a switching mean). The regimes follow a
    Markov chain; the result shows each regime's coefficients, the transition probabilities, the expected
    duration of each regime and the probability of each regime in each period.

    The regimes are unlabelled: read them from their estimates (a low-mean regime may be recessions). The smoothed
    probabilities use the whole sample and are the ones to date regimes with; a regime that is rarely visited, or a
    transition probability near 1 that makes a regime last for the whole sample, suggests too many regimes. The
    likelihood often has several local maxima, so the fit tries random starting values; if results change with the
    seed, raise the number tried. Standard tests of the number of regimes are not valid (a parameter is unidentified
    under the null), so do not compare a two-regime model with a linear one by a likelihood-ratio test.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, MarkovSwitching

        data = Dataset.example("sample_monthly").derive(gip="400*dlog(ip)")   # annualised monthly growth
        ms = MarkovSwitching("gip ~ 1", regimes=2, switching_variance=True).fit(data)
        print(ms.summary())        # each regime's mean const[k] and variance sigma2[k], transitions, expected durations
        ms.smoothed                # the probability of each regime in each month, from the whole sample
        MarkovSwitching("gip ~ 1", regimes=2).fit(data)     # only the mean switches; one variance
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("transition probabilities p[i->j], each regime's coefficients (const[k]) and, if switching, "
                     "variances sigma2[k]; the notes give the expected durations")),
        ("Transition probabilities", "from row to column; diagonal entries near 1 mean persistent regimes"),
        ("Expected durations", "1 / (1 - p[k->k]) periods: how long a spell in each regime lasts on average"),
        ("Regime probabilities, Graph", ("the smoothed probability of each regime in each period (ms.smoothed), drawn "
                                         "under the series: use them to date the regimes")),
    )

    topics: ClassVar[tuple[str, ...]] = ("markov-switching",)

    formula: Annotated[str, Widget("formula", help="dependent variable on the regressors whose coefficients switch; growth ~ 1 switches only the mean")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    regimes: Annotated[int, Widget("number", min=2, max=5, help="number of regimes; two (e.g. expansion and recession) is usual, more need a long sample")] = 2
    ar: Annotated[int, Widget("number", min=0, max=12, help="autoregressive lags whose coefficients also switch between regimes; 0 = none")] = 0
    switching_variance: Annotated[bool, Widget("checkbox", help="let the error variance differ between regimes")] = False
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "ms"
    search: Annotated[int, Widget("number", min=0, max=500, advanced=True, help="random starting values tried besides the default; more guards against a local maximum")] = 20
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True)] = 0

    def _spec(self) -> Any:
        from ..markov import MarkovSwitching

        return MarkovSwitching(self.formula, self.regimes, self.ar, self.switching_variance, search=self.search, seed=self.seed)

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def validate(self, session: Session) -> list[str]:
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:markov_switching.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=self._span())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
