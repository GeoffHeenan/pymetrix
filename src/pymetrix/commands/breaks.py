"""The dialog for Bai-Perron structural breaks (the Quandt-Andrews test is in Run test, as quandt_andrews)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Test/Structural breaks (Bai-Perron)...")
@dataclass(frozen=True)
class FindBreaks(Command):
    """Bai-Perron: the dates of structural breaks in a linear equation, at unknown dates, and how many.

    For each number of breaks up to the maximum, the dates that minimise the sum of squared residuals (every
    regime at least the trimmed share of the sample, all coefficients changing); the number of breaks is chosen by
    the Bayesian information criterion. The window shows the dates for each number, each regime's coefficients and
    the fit by regime. For a test of a single break at an unknown date, Test → Run test…
    with quandt_andrews.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, bai_perron

        data = Dataset.example("nile")             # the Nile's annual flow at Aswan, 1871-1970
        breaks = bai_perron(Equation("volume ~ 1").fit(data))   # breaks in the mean, up to 5
        print(breaks.summary())    # SSR, BIC and dates for each number of breaks, and the number BIC chooses
        # an AR(1), up to two breaks, each regime at least 20% of the sample
        bai_perron(Equation("volume ~ volume[-1]").fit(data), max_breaks=2, trim=0.2)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("for each number of breaks, the SSR, BIC and break dates; the notes give the number BIC chooses "
                     "(the lowest BIC) and the trimming")),
        ("Regimes", ("the coefficients and standard errors in each regime of the chosen breaks: compare them to see "
                     "what changed")),
        ("Graph", "the series, the fit by regime and a dashed line at each chosen break date"),
        ("breaks.dates, breaks.chosen", "the chosen break dates and their number, for use in code"),
    )

    topics: ClassVar[tuple[str, ...]] = ("diagnostic-tests",)

    fit: Annotated[str, Widget("object", source="FittedEquation", help="a linear equation fitted to a time series")]
    max_breaks: Annotated[int, Widget("number", min=1, max=10, help="the most breaks to consider")] = 5
    trim: Annotated[float, Widget("number", advanced=True, min=0.05, max=0.45, step=0.05, help="the smallest regime, as a share of the sample")] = 0.15
    name: str = "breaks"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:bai_perron", output=self.name, recorded_by=self.command_id, fit=Ref(self.fit),
                         max_breaks=self.max_breaks, trim=self.trim)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
