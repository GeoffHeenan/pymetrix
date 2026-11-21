"""Principal components and factor analysis dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS: tuple[str, ...] = ("factor-analysis",)


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _span(start: str, end: str) -> tuple[str | None, str | None] | None:
    return (start or None, end or None) if start or end else None


def _checks(session: Session, data: str, variables: str, at_least: int) -> list[str]:
    from ..data import Dataset

    obj = session.objects.get(data)
    if not isinstance(obj, Dataset):
        return [f"no dataset named {data!r}"]
    names = _names(variables)
    errors = [f"no series {n!r} in {data}" for n in names if n not in obj.names]
    if len(names) < at_least:
        errors.append(f"choose at least {at_least} series")
    return errors


@command(menu="Estimate/Principal components...")
@dataclass(frozen=True)
class FitPCA(Command):
    """Principal components: the weighted averages of a set of series that explain most of their variance.

    The first component is the combination that varies most; each later one varies most among those uncorrelated
    with the ones before. Standardized (the default), each series counts equally whatever its units (the
    correlation matrix); otherwise series with larger variances dominate (the covariance matrix). The window shows
    the loadings, the variance each component explains, Bai and Ng's criteria for the number of factors, and the
    components period by period; Next ▸ adds them to a dataset, for a regression or a bridge equation on factors.
    Only periods where every series is observed are used: for missing values and mixed frequencies, use the
    dynamic factor model (Estimate → Mixed frequency).
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series to summarise (two or more)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    n: Annotated[int, Widget("number", min=0, max=100, label="Components kept",
                             help="how many components to keep; 0 = all (the window's criteria help choose)")] = 0
    standardize: Annotated[bool, Widget("checkbox", help="scale each series to unit variance first (the correlation matrix): "
                                                         "do so unless the series share units")] = True
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "pca"

    def _spec(self) -> Any:
        from ..factors import PCA

        return PCA(tuple(_names(self.variables)), None if not self.n else self.n, self.standardize)

    def validate(self, session: Session) -> list[str]:
        errors = _checks(session, self.data, self.variables, 2)
        if self.n > len(_names(self.variables)):
            errors.append(f"at most {len(_names(self.variables))} components: one per series")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:pca.fit", self._spec(), output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Factor analysis...")
@dataclass(frozen=True)
class FitFactorModel(Command):
    """Maximum-likelihood factor analysis: each series a combination of a few common factors plus a part of its own.

    Unlike principal components, the model separates what the series share (their communalities) from what is
    particular to each (its uniqueness), and tests whether the number of factors is enough: a small p-value says
    more are needed. Rotation makes the loadings easier to read without changing the fit: varimax keeps the
    factors uncorrelated, promax lets them correlate (their correlations are shown). The window shows the
    loadings, uniquenesses, the test and regression scores; Next ▸ adds regression or Bartlett scores to a
    dataset. A uniqueness at its lower bound (0.005) is a Heywood case: a series the factors explain entirely,
    often a sign of too many factors. The results match R's factanal.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series to analyse (three or more)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    factors: Annotated[int, Widget("number", min=1, max=50, help="number of common factors; the test in the window says whether it is enough")] = 1
    rotation: Annotated[Literal["none", "varimax", "promax"],
                        Widget("choice", help="varimax: uncorrelated factors, loadings near 0 or large; promax: correlated factors (oblique)")] = "none"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "fa"

    def _spec(self) -> Any:
        from ..factors import FactorModel

        return FactorModel(tuple(_names(self.variables)), self.factors, rotation=self.rotation)

    def validate(self, session: Session) -> list[str]:
        errors = _checks(session, self.data, self.variables, 3)
        p, q = len(_names(self.variables)), self.factors
        if p >= 3 and (p - q) ** 2 < p + q:
            errors.append(f"{q} factors are too many for {p} series: a model is identified only when (series - factors)² ≥ "
                          "series + factors")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:factor_model.fit", self._spec(), output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
