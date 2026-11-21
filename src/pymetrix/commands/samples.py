"""Dialogs for samples: name one (Data > Sample > New sample), and make it a dataset's active sample (Data > Sample > Set sample).

A named sample is an object in the workspace, made by one line of code (``precrisis = Sample(end="2007Q4")``);
setting it rebinds the dataset (``data = data.with_sample(precrisis)``), so scripts and projects replay it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS = ("samples",)


def _levels(text: str) -> dict[str, list[str]]:
    """``country: DEU, FRA; sector: C`` -> {"country": ["DEU", "FRA"], "sector": ["C"]}."""
    out: dict[str, list[str]] = {}
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        axis, sep, values = part.partition(":")
        if not sep or not axis.strip().isidentifier():
            raise SpecificationError(f"give entities as axis: values, e.g. country: DEU, FRA (not {part!r})")
        out[axis.strip()] = [v.strip() for v in values.split(",") if v.strip()]
    return out


def _windows(text: str) -> dict[str, tuple[str | None, str | None]]:
    """``DEU=1991Q1:; *=1980Q1:2019Q4`` -> {"DEU": ("1991Q1", None), "*": ("1980Q1", "2019Q4")}."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        entity, sep, span = part.partition("=")
        first, colon, last = span.partition(":")
        if not sep or not colon or not entity.strip():
            raise SpecificationError(f"give each entity's periods as entity=first:last, e.g. DEU=1991Q1: (not {part!r})")
        out[entity.strip()] = (first.strip() or None, last.strip() or None)
    return out


def _make(start: str, end: str, where: str, levels: str, windows: str = "") -> Any:
    from ..data import Sample

    spans = _windows(windows)
    return Sample(start=start or None, end=end or None, where=where.strip() or None, **({"windows": spans} if spans else {}), **_levels(levels))


@command(menu="Data/New sample...")
@dataclass(frozen=True)
class NewSample(Command):
    """Name a sample: a span of periods, entities of a panel, and a condition on the data, any of them. Use it
    as a dataset's active sample (Data > Sample > Set sample), so that estimation, tests, tables and graphs use it.

    Lags still use earlier periods, and generated series are computed over all the data: a sample only chooses
    the rows commands use. In code, samples combine with & (and), | (or) and ~ (not).
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    name: Annotated[str, Widget("text", help="name of the sample in the workspace, e.g. precrisis")] = "sample"
    data: Annotated[str, Widget("object", source="Dataset", help="the dataset whose periods to choose from")] = "data"
    start: Annotated[str, Widget("period", group="Periods")] = ""
    end: Annotated[str, Widget("period", group="Periods")] = ""
    entities: Annotated[str, Widget("text", help="for panels: axis: values; e.g. country: DEU, FRA")] = ""
    where: Annotated[str, Widget("text", help="a condition on the data, e.g. pop > 1e6 and year != 2020")] = ""
    windows: Annotated[str, Widget("text", advanced=True, group="Periods",
                                   help="panels: each entity's own periods, e.g. DEU=1991Q1:; FRA=:2015Q4; *=1980Q1: for the rest")] = ""

    def validate(self, session: Session) -> list[str]:
        if not self.name.isidentifier():
            return [f"{self.name!r} is not a valid name"]
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        try:
            sample = _make(self.start, self.end, self.where, self.entities, self.windows)
            if not sample.mask(session[self.data]).any():
                return [f"the sample selects no rows of {self.data}"]
        except Exception as exc:  # noqa: BLE001 - shown in the dialog
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import VALUE, make_call

        return make_call(VALUE, output=self.name, recorded_by=self.command_id, value=_make(self.start, self.end, self.where, self.entities, self.windows))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Check data...")
@dataclass(frozen=True)
class CheckData(Command):
    """Check a dataset before using it. Errors: rows that appear twice, periods missing from the time axis, infinite
    values, text among numbers. Warnings: missing values inside a series' span, series that never change, values far
    from the rest. Notes: series without units or a vintage (release date), and series that end before the others.

    Problems of this kind rarely stop an estimator but can quietly change its results: a duplicated row double-counts
    an observation, a missing period makes lags and differences span two periods, and a single mistyped value can
    dominate a least-squares fit. Outliers are values far from the median in units of the median absolute deviation
    (scaled to match the standard deviation of a normal), so a few extreme values do not hide one another; lower the
    threshold to see more, but check flagged values against the source rather than dropping them automatically. A
    ragged edge (series ending at different periods) is normal in nowcasting but shortens the sample of any equation
    that uses all of the series.
    """

    topics: ClassVar[tuple[str, ...]] = ("samples",)

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "checks"
    strict: Annotated[bool, Widget("checkbox", advanced=True, help="stop with an error if the data have errors (duplicates, gaps, text)")] = False
    outlier: Annotated[float, Widget("number", min=1, max=100, step=0.5, advanced=True,
                                     help="flag values this many robust standard deviations from the median")] = 6.0

    def validate(self, session: Session) -> list[str]:
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.validate", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         **{k: getattr(self, k) for k, d in (("strict", False), ("outlier", 6.0)) if getattr(self, k) != d})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Set sample...")
@dataclass(frozen=True)
class SetSample(Command):
    """Make a sample the dataset's active sample (EViews' smpl): estimation, tests, tables and graphs on the
    dataset use it unless they are given another. Choose a named sample (Data > Sample > New sample), or give periods
    here; leave both empty for all the data again.

    The sample decides which observations an estimate rests on, so it is part of the specification: report it, and
    check that results are not driven by a particular episode (a crisis, a pandemic year) by re-estimating on
    subsamples. The data outside the sample are kept: lags at the start of the sample still use earlier periods, so
    the first observations are not lost to lags as they would be if the data were cut. Setting a sample replaces the
    dataset in the workspace with one that carries it, and is recorded, so scripts and projects replay it.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    sample: Annotated[str, Widget("object", source="Sample", help="a named sample (periods, entities, a condition); empty to give periods below")] = ""
    start: Annotated[str, Widget("period", group="Periods")] = ""
    end: Annotated[str, Widget("period", group="Periods")] = ""

    def validate(self, session: Session) -> list[str]:
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        if self.sample and (self.start or self.end):
            return ["choose a named sample or give periods, not both"]
        if self.sample and self.sample not in session.objects:
            return [f"no sample named {self.sample!r}"]
        return []

    def _value(self, session: Session) -> Any:
        if self.sample:
            return session[self.sample]
        if self.start or self.end:
            return _make(self.start, self.end, "", "")
        return None

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        value = Ref(self.sample) if self.sample else (_make(self.start, self.end, "", "") if self.start or self.end else None)
        return make_call("pymetrix:dataset.with_sample", Ref(self.data), output=self.data, recorded_by=self.command_id, sample=value)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
