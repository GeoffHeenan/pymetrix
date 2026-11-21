"""Changes made in the spreadsheet editor, as steps: recorded in the history and script,
saved in projects and replayed. Each replaces the dataset with an edited copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from ..errors import SpecificationError
from .registry import Command, Widget, command, existing_series

if TYPE_CHECKING:
    from .session import Session

_DATA = Widget("object", source="Dataset", help="the dataset to change (it is replaced by the changed copy)")


def _key(parts: list[str]) -> Any:
    return parts[0] if len(parts) == 1 else tuple(parts)


@command(menu=None)
@dataclass(frozen=True)
class EditData(Command):
    """Values typed or pasted into the spreadsheet editor."""

    label: ClassVar[str] = "Spreadsheet"  # where the step came from, in the History panel

    data: Annotated[str, _DATA]
    edits: Annotated[str, Widget("text", help='JSON list of [series, row key parts, value or null], e.g. [["c", ["2020Q1"], 1.2]]')]

    def cells(self) -> dict[str, dict[Any, Any]]:
        out: dict[str, dict[Any, Any]] = {}
        for name, parts, value in json.loads(self.edits):
            out.setdefault(name, {})[_key(parts)] = value
        return out

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.edit", Ref(self.data), output=self.data, recorded_by=self.command_id, values=self.cells())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)
@dataclass(frozen=True)
class RenameSeries(Command):
    """Rename a series; recipes of derived series that use it follow."""

    label: ClassVar[str] = "Spreadsheet"  # where the step came from, in the History panel

    data: Annotated[str, _DATA]
    old: Annotated[str, Widget("text", help="the series to rename")]
    new: Annotated[str, Widget("text", help="its new name")]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.rename", Ref(self.data), output=self.data, recorded_by=self.command_id, **{self.old: self.new})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)
@dataclass(frozen=True)
class DropSeries(Command):
    """Remove series from a dataset."""

    label: ClassVar[str] = "Spreadsheet"  # where the step came from, in the History panel

    data: Annotated[str, _DATA]
    series: Annotated[str, Widget("text", help="comma-separated series to remove")]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.drop", Ref(self.data), output=self.data, recorded_by=self.command_id,
                         names=tuple(s.strip() for s in self.series.split(",") if s.strip()))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)
@dataclass(frozen=True)
class FreezeSeries(Command):
    """Keep a derived series' values but drop its recipe, so it can be edited."""

    label: ClassVar[str] = "Spreadsheet"  # where the step came from, in the History panel

    data: Annotated[str, _DATA]
    series: Annotated[str, Widget("text", help="comma-separated derived series to freeze")]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.freeze", Ref(self.data), output=self.data, recorded_by=self.command_id,
                         names=tuple(s.strip() for s in self.series.split(",") if s.strip()))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)
@dataclass(frozen=True)
class SeriesInfo(Command):
    """A series' description: label, units, description, source and how it is measured."""

    label: ClassVar[str] = "Spreadsheet"  # where the step came from, in the History panel

    data: Annotated[str, _DATA]
    series: Annotated[str, Widget("text", help="the series to describe")]
    info: Annotated[str, Widget("text", help='JSON object of the fields to set, e.g. {"units": "USD bn"}')]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.with_info", Ref(self.data), output=self.data, recorded_by=self.command_id,
                         name=self.series, **json.loads(self.info))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Generate series...")
@dataclass(frozen=True)
class GenerateSeries(Command):
    """Add a series computed once from a formula; its values stay as computed (like EViews' genr).

    For a series that keeps following its inputs, tick Keep updated (a derived series). To make one for each of
    several series, fill in *For each*: a list (`gdp cons inv`), wildcards (`*_sa`, `ip_*`) or the series picked in
    the dataset's series browser (Generate for each…). Series and Formula are then templates: `{x}` is each series,
    `{1}`, `{2}`... the parts a wildcard matched (`ip_*` with `{1}_growth`). The preview lists every new series
    before it is made, and the step records them all, written out, in one call.
    """

    topics: ClassVar[tuple[str, ...]] = ("derived-series", "generate-for-each")

    series: Annotated[str, Widget("text", help="name of the new series: a plain name, e.g. ly or growth; "
                                               "with For each, a template such as {x}_g or dl_{x}")]
    formula: Annotated[str, Widget("formula", help="expression of existing series, e.g. log(y), c / y * 100, y[-1]; computed over "
                                                   "all periods; with For each, a template such as 100*d(log({x}))")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    keep_updated: Annotated[bool, Widget("checkbox", help="keep the formula, so the series follows its inputs (a derived series)")] = False
    replace: Annotated[bool, Widget("checkbox", help="replace a series of the same name, if there is one")] = False
    for_each: Annotated[str, Widget("text", group="For each", help="make one series for each of these: names (gdp cons inv), "
                                                                   "wildcards (*_sa, ip_*) or both; empty: just one series")] = ""

    schema_version: ClassVar[int] = 2  # 2: replace (schema-1 steps replaced silently); for_each added with a default

    def preview(self, session: Session) -> Any:
        """What *For each* would make (``pymetrix.data.each.preview_each``): a row per new series, checked."""
        from ..data.each import preview_each

        data = session.objects.get(self.data)
        return preview_each(data, self.for_each, self.series, self.formula, replace=self.replace, derived=self.keep_updated,
                            data_name=self.data)

    def preview_rows(self, session: Session) -> dict[str, Any] | None:
        """The dialog's preview table (None: nothing to preview, a single series): ``headings``, ``rows`` (new series,
        formula, note), ``marks`` per row (``"error"``, ``"warning"`` or ``""``), a ``summary`` and whether it can run
        (``ok``)."""
        if not self.for_each.strip():
            return None
        headings = ["New series", "Formula", "Note"]
        if self.data not in session.objects:
            return {"headings": headings, "rows": [], "marks": [], "summary": f"no dataset named {self.data!r}", "ok": False}
        found = self.preview(session)
        marks = {"error": "error", "exists": "error", "replaces": "warning"}
        errors = found.errors()
        replacing = sum(r.status == "replaces" for r in found.rows)
        summary = "; ".join(errors) if errors else (f"{len(found.rows)} new series" + (f", {replacing} replacing existing ones"
                                                                                         if replacing else ""))
        return {"headings": headings, "rows": [(r.name, r.formula, r.note) for r in found.rows],
                "marks": [marks.get(r.status, "") for r in found.rows], "summary": summary, "ok": not errors}

    def validate(self, session: Session) -> list[str]:
        if self.for_each.strip():
            return self.preview(session).errors() if self.data in session.objects else [f"no dataset named {self.data!r}"]
        if not self.series.isidentifier():
            return [f"{self.series!r} is not a valid series name"]
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        return existing_series(session, self.data, [self.series], self.replace)

    def formulas(self, session: Session | None = None) -> dict[str, str]:
        """New name -> formula: the one series, or with *For each* each series it stands for in the dataset."""
        if not self.for_each.strip():
            return {self.series: self.formula}
        if session is None or self.data not in session.objects:
            raise SpecificationError("For each needs the dataset, to find the series it names")
        found = self.preview(session)
        if not found.ok:
            raise SpecificationError("; ".join(found.errors()))
        return found.formulas()

    def to_call(self, session: Session | None = None) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        verb = "derive" if self.keep_updated else "with_series"
        return make_call(f"pymetrix:dataset.{verb}", Ref(self.data), output=self.data, recorded_by=self.command_id,
                         **self.formulas(session))

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        if self.for_each.strip():  # without the dataset to expand it in: the helper that does (the step records the expansion)
            options = (", derived=True" if self.keep_updated else "") + (", replace=True" if self.replace else "")
            return f"{self.data} = {self.data}.with_each({self.for_each!r}, {self.series!r}, {self.formula!r}{options})"
        return self.to_call().to_code()
