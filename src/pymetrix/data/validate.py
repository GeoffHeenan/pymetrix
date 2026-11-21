"""Checking a dataset before using it: ``data.validate()``.

    report = data.validate()
    report.ok                  # no errors (warnings and notes may remain)
    report.summary()           # one row per finding: level, series, check, what was found
    data.validate(strict=True) # raises DataError if there are errors

Errors are what makes results wrong or impossible: rows that appear twice, periods missing from the time axis
(a gap in a quarterly series is not a quarter), infinite values, series that are not numbers. Warnings are what
deserves a look: missing values inside a series' span, series that never change, values far from the rest
(more than 6 robust standard deviations from the median). Notes are about metadata: series without units, or
without a vintage (the release date of the data, so results can say which release they used), and series that end
earlier than the others (a ragged edge, normal for nowcasting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataError
from ..output import Table

__all__ = ["Validation", "validate"]

_LEVELS = ("error", "warning", "note")


@dataclass(frozen=True)
class Finding:
    level: str  # error, warning or note
    series: str  # "" for the dataset as a whole
    check: str
    message: str


@dataclass(frozen=True, eq=False)
class Validation:
    """The findings of ``data.validate()``: see the module docstring."""

    findings: tuple[Finding, ...]
    rows: int
    series: int

    @property
    def ok(self) -> bool:
        """No errors."""
        return not self.errors

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    def summary(self) -> Table:
        order = sorted(self.findings, key=lambda f: (_LEVELS.index(f.level), f.series, f.check))
        counts = {lv: sum(f.level == lv for f in self.findings) for lv in _LEVELS}
        verdict = "No problems found" if not self.findings else ", ".join(f"{n} {lv}{'s' if n != 1 else ''}" for lv, n in counts.items() if n)
        return Table(["level", "series", "check", "found"], [[f.level, f.series or "(dataset)", f.check, f.message] for f in order],
                     title="Data checks", notes=[f"{self.rows} rows, {self.series} series: {verdict}"])

    def __repr__(self) -> str:
        return f"<Validation: {len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.findings) - len(self.errors) - len(self.warnings)} notes>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _periods_missing(periods: pd.PeriodIndex) -> list[str]:
    if len(periods) < 2:
        return []
    full = pd.period_range(periods.min(), periods.max(), freq=periods.freq)
    return [str(p) for p in full.difference(periods)]


def _list(items: list[str], most: int = 5) -> str:
    return ", ".join(items[:most]) + (f" and {len(items) - most} more" if len(items) > most else "")


def validate(data: Any, strict: bool = False, outlier: float = 6.0) -> Validation:
    """Check ``data`` (a Dataset): see the module docstring. ``strict`` raises ``DataError`` if there are errors."""
    frame = data.frame
    found: list[Finding] = []

    def add(level: str, series: str, check: str, message: str) -> None:
        found.append(Finding(level, series, check, message))

    # the axes: duplicated rows and missing periods
    dupes = frame.index[frame.index.duplicated()]
    if len(dupes):
        add("error", "", "duplicate rows", f"{len(dupes)} row(s) appear more than once: {_list([str(d) for d in dupes.unique()])}")
    t = data.time_axis
    if t is not None and not len(dupes):
        periods = frame.index.get_level_values(t.name)
        if data.entity_axes:
            entity = data.entity_axes[0].name
            ids = frame.index.get_level_values(entity)
            for e in pd.unique(ids):
                missing = _periods_missing(pd.PeriodIndex(periods[ids == e]).sort_values())
                if missing:
                    add("error", "", "missing periods", f"{e}: no rows for {_list(missing)}")
        else:
            missing = _periods_missing(pd.PeriodIndex(periods).sort_values())
            if missing:
                add("error", "", "missing periods", f"no rows for {_list(missing)} (gaps in the time axis)")

    # the series
    lengths = {}
    no_units: list[str] = []
    no_vintage: list[str] = []
    for name in data.names:
        col = frame[name]
        if not pd.api.types.is_numeric_dtype(col.dtype):
            raw = col.dropna().astype(str)
            numeric = pd.to_numeric(raw, errors="coerce")
            if len(raw) and numeric.notna().mean() >= 0.5 and numeric.isna().any():  # mostly numbers, some text
                bad = raw[numeric.isna()]
                add("error", name, "not numbers", f"{len(bad)} value(s) are text among numbers, e.g. {bad.iloc[0]!r}")
            continue  # categories (regions, sectors) are fine as text
        values = col.to_numpy(dtype=float)
        inf = np.isinf(values)
        if inf.any():
            add("error", name, "infinite values", f"{int(inf.sum())} infinite value(s), e.g. at {frame.index[inf][0]}")
        present = ~np.isnan(values)  # infinite values are reported above, not as missing
        valid = np.isfinite(values)
        if not present.any():
            add("warning", name, "no values", "the series has no values")
            continue
        if t is not None and not data.entity_axes:
            first, last = int(present.argmax()), len(present) - 1 - int(present[::-1].argmax())
            inside = (~present[first:last + 1]).sum()
            if inside:
                gaps = [str(p) for p, ok in zip(frame.index[first:last + 1], present[first:last + 1]) if not ok]
                add("warning", name, "missing values", f"{int(inside)} missing inside its span: {_list(gaps)}")
            lengths[name] = frame.index[last]
        finite = values[valid]
        if len(finite) > 1 and np.ptp(finite) == 0:
            add("warning", name, "constant", f"every value is {finite[0]:g}")
        elif len(finite) > 10:
            med = np.median(finite)
            mad = 1.4826 * np.median(np.abs(finite - med))
            if mad > 0:
                far = np.abs(values - med) > outlier * mad
                far &= valid
                if far.any():
                    where = [str(p) for p in frame.index[far]]
                    add("warning", name, "outliers", f"{int(far.sum())} value(s) more than {outlier:g} robust standard deviations "
                        f"from the median: {_list(where)}")
        var = data.variables.get(name)
        if var is not None and var.recipe is None and not var.derived_from:
            if not var.units:
                no_units.append(name)
            if var.vintage is None:
                no_vintage.append(name)
    if no_units:
        add("note", "", "no units", f"no units recorded for {_list(no_units)}")
    if no_vintage:
        add("note", "", "no vintage", f"no release date (vintage) recorded for {_list(no_vintage)}, so results cannot say "
            "which release they used")
    if lengths:
        latest = max(lengths.values())
        early = sorted(n for n, end in lengths.items() if end < latest)
        if early and len(early) < len(lengths):
            add("note", "", "ragged edge", f"{_list(early)} end before {latest} (the last period of the others)")
    report = Validation(tuple(found), len(frame), len(data.names))
    if strict and not report.ok:
        raise DataError("the data have errors: " + "; ".join(f"{f.series or 'dataset'}: {f.message}" for f in report.errors))
    return report
