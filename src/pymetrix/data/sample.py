"""Samples: which periods, entities and rows a command uses, as a value that can be named, combined and reused.

    precrisis = Sample(end="2007Q4")
    core = Sample(("1995Q1", "2019Q4"), country=["DEU", "FRA", "ITA"])
    large = Sample(where="pop > 1e6")
    windows = Sample(windows={"DEU": ("1991Q1", None), "*": ("1980Q1", None)})   # a span per entity

    data = data.with_sample(core & large)      # the dataset's active sample (EViews: smpl)
    model.fit(data)                            # uses it
    model.fit(data, span=precrisis)            # an explicit sample replaces it for this call
    data = data.with_sample(None)              # all of it again (smpl @all)

A sample only selects rows: lags and other transformations are computed on all the data first, so the first
period of a sample still has its lagged values, and lags never cross from one entity to the next. Generating
series is not limited by the active sample. ``&`` (and), ``|`` (or) and ``~`` (not) combine samples. Conditions
are kept as text and evaluated on the data each time, so a revised dataset is selected by the same rules;
``describe(data)`` gives the rows, entities and periods a sample selects there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataError

__all__ = ["Sample", "as_span"]


def as_span(span: Any) -> Any:
    """A span as ``(start, end)``: a pair as given, or text ``"1990Q1:2019Q4"`` (``"1990Q1:"`` and ``":2019Q4"``
    leave one end open). ``None`` and ``Sample`` objects pass through."""
    if isinstance(span, str):
        if ":" not in span:
            raise DataError(f"a span is 'start:end', e.g. '1990Q1:2019Q4' (or a pair); got {span!r}")
        start, _, end = span.partition(":")
        return (start.strip() or None, end.strip() or None)
    return span


class Sample:
    """A selection of periods, entities and rows: see the module docstring."""

    __slots__ = ("start", "end", "where", "levels", "windows", "op", "parts")

    def __init__(self, span: tuple[Any, Any] | None = None, *, start: Any = None, end: Any = None, where: str | None = None,
                 windows: Mapping[str, tuple[Any, Any]] | None = None, **levels: Any) -> None:
        span = as_span(span)
        if span is not None:
            if start is not None or end is not None:
                raise DataError("give the periods as span=(start, end) or as start= and end=, not both")
            if isinstance(span, (str, pd.Period)) or len(span) != 2:
                raise DataError("span is a pair (start, end); use None for an open end, e.g. (None, '2007Q4')")
            start, end = span
        set_ = object.__setattr__
        set_(self, "start", None if start is None else str(start))
        set_(self, "end", None if end is None else str(end))
        if where is not None and not isinstance(where, str):
            raise TypeError("where= is a condition as text, e.g. \"pop > 1e6\"")
        set_(self, "where", where)
        set_(self, "levels", {k: tuple(v) if isinstance(v, (list, tuple, set, frozenset)) else (v,) for k, v in sorted(levels.items())})
        set_(self, "windows", {str(k): (None if a is None else str(a), None if b is None else str(b))
                               for k, (a, b) in (windows or {}).items()})
        set_(self, "op", "")
        set_(self, "parts", ())

    @classmethod
    def _combined(cls, op: str, parts: tuple[Sample, ...]) -> Sample:
        out = cls()
        object.__setattr__(out, "op", op)
        object.__setattr__(out, "parts", parts)
        return out

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a Sample cannot be changed; make a new one")

    def __and__(self, other: Sample) -> Sample:
        return Sample._combined("and", (self, _as_sample(other)))

    def __or__(self, other: Sample) -> Sample:
        return Sample._combined("or", (self, _as_sample(other)))

    def __invert__(self) -> Sample:
        return Sample._combined("not", (self,))

    @property
    def everything(self) -> bool:
        """True for a sample that selects every row."""
        return not self.op and self.start is None and self.end is None and self.where is None and not self.levels and not self.windows

    def mask(self, data: Any) -> np.ndarray:
        """The rows of ``data`` (a Dataset) this sample selects."""
        if self.op == "and":
            return self.parts[0].mask(data) & self.parts[1].mask(data)
        if self.op == "or":
            return self.parts[0].mask(data) | self.parts[1].mask(data)
        if self.op == "not":
            return ~self.parts[0].mask(data)
        mask = np.ones(len(data.frame), dtype=bool)
        if self.start is not None or self.end is not None:
            mask &= _between(data, self.start, self.end)
        if self.windows:
            if len(data.entity_axes) != 1:
                raise DataError("windows= needs one entity axis (a panel): one span per entity")
            ids = np.asarray(data.values_of(data.entity_axes[0].name)).astype(str)
            default = self.windows.get("*")
            chosen = np.zeros(len(mask), dtype=bool)
            for name in np.unique(ids):
                window = self.windows.get(name, default)
                rows = ids == name
                chosen |= rows & (_between(data, *window) if window is not None else True)
            mask &= chosen
        if self.where is not None:
            from .conditions import condition_mask

            mask &= condition_mask(data.frame, self.where)  # checked before it runs: a project's conditions may be anyone's
        for level, values in self.levels.items():
            mask &= np.asarray(data.values_of(level).isin(values))
        return mask

    def describe(self, data: Any) -> str:
        """What the sample selects in ``data``: rows, and entities and periods where there are such axes."""
        mask = self.mask(data)
        parts = [f"{int(mask.sum())} of {len(mask)} rows"]
        for axis in data.entity_axes:
            ids = data.values_of(axis.name)[mask]
            parts.append(f"{ids.nunique()} {axis.name}")
        if data.time_axis is not None and mask.any():
            periods = data.frame.index.get_level_values(data.time_axis.name)[mask]
            parts.append(f"{periods.min()} to {periods.max()}")
        return ", ".join(parts)

    # -- text, comparison, storage -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        if self.op:
            return {"op": self.op, "parts": [p.to_dict() for p in self.parts]}
        out: dict[str, Any] = {}
        if self.start is not None:
            out["start"] = self.start
        if self.end is not None:
            out["end"] = self.end
        if self.where is not None:
            out["where"] = self.where
        if self.levels:
            out["levels"] = {k: list(v) for k, v in self.levels.items()}
        if self.windows:
            out["windows"] = {k: list(v) for k, v in self.windows.items()}
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Sample:
        if "op" in d:
            return cls._combined(d["op"], tuple(cls.from_dict(p) for p in d["parts"]))
        return cls(start=d.get("start"), end=d.get("end"), where=d.get("where"),
                   windows={k: tuple(v) for k, v in d.get("windows", {}).items()} or None, **{k: list(v) for k, v in d.get("levels", {}).items()})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sample) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        if self.op == "not":
            return f"~{self.parts[0]!r}"
        if self.op:
            return f"({self.parts[0]!r} {'&' if self.op == 'and' else '|'} {self.parts[1]!r})"
        args = []
        if self.start is not None and self.end is not None:
            args.append(repr((self.start, self.end)))
        elif self.start is not None:
            args.append(f"start={self.start!r}")
        elif self.end is not None:
            args.append(f"end={self.end!r}")
        if self.where is not None:
            args.append(f"where={self.where!r}")
        if self.windows:
            args.append(f"windows={ {k: tuple(v) for k, v in self.windows.items()}!r}")
        for k, v in self.levels.items():
            args.append(f"{k}={list(v)!r}")
        return f"Sample({', '.join(args)})"

    def __str__(self) -> str:
        if self.op == "not":
            return f"not ({self.parts[0]})"
        if self.op:
            return f"({self.parts[0]}) {self.op} ({self.parts[1]})"
        bits = []
        if self.start is not None or self.end is not None:
            bits.append(_period_text(self.start, self.end))
        if self.windows:
            bits.append("; ".join(f"{k}: {_period_text(a, b)}" for k, (a, b) in self.windows.items()))
        for k, v in self.levels.items():
            bits.append(f"{k} in {', '.join(map(str, v))}")
        if self.where is not None:
            bits.append(f"where {self.where}")
        return ", ".join(bits) or "all"


def _period_text(start: Any, end: Any) -> str:
    if start is not None and end is not None:
        return f"{start} to {end}"
    return f"from {start}" if start is not None else f"to {end}" if end is not None else "all periods"


def _as_sample(value: Any) -> Sample:
    if isinstance(value, Sample):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return Sample(value)
    raise TypeError(f"combine a Sample with a Sample or a (start, end) span, not {type(value).__name__}")


def _between(data: Any, start: Any, end: Any) -> np.ndarray:
    t = data.time_axis
    if t is None:
        raise DataError("a span of periods needs a time axis; use where= to select a cross-section")
    periods = data.frame.index.get_level_values(t.name)
    mask = np.ones(len(periods), dtype=bool)
    if start is not None:
        mask &= np.asarray(periods >= pd.Period(str(start), freq=periods.freq))
    if end is not None:
        mask &= np.asarray(periods <= pd.Period(str(end), freq=periods.freq))
    return mask
