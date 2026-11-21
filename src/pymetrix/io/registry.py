"""One entry point for reading and writing, dispatched by URI scheme or file extension."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._api import renamed_arguments
from ..errors import DataError
from .. import plugins

if TYPE_CHECKING:
    from ..data import Dataset

Reader = Callable[..., "Dataset"]
Writer = Callable[..., None]

_READERS: dict[str, Reader] = {}
_WRITERS: dict[str, Writer] = {}
_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*)://", re.IGNORECASE)


@dataclass(frozen=True)
class Connector:
    """How File → Load from connector… writes a connector's address: ``scheme://codes?start=...&end=...``.

    ``codes`` says what goes after ``scheme://`` (with an example); ``start`` and ``end`` are the address's parameters
    for the first and last period ("" when the connector takes none); ``vintage`` is True when the reader honours
    ``Dataset.read(..., vintage=...)`` (the data as published on a day); ``search`` is True when its series can be
    searched and browsed (``pymetrix.connectors.search``, File → Browse series…), with ``search_note`` saying what
    that needs (an API key, say)."""

    label: str
    codes: str = ""
    start: str = ""
    end: str = ""
    vintage: bool = False
    search: bool = False
    search_note: str = ""


_CONNECTORS: dict[str, Connector] = {}


def register_reader(key: str, fn: Reader, connector: Connector | None = None) -> None:
    """``key`` is a URI scheme ("sdmx") or a file extension (".parquet"). A scheme's ``connector`` describes its
    addresses for the Load from connector dialog."""
    _READERS[key.lower()] = fn
    if connector is not None:
        _CONNECTORS[key.lower()] = connector


def connectors() -> dict[str, Connector]:
    """Every registered connector (a reader of a URI scheme, plugins' included), by scheme, with its description (a
    plain one for a scheme registered without)."""
    plugins.load("pymetrix.io")
    return {key: _CONNECTORS.get(key, Connector(key)) for key in sorted(_READERS) if not key.startswith(".")}


def register_writer(key: str, fn: Writer) -> None:
    _WRITERS[key.lower()] = fn


def _key(location: str) -> str:
    m = _SCHEME.match(location)
    return m.group(1).lower() if m else Path(location).suffix.lower()


def _lookup(table: dict[str, Any], location: str, what: str) -> Any:
    key = _key(location)
    if key not in table:
        plugins.load("pymetrix.io")
    if key not in table:
        raise DataError(f"no {what} for {key or location!r}; known: {sorted(table)}")
    return table[key]


def columns(source: str) -> list[str]:
    """Column names of a data file, read cheaply (no data), for forms. Unknown types give []."""
    import pandas as pd

    key = _key(source)
    try:
        if key == ".parquet":
            import pyarrow.parquet as pq

            return list(pq.read_schema(source).names)
        if key == ".csv":
            return [str(c) for c in pd.read_csv(source, nrows=0).columns]
        if key in (".xlsx", ".xls"):
            return [str(c) for c in pd.read_excel(source, nrows=0).columns]
    except (OSError, ValueError, ImportError):
        return []
    return []


@dataclass(frozen=True)
class Sniff:
    """What the first rows of a data file suggest: its time column and frequency, an entity column, and why."""

    head: Any  # the first rows, as a DataFrame
    time: str | None = None
    entity: str | None = None
    freq: str | None = None
    notes: tuple[str, ...] = ()
    axes_in_file: bool = False  # a pymetrix Parquet file: its axes are saved with it


_TIME_NAMES = ("date", "time", "period", "year", "quarter", "month", "obs", "week", "day")
_ENTITY_NAMES = ("id", "country", "firm", "state", "region", "name", "code", "person", "unit", "company", "household", "iso")
_PERIOD_PATTERNS = ((r"^\d{4}\s*[-_ ]?[Qq][1-4]$", "Q"), (r"^\d{4}[Qq][1-4]$", "Q"), (r"^\d{4}\s*[-M]\s*(0?[1-9]|1[0-2])$", "M"),
                    (r"^\d{4}$", "Y"))


def _frequency_of(values: Any) -> tuple[str | None, str]:
    """The frequency a column of times suggests, and how it was told; None when it is not a time column."""
    import re

    import pandas as pd

    s = pd.Series(values).dropna()
    if s.empty:
        return None, ""
    if pd.api.types.is_integer_dtype(s) or (pd.api.types.is_float_dtype(s) and (s == s.round()).all()):
        ints = s.astype(int)
        if ints.between(1800, 2200).all():
            return "Y", "whole numbers between 1800 and 2200: years"
        return None, ""
    text = s.astype(str).str.strip()
    for pattern, freq in _PERIOD_PATTERNS:
        if text.map(lambda v, p=pattern: re.match(p, v) is not None).all():
            return freq, {"Q": "periods like 2001Q1", "M": "periods like 2001-01", "Y": "four-digit years"}[freq]
    if pd.api.types.is_numeric_dtype(s):
        return None, ""
    try:
        dates = pd.to_datetime(text, errors="raise", format="mixed")
    except (ValueError, TypeError):
        return None, ""
    gaps = pd.Series(sorted(set(dates))).diff().dt.days.dropna()
    if gaps.empty:
        return None, ""
    gap = float(gaps.median())
    for freq, lo, hi, word in (("D", 0.5, 1.5, "daily"), ("W", 6, 8, "weekly"), ("M", 27, 32, "monthly"), ("Q", 85, 95, "quarterly"),
                               ("Y", 360, 370, "annual")):
        if lo <= gap <= hi:
            return freq, f"dates about {gap:.0f} days apart: {word}"
    return None, ""


def sniff(source: str, rows: int = 500) -> Sniff:
    """Read the first ``rows`` rows of a data file and guess, for Load data, which column holds the periods (and their
    frequency) and which identifies the units (a column that, with the period, picks out one row). Nothing is
    loaded; unreadable files give an empty guess."""
    import pandas as pd

    key = _key(source)
    try:
        if key == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(source)
            head = next(pf.iter_batches(batch_size=rows)).to_pandas() if pf.metadata.num_rows else pd.DataFrame()
            if pf.schema_arrow.metadata and any(k.startswith(b"pymetrix") for k in pf.schema_arrow.metadata):
                return Sniff(head.head(20), notes=("a pymetrix Parquet file: its periods and units are saved with it",), axes_in_file=True)
            head = head.reset_index() if not isinstance(head.index, pd.RangeIndex) else head
        elif key == ".csv":
            head = pd.read_csv(source, nrows=rows)
        elif key in (".xlsx", ".xls"):
            head = pd.read_excel(source, nrows=rows)
        else:
            return Sniff(pd.DataFrame(), notes=("this kind of file is not previewed",))
    except (OSError, ValueError, ImportError, StopIteration) as exc:
        return Sniff(pd.DataFrame(), notes=(f"could not be read: {exc}",))
    notes: list[str] = []
    time = freq = None
    ordered = sorted(head.columns, key=lambda c: (str(c).strip().lower() not in _TIME_NAMES, list(head.columns).index(c)))
    for col in ordered:
        guess, why = _frequency_of(head[col])
        if guess is not None and (str(col).strip().lower() in _TIME_NAMES or guess != "Y"):
            time, freq = str(col), guess
            notes.append(f"time: {col} ({why})")
            break
    entity = None
    rest = [c for c in head.columns if str(c) != time]
    named = sorted(rest, key=lambda c: (not any(w in str(c).lower() for w in _ENTITY_NAMES), list(head.columns).index(c)))
    repeats = time is not None and head[time].duplicated().any()
    for col in named:
        values = head[col]
        if pd.api.types.is_float_dtype(values) and not (values.dropna() == values.dropna().round()).all():
            continue  # measurements, not identifiers
        keys = head[[time, col]] if time is not None else head[[col]]
        unique = not keys.duplicated().any()
        if time is not None and repeats and unique and values.nunique() > 1:
            entity = str(col)
            notes.append(f"entity: {col} (with {time}, each row is one unit in one period: a panel)")
            break
        if time is None and unique and any(w in str(col).lower() for w in _ENTITY_NAMES):
            entity = str(col)
            notes.append(f"entity: {col} (a different value in each row: a cross section)")
            break
    if time is None and entity is None:
        notes.append("no time or identifier column found: choose them, or the rows cannot be told apart")
    elif time is not None and not repeats:
        notes.append("one row per period: a single time series")
    return Sniff(head.head(20), time, entity, freq, tuple(notes))


def read(source: str | Path, **kw: Any) -> Dataset:
    """A dataset from a file (by its extension) or a connector (``fred://GDPC1``); a path may be a string or a Path."""
    source = str(source)
    data = _lookup(_READERS, source, "reader")(source, **kw)
    return _with_load(data, source, kw)


def _with_load(data: Dataset, source: str, options: dict[str, Any]) -> Dataset:
    """The dataset with this load added to ``meta.extra["loads"]``: where from (a file or a connector's address), with
    which options, and when. Kept in Parquet files and projects; a list, so datasets combined later keep each load."""
    from dataclasses import replace
    from datetime import datetime

    entry = {"source": source, "retrieved": datetime.now().isoformat(timespec="seconds")}
    if options:
        entry["options"] = {k: str(v) for k, v in options.items()}
    loads = [*list((data.meta.extra or {}).get("loads", [])), entry]
    meta = replace(data.meta, extra={**dict(data.meta.extra or {}), "loads": loads})
    return type(data)(data.frame, data.axes, data.variables, data.flags, meta)


@renamed_arguments(target="path")
def write(dataset: Dataset, path: str | Path, **kw: Any) -> None:
    path = str(path)
    _lookup(_WRITERS, path, "writer")(dataset, path, **kw)
