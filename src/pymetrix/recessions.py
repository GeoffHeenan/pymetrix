"""Business-cycle dates: when recessions began and ended, for shading charts and for recession dummies.

- ``nber``: the United States, from the NBER's Business Cycle Dating Committee. The peaks and troughs are published
  monthly and quarterly; each is used at its own frequency, since the two are dated separately (in 1980 the monthly
  peak is January, the quarterly peak 1980Q1).
- ``cepr``: the euro area, from the CEPR-EABCN Euro Area Business Cycle Dating Committee, quarterly.

A recession runs from the period after a peak through the trough, as the NBER and FRED's recession indicators date
them. Dates as announced by October 2026; check the committees' pages for later announcements::

    data.plot("gdp_growth", shade="nber")            # shaded on any dated figure
    recessions("cepr", "Q")                         # (start, end, label) spans
    data.with_series(rec=recession_series(data))    # a 0/1 series
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .errors import SpecificationError

__all__ = ["SOURCES", "recession_series", "recessions", "turning_points"]

# peaks and troughs, as published: (peak, trough)
_NBER_MONTHLY = (("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"), ("1960-04", "1961-02"),
                 ("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
                 ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"))
_NBER_QUARTERLY = (("1948Q4", "1949Q4"), ("1953Q2", "1954Q2"), ("1957Q3", "1958Q2"), ("1960Q2", "1961Q1"),
                   ("1969Q4", "1970Q4"), ("1973Q4", "1975Q1"), ("1980Q1", "1980Q3"), ("1981Q3", "1982Q4"),
                   ("1990Q3", "1991Q1"), ("2001Q1", "2001Q4"), ("2007Q4", "2009Q2"), ("2019Q4", "2020Q2"))
_CEPR_QUARTERLY = (("1974Q3", "1975Q1"), ("1980Q1", "1982Q3"), ("1992Q1", "1993Q3"), ("2008Q1", "2009Q2"),
                   ("2011Q3", "2013Q1"), ("2019Q4", "2020Q2"))

#: source -> (label, {frequency: (peak, trough) pairs})
SOURCES: dict[str, tuple[str, dict[str, tuple[tuple[str, str], ...]]]] = {
    "nber": ("NBER recessions", {"M": _NBER_MONTHLY, "Q": _NBER_QUARTERLY}),
    "cepr": ("CEPR euro-area recessions", {"Q": _CEPR_QUARTERLY}),
}


def _source(source: str) -> tuple[str, dict[str, tuple[tuple[str, str], ...]]]:
    key = str(source).strip().lower()
    if key not in SOURCES:
        raise SpecificationError(f"no business-cycle dates {source!r}; available: {', '.join(SOURCES)}")
    return SOURCES[key]


def _code(freq: Any) -> str:
    """M, Q, Y, or another pandas frequency code."""
    text = str(getattr(freq, "freqstr", freq) or "Q").upper()
    base = text.split("-")[0].lstrip("0123456789").rstrip("ES")
    return {"A": "Y", "QS": "Q", "MS": "M", "YS": "Y"}.get(base, base[:1] if base[:1] in ("M", "Q", "Y") else text)


def turning_points(source: str = "nber", freq: Any = "Q") -> pd.DataFrame:
    """The peaks and troughs, as published at ``freq`` where the committee dates at it (else at the finest it does)."""
    _, tables = _source(source)
    code = _code(freq)
    table = tables.get(code) or tables.get("M") or tables["Q"]
    return pd.DataFrame(table, columns=["peak", "trough"])


def recessions(source: str = "nber", freq: Any = "Q") -> tuple[tuple[str, str, str], ...]:
    """The recessions as (first period, last period, label) at ``freq``: from the period after each peak through the
    trough. Monthly and quarterly use the committee's own dates at that frequency where it publishes them; other
    frequencies (and a frequency it does not date at) cover the same span of time."""
    label, tables = _source(source)
    code = _code(freq)
    if code in tables:
        return tuple((str(pd.Period(p, freq=code) + 1), str(pd.Period(t, freq=code)), label) for p, t in tables[code])
    native = "M" if "M" in tables else "Q"
    spans = []
    for p, t in tables[native]:
        first = pd.Period(p, freq=native) + 1
        last = pd.Period(t, freq=native)
        spans.append((str(pd.Period(first.start_time, freq=freq)), str(pd.Period(last.end_time.normalize(), freq=freq)), label))
    return tuple(spans)


def recession_series(data: Any, source: str = "nber") -> pd.Series:
    """1 in recession periods and 0 otherwise, for the periods of ``data`` (a dated dataset, Series or PeriodIndex),
    named ``<source>_recession``; periods before the committee's first date are 0."""
    if isinstance(data, pd.PeriodIndex):
        index = data
    elif isinstance(data, pd.Series):
        index = data.index
    else:  # a dataset
        index = getattr(getattr(data, "frame", None), "index", None)
    if not isinstance(index, pd.PeriodIndex):
        raise SpecificationError("recession dates need dated (time-series) data")
    flags = pd.Series(0.0, index=index, name=f"{str(source).lower()}_recession")
    for start, end, _ in recessions(source, index.freq):
        flags[(index >= pd.Period(start, freq=index.freq)) & (index <= pd.Period(end, freq=index.freq))] = 1.0
    return flags
