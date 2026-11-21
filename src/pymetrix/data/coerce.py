"""Plain pandas in, Dataset out: the simple path into pymetrix.

    fit = Equation("c ~ y").fit(df)          # a DataFrame with a PeriodIndex (or dates with a frequency) works
    data = as_dataset(df)                    # the same conversion, explicitly
    df = data.to_pandas()                    # and back: the series as plain columns, no metadata

Metadata, vintages and flags stay opt-in: a frame converted here has none until you add them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..errors import DataError

__all__ = ["as_dataset"]

_HOW = ("give the frame a PeriodIndex (df.index = pd.period_range('2000Q1', periods=len(df), freq='Q')), or build the "
        "dataset yourself: Dataset.from_frame(df, time='date', freq='Q') (entity='country' for panels)")


_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _period_freq(freq: str) -> str:
    """The period frequency of a date frequency: ``MS``/``ME`` → ``M``, ``QS-OCT``/``QE-DEC`` → ``Q``, ``YS-JUL`` →
    ``Y-JUN`` (a July-June year); anything pymetrix cannot map is refused."""
    base, _, anchor = freq.upper().partition("-")
    starts = base.endswith("S") and base not in ("S", "MS") or base in ("MS",)
    letter = base.rstrip("SE") or base
    letter = {"A": "Y", "BA": "Y", "BQ": "Q", "BM": "M", "SM": "M", "C": "B", "H": "h"}.get(letter, letter)
    if letter in ("M", "W", "D", "B", "h"):
        return letter if letter != "W" or not anchor else f"W-{anchor}"
    if letter in ("Q", "Y") and anchor in _MONTHS:
        month = _MONTHS.index(anchor)
        end = (month - 1) % 12 if starts else month  # a period starting in month m ends in month m - 1
        if letter == "Q":
            if end % 3 == 2:  # calendar quarters (ending Mar, Jun, Sep, Dec)
                return "Q"
            raise DataError(f"quarters ending in {_MONTHS[end]} are not calendar quarters; {_HOW}")
        return "Y" if end == 11 else f"Y-{_MONTHS[end]}"
    if letter in ("Q", "Y"):
        return letter
    raise DataError(f"pymetrix cannot turn dates of frequency {freq!r} into periods; {_HOW}")


def _user(data: Any) -> Any:
    from dataclasses import replace

    return type(data)(data.frame, data.axes, data.variables, data.flags, replace(data.meta, vintage={"policy": "user"}))


def as_dataset(value: Any) -> Any:
    """``value`` as a ``Dataset``: a Dataset unchanged; a DataFrame with a ``PeriodIndex``, with dates that have a
    frequency, or with a ``MultiIndex`` of entities and one period level, converted (each series recorded as coming
    from ``user:pandas``). Anything else is refused with a message saying how to make a dataset."""
    from .dataset import Dataset

    if isinstance(value, Dataset):
        return value
    if isinstance(value, pd.Series):
        value = value.to_frame(value.name or "value")
    if not isinstance(value, pd.DataFrame):
        raise DataError(f"expected a Dataset or a pandas DataFrame, got a {type(value).__name__}")
    index = value.index
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freqstr or pd.infer_freq(index)
        if not freq:
            raise DataError(f"the frame's dates have no regular frequency, so pymetrix cannot tell the periods; {_HOW}")
        value = value.copy()
        value.index = index.to_period(_period_freq(freq)).rename(index.name or "time")
        index = value.index
    if isinstance(index, pd.PeriodIndex):
        return _user(Dataset.from_frame(value, source="user:pandas"))
    if isinstance(index, pd.MultiIndex):
        periods = [n for n, lv in zip(index.names, index.levels) if isinstance(lv, pd.PeriodIndex)]
        if len(periods) == 1 and all(index.names):
            time = periods[0]
            entities = [n for n in index.names if n != time]
            flat = value.reset_index()
            return _user(Dataset.from_frame(flat, time=time, entity=entities, freq=index.levels[index.names.index(time)].freqstr,
                                            source="user:pandas"))
    raise DataError(f"pymetrix needs to know the frame's periods; {_HOW}")
