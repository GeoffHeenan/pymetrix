"""CSV and Excel. These formats carry no metadata, so the caller names the axes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..data import Dataset


def _flat(ds: Dataset) -> pd.DataFrame:
    df = ds.frame.reset_index()
    t = ds.time_axis
    if t is not None:
        df[t.name] = df[t.name].astype(str)
    return df


def _sourced(ds: Dataset, path: str) -> Dataset:
    """Each series names the file it came from and when it was read; a file has no release date, so no vintage."""
    from dataclasses import replace
    from datetime import datetime
    from pathlib import Path

    when = datetime.now().isoformat(timespec="seconds")
    name = Path(str(path)).name
    variables = {n: replace(v, source=v.source or f"file {name}", retrieved=v.retrieved or when) for n, v in ds.variables.items()}
    return Dataset(ds.frame, ds.axes, variables, ds.flags, ds.meta)


def read_csv(path: str, *, time: str | None = None, entity: Any = None, freq: str | None = None, start: str | None = None,
             **kw: Any) -> Dataset:
    """A CSV file: ``time`` names its column of dates, or ``start`` dates a file without one (rows in order, from that
    period at frequency ``freq``)."""
    return _sourced(Dataset.from_frame(pd.read_csv(path, **kw), time=time, entity=entity, freq=freq, start=start), path)


def write_csv(ds: Dataset, path: str, **kw: Any) -> None:
    _flat(ds).to_csv(path, index=False, **kw)


def read_excel(path: str, *, time: str | None = None, entity: Any = None, freq: str | None = None, start: str | None = None,
               **kw: Any) -> Dataset:
    """An Excel sheet, with its time column or a ``start`` as for ``read_csv``."""
    return _sourced(Dataset.from_frame(pd.read_excel(path, **kw), time=time, entity=entity, freq=freq, start=start), path)


def write_excel(ds: Dataset, path: str, **kw: Any) -> None:
    _flat(ds).to_excel(path, index=False, **kw)
