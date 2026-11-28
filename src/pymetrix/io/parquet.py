"""Native data format: Parquet, with axes and variable metadata in the file footer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..errors import DataError
from .._serial import to_jsonable
from ..data import CodeList, Dataset, EntityAxis, TimeAxis
from ..metadata import DatasetMeta, Variable

KEY = b"pymetrix"
LEGACY_KEYS = (b"pymetricsmodels",)  # files written before the package was renamed
FORMAT_VERSION = 1


def _axis_to_dict(axis: TimeAxis | EntityAxis) -> dict[str, Any]:
    if isinstance(axis, TimeAxis):
        return {"kind": "time", "name": axis.name, "freq": axis.freq, "calendar": axis.calendar}
    codes = None
    if axis.codes is not None:
        codes = {"codes": dict(axis.codes.codes), "groups": {k: sorted(v) for k, v in axis.codes.groups.items()}}
    return {"kind": "entity", "name": axis.name, "codes": codes}


def _axis_from_dict(d: dict[str, Any]) -> TimeAxis | EntityAxis:
    if d["kind"] == "time":
        return TimeAxis(d["name"], d["freq"], d.get("calendar"))
    codes = d.get("codes")
    if codes:
        codes = CodeList(codes["codes"], {k: frozenset(v) for k, v in codes.get("groups", {}).items()})
    return EntityAxis(d["name"], codes)


def write_parquet(ds: Dataset, path: str, **kw: Any) -> None:
    df = ds.frame.reset_index()
    t = ds.time_axis
    if t is not None:
        df[t.name] = df[t.name].astype(str)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {
        "format_version": FORMAT_VERSION,
        "axes": [_axis_to_dict(a) for a in ds.axes],
        "variables": {n: v.to_dict() for n, v in ds.variables.items()},
        "meta": {k: v for k, v in to_jsonable(ds.meta).items() if k != "type"},
        "flags": _flags_to_list(ds),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), KEY: json.dumps(meta).encode()})
    pq.write_table(table, path, **kw)


def read_parquet(path: Any, columns: Any = None, where: str | None = None, span: Any = None, **kw: Any) -> Dataset:
    """A dataset from a Parquet file, or a folder of them (partitioned as ``year=2010/country=DEU/...``). ``columns``,
    ``where`` and ``span`` read only part of it (``pymetrix.io.selection.read_selected``); without them a file is read
    whole, as it always was."""
    from pathlib import Path

    if columns is not None or where is not None or span is not None or (isinstance(path, (str, Path)) and Path(path).is_dir()):
        from .selection import read_selected

        return read_selected(str(path), columns, where, span, **kw)
    table = pq.read_table(path)
    return frame_to_dataset(table.to_pandas(), pymetrix_meta(table.schema.metadata or {}, path), kw)


def pymetrix_meta(stored: dict[bytes, bytes], path: Any) -> dict[str, Any] | None:
    """pymetrix's metadata from a Parquet schema's key-value metadata (None for a plain file)."""
    raw = next((stored[k] for k in (KEY, *LEGACY_KEYS) if k in stored), None)
    if raw is None:
        return None
    meta = json.loads(raw)
    if meta.get("format_version", 1) > FORMAT_VERSION:
        raise DataError(f"{path} was written by a newer version of pymetrix")
    return meta


def frame_to_dataset(df: pd.DataFrame, meta: dict[str, Any] | None, kw: dict[str, Any]) -> Dataset:
    """The dataset from a Parquet file's rows: by pymetrix's metadata, or, for a plain file, the axes named in ``kw``."""
    if meta is None:  # a plain Parquet file: the caller names the axes
        return Dataset.from_frame(df, **kw)
    axes = tuple(_axis_from_dict(a) for a in meta["axes"])
    for a in axes:
        if isinstance(a, TimeAxis):
            df[a.name] = pd.PeriodIndex(df[a.name], freq=a.freq)
    frame = df.set_index([a.name for a in axes])
    variables = {n: Variable.from_dict(v) for n, v in meta.get("variables", {}).items()}
    known = DatasetMeta.__dataclass_fields__
    ds_meta = DatasetMeta(**{k: v for k, v in meta.get("meta", {}).items() if k in known})
    ds = Dataset(frame, axes, variables, meta=ds_meta)
    return _with_flags(ds, meta.get("flags", []))


def _flags_to_list(ds: Dataset) -> list[list[Any]]:
    """Observation flags as [series, row key, status] triples (only flagged cells)."""
    if ds.flags is None:
        return []
    out = []
    for name in ds.flags.columns:
        column = ds.flags[name]
        for label, status in column[column.notna() & (column != "")].items():
            parts = label if isinstance(label, tuple) else (label,)
            out.append([str(name), [str(p) for p in parts], str(status)])
    return out


def _with_flags(ds: Dataset, triples: list[list[Any]]) -> Dataset:
    if not triples:
        return ds
    flags = pd.DataFrame(index=ds.frame.index)
    for name, parts, status in triples:
        if name not in ds.frame.columns:
            continue  # a series not read (columns=)
        key = ds.row_key(tuple(parts) if len(parts) > 1 else parts[0])
        if key not in flags.index:
            continue  # a row not read (where=, span=)
        if name not in flags.columns:
            flags[name] = pd.Series(None, index=ds.frame.index, dtype=object)
        flags.loc[key, name] = status
    if flags.columns.empty:
        return ds
    return Dataset(ds.frame, ds.axes, ds.variables, flags, ds.meta)
