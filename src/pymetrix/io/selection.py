"""Large Parquet data: look at a file's metadata, then read only the columns and rows needed.

pymetrix's datasets live in memory (pandas). A Parquet file too large for that can still be used by reading part of
it: ``scan`` reports a file's columns, row count, row groups and each column's minimum and maximum from its footer,
without reading data; ``Dataset.read(path, columns=..., where=..., span=...)`` (``read_selected``) then reads only
the columns asked for and, through ``pyarrow.dataset``, only the partition folders and row groups a condition does not
rule out. pyarrow does the reading and the pruning; ``pymetrix.io.pushdown`` turns pymetrix's row conditions into
pyarrow's filter expressions, and the condition is applied exactly afterwards, so the rows kept are those reading the
whole file and selecting would keep.

``fingerprint`` identifies a file's contents from its size and footer (which records every row group's place, size and
statistics), without reading the data: projects that keep only a reference to a large file check it on opening.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from ..data import Dataset
from ..errors import DataError
from .parquet import frame_to_dataset, pymetrix_meta

__all__ = ["LARGE", "SELECTION", "ColumnSummary", "ParquetScan", "fingerprint", "is_large_parquet", "read_selected", "scan"]

LARGE = 100 * 2**20  # bytes: Load data offers columns and a row condition for Parquet data larger than this
SELECTION = "_selection"  # what read_selected read: moved into the load's provenance by pymetrix.io.read


def _source(path: str) -> Any:
    import pyarrow.dataset as pads

    folder = Path(path).is_dir()
    try:
        return pads.dataset(path, format="parquet", partitioning="hive" if folder else None)
    except (OSError, pa.ArrowInvalid) as exc:
        raise DataError(f"{path} cannot be read as Parquet: {exc}") from None


def _pandas_index_columns(schema: pa.Schema) -> list[str]:
    """Columns pandas wrote its index to (named in the file's pandas metadata): read so the index comes back."""
    raw = (schema.metadata or {}).get(b"pandas")
    if not raw:
        return []
    try:
        return [c for c in json.loads(raw).get("index_columns", []) if isinstance(c, str)]
    except (ValueError, AttributeError):
        return []


def read_selected(path: str, columns: Any = None, where: str | None = None, span: Any = None, **kw: Any) -> Dataset:
    """Read only part of a Parquet file or a folder of them, with ``pyarrow.dataset``.

    - ``columns``: the series to keep (the axes' columns, and those ``where`` uses, are read too; columns not asked
      for are dropped once the condition has been applied).
    - ``where``: a row condition in pymetrix's syntax (``pymetrix.data.conditions``). The parts pyarrow can test
      (``pymetrix.io.pushdown``) are pushed down, so partition folders and row groups whose statistics rule them out
      are not read; the whole condition is then applied to the rows read as ``Dataset.select(where=...)`` applies it.
    - ``span``: ``(start, end)`` or ``"start:end"`` on the time axis (a pymetrix file's, or the ``time=`` column),
      pushed down the same way and then applied exactly.

    What was read goes in the load's provenance (``meta.extra["loads"][-1]["selection"]``): the columns, condition and
    span, the filter pushed down, the parts applied only after reading, and the rows, row groups and files read
    against those in the data."""
    from ..data.conditions import check_condition
    from ..data.sample import as_span
    from .pushdown import _backtick_names, condition_expression, span_expression

    source = _source(path)
    schema = source.schema
    meta = pymetrix_meta(schema.metadata or {}, path)
    periods: dict[str, str] = {}
    plain_time: tuple[str, str | None] | None = None
    if meta is not None:
        axis_names = [a["name"] for a in meta["axes"]]
        time_axis = next((a for a in meta["axes"] if a["kind"] == "time"), None)
        time, time_freq = (time_axis["name"], time_axis["freq"]) if time_axis else (None, None)
        if time is not None:
            periods[time] = time_freq
    else:
        entity = kw.get("entity")
        time, time_freq = kw.get("time") or None, None
        axis_names = ([entity] if isinstance(entity, str) else list(entity or [])) + ([time] if time else [])
        if time:
            plain_time = (time, kw.get("freq") or None)  # its values become periods only once read
    index_columns = [c for c in _pandas_index_columns(schema) if c in schema.names]
    known = [n for n in schema.names if n not in index_columns]
    if columns is not None:
        columns = [c.strip() for c in columns.split(",") if c.strip()] if isinstance(columns, str) else list(columns)
    used: list[str] = []
    expr = None
    left_out: list[str] = []
    if where is not None:
        tree = check_condition(where)
        names = _backtick_names(where)
        used = [n for n in dict.fromkeys(names.get(x.id, x.id) for x in ast.walk(tree) if isinstance(x, ast.Name)) if n in schema.names]
        expr, left_out = condition_expression(where, schema, plain_time, periods)
    span = as_span(span)
    span_note = None
    if span is not None:
        if time is None:
            raise DataError("span= needs a time axis: name the file's time column (time=), or select rows with where=")
        if time not in schema.names:
            raise DataError(f"{path} has no column {time} (its columns: {', '.join(known)})")
        cut = span_expression(time, schema.field(time).type, tuple(span), kw.get("freq") or time_freq, periods.get(time))
        if cut is None:
            span_note = f"span on {time} (a {schema.field(time).type} column): applied after reading"
        else:
            expr = cut if expr is None else expr & cut
    wanted = known if columns is None else columns
    needed = list(dict.fromkeys([*axis_names, *index_columns, *wanted, *used]))
    missing = [c for c in needed if c not in schema.names]
    if missing:
        raise DataError(f"{path} has no column{'s' if len(missing) > 1 else ''} {', '.join(missing)} "
                        f"(its columns: {', '.join(known)})")
    every = list(source.get_fragments())
    rows_in_file = sum(f.metadata.num_rows for f in every)
    groups_in_file = sum(f.metadata.num_row_groups for f in every)
    if expr is None:
        table = source.to_table(columns=needed)
        groups_read, rows_read, files_read = groups_in_file, rows_in_file, len(every)
    else:
        import pyarrow.dataset as pads

        kept = list(source.get_fragments(filter=expr))  # partition folders their names rule out are dropped here
        pieces = [g for f in kept for g in f.split_by_row_group(filter=expr, schema=schema)]  # row groups, by statistics
        groups_read = len(pieces)
        rows_read = sum(g.row_groups[0].num_rows for g in pieces)
        files_read = len({g.path for g in pieces})
        if pieces:
            table = pads.FileSystemDataset(pieces, schema, source.format, filesystem=source.filesystem).to_table(
                columns=needed, filter=expr)
        else:
            table = schema.empty_table().select(needed)
    rows_after_filter = table.num_rows
    if schema.metadata:
        table = table.replace_schema_metadata(schema.metadata)
    data = frame_to_dataset(table.to_pandas(), meta, kw)
    if where is not None or span is not None:
        data = data.select(where=where, span=span)  # exact: pyarrow's filter was a first cut
    if columns is not None:
        keep = [c for c in data.frame.columns if c in columns]
        if len(keep) < len(data.frame.columns):
            flags = data.flags[[c for c in data.flags.columns if c in keep]] if data.flags is not None else None
            data = Dataset(data.frame[keep], data.axes, {n: v for n, v in data._static().items() if n in keep},
                           flags if flags is not None and len(flags.columns) else None, data.meta)
    record: dict[str, Any] = {
        "columns": "all" if columns is None else list(columns),
        "where": where, "span": None if span is None else [None if p is None else str(p) for p in span],
        "pushed_down": None if expr is None else str(expr),
        "applied_after_reading": [*left_out, *([span_note] if span_note else [])] or None,
        "rows_in_file": rows_in_file, "rows_read": rows_read, "rows_after_filter": rows_after_filter, "rows_kept": len(data.frame),
        "row_groups_in_file": groups_in_file, "row_groups_read": groups_read,
        "files": len(every), "files_read": files_read,
    }
    record = {k: v for k, v in record.items() if v is not None}
    extra = {**dict(data.meta.extra or {}), SELECTION: record}
    return Dataset(data.frame, data.axes, data.variables, data.flags, replace(data.meta, extra=extra))


@dataclass(frozen=True)
class ColumnSummary:
    """A column as a Parquet file's metadata describes it: its type, its smallest and largest values (over the row
    groups whose statistics have them; None when none has), its missing values (None when not recorded), and whether
    it comes from the folder names of a partitioned dataset."""

    name: str
    type: str
    min: Any = None
    max: Any = None
    nulls: int | None = None
    partition: bool = False


@dataclass(frozen=True)
class ParquetScan:
    """What ``scan`` found: a Parquet file's (or folder's) size, rows, row groups and columns, from metadata only."""

    path: str
    bytes: int
    rows: int
    row_groups: int
    files: int
    columns: tuple[ColumnSummary, ...] = ()
    axes: tuple[str, ...] = ()  # a pymetrix file's axes (its time and entity columns)
    partitioning: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def large(self) -> bool:
        """Larger than ``LARGE``: Load data offers columns and a row condition before reading."""
        return self.bytes > LARGE

    def table(self) -> pd.DataFrame:
        """One row per column: type, minimum, maximum, missing values, and whether it is a partition column."""
        return pd.DataFrame([{"column": c.name, "type": c.type, "min": c.min, "max": c.max, "missing": c.nulls,
                              "partition": c.partition} for c in self.columns]).set_index("column")

    def __str__(self) -> str:
        size = f"{self.bytes / 2**20:,.1f} MB"
        head = (f"{self.path}: {self.rows:,} rows, {len(self.columns)} columns, {self.row_groups:,} row groups"
                f"{f' in {self.files:,} files' if self.files > 1 else ''}, {size}")
        lines = [head]
        if self.partitioning:
            lines.append(f"partitioned by {', '.join(self.partitioning)}")
        if self.axes:
            lines.append(f"pymetrix axes: {', '.join(self.axes)}")
        lines.append(self.table().to_string())
        return "\n".join(lines + list(self.notes))


def _files(source: Any) -> list[Any]:
    return list(source.get_fragments())


def scan(path: str | Path) -> ParquetScan:
    """A Parquet file's (or a partitioned folder's) schema, row count, row groups and each column's minimum, maximum
    and missing values, from its metadata (footers and folder names) only: no data is read. Use it to decide which
    columns and rows to load with ``Dataset.read(path, columns=..., where=..., span=...)``."""
    import pyarrow.dataset as pads

    path = str(path)
    source = _source(path)
    schema = source.schema
    every = _files(source)
    partition_names: list[str] = []
    part_values: dict[str, list[Any]] = {}
    for f in every:
        for k, v in pads.get_partition_keys(f.partition_expression).items():
            part_values.setdefault(k, []).append(v)
    partition_names = [n for n in schema.names if n in part_values]
    lows: dict[str, Any] = {}
    highs: dict[str, Any] = {}
    nulls: dict[str, int | None] = {}
    size = rows = groups = 0
    for f in every:
        md = f.metadata
        size += Path(f.path).stat().st_size
        rows += md.num_rows
        groups += md.num_row_groups
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for i, name in enumerate(names):
                stats = rg.column(i).statistics
                if stats is None:
                    nulls[name] = None
                    continue
                if name not in nulls or nulls[name] is not None:
                    nulls[name] = (nulls.get(name) or 0) + stats.null_count if stats.has_null_count else None
                if stats.has_min_max:
                    try:
                        lows[name] = stats.min if name not in lows else min(lows[name], stats.min)
                        highs[name] = stats.max if name not in highs else max(highs[name], stats.max)
                    except TypeError:
                        pass
    meta = pymetrix_meta(schema.metadata or {}, path)
    index_columns = set(_pandas_index_columns(schema))
    summaries = []
    for fld in schema:
        if fld.name in index_columns:
            continue
        if fld.name in part_values:
            values = part_values[fld.name]
            summaries.append(ColumnSummary(fld.name, str(fld.type), min(values), max(values), None, True))
        else:
            summaries.append(ColumnSummary(fld.name, str(fld.type), lows.get(fld.name), highs.get(fld.name), nulls.get(fld.name)))
    notes = ()
    if meta is not None and any(a["kind"] == "time" for a in meta["axes"]):
        notes = ("the time axis is stored as text (periods written like 2001Q1): its minimum and maximum are those of the text",)
    return ParquetScan(path, size, rows, groups, len(every), tuple(summaries),
                       tuple(a["name"] for a in meta["axes"]) if meta else (), tuple(partition_names), notes)


def is_large_parquet(path: str) -> bool:
    """True for a Parquet file or folder larger than ``LARGE`` (read cheaply: file sizes only)."""
    p = Path(str(path or ""))
    try:
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*.parquet")) > LARGE
        return p.suffix.lower() == ".parquet" and p.is_file() and p.stat().st_size > LARGE
    except OSError:
        return False


def _data_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file() and not f.name.startswith((".", "_")))
    return [path]


def fingerprint(path: str | Path) -> str:
    """A fingerprint of a Parquet file's (or folder's) contents from each file's name, size and footer, which records
    the schema and every row group's position, size and statistics: a file rewritten with other data gets another
    one. Only the footers are read, so it is quick for a file of any size. (Two files with equal footers and sizes but
    different values inside a row group whose statistics did not change would share a fingerprint: rare, but this is
    not a hash of every byte.)"""
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} does not exist")
    digest = hashlib.sha256()
    for f in _data_files(path):
        size = f.stat().st_size
        digest.update(f.relative_to(path).as_posix().encode() if f != path else b"")
        digest.update(size.to_bytes(8, "little"))
        with f.open("rb") as handle:
            if size >= 12:
                handle.seek(size - 8)
                tail = handle.read(8)
                length = int.from_bytes(tail[:4], "little")
                if tail[4:] == b"PAR1" and length + 8 <= size:
                    handle.seek(size - 8 - length)
                    digest.update(handle.read(length))
                    continue
            handle.seek(0)
            digest.update(handle.read())  # not a Parquet file (a small side file): all of it
    return "sha256:" + digest.hexdigest()
