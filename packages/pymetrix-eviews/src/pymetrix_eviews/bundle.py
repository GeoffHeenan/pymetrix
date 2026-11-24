"""EViews workfile bundles (format version 1): what ``pymetrix_export.prg`` writes, read into pymetrix.

A bundle is a folder with a JSON manifest (``<workfile>.evbundle``) and the files it names: each page's series as
CSV, long texts, small CSV files for vectors, matrices, value maps and each equation's estimates. The manifest's
schema ships with the plugin (``bundle_schema()``); docs/design/eviews-plugin.md, section 1, is the format's
description and this module follows it.

    quarterly = Dataset.read("C:/work/macro-export/macro.evbundle", page="quarterly")   # through pymetrix.io
    pages = bundle_pages("C:/work/macro-export/macro.evbundle")                          # what the bundle holds
    objects = eviews_objects(quarterly)       # its groups, samples, scalars, strings, matrices and models
    compare_estimates(quarterly)              # each least-squares equation refitted, against EViews's estimates

What does not read is refused at its place (the file, line and column, or the manifest's key), never misread: a
folder without its manifest (an incomplete export), a newer ``format_version``, a CSV without a column the manifest
lists, a value that is not a number, a date that does not start a period of the page's frequency, and frequencies
pandas has no period for (semi-annual, multi-year, intraday, bimonthly, fortnightly, ten-day: read such a page with
``undated=True``). Within version 1 changes are additive: keys this reader does not know are kept in
``meta.extra["eviews"]["unknown"]`` with a note. Files are read as UTF-8; one that is not falls back to Windows-1252,
with a note.
"""

from __future__ import annotations

import csv
import io
import json
import keyword
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pymetrix.errors import DataError, NotFoundError

__all__ = ["FORMAT", "FORMAT_VERSIONS", "PROGRAM_VERSION", "SUFFIX", "EViewsBundleNote", "bundle_model", "bundle_pages",
           "bundle_schema", "compare_estimates", "eviews_objects", "eviews_sample", "export_program", "read_bundle",
           "save_export_program"]

FORMAT = "pymetrix-eviews-bundle"
FORMAT_VERSIONS = (1,)  # the versions this reader reads; a newer one is refused with "update pymetrix-eviews"
PROGRAM_VERSION = "1.0"  # the version of the export program this plugin ships (export/pymetrix_export.prg)
SUFFIX = ".evbundle"

_HERE = Path(__file__).resolve().parent

# EViews's frequency codes as pandas period frequencies. Weekly takes its anchor from the dates (W-SUN: weeks from
# Monday to Sunday); the others pandas has no period for are refused with the reason (_refused_frequency).
_FREQ = {"A": "Y", "Y": "Y", "Q": "Q", "M": "M", "D5": "B", "5": "B", "D7": "D", "7": "D"}
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

_KNOWN = {
    "": {"format", "format_version", "exporter", "workfile", "na", "pages"},
    "exporter": {"program_version", "eviews_version", "exported"},
    "workfile": {"name", "path", "active_page"},
    "page": {"name", "structure", "frequency", "range", "sample", "file", "date_column", "id_columns", "series", "groups",
             "samples", "scalars", "strings", "arrays", "value_maps", "equations", "models", "skipped"},
    "series": {"name", "kind", "display_name", "description", "units", "source", "remarks", "attributes", "value_map"},
    "groups": {"name", "members"},
    "samples": {"name", "text"},
    "scalars": {"name", "value"},
    "strings": {"name", "value"},
    "arrays": {"name", "type", "rows", "cols", "file"},
    "value_maps": {"name", "file"},
    "equations": {"name", "command", "command_file", "method", "sample", "ncoef", "estimates"},
    "models": {"name", "text"},
    "skipped": {"name", "type", "reason"},
}


class EViewsBundleNote(UserWarning):
    """Something about a bundle the user should know, which did not stop it being read: a file that was not UTF-8, a
    bundle from an older export program, keys this reader does not know, names changed to be valid Python names."""


# -- the files -----------------------------------------------------------------------------------------------------


def _manifest_path(source: str | Path) -> Path:
    path = Path(source)
    if path.is_dir():
        found = sorted(path.glob(f"*{SUFFIX}"))
        if not found:
            raise DataError(f"{path} has no manifest (*{SUFFIX}): the export did not finish (the program writes the "
                            "manifest last), or this is not a bundle")
        if len(found) > 1:
            raise DataError(f"{path} has several manifests ({', '.join(p.name for p in found)}): open one of them")
        return found[0]
    if not path.is_file():
        raise DataError(f"no manifest {path}: a folder without its {SUFFIX} file is an incomplete export, which is "
                        "refused (run the export program again)")
    return path


def _decode(raw: bytes, where: str, notes: list[str]) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        notes.append(f"{where} is not UTF-8 (byte {exc.start + 1}): read as Windows-1252")
        return raw.decode("cp1252", errors="replace")


@dataclass
class _Bundle:
    manifest_path: Path
    manifest: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    unknown: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    def file(self, relative: Any, where: str) -> Path:
        """A file the manifest names, inside the bundle's folder."""
        text = str(relative).replace("\\", "/")
        parts = text.split("/")
        if text.startswith("/") or re.match(r"^[A-Za-z]:", text) or ".." in parts:
            raise DataError(f"{self.manifest_path.name}: {where} = {relative!r}: files must be inside the bundle's folder")
        path = self.root.joinpath(*parts)
        if not path.is_file():
            raise DataError(f"{self.manifest_path.name}: {where} names {text}, which is not in the bundle ({self.root})")
        return path

    def text(self, relative: Any, where: str) -> tuple[str, str]:
        path = self.file(relative, where)
        name = str(relative).replace("\\", "/")
        return _decode(path.read_bytes(), name, self.notes), name

    @property
    def na(self) -> str:
        return str(self.manifest.get("na", "NA"))

    def page(self, name: str | None) -> tuple[int, dict[str, Any]]:
        pages = self.manifest["pages"]
        names = [str(p["name"]) for p in pages]
        if name is None:
            if len(pages) == 1:
                return 0, pages[0]
            raise DataError(f"{self.manifest_path.name} has {len(pages)} pages ({', '.join(names)}): give page=")
        for i, p in enumerate(pages):
            if str(p["name"]).lower() == str(name).lower():
                return i, p
        raise NotFoundError(f"{self.manifest_path.name} has no page {name!r}; its pages: {', '.join(names)}")


def _open(source: str | Path) -> _Bundle:
    path = _manifest_path(source)
    notes: list[str] = []
    text = _decode(path.read_bytes(), path.name, notes)
    manifest = _load_json(text, path.name)
    bundle = _Bundle(path, manifest, notes)
    _check_manifest(bundle)
    return bundle


def _load_json(text: str, where: str) -> Any:
    """Strict JSON: no NaN or Infinity, no key twice in an object; an error says its line and column."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise DataError(f"{where}: the key {key!r} is given twice in one object")
            out[key] = value
        return out

    def constant(name: str) -> Any:
        raise DataError(f"{where}: {name} is not JSON (a missing number is null)")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as exc:
        raise DataError(f"{where}, line {exc.lineno}, column {exc.colno}: not valid JSON ({exc.msg})") from None


# -- the manifest's checks -----------------------------------------------------------------------------------------


def _need(obj: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, bundle: _Bundle) -> Any:
    if key not in obj:
        raise DataError(f"{bundle.manifest_path.name}: {where} has no {key!r}")
    value = obj[key]
    if not isinstance(value, kind) or (isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,))):
        raise DataError(f"{bundle.manifest_path.name}: {where}.{key} should be {_kind(kind)}, not {type(value).__name__}")
    return value


def _kind(kind: type | tuple[type, ...]) -> str:
    names = {str: "text", int: "a whole number", float: "a number", list: "a list", dict: "an object", type(None): "null"}
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if float in kinds:  # a whole number is a number
        kinds = tuple(k for k in kinds if k is not int)
    return " or ".join(names.get(k, k.__name__) for k in kinds)


def _unknown(obj: Mapping[str, Any], known: str, where: str, bundle: _Bundle) -> None:
    for key in obj:
        if key not in _KNOWN[known]:
            bundle.unknown[f"{where}.{key}" if where else key] = obj[key]


def _check_manifest(b: _Bundle) -> None:
    m = b.manifest
    name = b.manifest_path.name
    if not isinstance(m, dict):
        raise DataError(f"{name}: the manifest should be a JSON object")
    if m.get("format") != FORMAT:
        raise DataError(f"{name}: not a pymetrix EViews bundle (its \"format\" is {m.get('format')!r}, not {FORMAT!r})")
    version = _need(m, "format_version", int, "the manifest", b)
    if version > max(FORMAT_VERSIONS):
        raise DataError(f"{name}: format_version {version} is newer than this pymetrix-eviews reads "
                        f"({', '.join(map(str, FORMAT_VERSIONS))}): update pymetrix-eviews (pip install -U pymetrix-eviews)")
    if version not in FORMAT_VERSIONS:
        raise DataError(f"{name}: format_version {version} is not one this reader knows ({', '.join(map(str, FORMAT_VERSIONS))})")
    _unknown(m, "", "", b)
    exporter = _need(m, "exporter", dict, "the manifest", b)
    program = str(_need(exporter, "program_version", str, "exporter", b))
    _unknown(exporter, "exporter", "exporter", b)
    if _version_tuple(program) < _version_tuple(PROGRAM_VERSION):
        b.notes.append(f"the bundle was written by export program {program}; this plugin ships {PROGRAM_VERSION} "
                       "(Get the export program): export again with it for what it adds")
    workfile = _need(m, "workfile", dict, "the manifest", b)
    _need(workfile, "name", str, "workfile", b)
    _unknown(workfile, "workfile", "workfile", b)
    if "na" in m:
        _need(m, "na", str, "the manifest", b)
    pages = _need(m, "pages", list, "the manifest", b)
    if not pages:
        raise DataError(f"{name}: the bundle has no pages")
    seen: set[str] = set()
    for i, page in enumerate(pages):
        where = f"pages[{i}]"
        if not isinstance(page, dict):
            raise DataError(f"{name}: {where} should be an object")
        pname = _need(page, "name", str, where, b)
        if pname.lower() in seen:
            raise DataError(f"{name}: {where}: a second page called {pname!r}")
        seen.add(pname.lower())
        structure = _need(page, "structure", str, where, b)
        if structure not in ("dated", "undated", "panel"):
            raise DataError(f"{name}: {where}.structure is {structure!r}: dated, undated or panel")
        _need(page, "file", str, where, b)
        if structure in ("dated", "panel"):
            _need(page, "frequency", str, where, b)
            _need(page, "date_column", str, where, b)
        if structure == "panel" and not _need(page, "id_columns", list, where, b):
            raise DataError(f"{name}: {where}.id_columns: a panel page names its cross-section id column")
        _unknown(page, "page", where, b)
        series = _need(page, "series", list, where, b)
        for j, s in enumerate(series):
            sw = f"{where}.series[{j}]"
            if not isinstance(s, dict):
                raise DataError(f"{name}: {sw} should be an object")
            _need(s, "name", str, sw, b)
            kind = _need(s, "kind", str, sw, b)
            if kind not in ("numeric", "alpha"):
                raise DataError(f"{name}: {sw}.kind is {kind!r}: numeric or alpha")
            for key in ("display_name", "description", "units", "source", "remarks"):
                if key in s:
                    _need(s, key, str, sw, b)
            if "attributes" in s:
                attrs = _need(s, "attributes", dict, sw, b)
                bad = [k for k, v in attrs.items() if not isinstance(v, str)]
                if bad:
                    raise DataError(f"{name}: {sw}.attributes.{bad[0]} should be text")
            if s.get("value_map") is not None:
                _need(s, "value_map", str, sw, b)
            _unknown(s, "series", sw, b)
        for key, required in (("groups", {"name": str, "members": list}), ("samples", {"name": str, "text": str}),
                              ("scalars", {"name": str, "value": (int, float, type(None))}), ("strings", {"name": str, "value": str}),
                              ("arrays", {"name": str, "type": str, "rows": int, "cols": int, "file": str}),
                              ("value_maps", {"name": str, "file": str}), ("equations", {"name": str, "method": str}),
                              ("models", {"name": str, "text": str}), ("skipped", {"name": str, "type": str})):
            for j, item in enumerate(_need(page, key, list, where, b) if key in page else []):
                iw = f"{where}.{key}[{j}]"
                if not isinstance(item, dict):
                    raise DataError(f"{name}: {iw} should be an object")
                for k, kind in required.items():
                    _need(item, k, kind, iw, b)
                if key == "equations" and "command" not in item and "command_file" not in item:
                    raise DataError(f"{name}: {iw} has no 'command' (or 'command_file')")
                _unknown(item, key, iw, b)
    if b.unknown:
        b.notes.append(f"keys this reader does not know, kept in meta['eviews']['unknown']: {', '.join(sorted(b.unknown))}")


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)) or (0,)


# -- frequencies and names -----------------------------------------------------------------------------------------


def _refused_frequency(code: str) -> str | None:
    """Why pymetrix cannot give a page of this EViews frequency a time axis (None when it can)."""
    c = code.strip().upper()
    if c in _FREQ or c == "W":
        return None
    if c in ("S", "SA"):
        return "semi-annual: pandas has no half-year period (a known gap)"
    if re.fullmatch(r"\d+A|\d+Y", c):
        return "multi-year: pandas has no period of several years"
    if re.match(r"^\d*(H|MIN|SEC)", c) or c.endswith(("H", "MIN", "SEC")):
        return "intraday: pymetrix reads periods of a day or longer"
    return {"BM": "bimonthly: pandas has no two-month period", "F": "fortnightly: pandas has no two-week period",
            "T": "ten-day: pandas has no ten-day period",
            "D": "daily with a custom week: give the page five-day (D5) or seven-day (D7) dates in EViews"}.get(
        c, f"{code!r} is not a frequency this reader knows")


def _py_name(text: str) -> str:
    """A name pymetrix can use as a series or object: lower case, letters, digits and _, not a Python keyword."""
    name = re.sub(r"\W+", "_", str(text).strip().lower()).strip("_") or "x"
    if name[0].isdigit():
        name = f"x_{name}"
    if keyword.iskeyword(name) or keyword.issoftkeyword(name):
        name = f"{name}_v"
    return name


def _axis_name(column: str) -> str:
    """An id or date column as an axis name: pmx_crossid -> crossid (the export program's own columns)."""
    name = _py_name(column)
    return name[4:] if name.startswith("pmx_") and len(name) > 4 else name


# -- reading a page ------------------------------------------------------------------------------------------------


def read_bundle(source: str | Path, page: str | None = None, undated: bool = False) -> Any:
    """One page of an EViews bundle as a Dataset (``Dataset.read(path, page=...)`` comes here for ``.evbundle``).

    ``page`` is the page's name, as in EViews (any case); it may be left out when the bundle has one page.
    ``undated=True`` reads a dated page as undated (rows numbered from 1, the dates kept as the text series
    ``date``): for frequencies pandas has no period for, such as semi-annual.

    Each series keeps its EViews metadata: the label (the display name, else the description), description, units
    and source; the remarks, attributes, value map and EViews's own name, when changed, in ``Variable.extra``.
    ``meta.extra["eviews"]`` keeps the rest of the page: its frequency, range and sample, the groups, samples,
    scalars, strings, matrices, equations (with EViews's estimates) and models, the objects the program skipped, and
    the notes on the reading. ``eviews_objects(data)`` turns them into pymetrix objects."""
    import pandas as pd

    from pymetrix.data import Dataset
    from pymetrix.metadata import DatasetMeta, Variable

    b = _open(source)
    index, p = b.page(page)
    where = f"pages[{index}]"
    notes = b.notes
    structure = p["structure"]
    if undated and structure == "panel":
        raise DataError(f"{b.manifest_path.name}: page {p['name']!r} is a panel: it cannot be read as undated")
    text, file = b.text(p["file"], f"{where}.file")
    header, rows, lines = _csv(text, file)
    column = {h.strip().lower(): k for k, h in enumerate(header)}
    if len(column) < len(header):
        raise DataError(f"{file}, line 1: a column name is given twice (names are read in any case)")

    def col(name: str, what: str) -> int:
        k = column.get(str(name).strip().lower())
        if k is None:
            raise DataError(f"{file}: no column {name!r} ({what}); the file has {', '.join(header) or 'no columns'}")
        return k

    freq_code = str(p.get("frequency") or "")
    date_column = p.get("date_column") or None
    id_columns = list(p.get("id_columns") or [])
    used = set()
    data: dict[str, Any] = {}
    axes: list[str] = []
    frame_freq = None
    pd_freq = None
    if structure in ("dated", "panel") and not undated:
        why = _refused_frequency(freq_code)
        if why is not None:
            raise DataError(f"{b.manifest_path.name}: {where} ({p['name']}) has frequency {freq_code!r}, {why}; read the "
                            "page as undated (Dataset.read(path, page=..., undated=True))")
        k = col(date_column, f"{where}.date_column")
        used.add(k)
        dates = [_iso_date(r[k], file, lines[i], header[k]) for i, r in enumerate(rows)]
        pd_freq = _pandas_freq(freq_code, dates, file)
        periods = [pd.Period(d, freq=pd_freq) for d in dates]
        for d, per, line in zip(dates, periods, lines):
            if per.start_time.date() != d:
                raise DataError(f"{file}, line {line}, column {header[k]}: {d.isoformat()} is not the first day of a "
                                f"{freq_code} period (it falls in {per}): is the page's frequency right?")
        data["time"] = pd.PeriodIndex(periods, freq=pd_freq)
        axes.append("time")
        frame_freq = pd_freq
    elif structure in ("dated", "panel") and undated:
        k = col(date_column, f"{where}.date_column")
        used.add(k)
        data["obs"] = list(range(1, len(rows) + 1))
        axes.append("obs")
        dates_text = pd.array([r[k] or None for r in rows], dtype="string")
        notes.append(f"page {p['name']} ({freq_code}) read as undated: rows numbered from 1, its dates in the text series date")
    if structure == "panel" and not undated:
        for name in id_columns:
            k = col(name, f"{where}.id_columns")
            used.add(k)
            values = [r[k] for r in rows]
            data[_axis_name(name)] = [int(v) for v in values] if all(re.fullmatch(r"[+-]?\d+", v) for v in values) else values
            axes.append(_axis_name(name))
    if structure == "undated":
        names = id_columns or (["obs"] if "obs" in column else [])
        if names:
            for name in names:
                k = col(name, f"{where}.id_columns")
                used.add(k)
                values = [r[k] for r in rows]
                data[_axis_name(name)] = [int(v) for v in values] if all(re.fullmatch(r"[+-]?\d+", v) for v in values) else values
                axes.append(_axis_name(name))
        else:
            data["obs"] = list(range(1, len(rows) + 1))
            axes.append("obs")
            notes.append(f"page {p['name']} has no identifier column: rows numbered from 1 (obs)")

    value_maps = _value_maps(b, p, where)
    variables: dict[str, Variable] = {}
    renamed: dict[str, str] = {}
    retrieved = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005 - local time, as pymetrix's readers stamp it
    for j, s in enumerate(p["series"]):
        sw = f"{where}.series[{j}]"
        k = col(s["name"], f"the series {s['name']} in {sw}")
        used.add(k)
        name = _py_name(s["name"])
        while name in data or name in variables:
            name = f"{name}_v"
        if name != s["name"]:
            renamed[s["name"]] = name
        if s["kind"] == "numeric":
            data[name] = [_number(r[k], b.na, file, lines[i], header[k]) for i, r in enumerate(rows)]
        else:
            data[name] = pd.array([r[k] if r[k] != "" else None for r in rows], dtype="string")
        variables[name] = _variable(name, s, value_maps, b, sw, retrieved)
    if undated and structure in ("dated", "panel"):
        name = "date" if "date" not in data else "date_v"
        data[name] = dates_text
        variables[name] = Variable(name, label="the EViews dates", description=f"the page's {freq_code} dates (first days)")
    extra_cols = [header[k] for k in range(len(header)) if k not in used]
    if extra_cols:
        notes.append(f"{file}: columns the manifest does not list, left out: {', '.join(extra_cols)}")
    if renamed:
        notes.append("names changed to be valid Python names: " + ", ".join(f"{a} -> {c}" for a, c in renamed.items()))

    frame = pd.DataFrame({k: v for k, v in data.items() if k not in axes}, index=pd.RangeIndex(len(rows)))
    for a in axes:
        frame[a] = data[a]
    if structure == "dated" and not undated:
        frame = frame.set_index("time")
        if not frame.index.is_unique:
            dup = frame.index[frame.index.duplicated()][0]
            raise DataError(f"{file}: the period {dup} appears twice")
        ds = Dataset.from_frame(frame.sort_index(), variables=variables)
    else:
        time = "time" if "time" in axes else None
        entities = [a for a in axes if a != "time"]
        keys = frame[axes]
        if keys.duplicated().any():
            row = int(keys.duplicated().to_numpy().nonzero()[0][0])
            raise DataError(f"{file}, line {lines[row]}: the row's {' and '.join(axes)} appear on an earlier line")
        ds = Dataset.from_frame(frame, time=time, entity=entities, freq=frame_freq, variables=variables)
    meta = DatasetMeta(title=f"EViews {b.manifest['workfile']['name']}: page {p['name']}",
                       description=f"read from the bundle {b.manifest_path.name} (pymetrix-eviews)",
                       extra={"eviews": _page_meta(b, p, where, renamed)})
    out = Dataset(ds.frame, ds.axes, ds.variables, ds.flags, meta)
    for note in notes:
        warnings.warn(note, EViewsBundleNote, stacklevel=2)
    return out


def _csv(text: str, file: str) -> tuple[list[str], list[list[str]], list[int]]:
    """The header, the rows and each row's line in the file. A row with the wrong number of fields is refused."""
    reader = csv.reader(io.StringIO(text), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise DataError(f"{file} is empty: it should start with a row of column names") from None
    except csv.Error as exc:
        raise DataError(f"{file}, line 1: {exc}") from None
    rows, lines = [], []
    try:
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(header):
                raise DataError(f"{file}, line {reader.line_num}: {len(row)} fields; the header has {len(header)}")
            rows.append([v.strip() for v in row])
            lines.append(reader.line_num)
    except csv.Error as exc:
        raise DataError(f"{file}, line {reader.line_num}: {exc}") from None
    return [h.strip() for h in header], rows, lines


def _number(text: str, na: str, file: str, line: int, column: str) -> float:
    if text == "" or text == na:
        return math.nan
    if not _NUMBER.match(text):
        raise DataError(f"{file}, line {line}, column {column}: {text!r} is not a number (a missing value is {na!r} or empty)")
    return float(text)


def _iso_date(text: str, file: str, line: int, column: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise DataError(f"{file}, line {line}, column {column}: {text!r} is not a date written YYYY-MM-DD") from None


def _pandas_freq(code: str, dates: list[date], file: str) -> str:
    c = code.strip().upper()
    if c != "W":
        return _FREQ[c]
    days = {d.weekday() for d in dates}
    if len(days) > 1:
        raise DataError(f"{file}: weekly dates start on different days of the week ({', '.join(_WEEKDAYS[d] for d in sorted(days))})")
    start = days.pop() if days else 0
    return f"W-{_WEEKDAYS[(start - 1) % 7]}"  # a week that starts on Monday ends on Sunday: W-SUN


def _variable(name: str, s: Mapping[str, Any], maps: Mapping[str, dict[str, str]], b: _Bundle, where: str, retrieved: str) -> Any:
    from pymetrix.metadata import Variable

    display, description = s.get("display_name", ""), s.get("description", "")
    extra: dict[str, Any] = {}
    if name != s["name"]:
        extra["eviews_name"] = s["name"]
    if s["kind"] == "alpha":
        extra["eviews_kind"] = "alpha"
    if display and description:
        extra["display_name"] = display
    for key in ("remarks",):
        if s.get(key):
            extra[key] = s[key]
    if s.get("attributes"):
        extra["attributes"] = dict(s["attributes"])
    vm = s.get("value_map")
    if vm:
        if vm.lower() not in maps:
            raise DataError(f"{b.manifest_path.name}: {where}.value_map names {vm!r}, which the page's value_maps do not list")
        extra["value_map"] = {"name": vm.lower(), "labels": dict(maps[vm.lower()])}
    return Variable(name, label=display or description, description=description, units=s.get("units", ""),
                    source=s.get("source", ""), source_key=s["name"], retrieved=retrieved, extra=extra)


def _value_maps(b: _Bundle, p: Mapping[str, Any], where: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for j, vm in enumerate(p.get("value_maps", [])):
        text, file = b.text(vm["file"], f"{where}.value_maps[{j}].file")
        header, rows, lines = _csv(text, file)
        if [h.lower() for h in header] != ["value", "label"]:
            raise DataError(f"{file}, line 1: a value map's columns are value,label (found {','.join(header)})")
        labels: dict[str, str] = {}
        for r, line in zip(rows, lines):
            if r[0] in labels:
                raise DataError(f"{file}, line {line}: the value {r[0]!r} is mapped twice")
            labels[r[0]] = r[1]
        out[vm["name"].lower()] = labels
    return out


def _page_meta(b: _Bundle, p: Mapping[str, Any], where: str, renamed: Mapping[str, str]) -> dict[str, Any]:
    """What the page holds besides its series, as JSON values (kept in Parquet files and projects)."""
    m = b.manifest
    out: dict[str, Any] = {
        "format_version": m["format_version"], "exporter": dict(m["exporter"]), "workfile": dict(m["workfile"]),
        "bundle": str(b.manifest_path), "page": p["name"], "structure": p["structure"],
        "frequency": p.get("frequency"), "range": list(p.get("range", [])), "sample": p.get("sample", ""),
        "groups": {g["name"].lower(): [str(x).lower() for x in g["members"]] for g in p.get("groups", [])},
        "samples": {s["name"].lower(): s["text"] for s in p.get("samples", [])},
        "scalars": {s["name"].lower(): s["value"] for s in p.get("scalars", [])},
        "strings": {s["name"].lower(): s["value"] for s in p.get("strings", [])},
        "arrays": {}, "equations": [], "models": {},
        "skipped": [dict(s) for s in p.get("skipped", [])],
        "renamed": dict(renamed), "notes": list(b.notes), "unknown": dict(b.unknown),
    }
    for j, a in enumerate(p.get("arrays", [])):
        aw = f"{where}.arrays[{j}]"
        text, file = b.text(a["file"], f"{aw}.file")
        values = _array(text, file, b.na, a["rows"], a["cols"])
        out["arrays"][a["name"].lower()] = {"type": a["type"], "rows": a["rows"], "cols": a["cols"], "values": values}
    for j, e in enumerate(p.get("equations", [])):
        ew = f"{where}.equations[{j}]"
        command = e.get("command")
        if command is None:
            command, _ = b.text(e["command_file"], f"{ew}.command_file")
            command = " ".join(command.split())
        eq: dict[str, Any] = {"name": e["name"].lower(), "command": command, "method": e["method"], "sample": e.get("sample", ""),
                              "ncoef": e.get("ncoef")}
        if e.get("estimates"):
            text, file = b.text(e["estimates"], f"{ew}.estimates")
            eq.update(_estimates(text, file, b.na, e.get("ncoef")))
        out["equations"].append(eq)
    for j, mdl in enumerate(p.get("models", [])):
        text, _ = b.text(mdl["text"], f"{where}.models[{j}].text")
        out["models"][mdl["name"].lower()] = text.replace("\r\n", "\n")
    return out


def _array(text: str, file: str, na: str, rows: int, cols: int) -> list[list[float | None]]:
    """A matrix's CSV: one line per row, no header; missing values as the bundle's NA, read as null."""
    reader = csv.reader(io.StringIO(text))
    out: list[list[float | None]] = []
    for row in reader:
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != cols:
            raise DataError(f"{file}, line {reader.line_num}: {len(row)} values; the manifest says {cols} columns")
        values = []
        for k, v in enumerate(row):
            x = _number(v.strip(), na, file, reader.line_num, f"{k + 1}")
            values.append(None if math.isnan(x) else x)
        out.append(values)
    if len(out) != rows:
        raise DataError(f"{file}: {len(out)} rows; the manifest says {rows}")
    return out


def _estimates(text: str, file: str, na: str, ncoef: Any) -> dict[str, Any]:
    header, rows, lines = _csv(text, file)
    if [h.lower() for h in header] != ["coefficient", "estimate", "std_error"]:
        raise DataError(f"{file}, line 1: an equation's estimates have the columns coefficient,estimate,std_error "
                        f"(found {','.join(header)})")
    if ncoef is not None and len(rows) != ncoef:
        raise DataError(f"{file}: {len(rows)} coefficients; the manifest's ncoef is {ncoef}")

    def num(v: str, line: int, column: str) -> float | None:
        x = _number(v, na, file, line, column)
        return None if math.isnan(x) else x

    return {"coefficients": [r[0] for r in rows], "estimates": [num(r[1], ln, "estimate") for r, ln in zip(rows, lines)],
            "std_errors": [num(r[2], ln, "std_error") for r, ln in zip(rows, lines)]}


# -- what a bundle holds -------------------------------------------------------------------------------------------


def bundle_pages(source: str | Path) -> list[dict[str, Any]]:
    """The pages of a bundle, without reading their data: name, structure, frequency, how many series, and why the
    page cannot be read with a time axis (``refused``; None when it can)."""
    b = _open(source)
    out = []
    for p in b.manifest["pages"]:
        refused = _refused_frequency(str(p.get("frequency") or "")) if p["structure"] != "undated" else None
        out.append({"name": p["name"], "structure": p["structure"], "frequency": p.get("frequency"),
                    "series": len(p["series"]), "refused": refused})
    return out


def bundle_schema() -> dict[str, Any]:
    """The JSON Schema of the manifest (format version 1)."""
    return json.loads((_HERE / "bundle-1.schema.json").read_text(encoding="utf-8"))


def _eviews(data: Any) -> dict[str, Any]:
    ev = (getattr(getattr(data, "meta", None), "extra", None) or {}).get("eviews")
    if not isinstance(ev, Mapping):
        raise DataError("this dataset was not read from an EViews bundle (it has no meta.extra['eviews'])")
    return dict(ev)


def eviews_sample(text: str, frequency: str | None = None) -> Any:
    """An EViews sample's text as a pymetrix ``Sample``, or None when it does not translate: ``@all``, and one pair
    of ends (``1980q1 2007q4``, ``@first 2000q4``, ``1990:1 @last`` with the page's ``frequency``, ``Q`` or ``M``).
    Conditions (``if ...``) and several pairs are not translated."""
    from pymetrix import Sample

    tokens = str(text).strip().lower().split()
    if not tokens or tokens == ["@all"]:
        return Sample()
    if len(tokens) != 2:
        return None
    ends = []
    for token in tokens:
        if token in ("@first", "@last"):
            ends.append(None)
            continue
        period = _period(token, frequency)
        if period is None:
            return None
        ends.append(period)
    return Sample(start=ends[0], end=ends[1])


def _period(token: str, frequency: str | None) -> str | None:
    f = (frequency or "").upper()
    if m := re.fullmatch(r"(\d{4})q([1-4])", token):
        return f"{m.group(1)}Q{m.group(2)}"
    if m := re.fullmatch(r"(\d{4})m(\d{1,2})", token):
        return f"{m.group(1)}-{int(m.group(2)):02d}" if 1 <= int(m.group(2)) <= 12 else None
    if m := re.fullmatch(r"(\d{4}):(\d{1,2})", token):
        if f == "Q" and 1 <= int(m.group(2)) <= 4:
            return f"{m.group(1)}Q{m.group(2)}"
        if f == "M" and 1 <= int(m.group(2)) <= 12:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
        return None
    if re.fullmatch(r"\d{4}", token) and f in ("A", "Y"):
        return token
    return None


def eviews_objects(data: Any) -> dict[str, Any]:
    """The page's other EViews objects, as pymetrix's: groups as ``Selection`` (``data.series(...)``; a group with a
    member that is not a series of the page is left out), samples as ``Sample`` (where ``eviews_sample`` translates
    them), scalars as numbers, strings as text, vectors and matrices as ``Matrix``, and models as templates
    (``bundle_model``; a model the core reader refuses is left out). What is left out is listed under ``"_left_out"``
    with the reason."""
    import numpy as np

    from pymetrix import Matrix

    ev = _eviews(data)
    out: dict[str, Any] = {}
    left: dict[str, str] = {}
    columns = set(data.frame.columns)
    for name, members in ev.get("groups", {}).items():
        missing = [m for m in members if m not in columns]
        if missing:
            left[name] = f"group: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not series of the page"
            continue
        out[name] = data.series(*members)
    for name, text in ev.get("samples", {}).items():
        sample = eviews_sample(text, ev.get("frequency"))
        if sample is None:
            left[name] = f"sample {text!r}: only @all and one pair of ends translate"
        else:
            out[name] = sample
    for name, value in ev.get("scalars", {}).items():
        out[name] = math.nan if value is None else float(value)
    out.update(ev.get("strings", {}))
    for name, a in ev.get("arrays", {}).items():
        values = np.array([[math.nan if v is None else v for v in row] for row in a["values"]], dtype=float).reshape(a["rows"], a["cols"])
        out[name] = Matrix(values, title=name)
    for name in ev.get("models", {}):
        try:
            out[name] = bundle_model(data, name)
        except Exception as exc:  # noqa: BLE001 - the reason is given, the rest still returned
            left[name] = f"model: {exc}"
    if left:
        out["_left_out"] = left
    return out


def _model_program(ev: Mapping[str, Any], name: str) -> str:
    """The model as a program the core reader takes: its equations (``equation eq.ls ...`` on their samples), then
    ``model m``, ``m.merge eq`` for each linked equation (``:eq`` in the text) and ``m.append`` for the rest."""
    models = ev.get("models", {})
    if name.lower() not in models:
        raise NotFoundError(f"the page has no model {name!r}; its models: {', '.join(models) or 'none'}")
    equations = {e["name"]: e for e in ev.get("equations", [])}
    lines = [ln.strip() for ln in models[name.lower()].splitlines()]
    links = [ln[1:].strip().lower() for ln in lines if ln.startswith(":")]
    program = []
    for eq in links:
        if eq not in equations:
            raise DataError(f"model {name}: the linked equation {eq} is not in the bundle's equations")
        e = equations[eq]
        if e.get("sample"):
            program.append(f"smpl {e['sample']}")
        program.append(f"equation {eq}.{e['command']}")
    program.append(f"model {name.lower()}")
    for ln in lines:
        if not ln or ln.startswith("'"):
            continue
        program.append(f"{name.lower()}.merge {ln[1:].strip().lower()}" if ln.startswith(":") else f"{name.lower()}.append {ln}")
    return "\n".join(program) + "\n"


def bundle_model(data: Any, name: str) -> Any:
    """A model of the page, read by ``read_eviews``: its text, with the equations it links fitted to ``data`` by the
    core reader (least squares, on each equation's sample) and the page's scalars that the text names as its
    parameters. EViews's own estimates are not used; ``compare_estimates`` sets the two side by side."""
    from pymetrix import read_eviews

    ev = _eviews(data)
    program = _model_program(ev, name)
    scalars = {k: v for k, v in ev.get("scalars", {}).items()
               if v is not None and re.search(rf"(?<![\w@]){re.escape(k)}(?!\w)", program, flags=re.IGNORECASE)}
    return read_eviews(program, params=scalars or None, data=data, name=name.lower())


def _top_level_split(text: str) -> list[str]:
    out, depth, word = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if word:
                out.append(word)
            word = ""
        else:
            word += ch
    if word:
        out.append(word)
    return out


def compare_estimates(data: Any, tolerance: float = 1e-6) -> Any:
    """Each least-squares equation of the page refitted in pymetrix, against EViews's own estimates: a ``Table`` with
    a row per coefficient, saying *matches* (within ``tolerance``, relative, for the estimate and its standard error)
    or how much they differ. Equations by other methods, or with options (``ls(cov=white)``), are listed as not
    compared, with the reason. The refit is the core reader's (``read_eviews`` with the data), on each equation's
    sample; a list specification's ``c`` is pymetrix's ``const``."""
    import numpy as np
    import pandas as pd

    from pymetrix import read_eviews
    from pymetrix.extension import parse_eviews_expression, render_eviews_expression
    from pymetrix.output import Table

    def term(text: str) -> str:  # a regressor of a list specification as pymetrix writes it: cons(-1) -> cons[-1]
        return "const" if text.lower() == "c" else render_eviews_expression(parse_eviews_expression(text, 0, {"c"}))

    ev = _eviews(data)
    rows = []
    for e in ev.get("equations", []):
        command = str(e["command"]).strip()
        method = command.split()[0].lower() if command else ""
        reason = None
        if str(e.get("method", "")).lower() not in ("ls", "least squares") or not method.startswith("ls"):
            reason = f"method {e.get('method')}: only least squares (ls) is compared in this version"
        elif method != "ls":
            reason = f"options {method[2:]}: the core reader ignores them, so its fit would not be the same estimator"
        elif "estimates" not in e:
            reason = "the bundle has no estimates for it"
        if reason:
            rows.append({"equation": e["name"], "coefficient": "", "EViews": np.nan, "pymetrix": np.nan, "EViews s.e.": np.nan,
                         "pymetrix s.e.": np.nan, "result": f"not compared: {reason}"})
            continue
        program = (f"smpl {e['sample']}\n" if e.get("sample") else "") + f"equation {e['name']}.{command}\nmodel _m\n_m.merge {e['name']}\n"
        try:
            fit = read_eviews(program, data=data).model.equations[0]
            names = list(fit.params.names)
            values = np.asarray(fit.params.values, dtype=float)
            ses = np.sqrt(np.diag(np.asarray(fit.params.cov, dtype=float)))
            spec = command[2:].strip()
            if "=" in spec:  # an expression: c(i) is pymetrix's c_i
                order = [f"c_{i + 1}" for i in range(len(e["estimates"]))]
            else:  # a list: the dependent variable, then the regressors in EViews's order of c(1), c(2) ...
                order = [term(t) for t in _top_level_split(spec)[1:]]
        except (DataError, ValueError) as exc:
            rows.append({"equation": e["name"], "coefficient": "", "EViews": np.nan, "pymetrix": np.nan, "EViews s.e.": np.nan,
                         "pymetrix s.e.": np.nan, "result": f"not compared: pymetrix could not fit it ({exc})"})
            continue
        for i, (coef, b, s) in enumerate(zip(e["coefficients"], e["estimates"], e["std_errors"])):
            name = order[i] if i < len(order) else None
            if name not in names:
                rows.append({"equation": e["name"], "coefficient": coef, "EViews": b, "pymetrix": np.nan, "EViews s.e.": s,
                             "pymetrix s.e.": np.nan, "result": f"not compared: no term {name} in pymetrix's fit"})
                continue
            k = names.index(name)
            ours, our_se = float(values[k]), float(ses[k])
            gaps = [abs(ours - b) / max(abs(b), 1e-300) if b is not None else math.inf,
                    abs(our_se - s) / max(abs(s), 1e-300) if s is not None else math.inf]
            result = "matches" if max(gaps) <= tolerance else f"differs: relative {max(gaps):.3g}"
            rows.append({"equation": e["name"], "coefficient": f"{coef} ({name})", "EViews": b, "pymetrix": ours, "EViews s.e.": s,
                         "pymetrix s.e.": our_se, "result": result})
    frame = pd.DataFrame(rows, columns=["equation", "coefficient", "EViews", "pymetrix", "EViews s.e.", "pymetrix s.e.", "result"])
    frame.index = pd.RangeIndex(1, len(frame) + 1)
    return Table.from_frame(frame, title="EViews's estimates and pymetrix's refit",
                            notes=(f"matches: within {tolerance:g}, relative, for the estimate and its standard error",))


# -- the export program --------------------------------------------------------------------------------------------


def export_program() -> str:
    """The text of ``pymetrix_export.prg``, the program that writes a bundle from EViews. EXPERIMENTAL: written from
    EViews's documentation and never run in EViews (the project has no licence); every *(verify)* comment in it is
    a point to check. Saving the workfile from EViews as CSV or Excel is the supported way in."""
    return (_HERE / "export" / "pymetrix_export.prg").read_text(encoding="utf-8")


def save_export_program(path: str | Path) -> str:
    """Save a copy of the export program (EXPERIMENTAL, untested in EViews) at ``path`` (a folder: as
    pymetrix_export.prg in it), with Windows line ends, as EViews expects; returns where it was saved."""
    out = Path(path)
    if out.is_dir():
        out = out / "pymetrix_export.prg"
    out.write_bytes(export_program().replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    return str(out)
