"""Single results saved to a file, to send or keep outside a project.

    save_result(fit, "consumption.pymxr")       # every view of the result, its summary and provenance
    saved = load_result("consumption.pymxr")     # a SavedResult: the same tabs, summary() and to_frame(view)

A ``.pymxr`` file is a zip archive of plain files::

    manifest.json        format version, pymetrix version (and commit, from a checkout of its source:
                         ``pymetrix.build``), saved time, the result's kind and name, its views
    summary.txt          the result as text, as ``print(result.summary())`` shows it
    views/<n>.json       each view: a table (columns, rows, row labels, notes) or a figure (its panels and series)
    provenance.json      how the result was made (data, sample, options), when it records that
    record.json          for estimates: the result's own record (``results.to_record``): the tidy coefficients,
                         statistics with their definitions, and observations, so a saved result gives the same
                         ``coefficients()``, ``statistics()`` and ``observations()`` as the live one

A saved result is a record of the result as it was, not the live object: it shows the same tables
and figures (figures are redrawn from their series, in any theme) but cannot be forecast from or
estimated again; the project that made it can. Nothing in the file is code, so opening one runs
nothing: it is safe to open a file from someone else. Missing values are written as JSON ``NaN``.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .errors import PyMetrixError, SpecificationError

__all__ = ["SUFFIX", "SavedResult", "load_result", "result_bytes", "result_from_bytes", "save_result"]

FORMAT_VERSION = 1
SUFFIX = ".pymxr"


@dataclass(frozen=True, eq=False)
class SavedResult:
    """A result read from a ``.pymxr`` file: its views, summary and provenance as they were when saved.

    ``kind`` is the original result's type (``"FittedEquation"``), ``views`` its tabs in order as
    ``(title, Table or Figure)``, ``text`` its summary, ``provenance`` how it was made (a dictionary),
    and ``source`` the file it was read from."""

    kind: str
    name: str
    saved: str
    package_version: str
    views: tuple[tuple[str, Any], ...]
    text: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)  # the estimate's own record (results.to_record), when it had one
    build: Mapping[str, Any] = field(default_factory=dict)  # the saving pymetrix's commit (pymetrix.build); empty: not recorded

    def summary(self) -> str:
        """The result as text, as its ``summary()`` printed when it was saved."""
        return self.text

    def _table(self, name: str) -> pd.DataFrame:
        from .errors import NotSupportedError
        from .results import frame_from_json

        if name not in self.record:
            raise NotSupportedError(f"the saved {self.kind} has no {name} (saved before pymetrix recorded them, or not an estimate)")
        return frame_from_json(self.record[name])

    def coefficients(self) -> pd.DataFrame:
        """The tidy coefficient table, as the result's ``coefficients()`` gave it when saved."""
        return self._table("coefficients")

    def statistics(self) -> pd.DataFrame:
        """The tidy statistics, with their definitions, as when saved."""
        return self._table("statistics")

    def observations(self) -> pd.DataFrame:
        """Actual, fitted and residual values on the data's periods, as when saved."""
        return self._table("observations")

    @property
    def stats(self) -> dict[str, float]:
        """The fit statistics, by name."""
        table = self.statistics()
        return {str(k): float(v) for k, v in zip(table["statistic"], table["value"])}

    def __repr__(self) -> str:
        return f"<Saved {self.kind} {self.name!r}: {len(self.views)} views, saved {self.saved} by pymetrix {self.package_version}>"


def save_result(obj: Any, path: str | Path, name: str = "", call: Any = None) -> Path:
    """Write ``obj`` (an estimate, test, forecast, solution... anything with a window) to ``path``.

    The suffix ``.pymxr`` is added when missing. ``name`` is the result's name (default: the file's).

    The provenance keeps the call that made the result (``provenance["call"]``: the function's ID, its arguments as
    recorded, with the span and row condition as given, and the datasets and other objects it used, by name), so
    ``Call(**provenance["call"]).invoke({"data": data})`` makes it again. A result a session step made has its call
    found (``pymetrix.calls.call_of``); ``call`` gives one (a ``Call``) for a result made otherwise."""
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    path.write_bytes(result_bytes(obj, name or path.stem, call=call))
    return path


def load_result(path: str | Path) -> SavedResult:
    """Read a result saved by ``save_result``."""
    path = Path(path)
    return result_from_bytes(path.read_bytes(), source=str(path), name=path.stem)


def result_bytes(obj: Any, name: str = "", call: Any = None) -> bytes:
    """``obj`` as the contents of a ``.pymxr`` file (``call``: as ``save_result``)."""
    from .views import summary_text, view_sections

    if isinstance(obj, SavedResult):  # saved again: as it was read, not with its "Saved" tab
        kind, sections, text, provenance, record = obj.kind, list(obj.views), obj.text, dict(obj.provenance), dict(obj.record)
        name = name or obj.name
    else:
        kind, sections, text, provenance = type(obj).__name__, view_sections(obj), summary_text(obj), _provenance(obj)
        record = obj.to_record() if callable(getattr(obj, "to_record", None)) else {}
    provenance.update(_call_provenance(obj, call))
    views, skipped = [], []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for title, value in sections:
            try:
                encoded = _encode_view(value)
            except _NotSaved as exc:
                skipped.append(f"{title}: {exc}")
                continue
            file = f"views/{len(views)}.json"
            z.writestr(file, json.dumps(encoded, default=_plain))
            views.append({"title": str(title), "type": encoded["type"], "file": file})
        if not views:
            raise SpecificationError(f"{name or kind} has no views to save" + (f" ({'; '.join(skipped)})" if skipped else ""))
        z.writestr("summary.txt", text)
        if provenance:
            z.writestr("provenance.json", json.dumps(provenance, indent=2, default=_plain))
        if record:
            z.writestr("record.json", json.dumps(record, default=_plain))
        from .build import build_info

        manifest = {"format_version": FORMAT_VERSION, "package_version": __version__, **build_info(),
                    "saved":datetime.now().isoformat(timespec="seconds"), "kind": kind, "name": name,
                    "views": views, "not_saved": skipped}
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return buffer.getvalue()


def result_from_bytes(blob: bytes, source: str = "", name: str = "") -> SavedResult:
    """A ``SavedResult`` from the contents of a ``.pymxr`` file."""
    label = Path(source).name if source else "the saved result"
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("format_version", 1) > FORMAT_VERSION:
                raise SpecificationError(f"{label} was saved by a newer version of pymetrix ({manifest.get('package_version', '?')}); "
                                         "upgrade pymetrix to open it")
            views = tuple((v["title"], _decode_view(json.loads(z.read(v["file"])))) for v in manifest["views"])
            names = set(z.namelist())
            text = z.read("summary.txt").decode("utf-8") if "summary.txt" in names else ""
            provenance = json.loads(z.read("provenance.json")) if "provenance.json" in names else {}
            record = json.loads(z.read("record.json")) if "record.json" in names else {}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise SpecificationError(f"{label} is not a saved pymetrix result ({exc})") from exc
    return SavedResult(kind=str(manifest.get("kind", "")), name=str(manifest.get("name") or name), saved=str(manifest.get("saved", "")),
                       package_version=str(manifest.get("package_version", "")), views=views, text=text, provenance=provenance,
                       source=source, record=record, build={k: manifest[k] for k in ("commit", "dirty", "note") if k in manifest})


# ---------------------------------------------------------------------------------------------
# views as JSON


class _NotSaved(PyMetrixError):
    pass


def _provenance(obj: Any) -> dict[str, Any]:
    from ._serial import to_jsonable

    prov = getattr(obj, "provenance", None)
    if prov is None or callable(prov):
        return {}
    try:
        out = to_jsonable(prov)
    except Exception:  # noqa: BLE001 - provenance is a courtesy here; the views carry the result
        return {}
    return out if isinstance(out, dict) else {"provenance": out}


def _call_provenance(obj: Any, call: Any) -> dict[str, Any]:
    """The call that made ``obj`` (given, or the one a session step made it with), as saved provenance: ``call`` (the
    fields ``Call`` takes) and ``call_uses`` (the named objects it reads: datasets and results, by name)."""
    from .calls import Call, call_of, call_record

    if call is None:
        call = None if isinstance(obj, SavedResult) else call_of(obj)  # saved again: its provenance as it was read
        if call is None:
            return {}
    if not isinstance(call, Call):
        raise SpecificationError(f"call is a pymetrix.calls.Call (make_call(...)), not {type(call).__name__}")
    return {"call": call_record(call), "call_uses": sorted(call.refs())}


def _plain(value: Any) -> Any:
    """JSON for what ``json`` does not know: NumPy scalars as numbers, anything else (dates, periods) as text."""
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _encode_view(value: Any) -> dict[str, Any]:
    from .figures import Figure
    from .output import Table

    if isinstance(value, Table):
        return {"type": "table", "title": value.title, "columns": [str(c) for c in value.columns],
                "rows": [list(r) for r in value.rows], "index": None if value.index is None else [str(i) for i in value.index],
                "notes": [str(n) for n in value.notes], "index_name": value.index_name}
    if isinstance(value, Figure):
        value = value._housed()  # saved as it is drawn: the house style's calls kept with it, to look the same anywhere
        theme = value.theme if value.theme is None or isinstance(value.theme, str) else value.theme.name
        return {"type": "figure", "title": value.title, "columns": value.columns, "width": value.width, "height": value.height,
                "subtitle": value.subtitle, "footnote": value.footnote, "theme": theme,
                "span": None if value.span is None else [None if s is None else str(s) for s in value.span],
                "shade": [list(s) for s in value.shade], "events": [list(e) for e in value.events],
                "reference": [list(r) for r in value.reference], "log": value.log,
                "ylim": None if value.ylim is None else list(value.ylim), "sharey": value.sharey, "bar_labels": value.bar_labels,
                "line_labels": value.line_labels, "style": [c.to_dict() for c in value.style], "house": value.house,
                "ybreak": None if value.ybreak is None else list(value.ybreak), "xbreak": None if value.xbreak is None else list(value.xbreak),
                "panels": [{"title": p.title, "ylabel": p.ylabel, "xlabel": p.xlabel, "legend": p.legend, "ylabel_right": p.ylabel_right,
                            "zlabel": p.zlabel, "projection": p.projection, **({"legend_loc": p.legend_loc} if p.legend_loc else {}),
                            "layers": [_encode_layer(layer) for layer in p.layers]} for p in value.panels]}
    raise _NotSaved(f"a {type(value).__name__} view cannot be saved")


def _layers() -> dict[str, type]:
    from .figures import (QQ, Band, Bars, Box, Contour, Ecdf, Estimates, Fan, Heatmap, Hexbin, Hist2d, Histogram, Line, Line3d, Rule, Scatter,
                          Scatter3d, Surface, Violin)

    return {c.__name__: c for c in (Line, Band, Bars, Scatter, Histogram, Rule, Box, Fan, Estimates, Heatmap, Ecdf, Hist2d, QQ, Hexbin, Violin,
                                    Contour, Surface, Line3d, Scatter3d)}


def _encode_layer(layer: Any) -> dict[str, Any]:
    if type(layer).__name__ not in _layers() or not isinstance(layer, _layers()[type(layer).__name__]):
        raise _NotSaved(f"a figure layer of type {type(layer).__name__} cannot be saved")
    out: dict[str, Any] = {"type": type(layer).__name__}
    for f in fields(layer):
        value = getattr(layer, f.name)
        if isinstance(value, pd.Series):
            out[f.name] = {"series": _encode_series(value)}
        elif isinstance(value, pd.DataFrame):  # a heatmap's, contour's or surface's table
            out[f.name] = {"frame": {"index": _encode_index(value.index), "columns": _encode_index(value.columns),
                                     "values": [[None if pd.isna(v) else float(v) for v in row] for row in value.to_numpy(float)]}}
        elif isinstance(value, tuple) and value and all(isinstance(v, pd.Series) for v in value):  # a box or violin plot's series
            out[f.name] = {"series_list": [_encode_series(v) for v in value]}
        else:
            out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def _encode_series(s: pd.Series) -> dict[str, Any]:
    name = s.name if s.name is None or isinstance(s.name, (str, int, float)) else str(s.name)
    if isinstance(s.dtype, pd.PeriodDtype):  # dates as values (a 3-D layer's date axis): kept as periods
        return {"name": name, "index": _encode_index(s.index), "values": [None if pd.isna(v) else str(v) for v in s],
                "freq": pd.PeriodIndex(s).freqstr}
    return {"name": name, "index": _encode_index(s.index), "values": s.to_numpy().tolist()}


def _encode_index(idx: pd.Index) -> dict[str, Any]:
    if isinstance(idx, pd.MultiIndex):
        return {"kind": "multi", "names": [None if n is None else str(n) for n in idx.names],
                "levels": [_encode_index(idx.get_level_values(i)) for i in range(idx.nlevels)]}
    name = None if idx.name is None else str(idx.name)
    if isinstance(idx, pd.PeriodIndex):
        return {"kind": "period", "freq": idx.freqstr, "name": name, "values": [str(p) for p in idx]}
    if isinstance(idx, pd.DatetimeIndex):
        return {"kind": "datetime", "name": name, "values": [t.isoformat() for t in idx]}
    if pd.api.types.is_numeric_dtype(idx.dtype) and not pd.api.types.is_bool_dtype(idx.dtype):
        return {"kind": "number", "name": name, "values": idx.tolist()}
    return {"kind": "text", "name": name, "values": [str(v) for v in idx]}


def _decode_view(d: Mapping[str, Any]) -> Any:
    from .figures import Figure, Panel
    from .output import Table

    if d["type"] == "table":
        return Table(columns=list(d["columns"]), rows=[list(r) for r in d["rows"]], index=d.get("index"), title=d.get("title", ""),
                     notes=tuple(d.get("notes", ())), index_name=d.get("index_name", ""))
    if d["type"] == "figure":
        panels = tuple(Panel(layers=tuple(_decode_layer(layer) for layer in p["layers"]), title=p.get("title", ""),
                             ylabel=p.get("ylabel", ""), xlabel=p.get("xlabel", ""), legend=p.get("legend", True),
                             ylabel_right=p.get("ylabel_right", ""), zlabel=p.get("zlabel", ""), projection=p.get("projection", ""),
                             legend_loc=p.get("legend_loc", ""))
                       for p in d["panels"])
        span, ylim = d.get("span"), d.get("ylim")
        return Figure(panels, title=d.get("title", ""), columns=d.get("columns", 1), width=d.get("width", 7.5), height=d.get("height"),
                      subtitle=d.get("subtitle", ""), footnote=d.get("footnote", ""), theme=d.get("theme"),
                      span=None if span is None else tuple(span), shade=[tuple(s) for s in d.get("shade", [])],
                      events=[tuple(e) for e in d.get("events", [])], reference=[tuple(r) for r in d.get("reference", [])],
                      log=d.get("log", False), ylim=None if ylim is None else tuple(ylim), sharey=d.get("sharey", False),
                      bar_labels=d.get("bar_labels", False), line_labels=d.get("line_labels", False),
                      style=tuple(d.get("style", ())), house=d.get("house", True),
                      ybreak=None if d.get("ybreak") is None else tuple(d["ybreak"]),
                      xbreak=None if d.get("xbreak") is None else tuple(d["xbreak"]))
    raise SpecificationError(f"unknown kind of saved view: {d['type']!r}")


def _decode_layer(d: Mapping[str, Any]) -> Any:
    cls = _layers().get(d.get("type", ""))
    if cls is None:
        raise SpecificationError(f"unknown kind of figure layer: {d.get('type')!r}")
    known = {f.name for f in fields(cls)}
    def value(v: Any) -> Any:
        if isinstance(v, Mapping) and "series" in v:
            return _decode_series(v["series"])
        if isinstance(v, Mapping) and "frame" in v:  # a heatmap's table
            f = v["frame"]
            return pd.DataFrame([[np.nan if x is None else x for x in row] for row in f["values"]],
                                index=_decode_index(f["index"]),
                                columns=_decode_index(f["columns"]) if isinstance(f["columns"], Mapping) else f["columns"])
        if isinstance(v, Mapping) and "series_list" in v:  # a box plot's series
            return tuple(_decode_series(s) for s in v["series_list"])
        return tuple(v) if isinstance(v, list) else v

    return cls(**{k: value(v) for k, v in d.items() if k in known})


def _decode_series(d: Mapping[str, Any]) -> pd.Series:
    if "freq" in d:  # periods as values
        values = pd.PeriodIndex([pd.NaT if v is None else v for v in d["values"]], freq=d["freq"])
        return pd.Series(values, index=_decode_index(d["index"]), name=d.get("name"))
    return pd.Series(d["values"], index=_decode_index(d["index"]), name=d.get("name"))


def _decode_index(d: Mapping[str, Any]) -> pd.Index:
    kind, name = d["kind"], d.get("name")
    if kind == "multi":
        return pd.MultiIndex.from_arrays([_decode_index(level) for level in d["levels"]], names=d.get("names"))
    if kind == "period":
        return pd.PeriodIndex(d["values"], freq=d["freq"], name=name)
    if kind == "datetime":
        return pd.DatetimeIndex(d["values"], name=name)
    if kind == "number":
        return pd.Index(d["values"], name=name)
    return pd.Index([str(v) for v in d["values"]], name=name, dtype=object)
