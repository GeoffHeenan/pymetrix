"""pymetrix-psets: series from a PSETS database, read through PSETS's plug-in API (``psets`` 1.x).

The reader described in pymetrix's ``docs/design/plugins.md`` (Case 1), over PSETS's read-only host API
(``Econ_timeseries_DB``: ``docs/plugin_api.md``). It is written against pymetrix's Extension API only, as an outside
package would be.

    data = Dataset.read("psets://forecasts/NGDP_R.Q,NGDP.Q")              # alias "forecasts", two codes
    data = Dataset.read("psets:///PCPI_LOC.M?start=2010-01-01")           # the user's default database
    data = Dataset.read("psets://forecasts/PCPI_LOC.M?freq=Q&agg=avg")    # PSETS converts downward as it reads
    data = Dataset.read("psets://forecasts/NGDP_R.Q", vintage="2020-03-29")   # as released by that date

What it adds to each series:

- its PSETS metadata, as pymetrix's ``Variable`` (label, units, adjustment, source, the release it came from);
- ``external_ref`` = ``psets://<alias>/<code>@<vintage>``: PSETS's own record stays the authority;
- in ``extra``: frequency, coverage, dataset, sector, topic, geography, multiplier, what a canonical code is
  built from (``provenance``), and the publisher problems PSETS knows about (``issues``). Open issues are also
  raised as a ``PsetsIssueWarning`` when the data are read, where they would stop someone.

Not here, by design: writing to PSETS (its API is read-only; figures go in through its own tools), and the
Get data browser, which needs pymetrix's proposed catalogue hook (``PsetsCatalogue`` below is written to it).
"""

from __future__ import annotations

import re
import urllib.parse
import warnings
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

from pymetrix.data import Dataset
from pymetrix.errors import SpecificationError
from pymetrix.metadata import Adjustment, DatasetMeta, Variable, Vintage
from pymetrix.plugins import PluginInfo, add_plugin

from ._version import __version__

__all__ = ["PLUGIN", "PsetsCatalogue", "PsetsIssueWarning", "__version__", "psets_vintages", "read_psets", "register"]

PLUGIN = PluginInfo(
    name="pymetrix-psets",
    version=__version__,
    requires={"psets": ""},  # PSETS's API version is checked on use (_open): psets.__version__ is the API's, not the project's
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Series from a PSETS database: metadata, vintages, provenance and known publisher issues",
    install='pip install "PSETS @ git+https://github.com/GeoffHeenan/Econ_timeseries_DB#subdirectory=python"',
)

API = "1"  # the PSETS plug-in API major version this plugin is written against

#: PSETS frequency codes as pandas period frequencies. FY needs the series' fiscal year end (Y-JUN).
_FREQ = {"D": "D", "W": "W", "M": "M", "Q": "Q", "A": "Y"}
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_OPTIONS = ("start", "end", "freq", "agg")  # what an address's query may carry, passed to db.frame


class PsetsIssueWarning(UserWarning):
    """A publisher problem PSETS knows about, in a series just read."""


# -- reading -----------------------------------------------------------------------------------------------------


def read_psets(source: str, vintage: str | None = None, **options: Any) -> Dataset:
    """``psets://<alias>/<code>,<code>?start=&end=&freq=&agg=``: the series, as one dataset (one frequency).

    ``vintage`` (from ``Dataset.read(..., vintage=...)``) is PSETS's ``as_of``: the data as the publishers had
    released them by that date. Options in the address and keywords both reach ``db.frame``."""
    alias, codes, query = _parse(source)
    settings = {**query, **{k: str(v) for k, v in options.items() if k in _OPTIONS and v}}
    db = _open(alias)
    infos = {code: db.series(code) for code in codes}  # raises psets.UnknownSeries for a code it does not hold
    freqs = {_period_freq(info, settings.get("freq", "")) for info in infos.values()}
    if len(freqs) > 1:
        raise SpecificationError(f"the series have different frequencies ({', '.join(sorted(freqs))}): read them "
                                 "separately, or give freq= and agg= to convert downward")
    freq = freqs.pop()
    long = db.frame(codes, long=True, as_of=vintage or "", **{k: settings[k] for k in _OPTIONS if k in settings})
    names = _names(codes)
    columns, variables = {}, {}
    retrieved = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005 - local time, as pymetrix's own readers stamp it
    for code, info in infos.items():
        rows = long[long["series_id"] == code]
        index = pd.PeriodIndex(pd.to_datetime(rows["period"]), freq=freq)
        columns[names[code]] = pd.Series(rows["value"].astype(float).to_numpy(), index=index)
        released = _newest(rows["vintage"]) if "vintage" in rows else None  # a converted frame may not carry it
        variables[names[code]] = _variable(names[code], info, db, released, retrieved)
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "time"
    label = db.alias or db.path
    meta = DatasetMeta(title=f"PSETS {label}: {', '.join(codes)}",
                       extra={"retrieved": retrieved, "psets_database": label, "psets_api": _api_version()})
    data = Dataset.from_frame(frame, variables=variables, meta=meta)
    _warn_issues(variables)
    return data


def _parse(source: str) -> tuple[str, list[str], dict[str, str]]:
    parts = urllib.parse.urlsplit(source)
    if parts.scheme.lower() != "psets":
        raise SpecificationError(f"{source!r} is not a psets:// address")
    codes = [c.strip() for c in parts.path.strip("/").split(",") if c.strip()]
    if not codes:
        raise SpecificationError("give PSETS series codes, e.g. psets://forecasts/NGDP_R.Q,PCPI_LOC.M")
    query = dict(urllib.parse.parse_qsl(parts.query))
    unknown = sorted(set(query) - set(_OPTIONS))
    if unknown:
        raise SpecificationError(f"psets:// addresses take {', '.join(_OPTIONS)}, not {', '.join(unknown)}")
    return parts.netloc, codes, query


def _open(alias: str) -> Any:
    """The database by alias (or the user's default), refusing a PSETS API this plugin was not written for."""
    import psets

    found = str(getattr(psets, "__version__", ""))
    if found.split(".")[0] != API:
        raise SpecificationError(f"pymetrix-psets needs PSETS plug-in API {API}.x; this PSETS has {found or 'none'}")
    return psets.open(alias)


def _api_version() -> str:
    import psets

    return str(getattr(psets, "__version__", ""))


def _period_freq(info: Any, target: str) -> str:
    """The pandas frequency of a series as read: its own, or ``freq=`` when PSETS converts it."""
    code = (target or info.freq).upper()
    if code == "FY":
        month = _fiscal_month(info.fiscal_year_end)
        return f"Y-{month}" if month else "Y"
    if code not in _FREQ:
        raise SpecificationError(f"{info.series_id} has frequency {code!r}, which pymetrix does not read")
    return _FREQ[code]


def _fiscal_month(text: str) -> str:
    """``"06"``, ``"6"``, ``"June"`` or ``"--06-30"`` as JUN; "" when it cannot be told."""
    text = str(text or "").strip()
    for number in re.findall(r"\d+", text):  # the first number that can be a month: 06 in --06-30, 6 in "6"
        if 1 <= int(number) <= 12:
            return _MONTHS[int(number) - 1]
    return next((m for m in _MONTHS if text.upper().startswith(m)), "")


def _names(codes: Iterable[str]) -> dict[str, str]:
    """pymetrix series names from PSETS codes: PCPI_LOC.M -> pcpi_loc_m."""
    out = {}
    for code in codes:
        name = re.sub(r"\W+", "_", code).strip("_").lower()
        out[code] = name if name and not name[0].isdigit() else f"s_{name}"
    return out


def _newest(vintages: Iterable[Any]) -> date | None:
    found = [str(v) for v in vintages if v not in (None, "") and not pd.isna(v)]
    return date.fromisoformat(max(found)[:10]) if found else None


def _variable(name: str, info: Any, db: Any, released: date | None, retrieved: str) -> Variable:
    """A PSETS series' metadata as pymetrix keeps it; the rest in ``extra``, and PSETS's record by reference."""
    adjustment = _adjustment(info.seasonal_adj)
    label = db.alias or db.path
    extra: dict[str, Any] = {
        "psets_freq": info.freq, "psets_span": list(info.span), "psets_dataset": info.dataset_name or info.dataset,
        "psets_sector": info.sector, "psets_topic": info.topic, "psets_geo": info.geo,
        "psets_multiplier": info.multiplier, "psets_last_loaded": info.last_loaded_at,
    }
    if adjustment is None and info.seasonal_adj:
        extra["psets_seasonal_adj"] = info.seasonal_adj
    provenance = db.provenance(info.series_id)
    if provenance.canonical:
        extra["psets_provenance"] = [{"series_id": s.series_id, "role": s.role, "rank": s.rank, "provider": s.provider,
                                      "provider_code": s.provider_code, "link_period": s.link_period,
                                      "link_factor": s.link_factor} for s in provenance.segments]
    issues = [i for i in db.issues(info.series_id) if i.open]
    if issues:
        extra["psets_issues"] = [{"id": i.issue_id, "publisher": i.publisher, "kind": i.kind, "summary": i.summary,
                                  "status": i.status} for i in issues]
    return Variable(
        name,
        label=info.title,
        description=info.notes,
        units=info.unit,
        adjustment=adjustment or Adjustment.NSA,
        source=f"PSETS {label}: {info.source_name or info.source} ({info.series_id})",
        source_key=info.series_id,
        vintage=Vintage(released, retrieved=datetime.fromisoformat(retrieved)) if released else None,
        retrieved=retrieved,
        external_ref=f"psets://{db.alias}/{info.series_id}" + (f"@{released.isoformat()}" if released else ""),
        extra=extra,
    )


def _adjustment(text: str) -> Adjustment | None:
    value = str(text or "").strip().lower().replace(" ", "")
    mapping = {"sa": Adjustment.SA, "seasonallyadjusted": Adjustment.SA, "nsa": Adjustment.NSA,
               "notseasonallyadjusted": Adjustment.NSA, "swda": Adjustment.SWDA, "trend": Adjustment.TREND}
    return mapping.get(value)


def _warn_issues(variables: Mapping[str, Variable]) -> None:
    for name, var in variables.items():
        for issue in var.extra.get("psets_issues", ()):
            warnings.warn(f"{name} ({var.source_key}): {issue['publisher']} {issue['kind']}: {issue['summary']} "
                          f"(PSETS issue {issue['id']}, {issue['status']})", PsetsIssueWarning, stacklevel=3)


# -- real-time data for backtests -----------------------------------------------------------------------------


def psets_vintages(source: str, origins: Iterable[Any]) -> dict[Any, Dataset]:
    """``{origin: the data as released by the end of that period}``, for ``backtest(..., vintages=...)``:

        vintages = psets_vintages("psets://forecasts/NGDP_R.Q", pd.period_range("2015Q1", "2019Q4", freq="Q"))
        result = backtest(fit, data, ("2015Q1", "2019Q4"), horizon=4, vintages=vintages)
    """
    out = {}
    for origin in origins:
        period = origin if isinstance(origin, pd.Period) else pd.Period(origin)
        out[period] = read_psets(source, vintage=period.end_time.date().isoformat())
    return out


# -- browsing (pymetrix's proposed catalogue hook; not yet registered anywhere) ---------------------------------


class PsetsCatalogue:
    """What the proposed *Data → Get data* dialog asks a data plugin (docs/design/plugins.md, "Proposal for
    decision 1"). PSETS browses by facets, so ``children`` is not offered. Each method is one PSETS call."""

    def __init__(self, alias: str = "") -> None:
        self.alias = alias

    def search(self, text: str = "", filters: Mapping[str, Any] | None = None, limit: int = 100) -> list[Variable]:
        db = _open(self.alias)
        found = db.search(text, **dict(filters or {}))
        return [self._describe(db, code) for code in list(found["series_id"])[:limit]]

    def facets(self, filters: Mapping[str, Any] | None = None) -> dict[str, list[tuple[str, int]]]:
        return _open(self.alias).facets(**dict(filters or {}))

    def describe(self, keys: Iterable[str]) -> list[Variable]:
        db = _open(self.alias)
        return [self._describe(db, code) for code in keys]

    def vintages(self, key: str) -> list[str]:
        return _open(self.alias).vintages(key)

    def address(self, keys: Iterable[str]) -> str:
        """The address that reads these series: what the dialog's *Read* records."""
        return f"psets://{self.alias}/{','.join(keys)}"

    def _describe(self, db: Any, code: str) -> Variable:
        info = db.series(code)
        vintages = db.vintages(code)
        released = date.fromisoformat(vintages[-1][:10]) if vintages else None
        name = _names([code])[code]
        return _variable(name, info, db, released, datetime.now().isoformat(timespec="seconds"))  # noqa: DTZ005


# -- registration ---------------------------------------------------------------------------------------------


def register() -> None:
    """Called through the ``pymetrix.io`` entry point: ``psets://`` addresses read through this plugin."""
    from pymetrix.io.registry import register_reader

    add_plugin(PLUGIN)
    register_reader("psets", read_psets)
