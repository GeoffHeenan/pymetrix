"""Reference plugin: data from FRED, the World Bank and SDMX providers (ECB, Eurostat, OECD, IMF, BIS).

    Dataset.read("fred://GDPC1,UNRATE")                          # FRED series; ?vintage=2020-01-30 for ALFRED
    Dataset.read("fred://GDPC1?start=1990Q1&end=2019Q4")         # the periods between (kept after the download)
    Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON;FJI&date=2000:2023")   # a panel by country
    Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2015-01")
    Dataset.read("sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M?entity=COUNTRY")  # a panel by a dimension
    register_sdmx_provider("MYCB", "https://data.mycb.org/sdmx/data/{flow}/{key}?format=csvdata")

Written against the Extension API only (an ``io`` reader per URI scheme), so it could be a
separate package. It reads the providers' public web services with the standard library:
SDMX-CSV and SDMX-ML (structure-specific) data messages, FRED's CSV download, and the World
Bank's JSON API. Every series records where it came from; when a provider states when its
data were released or last updated, that date is the series' vintage (never the download
time). An institution's own SDMX service (a data warehouse) is one ``register_sdmx_provider``
call away, which is how institutional data stores are meant to connect to pymetrix.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from .._version import __version__
from ..io.registry import Connector, register_reader
from ..metadata import DatasetMeta, Variable, Vintage
from ..plugins import PluginInfo, add_plugin

if TYPE_CHECKING:
    from ..data import Dataset

PLUGIN = PluginInfo(
    name="pymetrix-connectors",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Data from FRED, the World Bank and SDMX providers (ECB, Eurostat, OECD, IMF, BIS, or your own)",
    install="included with pymetrix",
)

__all__ = ["PLUGIN", "read_fred", "read_sdmx", "read_worldbank", "register_sdmx_provider", "sdmx_providers"]

TIMEOUT = 60


def _get(url: str) -> bytes:
    """Fetch a URL (tests replace this to use recorded responses)."""
    request = urllib.request.Request(url, headers={"User-Agent": f"pymetrix/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - https URLs built here
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        message = _sdmx_error(body) or f"HTTP {exc.code} {exc.reason}"
        raise ConnectionError(f"{message} ({url})") from None
    except urllib.error.URLError as exc:
        raise ConnectionError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {exc.reason}") from None


def _ident(text: str) -> str:
    name = re.sub(r"\W+", "_", text).strip("_").lower()
    return name if name and not name[0].isdigit() else f"s_{name}"


def _parse_uri(source: str, scheme: str) -> tuple[str, dict[str, str]]:
    parts = urllib.parse.urlsplit(source)
    if parts.scheme.lower() != scheme:
        raise SpecificationError(f"{source!r} is not a {scheme}:// address")
    path = (parts.netloc + parts.path).strip("/")
    return path, dict(urllib.parse.parse_qsl(parts.query))


def _period_index(values: list[str]) -> pd.PeriodIndex:
    """Periods from SDMX, FRED and World Bank date strings: 2024, 2024-Q1, 2024Q1, 2024-01, 2024-M01, 2024-01-15."""
    sample = values[0]
    if re.fullmatch(r"\d{4}", sample):
        return pd.PeriodIndex(values, freq="Y")
    if re.fullmatch(r"\d{4}-?Q\d", sample):
        return pd.PeriodIndex([v.replace("-", "") for v in values], freq="Q")
    if re.fullmatch(r"\d{4}-M\d{2}", sample) or re.fullmatch(r"\d{4}M\d{2}", sample):
        return pd.PeriodIndex([re.sub(r"-?M", "-", v) for v in values], freq="M")
    if re.fullmatch(r"\d{4}-\d{2}", sample):
        return pd.PeriodIndex(values, freq="M")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", sample):
        return pd.PeriodIndex(values, freq="D")
    raise SpecificationError(f"unrecognised period format {sample!r}")


def _from_dates(dates: pd.DatetimeIndex) -> pd.PeriodIndex:
    """FRED gives the first day of each period: recover the frequency from the spacing."""
    if len(dates) < 3:
        return pd.PeriodIndex(dates, freq="D")
    gaps = np.median(np.diff(dates.values).astype("timedelta64[D]").astype(int))
    freq = "Y" if gaps > 300 else "Q" if gaps > 80 else "M" if gaps > 25 else "W" if gaps > 5 else "D"
    return pd.PeriodIndex(dates, freq=freq)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _vintage(released: date | None) -> Vintage | None:
    return Vintage(released, retrieved=datetime.now()) if released else None


def _dataset(frame: pd.DataFrame, variables: Mapping[str, Variable], title: str, entity: str | None = None) -> Dataset:
    from ..data import Dataset

    meta = DatasetMeta(title=title, extra={"retrieved": datetime.now().isoformat(timespec="seconds")})
    if entity:
        flat = frame.reset_index()
        return Dataset.from_frame(flat, time="time", entity=entity, variables=dict(variables), meta=meta)
    frame.index.name = "time"
    return Dataset.from_frame(frame.sort_index(), variables=dict(variables), meta=meta)


# ---------------------------------------------------------------------------------------------
# FRED


def read_fred(source: str, **kw: Any) -> Dataset:
    """``fred://ID1,ID2``; ``?vintage=YYYY-MM-DD`` gives the data as they were on that date (ALFRED); ``start=`` and
    ``end=`` (periods, e.g. 1990Q1) keep the periods between them: FRED's download is the whole history, cut here."""
    path, query = _parse_uri(source, "fred")
    ids = [i.strip().upper() for i in path.split(",") if i.strip()]
    if not ids:
        raise SpecificationError("give FRED series IDs, e.g. fred://GDPC1,UNRATE")
    vintage = query.get("vintage") or kw.get("vintage")
    columns, variables = {}, {}
    freqs = set()
    for sid in ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(sid)}"
        if vintage:
            url += f"&vintage_date={urllib.parse.quote(str(vintage))}"
        table = pd.read_csv(io.BytesIO(_get(url)), na_values=["."])
        date_col = table.columns[0]
        if sid not in table.columns:
            raise SpecificationError(f"FRED did not return {sid}: is the ID right?")
        index = _from_dates(pd.DatetimeIndex(pd.to_datetime(table[date_col])))
        freqs.add(index.freqstr)
        name = _ident(sid)
        columns[name] = pd.Series(table[sid].astype(float).to_numpy(), index=index)
        note = f" (ALFRED vintage as of {vintage})" if vintage else ""
        released = date.fromisoformat(str(vintage)) if vintage else None  # ALFRED: the data as released by that date
        variables[name] = Variable(name, label=sid, source=f"FRED, Federal Reserve Bank of St. Louis: {sid}{note}", source_key=sid,
                                   vintage=_vintage(released), retrieved=_now())
    if len(freqs) > 1:
        raise SpecificationError(f"the series have different frequencies ({', '.join(sorted(freqs))}); read them separately")
    frame = pd.DataFrame(columns)
    data = _dataset(frame, variables, f"FRED: {', '.join(ids)}")
    start, end = query.get("start") or kw.get("start"), query.get("end") or kw.get("end")
    return data.select(span=(start or None, end or None)) if start or end else data


# ---------------------------------------------------------------------------------------------
# World Bank


def read_worldbank(source: str, **kw: Any) -> Dataset:
    """``worldbank://INDICATOR1,INDICATOR2?country=WSM;TON&date=2000:2023`` (default: all countries), as a
    panel by country (ISO3 codes); ``start=2000&end=2023`` is the same as ``date=2000:2023``. The World Bank's
    last-update date is the vintage."""
    path, query = _parse_uri(source, "worldbank")
    start, end = query.pop("start", ""), query.pop("end", "")
    if (start or end) and "date" not in query:  # the API takes years: 2000:2023
        query["date"] = f"{str(start or 1960)[:4]}:{str(end or date.today().year)[:4]}"
    indicators = [i.strip() for i in path.split(",") if i.strip()]
    if not indicators:
        raise SpecificationError("give World Bank indicator codes, e.g. worldbank://NY.GDP.MKTP.CD")
    country = query.get("country", "all")
    rows: dict[tuple[str, str], dict[str, float]] = {}
    variables = {}
    for indicator in indicators:
        page, pages = 1, 1
        label, updated = indicator, None
        while page <= pages:
            params = {"format": "json", "per_page": query.get("per_page", "20000"), "page": str(page)}
            if "date" in query:
                params["date"] = query["date"]
            url = (f"https://api.worldbank.org/v2/country/{urllib.parse.quote(country, safe=';')}/indicator/"
                   f"{urllib.parse.quote(indicator)}?{urllib.parse.urlencode(params)}")
            payload = json.loads(_get(url))
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                message = payload[0].get("message", [{}])[0].get("value") if isinstance(payload, list) and payload else None
                raise SpecificationError(f"the World Bank returned no data for {indicator} ({message or 'check the codes'})")
            head, data = payload
            pages = int(head.get("pages", 1))
            updated = head.get("lastupdated", updated)
            for obs in data:
                label = obs["indicator"]["value"]
                code = obs.get("countryiso3code") or obs["country"]["id"]
                if obs["value"] is not None:
                    rows.setdefault((code, obs["date"]), {})[_ident(indicator)] = float(obs["value"])
            page += 1
        released = date.fromisoformat(updated) if updated else None
        variables[_ident(indicator)] = Variable(_ident(indicator), label=label, source=f"World Bank: {indicator}", source_key=indicator,
                                                vintage=_vintage(released), retrieved=_now())
    if not rows:
        raise SpecificationError("the World Bank returned no observations")
    keys = sorted(rows)
    periods = _period_index([k[1] for k in keys])
    frame = pd.DataFrame([rows[k] for k in keys], columns=list(variables))
    frame.insert(0, "country", [k[0] for k in keys])
    frame.insert(1, "time", periods)
    return _dataset(frame.set_index(["country", "time"]), variables, f"World Bank: {', '.join(indicators)}", entity="country")


# ---------------------------------------------------------------------------------------------
# SDMX


_PROVIDERS: dict[str, str] = {
    "ECB": "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata",
    "ESTAT": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{flow}/{key}?format=SDMX-CSV",
    "OECD": "https://sdmx.oecd.org/public/rest/data/{flow}/{key}?format=csvfile",
    "IMF": "https://api.imf.org/external/sdmx/2.1/data/{flow}/{key}",
    "BIS": "https://stats.bis.org/api/v1/data/{flow}/{key}?format=csv",
}
_ALIASES = {"EUROSTAT": "ESTAT"}


def register_sdmx_provider(name: str, url: str, structure_url: str = "", agency: str = "all") -> None:
    """Add an SDMX web service, e.g. an institution's own: ``url`` has ``{flow}`` and ``{key}`` placeholders
    and returns SDMX-CSV or SDMX-ML structure-specific data. ``structure_url``, the service's REST root (where
    ``/dataflow/{agency}`` answers), makes its dataflows searchable in the Series browser; ``agency`` is the
    maintenance agency to list ("all" for every one)."""
    if "{flow}" not in url or "{key}" not in url:
        raise SpecificationError("the URL template needs {flow} and {key}")
    _PROVIDERS[name.upper()] = url
    if structure_url:
        from .connector_search import SDMX_STRUCTURES

        SDMX_STRUCTURES[name.upper()] = (structure_url.rstrip("/"), agency, "all")


def sdmx_providers() -> dict[str, str]:
    return dict(_PROVIDERS)


_NOT_DIMENSIONS = {"DATAFLOW", "KEY", "LAST UPDATE", "TIME_PERIOD", "OBS_VALUE", "STRUCTURE", "STRUCTURE_ID", "ACTION"}


def _sdmx_error(body: bytes) -> str | None:
    text = body[:4000].decode("utf-8", "replace")
    if "Error" in text and "<" in text:
        found = re.search(r"<(?:\w+:)?Text[^>]*>(.*?)</(?:\w+:)?Text>", text, re.S)
        if found:
            return "SDMX: " + " ".join(found.group(1).split())
    return None


def _series_from_csv(body: bytes, n_dims: int | None) -> tuple[list[dict[str, Any]], date | None]:
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    if not rows or "TIME_PERIOD" not in rows[0] or "OBS_VALUE" not in rows[0]:
        raise SpecificationError("the response is not SDMX-CSV data (no TIME_PERIOD and OBS_VALUE columns)")
    header = list(rows[0])
    dims = header[: header.index("TIME_PERIOD")]
    dims = [d for d in dims if d.upper() not in _NOT_DIMENSIONS][:n_dims]  # the key's positions; the rest are attributes
    released = None
    if "LAST UPDATE" in header and rows[0]["LAST UPDATE"]:
        released = datetime.strptime(rows[0]["LAST UPDATE"].split()[0], "%d/%m/%y").date()
    series: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        key = tuple(r[d] for d in dims)
        s = series.setdefault(key, {"dims": dict(zip(dims, key)), "title": r.get("TITLE", ""), "obs": {}})
        value = r["OBS_VALUE"].strip()
        s["obs"][r["TIME_PERIOD"]] = float(value) if value not in ("", "NaN", "NA") else np.nan
    return list(series.values()), released


def _series_from_xml(body: bytes, n_dims: int | None) -> tuple[list[dict[str, Any]], date | None]:
    root = ET.fromstring(body)
    if root.tag.endswith("Error"):
        raise SpecificationError(_sdmx_error(body) or "the SDMX service returned an error")
    released = None
    out = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "DataSet":
            stamp = element.attrib.get("UPDATE_DATE") or element.attrib.get("PUBLICATION_DATE")
            if stamp:
                released = datetime.fromisoformat(stamp[:19]).date()
        elif tag == "Series":
            obs = {o.attrib["TIME_PERIOD"]: float(o.attrib.get("OBS_VALUE", "nan")) for o in element
                   if o.tag.rsplit("}", 1)[-1] == "Obs" and "TIME_PERIOD" in o.attrib}
            attrs = {k: v for k, v in element.attrib.items() if not k.startswith("{")}
            title = attrs.pop("TITLE", "")
            dims = dict(list(attrs.items())[:n_dims])  # dimensions come first, then attributes
            out.append({"dims": dims, "title": title, "obs": obs})
    if not out:
        raise SpecificationError("the SDMX message has no series")
    return out, released


def read_sdmx(source: str, **kw: Any) -> Dataset:
    """``sdmx://PROVIDER/FLOW/KEY?startPeriod=...&endPeriod=...``. Several series (a key with ``+`` or a
    wildcard) become several columns, named by the dimensions that differ; ``entity=DIMENSION`` makes a
    panel by that dimension instead; ``name=...`` names a single series."""
    path, query = _parse_uri(source, "sdmx")
    parts = path.split("/", 2)
    if len(parts) != 3:
        raise SpecificationError("write sdmx://PROVIDER/FLOW/KEY, e.g. sdmx://ECB/EXR/M.USD.EUR.SP00.A")
    provider, flow, key = parts
    provider = _ALIASES.get(provider.upper(), provider.upper())
    if provider not in _PROVIDERS:
        raise SpecificationError(f"unknown SDMX provider {provider!r}; known: {', '.join(_PROVIDERS)} (add one with register_sdmx_provider)")
    entity = query.pop("entity", None) or kw.get("entity")
    single_name = query.pop("name", None) or kw.get("name")
    url = _PROVIDERS[provider].format(flow=urllib.parse.quote(flow, safe=",@"), key=urllib.parse.quote(key, safe="+.*"))
    if query:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query)
    body = _get(url)
    n_dims = None if key.lower() in ("", "all", "*") else len(key.split("."))
    series, released = _series_from_xml(body, n_dims) if body.lstrip()[:1] == b"<" else _series_from_csv(body, n_dims)
    varying = [d for d in series[0]["dims"] if len({s["dims"].get(d) for s in series}) > 1]
    if entity and entity not in series[0]["dims"]:
        raise SpecificationError(f"no dimension {entity!r}; dimensions: {list(series[0]['dims'])}")
    name_dims = [d for d in varying if d != entity]
    vintage = _vintage(released)
    columns: dict[Any, pd.Series] = {}
    variables: dict[str, Variable] = {}
    base = single_name or (flow.split(",")[1] if "," in flow else flow)  # "IMF.STA,CPI" -> CPI
    for s in series:
        name = _ident("_".join(s["dims"][d] for d in name_dims)) if name_dims and not single_name else _ident(base)
        periods = list(s["obs"])
        values = pd.Series(list(s["obs"].values()), index=_period_index(periods) if periods else pd.PeriodIndex([], freq="M"))
        label = s["title"] or ".".join("*" if d == entity else v for d, v in s["dims"].items())
        series_key = ".".join(s["dims"].values())
        variables[name] = Variable(name, label=label.strip(), source=f"{provider} SDMX: {flow}/{key}", source_key=f"{flow}/{series_key}",
                                   vintage=vintage, retrieved=_now())
        columns[(s["dims"][entity], name) if entity else name] = values
    title = f"{provider}: {flow}/{key}"
    if entity:
        pieces = [pd.DataFrame({entity: ent, "time": values.index, name: values.to_numpy()}) for (ent, name), values in columns.items()]
        long = pd.concat(pieces).groupby([entity, "time"], sort=True).first()  # one row per entity and period
        return _dataset(long[list(variables)], variables, title, entity=entity)
    return _dataset(pd.DataFrame(columns), variables, title)


add_plugin(PLUGIN)
register_reader("fred", read_fred, Connector("FRED (Federal Reserve Bank of St. Louis)", "series IDs, e.g. GDPC1,UNRATE",
                                          "start", "end", vintage=True, search=True,
                                          search_note="searching needs a FRED API key (FRED_API_KEY); reading the data does not"))
register_reader("worldbank", read_worldbank, Connector("World Bank", "indicator codes, e.g. NY.GDP.MKTP.CD; a panel by country "
                                                                     "(Options: country=WSM;TON)", "start", "end", search=True,
                                                       search_note="indicators of a source (the World Development Indicators by default), topics and countries"))
register_reader("sdmx", read_sdmx, Connector("SDMX (ECB, Eurostat, OECD, IMF, BIS or your own)",
                                             "PROVIDER/FLOW/KEY, e.g. ECB/EXR/M.USD.EUR.SP00.A", "startPeriod", "endPeriod", search=True,
                                             search_note="a provider's dataflows, then their dimensions' codes for the series key"))

from . import connector_search  # noqa: E402,F401  (the searchers of these three connectors)
