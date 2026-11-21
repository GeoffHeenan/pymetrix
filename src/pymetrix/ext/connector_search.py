"""Reference plugin, continued: searching and browsing FRED, the World Bank and SDMX providers.

    FRED          series search and categories (``fred/series/search``, ``fred/category/children``,
                  ``fred/category/series``): FRED's web service, which needs an API key (``FRED_API_KEY`` or
                  ``pymetrix.connectors.set_api_key("fred", ...)``). Reading the data (``fred://``) needs none.
    World Bank    the indicators of a source (the World Development Indicators by default), searched here; sources,
                  topics and countries to browse (``/v2/source``, ``/v2/topic``, ``/v2/country``).
    SDMX          a provider's dataflows (``/dataflow/{agency}``), searched here, then a dataflow's data structure
                  with its codelists (``/dataflow/{agency}/{id}/latest?references=all``) for writing series keys;
                  :func:`dataflow_tree` puts the dataflows in the provider's own folders, its category schemes and
                  categorisations (``/categoryscheme/{agency}/all/latest?references=parentsandsiblings``), or groups
                  them otherwise (``pymetrix.ext.browse_folders``).

Each answers with ``pymetrix.connectors.SeriesInfo``; ``pymetrix.connectors`` keeps the answers in its cache.
Requests go through ``connectors._get`` (as the readers' do), so tests answer them from recorded responses.
"""

from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from ..connectors import Category, Fetch, Listing, SeriesInfo, Structure, api_key, rank, register_searcher
from ..errors import NotFoundError, PyMetrixError, SpecificationError
from . import connectors as _conn
from .browse_folders import DataflowTree, choose_tree, code_prefix, code_prefix_groups, parse_categories
from .sdmx_structure import Reference, dimensions, reference


def _transport(url: str) -> bytes:
    return _conn._get(url)  # looked up when called: tests replace it


# ---------------------------------------------------------------------------------------------------------------------
# FRED

FRED_API = "https://api.stlouisfed.org/fred"
NO_FRED_KEY = ("FRED's search needs an API key (free, from fred.stlouisfed.org/docs/api/api_key.html): set FRED_API_KEY "
               "or give it in the Series browser. Reading series by ID (fred://GDPC1) uses FRED's download, which needs none.")


def _fred_series(s: dict[str, Any]) -> SeriesInfo:
    return SeriesInfo("fred", str(s.get("id", "")), title=s.get("title", ""), frequency=s.get("frequency", ""), units=s.get("units", ""),
                      start=s.get("observation_start", ""), end=s.get("observation_end", ""), last_updated=s.get("last_updated", ""),
                      notes=(s.get("notes") or "").strip(), seasonal_adjustment=s.get("seasonal_adjustment", ""),
                      source="FRED, Federal Reserve Bank of St. Louis")


class FredSearch:
    secrets = ("api_key",)
    transport = staticmethod(_transport)

    def _get(self, fetch: Fetch, endpoint: str, kind: str, **params: Any) -> dict[str, Any]:
        key = api_key("fred")
        if not key:
            raise SpecificationError(NO_FRED_KEY)
        query = urllib.parse.urlencode({**params, "file_type": "json", "api_key": key})
        payload = json.loads(fetch(f"{FRED_API}/{endpoint}?{query}", kind))
        if "error_message" in payload:
            raise SpecificationError(f"FRED: {payload['error_message']}")
        return payload

    def search(self, text: str, *, fetch: Fetch, limit: int, source: str = "") -> list[SeriesInfo]:
        # FRED's own relevance order (search_rank, best first): kept as it comes, since its answer gives no score
        payload = self._get(fetch, "series/search", "search", search_text=text, limit=min(limit, 1000), order_by="search_rank",
                            sort_order="desc")
        return [_fred_series(s) for s in payload.get("seriess", [])][:limit]

    def browse(self, path: str, *, fetch: Fetch) -> Listing:
        category = path.rpartition("/")[2] or "0"  # "category/32991" or "32991"; "" is the top
        if not category.isdigit():
            raise SpecificationError(f"a FRED category is a number, e.g. 32991, not {path!r}")
        children = self._get(fetch, "category/children", "metadata", category_id=category).get("categories", [])
        series = self._get(fetch, "category/series", "metadata", category_id=category, limit=1000).get("seriess", []) if category != "0" else []
        return Listing(tuple(Category("fred", str(c["id"]), c.get("name", ""), c.get("notes", "") or "") for c in children),
                       tuple(_fred_series(s) for s in series))


# ---------------------------------------------------------------------------------------------------------------------
# World Bank

WB_API = "https://api.worldbank.org/v2"


def _wb_rows(fetch: Fetch, url: str, kind: str = "metadata") -> list[dict[str, Any]]:
    """Every row of a paged World Bank answer ([header, rows])."""
    rows: list[dict[str, Any]] = []
    page, pages = 1, 1
    while page <= pages:
        payload = json.loads(fetch(f"{url}{'&' if '?' in url else '?'}page={page}", kind))
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            message = payload[0].get("message", [{}])[0].get("value") if isinstance(payload, list) and payload else None
            if isinstance(payload, list) and payload and "message" not in payload[0]:
                return rows  # an empty list
            raise SpecificationError(f"the World Bank: {message or 'nothing found'} ({url.rpartition('/v2/')[2]})")
        pages = int(payload[0].get("pages", 1) or 1)
        rows += payload[1]
        page += 1
    return rows


WB_GROUP_AT = 40  # a source's or topic's list of more indicators than this is sub-grouped by its codes' first segment


class WorldBankSearch:
    transport = staticmethod(_transport)

    def _listing(self, fetch: Fetch, url: str, path: str, prefix: str) -> Listing:
        """The indicators at ``url``: those whose codes start ``prefix.`` (any, for ""); a long list (more than
        WB_GROUP_AT, of two or more prefixes) also gives a place under ``path`` for each prefix (``topic/3/NY``)."""
        found = self._indicators(fetch, url)
        if prefix:
            return Listing(series=tuple(s for s in found if code_prefix(s.code).upper() == prefix.upper()))
        groups = code_prefix_groups(s.code for s in found)
        places = tuple(Category("worldbank", f"{path}/{p}", title, f"{len(codes)} indicators whose codes start {p}.", leaf=True)
                       for p, title, codes in groups) if len(found) > WB_GROUP_AT and len(groups) > 1 else ()
        return Listing(places, tuple(found))

    def _sources(self, fetch: Fetch) -> dict[str, dict[str, Any]]:
        return {str(s["id"]): s for s in _wb_rows(fetch, f"{WB_API}/sources?format=json&per_page=1000")}

    def _indicators(self, fetch: Fetch, url: str) -> list[SeriesInfo]:
        sources = self._sources(fetch)
        out = []
        for r in _wb_rows(fetch, url):
            src = r.get("source") or {}
            updated = sources.get(str(src.get("id", "")), {}).get("lastupdated", "")
            topics = ", ".join(t.get("value", "").strip() for t in r.get("topics") or [] if t.get("value"))
            notes = (r.get("sourceNote") or "").strip()
            org = (r.get("sourceOrganization") or "").strip()
            notes = "\n\n".join(p for p in (notes, f"Source: {org}" if org else "", f"Topics: {topics}" if topics else "") if p)
            out.append(SeriesInfo("worldbank", r["id"], title=r.get("name", ""), units=r.get("unit", "") or "", last_updated=updated,
                                  notes=notes, source=src.get("value", "")))
        return out

    def _source_url(self, source: str) -> str:
        source = source.strip() or "2"
        if source.lower() == "all":
            return f"{WB_API}/indicator?format=json&per_page=20000"
        return f"{WB_API}/source/{urllib.parse.quote(source)}/indicator?format=json&per_page=20000"

    def search(self, text: str, *, fetch: Fetch, limit: int, source: str = "") -> list[SeriesInfo]:
        found = self._indicators(fetch, self._source_url(source))
        return rank(found, text, lambda s: (f"{s.title} {s.code}", s.notes), limit)

    def browse(self, path: str, *, fetch: Fetch) -> Listing:
        head, _, rest = path.partition("/")
        if not head:
            return Listing((Category("worldbank", "sources", "Sources", "databases: the World Development Indicators and others"),
                            Category("worldbank", "topics", "Topics", "indicators by subject"),
                            Category("worldbank", "countries", "Countries", "economies and aggregates, by ISO3 code (Options: country=...)")))
        if head == "sources" and not rest:
            return Listing(tuple(Category("worldbank", f"source/{s['id']}", s.get("name", ""), f"{s.get('code', '')}, updated {s.get('lastupdated', '')}")
                                 for s in self._sources(fetch).values()))
        if head == "source" and rest:
            source, _, prefix = rest.partition("/")
            return self._listing(fetch, self._source_url(source), f"source/{source}", prefix)
        if head == "topics" and not rest:
            rows = _wb_rows(fetch, f"{WB_API}/topic?format=json&per_page=1000")
            return Listing(tuple(Category("worldbank", f"topic/{t['id']}", t.get("value", "").strip(), (t.get("sourceNote") or "").strip())
                                 for t in rows))
        if head == "topic" and rest:
            topic, _, prefix = rest.partition("/")
            return self._listing(fetch, f"{WB_API}/topic/{urllib.parse.quote(topic)}/indicator?format=json&per_page=20000",
                                 f"topic/{topic}", prefix)
        if head == "countries" and not rest:
            rows = _wb_rows(fetch, f"{WB_API}/country?format=json&per_page=1000")
            return Listing(tuple(Category("worldbank", f"country/{c['id']}", c.get("name", ""), _country_note(c), leaf=True) for c in rows))
        raise NotFoundError(f"nothing at {path!r} on the World Bank: browse sources, source/2, topics, topic/3, topic/3/NY or countries")


def _country_note(c: dict[str, Any]) -> str:
    region = ((c.get("region") or {}).get("value") or "").strip()
    if region == "Aggregates":
        return "an aggregate"
    income = ((c.get("incomeLevel") or {}).get("value") or "").strip()
    return ", ".join(p for p in (region, income) if p)


# ---------------------------------------------------------------------------------------------------------------------
# SDMX

# provider -> (the REST root, the agency whose dataflows to list, the references to ask with a dataflow)
SDMX_STRUCTURES: dict[str, tuple[str, str, str]] = {
    "ECB": ("https://data-api.ecb.europa.eu/service", "ECB", "all"),
    "ESTAT": ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1", "ESTAT", "descendants"),
    "OECD": ("https://sdmx.oecd.org/public/rest", "all", "all"),
    "IMF": ("https://api.imf.org/external/sdmx/2.1", "all", "all"),
    "BIS": ("https://stats.bis.org/api/v1", "BIS", "all"),
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element, name: str) -> str:
    """The English (else the first) ``Name`` or ``Description`` of an SDMX artefact."""
    found = [c for c in element if _local(c.tag) == name]
    for c in found:
        if c.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "en") == "en":
            return " ".join((c.text or "").split())
    return " ".join((found[0].text or "").split()) if found else ""


def _sdmx_error(root: ET.Element) -> None:
    if _local(root.tag) == "Error":
        message = " ".join(" ".join((t.text or "").split()) for t in root.iter() if _local(t.tag) == "Text")
        raise SpecificationError(f"SDMX: {message or 'the service returned an error'}")


class SdmxSearch:
    transport = staticmethod(_transport)

    def _provider(self, name: str) -> tuple[str, str, str, str]:
        provider = _conn._ALIASES.get(name.strip().upper(), name.strip().upper())
        if not provider:
            raise SpecificationError(f"choose an SDMX provider: {', '.join(SDMX_STRUCTURES)}")
        if provider not in SDMX_STRUCTURES:
            if provider in _conn._PROVIDERS:
                raise SpecificationError(f"{provider}'s dataflows cannot be listed: register it with structure_url= (its REST root)")
            raise NotFoundError(f"unknown SDMX provider {provider!r}; known: {', '.join(SDMX_STRUCTURES)}")
        base, agency, refs = SDMX_STRUCTURES[provider]
        return provider, base, agency, refs

    def dataflows(self, provider: str, fetch: Fetch) -> list[SeriesInfo]:
        provider, base, agency, _ = self._provider(provider)
        root = ET.fromstring(fetch(f"{base}/dataflow/{agency}", "metadata"))
        _sdmx_error(root)
        out = []
        for flow in root.iter():
            if _local(flow.tag) != "Dataflow" or "id" not in flow.attrib:
                continue
            owner = flow.attrib.get("agencyID", "")
            code = flow.attrib["id"] if agency != "all" and owner == agency else f"{owner},{flow.attrib['id']}"
            out.append(SeriesInfo("sdmx", f"{provider}/{code}", title=_text(flow, "Name"), notes=_text(flow, "Description"),
                                  source=provider, kind="dataflow"))
        return out

    def search(self, text: str, *, fetch: Fetch, limit: int, source: str = "") -> list[SeriesInfo]:
        if not source.strip():
            raise SpecificationError(f"choose the SDMX provider to search (source=): {', '.join(SDMX_STRUCTURES)}")
        flows = self.dataflows(source, fetch)
        return rank(flows, text, lambda s: (f"{s.title} {s.code}", s.notes), limit)

    def browse(self, path: str, *, fetch: Fetch) -> Listing:
        if not path:
            return Listing(tuple(Category("sdmx", p, p, base) for p, (base, _, _) in SDMX_STRUCTURES.items()))
        if "/" in path:
            raise SpecificationError(f"browse a provider ({', '.join(SDMX_STRUCTURES)}); a dataflow's codes are in structure()")
        return Listing(tuple(Category("sdmx", s.code, s.title, s.notes, leaf=True) for s in self.dataflows(path, fetch)))

    def categories(self, provider: str, fetch: Fetch) -> Any:
        """The provider's category schemes and categorisations (``browse_folders.Categories``), kept a week."""
        provider, base, agency, _ = self._provider(provider)
        root = ET.fromstring(fetch(f"{base}/categoryscheme/{agency}/all/latest?references=parentsandsiblings", "metadata"))
        _sdmx_error(root)
        return parse_categories(root)

    def tree(self, provider: str, fetch: Fetch, flows: list[tuple[str, str]] | None = None) -> DataflowTree:
        """The provider's dataflows (``flows``, as (code, name), else asked for) in folders (``browse_folders``): its
        own categories where they are useful, else grouped by name or ID prefix. A category request that fails (an
        error, a provider with none, an answer that is not SDMX) only means another grouping, said in ``why``."""
        name, _, agency, _ = self._provider(provider)
        if flows is None:
            flows = [(s.code, s.title) for s in self.dataflows(name, fetch)]
        try:
            categories, why = self.categories(name, fetch), ""
        except (PyMetrixError, ConnectionError, ET.ParseError, ValueError) as exc:
            categories, why = None, str(exc.args[0]) if exc.args else type(exc).__name__
        return choose_tree(flows, categories, "" if agency == "all" else agency, why)

    def structure(self, code: str, *, fetch: Fetch) -> Structure:
        name, _, flow = code.partition("/")
        if not flow or "/" in flow:
            raise SpecificationError(f"give PROVIDER/FLOW, e.g. ECB/EXR, not {code!r}")
        provider, base, agency, refs = self._provider(name)
        parts = flow.split(",")  # ID, AGENCY,ID or AGENCY,ID,VERSION
        owner, flow_id = (agency, parts[0]) if len(parts) == 1 else (parts[0], parts[1])
        root = ET.fromstring(fetch(f"{base}/dataflow/{owner}/{flow_id}/latest?references={refs}", "metadata"))
        _sdmx_error(root)
        elements = list(root.iter())
        dataflow = next((e for e in elements if _local(e.tag) == "Dataflow" and e.attrib.get("id", "").upper() == flow_id.upper()), None)
        if dataflow is None:
            raise NotFoundError(f"{provider} has no dataflow {flow!r}")
        dsd_ref = reference(dataflow, "Structure")
        dsds = [e for e in elements if _local(e.tag) == "DataStructure" and "id" in e.attrib]
        dsd = next((e for e in dsds if dsd_ref is not None and e.attrib["id"] == dsd_ref.id), dsds[0] if dsds else None)
        if dsd is None:
            raise SpecificationError(f"{provider} sent no data structure for {flow}")

        def codelist(ref: Reference) -> bytes:  # a code list the answer left out (step 4 of sdmx_structure)
            return fetch(f"{base}/codelist/{ref.agency or owner}/{ref.id}/{ref.version or 'latest'}", "metadata")

        out = dimensions(root, Reference(dataflow.attrib.get("agencyID", owner), flow_id), dsd, dsd_ref, codelist)
        return Structure(provider, flow, _text(dataflow, "Name"), out, _text(dataflow, "Description"))


def dataflow_tree(provider: str, flows: list[tuple[str, str]] | None = None, *, refresh: bool = False) -> DataflowTree:
    """An SDMX provider's dataflows in folders, as File → Browse series… shows them (``browse_folders.choose_tree``):
    ``dataflow_tree("ECB").grouping`` is "categories", "names", "prefixes" or "flat". ``flows`` are (code, name) pairs
    already listed (``connectors.browse("sdmx", "ECB")``), else asked for. Answers are kept a week, as other metadata."""
    from .. import connectors as api

    searcher = api._searcher("sdmx")
    if not hasattr(searcher, "tree"):  # a plugin's own SDMX searcher: its dataflows by name
        return choose_tree(flows or [], None, why="the SDMX searcher gives no categories")
    return searcher.tree(provider, api._fetcher("sdmx", searcher, refresh), flows)


register_searcher("fred", FredSearch())
register_searcher("worldbank", WorldBankSearch())
register_searcher("sdmx", SdmxSearch())
