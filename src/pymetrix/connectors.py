"""Finding series on data services: search, browse and build SDMX keys, with what was found kept on disk for a while.

    from pymetrix import connectors
    connectors.search("worldbank", "gdp per capita")           # World Development Indicators (source=2) by default
    connectors.search("sdmx", "exchange rates", source="ECB")   # the provider's dataflows
    connectors.search("fred", "unemployment rate")             # needs a FRED API key: FRED_API_KEY or set_api_key
    connectors.browse("fred", "32991")                         # a category: its subcategories and its series
    connectors.browse("worldbank", "topic/3")                  # sources, topics and countries
    s = connectors.structure("sdmx", "ECB/EXR")                # a dataflow's dimensions and their codes
    s.address(FREQ="M", CURRENCY=["USD", "GBP"])               # "sdmx://ECB/EXR/M.USD+GBP..."
    Dataset.read(results[0].address)                           # what was found, read as any address is

Every connector that can be searched says so in its description (``pymetrix.io.connectors()[scheme].search``) and
answers with the same :class:`SeriesInfo` (code, title, frequency, units, first and last period, last update, notes
where the provider gives them), so the Series browser (File → Browse series…) shows every provider the same way.

Searches and metadata (category lists, indicator lists, dataflows, data structures) are kept in
``~/.pymetrix/cache/connectors`` (or under ``$PYMETRIX_HOME``) for a time to live (a day for searches, a week for
metadata), so browsing is quick and does not ask the provider twice; ``refresh=True`` asks again. Requests are kept
by their address with any credential (an API key) taken out, and only the provider's answer is kept: a key is never
written to disk. Searching is a lookup, not a step of the analysis, so it is not recorded; the data read with what
was found are recorded as ``Dataset.read(address)``, which needs neither the browser nor the cache to replay.

Plugins add a searcher for their own scheme with :func:`register_searcher`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import NotFoundError, SpecificationError

__all__ = ["Category", "Dimension", "Listing", "Searcher", "SeriesInfo", "Structure", "api_key", "browse", "cache_dir", "cache_entries",
           "clear_cache", "configure_cache", "rank", "redact", "register_searcher", "searchable", "search", "set_api_key", "structure",
           "to_frame"]

# -- what a search finds ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesInfo:
    """A series (or, for SDMX, a dataflow: ``kind="dataflow"``) as a provider describes it. ``code`` is what goes after
    ``provider://`` in the address; fields the provider does not give are empty."""

    provider: str
    code: str
    title: str = ""
    frequency: str = ""
    units: str = ""
    start: str = ""
    end: str = ""
    last_updated: str = ""
    notes: str = ""
    seasonal_adjustment: str = ""
    source: str = ""
    kind: str = "series"

    @property
    def address(self) -> str:
        """The address ``Dataset.read`` takes for this series (for a dataflow, every series of it: ``.../all``)."""
        return f"{self.provider}://{self.code}" + ("/all" if self.kind == "dataflow" else "")


@dataclass(frozen=True)
class Category:
    """A place to browse into: ``browse(provider, path)``. A ``leaf`` has nothing under it (a World Bank country, an
    SDMX dataflow: see :func:`structure`)."""

    provider: str
    path: str
    title: str
    notes: str = ""
    leaf: bool = False


@dataclass(frozen=True)
class Listing:
    """What is in a place: the places under it and the series in it."""

    categories: tuple[Category, ...] = ()
    series: tuple[SeriesInfo, ...] = ()


@dataclass(frozen=True)
class Dimension:
    """One position of an SDMX series key, with its codes as (code, name) pairs, the codes' longer
    ``descriptions`` where the provider gives them, and a ``note`` when its codes could not all be found (why;
    codes can still be typed)."""

    id: str
    name: str = ""
    codes: tuple[tuple[str, str], ...] = ()
    descriptions: tuple[tuple[str, str], ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Structure:
    """An SDMX dataflow's data structure: the dimensions of its series keys, in order. ``key(FREQ="M",
    CURRENCY=["USD", "GBP"])`` writes the key (``M.USD+GBP...``: a dimension not chosen is any code)."""

    provider: str
    flow: str
    title: str = ""
    dimensions: tuple[Dimension, ...] = ()
    notes: str = ""

    def dimension(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.id == name:
                return d
        raise NotFoundError(f"no dimension {name!r} in {self.flow}; dimensions: {', '.join(d.id for d in self.dimensions)}")

    def key(self, choices: Mapping[str, str | Iterable[str]] | None = None, **more: str | Iterable[str]) -> str:
        """The series key for the codes chosen per dimension (several joined by ``+``; none means any code)."""
        chosen = {**(choices or {}), **more}
        for name in chosen:
            self.dimension(name)
        parts = []
        for d in self.dimensions:
            value = chosen.get(d.id) or ()
            codes = [value] if isinstance(value, str) else list(value)
            codes = [c for c in codes if c]
            known = {c for c, _ in d.codes}
            bad = [c for c in codes if known and c not in known]
            if bad:
                raise SpecificationError(f"{d.id} has no code {bad[0]!r}; e.g. {', '.join(sorted(known)[:6])}")
            parts.append("+".join(codes))
        return ".".join(parts)

    def address(self, choices: Mapping[str, str | Iterable[str]] | None = None, **more: str | Iterable[str]) -> str:
        """The ``sdmx://PROVIDER/FLOW/KEY`` address of the series chosen."""
        return f"sdmx://{self.provider}/{self.flow}/{self.key(choices, **more)}"


# -- searchers ---------------------------------------------------------------------------------------------------------

Fetch = Callable[[str, str], bytes]  # (url, "search" or "metadata") -> the provider's answer, through the cache


class Searcher(Protocol):
    """What a connector's searcher provides. ``fetch(url, kind)`` gets a URL through the cache (``kind`` is "search"
    or "metadata", which sets the time to live); ``secrets`` lists query parameters never kept (``api_key``)."""

    def search(self, text: str, *, fetch: Fetch, limit: int, source: str) -> list[SeriesInfo]: ...

    def browse(self, path: str, *, fetch: Fetch) -> Listing: ...


_SEARCHERS: dict[str, Any] = {}
_KEYS: dict[str, str] = {}  # API keys given in this session (never written anywhere)


def register_searcher(scheme: str, searcher: Any) -> None:
    """Make ``scheme`` searchable: ``searcher`` has ``search(text, *, fetch, limit, source)`` and ``browse(path, *,
    fetch)`` (and, optionally, ``structure(code, *, fetch)``, ``transport(url)`` to fetch a URL, and ``secrets``, the
    query parameters that carry credentials). Its connector's description should say ``search=True``."""
    _SEARCHERS[scheme.lower()] = searcher


def _searcher(provider: str) -> Any:
    from .io import connectors

    connectors()  # loads the plugins (and the bundled connectors)
    key = provider.strip().lower()
    if key not in _SEARCHERS:
        searchable_ = ", ".join(sorted(_SEARCHERS)) or "none"
        raise NotFoundError(f"the {key} connector cannot be searched; searchable: {searchable_}")
    return _SEARCHERS[key]


def searchable() -> list[str]:
    """The schemes that can be searched and browsed."""
    from .io import connectors

    connectors()
    return sorted(_SEARCHERS)


def set_api_key(provider: str, key: str | None) -> None:
    """Give a provider's API key for this session (FRED's search needs one). It is kept in memory only: never saved,
    never in the cache, never in a script. ``FRED_API_KEY`` in the environment does the same for every session."""
    if key:
        _KEYS[provider.lower()] = key.strip()
    else:
        _KEYS.pop(provider.lower(), None)


def api_key(provider: str) -> str:
    """The API key for ``provider``: given with :func:`set_api_key`, or ``<PROVIDER>_API_KEY`` in the environment; ""
    when there is none."""
    return _KEYS.get(provider.lower()) or os.environ.get(f"{provider.upper()}_API_KEY", "").strip()


# -- the cache ---------------------------------------------------------------------------------------------------------

_TTL = {"search": 24 * 3600.0, "metadata": 7 * 24 * 3600.0}
_ENABLED = {"on": True}
SECRETS = ("api_key", "apikey", "token", "access_token", "subscription-key", "key_id", "password")


def configure_cache(*, search_ttl: float | None = None, metadata_ttl: float | None = None, enabled: bool | None = None) -> dict[str, Any]:
    """Change how long searches and metadata are kept (seconds), or turn the cache off; returns the settings."""
    if search_ttl is not None:
        _TTL["search"] = float(search_ttl)
    if metadata_ttl is not None:
        _TTL["metadata"] = float(metadata_ttl)
    if enabled is not None:
        _ENABLED["on"] = bool(enabled)
    return {"search_ttl": _TTL["search"], "metadata_ttl": _TTL["metadata"], "enabled": _ENABLED["on"], "folder": str(cache_dir())}


def cache_dir() -> Path:
    """Where searches and metadata are kept: ``cache/connectors`` in ``~/.pymetrix`` (or ``$PYMETRIX_HOME``)."""
    base = os.environ.get("PYMETRIX_HOME") or os.path.join(Path.home(), ".pymetrix")
    return Path(base) / "cache" / "connectors"


def redact(url: str, secrets: Iterable[str] = ()) -> str:
    """The URL with every credential parameter taken out (what the cache keeps and errors show)."""
    names = {s.lower() for s in (*SECRETS, *secrets)}
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in names]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _entry_path(provider: str, request: str) -> Path:
    digest = hashlib.sha256(request.encode("utf-8")).hexdigest()[:40]
    return cache_dir() / f"{re.sub(r'[^a-z0-9]+', '_', provider.lower())}-{digest}.json"


def _cached_fetch(provider: str, url: str, kind: str, *, refresh: bool, transport: Callable[[str], bytes],
                  secrets: Iterable[str] = ()) -> bytes:
    secrets = tuple(secrets)
    request = redact(url, secrets)
    path = _entry_path(provider, request)
    ttl = _TTL.get(kind, _TTL["metadata"])
    if _ENABLED["on"] and not refresh and path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("request") == request and time.time() - float(entry["stored"]) < ttl:
                return entry["body"].encode("utf-8")
        except (OSError, ValueError, KeyError, TypeError):
            pass  # unreadable: ask again
    try:
        body = transport(url)
    except ConnectionError as exc:  # its message may hold the URL: never with the key
        message = str(exc)
        for name in (*SECRETS, *secrets):
            message = re.sub(rf"({re.escape(name)}=)[^&\s)]+", r"\1***", message, flags=re.I)
        raise ConnectionError(message) from None
    if _ENABLED["on"]:
        text = body.decode("utf-8", "replace")
        names = {n.lower() for n in (*SECRETS, *secrets)}
        hidden = [v for k, v in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query) if k.lower() in names and len(v) > 3]
        for value in [*hidden, *(v for v in _KEYS.values() if len(v) > 3)]:  # an answer that echoes the key
            text = text.replace(value, "***")
        entry = {"provider": provider, "kind": kind, "request": request, "stored": time.time(),
                 "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ttl": ttl, "body": text}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(entry), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # a cache that cannot be written is only slower
    return body


def cache_entries(provider: str | None = None) -> list[dict[str, Any]]:
    """What is kept: each request (without credentials), its provider, kind, when it was stored and whether it has
    expired."""
    out = []
    folder = cache_dir()
    for path in sorted(folder.glob("*.json")) if folder.exists() else ():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if provider and entry.get("provider") != provider.lower():
            continue
        age = time.time() - float(entry.get("stored", 0))
        out.append({"provider": entry.get("provider"), "kind": entry.get("kind"), "request": entry.get("request"),
                    "stored_at": entry.get("stored_at"), "expired": age >= _TTL.get(entry.get("kind", ""), _TTL["metadata"]),
                    "file": str(path)})
    return out


def clear_cache(provider: str | None = None) -> int:
    """Forget what is kept (one provider's, or all); returns how many entries went."""
    gone = 0
    for entry in cache_entries(provider):
        try:
            Path(entry["file"]).unlink()
            gone += 1
        except OSError:
            pass
    return gone


def _fetcher(provider: str, searcher: Any, refresh: bool) -> Fetch:
    def default_transport(url: str) -> bytes:
        from .ext import connectors as ext

        return ext._get(url)

    transport = getattr(searcher, "transport", None) or default_transport
    secrets = tuple(getattr(searcher, "secrets", ()))

    def fetch(url: str, kind: str = "metadata") -> bytes:
        return _cached_fetch(provider, url, kind, refresh=refresh, transport=transport, secrets=secrets)

    return fetch


# -- the API -----------------------------------------------------------------------------------------------------------


def search(provider: str, text: str, *, limit: int = 50, source: str = "", refresh: bool = False) -> list[SeriesInfo]:
    """Series (or, for SDMX, dataflows) whose title or code match ``text``, best first, at most ``limit``.

    ``source`` narrows where to look: the SDMX provider (ECB, ESTAT, OECD, IMF, BIS or one registered), the World
    Bank source ("2", the World Development Indicators, by default; "all" for every indicator). ``refresh`` asks the
    provider again rather than using what was kept."""
    if not str(text).strip():
        raise SpecificationError("give words to search for, e.g. unemployment rate")
    if int(limit) < 1:
        raise SpecificationError("limit is at least 1")
    key = provider.strip().lower()
    searcher = _searcher(key)
    return list(searcher.search(str(text).strip(), fetch=_fetcher(key, searcher, refresh), limit=int(limit), source=source))


def browse(provider: str, path: str = "", *, refresh: bool = False) -> Listing:
    """What is at ``path`` ("" for the top): the places under it and the series in it. FRED: category IDs; World
    Bank: ``sources``, ``source/2``, ``topics``, ``topic/3``, ``countries``; SDMX: a provider, then its dataflows."""
    key = provider.strip().lower()
    searcher = _searcher(key)
    return searcher.browse(str(path).strip().strip("/"), fetch=_fetcher(key, searcher, refresh))


def structure(provider: str, code: str, *, refresh: bool = False) -> Structure:
    """The dimensions (and their codes) of a dataflow, for writing series keys: ``structure("sdmx", "ECB/EXR")``."""
    key = provider.strip().lower()
    searcher = _searcher(key)
    if not hasattr(searcher, "structure"):
        raise NotFoundError(f"the {key} connector has no data structures (its codes are whole series IDs)")
    return searcher.structure(str(code).strip().strip("/"), fetch=_fetcher(key, searcher, refresh))


def to_frame(found: Iterable[SeriesInfo]) -> Any:
    """Search results as a table (one row per series), for a quick look or a notebook."""
    import pandas as pd

    rows = [{"code": s.code, "title": s.title, "frequency": s.frequency, "units": s.units, "start": s.start, "end": s.end,
             "last_updated": s.last_updated, "address": s.address} for s in found]
    return pd.DataFrame(rows, columns=["code", "title", "frequency", "units", "start", "end", "last_updated", "address"])


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


def rank(items: Iterable[Any], text: str, fields: Callable[[Any], tuple[str, str]], limit: int) -> list[Any]:
    """Items whose (title and code, notes) hold every word of ``text``: those matching in the title or code first, then
    in the notes; the provider's order within each. For searchers that filter a list locally (World Bank, SDMX)."""
    words = _words(text)
    first, second = [], []
    for item in items:
        head, notes = fields(item)
        head_l = head.lower()
        if all(w in head_l for w in words):
            first.append(item)
        elif all(w in head_l or w in notes.lower() for w in words):
            second.append(item)
    return (first + second)[:limit]

