"""Folders for File → Browse series…: an SDMX provider's dataflows in its own category tree (or grouped by name or ID
prefix), and a long World Bank indicator list grouped by the first segment of its codes.

**SDMX.** A provider publishes its *category schemes* (folders, nested) and *categorisations* (each puts one dataflow in
one category) as SDMX 2.1 structures; :func:`parse_categories` reads them and :func:`category_tree` puts the provider's
dataflows in the folders. Categories with no dataflows (at any depth) are left out; a dataflow in two categories is in
both; dataflows in none go in a last folder, "Other". The categories of the scheme holding most dataflows are the top
folders; any other scheme (Eurostat's cross-cutting topics) is one folder after them; and a single top folder holding
only folders is opened (its contents become the top).

:func:`choose_tree` decides which grouping the browser shows. In order, the first that is **useful** is taken:

1. the provider's categories (:func:`category_tree`);
2. the families of the dataflows' names (:func:`name_tree`, ``pymetrix.ext.dataflow_groups``);
3. the first underscore-separated part of the dataflow's ID (:func:`prefix_tree`, Eurostat's ``nama_``, ``prc_``...).

A grouping is useful (:func:`judge`) when it has at least two top folders (not counting "Other"), its folders (not
"Other") hold at least half the dataflows, and no top folder holds more than four fifths of them. So a scheme that is
missing, could not be read, categorises few dataflows, or puts (nearly) everything in one folder gives way to the names,
as the IMF's must not be worse than its name families. When none is useful, the names or the prefixes are used if they
make any folder (whichever puts more dataflows in folders; the names on a tie, as before), else a flat list sorted by
name. Folders and rows are sorted by name, ignoring case, except categories, which keep the provider's order.

**World Bank.** :func:`code_prefix_groups` groups indicators by the first segment of their codes (``NY.GDP.MKTP.CD``
is under ``NY``), named from a small map (``NY``: "National accounts: income and GDP"), else by the prefix itself.

Nothing here touches Qt or the network.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .dataflow_groups import GroupedDataflow, group_dataflows
from .sdmx_structure import _URN, _child, _local, _text

__all__ = ["Categories", "Category", "DataflowTree", "Folder", "category_tree", "choose_tree", "code_prefix",
           "code_prefix_groups", "flat_tree", "flow_id", "flow_key", "id_prefix", "judge", "name_tree", "parse_categories", "prefix_tree"]

OTHER = "Other"
NOTES = {
    "categories": "Grouped by the provider's categories",
    "names": "Grouped by name",
    "prefixes": "Grouped by the start of the dataflow's ID",
    "flat": "Sorted by name",
}
HEADERS = {"categories": "by category", "names": "by name", "prefixes": "by ID prefix", "flat": "by name"}

# Eurostat's dataset codes start with a theme (nama_10_gdp, prc_hicp_manr); the commoner ones, named
EUROSTAT_THEMES = {
    "agr": "Agriculture", "apro": "Agricultural production", "bop": "Balance of payments", "demo": "Population",
    "earn": "Earnings", "educ": "Education", "ei": "Euro indicators", "env": "Environment", "ert": "Exchange rates",
    "ext": "International trade", "gov": "Government finance", "hlth": "Health", "ilc": "Income and living conditions",
    "irt": "Interest rates", "isoc": "Information society", "lc": "Labour costs", "lfsa": "Labour force survey, annual",
    "lfsi": "Labour market, indicators", "lfsq": "Labour force survey, quarterly", "migr": "Migration",
    "nama": "National accounts", "namq": "National accounts, quarterly", "nasa": "Sector accounts, annual",
    "nasq": "Sector accounts, quarterly", "nrg": "Energy", "prc": "Prices", "rd": "Research and development",
    "sts": "Short-term business statistics", "tour": "Tourism", "tran": "Transport", "une": "Unemployment",
}

# the World Development Indicators' code prefixes (the first segment of NY.GDP.MKTP.CD), named
WORLD_BANK_THEMES = {
    "AG": "Agriculture and land", "BG": "Balance of payments: goods and services", "BM": "Balance of payments: payments",
    "BN": "Balance of payments: net", "BX": "Balance of payments: receipts", "CM": "Capital markets",
    "DT": "External debt", "EG": "Energy", "EN": "Environment: emissions", "ER": "Environment: land and water",
    "FB": "Banking", "FD": "Financial depth", "FI": "Reserves", "FM": "Money", "FP": "Prices",
    "FR": "Interest rates", "FS": "Financial sector", "GB": "Research and development", "GC": "Government finance",
    "IC": "Business environment", "IE": "Infrastructure investment", "IP": "Intellectual property",
    "IQ": "Statistical capacity", "IS": "Transport", "IT": "Information and communications", "LP": "Logistics",
    "MS": "Military", "NE": "National accounts: expenditure", "NV": "National accounts: value added",
    "NY": "National accounts: income and GDP", "PA": "Exchange rates and prices", "SE": "Education", "SG": "Gender",
    "SH": "Health", "SI": "Poverty and income distribution", "SL": "Labour", "SM": "Migration", "SN": "Nutrition",
    "SP": "Population", "ST": "Tourism", "TG": "Trade: goods", "TM": "Trade: imports", "TX": "Trade: exports",
    "VC": "Violence and conflict",
}


# -- the tree the browser shows ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Folder:
    """A folder: its ``title``, ``notes`` (a category's description) and ``children``, folders and dataflows
    (:class:`~pymetrix.ext.dataflow_groups.GroupedDataflow`) in the order shown."""

    title: str
    children: tuple[Folder | GroupedDataflow, ...] = ()
    notes: str = ""

    @property
    def folders(self) -> list[Folder]:
        return [c for c in self.children if isinstance(c, Folder)]

    @property
    def flows(self) -> list[GroupedDataflow]:
        return [c for c in self.children if isinstance(c, GroupedDataflow)]

    def codes(self) -> set[str]:
        """The codes of every dataflow in the folder, at any depth."""
        out = {f.code for f in self.flows}
        for sub in self.folders:
            out |= sub.codes()
        return out


@dataclass(frozen=True)
class DataflowTree:
    """What the browser shows: the ``root``'s children are the top folders and rows. ``grouping`` is "categories",
    "names", "prefixes" or "flat"; ``note`` says so in words; ``why`` says why a better grouping was not used."""

    root: Folder
    grouping: str
    note: str = ""
    why: str = ""


def _row(code: str, name: str, label: str = "") -> GroupedDataflow:
    name = " ".join(str(name or code).split())
    return GroupedDataflow(code, name, label or name)


def _by_name(rows: Iterable[GroupedDataflow]) -> list[GroupedDataflow]:
    return sorted(rows, key=lambda r: (r.label.casefold(), r.label, r.code.casefold()))


def flat_tree(flows: Iterable[tuple[str, str]]) -> DataflowTree:
    """Every dataflow a row, sorted by name."""
    return DataflowTree(Folder("", tuple(_by_name(_row(c, n) for c, n in flows))), "flat", NOTES["flat"])


def name_tree(flows: Iterable[tuple[str, str]]) -> DataflowTree:
    """The families of the dataflows' names (``group_dataflows``): a folder for each family of two or more."""
    children: list[Folder | GroupedDataflow] = []
    for family in group_dataflows(flows):
        children.append(Folder(family.title, family.flows) if family.folder else family.flows[0])
    return DataflowTree(Folder("", tuple(children)), "names", NOTES["names"])


def flow_id(code: str) -> str:
    """A dataflow's ID from the browser's code: EXR for ECB/EXR, CPI for IMF/IMF.STA,CPI (or IMF.STA,CPI,1.0)."""
    parts = code.partition("/")[2].split(",") if "/" in code else code.split(",")
    return parts[1] if len(parts) > 1 else parts[0]


def flow_key(code: str, agency: str = "") -> tuple[str, str]:
    """(agency, ID) of a dataflow from the browser's code; ``agency`` is the provider's, for a code without one."""
    parts = code.partition("/")[2].split(",") if "/" in code else code.split(",")
    return (parts[0], parts[1]) if len(parts) > 1 else (agency, parts[0])


def id_prefix(flow: str) -> str:
    """The first underscore-separated part of a dataflow's ID, in lower case (nama for NAMA_10_GDP); "" for an ID with
    no underscore, which has no prefix."""
    head, sep, _ = flow.partition("_")
    return head.lower() if sep and head else ""


def prefix_tree(flows: Iterable[tuple[str, str]], themes: dict[str, str] | None = None) -> DataflowTree:
    """A folder for each ID prefix shared by two or more dataflows (``nama_``, ``prc_``), named by the prefix and, where
    ``themes`` (Eurostat's by default) knows it, its theme; the other dataflows are rows. Folders first, then rows."""
    themes = EUROSTAT_THEMES if themes is None else themes
    groups: dict[str, list[GroupedDataflow]] = {}
    for code, name in flows:
        groups.setdefault(id_prefix(flow_id(code)), []).append(_row(code, name))
    folders, rows = [], []
    for prefix, members in groups.items():
        if prefix and len(members) > 1:
            title = f"{prefix}_ {themes[prefix]}" if prefix in themes else f"{prefix}_"
            folders.append(Folder(title, tuple(_by_name(members)), f"dataflows whose IDs start {prefix.upper()}_"))
        else:
            rows += members
    folders.sort(key=lambda f: f.title.casefold())
    return DataflowTree(Folder("", (*folders, *_by_name(rows))), "prefixes", NOTES["prefixes"])


# -- category schemes ---------------------------------------------------------------------------------------------------


@dataclass
class Category:
    """A category of a scheme: its ``id`` (its own, not the path), name, description and sub-categories."""

    id: str
    name: str = ""
    notes: str = ""
    children: list[Category] = field(default_factory=list)


@dataclass
class Categories:
    """What a provider's category schemes say: ``schemes`` by (agency, id), each with its name and top categories;
    ``links``, one (dataflow agency, dataflow id, scheme agency, scheme id, category path) per categorisation."""

    schemes: dict[tuple[str, str], tuple[str, list[Category]]] = field(default_factory=dict)
    links: list[tuple[str, str, str, str, str]] = field(default_factory=list)


def _categories(element: ET.Element) -> list[Category]:
    return [Category(c.attrib["id"], _text(c, "Name"), _text(c, "Description"), _categories(c))
            for c in element if _local(c.tag) == "Category" and c.attrib.get("id")]


def _ref(node: ET.Element | None) -> tuple[str, str, str, str] | None:
    """(class, agency, id, maintainable parent) of a ``Ref`` or ``URN`` under ``node``; a category's id may be a path."""
    if node is None:
        return None
    ref = _child(node, "Ref")
    if ref is not None and ref.attrib.get("id"):
        a = ref.attrib
        return a.get("class", ""), a.get("agencyID", ""), a["id"], a.get("maintainableParentID", "")
    urn = _child(node, "URN")
    text = " ".join((urn.text or "").split()) if urn is not None else ""
    found = _URN.search(text)
    if found is None:
        return None
    cls = text[:found.start()].rpartition(".")[2]
    if found.group("item"):  # Category=AGENCY:SCHEME(VERSION).PATH
        return cls, found.group("agency"), found.group("item"), found.group("id")
    return cls, found.group("agency"), found.group("id"), ""


def parse_categories(message: bytes | ET.Element) -> Categories:
    """The category schemes and categorisations in an SDMX 2.1 structure message (by local tag names, so prefixes do
    not matter). Categorisations of anything but a dataflow (a data structure, a code list) are left out. Raises
    ``ET.ParseError`` for a message that is not XML."""
    root = ET.fromstring(message) if isinstance(message, bytes | str) else message
    out = Categories()
    for e in root.iter():
        tag = _local(e.tag)
        if tag == "CategoryScheme" and e.attrib.get("id"):
            out.schemes[(e.attrib.get("agencyID", ""), e.attrib["id"])] = (_text(e, "Name") or e.attrib["id"], _categories(e))
        elif tag == "Categorisation":
            source, target = _ref(_child(e, "Source")), _ref(_child(e, "Target"))
            if source is None or target is None or source[0] not in ("", "Dataflow") or target[0] not in ("", "Category"):
                continue
            out.links.append((source[1], source[2], target[1], target[3], target[2]))
    return out


def _find(categories: list[Category], path: str) -> Category | None:
    """The category at ``path`` (``07.07_1``, or its own id alone where that is unique in the scheme)."""
    level, node = categories, None
    for part in path.split("."):  # a nested category's id may itself hold no dot: try the path first
        node = next((c for c in level if c.id == part), None)
        if node is None:
            break
        level = node.children
    if node is not None:
        return node
    found: list[Category] = []

    def walk(level: list[Category]) -> None:
        for c in level:
            if c.id in (path, path.rpartition(".")[2]):
                found.append(c)
            walk(c.children)

    walk(categories)
    return found[0] if len(found) == 1 else None


def category_tree(flows: Sequence[tuple[str, str]], categories: Categories, agency: str = "") -> DataflowTree:
    """The dataflows ``(code, name)`` in the provider's categories, as the module's docstring says; ``agency`` is the
    provider's, for codes written without one (ECB/EXR)."""
    names = {code: name for code, name in flows}
    exact: dict[tuple[str, str], list[str]] = {}
    by_id: dict[str, list[str]] = {}
    for code, _ in flows:
        a, i = flow_key(code, agency)
        exact.setdefault((a.upper(), i.upper()), []).append(code)
        by_id.setdefault(i.upper(), []).append(code)
    placed: dict[int, list[str]] = {}  # id(category) -> codes, in the order categorised
    for f_agency, f_id, s_agency, s_id, path in categories.links:
        codes = exact.get((f_agency.upper(), f_id.upper())) or (by_id.get(f_id.upper(), []) if len(by_id.get(f_id.upper(), [])) == 1 else [])
        if not codes:
            continue  # a dataflow the provider does not list
        scheme = categories.schemes.get((s_agency, s_id)) or next(
            (v for (a, i), v in categories.schemes.items() if i == s_id), None)
        node = _find(scheme[1], path) if scheme else None
        if node is not None:
            for code in codes:
                if code not in placed.setdefault(id(node), []):
                    placed[id(node)].append(code)

    def build(c: Category) -> Folder | None:
        subs = [f for f in (build(x) for x in c.children) if f is not None]
        rows = _by_name(_row(code, names[code]) for code in placed.get(id(c), []))
        return Folder(c.name or c.id, (*subs, *rows), c.notes) if subs or rows else None

    schemes: list[Folder] = []
    for name, top in categories.schemes.values():
        built = [f for f in (build(c) for c in top) if f is not None]
        if built:
            schemes.append(Folder(name, tuple(built)))
    tops: list[Folder] = []
    if schemes:  # the scheme holding most dataflows gives the top folders; any other is a folder of its own after them
        main = max(schemes, key=lambda f: len(f.codes()))  # the first on a tie
        tops = [*main.folders, *(f for f in schemes if f is not main)]
    while len(tops) == 1 and not tops[0].flows:  # one folder holding the rest: open it
        tops = tops[0].folders
    inside = set().union(*(f.codes() for f in tops)) if tops else set()
    other = [_row(code, name) for code, name in flows if code not in inside]
    children: list[Folder | GroupedDataflow] = list(tops)
    if other:
        children.append(Folder(OTHER, tuple(_by_name(other)), "dataflows in none of the provider's categories"))
    return DataflowTree(Folder("", tuple(children)), "categories", NOTES["categories"])


# -- choosing ---------------------------------------------------------------------------------------------------------------


def _in_folders(tree: DataflowTree) -> set[str]:
    return set().union(*(f.codes() for f in tree.root.folders if f.title != OTHER or tree.grouping != "categories"))


def judge(tree: DataflowTree, count: int) -> str:
    """"" when ``tree`` is useful for ``count`` dataflows (the rule in the module's docstring), else why it is not."""
    tops = [f for f in tree.root.folders if not (tree.grouping == "categories" and f.title == OTHER)]
    if len(tops) < 2:
        return "fewer than two folders"
    if 2 * len(_in_folders(tree)) < count:
        return "its folders hold fewer than half the dataflows"
    if 5 * max(len(f.codes()) for f in tops) > 4 * count:
        return "one folder holds more than four fifths of the dataflows"
    return ""


def choose_tree(flows: Sequence[tuple[str, str]], categories: Categories | None, agency: str = "",
                why: str = "") -> DataflowTree:
    """The tree to show for a provider's dataflows ``(code, name)``: its categories, else the names' families, else
    the IDs' prefixes, else a flat list (the rule in the module's docstring). ``categories`` is None when the provider
    gave none (``why``: what went wrong)."""
    flows = list(flows)
    reasons = []
    if categories is None:
        reasons.append(f"no categories ({why})" if why else "no categories")
    elif not categories.links:
        reasons.append("the provider's categories hold no dataflows")
    else:
        tree = category_tree(flows, categories, agency)
        verdict = judge(tree, len(flows))
        if not verdict:
            return tree
        reasons.append(f"the provider's categories: {verdict}")
    names, prefixes = name_tree(flows), prefix_tree(flows)
    for tree in (names, prefixes):
        verdict = judge(tree, len(flows))
        if not verdict:
            return DataflowTree(tree.root, tree.grouping, tree.note, "; ".join(reasons))
        reasons.append(f"{HEADERS[tree.grouping]}: {verdict}")
    best = max((names, prefixes), key=lambda t: len(_in_folders(t)))  # max keeps the first (names) on a tie
    if _in_folders(best):
        return DataflowTree(best.root, best.grouping, best.note, "; ".join(reasons))
    flat = flat_tree(flows)
    return DataflowTree(flat.root, "flat", flat.note, "; ".join(reasons))


# -- World Bank indicators --------------------------------------------------------------------------------------------------


def code_prefix(code: str) -> str:
    """The first dot-separated segment of an indicator's code (NY for NY.GDP.MKTP.CD)."""
    return code.split(".", 1)[0]


def code_prefix_groups(codes: Iterable[str], themes: dict[str, str] | None = None) -> list[tuple[str, str, list[str]]]:
    """(prefix, folder name, codes) for each first segment of the ``codes``, sorted by prefix; the name is the prefix
    with its theme (``themes``, the World Bank's by default) where known, else the prefix alone (``"XY."``)."""
    themes = WORLD_BANK_THEMES if themes is None else themes
    groups: dict[str, list[str]] = {}
    for code in codes:
        groups.setdefault(code_prefix(code), []).append(code)
    return [(p, f"{p}. {themes[p.upper()]}" if p.upper() in themes else f"{p}.", groups[p])
            for p in sorted(groups, key=lambda p: (p.casefold(), p))]

