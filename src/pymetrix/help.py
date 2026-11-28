"""Help built from the extensions themselves, in three levels.

1. **Usage**: what the method, test or filter does and how to fill in its
   dialog. From the class docstring and the help line of each setting.
2. **Reference**: the wrapped library's own documentation, for classes that
   declare ``wraps = "linearmodels.iv.IVLIML"`` (and optionally ``docs``, a URL).
3. **Background**: the statistics and econometrics, through ``topics`` (ids of
   help topics such as ``"garch"``) and ``references`` (citations).

Plugins add topics with ``add_topic``. ``search`` looks through everything,
so the Help menu and the dialogs' Help buttons read the same pages.
"""

from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._api import renamed_arguments
from .errors import NotFoundError
from .forms import form_spec
from .plugins import Entry, registry

__all__ = ["Course", "HelpPage", "Note", "Topic", "add_topic", "contents", "courses", "load_course", "link_of", "unload_course", "dialog_markdown", "help_page", "link", "page_for_link", "search", "topic", "topics"]

HELP_GROUPS = ("methods", "variances", "tests", "filters", "solvers", "models", "commands")


@dataclass(frozen=True)
class Topic:
    """Background on a statistical or econometric idea, shared by the extensions that link to it."""

    id: str
    title: str
    text: str
    references: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()


_TOPICS: dict[str, Topic] = {}


# a DOI runs to the next space; old Wiley DOIs (SICI) hold ; ( ) < > themselves, so only a final . , or ; is left out
_DOI = re.compile(r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/\S+?)(?=[.,;]?(?:\s|$))", re.IGNORECASE)
_URL = re.compile(r"(?<![(<])\bhttps?://[^\s)>]+")
# a JSTOR stable ID, for an article with no DOI: digits, or JSTOR's j.ctt... form for books and their chapters
_JSTOR = re.compile(r"\bjstor:\s*(\d+|j\.[a-z0-9]+(?:\.\d+)?)(?=[.,;]?(?:\s|$))", re.IGNORECASE)


def reference_markdown(citation: str) -> str:
    """A citation as a list item with a link: its DOI through doi.org, which keeps resolving when a publisher moves
    the article; for an article with no DOI, its JSTOR stable URL (``jstor:1827924``); a web address it already gives;
    otherwise a search for the citation on Google Scholar, which needs no
    upkeep and finds the article and its free copies."""
    from urllib.parse import quote, quote_plus

    found = _DOI.search(citation)
    if found:
        doi = found.group(1)
        url = "https://doi.org/" + quote(doi, safe="/:()")  # < > and ; escaped; balanced ( ) are fine in a link
        return "- " + citation[: found.start()] + f"[doi:{doi}]({url})" + citation[found.end():]
    found = _JSTOR.search(citation)
    if found:
        stable = found.group(1)
        return "- " + citation[: found.start()] + f"[jstor:{stable}](https://www.jstor.org/stable/{stable})" + citation[found.end():]
    if _URL.search(citation):
        return "- " + _URL.sub(lambda m: f"<{m.group(0)}>", citation)
    return f"- {citation} ([search](https://scholar.google.com/scholar?q={quote_plus(citation)}))"


def add_topic(id: str, title: str, text: str, references: tuple[str, ...] = (), see_also: tuple[str, ...] = ()) -> Topic:
    _TOPICS[id] = Topic(id, title, inspect.cleandoc(text), tuple(references), tuple(see_also))
    return _TOPICS[id]


def topic(id: str) -> Topic:
    try:
        return _TOPICS[id]
    except KeyError:
        raise NotFoundError(f"no help topic {id!r}; topics: {sorted(_TOPICS)}") from None


def topics() -> list[Topic]:
    return sorted(_TOPICS.values(), key=lambda t: t.title)


SCHEME = "pymetrix:"


def link(kind: str, key: str = "") -> str:
    """A help link: ``pymetrix:topic/garch``, ``pymetrix:methods/linearmodels:liml``,
    ``pymetrix:reference/methods/linearmodels:liml`` or ``pymetrix:home``."""
    return f"{SCHEME}{kind}/{key}" if key else f"{SCHEME}{kind}"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


@dataclass(frozen=True)
class HelpPage:
    title: str
    summary: str
    usage: str
    settings: list[tuple[str, str, Any, bool]] = field(default_factory=list)  # name, help, default, advanced
    plugin: str = ""
    wraps: str = ""
    docs: str = ""
    references: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    group: str = ""  # registry group and qualified name, for links
    key: str = ""
    choices: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()  # dialogs: (setting, ((title, link), ...))
    mapping: str = ""  # for wrapped estimators: how settings become the library's arguments (pymetrix.adapters)
    example: str = ""  # a short script; tests/test_help_examples.py runs every one
    menu: str = ""  # dialogs: where the app's menus put it ("Estimate → Equation...")
    results: tuple[tuple[str, str], ...] = ()  # what the output holds: (item, how to read it)
    maturity: str = ""  # "stable", "beta" or "experimental" (the class's ``maturity``); "" shows nothing

    @property
    def link(self) -> str:
        return link(self.group, self.key) if self.group else ""

    def reference(self) -> str:
        """The wrapped library's documentation for the wrapped function or class ("" if not installed)."""
        if not self.wraps:
            return ""
        try:
            target = resolve(self.wraps)
        except (ImportError, AttributeError):
            return f"{self.wraps} is not installed."
        return inspect.getdoc(target) or ""

    def to_markdown(self, reference: bool = False, level: int = 1) -> str:
        """The page as Markdown with ``pymetrix:`` links; ``level`` sets the heading depth
        (2 when the page sits below a dialog's own page)."""
        h = "#" * level
        out = [f"{h} {self.title}", "", self.summary]
        if maturity_line(self.maturity):
            out += ["", maturity_line(self.maturity)]
        if self.usage:
            out += ["", self.usage]
        if self.menu:
            out += ["", f"In the app: **{self.menu}**"]
        if self.example:
            out += ["", f"{h}# Example", "", "```python", self.example, "```"]
        if self.link:
            out += _notes_markdown(self.link, f"{h}#")
        if self.group == "commands" and self.key:
            out += _howto_markdown(self.key, f"{h}#")
        for label, options in self.choices:
            out += ["", f"**{label}:** " + ", ".join(f"[{t}]({u})" for t, u in options)]
        if self.settings:
            out += ["", f"{h}# Settings", "", "| Setting | Default | Description |", "| --- | --- | --- |"]
            out += [f"| {n}{' (advanced)' if adv else ''} | {_default_cell(d)} | {_cell(text[:1].upper() + text[1:])} |"
                    for n, text, d, adv in self.settings]
        if self.results:
            out += ["", f"{h}# Results", "", "| Output | How to read it |", "| --- | --- |"]
            out += [f"| {_cell(item)} | {_cell(text)} |" for item, text in self.results]
        if self.wraps or self.docs:
            out += ["", f"{h}# Reference", ""]
            if self.wraps:
                shown = f"[its documentation]({link('reference', f'{self.group}/{self.key}')})" if self.group and not reference else "its documentation below"
                out += [f"Wraps `{self.wraps}`; see {shown}."]
            if self.docs:
                out += ["", f"Online: [{self.docs}]({self.docs})"]
            if self.mapping:
                out += ["", f"{h}## How the settings reach the library", "", self.mapping]
            if reference:
                out += ["", "```text", self.reference(), "```"]
        if self.topics:
            out += ["", f"{h}# Background", ""] + [f"- [{_TOPICS[t].title if t in _TOPICS else t}]({link('topic', t)})" for t in self.topics]
        if self.references:
            out += ["", f"{h}# References", ""] + [reference_markdown(r) for r in self.references]
        if self.plugin:
            out += ["", f"_Provided by {self.plugin}._"]
        return "\n".join(out).strip() + "\n"


MATURITY = {
    "stable": "checked against an independent implementation and on the main, well-used path",
    "beta": "validated in tests, little use so far",
    "experimental": "working but thinly checked, likely to change",
}
"""The maturity tiers a class may declare (``maturity: ClassVar[str]``), with what each means; docs/features.md
defines them."""



def maturity_line(tier: str) -> str:
    """The line a help page shows under its summary for a maturity tier ("" for anything else)."""
    if tier not in MATURITY:
        return ""
    return f"*Maturity: {tier}: {MATURITY[tier]}.*"  # no link: the help also runs offline


def resolve(path: str) -> Any:
    """``"linearmodels.iv.IVLIML"`` -> the object (ImportError if the library is missing)."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for name in parts[i:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(f"cannot import {path}")


def _split_doc(cls: type) -> tuple[str, str]:
    """(summary, the rest) of the class's own docstring; dataclasses' generated signature does not count."""
    doc = cls.__dict__.get("__doc__")
    if not doc or doc.startswith(cls.__name__ + "("):
        return "", ""
    summary, _, rest = inspect.cleandoc(doc).partition("\n\n")
    return " ".join(summary.split()), rest.strip()


@renamed_arguments(target="obj")
def help_page(obj: Any) -> HelpPage:
    """The help page for a registered extension, a class, or an instance."""
    entry = obj if isinstance(obj, Entry) else None
    cls = entry.obj if entry else (obj if isinstance(obj, type) else type(obj))
    if entry is None:
        entry = next((e for g in HELP_GROUPS for e in [registry(g).of(cls)] if e is not None), None)
    summary, usage = _split_doc(cls)
    try:
        specs = form_spec(cls)
    except (TypeError, KeyError):
        specs = []
    preset = entry.preset if entry else {}
    settings = [(s.label, s.help, preset.get(s.name, s.default), s.advanced) for s in specs if s.kind != "options"]
    plugin = f"{entry.plugin.name} {entry.plugin.version}".strip() if entry and entry.plugin.name != "pymetrix" else ""
    choices = []
    for s in specs:  # dialogs: link to the pages of what each choice field offers
        group = {"extension": s.source, "method": "methods", "test": "tests"}.get(s.kind)
        if group:
            reg = registry(group)
            choices.append((s.label, tuple((reg.display(e), link(group, e.qualified)) for e in reg.entries())))
    return HelpPage(
        title=entry.label if entry else cls.__name__,
        summary=summary,
        usage=usage,
        settings=settings,
        plugin=plugin,
        wraps=getattr(cls, "wraps", "") or "",
        docs=getattr(cls, "docs", "") or "",
        references=tuple(getattr(cls, "references", ()) or ()),
        topics=tuple(getattr(cls, "topics", ()) or ()),
        group=entry.group if entry else "",
        key=entry.qualified if entry else "",
        choices=tuple(choices),
        mapping=spec.describe() if (spec := getattr(cls, "adapter_spec", None)) is not None else "",
        example=inspect.cleandoc(getattr(cls, "example", "") or ""),
        results=tuple(getattr(cls, "results", ()) or ()),
        maturity=getattr(cls, "maturity", "") or "",
        menu=menu_path(cls) if entry and entry.group == "commands" else "",
    )


def menu_path(cls: type) -> str:
    """Where the app's menus put a dialog ("Estimate → Time series → ARDL and bounds test...");
    "" if it has no menu entry."""
    from .commands.menus import menu_order
    from .commands.registry import COMMANDS

    paths = {info.cls: path for info, path in menu_order([i for i in COMMANDS.values() if i.menu])}
    return paths.get(cls, "")


GROUP_TITLES = {"commands": "Commands", "methods": "Estimation methods", "variances": "Variance models", "tests": "Tests",
                "filters": "Filters", "solvers": "Solvers", "models": "Models"}


_ENV_LOADED = False


def _load_all() -> None:
    """Built-in dialogs, plugins, and course folders named in the PYMETRIX_COURSES environment variable."""
    global _ENV_LOADED
    import os

    import pymetrix.commands  # noqa: F401  (registers the built-in dialogs)
    from .plugins import load_all

    load_all()  # includes "pymetrix.courses" entry points: functions that call load_course
    if not _ENV_LOADED:
        _ENV_LOADED = True
        for folder in filter(None, os.environ.get("PYMETRIX_COURSES", "").split(os.pathsep)):
            try:
                load_course(folder)
            except FileNotFoundError:
                pass


def contents() -> list[tuple[str, list[tuple[str, str]]]]:
    """Sections of the help contents: (section title, [(page title, link), ...])."""
    from .howto import is_howto, section

    _load_all()
    out = [section()]  # task pages first ("How do I…"), then a section per kind of page
    for group in ("commands", *(g for g in HELP_GROUPS if g != "commands")):
        reg = registry(group)
        pages = sorted(((_page_title(group, reg, e), link(group, e.qualified)) for e in reg.entries()), key=lambda p: p[0].lower())
        if pages:
            out.append((GROUP_TITLES[group], pages))
    out.append(("Background", [(t.title, link("topic", t.id)) for t in topics() if not is_howto(t.id)]))
    for course in _COURSES.values():
        out.append((f"Course: {course.name}", [(n.title, n.link) for n in course.notes]))
    return out


_TEST_NAMES = {  # tests whose description does not start with their name
    "breitung": "Breitung panel unit-root test", "breusch_godfrey": "Breusch-Godfrey LM test", "breusch_pagan": "Breusch-Pagan test", "cusumsq": "CUSUM of squares test",
    "granger": "Granger causality test", "hansen_j": "Hansen J test", "jarque_bera": "Jarque-Bera test",
    "normality": "Normality of residuals", "wald": "Wald test", "zivot_andrews": "Zivot-Andrews test",
}


def _page_title(group: str, reg: Any, entry: Any) -> str:
    """A page's name in the contents. Tests are registered by the code scripts use (adf, breusch_godfrey); the contents
    give their name too, from the first words of their description: "Augmented Dickey-Fuller test (adf)"."""
    label = reg.display(entry)
    if group != "tests":
        return label
    first = ((getattr(entry.obj, "__doc__", "") or "").strip().splitlines() or [""])[0]
    name = _TEST_NAMES.get(label) or first.split(":")[0].split(" (")[0].rstrip(".").strip()
    return f"{name} ({label})" if name and name.lower() != label.lower() and len(name) <= 60 else label


def _section_key(title: str) -> str:
    """A contents section's key in a link: "Course: Forecasting" -> "course-forecasting"."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def home_markdown(open_sections: Any = None, details: bool = False) -> str:
    """The help contents: a section per kind of page, one link per line.

    ``open_sections``: the keys of the sections shown open (``"all"`` for every one); the others show only their
    heading, a link that opens it (``pymetrix:home/commands,tests``: the sections open after the click), so the
    viewer folds sections with no state of its own. ``details=True`` writes each section as an HTML ``<details>``
    block instead, which notebooks fold themselves."""
    sections = contents()
    keys = [_section_key(title) for title, _ in sections]
    opened = set(keys) if open_sections == "all" or details else set(open_sections or ()) & set(keys)  # unknown keys: ignored
    out = ["# PyMetrix help", "", "Search above, or choose a page. Each dialog's **Help** button shows the page for that dialog "
           "and for what is chosen in it."]
    if "coming-from-eviews" in _TOPICS and "coming-from-iris" in _TOPICS:  # orientation for users of other packages
        out += ["", f"Used to another package? [{_TOPICS['coming-from-eviews'].title}]({link('topic', 'coming-from-eviews')}) · "
                    f"[{_TOPICS['coming-from-iris'].title}]({link('topic', 'coming-from-iris')})"]
    if "how-do-i" in _TOPICS:  # task pages, start to finish
        out += ["", f"A task from start to finish? [{_TOPICS['how-do-i'].title}]({link('topic', 'how-do-i')})"]
    if not details:
        everything, nothing = link("home", ",".join(keys)), link("home", "none")
        out += ["", f"[Open all]({everything}) · [Close all]({nothing})"]
    for key, (title, pages) in zip(keys, sections):
        lines = [f"- [{t}]({u})" for t, u in pages]
        if details:
            out += ["", f"<details><summary><b>{title}</b> ({len(pages)})</summary>", "", *lines, "", "</details>"]
            continue
        is_open = key in opened
        toggled = sorted(opened ^ {key}, key=keys.index)
        heading = f"[{'▾' if is_open else '▸'} {title}]({link('home', ','.join(toggled) or 'none')}) ({len(pages)})"
        out += ["", f"## {heading}"]
        if is_open:
            out += ["", *lines]
    return "\n".join(out) + "\n"


def topic_markdown(t: Topic) -> str:
    used = [(registry(g).display(e), link(g, e.qualified)) for g in HELP_GROUPS for e in registry(g).entries()
            if t.id in (getattr(e.obj, "topics", ()) or ())]
    out = [f"# {t.title}", "", t.text] + _notes_markdown(link("topic", t.id))
    if used:
        out += ["", "## Used by", "", " · ".join(f"[{title}]({u})" for title, u in used)]
    if t.see_also:
        out += ["", "## See also", ""] + [f"- [{_TOPICS[s].title if s in _TOPICS else s}]({link('topic', s)})" for s in t.see_also]
    if t.references:
        out += ["", "## References", ""] + [reference_markdown(r) for r in t.references]
    return "\n".join(out) + "\n"


def _default_cell(value: Any) -> str:
    """A setting's default as the reader sees it: a word as it is (data, mean), not its Python repr ('data'), and
    other values (numbers, True, lists) as Python writes them."""
    if value is None or value == "":
        return ""
    return _cell(value if isinstance(value, str) else repr(value))


def page_for_link(url: str) -> str:
    """The Markdown for a ``pymetrix:`` help link."""
    _load_all()
    kind, _, key = url.removeprefix(SCHEME).partition("/")
    if kind in ("", "home"):  # pymetrix:home/commands,tests: the contents with those sections open
        return home_markdown([k for k in key.split(",") if k and k != "none"])
    if kind == "topic":
        return topic_markdown(topic(key))
    if kind == "note":
        note = next((n for n in _notes() if n.id == key), None)
        if note is None:
            raise NotFoundError(f"no course note {key!r}")
        return note_markdown(note)
    if kind == "reference":
        group, _, name = key.partition("/")
        return help_page(registry(group).get(name)).to_markdown(reference=True)
    return help_page(registry(kind).get(key)).to_markdown()


def dialog_markdown(command_cls: type, values: dict[str, Any]) -> str:
    """Context help for a dialog: its own page, then the page of each extension chosen in it."""
    from .forms import form_spec as specs_of

    out = [help_page(command_cls).to_markdown()]
    for spec in specs_of(command_cls):
        group = {"extension": spec.source, "method": "methods", "test": "tests"}.get(spec.kind)
        chosen = values.get(spec.name)
        if not group or not chosen:
            continue
        try:
            entry = registry(group).find(str(chosen))
        except KeyError:
            entry = None
        if entry is not None:
            out += ["---", f"*{spec.label}: {registry(group).display(entry)}*", "", help_page(entry).to_markdown(level=2)]
    return "\n\n".join(out)


# Course material: an instructor's notes and example notebooks, linked into the help.


@dataclass(frozen=True)
class Note:
    """A lecture note (Markdown) or example notebook (``.ipynb``) from a course folder.

    ``links`` say which help pages it belongs to: topic ids (``garch``) or
    ``group/name`` (``methods/LIML``, ``variances/GARCH``, ``commands/Equation``).
    """

    id: str
    title: str
    course: str
    text: str
    path: Path
    links: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    kind: str = "note"  # "note" or "notebook"

    @property
    def link(self) -> str:
        return link("note", self.id)


@dataclass(frozen=True)
class Course:
    name: str
    path: Path
    notes: tuple[Note, ...]
    problems: tuple[str, ...] = ()  # links that match no help page, local files that are missing
    assistant: str = ""  # what an AI assistant may do while the course is loaded: "explain" (no steps proposed or run), "off", or "" (no rule)


_COURSES: dict[str, Course] = {}


def _front_matter(text: str) -> tuple[dict[str, Any], str]:
    """``---``-delimited ``key: value`` lines at the top of a note (lists as ``[a, b]`` or ``- item`` lines)."""
    if not text.startswith("---"):
        return {}, text
    head, sep, body = text[3:].partition("\n---")
    if not sep:
        return {}, text
    meta: dict[str, Any] = {}
    key = ""
    for line in head.strip().splitlines():
        if line.lstrip().startswith("- ") and key:
            meta.setdefault(key, [])
            if not isinstance(meta[key], list):
                meta[key] = [meta[key]] if meta[key] else []
            meta[key].append(line.lstrip()[2:].strip().strip("'\""))
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if value.startswith("[") and value.endswith("]"):
                meta[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
            else:
                meta[key] = value.strip("'\"")
    return meta, body.lstrip("\n")


def _as_list(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(value) if isinstance(value, list) else (str(value),)


def _notebook(path: Path) -> tuple[dict[str, Any], str]:
    """A notebook as Markdown: its Markdown cells, and code cells as code blocks (outputs are not shown).
    Settings come from the notebook metadata ``{"pymetrix": {...}}`` or front matter in its first cell."""
    import json

    nb = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(nb.get("metadata", {}).get("pymetrix", {}))
    parts = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if i == 0 and cell.get("cell_type") in ("markdown", "raw") and source.startswith("---"):
            more, source = _front_matter(source)
            meta = {**more, **meta}
        if cell.get("cell_type") == "markdown":
            parts.append(source)
        elif cell.get("cell_type") == "code" and source.strip():
            parts.append(f"```python\n{source}\n```")
    return meta, "\n\n".join(p for p in parts if p.strip())


def _absolute_links(text: str, folder: Path, root: Path | None = None) -> tuple[str, list[str]]:
    """Relative links in a note (slides, data, other notes) become file links. Reported: missing files,
    and files outside the course folder ``root`` (projects save only the folder)."""
    missing = []

    def fix(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
            return m.group(0)
        file = (folder / target.split("#")[0]).resolve()
        if not file.exists():
            missing.append(f"linked file {target} not found")
        elif root is not None and not file.is_relative_to(root):
            missing.append(f"linked file {target} is outside the course folder, so projects will not include it")
        return f"[{label}]({file.as_uri()})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", fix, text), missing


def _resolve_note_link(target: str) -> str | None:
    """A note's link (``garch``, ``methods/LIML``) as the help link it belongs to, or None."""
    if "/" not in target:
        return link("topic", target) if target in _TOPICS else None
    group, _, key = target.partition("/")
    group = {"method": "methods", "test": "tests", "filter": "filters", "variance": "variances",
             "solver": "solvers", "model": "models", "command": "commands", "dialog": "commands", "dialogs": "commands"}.get(group, group)
    if group not in HELP_GROUPS:
        return None
    try:
        entry = registry(group).find(key)
    except KeyError:
        entry = None
    if entry is None and group == "commands":  # a dialog by its class name, e.g. FitEquation
        entry = next((e for e in registry(group).entries() if e.obj.__name__.lower() == key.lower()), None)
    return link(group, entry.qualified) if entry is not None else None


def load_course(path: str | Path, name: str | None = None) -> Course:
    """Add a folder of course material to the help: ``*.md`` notes and ``*.ipynb`` example notebooks.

    Each file may start with front matter::

        ---
        title: Lecture 6. Volatility
        course: ECON 5010 Applied Time Series
        links: [garch, variances/GARCH, commands/Equation]
        references:
          - Tsay, R. (2010). Analysis of Financial Time Series, chapter 3.
        assistant: explain
        ---

    (notebooks: the same keys in the first cell, or in metadata ``"pymetrix"``).
    The notes appear in the help contents and search, and on every page they link to. ``assistant``, in any note,
    sets what an AI assistant may do while the course is loaded (``Course.assistant``): ``explain`` (it explains
    methods and results, and proposes or runs no steps) or ``off``; the strictest any note sets applies.
    """
    _load_all()
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"no course folder {folder}")
    read = []
    for file in sorted([*folder.rglob("*.md"), *folder.rglob("*.ipynb")]):
        if ".ipynb_checkpoints" in file.parts:
            continue
        meta, text = _notebook(file) if file.suffix == ".ipynb" else _front_matter(file.read_text(encoding="utf-8"))
        read.append((file, meta, text))
    # one folder is one course: named by the argument, else the first note that names it, else the folder
    course = name or next((str(m["course"]) for _, m, _ in read if m.get("course")), folder.name)
    slug = re.sub(r"[^a-z0-9]+", "-", course.lower()).strip("-")
    notes, problems = [], []
    rules = {"": 0, "explain": 1, "off": 2}
    assistant = ""
    for file, meta, text in read:
        rule = str(meta.get("assistant", "")).strip().lower()
        if rule not in rules:
            problems.append(f"{file.name}: assistant {rule!r} is not explain or off")
        elif rules[rule] > rules[assistant]:
            assistant = rule
        text, missing = _absolute_links(text, file.parent, folder)
        problems += [f"{file.name}: {m}" for m in missing]
        heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = str(meta.get("title") or (heading.group(1).strip() if heading else file.stem.replace("_", " ")))
        links = _as_list(meta.get("links"))
        problems += [f"{file.name}: link {t!r} matches no help page" for t in links if _resolve_note_link(t) is None]
        notes.append(Note(f"{slug}/{file.relative_to(folder).as_posix()}", title, course, text, file, links,
                          _as_list(meta.get("references")), "notebook" if file.suffix == ".ipynb" else "note"))
    result = Course(course, folder, tuple(notes), tuple(problems), assistant)
    _COURSES[str(folder)] = result
    return result


def unload_course(path: str | Path) -> None:
    _COURSES.pop(str(Path(path).expanduser().resolve()), None)


def courses() -> list[Course]:
    return list(_COURSES.values())


def _notes() -> list[Note]:
    return [n for c in _COURSES.values() for n in c.notes]


def notes_for(page_link: str) -> list[Note]:
    """Course notes and notebooks that link to a help page."""
    return [n for n in _notes() if any(_resolve_note_link(t) == page_link for t in n.links)]


def _notes_markdown(page_link: str, h: str = "##") -> list[str]:
    found = notes_for(page_link)
    if not found:
        return []
    return ["", f"{h} Course material", ""] + [f"- [{n.title}]({n.link}) ({'example notebook, ' if n.kind == 'notebook' else ''}{n.course})" for n in found]


def _howto_markdown(command: str, h: str = "##") -> list[str]:
    """The task pages ("How do I…") whose steps open the dialog ``command``, for its help page."""
    from .commands import COMMANDS
    from .howto import pages_for_dialog

    tail = command.rsplit(":", 1)[-1]  # the help's key, "pymetrix:fitvar", is the class's name in lower case
    name = next((i.cls.__name__ for i in COMMANDS.values() if i.cls.__name__.lower() == tail), tail)
    found = pages_for_dialog(name)
    if not found:
        return []
    return ["", f"{h} How do I…", ""] + [f"- [{p.title}]({link('topic', p.id)}): {p.summary}" for p in found]


def note_markdown(note: Note) -> str:
    out = [f"*{note.course}*", "", note.text if note.text.lstrip().startswith("#") else f"# {note.title}\n\n{note.text}"]
    if note.kind == "notebook":
        out += ["", f"[Open the notebook]({note.path.as_uri()}) (code cells are shown without their output)"]
    pages = [(t, _resolve_note_link(t)) for t in note.links]
    if any(u for _, u in pages):
        out += ["", "## See in the help", "", " · ".join(f"[{t}]({u})" for t, u in pages if u)]
    if note.references:
        out += ["", "## References", ""] + [reference_markdown(r) for r in note.references]
    return "\n".join(out) + "\n"


def link_of(obj: Any) -> str:
    """The help link for a search result: an entry, a topic or a note."""
    if isinstance(obj, Entry):
        return link(obj.group, obj.qualified)
    if isinstance(obj, Topic):
        return link("topic", obj.id)
    return obj.link


def search(text: str) -> list[tuple[str, str, Any]]:
    """Help pages, topics and course notes matching every word of ``text``: (kind, title, entry, topic or note), best first."""
    words = [w for w in re.split(r"\W+", text.lower()) if w]
    found: list[tuple[int, str, str, Any]] = []
    for note in _notes():
        body = f"{note.title} {note.course} {note.text}".lower()
        if all(w in body for w in words):
            found.append((sum(w in note.title.lower() for w in words) * 10 + sum(body.count(w) for w in words), note.kind, note.title, note))
    for group in HELP_GROUPS:
        for entry in registry(group).entries():
            page = help_page(entry)
            title = page.title.lower()
            body = " ".join([title, entry.name, page.summary, page.usage, " ".join(page.topics),
                             " ".join(h for _, h, _, _ in page.settings)]).lower()
            if all(w in body for w in words):
                found.append((sum(w in title for w in words) * 10 + sum(body.count(w) for w in words), group, page.title, entry))
    for t in _TOPICS.values():
        body = f"{t.id} {t.title} {t.text}".lower()
        if all(w in body for w in words):
            named = all(w in t.title.lower() for w in words)  # a topic its title names leads: it introduces the methods and dialogs
            found.append((sum(w in t.title.lower() for w in words) * 10 + sum(body.count(w) for w in words) + 50 * named,
                          "topic", t.title, t))
    return [(kind, title, obj) for _, kind, title, obj in sorted(found, key=lambda f: -f[0])]


# Background topics for the built-in components. Short by design: what the idea is,
# when to use it, and where to read more.

add_topic("least-squares", "Least squares", """
    Ordinary least squares chooses the coefficients that minimise the sum of squared
    residuals. It is unbiased when the regressors are uncorrelated with the error, and
    efficient when the errors are also homoskedastic and serially uncorrelated; robust
    (White) and HAC (Newey-West) covariances keep inference valid when they are not.
    Equations written in explicit form are estimated by nonlinear least squares.
    In code, pymetrix.ols(y, X) estimates without a formula (see "Regression without a formula").

    Restricted least squares imposes linear restrictions on the coefficients, such as
    constant returns (a + b = 1) or a coefficient fixed at a value: type them under
    Method settings → Restrictions for OLS or IV (in code, OLS(restrictions=["a + b = 1"])),
    naming coefficients as the summary does (const, y, x[-1]), several separated by
    commas. The estimate is b - (X'X)^-1 R' [R (X'X)^-1 R']^-1 (R b - r), the same as
    substituting the restrictions into the equation; standard errors follow the chosen
    covariance, with one residual degree of freedom more for each restriction, and a
    coefficient fixed by a restriction has a standard error of zero. Restrictions must be
    linear; a nonlinear one belongs in an explicit-form equation, written in.
    """, ("Greene, W. H. (2018). Econometric Analysis, 8th ed., chapters 2 (\"The linear regression model\"), 3 (\"Least squares regression\") and 4 (\"Estimating the regression model by least squares\"). Pearson.",), ("instrumental-variables", "regression-arrays"))
add_topic("regression-arrays", "Regression without a formula", """
    For code that runs many small regressions (loops, simulations, model building),
    pymetrix.ols(y, X) skips the formula and the dataset: y is a Series (or a 1-d array),
    X a DataFrame (or a 2-d array, with names= for its columns). The options are those
    of OLS: cov="classic", "white", "hac" (with hac_lags and hac_correction), or
    Cluster(...) with clusters= (one- or two-way); constant=False leaves out the
    intercept; weights= gives weighted least squares; missing="raise" refuses missing
    values instead of dropping their rows.

    The result is the fitted equation a formula fit gives on the same data
    (Equation("y ~ x1 + x2").fit(data, method=OLS(...))): the same numbers, views,
    tests and diagnostics, and the same provenance, except that it records the data as
    passed directly (vintage request "direct"). A PeriodIndex makes the data a time
    series. In a session the call is recorded as the code you typed; the arrays are
    kept with the project only if you keep them there yourself. There is no dialog:
    Estimate → Equation estimates by formula.
    """, (), ("least-squares",))
add_topic("instrumental-variables", "Instrumental variables", """
    When a regressor is correlated with the error (simultaneity, measurement error,
    omitted variables), least squares is inconsistent. Instruments are variables that
    move the endogenous regressor but are unrelated to the error. Two-stage least
    squares, LIML and GMM use them; weak instruments (a low first-stage F) bias the
    estimates towards least squares, which LIML and Fuller's estimator resist better.
    Test → Weak-instrument tests runs tests that stay valid however weak the instruments
    are (Anderson-Rubin, its confidence set, Stock-Wright) and the C test of a subset.
    """, ("Stock, J. H. and Yogo, M. (2005). Testing for weak instruments in linear IV regression. In Andrews, D. W. K. and Stock, J. H. (eds.), Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg, 80-108. Cambridge University Press. doi:10.1017/cbo9780511614491.006",
          "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed., chapter 5, \"Instrumental variables estimation of single-equation linear models\". MIT Press."), ("least-squares",))
add_topic("systems", "Systems of equations: SUR and 3SLS", """
    Estimating equations together pays when their errors are correlated, or when their
    regressors are endogenous variables of the system.

    Seemingly unrelated regressions (SUR) apply generalised least squares to the stacked
    equations, weighting by the inverse of the residual covariance across equations. When
    the equations have different regressors and correlated errors this is more precise
    than least squares one equation at a time; with identical regressors, or uncorrelated
    errors, the two coincide. The Breusch-Pagan test checks whether the errors are
    correlated at all. Iterating until the covariance settles gives maximum likelihood.

    Three-stage least squares (3SLS) does the same with instruments: two-stage least
    squares for each equation, then the residual covariance, then generalised least
    squares on the instrumented system. In a simultaneous model the left-hand sides are
    endogenous, and the constant, exogenous variables and lagged variables are the
    instruments. The Hansen-Sargan test checks the overidentifying restrictions.

    Joint estimation also allows restrictions across equations, such as the same
    coefficient in two equations, or a parameter shared by explicit-form equations.
    """, ("Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias. Journal of the American Statistical Association 57(298), 348-368. doi:10.1080/01621459.1962.10480664",
          "Zellner, A. and Theil, H. (1962). Three-stage least squares: simultaneous estimation of simultaneous equations. Econometrica 30(1), 54-78. doi:10.2307/1911287",
          "Breusch, T. S. and Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. Review of Economic Studies 47(1), 239-253. doi:10.2307/2297111",
          "Greene, W. H. (2018). Econometric Analysis, 8th ed., chapters 10 (\"Systems of regression equations\") and 15 (\"Simulation-based estimation and inference and random parameter models\"). Pearson."),
          ("least-squares", "instrumental-variables", "model-solution"))
add_topic("panel-data", "Panel data estimators", """
    Panels follow many entities over time. Fixed effects (fe() terms) remove each
    entity's mean; random effects treat entity effects as random and uncorrelated with
    the regressors (test with Hausman); first differences remove fixed effects by
    differencing; the between estimator uses entity means; Fama-MacBeth averages
    period-by-period cross-section regressions.
    """, ("Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed., chapter 10, \"Basic linear unobserved effects panel data models\". MIT Press.",))
add_topic("panel-data-tools", "Working with panel data", """
    A panel holds the same series for several entities (countries, firms, sectors) over
    time. PyMetrix keeps it in long form, one row per entity and period, indexed by a
    pandas MultiIndex; lags are always taken within each entity.

    Pivot tables summarise series by axes or categories: the mean of inflation by region
    and year, or counts of observations by country. Wide form (one column per entity,
    y_AUS, y_NZL) suits graphs and spreadsheets; long form suits estimation; reshape with
    wide() and long(). Take one country's time series with entity(), or one period's
    cross-section with cross_section().

    Group-wise functions work in any expression or derived series: mean(y, country) is
    each country's mean on every row, mean(y, time) the cross-country mean each period,
    and demean(), share(), rank(), sum(), sd(), count(), min(), max() and median() follow
    the same pattern; group by an axis or by a series of categories such as region, as
    mean(y, country) or mean(y, by=country). For max and min, write by=: max(a, b) is the
    larger of two values in each period.

    Before estimating, check coverage: which entity has data when, and where the gaps
    are. balance() keeps the periods (or entities) with complete data. The overall,
    between and within standard deviations show whether a series varies mostly across
    entities or over time, which matters for fixed-effects estimation: a series that
    hardly varies within entities is poorly identified once their means are removed.
    """, ("Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed. MIT Press.",
          "McKinney, W. (2022). Python for Data Analysis, 3rd ed., chapters 8 (\"Data wrangling: join, combine, and reshape\") and 10 (\"Data aggregation and group operations\"). O'Reilly."),
          ("panel-data", "derived-series"))
add_topic("dynamic-panels", "Dynamic panels: Arellano-Bond and system GMM", """
    In a panel with a lagged dependent variable and entity effects, least squares is
    biased up (the lag is correlated with the effect) and the within estimator biased
    down when there are few periods (Nickell bias). Differencing removes the effects but
    makes the differenced lag correlated with the differenced error; lags two and
    earlier of the dependent variable are then valid instruments (Arellano and Bond).
    When the series are persistent, lagged levels say little about differences; system
    GMM adds the equation in levels, instrumented by lagged differences, under the extra
    assumption that changes are uncorrelated with the entity effects (Blundell and Bond).

    Two-step GMM is efficient, but its usual standard errors are far too small in finite
    samples; the Windmeijer correction fixes most of that. Check the model with the
    Arellano-Bond tests: first-order serial correlation in the differenced residuals is
    expected, second-order would invalidate the instruments. Hansen's J tests the
    overidentifying restrictions, but too many instruments weaken it (a p-value near
    one is a warning sign): keep the instrument count below the number of groups, by
    collapsing instruments or limiting lags.
    """, ("Arellano, M. and Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. Review of Economic Studies 58(2), 277-297. doi:10.2307/2297968",
          "Blundell, R. and Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. Journal of Econometrics 87(1), 115-143. doi:10.1016/s0304-4076(98)00009-8",
          "Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. Journal of Econometrics 126(1), 25-51. doi:10.1016/j.jeconom.2004.02.005",
          "Roodman, D. (2009). How to do xtabond2: an introduction to difference and system GMM in Stata. Stata Journal 9(1), 86-136. doi:10.1177/1536867X0900900106",
          "Nickell, S. (1981). Biases in dynamic models with fixed effects. Econometrica 49(6), 1417-1426. doi:10.2307/1911408"),
          ("panel-data", "instrumental-variables"))
add_topic("quantile-regression", "Quantile regression", """
    Least squares estimates how the mean of the dependent variable moves with the
    regressors. Quantile regression estimates how a quantile moves: the median, or the
    10th percentile, by minimising absolute residuals weighted q above the line and
    1 - q below it. It is robust to outliers in the dependent variable, and it shows
    whether a regressor matters more in the tails, which is the idea behind growth at
    risk: financial conditions predict the lower quantiles of future growth much more
    than its median.

    Plotting each coefficient across quantiles, with OLS for comparison, shows where in
    the distribution an effect lies. Standard errors use a kernel estimate of the error
    density at the quantile (robust), or assume identically distributed errors (iid).
    Estimate → Quantile process draws that plot for a fitted equation.
    """, ("Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46(1), 33-50. doi:10.2307/1913643",
          "Koenker, R. (2005). Quantile Regression. Cambridge University Press. doi:10.1017/cbo9780511754098",
          "Adrian, T., Boyarchenko, N. and Giannone, D. (2019). Vulnerable growth. American Economic Review 109(4), 1263-1289. doi:10.1257/aer.20161923"),
          ("least-squares",))
add_topic("robust-regression", "Robust regression", """
    A few outlying values of the dependent variable can pull least squares a long way.
    Robust regression (M-estimation) replaces the sum of squared residuals by a sum of
    rho(u / s) that grows more slowly for large residuals: Huber's norm counts them
    linearly; Tukey's bisquare counts residuals beyond its tuning constant not at all.
    The estimates are found by iteratively reweighted least squares from the least-squares
    start, each observation weighted by how far its residual lies out, and the scale s is
    re-estimated each round by the median absolute residual over 0.6745 (the MAD).

    Choose Robust regression (M) as the method in Estimate → Equation; the norm is in
    Method settings, and the tuning constant (1.345 for Huber, 4.685 for the bisquare:
    95% efficient with normal errors; smaller is more robust), a fixed scale and the
    convergence settings under Advanced. Standard errors are Huber's H1, as statsmodels'
    RLM. Robust regression does not protect against outlying regressors (leverage), and
    the bisquare's answer can depend on its least-squares start. fit.extra["robust_weights"]
    lists each observation's final weight: those near zero are the outliers.
    """, ("Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics 35(1), 73-101. doi:10.1214/aoms/1177703732",
          "Huber, P. J. (1973). Robust regression: asymptotics, conjectures and Monte Carlo. Annals of Statistics 1(5), 799-821. doi:10.1214/aos/1176342503",
          "Holland, P. W. and Welsch, R. E. (1977). Robust regression using iteratively reweighted least-squares. Communications in Statistics - Theory and Methods 6(9), 813-827. doi:10.1080/03610927708827533"),
          ("least-squares", "quantile-regression"))
add_topic("maximum-likelihood", "Maximum likelihood", """
    Maximum likelihood chooses the parameters that make the observed data most probable
    under the model: logit and probit for 0/1 outcomes, Poisson for counts, Tobit and
    Heckman's model for censored and selected samples, and regressions with ARMA errors
    or GARCH variance. If the model is right the estimates are consistent and, in large
    samples, normal and as precise as any consistent estimator can be.

    Their precision comes from the curvature of the log likelihood at its maximum, the
    information matrix, whose inverse is their covariance. cov chooses how it is
    estimated: classic, the inverse Hessian (second derivatives), right when the
    likelihood is the true one; opg (logit, probit, Poisson), the outer product of each
    observation's scores (first derivatives), equivalent under the same assumption but
    often worse in small samples; white, the quasi-ML sandwich of the two, still valid when the distribution
    is wrong but the part that matters (a Poisson's mean, a GARCH's variance equation)
    is right. The estimates are then quasi-ML, and only the sandwich can be trusted.

    Three tests compare a model with a restricted one, each chi-squared with as many
    degrees of freedom as restrictions. The likelihood ratio (LR) test fits both and
    doubles the gap in log likelihood; the Wald test fits only the unrestricted model
    and asks whether the restrictions nearly hold; the Lagrange multiplier (LM, score)
    test fits only the restricted one and asks whether the scores of the extra
    parameters are far from zero. They agree in large samples but not always in small
    ones, and Wald depends on how a nonlinear restriction is written.

    The maximum is found numerically from starting values. A parameter is not
    identified when the likelihood is flat in it: collinear regressors, a regressor
    that predicts a 0/1 outcome perfectly (separation: estimates running off to
    infinity), or a parameter that matters only when another is not zero. A singular
    Hessian, huge standard errors or a fit stopped at its iteration limit are the signs.
    A likelihood can have several maxima, so check that other starting values give the
    same estimates and log likelihood.
    """, ("Greene, W. H. (2018). Econometric Analysis, 8th ed., chapter 14, \"Maximum likelihood estimation\". Pearson.",
          "White, H. (1982). Maximum likelihood estimation of misspecified models. Econometrica 50(1), 1-25. doi:10.2307/1912526",
          ("Engle, R. F. (1984). Wald, likelihood ratio, and Lagrange multiplier tests in econometrics. In Griliches, Z. and Intriligator, M. D. (eds.), Handbook of Econometrics, volume 2, chapter 13, 775-826. North-Holland. doi:10.1016/s1573-4412(84)02005-5")))
add_topic("garch", "Volatility models (GARCH)", """
    GARCH models let the variance of shocks change over time: today's variance depends
    on recent squared shocks (ARCH terms) and recent variances (GARCH terms). GJR,
    TARCH, EGARCH and APARCH let negative shocks raise volatility more than positive
    ones; FIGARCH gives volatility long memory. Fat-tailed (t, GED) and skewed error
    distributions are common for financial returns.
    """, ("Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics 31(3), 307-327. doi:10.1016/0304-4076(86)90063-1",
          ("Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance 48(5), 1779-1801. doi:10.1111/j.1540-6261.1993.tb05128.x"),
          ("Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach. Econometrica 59(2), 347-370. doi:10.2307/2938260")), ("maximum-likelihood",))
add_topic("trend-cycle-filters", "Trend-cycle filters", """
    Filters split a series into a slow-moving trend and a cycle. The HP filter penalises
    changes in the trend's growth (lambda sets the smoothness); band-pass filters
    (Baxter-King, Christiano-Fitzgerald) keep cycles between two lengths; Hamilton's
    filter uses the error in a two-year-ahead regression forecast. Two-sided filters
    revise the end of the sample as data arrive; one-sided versions use only the past.
    """, (("Hodrick, R. J. and Prescott, E. C. (1997). Postwar U.S. business cycles: an empirical investigation. Journal of Money, Credit and Banking 29(1), 1-16. doi:10.2307/2953682"),
          "Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. Review of Economics and Statistics 100(5), 831-843. doi:10.1162/rest_a_00706",
          "Christiano, L. and Fitzgerald, T. (2003). The band pass filter. International Economic Review 44(2), 435-465. doi:10.1111/1468-2354.t01-1-00076"))
add_topic("seasonal-adjustment", "Seasonal adjustment and calendar effects", """
    Monthly and quarterly data move with the seasons: retail sales rise before
    Christmas, construction slows in winter, tourism peaks in the dry season. Seasonal
    adjustment removes the regular within-year pattern so that underlying movements can
    be seen and compared from one period to the next.

    STL (seasonal-trend decomposition by LOESS) fits the seasonal pattern with local
    regressions, letting it change slowly; a longer seasonal smoother keeps it more
    stable, and the robust option stops outliers from distorting it. X-13ARIMA-SEATS,
    the US Census Bureau's program used by most statistical offices, first fits an ARIMA
    model (extending the series with forecasts, correcting outliers and, if asked,
    trading-day and Easter effects), then applies the X-11 moving-average filters
    (multiplicative, additive, pseudo-additive or log-additive; filters chosen by X-13 or
    set) or SEATS, which derives the filters from the ARIMA model itself. TRAMO-SEATS,
    from the Bank of Spain and the standard in Europe, does the same with TRAMO's model:
    here as Eurostat's JDemetra+ computes it, from its predefined specifications (RSA0, the
    airline model in levels, to RSAfull, with an automatic model, outliers, trading days
    and Easter), which any setting can change. The
    classical decomposition takes a centred moving average as the trend and averages
    each season's deviation from it, so its seasonal pattern never changes.

    Multiplicative adjustment, used for positive series, treats the seasonal effect as
    a percentage (a factor around 1); additive adjustment treats it as an amount. Months
    differ in their numbers of each weekday, and Easter moves between March and April:
    calendar regressors capture these effects, in regressions or in X-13.

    In an equation, seasonal() adds seasonal dummies: one 0/1 regressor per quarter of
    quarterly data or month of monthly data, leaving out the first when the equation has
    a constant (named seasonal[Q2], seasonal[Q3], ...; with - 1 every season has one).
    C(quarter) and C(month) do the same with the quarter or month taken from the dates.
    The dummies follow each observation's period, whatever the sample's start; a Wald
    test of them all being zero (seasonal[Q2] = 0, seasonal[Q3] = 0, seasonal[Q4] = 0)
    tests for deterministic seasonality.

    Adjusted series are revised as new data arrive, because two-sided filters use
    observations on both sides; this is why adjusted series here are derived series,
    recomputed when the original changes.
    """, ("Cleveland, R. B., Cleveland, W. S., McRae, J. E. and Terpenning, I. (1990). STL: a seasonal-trend decomposition procedure based on loess. Journal of Official Statistics 6.",
          "Findley, D. F., Monsell, B. C., Bell, W. R., Otto, M. C. and Chen, B.-C. (1998). New capabilities and methods of the X-12-ARIMA seasonal-adjustment program. Journal of Business and Economic Statistics 16(2), 127-152. doi:10.1080/07350015.1998.10524743",
          "US Census Bureau (2017). X-13ARIMA-SEATS Reference Manual, version 1.1.",
          "Ladiray, D. and Quenneville, B. (2001). Seasonal Adjustment with the X-11 Method. Springer. doi:10.1007/978-1-4613-0175-2",
          "Gómez, V. and Maravall, A. (1996). Programs TRAMO and SEATS: instructions for the user (beta version: September 1996). Banco de España working paper 9628.",
          "Eurostat (2015). ESS Guidelines on Seasonal Adjustment."),
          ("trend-cycle-filters", "derived-series"))
add_topic("forecast-workflow", "The forecast workflow", """
    A forecast is rarely one model's output. A forecasting round combines models,
    judgement and a story, is published, and is later compared with what happened.

    A forecast round records the paths of each variable with when they were made, the
    last period of data used, notes, and every judgemental adjustment: which period was
    changed, from what, to what, and why. The paths before judgement are kept, so the
    effect of judgement can be measured. A forecast store keeps every round, so rounds
    can be compared (the revisions between them) and evaluated: the errors of past
    rounds by horizon are a real-time track record. A fan chart shows the central path
    with bands that widen with uncertainty (here 30, 60 and 90 per cent).

    A backtest asks how a model would have forecast in the past: re-estimate it on the
    data up to each origin, forecast ahead, and compare with what happened. With
    vintages, the data as they were at each origin, the evaluation is real-time; with
    revised data, it flatters. The Diebold-Mariano test asks whether two forecasts'
    accuracy differs by more than chance, allowing for the correlation of multi-step
    errors; the Harvey-Leybourne-Newbold correction improves it in small samples.
    """, ("Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business and Economic Statistics 13(3), 253-263. doi:10.1080/07350015.1995.10524599",
          "Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting 13(2), 281-291. doi:10.1016/s0169-2070(96)00719-4",
          "Britton, E., Fisher, P. and Whitley, J. (1998). The Inflation Report projections: understanding the fan chart. Bank of England Quarterly Bulletin.",
          "Croushore, D. and Stark, T. (2001). A real-time data set for macroeconomists. Journal of Econometrics 105(1), 111-130. doi:10.1016/s0304-4076(01)00072-0"),
          ("var", "bayesian-var", "model-solution"))
add_topic("reports", "Reports", """
    A report collects results for others to read: headings and text, tables and
    figures, in Excel, Word or HTML. Each result adds the tables and graph its window
    shows: by default its summary and graph, or chosen tabs, or all of them. Excel puts
    each table on its own sheet, with a contents sheet linking to them, and keeps the
    numbers as numbers, so they can be reused. Word and HTML follow the report's order,
    with numbered tables and figures.

    An appendix lists where every number came from: the data's vintages (when each
    series was released), the sample, the method and the version of pymetrix. A reader
    can then tell whether two reports used the same data, and reproduce either.
    """, (), ("forecast-workflow",))
add_topic("programs-and-projects", "Reading and writing projects from programs", """
    Every step of a session is a call of a public function: its stable ID, its arguments
    (settings written out in full, datasets and results by name) and the name its result
    is kept under. Another program can build such calls (make_call), run them in a
    session (Session.run), or record a slow one without running it
    (Session.add(cmd, deferred=True): listed as not computed until Compute runs it,
    against the objects it uses as they were where it stands in the history), and
    save the project for the app to open. It can keep its own files in the project
    (session.attachments), and read a project's steps, as calls, without opening it.

    A saved result (File → Save result…) keeps the call that made it in its provenance,
    so Call(**provenance["call"]).invoke({"data": data}) makes it again from the same
    data. Each result's provenance, each save of a project and each saved result also
    record the pymetrix commit beside the version, when pymetrix runs from a checkout of
    its source, and whether the code differed from that commit.

    These names (Call, make_call, the codec's encode and decode, Session.attachments) are
    a public interface, provisional before version 1.0: documented and tested as public,
    but still free to change before 1.0. The user guide's section "Reading and writing
    projects from programs" has examples.
    """, (), ("reports",))
add_topic("temporal-disaggregation", "Temporal disaggregation", """
    National accounts are often annual, while the decisions they inform are quarterly or
    monthly. Temporal disaggregation builds a high-frequency series that adds up to the
    low-frequency data (sums of flows, averages of rates and indices, first or last
    values of stocks) and moves like related high-frequency indicators.

    Chow and Lin regress the low-frequency series on the indicators, aggregated, by
    generalised least squares, assuming the high-frequency residual follows an AR(1);
    the low-frequency residuals are then spread over the high-frequency periods in the
    way that AR(1) implies. Rho is chosen by maximum likelihood (or by minimising the
    residual sum of squares, which often drives it to zero). Fernández assumes a random
    walk residual, which suits indicators that are not cointegrated with the target.
    Litterman's residual is a random walk whose steps follow an AR(1); with rho at zero it
    is Fernández. Rho, for Chow-Lin and Litterman, is searched between 0 and 0.999 (as in
    R's tempdisagg); widen the range (-0.999, 0.999) to allow negative values. An estimate
    at an end of the range is noted in the summary: often a sign that the data say little
    about rho. Denton (in Cholette's form) moves a preliminary series, the indicator or a
    constant, as little as possible in first differences (proportional or additive) to meet
    the totals: no regression, just benchmarking. In seasonal differences (lag 4 or 12) it
    keeps the indicator's seasonal pattern, with a light weight on first differences to pin
    what seasonal differences leave free. Where the indicators continue past the last year,
    the regression methods extrapolate, giving early estimates for the current year.

    An aggregation matrix replaces the conversion when the totals are not all of one kind:
    calendar-year and fiscal-year totals together, overlapping or not, or quarters for the
    latest year. Each row says which high-frequency periods a low-frequency value adds up
    (the weights), and the result meets every value exactly; a value that is missing drops
    its row. Soft constraints let the series miss the totals by about a given standard
    deviation, for totals known to be rough or to be revised: the regression methods treat
    it as measurement error, Denton as a penalty. Very small, it is the exact result; very
    large, the series follows the indicator alone.

    Components disaggregated one by one (consumption, investment, government) need not add
    up to the total disaggregated on its own. Data → Transform → Reconcile components…
    moves them as little as possible, each keeping its own annual totals, so that they add
    up every quarter; components known well can be held fixed. Least squares moves each
    value in proportion to its size; weighted least squares gives each component a share of
    every gap set by its weight, whatever its level or sign (imports, entered as negative
    values, move as any other component). Its summary gives the size of each component's
    adjustment.
    """, ("Chow, G. C. and Lin, A.-L. (1971). Best linear unbiased interpolation, distribution, and extrapolation of time series by related series. Review of Economics and Statistics 53(4), 372-375. doi:10.2307/1928739",
          "Fernández, R. B. (1981). A methodological note on the estimation of time series. Review of Economics and Statistics 63(3), 471-476. doi:10.2307/1924371",
          "Denton, F. T. (1971). Adjustment of monthly or quarterly series to annual totals: an approach based on quadratic minimization. Journal of the American Statistical Association 66(333), 99-102. doi:10.1080/01621459.1971.10482227",
          "Dagum, E. B. and Cholette, P. A. (2006). Benchmarking, Temporal Distribution, and Reconciliation Methods for Time Series. Springer. doi:10.1007/0-387-35439-5",
          "Litterman, R. B. (1983). A random walk, Markov model for the distribution of time series. Journal of Business & Economic Statistics 1(2), 169-173. doi:10.1080/07350015.1983.10509336",
          "Di Fonzo, T. and Marini, M. (2011). Simultaneous and two-step reconciliation of systems of time series: methodological and practical issues. Journal of the Royal Statistical Society, Series C (Applied Statistics) 60(2), 143-164. doi:10.1111/j.1467-9876.2010.00733.x",
          "Stone, R., Champernowne, D. G. and Meade, J. E. (1942). The precision of national income estimates. Review of Economic Studies 9(2), 111-125. doi:10.2307/2967664",
          "Sax, C. and Steiner, P. (2013). Temporal disaggregation of time series. The R Journal 5(2), 80-87. doi:10.32614/rj-2013-028"),
          ("seasonal-adjustment",))
add_topic("midas", "MIDAS: mixed-frequency regression", """
    Monthly indicators (industrial production, retail sales, surveys) arrive long before
    quarterly GDP. A MIDAS regression relates the quarterly series to many monthly lags
    of an indicator, constrained to a smooth lag polynomial so that twelve lags cost a
    slope and two shape parameters: exponential Almon or a normalised Beta function.
    Unrestricted MIDAS estimates every lag, which works when the frequency gap is small
    (months to quarters) and lags are few.

    For a nowcast, the regression can be estimated as if some months of each quarter
    were missing (data short at the end), matching what is known when the forecast is
    made; the nowcast then uses the months available for the current quarter.
    """, ("Ghysels, E., Santa-Clara, P. and Valkanov, R. (2004). The MIDAS touch: mixed data sampling regression models. Working paper, UCLA and UNC.",
          "Ghysels, E., Sinko, A. and Valkanov, R. (2007). MIDAS regressions: further results and new directions. Econometric Reviews 26(1), 53-90. doi:10.1080/07474930600972467",
          "Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted mixed data sampling (MIDAS): MIDAS regressions with unrestricted lag polynomials. Journal of the Royal Statistical Society, Series A (Statistics in Society) 178(1), 57-82. doi:10.1111/rssa.12043"),
          ("temporal-disaggregation", "forecast-workflow"))
# -- data and nowcasting dialogs: frequency conversion, the ragged edge, factors, connectors ----------------------------
add_topic("frequency-conversion", "Changing frequency and aggregating", """
    Data → Transform → Convert frequency… turns monthly data into quarterly or annual
    (or a July-June fiscal year) data, and Data → Panel → Aggregate over entities… adds a
    panel's countries or firms into totals. Both combine values the way the series'
    nature requires: flows (GDP, sales) and counts are summed, stocks (the money supply,
    employment at the end of a period) take the last value over time and are summed
    across entities, rates, indices and ratios are averaged. pymetrix reads this from
    each series' measure (Variable.measure); one choice for every series, or a rule per
    series, overrides it.

    A new period with one of its sub-periods missing is left missing rather than
    computed from part of the data, so a quarter is not reported from two of its
    months. Averages across entities are unweighted: for a weighted average (inflation
    across countries weighted by GDP) derive the weighted series first. Going the other
    way, from annual to quarterly, needs a model of how the total is spread: that is
    temporal disaggregation.
    """, (), ("temporal-disaggregation", "panel-data-tools"))
add_topic("nowcasting", "Nowcasting and the ragged edge", """
    A nowcast estimates the current quarter (or the one just ended) before its official
    figure, from monthly indicators that come out sooner. Each indicator is published
    with its own delay, so on any day they end in different months: the ragged edge.
    Estimate → Mixed frequency → Ragged edge… shows it; Data as known on a day… rebuilds
    it for any past day from each series' publication lag (Variable.release_lag), so a
    model can be evaluated with only the data that were known when its nowcast would
    have been made.

    There are three ways of handling the edge. A model that handles missing values (the
    dynamic factor model, by the Kalman filter) takes the data as they are. Bridge
    equations first fill the missing months with forecasts from a small autoregression
    of each indicator (Fill the ragged edge…). Vertical realignment shifts each series so
    that its latest value is treated as current (Realign the ragged edge…). Different
    models' nowcasts are often combined (Combine nowcasts…): averages of several models
    are rarely the best in any one quarter but are seldom far off, and inverse mean
    squared errors from a backtest make reasonable weights.
    """, ("Bańbura, M., Giannone, D., Modugno, M. and Reichlin, L. (2013). Now-casting and the real-time data flow. In Handbook of Economic Forecasting, vol. 2A, 195-237. Elsevier. doi:10.1016/B978-0-444-53683-9.00004-9",
          "Altissimo, F., Cristadoro, R., Forni, M., Lippi, M. and Veronese, G. (2010). New Eurocoin: tracking economic growth in real time. Review of Economics and Statistics 92(4), 1024-1034. doi:10.1162/REST_a_00045",
          "Baffigi, A., Golinelli, R. and Parigi, G. (2004). Bridge models to forecast the euro area GDP. International Journal of Forecasting 20(3), 447-460. doi:10.1016/S0169-2070(03)00067-0"),
          ("midas", "forecast-workflow"))
add_topic("factor-analysis", "Principal components and factor analysis", """
    Many series often move together because a few forces drive them. Principal
    components (Estimate → Factors → Principal components…) find the weighted averages
    of the series that explain most of their variance, in order; they are descriptive,
    and the usual way of summarising a large panel of indicators by a few factors for a
    regression or a forecast. Bai and Ng's criteria suggest how many to keep.

    Factor analysis (Estimate → Factors → Factor analysis…) is a model: each series is a
    combination of a few common factors plus a part of its own, estimated by maximum
    likelihood. It separates the shared variation (communalities) from each series' own
    (uniquenesses) and tests whether the number of factors is enough. The factors are
    defined only up to a rotation: varimax makes the loadings close to zero or large
    while keeping the factors uncorrelated, promax lets them correlate. Scores estimate
    each observation's factors, by regression (Thomson) or by weighted least squares
    (Bartlett); Data → Add results to dataset… adds them as series.
    """, ("Jolliffe, I. T. (2002). Principal Component Analysis, 2nd ed. Springer. doi:10.1007/b98835",
          "Lawley, D. N. and Maxwell, A. E. (1971). Factor Analysis as a Statistical Method, 2nd ed. Butterworths.",
          "Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica 70(1), 191-221. doi:10.1111/1468-0262.00273"),
          ("nowcasting",))
add_topic("data-connectors", "Data connectors", """
    Data services are read through connectors, each named by the start of an address:
    fred:// for FRED (and ALFRED's vintages), worldbank:// for the World Bank's
    indicators, sdmx:// for the ECB, Eurostat, the OECD, the IMF, the BIS or an
    institution's own SDMX service, and those plugins add. File → Load from connector…
    writes the address from a provider, the series codes and the periods wanted; the
    same address typed as the path in File → Load data…, or given to Dataset.read in
    code, reads the same data.

    Every series records where it came from and, where the provider says, when the data
    were released: its vintage. Where a provider keeps earlier releases (FRED through
    ALFRED), a vintage date gives the data as published that day, so an analysis can be
    repeated with what was known then. A project keeps the data as it was read: it
    opens with the same numbers when the provider has revised them since. File →
    Refresh data… reads the sources again, runs again the steps that changed the data
    and recomputes the results that use it.

    File → Browse series… finds the codes: choose a provider, search by words and pick
    from the results (title, code, frequency, units, first and last period, last
    update, with the provider's notes beside them), or browse FRED's categories, the
    World Bank's sources, topics and countries (a long topic has a folder for each
    indicator-code prefix, NY., SL., ...), or an SDMX provider's dataflows: in the
    provider's own category folders where it publishes useful ones, else in a folder for
    each family such as the IMF's "Monetary and Financial Statistics (MFS)" (newest
    releases first), else by the first part of their IDs (Eurostat's nama_, prc_); the
    tree's header says which, and the search box's words filter it as you type. Click a
    column's heading to sort the results (a third click returns to the provider's order,
    FRED's relevance for a search). With
    a dataflow chosen, the search box searches its codes' names and descriptions ("In
    WEO's codes" beside Search; untick it to search dataflows), and double-clicking a
    code found, or Enter, chooses it. A dimension whose codes cannot be loaded says why
    and takes typed codes. For SDMX, a dataflow's dimensions each get a list of codes, and the series key is
    previewed as it is built (a dimension with nothing chosen is any code). Load reads
    the series at once, recorded as Dataset.read with the address, so the script replays
    without the browser; Add to Load from connector… opens that dialog with the codes
    filled in, to add periods, a vintage or options. In code:
    pymetrix.connectors.search("worldbank", "gdp per capita"), browse(...) and
    structure("sdmx", "ECB/EXR").key(FREQ="M", CURRENCY=["USD", "GBP"]).

    FRED's search needs an API key (free from FRED): set FRED_API_KEY, or type it in the
    browser, where it is kept for the session only. Reading FRED series by ID needs none.
    Searches are kept on disk for a day and lists of categories, indicators, dataflows
    and structures for a week (in ~/.pymetrix/cache/connectors), so browsing again is
    quick; tick "Ask again" to skip them. A key is never written there.
    """, (), ("nowcasting",))
add_topic("large-data", "Large Parquet data", """
    pymetrix keeps a dataset in memory, so a Parquet file larger than memory, or than is
    comfortable, is read in part: only the columns and rows needed. pyarrow does the
    reading (pyarrow.dataset), from one file or a folder of files partitioned by their
    folder names (year=2010/country=DEU/...).

    Look first, without reading data: pymetrix.io.scan("big.parquet") gives the rows,
    row groups and files, and each column's type, smallest and largest value and
    missing values, from the files' footers. Then read part of it:
    Dataset.read("big.parquet", columns=["gdp", "cpi"], where="country in ['DEU', 'FRA']
    and gdp > 0", span=("2000Q1", "2019Q4")). The time and entity columns are always
    read; so are the columns the condition uses, dropped afterwards if not asked for.

    What is pushed down to pyarrow: comparisons of a column with a value (x > 3,
    country == 'DEU', x != 0), in [...] and not in [...], and, on the time axis and date
    columns, < <= > >= with a period or date in quotes; joined by and (each part that
    can be pushed is) or by or (when every part can). span= is pushed the same way. A
    partition folder or a row group whose statistics show no row can match is not read,
    so a condition on a column the file is sorted or partitioned by saves the most.
    What is not pushed: arithmetic and functions (log(x) > 1), comparisons of two
    columns, not, and ==, != and in on periods and dates. Those parts are applied to the
    rows read, and the whole condition is then applied exactly as Dataset.select applies
    it, so the rows kept are those reading the whole file and selecting would keep. The
    load's provenance (meta.extra["loads"]) says what was read: the columns, condition
    and span, the filter pushed down, the parts applied after reading, and the rows, row
    groups and files read against those in the data.

    In the app, File → Load data… on a Parquet file larger than 100 MB shows Part of a
    large file: the columns to tick, a row condition and the first and last periods.

    Projects keep a dataset as it was read: a selection is saved as selected, not the
    whole file. For a read too large to copy into the project, untick Keep a copy in the
    project (embed=False in code): the project keeps the file's path and a fingerprint
    of its contents (from each file's size and footer), reads the file again on opening,
    and refuses to open if the file has changed or gone, rather than compute other numbers
    without saying so; load_project(path, changed_data="read") reads it as it is now.

    Everything after reading is in memory as before: estimation, models and matrices
    (dense) work on the rows and columns read. The known gaps page lists what a lazy
    engine would add.
    """, ("Apache Parquet. File format. https://parquet.apache.org/docs/file-format/",),
    ("samples", "data-connectors", "programs-and-projects"))
add_topic("algebraic-notation", "Algebraic notation", """
    Equations can be shown as an economist writes them: cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ
    for the formula c ~ y + c[-1]. Turn it on with View → Algebraic notation, or
    set_notation("algebraic") in code. Summaries are then titled with the equation and
    their coefficients labelled β₀, β₁ and so on, and notebooks typeset equations. Every
    fitted equation's window has an Equation tab with the code, algebraic, estimated and
    LaTeX forms whichever notation is on.

    Parameters with Greek names, typed as α and β or spelled out as alpha and beta, show
    as Greek letters; a parameter's digits and anything after an underscore become
    subscripts (b1 is b₁, phi_pi is φ_π). Time is a subscript (c[-1] is cₜ₋₁), and d(),
    dlog() and pch() show as Δ, Δlog and %Δ. A parameter's or series' own symbol, set in
    its metadata, is used instead when given.

    Notation only changes what is shown: scripts, generated code and saved projects keep
    the formulas as typed, so a project means the same whichever notation is on.
    """, (), ("least-squares", "model-solution"))
add_topic("chart-look", "How charts look", """
    Every PyMetrix chart has one look, in the app, scripts, notebooks, reports and saved
    files: a light grey grid of major lines behind the data (on bar charts across the
    values only), no top or right frame, short ticks pointing outwards, and series in
    the Okabe-Ito colours, in a fixed order, which stay distinct with the common kinds of
    colour blindness. Text is in the app's own font in the app and in DejaVu Sans
    elsewhere. A theme (publication, teaching, policy, monochrome) changes sizes, colours
    and fonts on top of this look; Edit… styling comes after that.

    The look is set for PyMetrix's charts only: matplotlib's global settings, and so your
    own matplotlib plots, are left as you set them. Under a dark app palette charts are
    drawn on the window's colour with light text, and dark greys such as an "actual" line
    turn light; anything copied, saved or put in a report is light. Charts are drawn at
    the screen's full resolution, and a chart's size in inches never changes with the
    screen or the theme.
    """, ("Okabe, M. and Ito, K. (2008). Color Universal Design (CUD): how to make figures and presentations that are friendly to colorblind people. J*Fly, jfly.uni-koeln.de/color.",), ())
add_topic("defaults", "Defaults, and other packages' defaults", """
    Results that differ from another package's usually differ in a default, not in the
    method. These are PyMetrix's defaults where packages are known to differ; set the
    option shown to match. Where a comparison is not listed, check the other package's
    documentation before comparing numbers.

    Robust (White) standard errors: HC1, scaled by n/(n-k), as Stata's vce(robust).
    Cluster-robust: the G/(G-1) x (n-1)/(n-k) small-sample factor, as Stata.

    Newey-West (HAC): Bartlett kernel with floor(4(n/100)^(2/9)) lags, scaled by
    n/(n-k); give hac_lags to fix the lag. Stata's newey has no default lag.

    ADF unit-root test: lags chosen by AIC up to 12(n/100)^(1/4). EViews chooses by the
    Schwarz criterion (lags="bic" matches it); Stata's dfuller uses no lags unless told.

    VAR residual covariance: degrees-of-freedom adjusted (divided by T - k), as
    statsmodels; impulse responses and their bands use it. Stata's var divides by T
    unless dfk is given, so its responses are slightly smaller.

    SUR and 3SLS: residual covariance divided by n, as Stata's sureg and reg3 without
    dfk; df_correction=True divides by sqrt((n - k_i)(n - k_j)).

    HP filter: lambda = 1600 (f/4)^2, that is 100, 1600 and 14,400 for annual,
    quarterly and monthly data, as EViews; power=4 gives Ravn and Uhlig's rule.

    Seasonal adjustment: STL by default (no external program); X13() uses the Census
    Bureau's X-13ARIMA-SEATS, which statistical offices and EViews use; TramoSeats()
    runs TRAMO-SEATS in Eurostat's JDemetra+ 3 (Java), the ESS's reference software.
    Both are in the pymetrix-seasonal plugin.

    Bayesian VAR: Minnesota prior with tightness 0.2 and lag decay 2, in conjugate
    form, as in Giannone, Lenza and Primiceri; other packages' Minnesota priors differ
    in their scaling, so compare hyperparameters with care.

    Coefficient tests: t distributions for single-equation least squares; normal (z)
    for systems, GMM and maximum likelihood. Intervals and bands are 95 per cent unless
    stated (fan charts 30, 60 and 90).
    """, (), ("least-squares", "unit-roots", "var", "systems", "trend-cycle-filters", "seasonal-adjustment", "bayesian-var"))
add_topic("ardl", "ARDL models and the bounds test", """
    An autoregressive distributed lag (ARDL) model regresses a series on its own lags and
    on current and lagged values of other series. Its long-run multipliers, the sums of
    each regressor's coefficients divided by one less the sum of the autoregressive
    coefficients, describe the level relationship the series settle into.

    Pesaran, Shin and Smith's bounds test asks whether such a level relationship exists
    without first deciding whether each series is I(0) or I(1): an F test whose critical
    values lie between two bounds, the all-I(0) and the all-I(1) cases. Above the upper
    bound there is a relationship; below the lower bound there is not; between them the
    test cannot decide. The error-correction form rewrites the model in differences
    with the lagged levels: the coefficient on the lagged dependent level is the speed
    at which deviations from the long run close.
    """, ("Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. Journal of Applied Econometrics 16(3), 289-326. doi:10.1002/jae.616",
          "Pesaran, M. H. and Shin, Y. (1999). An autoregressive distributed-lag modelling approach to cointegration analysis. In Strøm, S. (ed.), Econometrics and Economic Theory in the 20th Century: The Ragnar Frisch Centennial Symposium, 371-413. Cambridge University Press. doi:10.1017/CCOL0521633230.011"),
          ("cointegration", "unit-roots"))
add_topic("markov-switching", "Markov-switching models", """
    A Markov-switching regression lets the intercept, and optionally the coefficients and
    the error variance, take different values in a small number of unobserved regimes:
    expansion and recession, high and low inflation. Which regime holds in a period is
    not observed; it follows a Markov chain whose transition probabilities are estimated
    with the coefficients, by maximum likelihood with Hamilton's filter.

    The transition matrix gives the probability of moving from each regime to each other
    one; the expected duration of a regime is one over the probability of leaving it.
    Filtered probabilities use the data up to each period; smoothed probabilities use
    the whole sample. The likelihood often has several local maxima, so the fit tries
    random starting values (seeded, so results repeat). Regimes are numbered in the order
    estimated: read each regime's coefficients before naming it.
    """, ("Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica 57(2), 357-384. doi:10.2307/1912559",
          "Kim, C.-J. and Nelson, C. R. (1999). State-Space Models with Regime Switching. MIT Press. doi:10.7551/mitpress/6444.001.0001"),
          ("state-space",))
add_topic("unit-roots", "Unit roots and stationarity", """
    A series with a unit root has shocks that never die out, and regressions between
    such series can be spurious. ADF, Phillips-Perron and DF-GLS test the null of a
    unit root; KPSS tests the null of stationarity; Zivot-Andrews allows one break at an unknown date and Perron's
    test one at a known date. Ng and Perron's M tests and Elliott, Rothenberg and Stock's point-optimal test detrend
    by GLS for better power and size.
    Cointegration (Engle-Granger, Phillips-Ouliaris, Johansen) asks whether a
    combination of unit-root series is stationary.

    ADF's and Engle-Granger's critical values and p-values are MacKinnon's response surfaces, which are asymptotic; in
    short samples (13 to 30 observations) they are far off, and the lag search moves them further. Choose Critical
    values: simulated to simulate them for the sample instead: random walks of the sample's length, each tested exactly
    as the data are (the same trend terms and lag rule). The p-value then has a Monte Carlo standard error, shown with
    it; more draws make it smaller. Impulse or break dummies in Engle-Granger's levels regression go in Exog, and enter
    every simulated regression as given.

    In panels, short series make single-series tests weak, so tests pool across
    entities. Fisher-type tests (Maddala and Wu; Choi) combine the entities' ADF
    p-values; their null is that every series has a unit root, and rejection means at
    least some are stationary. Im, Pesaran and Shin average the entities' ADF
    t-statistics, with the same null and alternative. Levin, Lin and Chu pool the ADF
    regressions with one common coefficient, so their alternative is that every series
    is stationary with the same persistence: a strong assumption, though the test is
    powerful when it holds. These tests assume the entities are independent; a common
    shock, such as a world business cycle, makes them reject too often. Pesaran's CIPS
    test adds the cross-section average to each entity's ADF regression to absorb one
    common factor, and is the usual choice for macroeconomic panels; it takes unbalanced panels, with the null simulated
    for the same missing periods. Breitung's test pools like Levin, Lin and Chu but needs no bias adjustment, so it keeps
    its power when the series have trends; Breitung and Das's robust version allows a common factor (it needs many
    entities and more periods than entities). Hadri's test reverses the null (every series stationary)
    by averaging the entities' KPSS statistics; with autocorrelated errors it rejects
    far too often unless a long-run variance is used (give lags).
    """, ("Hamilton, J. D. (1994). Time Series Analysis, chapters 17 (\"Univariate processes with unit roots\"), 18 (\"Unit roots in multivariate time series\") and 19 (\"Cointegration\"). Princeton University Press. doi:10.2307/j.ctv14jx6sm",
          "Phillips, P. C. B. and Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. Econometrica 58(1), 165-193. doi:10.2307/2938339",
          "Maddala, G. S. and Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. Oxford Bulletin of Economics and Statistics 61(S1), 631-652. doi:10.1111/1468-0084.0610s1631",
          "Choi, I. (2001). Unit root tests for panel data. Journal of International Money and Finance 20(2), 249-272. doi:10.1016/s0261-5606(00)00048-6",
          "Im, K. S., Pesaran, M. H. and Shin, Y. (2003). Testing for unit roots in heterogeneous panels. Journal of Econometrics 115(1), 53-74. doi:10.1016/s0304-4076(03)00092-7",
          "Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. Journal of Applied Econometrics 22(2), 265-312. doi:10.1002/jae.951",
          "Levin, A., Lin, C.-F. and Chu, C.-S. J. (2002). Unit root tests in panel data: asymptotic and finite-sample properties. Journal of Econometrics 108(1), 1-24. doi:10.1016/s0304-4076(01)00098-7",
          "Hadri, K. (2000). Testing for stationarity in heterogeneous panel data. Econometrics Journal 3(2), 148-161. doi:10.1111/1368-423x.00043",
          "Breitung, J. (2000). The local power of some unit root tests for panel data. Advances in Econometrics 15, 161-178. doi:10.1016/s0731-9053(00)15006-6",
          "Breitung, J. and Das, S. (2005). Panel unit root tests under cross-sectional dependence. Statistica Neerlandica 59(4), 414-433. doi:10.1111/j.1467-9574.2005.00299.x"), ("cointegration", "panel-data"))
add_topic("var", "Vector autoregressions", """
    A VAR regresses each variable on lags of all variables. Lag length is chosen by
    information criteria; impulse responses and variance decompositions need an
    identification scheme (Cholesky ordering, long-run restrictions) to give shocks
    an economic meaning.
    """, ("Kilian, L. and Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis. doi:10.1017/9781108164818",), ("cointegration",))
add_topic("staggered-did", "Staggered difference-in-differences", """
    When units adopt a policy at different times, the usual two-way fixed effects regression compares later adopters
    with units already treated. If effects change over time those comparisons are biased, and their weights can even
    be negative, so the estimate can have the wrong sign. The Goodman-Bacon decomposition shows how much rests on them.
    Callaway and Sant'Anna's and Sun and Abraham's estimators compare each adoption cohort only with units not yet
    treated, and report effects by time since adoption; the effects before adoption test parallel trends.
    """, ("Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics 225(2), 200-230. doi:10.1016/j.jeconom.2020.12.001",
          "Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. Journal of Econometrics 225(2), 254-277. doi:10.1016/j.jeconom.2021.03.014",
          "Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics 225(2), 175-199. doi:10.1016/j.jeconom.2020.09.006"), ("treatment-effects",))
add_topic("simulation-estimation", "Estimation by simulation", """
    Some models are easy to simulate but have no likelihood that can be written down: stochastic volatility,
    diffusions observed at intervals, DSGE models with nonlinear solutions. Estimation by simulation picks the
    parameters whose simulated data look like the real data. Indirect inference matches the estimates of a simpler
    auxiliary model (often an AR); the simulated method of moments matches chosen moments; the efficient method of
    moments sets the auxiliary model's score, averaged over a long simulation, to zero. The shocks are drawn once
    and held fixed, so the objective is smooth; standard errors are inflated by (1 + 1/S) for S simulated paths, and
    with more summaries than parameters the J statistic tests the model.
    """, ("Duffie, D. and Singleton, K. J. (1993). Simulated moments estimation of Markov models of asset prices. Econometrica 61(4), 929-952. doi:10.2307/2951768",
          "Gallant, A. R. and Tauchen, G. (1996). Which moments to match? Econometric Theory 12(4), 657-681. doi:10.1017/s0266466600006976",
          "Gourieroux, C., Monfort, A. and Renault, E. (1993). Indirect inference. Journal of Applied Econometrics 8(S1), S85-S118. doi:10.1002/jae.3950080507",
          "Martin, V., Hurn, S. and Harris, D. (2013). Econometric Modelling with Time Series, chapter 12, \"Estimation by simulation\". Cambridge University Press. doi:10.1017/cbo9781139043205"), ("maximum-likelihood",))
add_topic("structural-var", "Structural VARs by A-B restrictions", """
    A structural VAR relates the reduced-form residuals u to independent structural shocks e by A u = B e. Fixing
    some entries of A and B (often zeros) identifies the shocks; the rest are estimated by maximum likelihood. A
    recursive A with a diagonal B is the Cholesky ordering; zeros elsewhere express economic timing (output does not
    respond to policy within the quarter). Zeros of the long-run effects can be combined with zeros on impact. More
    restrictions than needed can be tested against the data with a likelihood-ratio test.

    Other routes: the shock that explains most of a variable's forecast-error variance over some horizons (max
    share; with no effect on impact, a news shock); sign restrictions sharpened by narrative ones (the shock was
    positive, or the main driver, at a known date); and shifts in the shocks' variances between regimes, which
    identify the impact matrix with no zeros at all if the shocks' variances change by different amounts.
    """, ("Amisano, G. and Giannini, C. (1997). Topics in Structural VAR Econometrics, 2nd ed. Springer. doi:10.1007/978-3-642-60623-6",
          "Blanchard, O. J. and Quah, D. (1989). The dynamic effects of aggregate demand and supply disturbances. American Economic Review 79(4), 655-673. jstor:1827924",
          "Galí, J. (1999). Technology, employment, and the business cycle: do technology shocks explain aggregate fluctuations? American Economic Review 89(1), 249-271. doi:10.1257/aer.89.1.249",
          "Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis, chapter 9, \"Structural VARs and VECMs\". Springer. doi:10.1007/978-3-540-27752-1",
          "Antolín-Díaz, J. and Rubio-Ramírez, J. F. (2018). Narrative sign restrictions for SVARs. American Economic Review 108(10), 2802-2829. doi:10.1257/aer.20161852",
          "Barsky, R. B. and Sims, E. R. (2011). News shocks and business cycles. Journal of Monetary Economics 58(3), 273-289. doi:10.1016/j.jmoneco.2011.03.001",
          "Rigobon, R. (2003). Identification through heteroskedasticity. Review of Economics and Statistics 85(4), 777-792. doi:10.1162/003465303772815727",
          "Uhlig, H. (2004). Do technology shocks lead to a fall in total hours worked? Journal of the European Economic Association 2(2-3), 361-371. doi:10.1162/154247604323068050"), ("var",))
add_topic("nonparametric", "Nonparametric estimation", """
    Nonparametric methods let the data choose the shape: a kernel density estimates a distribution without assuming
    it normal; kernel regression estimates E[y | x] as a weighted average of nearby observations (Nadaraya-Watson) or
    a weighted local line (local linear), the bandwidth setting how near is near, chosen by cross-validation.
    Robinson's partially linear model keeps a linear part for the coefficients of interest and leaves one control's
    effect free. A nonparametric autoregression asks how y depends on its past with no functional form; the
    semiparametric GARCH estimates the shocks' density instead of assuming it normal. Many regressors need a lot of
    data: the curse of dimensionality.
    """, ("Engle, R. F. and Gonzalez-Rivera, G. (1991). Semiparametric ARCH models. Journal of Business and Economic Statistics 9(4), 345-359. doi:10.1080/07350015.1991.10509863",
          "Li, Q. and Racine, J. S. (2007). Nonparametric Econometrics: Theory and Practice. Princeton University Press.",
          "Martin, V., Hurn, S. and Harris, D. (2013). Econometric Modelling with Time Series, chapters 11 (\"Nonparametric estimation\") and 20 (\"Nonlinearities in mean\"). Cambridge University Press. doi:10.1017/cbo9781139043205",
          "Robinson, P. M. (1988). Root-N-consistent semiparametric regression. Econometrica 56(4), 931-954. doi:10.2307/1912705",
          "Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. Chapman and Hall."), ("garch",))
add_topic("count-series", "Counts and durations over time", """
    Counts (claims, strikes, trades) and durations (the time between trades) need models that respect their nature.
    INAR(1) carries over each unit of last period's count with some probability and adds new arrivals; a Poisson
    autoregression lets the expected count depend on last period's count and expectation, as GARCH does for
    variance. The ACD model does the same for expected durations, and the ACH turns them into the probability of
    an event each period. A dynamic probit is a probit with the lagged
    outcome among its regressors. A bilinear model lets the effect of last period's shock depend on last period's
    level.
    """, ("Al-Osh, M. A. and Alzaid, A. A. (1987). First-order integer-valued autoregressive (INAR(1)) process. Journal of Time Series Analysis 8(3), 261-275. doi:10.1111/j.1467-9892.1987.tb00438.x",
          "Engle, R. F. and Russell, J. R. (1998). Autoregressive conditional duration: a new model for irregularly spaced transaction data. Econometrica 66(5), 1127-1162. doi:10.2307/2999632",
          "Fokianos, K., Rahbek, A. and Tjostheim, D. (2009). Poisson autoregression. Journal of the American Statistical Association 104(488), 1430-1439. doi:10.1198/jasa.2009.tm08270"), ("maximum-likelihood", "garch"))
add_topic("synthetic-did", "Synthetic difference-in-differences", """
    Difference-in-differences needs the treated and control groups to trend in parallel; the synthetic control needs
    a weighted control to match the treated units' level. Synthetic DiD asks for less of each: it weights the control
    units so that their pre-treatment trend parallels the treated units', weights the pre-treatment periods to look
    like the post-treatment ones, and then takes the weighted double difference. Standard errors come from the
    jackknife over units, or from placebo treatments of control units when only one unit is treated.
    """, ("Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S. (2021). Synthetic difference-in-"
          "differences. American Economic Review 111(12), 4088-4118. doi:10.1257/aer.20190159",), ("treatment-effects", "staggered-did"))
add_topic("double-ml", "Double/debiased machine learning", """
    When the controls are many, or enter in unknown ways, the effect of a treatment can be estimated by learning the
    controls' part flexibly (lasso, random forests, boosted trees, or least squares) and then solving a score that is
    insensitive to small mistakes in what was learned (Neyman orthogonality). Cross-fitting learns each fold's
    nuisance functions on the other folds, so overfitting does not leak into the estimate. The partially linear model
    gives a constant effect (with an instrument, the partially linear IV model); the interactive model gives the
    average effect, or the average effect on the treated, of a 0/1 treatment by the doubly robust (AIPW) score.
    Splitting the sample several times and taking the median removes most of the dependence on one random split.
    Machine learning does not fix confounding: the controls must still contain every confounder.
    """, ("Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018). Double/debiased "
          "machine learning for treatment and structural parameters. Econometrics Journal 21(1), C1-C68. doi:10.1111/ectj.12097",
          "Robinson, P. M. (1988). Root-N-consistent semiparametric regression. Econometrica 56(4), 931-954. doi:10.2307/1912705"),
          ("treatment-effects",))
add_topic("shap-values", "SHAP values", """
    A machine-learning fit has no coefficients to read. SHAP values split each prediction into a base value (the
    average prediction) and one contribution per regressor: the Shapley values of a game whose players are the
    regressors and whose payoff is the prediction with only some of them known. They add up to the prediction
    exactly, a regressor the model ignores gets zero, and two regressors the model uses alike get the same. Averaged
    as mean |SHAP|, they rank the regressors by how much the predictions move with each, a companion to permutation
    importance.

    For trees (random forests, boosted trees, LightGBM, XGBoost), TreeSHAP computes them exactly from the trees
    themselves, with an absent regressor handled by following each split in proportion to the training observations
    that went each way. For other learners they come from the model's predictions, with absent regressors drawn from
    a background sample of the training periods: exactly, over every subset of regressors, when there are at most
    ten; otherwise from a seeded sample of orderings, whose seed and number of evaluations are recorded. Either way
    they describe the fitted model, not causal effects, and with correlated regressors the credit each gets depends
    on how absence is defined.
    """, ("Shapley, L. S. (1953). A value for n-person games. In Kuhn, H. W. and Tucker, A. W. (eds.), Contributions to the Theory of "
          "Games, volume 2, 307-317. Princeton University Press.",
          "Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information "
          "Processing Systems 30, 4765-4774.",
          "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. "
          "(2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2(1), 56-67."),
          ("conformal-prediction", "forecast-workflow"))
add_topic("conformal-prediction", "Conformal prediction intervals", """
    A conformal interval covers a new outcome with a chosen probability without assuming anything about the model:
    it is calibrated on the model's own errors. Split conformal holds back the last part of the sample, estimates the
    model on the rest, and takes the interval as the prediction plus or minus the k-th smallest absolute error of the
    n held-back periods, k = ceil((n + 1)(1 - alpha)). If the held-back periods and the new one are exchangeable, the
    interval covers with probability at least 1 - alpha, and no more than 1 - alpha + 1/(n + 1). Conformalised
    quantile regression calibrates the band between two quantile learners in the same way, so its width follows the
    learners' estimate of the spread.

    Exchangeability is the catch for time series: a trend, a break or changing volatility makes the past errors a
    poor guide to the next, and the coverage can fall short. Adaptive conformal inference moves the level after each
    outcome, alpha_t+1 = alpha_t + gamma (alpha - miss_t), so a run of misses widens the intervals until they cover
    again; over T periods the share of misses is within (max(alpha, 1 - alpha) + gamma) / (gamma T) of alpha whatever
    the data. That is coverage on average over time, not in each period, and the guarantee needs outcomes as they
    arrive: true forecasts beyond the data keep the last level.
    """, ("Vovk, V., Gammerman, A. and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.",
          "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018). Distribution-free predictive inference for "
          "regression. Journal of the American Statistical Association 113(523), 1094-1111.",
          "Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized quantile regression. Advances in Neural Information Processing "
          "Systems 32, 3543-3553.",
          "Gibbs, I. and Candès, E. (2021). Adaptive conformal inference under distribution shift. Advances in Neural Information "
          "Processing Systems 34, 1660-1672."),
          ("shap-values", "forecast-workflow"))
add_topic("shift-share", "Shift-share (Bartik) instruments", """
    A shift-share instrument combines regions' exposure to industries (the shares) with shocks to those industries
    (the shifts): regions exposed to industries that boomed get a large instrument. Identification can rest on the
    shocks being as good as randomly assigned across industries (Borusyak, Hull and Jaravel), checked and estimated
    in the equivalent regression across industries; or on the shares being exogenous (Goldsmith-Pinkham, Sorkin and
    Swift), where the Rotemberg weights show which industries' shares matter most. Either way, regions with similar
    shares have correlated instruments, so regional standard errors are too small: Adao, Kolesar and Morales's
    errors (AKM, and AKM0 with the null imposed) and the shock-level regression's errors allow for that.
    """, ("Adao, R., Kolesar, M. and Morales, E. (2019). Shift-share designs: theory and inference. Quarterly Journal of Economics 134(4), "
          "1949-2010. doi:10.1093/qje/qjz025",
          "Borusyak, K., Hull, P. and Jaravel, X. (2022). Quasi-experimental shift-share research designs. Review of Economic Studies 89(1), "
          "181-213. doi:10.1093/restud/rdab030",
          "Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik instruments: what, when, why, and how. American Economic Review "
          "110(8), 2586-2624. doi:10.1257/aer.20181047"), ("treatment-effects",))
add_topic("interactive-fixed-effects", "Interactive fixed effects", """
    Two-way fixed effects remove a level for each unit and a shock for each period that hits every unit alike.
    Interactive fixed effects let units respond differently to common shocks: a few unobserved factors F_t with
    loadings lambda_i, which may be correlated with the regressors (a recession that hits manufacturing regions harder,
    with their regressors moving too). Bai's estimator alternates between least squares given the factors and
    principal components of the residuals given the coefficients. The number of factors can be chosen by Bai and Ng's
    information criteria. It needs many units and many periods; with additive effects removed first and no factors it
    is the two-way fixed effects estimator.
    """, ("Bai, J. (2009). Panel data models with interactive fixed effects. Econometrica 77(4), 1229-1279. doi:10.3982/ECTA6135",
          "Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica 70(1), 191-221. "
          "doi:10.1111/1468-0262.00273"), ("panel-data",))
add_topic("conditional-logit", "Conditional logit", """
    Two models share one likelihood. Chamberlain's fixed-effects logit conditions on each unit's number of ones,
    which removes the unit's own intercept, so the coefficients are consistent with a short panel, where a logit with
    unit dummies is biased (the incidental parameters problem); only units whose outcome changes contribute.
    McFadden's conditional logit models which of several alternatives each decision maker picks, from the
    alternatives' attributes: with exactly one choice per case it is the same likelihood, each case a group. Its
    probabilities assume the independence of irrelevant alternatives.

    Marginal effects (Next, Marginal effects, or clogit_effects): after the choice model, how each alternative's
    probability moves with each attribute of each alternative (own effects negative for a cost, cross effects
    positive, each column summing to zero) and with each characteristic of the decision maker, as derivatives and
    elasticities, averaged over the cases or at the means, with delta-method standard errors. The fixed-effects
    logit's probabilities depend on the fixed effects it conditions away, so it has no marginal effects; it reports
    odds ratios, and, only if asked, effects for a unit with a fixed effect of a chosen value. For 0/1 variables,
    tick 0/1 variables for discrete changes (the change in the probabilities when the variable goes from 0 to 1), and
    list others under Changes (e.g. cost: 1) for the change when a variable rises by that amount.
    """, ("Chamberlain, G. (1980). Analysis of covariance with qualitative data. Review of Economic Studies 47(1), 225-238. doi:10.2307/2297110",
          "McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (ed.), Frontiers in Econometrics, "
          "105-142. New York: Academic Press."), ("maximum-likelihood",))
add_topic("nested-and-mixed-logit", "Nested and mixed logit", """
    The conditional logit assumes the independence of irrelevant alternatives: the odds between two alternatives do
    not depend on the others, so a new alternative draws from every other in proportion to its share. Two models
    relax it, for the same long data (one row per case and alternative).

    The nested logit groups alternatives in nests (bus and train in "public") whose unobserved utilities are
    correlated: IIA holds within a nest, not across nests. Each nest has a dissimilarity (log-sum) parameter lambda.
    pymetrix writes the model in McFadden's form consistent with random utility maximisation (Hensher and Greene's
    "RU2": utilities divided by their nest's lambda inside the nest's inclusive value). With every lambda 1 it is the
    conditional logit exactly; between 0 and 1 it is consistent with utility maximisation; near 0 the nest's
    alternatives are near-perfect substitutes; above 1 is a warning sign. A nest of one alternative has no lambda (it
    cancels), so it is fixed at 1; lambdas can be constrained equal. Estimation is full-information maximum
    likelihood, and the likelihood-ratio test of every lambda = 1 against the conditional logit is a test of IIA.

    The mixed (random-parameters) logit lets coefficients vary across decision makers, normal or lognormal (positive:
    enter the negative of a price), uncorrelated or correlated; with repeated choices by one decision maker (a
    panel), their choices share one draw of the coefficients. Its probabilities are integrals over the coefficients'
    distribution, simulated with Halton draws (deterministic) or pseudo-random ones, and the simulated likelihood is
    maximised. Too few draws bias the estimates: estimate again with more and compare. Each decision maker's
    conditional mean of a coefficient, given their choices, describes where they lie in the distribution.

    Marginal effects (fit.marginal_effects()): derivatives and elasticities of each alternative's probability in
    each attribute of each alternative and each characteristic of the case, with delta-method standard errors, as
    after the conditional logit; unlike there, a column's cross elasticities differ.
    """, ("McFadden, D. (1978). Modelling the choice of residential location. In A. Karlqvist, L. Lundqvist, F. Snickars and J. W. Weibull (eds.), "
          "Spatial Interaction Theory and Planning Models, 75-96. Amsterdam: North-Holland.",
          "Hensher, D. A. and Greene, W. H. (2002). Specification and estimation of the nested logit model: alternative normalisations. "
          "Transportation Research Part B: Methodological 36(1), 1-17.",
          "McFadden, D. and Train, K. (2000). Mixed MNL models for discrete response. Journal of Applied Econometrics 15(5), 447-470.",
          'Train, K. E. (2009). Discrete Choice Methods with Simulation, 2nd ed., chapters 4 ("GEV"), 6 ("Mixed logit"), 9 ("Drawing from '
          'densities"), 10 ("Simulation-assisted estimation") and 11 ("Individual-level parameters"). Cambridge University Press.'),
          ("conditional-logit", "maximum-likelihood"))
add_topic("survival-analysis", "Survival analysis", """
    Durations, such as time to finding a job, to default or to relapse, are often censored: for some subjects the
    event has not happened when observation ends. The Kaplan-Meier estimator gives the probability of surviving past
    each time as a product over the event times of the share of those at risk who survive it; censored subjects count
    as at risk until they leave. Its intervals are pointwise (log-log by default, as Stata; log as R's survfit), from
    Greenwood's variance. The log-rank test compares the events seen in each group with those expected if survival
    were the same in every group; the Wilcoxon (Gehan-Breslow) and Tarone-Ware versions give early events more
    weight.

    Cox's proportional hazards model leaves the baseline hazard unspecified and estimates how regressors shift it:
    exp(coefficient) is the hazard ratio, the same at every time. The partial likelihood needs a rule for events at
    the same time: Breslow's (the default here, as in Stata and statsmodels) or Efron's (R's default, closer to the
    exact likelihood with many ties). Strata give each stratum its own baseline hazard; an entry time handles
    subjects observed only after some time at risk (left truncation); robust standard errors are Lin and Wei's. The
    proportional-hazards assumption is checked by Grambsch and Therneau's test: the scaled Schoenfeld residuals
    should not trend with time.
    """, ("Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical "
          "Association 53(282), 457-481.",
          "Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. Cancer Chemotherapy "
          "Reports 50(3), 163-170.",
          "Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society, Series B (Methodological) 34(2), 187-220.",
          "Breslow, N. (1974). Covariance analysis of censored survival data. Biometrics 30(1), 89-99.",
          "Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. Journal of the American Statistical Association "
          "72(359), 557-565.",
          "Lin, D. Y. and Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. Journal of the American Statistical "
          "Association 84(408), 1074-1078.",
          "Grambsch, P. M. and Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. Biometrika 81(3), "
          "515-526."), ("maximum-likelihood",))
add_topic("large-and-switching-vars", "BVARs with stochastic volatility, FAVARs and Markov-switching VARs", """
    Three ways to let a VAR handle more. A Bayesian VAR with stochastic volatility keeps its coefficients fixed but
    lets each shock's size drift, which matters for forecast bands and for telling large shocks from changed
    dynamics. A factor-augmented VAR summarises a large panel of series by a few principal components and puts them in
    a VAR with the policy variable, so a policy shock's effect can be read on any series in the panel; the number of
    factors can be left to Bai and Ng's criteria, and the bands bootstrap the whole two-step procedure. A
    Markov-switching VAR lets the intercepts, the variances or everything change with an unobserved regime, such as
    recession and expansion, whose probabilities it estimates period by period. Its likelihood has local maxima:
    EM starts from regime paths read off the data (splits by levels and by residual variances, clustering,
    one-variable fits) as well as random ones, so clearly separated regimes are found; if the estimates change with
    Starts or the seed, keep the fit with the highest log likelihood.

    Their variance and historical decompositions are recursive. A BVAR with stochastic volatility splits the
    forecast-error variance with one date's volatilities, and the history draw by draw with each period's own impact
    matrix, so both come with bands over the posterior draws. A Markov-switching VAR splits the variance within a
    regime held fixed, and the history along its most likely regime path, that regime's intercepts in the baseline;
    its responses and decompositions get bands from Draws: draws of its parameters from the maximum-likelihood
    estimates' asymptotic normal distribution (the regime path held fixed in the historical decomposition). A
    quantile VAR splits the variance of its quantile system (quantile connectedness); it has no historical
    decomposition, since away from the median its residuals are not centred.
    """, ("Bernanke, B. S., Boivin, J. and Eliasz, P. (2005). Measuring the effects of monetary policy: a factor-augmented vector autoregressive (FAVAR) approach. Quarterly Journal of Economics 120(1), 387-422. doi:10.1162/0033553053327452",
          "Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica 70(1), 191-221. doi:10.1111/1468-0262.00273",
          "Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. Review of Economics and Statistics 80(2), 218-230. doi:10.1162/003465398557465",
          "Carriero, A., Clark, T. E. and Marcellino, M. (2019). Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors. Journal of Econometrics 212(1), 137-154. doi:10.1016/j.jeconom.2019.04.024",
          "Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica 57(2), 357-384. doi:10.2307/1912559",
          "Krolzig, H.-M. (1997). Markov-Switching Vector Autoregressions, chapter 2, \"The Markov-switching vector autoregressive model\". Springer. doi:10.1007/978-3-642-51684-9_2",
          "Ando, T., Greenwood-Nimmo, M. and Shin, Y. (2022). Quantile connectedness: modeling tail behavior in the topology of financial networks. Management Science 68(4), 2401-2431. doi:10.1287/mnsc.2021.3984",
          "Diebold, F. X. and Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers, with application to global equity markets. Economic Journal 119(534), 158-171. doi:10.1111/j.1468-0297.2008.02208.x"), ("var", "bayesian-var", "tvp-var"))
add_topic("spectral-analysis", "Spectra and long memory", """
    The spectrum splits a series' variance by frequency: how much comes from slow swings, from business-cycle
    fluctuations and from quarter-to-quarter noise. The periodogram is its raw estimate; smoothing over neighbouring
    frequencies trades resolution for precision. The cross-spectrum's coherence measures how closely two series move
    together at each frequency, and its phase which leads. A long-memory series has a spectrum that rises without
    bound at frequency zero, like f(lambda) ~ lambda^-2d: d between 0 and 0.5 is stationary but persistent, between 0.5
    and 1 mean-reverting but not stationary. GPH and the local Whittle estimate d from the lowest frequencies; ARFIMA
    models it with short-run ARMA dynamics.
    """, ("Geweke, J. and Porter-Hudak, S. (1983). The estimation and application of long memory time series models. Journal of Time Series Analysis 4(4), 221-238. doi:10.1111/j.1467-9892.1983.tb00371.x",
          "Granger, C. W. J. and Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. Journal of Time Series Analysis 1(1), 15-29. doi:10.1111/j.1467-9892.1980.tb00297.x",
          "Hamilton, J. D. (1994). Time Series Analysis, chapters 6 (\"Spectral analysis\") and 10 (\"Covariance-stationary vector processes\"). Princeton University Press. doi:10.2307/j.ctv14jx6sm",
          "Robinson, P. M. (1995). Gaussian semiparametric estimation of long range dependence. Annals of Statistics 23(5), 1630-1661. doi:10.1214/aos/1176324317"), ("unit-roots",))
add_topic("spatial", "Spatial econometrics", """
    Neighbouring regions, firms or households affect each other. A spatial weights matrix says who neighbours whom
    (here the k nearest, or any matrix supplied), usually scaled so each row sums to one. Moran's I measures whether
    neighbours' residuals are alike; Anselin's LM tests say whether a spatial lag (neighbours' outcomes feed into one's
    own: y = rho W y + X b + e) or a spatial error (neighbours share unobserved shocks) fits better. Both models are
    estimated by maximum likelihood; least squares on a spatial lag model is inconsistent.
    """, ("Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer. doi:10.1007/978-94-015-7799-1",
          "Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. Regional Science and Urban Economics 26(1), 77-104. doi:10.1016/0166-0462(95)02111-6",
          "Ord, K. (1975). Estimation methods for models of spatial interaction. Journal of the American Statistical Association 70(349), 120-126. doi:10.1080/01621459.1975.10480272"), ("least-squares", "maximum-likelihood"))
add_topic("tvp-var", "TVP-VARs with stochastic volatility", """
    A VAR assumes the economy's dynamics and the size of its shocks never change. A time-varying-parameter VAR lets
    its coefficients, the contemporaneous relations and the shocks' volatilities drift as random walks, so it can tell
    whether responses changed (did monetary policy's effect change?) or only the size of the shocks did (good policy
    or good luck?). It is estimated by Gibbs sampling, the priors from a training sample; responses come at any date.
    """, ("Del Negro, M. and Primiceri, G. E. (2015). Time varying structural vector autoregressions and monetary policy: a corrigendum. Review of Economic Studies 82(4), 1342-1345. doi:10.1093/restud/rdv024",
          "Kim, S., Shephard, N. and Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. Review of Economic Studies 65(3), 361-393. doi:10.1111/1467-937x.00050",
          "Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. Review of Economic Studies 72(3), 821-852. doi:10.1111/j.1467-937x.2005.00353.x"), ("var", "bayesian-var"))
add_topic("proxy-svar", "Proxy SVARs", """
    A proxy (or external-instrument) SVAR identifies one structural shock with a series from outside the VAR, such as
    the surprise in futures prices around policy announcements, or a narrative record of tax changes. If the proxy is
    correlated with that shock and with no other, the covariance of the VAR's residuals with it gives the shock's
    impact on every variable, without ordering them. Its responses are those of the VAR to that impact. At horizon 0
    they equal the LP-IV estimate with the same lags; beyond it, the VAR's dynamics extrapolate where LP-IV does not.

    Its historical decomposition splits each variable into the identified shock's contribution, the other shocks'
    contribution together (they are not identified one by one) and the baseline from the first observations; the
    shock is the part of the residuals along its impact column. Bands come from the same moving-block bootstrap.
    """, ("Jentsch, C. and Lunsford, K. G. (2019). The dynamic effects of personal and corporate income tax changes in the United States: comment. American Economic Review 109(7), 2655-2678. doi:10.1257/aer.20162011",
          "Mertens, K. and Ravn, M. O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. American Economic Review 103(4), 1212-1247. doi:10.1257/aer.103.4.1212",
          "Stock, J. H. and Watson, M. W. (2018). Identification and estimation of dynamic causal effects in macroeconomics using external instruments. Economic Journal 128(610), 917-948. doi:10.1111/ecoj.12593"), ("var", "local-projections"))
add_topic("smooth-transition", "Smooth-transition autoregressions", """
    A smooth-transition autoregression lets a series follow one AR in one state and another in a second, moving
    between them smoothly as a lag of the series changes, rather than jumping at a threshold. The logistic form (LSTAR)
    suits recessions and expansions; the exponential form (ESTAR) suits mean reversion that strengthens far from
    equilibrium, as with real exchange rates. Test linearity first (Teräsvirta): the transition's speed and centre
    are not identified when there is none.
    """, ("Teräsvirta, T. (1994). Specification, estimation, and evaluation of smooth transition autoregressive models. Journal of the American Statistical Association 89(425), 208-218. doi:10.1080/01621459.1994.10476462",
          "van Dijk, D., Teräsvirta, T. and Franses, P. H. (2002). Smooth transition autoregressive models: a survey of recent developments. Econometric Reviews 21(1), 1-47. doi:10.1081/etc-120008723"), ("forecast-workflow",))
add_topic("growth-at-risk", "Growth at risk", """
    Growth at risk asks how bad growth could be, not only what it is expected to be. Quantile regressions of future
    growth on today's conditions show that tight financial conditions lower the low quantiles of growth much more
    than the median: downside risk moves, while the centre stays put. Matching each period's predicted quantiles with
    a skewed t gives a whole predictive distribution, its 5% quantile (growth at risk) and its expected shortfall.
    """, ("Adrian, T., Boyarchenko, N. and Giannone, D. (2019). Vulnerable growth. American Economic Review 109(4), 1263-1289. doi:10.1257/aer.20161923",
          "Azzalini, A. and Capitanio, A. (2003). Distributions generated by perturbation of symmetry with emphasis on a multivariate skew t-distribution. Journal of the Royal Statistical Society, Series B (Statistical Methodology) 65(2), 367-389. doi:10.1111/1467-9868.00391",
          "Chernozhukov, V., Fernández-Val, I. and Galichon, A. (2010). Quantile and probability curves without crossing. Econometrica 78(3), 1093-1125. doi:10.3982/ecta7880"), ("quantile-regression", "local-projections"))
add_topic("panel-var", "Panel VARs", """
    A panel VAR fits one VAR to many entities (firms, countries, regions), each with its own fixed effect. With few
    periods per entity, demeaning biases the lags towards zero (Nickell), so the standard estimator transforms each
    variable by forward orthogonal deviations, which removes the fixed effect without that bias, and instruments the
    transformed lags with their levels (GMM). More instrument lags than lags over-identify the model, and Hansen's J
    tests them. Granger tests, impulse responses and variance decompositions work as for a VAR.

    A historical decomposition is entity by entity: each entity's data split into the shocks' contributions and a
    baseline from its first observations and its fixed effect (its mean of the data less the lags' effect). The
    window shows one entity at a time; the dialog's Entity, or hd.select(entity), picks it.
    """, ("Abrigo, M. R. M. and Love, I. (2016). Estimation of panel vector autoregression in Stata. Stata Journal 16(3), 778-804. doi:10.1177/1536867x1601600314",
          "Arellano, M. and Bover, O. (1995). Another look at the instrumental variable estimation of error-components models. Journal of Econometrics 68(1), 29-51. doi:10.1016/0304-4076(94)01642-d",
          "Holtz-Eakin, D., Newey, W. and Rosen, H. S. (1988). Estimating vector autoregressions with panel data. Econometrica 56(6), 1371-1395. doi:10.2307/1913103",
          "Love, I. and Zicchino, L. (2006). Financial development and dynamic investment behavior: evidence from panel VAR. Quarterly Review of Economics and Finance 46(2), 190-210. doi:10.1016/j.qref.2005.11.007"), ("var",))
add_topic("bayesian-equations", "Bayesian estimation of an equation", """
    Bayesian estimation combines what the data say with what is known beforehand: a
    prior for each coefficient (and the error's standard deviation), stated here by
    its mean and standard deviation. The result is the posterior: a distribution for
    each coefficient, summarised by its mean, median, standard deviation and highest
    posterior density (HPD) interval, the shortest interval holding, say, 90% of it.
    A tight prior pulls an estimate towards its mean; a coefficient with no prior is
    left to the data (a flat prior), so with no priors at all the posterior mean is
    close to least squares.

    The posterior is explored by drawing from it. With normal (or no) priors on the
    coefficients of a linear equation, each step can be drawn exactly (Gibbs
    sampling); otherwise, and for logit, probit and Poisson equations, random-walk
    Metropolis-Hastings proposes moves and accepts some. The No-U-Turn sampler (NUTS,
    sampler "nuts") instead follows the posterior's slope, moving far at each step, so
    fewer draws carry as much information; its burn-in tunes its step size and scales,
    and a divergent transition warns that the draws near it may be biased. Its mass
    matrix (Metric) is diagonal by default; dense also undoes correlations between
    coefficients, and mode keeps the curvature at the posterior mode. Several chains started apart
    should agree: R-hat near 1 and large effective sample sizes suggest they do.
    Draws at the start of each chain (the burn-in) are discarded.

    The posterior mean of the draws is itself an estimate of the posterior mean, with a
    Monte Carlo standard error (MCSE): the posterior standard deviation over the square
    root of the effective sample size. Correlated draws are worth fewer independent ones,
    so their MCSE is larger. Draw more until it is small beside the standard deviation and
    the precision you report.
    """, ("Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15, 1593-1623.",
          "Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.",
          "Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013). Bayesian Data Analysis, 3rd ed. CRC Press. doi:10.1201/b16018",
          "Koop, G. (2003). Bayesian Econometrics. Wiley."), ("bayesian-var", "least-squares"))
add_topic("bayesian-var", "Bayesian VARs", """
    A VAR has many coefficients for the data available: with 4 variables and 4 lags,
    each equation estimates 17. Least squares then fits noise, and forecasts suffer.
    A Bayesian VAR shrinks the coefficients towards a simple benchmark, by an amount
    the data can override.

    The Minnesota prior (Litterman) centres each equation on a random walk (each
    variable's own first lag 1, everything else 0; use 0 for growth rates), shrinks
    longer lags more, and scales each coefficient by the relative volatility of the
    two variables. (Litterman's original also shrinks other variables' lags more than
    a variable's own; the conjugate form cannot, the independent form can, see below.) Its tightness
    sets the strength:
    smaller values shrink more. The sum-of-coefficients prior and the single-unit-root
    (initial observation) prior, added as dummy observations, pull towards unit roots
    and cointegration, which helps forecasts of trending data.

    In its conjugate normal-inverse-Wishart form the posterior is known exactly, so
    draws are cheap and exact, and the marginal likelihood (how well the prior and
    model predicted the data) can be computed. Choosing the hyperparameters to maximise
    it lets the data decide how much to shrink (Giannone, Lenza and Primiceri). Impulse
    responses and forecasts from the posterior draws include parameter uncertainty;
    their bands are credible intervals. The conjugate form ties cross-variable
    shrinkage to own-variable shrinkage. The independent normal-Wishart form
    (NormalWishart) frees it: other variables' lags get their own tightness
    (cross_tightness), and each equation's prior no longer depends on the others' error
    variances. Its posterior is sampled by Gibbs, alternating the coefficients given
    the error covariance and the covariance given the coefficients; it has no closed-form
    marginal likelihood, but Chib's method estimates it from the draws, and its
    hyperparameters can be chosen by that estimate (more slowly).

    The hierarchical prior of Giannone, Lenza and Primiceri (GLP) treats the
    hyperparameters as unknowns with priors of their own: rather than one best
    tightness, it weighs each by how well it predicted the data and by how plausible
    it is beforehand. Its hyperparameters can be fixed at their most probable values
    or integrated over (MCMC), so that the bands also reflect doubt about how much
    to shrink.
    """, ("Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: five years of experience. Journal of Business and Economic Statistics 4(1), 25-38. doi:10.2307/1391384",
          "Doan, T., Litterman, R. and Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. Econometric Reviews 3(1), 1-100. doi:10.1080/07474938408800053",
          "Sims, C. A. and Zha, T. (1998). Bayesian methods for dynamic multivariate models. International Economic Review 39(4), 949-968. doi:10.2307/2527347",
          "Bańbura, M., Giannone, D. and Reichlin, L. (2010). Large Bayesian vector auto regressions. Journal of Applied Econometrics 25(1), 71-92. doi:10.1002/jae.1137",
          "Giannone, D., Lenza, M. and Primiceri, G. E. (2015). Prior selection for vector autoregressions. Review of Economics and Statistics 97(2), 436-451. doi:10.1162/rest_a_00483",
          "Chib, S. (1995). Marginal likelihood from the Gibbs output. Journal of the American Statistical Association 90(432), 1313-1321. doi:10.1080/01621459.1995.10476635",
          "Kadiyala, K. R. and Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR models. Journal of Applied Econometrics 12(2), 99-132. doi:10.1002/(sici)1099-1255(199703)12:2<99::aid-jae429>3.0.co;2-a"),
          ("var",))
add_topic("cointegration", "Cointegration and VECMs", """
    Unit-root series are cointegrated when a combination of them is stationary: they
    share a long-run equilibrium. A vector error-correction model adds the deviations
    from that equilibrium to a VAR in differences; Johansen's trace and
    maximum-eigenvalue tests choose the number of cointegrating relations. A single relation can be estimated by
    fully modified OLS, canonical cointegrating regression or dynamic OLS, whose z statistics are standard normal
    (Estimate → VARs and local projections → Cointegrating regression; Start and End limit the sample, and a
    condition on the rows under Advanced leaves rows out, the rows kept then taken as consecutive periods). Their
    long-run covariance uses the Bartlett kernel with Newey and West's (1994) automatic bandwidth, as arch 9 and
    R's sandwich choose it. Gregory and Hansen's test allows the relation to break once. In panels with long time
    series, the mean group and pooled mean group estimators let each unit have its own dynamics, the latter with a common long run, and
    common correlated effects absorb factors common to all units (Test → Gregory-Hansen cointegration test, and
    Estimate → VARs and local projections → Panel mean group and CCE).

    Panel cointegration tests have the null of no cointegration. Pedroni's seven statistics test each unit's
    cointegrating regression residuals for a unit root: the four panel statistics pool them (a common autoregressive
    coefficient under the alternative), the three group statistics average them (each unit its own). Kao's tests pool
    one regression with unit constants and a common slope. Westerlund's tests ask whether each unit's error-correction
    coefficient is zero: Gt and Ga average the units' (the alternative: some units correct), Pt and Pa pool them (all
    do). All assume the units are independent; with a strong common factor, demean the data by period first.
    """, ("Johansen, S. (1995). Likelihood-Based Inference in Cointegrated Vector Autoregressive Models. doi:10.1093/0198774508.001.0001",
          "Gregory, A. W. and Hansen, B. E. (1996). Residual-based tests for cointegration in models with regime shifts. Journal of Econometrics 70(1), 99-126. doi:10.1016/0304-4076(69)41685-7",
          "Pesaran, M. H. (2006). Estimation and inference in large heterogeneous panels with a multifactor error structure. Econometrica 74(4), 967-1012. doi:10.1111/j.1468-0262.2006.00692.x",
          "Pesaran, M. H., Shin, Y. and Smith, R. P. (1999). Pooled mean group estimation of dynamic heterogeneous panels. Journal of the American Statistical Association 94(446), 621-634. doi:10.1080/01621459.1999.10474156",
          "Phillips, P. C. B. and Hansen, B. E. (1990). Statistical inference in instrumental variables regression with I(1) processes. Review of Economic Studies 57(1), 99-125. doi:10.2307/2297545",
          "Stock, J. H. and Watson, M. W. (1993). A simple estimator of cointegrating vectors in higher order integrated systems. Econometrica 61(4), 783-820. doi:10.2307/2951763",
          "Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. Oxford Bulletin of Economics and Statistics 61(s1), 653-670. doi:10.1111/1468-0084.61.s1.14",
          "Pedroni, P. (2004). Panel cointegration: asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. Econometric Theory 20(3), 597-625. doi:10.1017/s0266466604203073",
          "Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. Journal of Econometrics 90(1), 1-44. doi:10.1016/s0304-4076(98)00023-2",
          "Westerlund, J. (2007). Testing for error correction in panel data. Oxford Bulletin of Economics and Statistics 69(6), 709-748. doi:10.1111/j.1468-0084.2007.00477.x"), ("unit-roots", "var", "panel-data"))
add_topic("derived-series", "Derived series", """
    A derived series keeps the recipe that made it: a formula such as dlog(y), or a
    filter such as the HP trend of y. When a series it uses changes, it is recomputed,
    and so is anything derived from it. Plain series made with with_series keep their
    values (like EViews' genr); derived series follow their inputs (like its
    auto-updating series). Recipes are saved with the data. Two-sided filters revise
    their whole history when the series changes, just as they do when a new release
    arrives; results estimated on the old values are then flagged out of date. Taking
    a subset of a dataset, or converting its frequency, keeps the values but drops the
    recipes. Formulas can hold conditions, which give 1 where true and 0 where false
    (missing where a value is missing): comparisons (<, <=, >, >=, ==, !=) joined with
    and, or and not, as in recession = gy < 0. They make dummies, regressors written in
    an equation (c ~ y + (gy < 0)) and the shading of charts. Besides d(), dlog(),
    pch(), log(), exp() and the rest, formulas can use distribution functions such as
    normcdf(x) and tinv(p, df), and moving-window statistics such as movavg(x, 4) and
    movcorr(x, y, 12) (see Distribution functions and moments).

    Rounding: round(x) and round(x, n) round to n decimals with halves away from zero,
    as MATLAB and Excel do (round(2.5) is 3); floor(x) and ceil(x) round down and up;
    log10(x) is the base-10 logarithm. Missing values: na is the missing value, isna(x)
    is 1 where x is missing and 0 where not, and fillna(x, v) is v where x is missing.
    iif(c, a, b) is a where the condition holds, b where not, and missing where the
    condition is: iif(gy > 0, gy, na). From the data's rows: year(), quarter(),
    month() and day() of each period's first day, and obsnum(), the observation's
    number within each entity; normrnd(mu, sigma, seed=s) and unifrnd(a, b, seed=s)
    draw one number a row, seeded so the draws are the same each time and when the
    project is replayed. These work on data (Generate series, data.eval, regressors
    written in an equation), not in models; random draws are values, so make them
    with Generate series rather than as a derived series.
    """, (), ("trend-cycle-filters", "distributions-and-moments"))
add_topic("distributions-and-moments", "Distribution functions and moments", """
    Formulas (derived series, equations, measurement maps and model text) can use
    distribution functions named as in MATLAB and Dynare: normpdf, normcdf and norminv
    (x, then optionally mu and sigma: normcdf(x) is the standard normal), tpdf, tcdf
    and tinv (x, df), chi2pdf, chi2cdf and chi2inv (x, df), fpdf, fcdf and finv (x,
    d1, d2), gampdf, gamcdf and gaminv (x, a, b: shape a and scale b, so the mean is
    a*b, as in MATLAB), betapdf, betacdf and betainv (x, a, b), lognpdf, logncdf and
    logninv (x, mu, sigma of log x), unifpdf, unifcdf and unifinv (x, a, b, default 0
    and 1), exppdf, expcdf and expinv (x, mu: the mean, not the rate), poisspdf and
    poisscdf (k, lambda), binopdf and binocdf (k, n, p), and erf and erfc. The inv
    functions are quantiles: norminv(0.975) is 1.96; poissinv (p, lambda) and binoinv
    (p, n, p) are MATLAB's, the smallest integer k with cdf(k) >= p, and have no
    derivative (a model cannot be linearised through them). Arguments may be expressions and
    parameters, and models using these functions are linearised with their exact
    derivatives; where no derivative is offered (in the degrees of freedom of t,
    chi-squared and F, the shape parameters of gamma and beta, a binomial's n),
    linearisation stops with a message rather than an approximation. Dynare and IRIS
    files can use normcdf, normpdf and erf; the others cannot be exported to them.

    Moving-window statistics look back over the last n periods, within each entity of a
    panel: movavg(x, n), movsum, movsd, movvar, movmin and movmax, and between two
    series movcov(x, y, n), movcorr(x, y, n) and movbeta(y, x, n), the slope of a
    rolling regression of y on x. They are missing until a full window of data is
    there, and variances use n - 1. Centred windows put the window around t:
    cmovavg(x, 5) uses x[+2] ... x[-2] (an even n one more lag than leads), and every
    moving function has one (cmovsum ... cmovbeta); a model reads their leads as
    leads. Expanding windows use every period up to t, within each entity: cumavg(x),
    cumsum, cumsd, cumvar, cummin, cummax, cumcov(x, y), cumcorr(x, y) and
    cumbeta(y, x), as pandas' expanding(); they work on data only, not in models.

    For a dataset, describe() gives count, mean, standard deviation, percentiles,
    skewness and excess kurtosis (bias-corrected, as pandas and Excel compute them; 0
    for a normal); corr() Pearson, Spearman or Kendall correlations; cov() covariances;
    partial_corr() correlations holding other series constant; crosscorrelogram()
    the correlations of x(t) with y(t+k) at leads and lags; and autocov() the
    autocovariances of a series.
    """, ("Johnson, N. L., Kotz, S. and Balakrishnan, N. (1994). Continuous Univariate Distributions, 2nd ed.",
          "Joanes, D. N. and Gill, C. A. (1998). Comparing measures of sample skewness and kurtosis. The Statistician 47(1), 183-189.",
          "Box, G. E. P., Jenkins, G. M., Reinsel, G. C. and Ljung, G. M. (2015). Time Series Analysis, 5th ed., chapter 12 (cross-correlations)."),
          ("derived-series", "diagnostic-tests"))
add_topic("selections-and-loops", "Selections and loops", """
    A selection is an ordered set of series names: listed, chosen by pattern (*_cycle) or
    by metadata (units, adjustment), like column selectors in R, polars and Julia. Loops
    over a selection replace EViews' for %s loops: the dialogs (Derived series for each
    selected, Equation for each selected series, Test each selected series) and the
    selection methods (formulas, fit, test, map) use templates with {s} for the series
    name, and return results side by side. In Python, for name in selection: works
    as for any list, f-strings replace %s substitution, and ordinary variables replace !i
    scalars. Everything run in the Immediate or Code window is recorded, loops included.
    """, (), ("derived-series",))
add_topic("generate-for-each", "Generate series for each", """
    Data > Generate series makes one series from a formula; fill in For each and it
    makes one for each of several series. For each takes names (gdp cons inv),
    wildcards (*_sa, ip_*: * any run of characters, ? one character) or both; in a
    dataset's window, pick series in the series list and press Generate for each.
    Series and Formula are then templates: {x} is each series, so {x}_g with
    100*d(log({x})) makes gdp_g = 100*d(log(gdp)) and so on, and {1}, {2}... are the
    parts the wildcards matched, so ip_* with {1}_growth turns ip_us into us_growth
    (each * takes as much as it can, left first). {{ and }} are literal braces. The
    preview lists every new series and its formula before anything is made, and
    flags names that are not valid, formulas that do not parse or use series the
    dataset lacks, names made twice, and series that exist already: tick Replace to
    overwrite them, as for a single series. Keep updated makes them derived series.
    The step records the expansion as one ordinary call listing every new series,
    data = data.with_series(gdp_g="100*d(log(gdp))", cons_g=...), or derive with Keep
    updated, so the script says exactly what was made and replays the same whatever
    series are added later; reopening the step shows the templates again. In Python,
    data.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))") does the same and is
    recorded the same way, and pymetrix.data.each.preview_each shows the rows. On a
    panel each series is computed within each entity, as for any formula.
    """, (), ("derived-series", "selections-and-loops"))
add_topic("samples", "Samples", """
    A sample chooses the rows a command uses: a span of periods, entities of a panel
    (country: DEU, FRA), a span per entity, and a condition on the data, in any
    combination. Name one with Data > Sample > New sample (Sample(...) in code) and make it a
    dataset's active sample with Data > Sample > Set sample (data.with_sample(...)), the
    equivalent of EViews' smpl: estimation, tests, tables and graphs on that dataset
    then use it unless they are given another span. It belongs to the dataset, and the
    line that sets it is recorded, so scripts and projects replay it. A sample only
    selects rows: lags still reach back before it, never across entities, and generated
    series are computed over all the data. Conditions are kept as
    text and applied to the data each time; results record the sample they used. In
    code, samples combine with & (and), | (or) and ~ (not).

    To assign a series only within a sample, as EViews' smpl followed by y = ... does,
    give the sample under Within in Data > Generate series (with_series(y="...",
    within=Sample(...)) in code): the sample's rows are assigned and the others keep
    their values, missing in a new series. A formula that uses the series it assigns,
    such as y[-1]*1.02, is then computed one period at a time, each period seeing the
    values just assigned, as in EViews. with_series(y=None) makes a series of missing
    values (EViews' series y). File > New dataset makes an empty dataset of a frequency
    from a first to a last period (Dataset.empty("Q", "1990Q1", "2030Q4"); EViews'
    wfcreate).

    A condition uses column names, numbers, text and dates in quotes (time >= '2000Q1'),
    comparisons, in [...] (country in ['DEU', 'FRA']), and, or, not, arithmetic and the
    functions abs, sqrt, exp, log, log10, log1p, expm1, floor, ceil, the trigonometric ones
    and arctan2. Anything else (@ references, attribute access with a dot, other functions)
    is refused before it runs, so a condition in someone else's project cannot run code.
    """, (), ("selections-and-loops",))
add_topic("state-space", "State space and the Kalman filter", """
    A state-space model links observed series to unobserved states, such as potential
    output, the trend, the cycle or the neutral rate, that evolve with shocks. The Kalman
    filter estimates the states period by period from the data up to then (filtered,
    real-time estimates); the smoother uses all the data (smoothed, two-sided estimates).
    The likelihood from the filter gives maximum-likelihood estimates of the parameters,
    or the posterior mode when parameters have priors. Unit-root states such as trends
    start diffuse (unknown); stationary states such as cycles and gaps start from their
    stationary distribution. The HP filter is the smoothed trend of a smooth-trend model
    with signal-to-noise ratio 1/lambda.

    With diffuse states the log likelihood is the exact diffuse likelihood (Durbin and
    Koopman 2012, section 7.2.2): the first d periods, until the diffuse states are pinned
    down, each add -(log 2 pi + log F_inf)/2 per observation, the later ones the usual
    -(log 2 pi + log F + v^2/F)/2. statsmodels' UnobservedComponents and SARIMAX report by
    default the likelihood of the later periods given the first d (Harvey 1989), higher by
    d/2 log 2 pi + 1/2 sum log F_inf: 7.367 for a local linear trend with a quarterly
    seasonal (d = 5). Diffuse periods in the likelihood (Advanced; diffuse_terms="exclude"
    in code) reports that convention. Estimates and standard errors do not change, only
    the level of the log likelihood, AIC, BIC and HQ. Compare criteria across models only
    in one convention and with the same diffuse states; never compare one package's level
    with another's without checking which terms each counts.
    """, ("Durbin, J. and Koopman, S. J. (2012). Time Series Analysis by State Space Methods, 2nd ed. doi:10.1093/acprof:oso/9780199641178.001.0001",
          "Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. doi:10.1017/cbo9781107049994",
          "Hamilton, J. D. (1994). Time Series Analysis, chapter 13, \"The Kalman filter\". Princeton University Press. doi:10.2307/j.ctv14jx6sm"), ("trend-cycle-filters",))
add_topic("custom-estimators", "Writing your own estimator or solver", """
    NumPy, pandas and SciPy are ready in the Immediate and Code windows as np, pd and
    sp, and pymetrix.toolkit has the building blocks PyMetrix's own estimators use.
    toolkit.design(equation, data) turns a formula and a sample into arrays (y, X, names,
    rows used; lags, fixed effects and missing values handled); toolkit.least_squares,
    toolkit.covariance (classic, white, hac, Cluster), toolkit.sandwich, toolkit.hessian
    and toolkit.gradient do the inference; toolkit.result builds the fitted equation,
    with statistics and provenance. An estimator is a frozen dataclass of settings with
    an estimate(eq, data, ...) method; a solver has solve_period(system). Register either
    with toolkit.register("methods", ...) or ("solvers", ...) and it appears in the
    dialogs and works in scripts, projects, group loops and side-by-side tables like a
    built-in one. The repository's examples folder has a ridge-regression estimator and
    a damped Newton solver; pymetrix.testing.check_entry checks yours.
    """, (), ("least-squares", "model-solution"))
add_topic("local-projections", "Local projections", """
    Local projections estimate impulse responses directly: for each horizon h, regress the
    response at t+h on the shock at t and on lags of the variables; the coefficient on the
    shock is the response at horizon h. They need no model of the whole system, so they
    are robust to its misspecification, at the cost of less precision than a VAR. With the
    same lags and recursive ordering they reproduce a VAR's responses at impact, and the
    two estimate the same responses in large samples. Errors overlap across horizons, so
    standard errors use Newey-West with h+1 lags, or heteroskedasticity-robust errors
    after adding one more lag of every variable (lag augmentation). A state indicator
    gives state-dependent responses, e.g. in recessions and expansions.
    """, ("Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. American Economic Review 95(1), 161-182. doi:10.1257/0002828053828518",
          "Montiel Olea, J. L. and Plagborg-Møller, M. (2021). Local projection inference is simpler and more robust than you think. Econometrica 89(4), 1789-1823. doi:10.3982/ecta18756",
          "Plagborg-Møller, M. and Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. Econometrica 89(2), 955-980. doi:10.3982/ecta17813",
          "Ramey, V. A. and Zubairy, S. (2018). Government spending multipliers in good times and in bad: evidence from US historical data. Journal of Political Economy 126(2), 850-901. doi:10.1086/696277"),
          ("var",))
add_topic("model-solution", "Solving models", """
    A model is a system of estimated equations and identities solved period by period
    for its endogenous variables. Newton's method solves all equations jointly using
    their derivatives; Gauss-Seidel updates each equation's own variable in turn and is
    cheaper per iteration but may need damping to converge. Models with expected future
    values need the forward-looking methods.

    By default a period has converged when every equation's residual is below the
    tolerance, a test in the equations' own units; the step test (Convergence: step)
    stops instead when no variable moves by more than the tolerance relative to its size,
    whatever the units. Newton's line search halves a step that would not reduce the
    residuals, for starting values far from the solution, and Refresh: period keeps one
    Jacobian a period while convergence is fast. Fallback retries a period the solver
    cannot solve with damped Gauss-Seidel, then Newton with a line search; Ordering:
    blocks solves the recursive equations once each, in order, and iterates only on the
    simultaneous blocks, much less work for a mostly recursive model.

    A stochastic simulation solves the model many times, each time with errors drawn for
    the estimated equations (normal, with their standard errors and correlations, or the
    residuals resampled), and perhaps their coefficients too. The spread of the draws
    shows how uncertain the paths are: quantiles give fan charts. For a nonlinear model
    the draws' mean can differ from the deterministic solution. A sample of residuals can
    be given instead (Error draws: The data's residuals): historical errors by equation, resampled a whole
    period at a time, as the Federal Reserve's stochastic simulations of FRB/US draw them.

    Add factors (Model > Solution > Add factors) are each equation's residual at the data over a
    span: solved with them, the model reproduces the data, so a scenario reads as a
    deviation from that baseline. Optimal control (Model > Policy > Optimal control) moves
    exogenous instruments, period by period, to minimise a discounted quadratic loss with
    the model solved as the constraint. Models in the FRB/US XML format, and EViews
    state-space objects, are read under Model > Import.

    A fixed or held variable can apply in some periods only (Scenario partial; in Model >
    Solution > Solve model, Hold only where given under Advanced, for holds): where its
    path has a value the variable is exogenous, elsewhere its equation determines it. Before solving, Model > Check model (or
    missing_inputs in code) lists each period, equation and input the solve would lack.
    A solution's Convergence view shows, period by period, the iterations and the largest
    residual left with its equation; a period that fails to converge stops the solve, and
    the error (its convergence table in code) names that period.
    """, ("Hall, S. G. and Henry, S. G. B. (1988). Macroeconomic Modelling. North-Holland.",
          "Fair, R. C. (1980). Estimating the expected predictive accuracy of econometric models. International Economic Review 21(2), 355-378. doi:10.2307/2526186"),
          ("howto-solve-a-large-backward-looking-model",))
add_topic("matrices", "Matrices with names", """
    A Matrix keeps names for its rows and columns, so matrix algebra checks that they meet: A @ B needs A's columns
    to be B's rows, in the same order, and a mistake in ordering is an error rather than a wrong number. The usual
    functions are methods (A.T, A.inv(), A.pinv(), A.chol(), A.power(2), A.svd(), A.eig(), A.kron(B)) or Matrix's
    (Matrix.cov(fit), Matrix.var_lag(var, 1), Matrix.lyapunov(F, Q), Matrix.hcat(A, B)).

    In the app, Model > Matrix makes a named matrix as a step: from an expression in that syntax, using the
    session's objects by name ((X.T @ X).inv() @ X.T @ y); from a fit's coefficients or covariance matrix, a VAR's
    lag matrix or error covariance, a dataset's series over a span, or a formula's regressors and dependent variable;
    or from numbers typed or pasted from Excel into a grid, with names for the rows and columns. A matrix window's
    Matrix menu transposes, inverts, factors (Cholesky, lower or upper) and decomposes it (SVD, eigenvalues, QR, LU,
    Schur), raises it to a power and solves the Lyapunov equation X = A X A' + Q, each as a new named matrix; it
    also copies the matrix as LaTeX (a bmatrix, or a table with the names), CSV or tab-separated text, and exports
    it to Excel. A fitted result's right-click menu in the workspace offers it As matrix.

    Each step records one line, name = expression, which the script runs to give the same matrix; it is recomputed
    when the objects it uses change. The expression is checked before it runs: names, attribute access, calls,
    operators, indexing and literals only; imports, lambda, attributes starting with an underscore and anything
    that reads or writes files are refused. The dialog checks it as you type, showing the problem (and keeping OK
    disabled) or, when it is cheap to compute, the result's shape and names. The user guide's section on matrices
    covers the methods in full.
    """, ("pymetrix user guide: Matrices with names. https://github.com/GeoffHeenan/pymetrix/blob/main/docs/guide.md#matrices-with-names",
          "Lütkepohl, H. (1996). Handbook of Matrices. Wiley.",
          "Magnus, J. R. and Neudecker, H. (2019). Matrix Differential Calculus with Applications in Statistics and Econometrics, 3rd ed. Wiley."),
          ("state-space",))
add_topic("forward-looking-models", "Forward-looking models", """
    When equations contain expected future values (x[+1]), today's outcome depends on
    tomorrow's, so periods cannot be solved one at a time. Two approaches:

    Perfect foresight (deterministic simulation) assumes agents know the whole future path
    of the exogenous variables. All periods are stacked into one system and solved jointly
    by Newton's method with a sparse Jacobian; values beyond the end of the span come from
    a terminal condition (the last value, the data, or the steady state). It keeps every
    non-linearity, so it suits large, anticipated policy changes.

    Linear rational expectations: linearise the model around its steady state, giving
    A0 E[x(t+1)] + A1 x(t) + A2 x(t-1) + B e(t) = 0, and find the unique stable solution
    x(t) = P x(t-1) + Q e(t) by the QZ (generalised Schur) decomposition. Blanchard and
    Kahn's condition: there must be exactly as many stable roots as predetermined
    variables. Too few, and no stable solution exists; too many, and the solution is not
    unique (indeterminacy, e.g. a Taylor rule responding less than one for one to
    inflation). The solution gives impulse responses (Model > Analysis > Impulse responses:
    one shock, its size and the horizon; in code, irf.paths["y <- e"] is the response of y
    to the shock e by horizon), stochastic simulations, and a state-space form
    for the Kalman filter, to estimate unobserved variables such as the output gap from
    observed data.

    Models written for Dynare can be read from their .mod files: variables, parameters,
    the model block (timing x(+1), model-local variables), steady-state guesses and shock
    standard deviations. Each equation is paired with a variable to solve for, and the
    model solves as any other. Model > Export model writes a model in pymetrix's, Dynare's,
    IRIS's or EViews's syntax, with notes on what does not carry across.

    Once a model is estimated, Model > Analysis > Shock decomposition splits the
    smoothed history of each variable into the contributions of the shocks and of the
    starting point; Data contributions splits it by observed series instead.
    """, ("Blanchard, O. J. and Kahn, C. M. (1980). The solution of linear difference models under rational expectations. Econometrica 48(5), 1305-1311. doi:10.2307/1912186",
          "Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational expectations model. Journal of Economic Dynamics and Control 24(10), 1405-1423. doi:10.1016/s0165-1889(99)00045-7",
          "Uhlig, H. (1999). A toolkit for analysing nonlinear dynamic stochastic models easily. In Marimon, R. and Scott, A. (eds.), Computational Methods for the Study of Dynamic Economies, 30-61. Oxford University Press. doi:10.1093/0199248273.003.0003",
          "Juillard, M. (1996). Dynare: a program for the resolution and simulation of dynamic models with forward variables through the use of a relaxation algorithm. CEPREMAP.",
          "Galí, J. (2015). Monetary Policy, Inflation, and the Business Cycle, 2nd ed."),
          ("model-solution", "state-space"))
add_topic("dynare-files", "Dynare files: measurement errors, generalised priors and macros", """
    A Dynare file read with read_mod (Model > Import > Dynare file...) keeps Dynare's
    meaning in three places that are easy to miss.

    Measurement errors. In the shocks block, stderr or var on a varobs variable is the
    size of its measurement error, and corr y, c = 0.3 (or var y, c = covariance) on two
    varobs variables is the correlation of their errors. The Kalman filter's
    measurement-error covariance is then H = D C D, D the errors' standard deviations
    and C their correlations, as the shocks' covariance is built. In estimated_params,
    corr y, c, ... estimates that correlation, named corr_y_c (the two in varobs order),
    within (-1, 1). For a model of your own, MeasurementMap(..., error_corr={("y", "c"):
    0.3}); naming corr_y_c in estimate= estimates it.

    Generalised priors. A prior line's third and fourth numbers after the mean and
    standard deviation make Dynare's generalised distributions, and the mean and
    standard deviation are those of the generalised distribution: beta_pdf with 3 and 4
    is a beta on (3, 4) (GeneralisedBeta); gamma_pdf, inv_gamma_pdf and inv_gamma2_pdf
    with a third number are shifted to start there (ShiftedGamma, ShiftedInvGamma1,
    ShiftedInvGamma2); for normal_pdf the two are bounds. estimation(prior_trunc=p) keeps
    each parameter between its prior's p and 1 - p quantiles, as Dynare's estimation
    does; without the option no such bounds are set (Dynare's default is 1e-10, which
    matters only far in the tails). One difference: Dynare 6.0's shifted inv_gamma2_pdf
    takes its scale from the mean before the shift, so its mean is the mean given plus
    the shift; pymetrix keeps the mean given, as Dynare's manual describes the prior.

    Macros. Besides @#define, @#if, @#for and @#include, Dynare 6's operations on arrays
    work: a | b (union: a, then what of b is not in it), a & b (intersection), a - b
    (difference), a * b and a ^ n (Cartesian product and power, arrays of tuples), with
    & binding tighter than |; the type tests isreal, isstring, isboolean, istuple and
    isarray; the casts (real), (string), (bool), (tuple) and (array). @{...} writes numbers
    to 15 significant digits and strings without quotes, as Dynare's preprocessor.
    @#line is accepted; problems are still reported at the model file's lines.
    """, ("Adjemian, S., Juillard, M., Karamé, F., Mutschler, W., Pfeifer, J., Ratto, M., Rion, N. and Villemot, S. (2024). Dynare: reference manual, version 6. Dynare Working Papers 80, CEPREMAP.",),
          ("model-solution", "state-space"))
add_topic("global-sensitivity", "Global sensitivity analysis: prior stability mapping", """
    Before estimating a forward-looking model, it is worth knowing where its priors put
    the parameters: in a region where the model has a unique stable solution, or partly
    where it has none or many. Ratto's (2008) prior stability mapping, Dynare's
    dynare_sensitivity, answers this by Monte Carlo filtering. Model > Analysis >
    Stability mapping draws the parameters from their priors, solves the model at each
    draw, and puts each draw in one class: determinate (a unique stable solution:
    Blanchard and Kahn's condition holds), indeterminate (more stable roots than
    predetermined variables), explosive (fewer), no steady state, or no solution (the
    roots count right but the stable roots do not pin down the predetermined variables).

    Then each parameter's values in the determinate draws are compared with its values
    in the others by the Smirnov test (the two-sample Kolmogorov-Smirnov test). Its
    statistic is the largest distance between the two cumulative distributions, drawn on
    the Graph tab. A parameter whose p-value is below 0.001 (Dynare's pvalue_ks) drives
    the model in or out of the determinate region: in a New Keynesian model, the
    inflation response of the Taylor rule, whose determinate draws all lie above one.
    A parameter whose two distributions coincide does not matter for stability on its
    own. The shares are Monte Carlo estimates: their standard error is
    sqrt(p (1 - p) / draws), shown with them. Draws are random from the session's
    generator and seed (Dynare's default is a quasi-random Sobol sequence), so the two
    agree within that error. Shock sizes and measurement errors are not drawn: they do
    not change whether the model solves.

    In code: stability_mapping(model, draws=2000, seed=1) from pymetrix.models, with a
    Dynare model (its estimated_params), a template or a model whose parameters carry
    priors, or priors={"phi_pi": Uniform(0, 3)}. The result's shares, smirnov and
    smirnov_by_class() hold the numbers; draws holds each draw and its class.
    """, ("Ratto, M. (2008). Analysing DSGE models with global sensitivity analysis. Computational Economics 31(2), 115-139.",
          "Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M. and Tarantola, S. (2008). Global Sensitivity Analysis: The Primer. Wiley.",
          "Smirnov, N. (1948). Table for estimating the goodness of fit of empirical distributions. Annals of Mathematical Statistics 19(2), 279-281.",
          "Adjemian, S., Juillard, M., Karamé, F., Mutschler, W., Pfeifer, J., Ratto, M., Rion, N. and Villemot, S. (2024). Dynare: reference manual, version 6. Dynare Working Papers 80, CEPREMAP."),
          ("model-solution", "dynare-files"))
add_topic("likelihood-gradient", "The gradient of a model's likelihood", """
    Estimating a linearised model means climbing its log likelihood (or log posterior) in
    the parameters, and the No-U-Turn sampler moves along its slope: both need the
    gradient. Differencing the likelihood costs two model solutions and Kalman filters per
    parameter and is accurate to about seven digits. pymetrix differentiates it exactly
    instead, for all parameters at once: the steady state (by the implicit-function
    theorem on the static equations), the linearised equations (second derivatives of the
    model's equations), the solution x(t) = P x(t-1) + Q e(t) (a Sylvester-type equation
    from A0 P^2 + A1 P + A2 = 0, as in Iskrev's identification analysis), the measurement
    equations, the shocks' covariance (sizes and correlations), the measurement errors,
    the stationary initial variance (a Lyapunov equation), and then each step of the
    Kalman filter. The optimiser uses it through the transforms that keep parameters
    inside their bounds; the Hessian, and so the standard errors, come from differences of
    the exact gradient; NUTS takes it at every leapfrog step. The filter is differentiated
    backwards (its adjoint, as reverse-mode automatic differentiation would): one pass
    forwards, one back, whatever the number of parameters, so a gradient costs two to four
    likelihood evaluations (seven parameters of the New Keynesian template, 36 of Smets and
    Wouters 2007), not one per parameter. Where it does not apply
    (known-date regimes, the OccBin filter, DSGE-VARs, diffuse starts, growth models) central
    differences are used as before; Gradient in the dialogs chooses. With numba
    installed (pip install pymetrix[fast]) the filter's loops run compiled, about twice
    as fast on small models. Which sampler? NUTS with a dense mass matrix (Metric, the
    default) undoes the correlations between parameters that DSGE posteriors have, and
    gave more effective draws a second than random-walk Metropolis-Hastings on the New
    Keynesian template and on Smets and Wouters; Metric mode keeps the curvature at the
    posterior mode (a short burn-in, good on nearly normal posteriors); with Workers its
    chains run at once in other processes. Where the gradient is by differences,
    Metropolis-Hastings is usually faster.
    """, ("Iskrev, N. (2010). Local identification in DSGE models. Journal of Monetary Economics 57(2), 189-202. doi:10.1016/j.jmoneco.2009.12.007",
          "Ratto, M. and Iskrev, N. (2011). Algorithms for identification analysis under the DYNARE environment: final version of the software. Joint Research Centre, European Commission.",
          "Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press. doi:10.1017/cbo9781107049994",
          "Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15, 1593-1623."),
          ("forward-looking-models", "particle-smoother"))
add_topic("particle-smoother", "The particle smoother", """
    The particle filter estimates a nonlinear model's state from the data up to each
    period; the smoother uses all of the data, as the Kalman smoother does for a linear
    model. pymetrix draws whole paths backwards (backward simulation): starting from a
    particle at the end of the sample, it picks the particle the period before in
    proportion to its filter weight times how well the later data fit when the model is
    run on from that particle with the path's shocks. A pruned second-order model moves
    deterministically once its shocks are known, so this is done in the shocks: each
    path is a start and a sequence of shocks, re-simulated, so it is a path the model
    can produce, with its smoothed shocks. Means and quantiles over the paths are the
    smoothed values (an estimate's smoothed variables when it was estimated with the
    particle filter). More paths reduce the noise; the cost grows with the square of the
    sample's length.
    """, ("Godsill, S. J., Doucet, A. and West, M. (2004). Monte Carlo smoothing for nonlinear time series. Journal of the American Statistical Association 99(465), 156-168. doi:10.1198/016214504000000151",
          "Lindsten, F. and Schön, T. B. (2013). Backward simulation methods for Monte Carlo statistical inference. Foundations and Trends in Machine Learning 6(1), 1-143. doi:10.1561/2200000045",
          "Herbst, E. and Schorfheide, F. (2015). Bayesian Estimation of DSGE Models. Princeton University Press. doi:10.1515/9781400873739"),
          ("forward-looking-models", "likelihood-gradient"))
add_topic("perturbation", "Second- and third-order solutions", """
    A first-order solution treats the future as if it were certain: risk does not change
    decisions, and responses to shocks are proportional to their size. Higher-order
    perturbation approximates the decision rules by a polynomial in last period's states,
    this period's shocks and the size of future uncertainty, from the model's exact
    derivatives at the steady state.

    Second order (Dynare's stoch_simul(order=2)) adds quadratic terms and a constant, the
    correction for risk: means differ from the steady state (precautionary saving, risk
    premia), and responses depend on the size and sign of a shock. Third order (order=3)
    adds cubic terms and corrections for risk to the responses themselves (ghxss, ghuss in
    Dynare's names): how strongly the economy reacts depends on how uncertain the future
    is. The shocks are taken to be normal, so the third-order constant is zero.

    Simulating the polynomial rule directly can explode, because its higher powers feed on
    themselves. Pruning keeps each order's terms acting on the lower-order parts of the
    state only (Kim, Kim, Schaumburg and Sims at second order; Andreasen,
    Fernández-Villaverde and Rubio-Ramírez at third), which keeps paths stable and gives a
    process with well-defined moments; pymetrix prunes by default, as Dynare does with
    its pruning option. The pruned third-order mean equals the pruned second-order mean.

    Impulse responses start from the stochastic steady state (where the model rests when
    shocks are expected but none occur) or from a given state. At third order they can be
    generalised responses: the mean difference, over draws of the other shocks, between
    paths with and without the impulse. Model > Solution > Second-order solution solves to
    order 2 or 3. Second-order solutions can be estimated with the particle filter; third-
    order ones cannot yet.
    """, ("Schmitt-Grohé, S. and Uribe, M. (2004). Solving dynamic general equilibrium models using a second-order approximation to the policy function. Journal of Economic Dynamics and Control 28(4), 755-775.",
          "Kim, J., Kim, S., Schaumburg, E. and Sims, C. A. (2008). Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models. Journal of Economic Dynamics and Control 32(11), 3397-3414.",
          "Andreasen, M. M., Fernández-Villaverde, J. and Rubio-Ramírez, J. F. (2018). The pruned state-space system for non-linear DSGE models: theory and empirical applications. Review of Economic Studies 85(1), 1-49.",
          "Levintal, O. (2017). Fifth-order perturbation solution to DSGE models. Journal of Economic Dynamics and Control 80, 1-16.",
          "Koop, G., Pesaran, M. H. and Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. Journal of Econometrics 74(1), 119-147."),
          ("forward-looking-models", "particle-smoother"))
add_topic("treatment-effects", "Treatment effects and policy evaluation", """
    What would the outcome have been without the policy? Difference-in-differences compares the change in treated
    units with the change in untreated ones, assuming they would have moved in parallel; an event study shows the
    effect period by period, and effects before adoption near zero support the assumption. Regression
    discontinuity compares those just above and just below a cutoff that decides treatment. Weighting on the
    propensity score balances treated and untreated on observed characteristics, assuming nothing unobserved
    drives both. Synthetic control builds one treated unit's counterfactual from a weighted mix of untreated units.
    """, ("Angrist, J. D. and Pischke, J.-S. (2009). Mostly Harmless Econometrics. Princeton University Press. doi:10.1515/9781400829828",
          "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press. doi:10.12987/9780300255881",
          "Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic control methods for comparative case studies: estimating the effect of California's tobacco control program. Journal of the American Statistical Association 105(490), 493-505. doi:10.1198/jasa.2009.ap08746",
          "Imbens, G. W. and Lemieux, T. (2008). Regression discontinuity designs: a guide to practice. Journal of Econometrics 142(2), 615-635. doi:10.1016/j.jeconom.2007.05.001"))

add_topic("diagnostic-tests", "Specification and diagnostic tests", """
    After estimating, test the assumptions behind the standard errors: serial correlation
    (Breusch-Godfrey, Ljung-Box), heteroskedasticity (Breusch-Pagan, White, ARCH),
    normality (Jarque-Bera), functional form (RESET) and stability (Chow, CUSUM).

    Breusch-Godfrey's auxiliary regression uses the equation's own regressors, so an
    equation without a constant has none there either; statsmodels always adds one.
    Tick "Add a constant to the auxiliary regression" (auxiliary_constant=True) for
    statsmodels' number; with a constant in the equation the two agree.
    """, ("Greene, W. H. (2018). Econometric Analysis, 8th ed. Pearson.",))


def _terms_topic() -> None:
    """The help topic "Terms used in pymetrix": the glossary the app's words follow (``pymetrix.wording.GLOSSARY``)."""
    from .wording import glossary_markdown

    add_topic("terms", "Terms used in pymetrix", glossary_markdown())


_terms_topic()


def _coming_from_topics() -> None:
    """The help topics "Coming from EViews" and "Coming from MATLAB/IRIS and Dynare" (``pymetrix.coming_from``)."""
    from .coming_from import add_topics

    add_topics()


_coming_from_topics()


def _howto_topics() -> None:
    """The task pages ("How do I…", ``pymetrix.howto``): an index topic and a topic per page."""
    from .howto import add_topics

    add_topics()


_howto_topics()


def _exercises_topic() -> None:
    """The help topic on exercise packs (``pymetrix.exercises``)."""
    from .exercises import add_help_topic

    add_help_topic()


_exercises_topic()
