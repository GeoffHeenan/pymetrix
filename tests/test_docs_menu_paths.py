"""Menu paths written in the docs and the in-app help ("Estimate → VARs and local projections → VAR…") name menus that
exist: each is followed through the main window's menu bar as built, so a moved or renamed entry fails here rather than
in a reader's hands. Paths through a figure's Edit… button or a window's own menu are not the menu bar's and are not
checked; a plugin's menu is checked only when the plugin is installed, and so is a path in a paragraph that names a
plugin of this repository (``pymetrix-guide``, ``pymetrix[guide]``): "View → Guide" is the guide plugin's panel."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Docs that describe the app as it is now (the changelog, design notes and reviews are history and keep old names).
DOCS = ["README.md", "docs/guide.md", "docs/features.md", "docs/known-gaps.md", "docs/concepts.md", "docs/install.md",
        "docs/modelling-quickstart.md", "docs/how-to-help.md", "docs/tutorials/frbus.md",
        "docs/solving-large-models.md", "docs/how-do-i/README.md",
        *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("src/pymetrix/howto/*.md")),  # the task pages
        *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("packages/*/docs/*.md"))]

# The plugins kept in this repository (packages/pymetrix-*), by their short name: "guide", "finance", ...
BUNDLED = sorted(p.name.removeprefix("pymetrix-") for p in ROOT.glob("packages/pymetrix-*") if p.is_dir())

_ARROW = re.compile(r"\s*(?:→|->)\s*")
_HEAD = re.compile(r"(?<![\w.…])(\w[\w ]*?)\s*(?:→|->)\s*")


def _norm(label: str) -> str:
    return label.replace("&", "").replace("...", "…").rstrip("…").strip().lower()


def menu_tree(window) -> dict:
    """The menu bar as nested dicts of normalised labels."""

    def walk(menu) -> dict:
        out: dict = {}
        for action in menu.actions():
            if action.menu() is not None:
                out[_norm(action.menu().title())] = walk(action.menu())
            elif not action.isSeparator() and action.text():
                out.setdefault(_norm(action.text()), {})
        return out

    tops: dict = {}
    for top in window.menuBar().actions():
        if top.menu() is not None:
            tops[top.menu().title().replace("&", "")] = walk(top.menu())
    return tops


def _without_plugins(text: str, absent: list[str]) -> str:
    """``text`` less its paragraphs that name one of the ``absent`` plugins (their menu paths are theirs)."""
    if not absent:
        return text
    names = re.compile(r"pymetrix[-_](?:%s)\b|pymetrix\[(?:%s)\]" % ("|".join(absent), "|".join(absent)), re.IGNORECASE)
    return "\n\n".join(par for par in re.split(r"\n[ \t]*\n", text) if not names.search(par))


def bad_paths(text: str, tree: dict, known_tops: set[str], absent: list[str] = ()) -> list[str]:
    """The menu paths of ``text`` that the menus do not have: "Top → A → B", each step matched against the labels
    (longest first, the trailing "…" optional), the path ending where the text stops naming a submenu's entry.
    Paragraphs naming a plugin of ``absent`` (not installed) are left out."""
    text = _without_plugins(text, list(absent))
    flat = re.sub(r"[ \t]*\n[ \t]*(?:#\s?)?(?=\S)", " ", text).replace("**", "").replace("`", "")
    bad = []
    for m in _HEAD.finditer(flat):
        head = m.group(1)
        top = next((t for t in sorted(known_tops, key=len, reverse=True)
                    if head.endswith(t) and (len(head) == len(t) or not head[-len(t) - 1].isalnum())), None)
        if top is None or top not in tree:
            continue  # not a menu (or a plugin's menu, the plugin not installed)
        node, rest, path = tree[top], flat[m.end():], [top]
        while True:
            low = rest.lower().replace("...", "…")
            labels = [k for k in node if k and low.startswith(k)]
            if not labels:
                bad.append(" → ".join(path) + " → " + re.split(r"[,;:.)]|\s(?:→|->)\s", rest)[0][:60].strip())
                break
            label = max(labels, key=len)
            path.append(label)
            rest = re.sub(r"^(?:…|\.\.\.)", "", rest[len(label):])
            arrow = _ARROW.match(rest)
            if not arrow:
                break
            if not node[label]:
                bad.append(" → ".join(path) + " is not a submenu")
                break
            node, rest = node[label], rest[arrow.end():]
    return bad


@pytest.fixture(scope="module")
def menus():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841 (kept alive for the window)
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        yield menu_tree(window)
    finally:
        window.close()
        window.deleteLater()


@pytest.fixture(scope="module")
def absent(menus) -> list[str]:
    """The bundled plugins not loaded in the window (not installed here, as in the installed-wheel job)."""
    from pymetrix.plugins import plugins

    loaded = {p.prefix.replace("_", "-") for p in plugins()}
    return [name for name in BUNDLED if name not in loaded]


# Every top-level menu the app or a bundled plugin may show; one missing from the window (a plugin not installed) is
# skipped, while one present is checked.
TOPS = {"File", "Edit", "View", "Data", "Estimate", "Test", "Model", "Forecast", "Calculate", "Finance", "Help"}


def test_the_checker_finds_a_wrong_path_and_passes_right_ones():
    tree = {"Estimate": {"equation": {}, "nonlinear": {"threshold var": {}}}, "Help": {"about pymetrix": {}}}
    assert bad_paths("**Estimate → Nonlinear → Threshold VAR…** and Estimate → Equation…", tree, TOPS) == []
    assert bad_paths("Estimate → Threshold VAR…", tree, TOPS) == ["Estimate → Threshold VAR…"]
    assert bad_paths("In the app: **Estimate →\n  Nonlinear → Bilinear model…**", tree, TOPS) == ["Estimate → nonlinear → Bilinear model…"]
    assert bad_paths("Help → About PyMetrix; Finance → Anything (no plugin)", tree, TOPS) == []
    assert bad_paths("the figure's Edit… → Style", tree, TOPS) == []
    plugin = "The **Guide** panel (`pymetrix-guide`; `pip install pymetrix[guide]`): View → Guide."
    tree["View"] = {"workspace": {}}
    assert bad_paths(plugin, tree, TOPS) == ["View → Guide"]
    assert bad_paths(plugin + "\n\nAnd View → Nothing.", tree, TOPS, ["guide"]) == ["View → Nothing"]


@pytest.mark.parametrize("doc", DOCS)
def test_menu_paths_in_the_docs_exist(menus, absent, doc):
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not in this checkout")
    assert bad_paths(path.read_text(encoding="utf-8"), menus, TOPS, absent) == []


def test_menu_paths_in_the_help_exist(menus, absent):
    from pymetrix.commands import COMMANDS
    from pymetrix.help import _load_all, topics

    _load_all()
    bad = {}
    for t in topics():
        if found := bad_paths(t.text, menus, TOPS, absent):
            bad[f"topic {t.id}"] = found
    for info in COMMANDS.values():
        if found := bad_paths(info.cls.__doc__ or "", menus, TOPS, absent):
            bad[f"dialog {info.cls.__name__}"] = found
    assert bad == {}
