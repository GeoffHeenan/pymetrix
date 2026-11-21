"""The command palette (Ctrl+K): one search box over everything the app can do.

Every menu item (commands and the app's own actions, under their full menu path), the workspace's objects and the
help pages, found by typing a few letters of any words ("seas adj", "var lag", "gdp"). Enter runs the first; the
arrows choose another. Nothing is recorded by the palette itself: it opens what the menu would open.

EViews', IRIS's and Dynare's names find their pymetrix equivalents too ("genr" offers "Generate series… (EViews:
genr)"), from the translation tables of ``pymetrix.coming_from``; they come after the items pymetrix's own names
match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

__all__ = ["PaletteEntry", "palette_dialog", "rank", "synonym_entries"]

KIND_ORDER = {"Menu": 0, "Object": 1, "Help": 2}  # commands first, then the workspace, then help


@dataclass
class PaletteEntry:
    title: str  # "Data → Transform → Seasonal adjustment…", an object's name, a help page's title
    kind: str  # "Menu", "Object" or "Help"
    run: Callable[[], Any]
    detail: str = ""  # searched too: an object's type, a help page's section
    synonym: str = ""  # another package's name for it ("genr"): the entry is found by that name alone
    icon: Any = None  # a QIcon shown beside it (an object's kind: ui.icons); decoration only


def synonym_entries(entries: list[PaletteEntry], synonyms: Iterable[tuple[str, str, str, str]],
                    open_topic: Callable[[str], Any]) -> list[PaletteEntry]:
    """Entries for other packages' names: each ``(word, package, menu path, topic id)`` (``pymetrix.coming_from.
    palette_synonyms``) as "Generate series… (EViews: genr)", running the menu item of ``entries`` at that path, or
    opening the help topic when it has no menu path. A word the item's own name already holds adds nothing; a menu
    item not in ``entries`` (another workspace's, or not available now) is left out. Packages sharing a word for
    the same item share an entry: "(IRIS, Dynare: forecast)"."""
    from ..help import topic as help_topic

    menus = {e.title: e for e in entries if e.kind == "Menu" and not e.synonym}
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for word, package, menu, topic in synonyms:
        if menu and menu not in menus:
            continue
        if menu and word.lower() in _words(menu.rsplit("→", 1)[-1]):
            continue
        grouped.setdefault((word.lower(), menu, "" if menu else topic), []).append(package)
    out = []
    for (word, menu, topic), packages in grouped.items():
        label = f"({', '.join(dict.fromkeys(packages))}: {word})"
        if menu:
            target = menus[menu]
            out.append(PaletteEntry(f"{menu.rsplit('→', 1)[-1].strip()} {label}", "Menu", target.run, menu, synonym=word))
        else:  # code only: the page that says how
            out.append(PaletteEntry(f"{help_topic(topic).title} {label}", "Help", lambda t=topic: open_topic(t),
                                    "Background", synonym=word))
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s→/()_.,:-]+", text.lower()) if w]


def rank(entries: list[PaletteEntry], text: str, limit: int = 60) -> list[PaletteEntry]:
    """The entries every typed word is part of, best first: a word that starts the item's own name, then words that
    start words of the title, then shorter titles. With nothing typed, the entries as given.

    Another package's name (an entry's ``synonym``: "genr") is found only by that name, typed in full or from its
    first two letters, and comes after every item whose own name has a word starting with each typed word (an item
    named in pymetrix's words), ahead of items that only hold the letters in their menu path or inside a word."""
    typed = text.lower().split()
    if not typed:
        return [e for e in entries if not e.synonym][:limit]
    # a dialog's help page has the dialog's own name: its menu item stands for both (F1 in the dialog opens the page),
    # so "VAR · Help (Commands)" does not crowd the list beside "Estimate → ... → VAR…" and "VAR · Help (Models)"
    commands = {e.title.rsplit("→", 1)[-1].strip().rstrip(".…").lower() for e in entries if e.kind == "Menu" and not e.synonym}
    entries = [e for e in entries if not (e.kind == "Help" and e.detail == "Commands" and e.title.lower() in commands)]
    scored, synonyms, named = [], [], set()
    typed_text = " ".join(typed)
    for order, entry in enumerate(entries):
        if entry.synonym:
            if len(typed_text) >= 2 and entry.synonym.startswith(typed_text):
                exact = 0 if entry.synonym == typed_text else 1
                synonyms.append((exact, len(entry.title), order, entry))
            continue
        title = entry.title.lower()
        hay = f"{title} {entry.detail.lower()}"
        if not all(word in hay for word in typed):
            continue
        own = title.rsplit("→", 1)[-1].strip()  # the item itself, not its menu
        starts = _words(title)
        score = 0.0
        if own.rstrip(".…") == " ".join(typed):  # exactly its name: an object called data, for "data"
            score -= 6
        elif own.startswith(typed[0]):
            score -= 4
        score -= sum(1.5 for word in typed if any(w.startswith(word) for w in starts))
        score -= sum(0.5 for word in typed if word in own)
        score += len(title) / 200 + 1.0 * KIND_ORDER.get(entry.kind, 3)  # on a like match, commands before help pages
        scored.append((score, KIND_ORDER.get(entry.kind, 3), order, entry))
        own_words = _words(own)
        if all(any(w.startswith(word) for w in own_words) for word in typed):
            named.add(id(entry))
    scored.sort(key=lambda s: s[:3])
    ranked = [entry for *_, entry in scored]
    last_named = max((k for k, entry in enumerate(ranked) if id(entry) in named), default=-1)
    found = [entry for *_, entry in sorted(synonyms, key=lambda s: s[:3])]
    return (ranked[:last_named + 1] + found + ranked[last_named + 1:])[:limit]


def palette_dialog(entries: Callable[[], list[PaletteEntry]], parent: Any = None) -> Any:
    """The palette: a search line and the matches under it, each with its kind. ``entries`` is called when it opens,
    so it lists the menus and objects as they are now."""
    from PySide6 import QtWidgets
    from PySide6.QtCore import QEvent, Qt

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Find anything")
    from .fit import fit_to_screen

    fit_to_screen(dialog, 620, 420)
    layout = QtWidgets.QVBoxLayout(dialog)
    from .metrics import set_margins

    set_margins(layout, "dialog_margin")
    search = QtWidgets.QLineEdit()
    search.setPlaceholderText("Type to find a command, an object or a help page…  (Enter opens, Esc closes)")
    search.setClearButtonEnabled(True)
    found = QtWidgets.QListWidget()
    from .icons import size_icons

    size_icons(found)  # the objects' icons at the text size
    layout.addWidget(search)
    layout.addWidget(found, 1)
    everything = entries()
    shown: list[PaletteEntry] = []

    def refresh() -> None:
        shown[:] = rank(everything, search.text())
        found.clear()
        for entry in shown:
            item = QtWidgets.QListWidgetItem(f"{entry.title}    ·  {entry.kind}{f' ({entry.detail})' if entry.detail else ''}")
            item.setToolTip(entry.detail or entry.title)
            if entry.icon is not None:
                item.setIcon(entry.icon)
            found.addItem(item)
        if shown:
            found.setCurrentRow(0)

    def run(row: int | None = None) -> None:
        row = found.currentRow() if row is None else row
        if 0 <= row < len(shown):
            dialog.accept()
            dialog.chosen = shown[row]
            shown[row].run()

    class Keys(QtWidgets.QWidget):  # Up and Down in the search line move through the matches
        def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                step = 1 if event.key() == Qt.Key.Key_Down else -1
                found.setCurrentRow(max(0, min(found.count() - 1, found.currentRow() + step)))
                return True
            return False

    keys = Keys(dialog)
    search.installEventFilter(keys)
    search.textChanged.connect(lambda _text: refresh())
    search.returnPressed.connect(lambda: run())
    found.itemActivated.connect(lambda item: run(found.row(item)))
    dialog.search, dialog.found, dialog.shown, dialog.run, dialog.chosen = search, found, shown, run, None
    refresh()
    search.setFocus()
    return dialog
