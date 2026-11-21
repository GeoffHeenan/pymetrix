"""The Report panel: the report being put together, from every workspace.

Figures, tables and results go in with **Add to report** (under a figure, under a table, or a result window's corner);
here they are put in order, given headings and text, and exported (Export report: the outline's items, written out as
the code that makes the report). Putting the outline together is not recorded; exporting it is.
"""

from __future__ import annotations

from typing import Any, Callable

from . import theme
from .metrics import set_margins


def _describe(item: dict[str, Any]) -> str:
    if item["kind"] == "heading":
        return f"{'#' * item.get('level', 2)} {item['text']}"
    if item["kind"] == "text":
        text = " ".join(str(item["text"]).split())
        return f"¶ {text[:80]}{'…' if len(text) > 80 else ''}"
    what = f"{item['source']} — {item['tab']}" if item.get("tab") else f"{item['source']} (summary and graph)"
    return what + (f"  “{item['caption']}”" if item.get("caption") else "")


def report_panel(session: Callable[[], Any], export: Callable[[], Any], changed: Callable[[], Any] = lambda: None,
                 show_now: bool = True, open_result: Callable[[str], Any] | None = None) -> Any:
    """The panel. ``session()`` gives the current session (whose ``report_outline`` it shows and changes),
    ``export()`` opens Export report on the outline, ``changed()`` is told when the outline changes. With
    ``show_now=False`` it waits for its first ``refresh()`` (the main window's, once its session is there).
    A right-click on an entry (``panel.item_menu(row)``): edit its caption or words, move it (up, down, to the top
    or the end), open its result (``open_result(name)``), add a heading or a paragraph after it, remove it."""
    from PySide6 import QtWidgets

    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    set_margins(layout, "tight")
    from .messages import EMPTY_STATES

    hint = QtWidgets.QLabel(EMPTY_STATES["report"])
    hint.setWordWrap(True)
    hint.setStyleSheet(theme.muted_css())  # follows the light or dark theme
    listing = QtWidgets.QListWidget()
    layout.addWidget(hint)
    layout.addWidget(listing, 1)
    row = QtWidgets.QHBoxLayout()
    buttons: dict[str, Any] = {}

    def outline() -> list[dict[str, Any]]:
        return session().report_outline

    def refresh(select: int | None = None) -> None:
        keep = listing.currentRow() if select is None else select
        listing.clear()
        for item in outline():
            listing.addItem(_describe(item))
        empty = not outline()
        hint.setVisible(empty)
        listing.setVisible(not empty)
        if not empty:
            listing.setCurrentRow(max(0, min(keep, listing.count() - 1)))
        update()

    def update() -> None:
        row_ = listing.currentRow()
        for key in ("up", "down", "remove"):
            buttons[key].setEnabled(row_ >= 0)
        buttons["up"].setEnabled(row_ > 0)
        buttons["down"].setEnabled(0 <= row_ < listing.count() - 1)
        buttons["export"].setEnabled(any(i["kind"] == "item" for i in outline()))

    def move(by: int) -> None:
        i = listing.currentRow()
        if 0 <= i + by < len(outline()):
            session().move_report_item(i, i + by)
            refresh(i + by)
            changed()

    def remove() -> None:
        i = listing.currentRow()
        if i >= 0:
            session().remove_report_item(i)
            refresh(i)
            changed()

    def add_heading() -> None:
        text, ok = QtWidgets.QInputDialog.getText(panel, "Heading", "A heading in the report:")
        if ok and text.strip():
            session().add_report_heading(text.strip())
            refresh(len(outline()) - 1)
            changed()

    def add_text() -> None:
        text, ok = QtWidgets.QInputDialog.getMultiLineText(panel, "Text", "A paragraph in the report:")
        if ok and text.strip():
            session().add_report_text(text.strip())
            refresh(len(outline()) - 1)
            changed()

    def edit(index: int) -> None:
        """Double-click: an item's caption, a heading's or a paragraph's text."""
        item = outline()[index]
        if item["kind"] == "text":
            text, ok = QtWidgets.QInputDialog.getMultiLineText(panel, "Text", "The paragraph:", item["text"])
        else:
            label = "The heading:" if item["kind"] == "heading" else "A caption (empty: the result's own title):"
            text, ok = QtWidgets.QInputDialog.getText(panel, "Report", label, text=item.get("text" if item["kind"] == "heading" else "caption", ""))
        if ok:
            item["text" if item["kind"] in ("heading", "text") else "caption"] = text.strip()
            session().dirty = True
            refresh(index)
            changed()

    def move_to(target: int) -> None:
        i = listing.currentRow()
        if i >= 0 and i != target:
            session().move_report_item(i, target)
            refresh(target)
            changed()

    def insert_after(row_: int, kind: str) -> None:
        """A heading or a paragraph, added at the end and moved to just after the entry."""
        before = len(outline())
        (add_heading if kind == "heading" else add_text)()
        if len(outline()) > before and row_ + 1 < len(outline()) - 1:
            session().move_report_item(len(outline()) - 1, row_ + 1)
            refresh(row_ + 1)
            changed()

    def item_menu(row_: int) -> Any:
        menu = QtWidgets.QMenu(panel)
        if not 0 <= row_ < len(outline()):
            menu.addAction("Add a heading…", add_heading)
            menu.addAction("Add a paragraph…", add_text)
            return menu
        listing.setCurrentRow(row_)
        item = outline()[row_]
        words = {"heading": "Edit the heading…", "text": "Edit the paragraph…"}.get(item["kind"], "Edit the caption…")
        menu.addAction(words, lambda: edit(row_))
        if item["kind"] == "item" and open_result is not None:
            menu.addAction(f"Open “{item['source']}”", lambda: open_result(item["source"]))
        menu.addSeparator()
        last = len(outline()) - 1
        menu.addAction("Move up", lambda: move(-1)).setEnabled(row_ > 0)
        menu.addAction("Move down", lambda: move(1)).setEnabled(row_ < last)
        menu.addAction("Move to the top", lambda: move_to(0)).setEnabled(row_ > 0)
        menu.addAction("Move to the end", lambda: move_to(last)).setEnabled(row_ < last)
        menu.addSeparator()
        menu.addAction("Add a heading after it…", lambda: insert_after(row_, "heading"))
        menu.addAction("Add a paragraph after it…", lambda: insert_after(row_, "text"))
        menu.addSeparator()
        menu.addAction("Remove from the report", remove)
        return menu

    from PySide6.QtCore import Qt as QtNS

    listing.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
    listing.customContextMenuRequested.connect(
        lambda pos: item_menu(listing.row(listing.itemAt(pos)) if listing.itemAt(pos) else -1).popup(listing.viewport().mapToGlobal(pos)))

    for key, label, tip, action in (("up", "Up", "earlier in the report", lambda: move(-1)),
                                    ("down", "Down", "later in the report", lambda: move(1)),
                                    ("remove", "Remove", "take it out of the report", remove),
                                    ("heading", "Heading…", "add a heading at the end", add_heading),
                                    ("text", "Text…", "add a paragraph at the end", add_text),
                                    ("export", "Export…", "write the report: Word, HTML, Markdown or Excel (a recorded step)", lambda: export())):
        button = QtWidgets.QPushButton(label)
        button.setToolTip(tip)
        button.clicked.connect(action)
        row.addWidget(button)
        buttons[key] = button
    layout.addLayout(row)
    listing.currentRowChanged.connect(lambda _row: update())
    listing.itemDoubleClicked.connect(lambda it: edit(listing.row(it)))
    panel.listing, panel.buttons, panel.refresh, panel.edit, panel.hint = listing, buttons, refresh, edit, hint
    panel.item_menu = item_menu
    if show_now:
        refresh()
    return panel
