"""File → Open demo project…: a small dialog listing the demos, each with what it shows and the windows it opens,
in the Start window's list (``pymetrix.ui.fit.EntryList``) rather than a dropdown of cut lines.

Up and Down move between the demos, Enter (or a click) opens one, Open opens the one with the focus, Esc cancels.
``choose_demo(parent)`` asks and returns the demo's name (None when cancelled); the window then builds and opens it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DemoChooser", "choose_demo", "demo_entries"]

TITLE = "Open demo project"
WIDTH = 560  # pixels: each demo's description on two or three lines
INTRO = ("Each demo is a project built step by step from the app's own dialogs: its results open in windows, and "
         "History and Script show every step.")


def opens(name: str) -> str:
    """What opening the demo shows, from ``pymetrix.demos.headline``: "Opens forecast_round (Graph), with var
    (Summary) behind it"."""
    from ..demos import headline

    windows = [f"{obj} ({view})" for obj, view in headline(name)[0]]
    if not windows:
        return ""
    if len(windows) == 1:
        return f"Opens {windows[0]}"
    return f"Opens {windows[-1]}, with {', '.join(windows[:-1])} behind it"


def demo_entries() -> list[tuple[str, str, str]]:
    """(name, the entry's text, its tooltip) for each demo: its name, what it shows, and what it opens."""
    from ..demos import demos

    out = []
    for name, description in demos().items():
        shown = opens(name)
        text = f"{name}\n{description}" + (f"\n{shown}." if shown else "")
        out.append((name, text, f"{name}: {description}"))
    return out


def DemoChooser(parent: Any = None) -> Any:  # noqa: N802 - named as the dialog it makes (Qt imported only then)
    """The dialog: ``entries`` (the list), ``names`` (in order), ``chosen`` (the name once one is opened), ``open_row``
    (open the demo at a row, as a click does), ``open_button``."""
    from PySide6 import QtCore, QtWidgets

    from .fit import EntryList, scroll_area

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(TITLE)
    layout = QtWidgets.QVBoxLayout(dialog)
    from .metrics import set_margins

    set_margins(layout, "dialog_margin")
    intro = QtWidgets.QLabel(INTRO)
    intro.setWordWrap(True)
    layout.addWidget(intro)
    rows = demo_entries()
    dialog.names = [name for name, _, _ in rows]
    dialog.chosen = None
    entries = EntryList([(text, tip) for _, text, tip in rows])
    entries.setAccessibleName("Demo projects")
    for entry in entries.entries:  # the demo's name on the first line, the rest under it
        entry.setAccessibleName(entry.text().split("\n", 1)[0])
    page = QtWidgets.QWidget()
    inner = QtWidgets.QVBoxLayout(page)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.addWidget(entries)
    inner.addStretch()
    layout.addWidget(scroll_area(page, sideways=False), 1)  # a small screen: the list scrolls, never sideways
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Open
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)
    open_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Open)
    open_button.setToolTip("open the demo with the focus (or click a demo, or press Enter on it)")

    def open_row(row: int) -> None:
        if 0 <= row < len(dialog.names):
            dialog.chosen = dialog.names[row]
            dialog.accept()

    def focused_row() -> int:
        focus = QtWidgets.QApplication.focusWidget()
        return entries.entries.index(focus) if focus in entries.entries else 0

    entries.activated.connect(open_row)
    buttons.accepted.connect(lambda: open_row(focused_row()))
    buttons.rejected.connect(dialog.reject)
    dialog.entries, dialog.open_row, dialog.open_button = entries, open_row, open_button
    m = layout.contentsMargins()  # as tall as the list at this width (up to the screen's share): no empty room under it
    inner_width = WIDTH - m.left() - m.right()
    height = (m.top() + m.bottom() + intro.heightForWidth(inner_width) + entries.heightForWidth(inner_width)
              + buttons.sizeHint().height() + 3 * max(layout.spacing(), 6))
    from .fit import fit_to_screen

    fit_to_screen(dialog, WIDTH, max(height, 240))
    if entries.entries:
        entries.entries[0].setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
    return dialog


def choose_demo(parent: Any = None) -> str | None:
    """Ask which demo to open: its name, or None when cancelled."""
    dialog = DemoChooser(parent)
    try:
        dialog.exec()
        return dialog.chosen
    finally:
        dialog.deleteLater()
