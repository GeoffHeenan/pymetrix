"""A command dialog's Advanced settings section: the one collapsed place for settings most steps leave alone.

The fields a command marks ``Widget(..., advanced=True)`` go here, after the form's own fields, with the advanced
settings of the method or test chosen in a sub-form (Method settings, Test settings), each under that sub-form's label.

- **Collapsed by default**, and remembered per command: opening or closing it by hand (a click, its access key) is kept
  in the app's settings (``QSettings``, key ``dialogs/<Command>/advanced``) for the next time that dialog opens.
- **Never hiding what matters**: the section opens by itself when a dialog is reopened from a step whose advanced
  settings differ from their defaults, or when an advanced setting holds a value that is not valid; while closed, its
  header says how many settings in it differ from their defaults ("Advanced settings (2 changed)").
- **Keyboard**: the header has an access key of its own; a hidden field's access key opens the section and goes to
  the field (``pymetrix.ui.form_keys``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["AdvancedSection", "SETTINGS_KEY", "remembered", "remember"]

SETTINGS_KEY = "dialogs/{command}/advanced"  # in QSettings("pymetrix", "desktop"): true while the section was left open


def _settings() -> Any:
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def remembered(command: str) -> bool:
    """Whether the user last left ``command``'s Advanced section open (False when nothing is stored, or the settings
    cannot be read)."""
    if not command:
        return False
    try:
        value = _settings().value(SETTINGS_KEY.format(command=command), False)
    except Exception:  # noqa: BLE001 - unreadable settings: the default, collapsed
        return False
    return value is True or str(value).strip().lower() in ("true", "1")


def remember(command: str, open_: bool) -> None:
    """Keep the section's state for ``command``'s next dialog; a failure to write leaves the default."""
    if not command:
        return
    try:
        settings = _settings()
        settings.setValue(SETTINGS_KEY.format(command=command), bool(open_))
        settings.sync()
    except Exception:  # noqa: BLE001 - settings that cannot be written: nothing is remembered
        pass


def _same(value: Any, default: Any) -> bool:
    """Whether a field shows its default: equal, both empty, or the same text (a box showing ``()`` for a tuple)."""
    from ..forms import is_blank

    if value == default:
        return True
    if is_blank(value) and is_blank(default):
        return True
    return str(value).strip() == str(default).strip()


def invalid(editor: Any) -> bool:
    """Whether a field holds a value that cannot be used as typed: one that cannot be read (a tolerance ``1e-``),
    typed text its validator does not accept, or a field a live check marked (a formula with an error)."""
    widget = getattr(editor, "widget", None)
    try:
        editor.get()
    except Exception:  # noqa: BLE001 - what the field shows is not a value
        return True
    if widget is None:
        return False
    try:
        if widget.property("invalid"):
            return True
        if hasattr(widget, "validator") and widget.validator() is not None and hasattr(widget, "hasAcceptableInput"):
            return bool(widget.text().strip()) and not widget.hasAcceptableInput()
    except RuntimeError:  # the widget is gone
        return False
    return False


class AdvancedSection:
    """A dialog's one place for settings most steps leave alone: a "▸ Advanced settings" toggle at the bottom of the
    form, and under it, collapsed until opened, the command's advanced fields and the advanced settings of the
    extension chosen in a sub-form (Method settings, Test settings...), each under that sub-form's label. The toggle
    shows only while some advanced setting applies; its text counts the settings in it that differ from their
    defaults, and has its own access key (``set_key``)."""

    TEXT = "Advanced settings"

    def __init__(self, Qt: Any, form: Any, dialog: Any, command: str = "") -> None:
        from PySide6.QtCore import Qt as QtNS

        self.form, self.dialog, self.QtNS, self.command = form, dialog, QtNS, command
        self.toggle = Qt.QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(QtNS.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtNS.ArrowType.RightArrow)
        self.toggle.setToolTip("Settings most steps leave at their defaults")
        self.toggle.setAccessibleName(self.TEXT)
        self.keyed = self.TEXT  # the header's words with their access key (&), as form_keys gives it
        self.changed = 0
        self._show_text()
        from .metrics import spaced

        form.addRow(spaced(self.toggle))  # a section's room above it, as above every section's heading (metrics)
        self.toggle_row = form.rowCount() - 1
        self.rows: list[tuple[int, Any]] = []  # (row, applies()): what decides whether the row has anything to show
        self.toggle.toggled.connect(self._flip)
        self.toggle.clicked.connect(lambda on: remember(self.command, on))  # by hand (or its key): kept for next time
        self.on_change: Any = None
        self.count: Any = None  # () -> (changed, invalid): set by the dialog once its editors exist
        self.shortcuts: dict[str, Any] = {}  # access key -> the shortcut that opens the section for a hidden field
        self.bad_labels: list[str] = []  # the settings in it that are not valid, as last counted

    @property
    def expanded(self) -> bool:
        return self.toggle.isChecked()

    def add(self, row: int, applies: Any) -> None:
        self.rows.append((row, applies))

    def applies(self) -> bool:
        """Whether any advanced setting applies now (the toggle shows)."""
        return any(bool(applies()) for _, applies in self.rows)

    def set_key(self, keyed: str) -> None:
        """The header's words with their access key, e.g. "&Advanced settings"."""
        if keyed != self.keyed:
            self.keyed = keyed
            self._show_text()

    def _show_text(self) -> None:
        text = self.keyed + (f" ({self.changed} changed)" if self.changed else "")
        if self.toggle.text() != text:
            self.toggle.setText(text)
        self.toggle.setAccessibleDescription(f"{self.changed} changed from their defaults" if self.changed else "")

    def update_header(self) -> tuple[int, int]:
        """Count the settings in the section that differ from their defaults (and those not valid), and say so in the
        header. Returns (changed, invalid)."""
        changed, bad = (0, 0)
        if self.count is not None:
            try:
                changed, bad = self.count()
            except RuntimeError:  # the dialog is gone
                return 0, 0
        self.changed = changed
        self._show_text()
        return changed, bad

    def problem(self) -> str:
        """What OK waits for while a setting in the section is not valid, e.g. "Fix Tolerance (not valid) to run this
        step." """
        names = self.bad_labels or ["the marked settings"]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return f"Fix {listed} ({'in ' + self.TEXT if not self.expanded else 'not valid'}) to run this step."

    def _flip(self, on: bool) -> None:
        self.toggle.setArrowType(self.QtNS.ArrowType.DownArrow if on else self.QtNS.ArrowType.RightArrow)
        if self.on_change is not None:
            self.on_change()
        self._enable_shortcuts()
        if self.dialog.isVisible():  # opened: room for it, within the screen (a long form scrolls); closed: the room back
            from .timers import single_shot

            def fit() -> None:  # once the form's new size has reached the window: its layout settles over two passes
                single_shot(0, self.dialog, self.dialog.adjustSize)

            single_shot(0, self.dialog, fit)

    def sync(self) -> None:
        """Rows show when the section is open and they apply; the toggle while any of them applies."""
        applying = [(row, bool(applies())) for row, applies in self.rows]
        self.form.setRowVisible(self.toggle_row, any(on for _, on in applying))
        for row, on in applying:
            self.form.setRowVisible(row, on and self.expanded)
        self.update_header()
        self._enable_shortcuts()

    # ------------------------------------------------------------------ a hidden field's access key opens the section

    def reveal_keys(self, keyed: list[tuple[str, Any]]) -> None:
        """``keyed``: (label text with its &, input) of each field in the section. While the section is closed its
        fields' labels are hidden, and a hidden label's key does nothing; these shortcuts take the keys then: open the
        section and put the cursor in the field."""
        from PySide6.QtGui import QKeySequence, QShortcut

        from ..forms import access_key

        wanted: dict[str, Any] = {}
        for text, target in keyed:
            key = access_key(text)
            if key and key not in wanted:
                wanted[key] = target
        for key, shortcut in self.shortcuts.items():
            if key not in wanted:
                shortcut.setEnabled(False)
                shortcut.target = None
        for key, target in wanted.items():
            shortcut = self.shortcuts.get(key)
            if shortcut is None:
                shortcut = QShortcut(QKeySequence.mnemonic(f"&{key}"), self.dialog)
                shortcut.setAutoRepeat(False)
                shortcut.activated.connect(lambda s=shortcut: self._reveal(s.target))
                self.shortcuts[key] = shortcut
            shortcut.target = target
        self._enable_shortcuts()

    def _enable_shortcuts(self) -> None:
        on = not self.expanded and self.applies()
        for shortcut in self.shortcuts.values():
            shortcut.setEnabled(on and getattr(shortcut, "target", None) is not None)

    def _reveal(self, target: Any) -> None:
        """Open the section (not remembered: the key went to a field, not to the section) and go to ``target``."""
        if target is None:
            return
        if not self.expanded:
            self.toggle.setChecked(True)
        from PySide6.QtCore import Qt as QtNS

        from .timers import single_shot

        def focus() -> None:
            try:
                if target.isVisibleTo(self.dialog):
                    target.setFocus(QtNS.FocusReason.ShortcutFocusReason)
                    scroll = getattr(self.dialog, "form_scroll", None)
                    if scroll is not None:
                        scroll.ensureWidgetVisible(target)
            except RuntimeError:  # the dialog is gone
                pass

        focus()
        single_shot(0, self.dialog, focus)  # again once the form has laid the rows out


def present(dialog: Any, section: AdvancedSection | None, specs: list[Any], editors: dict[str, Any], initial: Any,
            refresh: Any, shown: dict[str, bool]) -> None:
    """Wire ``section`` into ``dialog`` once its editors exist: for tests and scripting, ``advanced_toggle``,
    ``advanced_section``, ``show_advanced(on)``, ``in_advanced(name)`` (``on_advanced_tab``, its older name) and
    ``advanced_changed()``. Then the section's state: remembered for this command, and opened whatever was remembered
    when a reopened step's advanced settings differ from their defaults or an advanced setting is not valid."""
    names = {s.name for s in specs if s.advanced}
    dialog.in_advanced = dialog.on_advanced_tab = lambda name: name in names
    dialog.advanced_toggle = section.toggle if section is not None else None
    dialog.advanced_section = section

    def show_advanced(on: bool = True) -> None:
        if section is not None:
            section.toggle.setChecked(on)

    dialog.show_advanced = show_advanced
    dialog.advanced_changed = lambda: section.update_header() if section is not None else (0, 0)
    if section is None:
        return
    section.on_change = refresh
    by_name = {s.name: s for s in specs}
    baselines = _baselines(specs, editors, initial)

    def count() -> tuple[int, int]:
        """(settings that differ from their defaults, settings not valid) among those in the section that apply."""
        from ..forms import parse_options

        changed = bad = 0
        section.bad_labels = []
        for name in names:
            if not shown.get(name, True):
                continue
            editor = editors[name]
            if invalid(editor):
                changed, bad = changed + 1, bad + 1
                section.bad_labels.append(by_name[name].label)
                continue
            if by_name[name].kind == "options":
                changed += len(parse_options(editor.get())[1])
            elif not _same(editor.get(), baselines.get(name, by_name[name].default)):
                changed += 1
        for spec in specs:  # a sub-form's advanced settings, for the extension chosen now
            if spec.kind != "options" or spec.advanced or not shown.get(spec.name, True):
                continue
            part = editors[spec.name]
            try:
                given = parse_options(part.get())[1]
            except Exception:  # noqa: BLE001 - a setting typed so it cannot be read
                given = {}
            for sub_spec in part.sub_specs:
                if not sub_spec.advanced:
                    continue
                sub = part.sub.get(sub_spec.name)
                if sub is not None and invalid(sub):
                    changed, bad = changed + 1, bad + 1
                    section.bad_labels.append(sub_spec.label)
                elif sub_spec.name in given:
                    changed += 1
        return changed, bad

    section.count = count
    timer = getattr(dialog, "formula_timer", None)
    if timer is not None:  # a formula checked as typed may be marked invalid after the form refreshed
        timer.timeout.connect(section.update_header)
    changed, bad = section.update_header()
    reopened = initial is not None and _reopened_with_changes(specs, editors, initial)
    if bad or reopened or (initial is not None and changed) or remembered(section.command):
        show_advanced(True)


def _baselines(specs: list[Any], editors: dict[str, Any], initial: Any) -> dict[str, Any]:
    """What each advanced field shows when left at its default: a fresh editor's value for a fixed field (a spin box
    rounds, a text box shows a tuple as text), the value a fresh dialog picked for one filled from the session (a
    dataset, a series list); the declared default where a reopened step set that field otherwise."""
    from ..forms import is_blank

    out: dict[str, Any] = {}
    for spec in specs:
        if not spec.advanced or spec.kind == "options":
            continue
        editor = editors[spec.name]
        if spec.dynamic:
            given = getattr(initial, spec.name, spec.default) if initial is not None else spec.default
            if _same(given, spec.default) or is_blank(given):
                try:
                    out[spec.name] = editor.get()
                except Exception:  # noqa: BLE001
                    out[spec.name] = spec.default
            else:
                out[spec.name] = spec.default
            continue
        try:
            fresh = type(editor)(editor.Qt, spec)
            out[spec.name] = fresh.get()
            fresh.outer.deleteLater()
        except Exception:  # noqa: BLE001 - an editor that cannot be made on its own: the declared default
            out[spec.name] = spec.default
    return out


def _reopened_with_changes(specs: list[Any], editors: dict[str, Any], initial: Any) -> bool:
    """Whether the step being reopened set an advanced field, or an advanced setting of a sub-form, otherwise than its
    default (judged on the step's own values, so a field the dialog fills from the session does not count)."""
    from ..forms import parse_options

    if any(not _same(getattr(initial, s.name, s.default), s.default) for s in specs if s.advanced):
        return True
    for s in specs:  # a sub-form's advanced settings, as the reopened step set them
        if s.kind == "options":
            part = editors[s.name]
            try:
                given = parse_options(part.get())[1]  # the settings that differ from their defaults
            except Exception:  # noqa: BLE001
                continue
            if any(sub.advanced and sub.name in given for sub in part.sub_specs):
                return True
    return False
