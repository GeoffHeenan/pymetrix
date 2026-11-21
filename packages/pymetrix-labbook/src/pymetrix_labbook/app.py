"""The Lab book panel in the desktop app: the project brief, notes, the objectives steps are linked to, and what
``verify`` would say about them, as the work goes on.

The panel docks on the right (View → Lab book). It shows:

- the brief, or a prompt to write one before the work (or to say why there is none);
- **Brief…**, **Skip…** and **Note…**, which write ``brief``, ``brief_skipped`` and ``note`` entries;
- the brief's objectives and hypotheses, with ticks: steps run while one is ticked are linked to it;
- the brief's checks (``verify.brief_report``) and the latest entries.

Qt is imported only when the panel is built, so the lab book works without the app. The forms' text is parsed by
plain functions (``parse_pairs``, ``parse_lines``), tested without a window.
"""

from __future__ import annotations

from typing import Any

from .brief import (
    _has_brief,
    active_objectives,
    brief,
    commit_brief,
    current_brief,
    note,
    reveal_brief,
    set_active_objectives,
    skip_brief,
)
from .plan import current_plan, lock_plan, plan_report
from .recorder import lab_book
from .tried import fits, outcome_counts, specification_curve, tried
from .verify import brief_report, describe_actor

__all__ = ["BRIEF_FIELDS", "create_panel", "curve_dialog", "format_pairs", "parse_lines", "parse_pairs", "plan_dialog"]

# the brief dialog's fields: (keyword, label, kind) with kind "line", "text", "pairs" (id: text a line) or "lines"
BRIEF_FIELDS = (
    ("title", "Title", "line"),
    ("question", "Research question", "text"),
    ("why", "Why it matters", "text"),
    ("objectives", "Objectives (one a line, id: text)", "pairs"),
    ("hypotheses", "Hypotheses (one a line, id: statement)", "pairs"),
    ("primary_outcomes", "Primary outcomes (one a line)", "lines"),
    ("secondary_outcomes", "Secondary outcomes (one a line)", "lines"),
    ("data", "Intended data (one a line, e.g. sources: ..., sample: ...)", "pairs"),
    ("methods", "Intended methods, broadly", "text"),
    ("limits", "What the project leaves out", "text"),
    ("team", "Team (one a line, name: role)", "pairs"),
    ("funder", "Funder", "line"),
    ("conditions", "Ethics or data-access conditions", "text"),
)


def parse_pairs(text: str) -> dict[str, str]:
    """``"O1: estimate it\\nO2: test it"`` -> {"O1": "estimate it", "O2": "test it"}; a line without a colon is an
    error, so a typo is not silently dropped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"write this line as 'id: text': {line.strip()!r}")
        out[key.strip()] = value.strip()
    return out


def parse_lines(text: str) -> list[str]:
    """One item a line, blank lines left out."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def format_pairs(pairs: dict[str, Any]) -> str:
    """The inverse of ``parse_pairs``, for filling in the form from the brief as it stands (a hypothesis's
    statement)."""
    return "\n".join(f"{k}: {v['statement'] if isinstance(v, dict) else v}" for k, v in pairs.items())


def _summary(entry: dict[str, Any]) -> str:
    data = entry.get("data", {})
    what = (data.get("code") or ", ".join(data.get("codes", [])) or data.get("title") or data.get("text")
            or data.get("reason") or data.get("path") or "")
    first = str(what).strip().splitlines()[0] if str(what).strip() else ""
    linked = data.get("objectives") if entry.get("event") in ("step", "failed") else None  # a brief's are its own
    tags = f" [{', '.join(linked)}]" if linked else ""
    if entry.get("event") in ("step", "failed") and data.get("plan"):
        tags += f" ({data['plan']})"
    if entry.get("event") == "brief_commitment":
        first = f"private brief, hash {data.get('hash', '')[:16]}…"
    if entry.get("event") == "brief_revealed":
        first = f"revealed: {data.get('brief', {}).get('title', '')}"
    if entry.get("event") == "plan":
        first = f"{len(data.get('specifications', []))} specification(s), locked"
    event = {"start": "lab book started", "brief_skipped": "no brief", "brief_commitment": "brief committed",
             "brief_revealed": "brief revealed"}.get(entry.get("event", "?"), entry.get("event", "?"))
    by = f"  · {describe_actor(entry['by'])}" if entry.get("by") else ""  # not "· not recorded" on the first entry
    return f"{entry.get('seq', '?')}  {event}  {first}{tags}{by}".rstrip()


def create_panel(app: Any) -> Any:
    """The panel's widget (``AppPanel.create``): ``app`` is pymetrix's ``AppInterface``."""
    from PySide6 import QtCore, QtWidgets

    panel = QtWidgets.QWidget()
    panel.setObjectName("labbook-panel")
    layout = QtWidgets.QVBoxLayout(panel)
    status = QtWidgets.QLabel()
    status.setWordWrap(True)
    layout.addWidget(status)
    buttons = QtWidgets.QHBoxLayout()
    brief_button = QtWidgets.QPushButton("Brief…")
    skip_button = QtWidgets.QPushButton("Skip…")
    note_button = QtWidgets.QPushButton("Note…")
    plan_button = QtWidgets.QPushButton("Plan…")
    reveal_button = QtWidgets.QPushButton("Reveal…")
    for b, tip in ((brief_button, "Write the project brief before the work, or amend it (with a reason)"),
                   (skip_button, "Record that this project has no brief, and why"),
                   (note_button, "Record the reasoning behind a choice"),
                   (plan_button, "Lock the analysis plan (exact specifications) before reading the data"),
                   (reveal_button, "Disclose a private brief from its file, checked against its commitment")):
        b.setToolTip(tip)
        buttons.addWidget(b)
    layout.addLayout(buttons)
    more = QtWidgets.QHBoxLayout()
    curve_button = QtWidgets.QPushButton("Specification curve…")
    curve_button.setToolTip("Every specification fitted for an outcome, with the selected result highlighted")
    more.addWidget(curve_button)
    more.addStretch(1)
    layout.addLayout(more)
    ticks_label = QtWidgets.QLabel("Objectives the next steps address (tick them):")
    ticks_label.setWordWrap(True)
    layout.addWidget(ticks_label)
    ticks = QtWidgets.QListWidget()
    ticks.setObjectName("labbook-objectives")
    layout.addWidget(ticks)
    no_ticks = QtWidgets.QLabel("No objectives yet: a brief's objectives and hypotheses are listed here, to tick the ones "
                                "the next steps address.")
    no_ticks.setWordWrap(True)
    no_ticks.setObjectName("labbook-objectives-empty")
    no_ticks.setStyleSheet("color: palette(placeholder-text);")
    layout.addWidget(no_ticks)
    layout.addWidget(QtWidgets.QLabel("Checks:"))
    checks = QtWidgets.QListWidget()
    checks.setObjectName("labbook-checks")
    layout.addWidget(checks)
    layout.addWidget(QtWidgets.QLabel("Latest lab-book entries:"))
    entries = QtWidgets.QListWidget()
    entries.setObjectName("labbook-entries")
    layout.addWidget(entries)
    for listing in (checks, entries):  # long lines wrap, in full in their tooltips, with no sideways scroll bar
        listing.setWordWrap(True)
        listing.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        listing.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)

    state = {"filling": False}

    def refresh() -> None:
        import shiboken6

        if not (shiboken6.isValid(panel) and shiboken6.isValid(status)):
            return  # the panel, or its window, is gone (an app may still call it): nothing to show
        session = app.session
        book = lab_book(session)
        current = current_brief(session)
        skipped = any(e.get("event") == "brief_skipped" for e in book.entries)
        amendments = sum(1 for e in book.entries if e.get("event") == "brief") - 1
        plan = current_plan(book)
        locked = f"<br>Plan locked: {len(plan['specifications'])} specification(s)" if plan else ""
        if current:
            status.setText(f"<b>{current['title']}</b>" + (f" (amended {amendments}×)" if amendments > 0 else "") + locked)
        elif any(e.get("event") == "brief_commitment" for e in book.entries):
            status.setText("<b>Private brief committed</b>: only its hash is in the lab book (Reveal… at publication)."
                           + locked)
        elif skipped:
            status.setText("No brief: skipped (see Checks)." + locked)
        else:
            status.setText("<b>No brief yet.</b> Write one before the work (Brief…), or say why there is none (Skip…)."
                           + locked)
        state["filling"] = True
        ticks.clear()
        ticked = set(active_objectives(session))
        for key, value in {**(current or {}).get("objectives", {}), **(current or {}).get("hypotheses", {})}.items():
            text = value["statement"] if isinstance(value, dict) else value
            item = QtWidgets.QListWidgetItem(f"{key}: {text}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if key in ticked else QtCore.Qt.CheckState.Unchecked)
            ticks.addItem(item)
        state["filling"] = False
        ticks.setVisible(bool(ticks.count()))
        no_ticks.setVisible(not ticks.count())
        warnings, notes, checked = brief_report(book)
        plan_warnings, plan_notes, plan_checked = plan_report(book)
        warnings, notes, checked = warnings + plan_warnings, notes + plan_notes, checked + plan_checked
        notes = notes + [f"{o}: {k} specification(s) fitted ({n} fit(s))" for o, (k, n) in sorted(outcome_counts(book).items())]
        selected = app.selected()
        if selected:
            try:
                notes.append(tried(app.session, selected))
            except ValueError:
                pass
        checks.clear()
        for prefix, items in (("⚠ ", warnings), ("· ", notes), ("✓ ", checked)):
            for text in items:
                checks.addItem(prefix + text)
                checks.item(checks.count() - 1).setToolTip(text)
        if not checks.count():
            checks.addItem("No checks yet: they appear once there is a brief, a plan or a fitted step.")
        entries.clear()
        for entry in book.entries[-12:]:
            entries.addItem(_summary(entry))
            entries.item(entries.count() - 1).setToolTip(_summary(entry))
            entries.item(entries.count() - 1).setData(QtCore.Qt.ItemDataRole.UserRole, entry.get("seq"))
        entries.scrollToBottom()

    def ticked_changed(_item: Any) -> None:
        if state["filling"]:
            return
        ids = [ticks.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(ticks.count())
               if ticks.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        set_active_objectives(app.session, ids)
        app.status(f"Steps now address {', '.join(ids)}" if ids else "Steps now address no objective")

    def write_brief() -> None:
        dialog = brief_dialog(panel, current_brief(app.session), _has_brief(app.session))
        while dialog.exec():
            try:
                if dialog.private.isChecked():
                    path, _ = QtWidgets.QFileDialog.getSaveFileName(panel, "Keep the private brief in", "brief.json",
                                                                    "JSON (*.json)")
                    if not path:
                        continue
                    digest = commit_brief(app.session, path, **dialog.values())
                    QtWidgets.QApplication.clipboard().setText(digest)
                    QtWidgets.QMessageBox.information(
                        panel, "Private brief", f"The brief is in {path}: keep it private and unchanged.\n\nIts hash, "
                        f"now on the clipboard, is in the lab book:\n{digest}\n\nRegister the hash with a time-stamping "
                        "service (OSF, an RFC 3161 authority) to prove when it was written. Reveal… discloses it later.")
                    break
                brief(app.session, **dialog.values())
            except (ValueError, OSError) as exc:
                QtWidgets.QMessageBox.warning(panel, "Project brief", str(exc))
                continue
            app.status("Brief recorded in the lab book")
            break
        refresh()

    def skip() -> None:
        reason, ok = QtWidgets.QInputDialog.getText(panel, "No project brief", "Why is there no brief?")
        if ok and reason.strip():
            skip_brief(app.session, reason.strip())
            refresh()

    def entry_menu(item: Any) -> Any:
        """The right-click menu on a lab-book entry: a note about it (linked to its number), or the entry copied, as
        shown or as its full record (JSON)."""
        import json

        menu = QtWidgets.QMenu(panel)
        seq = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if seq is None:
            return menu
        record = next((e for e in lab_book(app.session).entries if e.get("seq") == seq), None)

        def note_on() -> None:
            current = current_brief(app.session) or {}
            dialog = note_dialog(panel, [*current.get("objectives", {}), *current.get("hypotheses", {})])
            dialog.setWindowTitle(f"Note on lab-book entry {seq}")
            if dialog.exec():
                text, about = dialog.values()
                if text:
                    note(app.session, text, about=about, step=seq)
                    refresh()

        clip = QtWidgets.QApplication.clipboard()
        menu.addAction(f"Add a note on entry {seq}…", note_on)
        menu.addAction("Copy the entry", lambda: clip.setText(item.text()))
        menu.addAction("Copy its full record (JSON)",
                       lambda: clip.setText(json.dumps(record, ensure_ascii=False, indent=1, default=str))).setEnabled(record is not None)
        return menu

    entries.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    entries.customContextMenuRequested.connect(
        lambda pos: entry_menu(entries.itemAt(pos)).popup(entries.viewport().mapToGlobal(pos)) if entries.itemAt(pos) else None)
    checks.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)

    def check_menu(pos: Any) -> None:
        item = checks.itemAt(pos)
        if item is not None:
            menu = QtWidgets.QMenu(panel)
            menu.addAction("Copy the check", lambda: QtWidgets.QApplication.clipboard().setText(item.toolTip() or item.text()))
            menu.popup(checks.viewport().mapToGlobal(pos))

    checks.customContextMenuRequested.connect(check_menu)
    panel.entry_menu = entry_menu

    def add_note() -> None:
        current = current_brief(app.session) or {}
        dialog = note_dialog(panel, [*current.get("objectives", {}), *current.get("hypotheses", {})])
        if dialog.exec():
            text, about = dialog.values()
            if text:
                note(app.session, text, about=about)
                refresh()

    def write_plan() -> None:
        session = app.session
        dialog = plan_dialog(panel, current_plan(session), [r.code for r in session.history])
        while dialog.exec():
            try:
                lock_plan(session, **dialog.values())
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(panel, "Analysis plan", str(exc))
                continue
            app.status("Plan locked in the lab book: later steps are labelled planned or exploratory")
            break
        refresh()

    def reveal() -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(panel, "The private brief's file", "", "JSON (*.json)")
        if not path:
            return
        try:
            reveal_brief(app.session, path)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(panel, "Reveal the brief", str(exc))
            return
        app.status("Brief revealed: it matches its commitment")
        refresh()

    brief_button.clicked.connect(write_brief)
    reveal_button.clicked.connect(reveal)

    def curve() -> None:
        session = app.session
        counts = outcome_counts(session)
        if not counts:
            QtWidgets.QMessageBox.information(panel, "Specification curve", "No fitted model with an outcome yet.")
            return
        dialog = curve_dialog(panel, session, sorted(counts), app.selected())
        dialog.exec()

    curve_button.clicked.connect(curve)
    plan_button.clicked.connect(write_plan)
    skip_button.clicked.connect(skip)
    note_button.clicked.connect(add_note)
    ticks.itemChanged.connect(ticked_changed)
    app.on_history_changed(refresh)
    panel.refresh = refresh  # for tests and the app
    refresh()
    return panel


def brief_dialog(parent: Any, current: dict[str, Any] | None, amending: bool | None = None) -> Any:
    """The project brief form, filled in from the brief as it stands; an amendment also asks for its reason.
    ``dialog.values()`` gives ``brief``'s keywords."""
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    amending = bool(current) if amending is None else amending  # a private brief not yet revealed is amended too
    dialog.setWindowTitle("Amend the project brief" if amending else "Project brief")
    form = QtWidgets.QFormLayout(dialog)
    widgets: dict[str, Any] = {}
    for key, label, kind in BRIEF_FIELDS:
        value = (current or {}).get(key)
        if kind == "line":
            widget = QtWidgets.QLineEdit(value or "")
        else:
            widget = QtWidgets.QPlainTextEdit()
            widget.setFixedHeight(56)
            if kind == "pairs":
                widget.setPlainText(format_pairs(value or {}))
            elif kind == "lines":
                widget.setPlainText("\n".join(value or []))
            else:
                widget.setPlainText(value or "")
        widget.setObjectName(f"brief-{key}")
        widgets[key] = (widget, kind)
        form.addRow(label, widget)
    private = QtWidgets.QCheckBox("Keep it private until publication: the lab book records only its hash")
    private.setObjectName("brief-private")
    form.addRow("", private)
    if amending:
        reason = QtWidgets.QLineEdit()
        reason.setObjectName("brief-reason")
        reason.setPlaceholderText("why the brief changes (kept in the lab book)")
        widgets["reason"] = (reason, "line")
        form.addRow("Reason for the amendment", reason)
    box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    form.addRow(box)
    hypotheses = (current or {}).get("hypotheses", {})

    def values() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, (widget, kind) in widgets.items():
            text = widget.text() if kind == "line" else widget.toPlainText()
            if kind == "pairs":
                out[key] = parse_pairs(text)
            elif kind == "lines":
                out[key] = parse_lines(text)
            else:
                out[key] = text.strip()
        # keep what the form does not show (a hypothesis's supports and contradicts) for unchanged statements
        out["hypotheses"] = {k: ({**hypotheses[k], "statement": v} if k in hypotheses else v)
                             for k, v in out["hypotheses"].items()}
        return out

    dialog.widgets = widgets
    dialog.values = values
    dialog.private = private
    return dialog


def note_dialog(parent: Any, ids: list[str]) -> Any:
    """A note's text and what it concerns. ``dialog.values()`` gives (text, about)."""
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Note in the lab book")
    form = QtWidgets.QFormLayout(dialog)
    text = QtWidgets.QPlainTextEdit()
    text.setObjectName("note-text")
    about = QtWidgets.QComboBox()
    about.setObjectName("note-about")
    about.addItems(["(none)", *ids])
    form.addRow("Note", text)
    form.addRow("About", about)
    box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    form.addRow(box)
    dialog.values = lambda: (text.toPlainText().strip(), "" if about.currentIndex() == 0 else about.currentText())
    return dialog


def plan_dialog(parent: Any, current: dict[str, Any] | None, history: list[str]) -> Any:
    """The analysis-plan form: the specifications (a step's code a line; the session's steps are offered to copy
    from), the datasets, the sample and exclusion rules, the hypotheses and the primary outcomes. Changing a locked
    plan also asks for the reason. ``dialog.values()`` gives ``lock_plan``'s keywords."""
    from PySide6 import QtWidgets

    current = current or {}
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Change the analysis plan" if current else "Lock the analysis plan")
    form = QtWidgets.QFormLayout(dialog)
    specs = QtWidgets.QPlainTextEdit("\n".join(current.get("specifications", [])))
    specs.setObjectName("plan-specifications")
    specs.setPlaceholderText("fit = Equation('c ~ y').fit(data, method=OLS())")
    form.addRow("Specifications (a step's code a line)", specs)
    if history:
        steps = QtWidgets.QComboBox()
        steps.setObjectName("plan-history")
        steps.addItems(["Add a step from the history…", *history])

        def add(index: int) -> None:
            if index > 0:
                specs.appendPlainText(history[index - 1])
                steps.setCurrentIndex(0)

        steps.currentIndexChanged.connect(add)
        form.addRow("", steps)
    fields: dict[str, Any] = {}
    for key, label, value in (("data", "Datasets it uses (by name, comma-separated)", ", ".join(current.get("data", []))),
                              ("sample", "Sample", current.get("sample", "")),
                              ("exclusions", "Exclusion rules", current.get("exclusions", "")),
                              ("hypotheses", "Hypotheses tested (ids, comma-separated)", ", ".join(current.get("hypotheses", []))),
                              ("primary_outcomes", "Primary outcomes (comma-separated)",
                               ", ".join(current.get("primary_outcomes", []))),
                              ("data_request", "Data request", current.get("data_request", ""))):
        widget = QtWidgets.QLineEdit(value)
        widget.setObjectName(f"plan-{key}")
        fields[key] = widget
        form.addRow(label, widget)
    if current:
        reason = QtWidgets.QLineEdit()
        reason.setObjectName("plan-reason")
        reason.setPlaceholderText("why the plan changes (kept in the lab book, with the work done before it)")
        fields["reason"] = reason
        form.addRow("Reason for the change", reason)
    box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    form.addRow(box)

    def values() -> dict[str, Any]:
        def items(text: str) -> list[str]:
            return [t.strip() for t in text.split(",") if t.strip()]

        out: dict[str, Any] = {"specifications": parse_lines(specs.toPlainText())}
        for key, widget in fields.items():
            out[key] = items(widget.text()) if key in ("data", "hypotheses", "primary_outcomes") else widget.text().strip()
        return out

    dialog.specs, dialog.fields, dialog.values = specs, fields, values
    return dialog


def curve_dialog(parent: Any, session: Any, outcomes: list[str], selected: str | None) -> Any:
    """The specification curve, in a window: choose the outcome and the coefficient; the result selected in the app
    (``selected``) is highlighted. ``dialog.draw()`` redraws; ``dialog.figure`` is the current figure."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Specification curve")
    layout = QtWidgets.QVBoxLayout(dialog)
    row = QtWidgets.QHBoxLayout()
    outcome = QtWidgets.QComboBox()
    outcome.setObjectName("curve-outcome")
    outcome.addItems(outcomes)
    term = QtWidgets.QComboBox()
    term.setObjectName("curve-term")
    row.addWidget(QtWidgets.QLabel("Outcome"))
    row.addWidget(outcome)
    row.addWidget(QtWidgets.QLabel("Coefficient"))
    row.addWidget(term)
    layout.addLayout(row)
    holder = QtWidgets.QVBoxLayout()
    layout.addLayout(holder)
    mine = [f for f in fits(session) if f["name"] == selected]
    if mine and mine[-1]["outcome"] in outcomes:
        outcome.setCurrentText(mine[-1]["outcome"])

    def terms() -> None:
        names = list(dict.fromkeys(t for f in fits(session, outcome.currentText()) for t in f["estimates"]))
        term.blockSignals(True)
        term.clear()
        term.addItems(names)
        slopes = [t for t in names if t not in ("const", "Intercept")]
        if slopes:
            term.setCurrentText(slopes[0])
        term.blockSignals(False)
        draw()

    def draw() -> None:
        while holder.count():
            old = holder.takeAt(0).widget()
            if old is not None:
                old.deleteLater()
        if not term.currentText():
            return
        dialog.figure = specification_curve(session, outcome.currentText(), term.currentText(), highlight=selected)
        holder.addWidget(FigureCanvasQTAgg(dialog.figure))

    outcome.currentIndexChanged.connect(lambda _i: terms())
    term.currentIndexChanged.connect(lambda _i: draw())
    dialog.figure = None
    dialog.draw = draw
    terms()
    return dialog
