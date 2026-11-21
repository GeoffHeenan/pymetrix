"""Plugins in the app: the manager (Help → Plugins…), the question asked about a new plugin before it runs, and the
note shown when a project needs a plugin that is not here."""

from __future__ import annotations

from typing import Any


def _provides(name: str) -> dict[str, list[str]]:
    """What a loaded plugin registered, by group: methods, tests, data readers (with their address schemes)... and the
    dialogs, with their menus."""
    from .. import plugins
    from ..commands.registry import COMMAND_IDS

    out: dict[str, list[str]] = {}
    for group, reg in plugins._REGISTRIES.items():
        for entry in reg._entries.values():
            if entry.plugin.name == name:
                label = f"{entry.name}:// (reader)" if group == "io" and not entry.name.startswith(("write", "read")) else entry.label
                out.setdefault(group, []).append(label)
    for cid, info in COMMAND_IDS.items():
        if info.plugin == name:
            from ..wording import menu_label  # only the first "/" separates the menu: "Model/Import FRB/US XML file..."

            out.setdefault("dialogs", []).append(f"{info.menu.partition('/')[0]} → {menu_label(info.menu)}" if info.menu else "(no menu)")
    return out


# What each registry group holds, in words: (one, many)
GROUP_WORDS = {"methods": ("estimation method", "estimation methods"), "systems": ("system estimator", "system estimators"),
               "models": ("model", "models"), "variances": ("variance model", "variance models"), "tests": ("test", "tests"),
               "filters": ("filter", "filters"), "seasonal": ("seasonal adjuster", "seasonal adjusters"),
               "solvers": ("solver", "solvers"), "io": ("data reader or writer", "data readers and writers"),
               "views": ("result view", "result views"), "commands": ("dialog", "dialogs"), "dialogs": ("dialog", "dialogs"),
               "courses": ("course", "courses"), "workspaces": ("workspace", "workspaces"), "menus": ("menu", "menus"),
               "panels": ("panel", "panels"), "report_templates": ("report template", "report templates"),
               "datasets": ("example dataset", "example datasets"), "observers": ("step observer", "step observers"),
               "report_notes": ("report note", "report notes")}


def group_words(group: str, count: int | None = None) -> str:
    """A registry group in words, made plural by ``count`` ("1 panel", "2 report notes"); without a count, its plural
    as a heading ("Report notes")."""
    one, many = GROUP_WORDS.get(group, (group.replace("_", " "), group.replace("_", " ")))
    if count is None:
        return many[:1].upper() + many[1:]
    return f"{count:,} {one if count == 1 else many}"


def _rows() -> list[dict[str, Any]]:
    """Every plugin the app knows of: loaded, disabled (an installed one is then not loaded) or new."""
    from .. import plugins

    rows: dict[str, dict[str, Any]] = {}
    table = {r[0]: r for r in plugins.plugin_table().rows}
    for info in plugins.plugins():
        provided = _provides(info.name)
        rows[info.name] = {"name": info.name, "version": info.version, "status": table.get(info.name, [""] * 7)[-1] or "available",
                           "licence": info.license, "maintainer": info.maintainer, "description": info.description,
                           "requires": ", ".join(f"{k} {v}".strip() for k, v in info.requires.items()), "provides": provided,
                           "install": info.install_hint, "core": info.name == plugins.CORE.name, "new": False}
    for name in plugins.plugin_state()["disabled"]:
        rows.setdefault(name, {"name": name, "version": "", "status": "disabled", "licence": "", "maintainer": "", "description": "",
                               "requires": "", "provides": {}, "install": f"pip install {name}", "core": False, "new": False})
    for p in plugins.pending():
        rows.setdefault(p.name, {"name": p.name, "version": p.version, "status": "new: not enabled yet", "licence": p.license,
                                 "maintainer": p.author, "description": p.summary, "requires": "",
                                 "provides": {g: ["(enable it to see)"] for g in p.groups}, "install": f"pip install {p.name}",
                                 "core": False, "new": True})
    for name, why in plugins.load_problems().items():  # installed but not importable: known only by its entry points
        rows.setdefault(name, {"name": name, "version": "", "status": why, "licence": "", "maintainer": "", "description": "",
                               "requires": "", "provides": {}, "install": f"pip install {name}", "core": False, "new": False})
    for row in rows.values():
        detail = plugins.problem_detail(row["name"])
        row["problem"] = detail["detail"] if detail else ""
        if detail and detail["command"]:
            row["install"] = detail["command"]
    return list(rows.values())


def plugin_manager(parent: Any = None) -> Any:
    """Help → Plugins…: each plugin's status and what it provides; enable, disable, copy an install command."""
    from html import escape

    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    from .. import plugins

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Plugins")
    layout = QtWidgets.QVBoxLayout(dialog)
    from .metrics import set_margins

    set_margins(layout, "dialog_margin")
    note = QtWidgets.QLabel("The app and scripts share these settings. Bundled plugins are hidden or shown at once; "
                            "installed ones change when PyMetrix starts again.")
    note.setWordWrap(True)
    layout.addWidget(note)
    table = QtWidgets.QTableWidget(0, 5)
    table.setHorizontalHeaderLabels(["Plugin", "Version", "Status", "Provides", "Licence"])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    layout.addWidget(table, 1)
    details = QtWidgets.QTextBrowser()
    details.setMaximumHeight(180)
    layout.addWidget(details)
    buttons = QtWidgets.QHBoxLayout()
    enable = QtWidgets.QPushButton("Enable")
    disable = QtWidgets.QPushButton("Disable")
    copy = QtWidgets.QPushButton("Copy command")
    copy.setToolTip("copy the command that installs or repairs the chosen plugin")
    check = QtWidgets.QPushButton("Check plugin")
    check.setToolTip("run PyMetrix's conformance checks on what the plugin registers")
    close = QtWidgets.QPushButton("Close")
    for b in (enable, disable, copy, check):
        buttons.addWidget(b)
    buttons.addStretch(1)
    buttons.addWidget(close)
    layout.addLayout(buttons)
    state: dict[str, Any] = {"rows": []}

    def fill() -> None:
        state["rows"] = _rows()
        table.setRowCount(len(state["rows"]))
        for r, row in enumerate(state["rows"]):
            counts = ", ".join(group_words(k, len(v)) for k, v in row["provides"].items()) or "—"
            for c, text in enumerate((row["name"], row["version"], row["status"], counts, row["licence"])):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        header = table.horizontalHeader()  # Provides takes the spare width and wraps (its full text in the tooltip and below)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.setWordWrap(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.resizeRowsToContents()
        show()

    def current() -> dict[str, Any] | None:
        rows = table.selectionModel().selectedRows()
        return state["rows"][rows[0].row()] if rows else None

    def show() -> None:
        row = current()
        enable.setEnabled(bool(row) and (row["status"] == "disabled" or row["new"]))
        disable.setEnabled(bool(row) and not row["core"] and row["status"] != "disabled")
        copy.setEnabled(bool(row) and not row["core"])
        check.setEnabled(bool(row) and not row["core"] and not row["new"] and row["status"] != "disabled")
        if row is None:
            details.setHtml("<p>Choose a plugin to see what it provides.</p>")
            return
        parts = [f"<h3>{row['name']} {row['version']}</h3>", f"<p>{row['description']}</p>" if row["description"] else ""]
        meta = [("Status", row["status"]), ("Maintainer", row["maintainer"]), ("Licence", row["licence"]), ("Requires", row["requires"])]
        parts.append("".join(f"<p><b>{k}:</b> {v}</p>" for k, v in meta if v))
        for group, items in row["provides"].items():
            parts.append(f"<p><b>{group_words(group)}:</b> {', '.join(items[:30])}{'…' if len(items) > 30 else ''}</p>")
        if row.get("problem"):
            parts.append(f"<p><b>Why it could not load:</b> {escape(row['problem'])}</p>"
                         f"<p><b>Command</b> (Copy command copies it):</p><pre>{escape(row['install'])}</pre>")
        if row["new"]:
            parts.append("<p>Read from the package's metadata: nothing of it has run. Enable it to load it.</p>")
        details.setHtml("".join(parts))

    def do_enable() -> None:
        row = current()
        if row:
            plugins.enable(row["name"], row["version"])
            fill()

    def do_disable() -> None:
        row = current()
        if row:
            plugins.disable(row["name"])
            fill()

    def do_check() -> None:
        row = current()
        if not row:
            return
        from html import escape

        details.setHtml(f"<h3>Checking {escape(row['name'])}</h3><p>{escape(check_report(row['name']))}</p>")

    def do_copy() -> None:
        row = current()
        if row:
            QtWidgets.QApplication.clipboard().setText(row["install"])

    table.itemSelectionChanged.connect(show)
    enable.clicked.connect(do_enable)
    disable.clicked.connect(do_disable)
    copy.clicked.connect(do_copy)
    check.clicked.connect(do_check)
    close.clicked.connect(dialog.accept)
    fill()
    from .fit import fit_to_screen

    fit_to_screen(dialog, 760, 520)
    dialog.table, dialog.details, dialog.refresh = table, details, fill
    dialog.enable_button, dialog.disable_button, dialog.copy_button, dialog.check_button = enable, disable, copy, check
    dialog.rows = lambda: state["rows"]
    _ = Qt
    return dialog


def check_report(name: str) -> str:
    """Run the conformance kit (``testing.check_plugin``) on a loaded plugin: what passed, or the first failure."""
    from .. import plugins
    from ..testing import check_plugin

    problem = plugins.problem_detail(name)
    if problem is not None:
        return f"{name} could not be loaded, so there is nothing to check: {problem['detail']}"
    info = next((p for p in plugins.plugins() if p.name == name), None)
    if info is None:
        return f"{name} is not loaded, so there is nothing to check."
    try:
        entries = check_plugin(info)
    except AssertionError as exc:
        return f"A check failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - a plugin's error is the report
        return f"Checking stopped with an error: {type(exc).__name__}: {exc}"
    return f"Passed: {len(entries)} extension{'s' if len(entries) != 1 else ''} conform (settings, help, round trips)."


def ask_new_plugins(parent: Any = None, answer: Any = None) -> list[str]:
    """Ask about each new plugin before it runs: enable it, disable it, or leave it out for now. The question shows
    only the package's metadata. ``answer(plugin) -> "enable" | "disable" | "later"`` replaces the question (tests).
    Returns the names enabled."""
    from PySide6 import QtWidgets

    from .. import plugins

    enabled = []
    for p in plugins.pending():
        if answer is not None:
            choice = answer(p)
        else:
            box = QtWidgets.QMessageBox(parent)
            box.setWindowTitle("A new plugin")
            box.setText(f"<b>{p.name} {p.version}</b> is installed and has not been used here before.")
            facts = [f"by {p.author}" if p.author else "", f"licence {p.license}" if p.license else "", p.summary]
            box.setInformativeText(" · ".join(f for f in facts if f) + "\n\nIt registers in: " + (", ".join(p.groups) or "nothing listed")
                                   + ".\nA plugin's code runs when it loads, as typed code does. Enable it only if you trust it.")
            yes = box.addButton("Enable", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            no = box.addButton("Disable", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Not now", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            choice = "enable" if clicked is yes else "disable" if clicked is no else "later"
        if choice == "enable":
            plugins.enable(p.name, p.version)
            enabled.append(p.name)
        elif choice == "disable":
            plugins.disable(p.name)
        else:
            plugins.skip_for_now(p.name)
    return enabled


def missing_plugins_note(missing: dict[str, str]) -> str:
    """The note shown after opening a project whose steps need plugins that are not here."""
    lines = "\n".join(f"• {name}: {why}" for name, why in missing.items())
    return ("Some steps of this project need plugins that are not available here. They are kept in the History, "
            f"marked as not run, and saved again unchanged:\n\n{lines}")
