"""Every line of every menu has an access key (Alt+M, then K: Model → Analysis → Kalman filter…), unique within its
menu: core and plugin menus, in every workspace, including the lines a menu adds as it opens. Keys are given from the
labels (pymetrix.commands.menus.with_access_keys) unless a line marks its own (E&xit, &1 for a recent project)."""

import importlib.metadata
import os

import pytest

from pymetrix import plugins
from pymetrix.commands.menus import access_key, with_access_keys


def test_keys_take_a_significant_words_first_letter_and_never_clash():
    assert with_access_keys(["Model editor…", "Define equation…", "Template model…"]) == \
        ["&Model editor…", "&Define equation…", "&Template model…"]
    # a clash takes the next significant word's first letter, then any free letter; minor words come last
    assert with_access_keys(["Estimate model…", "Estimate DSGE-VAR…", "Error of fit"]) == \
        ["&Estimate model…", "Estimate &DSGE-VAR…", "Error of &fit"]
    assert with_access_keys(["Measure of fit", "Model"]) == ["&Measure of fit", "M&odel"]
    assert with_access_keys(["Set of things", "Sat"]) == ["&Set of things", "S&at"]
    # a line that marks its own keeps it, and its letter is not given to another
    assert with_access_keys(["Exit", "E&xit", "Export"]) == ["&Exit", "E&xit", "Ex&port"]
    assert with_access_keys(["&1 project.pmx", "Clear recent"]) == ["&1 project.pmx", "&Clear recent"]
    # "&&" is a literal ampersand, not a key; letters already used elsewhere are passed in
    assert access_key("R&&D") == "" and with_access_keys(["R&&D"]) == ["&R&&D"]
    assert with_access_keys(["File"], taken="f") == ["F&ile"]
    # nothing free: left without a key rather than doubled
    assert with_access_keys(["a", "A"]) == ["&a", "A"]


def _check(menu, where, problems):
    keys: dict[str, str] = {}
    for action in menu.actions():
        if action.isSeparator() or not action.text():
            continue
        text = action.text()
        key = access_key(text)
        if not key:
            problems.append(f"{where} → {text}: no access key")
        elif key in keys:
            problems.append(f"{where}: {keys[key]!r} and {text!r} share Alt+{key.upper()}")
        else:
            keys[key] = text
        if action.menu() is not None and action.menu() is not menu:
            _check(action.menu(), f"{where} → {text.replace('&', '')}", problems)


def test_every_menu_line_has_its_own_access_key_in_every_workspace():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.ui.qt import main_window
    from pymetrix.ui.workspaces import all_workspaces

    s = Session()
    window = main_window(s)
    try:
        plugin_menus = {i.menu.partition("/")[0] for i in COMMANDS.values() if i.menu and i.plugin != "pymetrix"}
        tops = {a.text().replace("&", ""): a for a in window.menuBar().actions() if a.menu() is not None}
        assert plugin_menus <= set(tops)  # the plugins' menus are there and checked too
        # installed as the app decides it: from the plugin's entry points (a module that can be imported but whose
        # entry points are hidden or disabled registers nothing), not from find_spec
        finance = [ep for ep in importlib.metadata.entry_points(group="pymetrix.plugins")
                   if getattr(ep.dist, "name", "") == "pymetrix-finance" and not plugins.is_disabled("pymetrix-finance")]
        if finance:
            assert "Finance" in plugin_menus
        for ws in all_workspaces():
            window.switch_workspace(ws.key)
            problems: list[str] = []
            for title, top in tops.items():
                if top.isVisible():
                    menu = top.menu()
                    menu.aboutToShow.emit()  # as it opens: the lines it adds then (Undo's step, the open windows)
                    for sub in menu.findChildren(QtWidgets.QMenu):
                        sub.aboutToShow.emit()
                    _check(menu, title, problems)
            assert not problems, f"{ws.key}: " + "; ".join(problems)
        model = window.command_menus["Model"]
        assert [a.text() for a in model.actions() if not a.isSeparator()][:3] == \
            ["&Model editor…", "&Define equation…", "&Template model…"]
        assert window.quit_action.text() in ("E&xit", "&Quit")
    finally:
        s.dirty = False
        window.switch_workspace("all")
        window.close()
