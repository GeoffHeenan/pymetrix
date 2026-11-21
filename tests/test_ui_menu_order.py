"""The menus as a first-time user meets them: items where they are looked for, access keys, Exit, plain shortcuts."""

import os

import pytest

from pymetrix.commands import COMMANDS, Session


@pytest.fixture(scope="module")
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    from pymetrix.ui.qt import main_window

    w = main_window(Session())
    yield w
    w.state["session"].dirty = False
    w.close()


def _items(menu, path=""):
    for action in menu.actions():
        if action.menu() is not None:
            yield from _items(action.menu(), f"{path}{action.menu().title().replace('&', '')}/")
        elif not action.isSeparator():
            yield f"{path}{action.text().replace('&', '')}", action


def _menu(window, title):
    return next(a.menu() for a in window.menuBar().actions() if a.menu() and a.text().replace("&", "") == title)


def test_items_are_where_they_are_looked_for(window):
    forecast = dict(_items(_menu(window, "Forecast")))
    assert list(forecast)[0] == "Forecast…"  # a forecast from an estimated model: in the Forecast menu, first
    assert "Forecast…" not in dict(_items(_menu(window, "Estimate")))
    tests = dict(_items(_menu(window, "Test")))
    assert "Gregory-Hansen cointegration test…" in tests  # beside Johansen, not among the estimators
    names = [t for t in tests]
    assert names.index("Gregory-Hansen cointegration test…") == names.index("Johansen cointegration…") + 1
    calculate = dict(_items(_menu(window, "Calculate")))
    assert "Random generator/PCG64 (default)" in calculate  # a setting of the calculation, not of the view
    assert not any(p.startswith("Random generator") for p in dict(_items(_menu(window, "View"))))


def test_the_menu_bar_has_access_keys_each_letter_once(window):
    titles = [a.text() for a in window.menuBar().actions() if a.menu() is not None]
    assert all("&" in t for t in titles), titles
    letters = [t[t.index("&") + 1].lower() for t in titles]
    assert len(letters) == len(set(letters)), titles
    assert "&File" in titles and "&Help" in titles and "E&stimate" in titles


def test_a_plugin_menu_takes_a_free_letter():
    from pymetrix.ui.qt import with_access_key

    assert with_access_key("Finance") == "F&inance"  # F is File's
    assert with_access_key("Finance", taken="i") == "Fi&nance"
    assert with_access_key("Data") == "&Data"


def test_file_ends_with_exit_on_ctrl_q(window):
    file_menu = _menu(window, "File")
    last = [a for a in file_menu.actions() if not a.isSeparator()][-1]
    assert last is window.quit_action and last.text().replace("&", "") in ("Exit", "Quit")
    assert last.shortcut().toString() == "Ctrl+Q"


def test_redo_lists_the_documented_keys_only(window):
    from pymetrix.ui.qt import shortcut_list

    from PySide6.QtGui import QKeySequence

    def native(*keys):  # as the list shows them: the platform's own words ("⌘Y, ⇧⌘Z" on a Mac)
        return ", ".join(QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) for k in keys)

    redo = [keys for where, label, keys in shortcut_list(window) if label == "Redo"]
    assert redo == [native("Ctrl+Y", "Ctrl+Shift+Z")]


def test_core_workspaces_keep_ctrl_1_to_6_with_a_plugin_workspace():
    from pymetrix.extension import Workspace
    from pymetrix.ui.workspaces import WORKSPACES, workspace_numbers

    plugin = Workspace("finance", "Finance", ("Finance",), ("Objects",), (), "a plugin's")
    order = WORKSPACES[:-1] + (plugin,) + WORKSPACES[-1:]  # plugins sit before Report on the bar
    numbers = workspace_numbers(order)
    assert numbers["report"] == 6 and numbers["all"] == 1 and numbers["finance"] == 7


def test_finance_groups_its_items_in_submenus(window):
    if "Finance" not in [a.text().replace("&", "") for a in window.menuBar().actions()]:
        pytest.skip("the finance plugin is not installed")
    finance = _menu(window, "Finance")
    tops = [a.text().replace("&", "") for a in finance.actions() if not a.isSeparator()]
    assert tops[:2] == ["Returns…", "Performance…"]
    assert {"Risk", "Copulas", "Portfolios", "Options", "Fixed income", "Credit"} <= set(tops)
    assert len(tops) <= 10
    items = dict(_items(finance))
    assert "Risk/Value at risk…" in items and "Copulas/DCC copula…" in items and "Credit/CDS curve…" in items
