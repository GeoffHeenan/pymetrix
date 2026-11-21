"""The reviewer demo, rehearsed: the 15-minute demo script and the first hour of the review brief, step by step
through the desktop app's own menus (off-screen Qt). Dialogs are filled in and accepted as a presenter would; any
error box fails the test. A change that breaks the path reviewers are shown fails here first."""

import os
import subprocess
import sys

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")

import PySide6  # noqa: E402
from packaging.version import Version  # noqa: E402

if Version(PySide6.__version__) < Version("6.7"):
    # PySide6 6.6 (the oldest-versions CI job) can hand back a Python object for a live menu action that says it is
    # already deleted: looking menus up from Python, as this test does, fails, while the app's menus work as ever.
    pytest.skip("menu actions cannot be looked up reliably from Python before PySide6 6.7", allow_module_level=True)


@pytest.fixture
def app(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    state = {"errors": [], "pending": [], "item": None, "save": tmp_path / "script.py"}

    def box(kind):
        def answer(parent, title, text, *args, **kwargs):
            if kind == "critical":
                state["errors"].append(f"{title}: {text}")
            if kind == "question" and "save" in str(text).lower():
                return QtWidgets.QMessageBox.StandardButton.Discard  # switching demos: the rehearsal's changes go
            return QtWidgets.QMessageBox.StandardButton.Yes
        return staticmethod(answer)

    for kind in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, box(kind))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(state["save"]), "")))

    from pymetrix.ui import demo_chooser

    monkeypatch.setattr(demo_chooser, "choose_demo", lambda parent=None: state["item"])  # File → Open demo project: the demo

    def driven_exec(dialog):
        """A dialog as a presenter fills it: the pending step's values, then OK."""
        dialog.show()
        qapp.processEvents()
        values = state["pending"].pop(0)
        for name, value in values.items():
            dialog.editors[name].set(value)
        qapp.processEvents()
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        qapp.processEvents()
        return 0 if dialog.isVisible() else 1

    monkeypatch.setattr(QtWidgets.QDialog, "exec", driven_exec)
    state["qapp"] = qapp
    return state


def _menu(window, path):
    """Trigger a menu item by its path, e.g. 'Estimate/Equation'."""
    from shiboken6 import isValid

    def walk(menu, prefix):
        for action in menu.actions():
            # the View menu's window list is rebuilt as windows close: older PySide6 still lists deleted entries
            if not isValid(action):
                continue
            try:
                text, sub = prefix + action.text().replace("&", "").rstrip(".…"), action.menu()
            except RuntimeError:
                continue
            if sub is not None and isValid(sub):
                yield from walk(sub, text + "/")
            elif sub is None:
                yield text, action
    found = dict(item for top in window.menuBar().actions() if top.menu()
                 for item in walk(top.menu(), top.text().replace("&", "") + "/"))
    assert path in found, f"no menu item {path!r}"
    found[path].trigger()


def test_the_demo_script_runs_through_the_app(app):
    from pymetrix import demo
    from pymetrix.ui.qt import main_window

    qapp = app["qapp"]
    window = main_window(demo("consumption"))
    window.resize(1400, 900)
    window.show()
    try:
        # 0:00 the app and a demo: a result's Fields name the Python that reaches each number
        window.show_object("dynamic")
        tabs = window.mdi.activeSubWindow().widget()
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        assert "Summary" in titles and "Python access" in titles
        tabs.setCurrentIndex(titles.index("Python access"))
        table = tabs.currentWidget().findChild(QtWidgets.QTableWidget)
        assert 'dynamic.stats["aic"]' in [table.verticalHeaderItem(r).text() for r in range(table.rowCount())]
        _menu(window, "View/Zoom in")
        _menu(window, "View/Actual size")

        # 0:03 a step of your own, a test on it, a new sample, recalculation
        session = window.state["session"]
        steps = len(session.history)
        app["pending"].append({"formula": "c ~ y + c[-1] + i"})
        _menu(window, "Estimate/Equation")
        assert len(session.history) == steps + 1 and "c ~ y + c[-1] + i" in session.history[-1].code
        app["pending"].append({"source": "fit", "test": "breusch_godfrey"})
        _menu(window, "Test/Run test")
        assert len(session.history) == steps + 2 and session.history[-1].ok
        app["pending"].append({"start": "2010Q1", "end": "2024Q4"})
        _menu(window, "Data/Sample/Set sample")
        assert {"static", "dynamic", "fit"} <= set(session.stale())
        _menu(window, "Calculate/Recalculate")
        assert not session.stale()

        # the exported script runs without the app
        _menu(window, "File/Export script")
        run = subprocess.run([sys.executable, str(app["save"])], capture_output=True, text=True, timeout=600)
        assert run.returncode == 0, run.stderr[-500:]

        # every result knows the step that made it
        window.show_object("fit")
        button = next(b for b in window.mdi.activeSubWindow().widget().findChildren(QtWidgets.QAbstractButton)
                      if b.text() == "Copy the step's code")
        button.click()
        assert QtWidgets.QApplication.clipboard().text().startswith("fit = Equation('c ~ y + c[-1] + i')")

        # 0:07 models and Bayesian estimation
        app["item"] = "nk_estimation"
        _menu(window, "File/Open demo project")
        session = window.state["session"]
        assert session.demo == "nk_estimation" and "posterior" in session.objects
        window.show_object("posterior")
        notes = " ".join(session["posterior"].summary().notes)
        assert "Acceptance rate" in notes and "Log marginal data density" in notes
        app["item"] = "policy_and_nonlinear"
        _menu(window, "File/Open demo project")
        for name in ("zlb_episode", "commitment", "discretion"):
            window.show_object(name)
        area = window.mdi.viewport().rect()
        assert all(area.contains(sub.geometry()) for sub in window.mdi.subWindowList())  # every window in view

        # if something goes wrong
        _menu(window, "File/Reload demo project")
        assert window.state["session"].demo == "policy_and_nonlinear"
        assert not app["errors"], app["errors"]
    finally:
        window.state["session"].dirty = False
        window.close()
        qapp.processEvents()
