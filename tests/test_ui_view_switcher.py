"""Result windows list their views down the side, grouped, every one in view (a fitted equation has 13 views, which as
tabs ran behind scroll arrows); the keyboard moves between them, and the view picked last opens next time."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import FitEquation, Session  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fit():
    data = Dataset.example("sample_macro")
    from pymetrix import Equation

    return Equation("c ~ y + c[-1]").fit(data)


def _settings():
    return QtCore.QSettings(QtCore.QSettings.defaultFormat(), QtCore.QSettings.Scope.UserScope, "pymetrix", "desktop")


def _view(obj):
    from pymetrix.ui.qt import object_view

    return object_view(obj, name="fit", code="fit = ...")


def test_every_view_of_a_fitted_equation_is_listed_and_grouped(app, fit):
    from pymetrix.ui.view_switcher import ViewSwitcher

    _settings().remove("result_views/FittedEquation")
    view = _view(fit)
    assert isinstance(view, ViewSwitcher)
    titles = view.titles()
    assert len(titles) == 13 and titles[0] == "Summary" and titles[-1] == "Python access"
    listed = [view.tabText(i) for i in view.listed()]
    assert sorted(listed) == sorted(titles)  # every view, once
    groups = view.groups()
    assert list(groups) == ["Summary", "Tables", "Graphs", "Diagnostics", "Provenance"]
    assert groups["Summary"] == ["Summary"] and groups["Provenance"] == ["Provenance", "Python access"]
    assert "Graph" in groups["Graphs"] and {"Residuals", "Diagnostics", "Q-Q plot", "Correlogram"} <= set(groups["Diagnostics"])
    assert view.headings_shown()  # 13 views: under their group headings
    assert view.currentIndex() == 0 and view.current_title() == "Summary"
    assert view.select("Coefficients") and view.currentWidget() is view.widget(titles.index("Coefficients"))
    assert not view.select("No such view")
    view.deleteLater()


def test_a_short_list_has_no_headings_and_one_view_no_list(app):
    from pymetrix.ui.qt import object_view

    data = Dataset.example("sample_macro")
    view = object_view(data, name="data")
    assert 1 < view.count() < 6 and not view.headings_shown() and view.side.isVisibleTo(view)
    from pymetrix.figures import Figure, Line, Panel

    single = object_view(Figure((Panel((Line(data["y"], "y"),)),)), name="fig")
    assert single.count() == 1 and not single.side.isVisibleTo(single)  # one view: nothing to choose
    view.deleteLater()
    single.deleteLater()


def test_the_keyboard_moves_between_views_in_listed_order(app, fit):
    _settings().remove("result_views/FittedEquation")
    view = _view(fit)
    view.resize(680, 460)
    view.show()
    order = view.listed()
    view.side.setFocus()
    QTest.keyClick(view.side, QtCore.Qt.Key.Key_Down)
    assert view.currentIndex() == order[1]  # Down: the next view, past the group heading
    QTest.keyClick(view.side, QtCore.Qt.Key.Key_PageDown, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert view.currentIndex() == order[2]  # Ctrl+PgDown, as between tabs
    view.step(-1)
    view.step(-1)
    view.step(-1)
    assert view.currentIndex() == order[-1]  # round from the start to the end
    view.close()
    _settings().remove("result_views/FittedEquation")


def test_the_view_picked_last_opens_next_time_and_a_refresh_keeps_it(app, fit):
    _settings().remove("result_views/FittedEquation")
    view = _view(fit)
    view.setCurrentIndex(view.titles().index("Diagnostics"))  # set in code: not remembered
    assert _view(fit).current_title() == "Summary"
    view.side.setCurrentRow(view._row_of(view.titles().index("Coefficients")))  # the reader's pick
    assert _view(fit).current_title() == "Coefficients"
    _settings().remove("result_views/FittedEquation")

    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y", name="fit"))
    window = main_window(s)
    try:
        window.show_object("fit")
        sub = window.mdi.activeSubWindow()
        sub.widget().select("Residuals")
        s.run(FitEquation("c ~ y + c[-1]", name="fit"))  # the result changes: its window is rebuilt
        window.refresh()
        assert sub.widget().current_title() == "Residuals"
    finally:
        s.dirty = False
        window.close()


def test_groups_by_title():
    from pymetrix.ui.view_switcher import view_group

    assert view_group("Summary") == "Summary" and view_group("Paths", first=True) == "Summary"
    assert view_group("Residual covariance") == "Tables" and view_group("Residuals") == "Diagnostics"
    assert view_group("Residual correlogram graph", figure=True) == "Diagnostics" and view_group("Graph", figure=True) == "Graphs"
    assert view_group("Python access") == "Provenance" and view_group("Coefficients") == "Tables"
