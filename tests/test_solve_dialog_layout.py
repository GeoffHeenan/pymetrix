"""Model > Solve model…: the chosen solver's own settings (Newton's line search and Jacobian refresh...) sit with the
other solver settings in the Advanced section, under the solver's name, and change with the solver chosen; the step
the dialog runs is the API's call, as before. File > Browse series… names its two choices once each ("Provider" was
shown twice for SDMX). The layout checks use the forms' rows and the widgets' order, not sizes, so no font or screen
enters them."""

import os
import warnings

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix import Dataset, Equation, Model  # noqa: E402
from pymetrix.commands import FitEquation, DefineEquation, Session, SolveModel  # noqa: E402
from pymetrix.models.solvers import GaussSeidel, Newton  # noqa: E402

Role = QtWidgets.QFormLayout.ItemRole


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    s.run(FitEquation("i ~ y[-1] + i[-1]", name="inv"))
    s.run(DefineEquation("y = c + i + g", name="identity"))
    return s


def _dialog(cls, session):
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return command_dialog(cls, session)


def _row(widget):
    """(the form layout holding ``widget``, its row, the row's label text)."""
    for form in widget.window().findChildren(QtWidgets.QFormLayout):
        if form.indexOf(widget) >= 0:
            row, _ = form.getWidgetPosition(widget)
            label = form.itemAt(row, Role.LabelRole)
            return form, row, label.widget().text() if label is not None and label.widget() is not None else ""
    raise AssertionError(f"{widget} is in no form")


def _sub_labels(part):
    form = part.advanced_form
    return [form.itemAt(i, Role.LabelRole).widget().text().replace("&", "") for i in range(form.rowCount())]


def test_the_solvers_settings_sit_with_the_solver_settings(app, session):
    d = _dialog(SolveModel, session)
    try:
        d.show_advanced(True)
        app.processEvents()
        part = d.editors["solver_settings"]
        form, row, label = _row(part.advanced_part)
        fallback_form, fallback_row, _ = _row(d.editors["fallback"].outer)
        ordering_form, ordering_row, _ = _row(d.editors["ordering"].outer)
        heading = form.itemAt(ordering_row - 1, Role.SpanningRole).widget().text()
        # in the same form, just after Ordering and Fallback under the one "Solver settings" heading, named Newton
        assert form is fallback_form is ordering_form and row == fallback_row + 1 == ordering_row + 2
        assert heading == "<b>Solver settings</b>" and label == "Newton"
        assert {"Line search", "Refresh"} <= set(_sub_labels(part)) and not part.advanced_part.isHidden()
        texts = [w.text() for w in d.findChildren(QtWidgets.QLabel) if not w.isHidden()]
        assert sum(t.replace("<b>", "").replace("</b>", "") == "Solver settings" for t in texts) == 1  # no second label
        d.editors["solver"].set("Gauss-Seidel")  # its own settings, under its name; no line search
        app.processEvents()
        assert _row(part.advanced_part)[2] == "Gauss-Seidel" and "Line search" not in _sub_labels(part)
        d.editors["solver"].set("First order (linearised)")  # no settings of its own: the row goes
        app.processEvents()
        assert part.advanced_part.isHidden() and not d.editors["fallback"].outer.isHidden()
        d.editors["solver"].set("Newton")
        app.processEvents()
        assert _row(part.advanced_part)[2] == "Newton" and "Line search" in _sub_labels(part) and not part.advanced_part.isHidden()
    finally:
        d.close()


def test_the_dialogs_step_is_unchanged(app, session):
    """Newton's line search and modified Newton set in the dialog: the step is SolveModel with those settings, its
    solution the API's (Model.solve with Newton(line_search=True, refresh="period")), and its script replays."""
    d = _dialog(SolveModel, session)
    try:
        d.editors["equations"].set("cons, inv, identity")
        d.editors["start"].set("2020Q1")
        d.editors["end"].set("2024Q4")
        app.processEvents()
        part = d.editors["solver_settings"]
        part.sub["line_search"].set(True)
        part.sub["refresh"].set("period")
        app.processEvents()
        ok = d.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok.click()
        app.processEvents()
        want = SolveModel("cons, identity, inv", "2020Q1", "2024Q4", solver_settings="line_search=True, refresh='period'")
        assert d.command == want and d.command.to_code() == want.to_code()
    finally:
        d.close()
    o = session.objects
    api = Model([o["cons"], o["inv"], o["identity"]]).solve(o["data"], ("2020Q1", "2024Q4"),
                                                            solver=Newton(line_search=True, refresh="period"))
    got = session["solution"]
    np.testing.assert_allclose(got.frame[["c", "i", "y"]].to_numpy(), api.frame[["c", "i", "y"]].to_numpy(), rtol=0, atol=1e-12)
    assert "solver=Newton(line_search=True, refresh='period')" in session.script()
    ns = {"data": o["data"]}  # put in the session by hand, not by a step
    exec(session.script(), ns)
    np.testing.assert_allclose(ns["solution"].frame[["c", "i", "y"]].to_numpy(), got.frame[["c", "i", "y"]].to_numpy(),
                               rtol=0, atol=1e-12)


def test_the_default_step_is_unchanged(app, session):
    """Left at its defaults the dialog makes the same step as before: Newton with no settings."""
    d = _dialog(SolveModel, session)
    try:
        d.editors["equations"].set("cons, inv, identity")
        app.processEvents()
        values = {name: editor.get() for name, editor in d.editors.items()}
        assert values["solver"] == "Newton" and values["solver_settings"] == ""
        assert SolveModel(**values)._solver() == Newton() and SolveModel(**values).ordering == "joint"
        d.editors["solver"].set("Gauss-Seidel")
        app.processEvents()
        assert SolveModel(**{name: editor.get() for name, editor in d.editors.items()})._solver() == GaussSeidel()
    finally:
        d.close()


def test_browse_series_names_the_provider_once(app):
    """With SDMX chosen, the row read "Provider [SDMX]  Provider [ECB]": the second is the source (ECB, IMF...),
    source= in connectors.search, as it is called for the World Bank."""
    from pymetrix.ui.series_browser_qt import SeriesBrowser

    b = SeriesBrowser(Session())
    try:
        for key, want in (("fred", ["Provider"]), ("worldbank", ["Provider", "Source"]), ("sdmx", ["Provider", "Source"])):
            b.set_provider(key)
            app.processEvents()
            labels = [w.text() for w in b.dialog.findChildren(QtWidgets.QLabel)
                      if not w.isHidden() and w.text() in ("Provider", "Source")]
            assert sorted(labels) == want, key
    finally:
        b.dialog.close()
