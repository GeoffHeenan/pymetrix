"""Formulas checked as they are typed (pymetrix.ui.formula_check): a parse error shows under the field in red, with
the parser's reason and where, and holds OK until it is fixed; a name the data do not have is a grey hint with the
nearest name, and does not hold OK; nothing is checked until typing stops (300 ms)."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import COMMANDS, GenerateSeries, Session  # noqa: E402
from pymetrix.ui import theme  # noqa: E402
from pymetrix.ui.formula_check import DELAY_MS, check_formula, formula_fields  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    host = main_window(Session())  # plugins' commands join the registry as the window loads them
    host.hide()
    yield app
    host.state["session"].dirty = False
    host.close()


def _session():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"consumption": rng.normal(size=40).cumsum() + 50, "income": rng.normal(size=40).cumsum() + 60,
                          "rate": rng.normal(size=40)}, index=pd.period_range("2000Q1", periods=40, freq="Q", name="time"))
    session = Session()
    session.objects["data"] = Dataset.from_frame(frame)
    return session


def _dialog(cls, session):
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return command_dialog(cls, session)


def _ok(dialog):
    return dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)


def _wait_for(condition, timeout_s=5.0) -> bool:
    """Run the event loop until ``condition()`` holds or ``timeout_s`` has passed; whether it holds. A fixed wait
    assumed the check's timer had fired within DELAY_MS + 150 ms, which a loaded CI runner did not always give
    (oldest job, PySide6 6.6). PySide6 has no QTest.qWaitFor."""
    import time

    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(20)
    return bool(condition())


# --- the check itself (no Qt) --------------------------------------------------------------------------------------

def test_a_parse_error_gives_the_parsers_reason_and_where():
    found = check_formula("log(y", "expression")
    assert found.error and found.position == 5  # the parser's own position: the end, where ')' was expected
    assert found.message == "Expected ')' at character 6, after “log(y”"
    assert check_formula("y +* 2", "expression").message.startswith("Expected a number, a name or '(' at character 4")
    assert check_formula("", "equation").level == "ok" and check_formula("   ", "expression").level == "ok"  # empty: the * check


def test_equations_and_expressions_are_told_apart():
    assert check_formula("c ~ y + c[-1]", "equation").level == "ok"
    assert "not an equation" in check_formula("c + y", "equation").message  # Estimate → Equation needs a ~
    assert check_formula("c ~ y", "expression").error  # Generate series takes an expression
    assert check_formula("y > 0 and c <= 2", "auto").level == "ok"  # comparisons are not the = of an identity
    assert check_formula("y = c + i", "auto").level == "ok"


def test_an_unknown_function_is_named_without_the_list_of_all_functions():
    found = check_formula("diff(y)", "expression")
    assert found.error and "did you mean d(x)" in found.message and "known:" not in found.message
    assert "known:" not in check_formula("foo(y)", "expression").message


def test_names_not_in_the_data_are_a_hint_with_the_nearest_name():
    names = ["consumption", "income", "rate", "time"]
    found = check_formula("consumptoin ~ income + rate[-1]", "equation", names)
    assert found.level == "hint" and found.unknown == ("consumptoin",)
    assert found.message == "Not in the data: consumptoin (did you mean consumption?)."
    assert check_formula("consumption ~ income + fe(time)", "equation", names).level == "ok"  # an axis is a name too
    assert check_formula("x ~ y", "equation", None).level == "ok"  # no dataset (a model's equation): not looked up


def test_parameters_and_template_stand_ins_are_not_series():
    names = ["c", "y"]
    assert check_formula("c ~ rho*c[-1] + b*y", "equation", names, "rho, b").level == "ok"
    assert check_formula("d{s} ~ d{s}[-1] + y", "equation", names).level == "ok"  # Equation for each selected series
    assert check_formula("100*d(log({x}))", "expression", names).level == "ok"  # Generate series, For each
    found = check_formula("d{s} ~ (y", "equation", names)
    assert found.error and "d{s} ~ (y" in found.message  # the stand-in as typed, not what it was parsed as
    found = check_formula("c ~ rho*c[-1]", "equation", names, "rho, b", equation_class=True)
    assert found.error and "not used" in found.message  # what Equation itself checks: a parameter named but not used


def test_the_fields_checked():
    from pymetrix.commands.builtin import FitEquation
    from pymetrix.forms import form_spec

    assert formula_fields(FitEquation, form_spec(FitEquation)) == {"formula": "equation", "where": "expression"}
    assert formula_fields(GenerateSeries, form_spec(GenerateSeries)) == {"formula": "expression"}


# --- in the dialog -------------------------------------------------------------------------------------------------

def test_typing_a_bad_formula_shows_the_message_and_holds_ok(app):
    session = _session()
    dialog = _dialog(GenerateSeries, session)
    dialog.show()
    dialog.editors["series"].set("lc")
    dialog.editors["formula"].set("log(consumption")
    line = dialog.editors["formula"].message  # not while typing: a moment after it stops (asserted before the event
    assert not line.isVisible() and _ok(dialog).isEnabled() and dialog.formula_timer.isActive()  # loop can fire the timer)
    assert dialog.formula_timer.isSingleShot() and dialog.formula_timer.interval() == DELAY_MS
    assert _wait_for(lambda: line.isVisible() and not dialog.formula_timer.isActive())  # the check, once typing stops
    assert line.text() == "Expected ')' at character 16, after “log(consumption”"
    assert theme.css("error") in line.styleSheet() and dialog.editors["formula"].widget.property("invalid") is True
    assert not _ok(dialog).isEnabled() and _ok(dialog).toolTip().startswith("Formula: Expected ')'")
    _ok(dialog).click()  # OK forced (Enter in a field): nothing runs or is recorded
    assert session.history == [] and "lc" not in session.objects["data"].names and dialog.result_value is None
    dialog.editors["formula"].set("log(consumption)")  # fixed: the hold goes
    assert _wait_for(lambda: not dialog.formula_timer.isActive() and not line.isVisible())
    assert _ok(dialog).isEnabled() and not dialog.ok_gate.held()
    dialog.close()


def test_an_unknown_series_suggests_a_close_name_without_holding_ok(app):
    dialog = _dialog(GenerateSeries, _session())
    dialog.editors["series"].set("x")
    dialog.editors["formula"].set("incme / 4")
    dialog.check_formulas()
    line = dialog.editors["formula"].message
    assert not line.isHidden() and line.text() == "Not in the data: incme (did you mean income?)."
    assert theme.css("error") not in line.styleSheet() and _ok(dialog).isEnabled()
    dialog.close()


def test_rows_where_and_estimate_equation_are_checked_too(app):
    from pymetrix.commands.builtin import FitEquation

    dialog = _dialog(FitEquation, _session())
    dialog.editors["formula"].set("consumption ~ income +")
    dialog.check_formulas()
    assert dialog.formula_checks["formula"].error and not _ok(dialog).isEnabled()
    dialog.editors["formula"].set("consumption ~ income")
    dialog.editors["where"].set("rate >")
    dialog.check_formulas()
    assert dialog.formula_checks["where"].error and "(in Advanced settings)" in _ok(dialog).toolTip()
    dialog.editors["where"].set("rate > 0")
    dialog.check_formulas()
    assert _ok(dialog).isEnabled()
    dialog.close()


def test_a_hidden_formula_field_holds_nothing(app):
    info = next(i for i in COMMANDS.values() if i.cls.__name__ == "MatrixFrom")
    dialog = _dialog(info.cls, _session())
    from pymetrix.forms import form_spec

    spec = next(s for s in form_spec(info.cls) if s.name == "formula")
    dialog.editors["formula"].set("y ~ (")
    dialog.check_formulas()
    current = {k: e.get() for k, e in dialog.editors.items()}
    assert dialog.formula_checks["formula"].error == spec.visible(current)
    assert ("formula:formula" in dialog.ok_gate.reasons) == spec.visible(current)
    dialog.close()


def test_no_dialog_opens_with_its_formula_held(app):
    """Every dialog as it opens (their defaults): no formula holds OK, and each formula field has its line."""
    session = _session()
    from pymetrix.forms import form_spec

    checked = 0
    for info in list(COMMANDS.values()):
        fields = formula_fields(info.cls, form_spec(info.cls))
        if not fields:
            continue
        dialog = _dialog(info.cls, session)
        try:
            assert not [k for k in dialog.ok_gate.reasons if k.startswith("formula:")], info.cls.__name__
            assert all(hasattr(dialog.editors[n], "message") for n in fields), info.cls.__name__
            checked += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert checked >= 12
