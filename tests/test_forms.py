"""Generated dialogs: widget choice, conditional fields, series pickers, test settings sub-forms."""

import importlib.util
import os

import pytest

from pymetrix import Dataset, Equation
from pymetrix.commands import FitEquation, FitVAR, RunTest, Session, form_spec

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def session(ts):
    s = Session()
    s.objects["data"] = ts
    s.objects["fit"] = Equation("y ~ x + z").fit(ts)
    return s


def test_widget_choice(app, session):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    ed = dialog.editors
    methods = [ed["method"].widget.itemText(i) for i in range(ed["method"].widget.count())]
    assert methods[0] == "OLS" and "Robust regression (M)" in methods
    assert ("LIML" in methods) == (importlib.util.find_spec("linearmodels") is not None)  # a plugin's, when installed
    assert ed["missing"].group.buttons()[0].text() == "Drop rows with missing values"  # two choices: radio buttons, named
    assert ed["missing"].group.buttons()[0].choice == "drop"  # the code is what is stored
    assert isinstance(ed["family"].widget, QtWidgets.QComboBox)  # four choices: dropdown
    assert isinstance(ed["endog"].widget, QtWidgets.QListWidget)  # series checklist
    assert [ed["endog"].widget.item(i).text() for i in range(ed["endog"].widget.count())] == ts_names(session)


def ts_names(session):
    return session["data"].names


def test_fields_appear_when_relevant(app, session):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    assert not dialog.row_visible("endog") and not dialog.row_visible("ar")
    dialog.editors["method"].widget.setCurrentText("IV")
    assert dialog.row_visible("endog") and dialog.row_visible("instruments") and not dialog.row_visible("ar")
    dialog.editors["method"].widget.setCurrentText("ML")
    assert dialog.row_visible("ar") and not dialog.row_visible("dist")
    dialog.editors["variance"].widget.setCurrentText("GARCH")
    assert dialog.row_visible("dist")


def test_series_checklist_builds_the_command(app, session):
    from PySide6.QtCore import Qt

    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitVAR, session)
    items = dialog.editors["variables"].widget
    for i in range(items.count()):
        if items.item(i).text() in ("y", "x"):
            items.item(i).setCheckState(Qt.CheckState.Checked)
    assert dialog.editors["variables"].get() == "y, x"
    assert isinstance(dialog.editors["max_lags"].widget, QtWidgets.QSpinBox)
    assert dialog.editors["max_lags"].widget.maximum() == 24


def test_test_settings_sub_form(app, session):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(RunTest, session)
    dialog.editors["source"].widget.setCurrentText("fit")
    dialog.editors["test"].widget.setCurrentText("breusch_godfrey")
    options = dialog.editors["options"]
    assert list(options.sub) == ["lags", "auxiliary_constant"] and isinstance(options.sub["lags"].widget, QtWidgets.QSpinBox)
    assert options.get() == ""  # default lags
    options.sub["lags"].widget.setValue(6)
    assert options.get() == "lags=6"
    options.sub["auxiliary_constant"].widget.setChecked(True)  # statsmodels' convention without a constant
    assert options.get() == "lags=6, auxiliary_constant=True"
    options.sub["auxiliary_constant"].widget.setChecked(False)
    dialog.editors["test"].widget.setCurrentText("chow")
    assert list(dialog.editors["options"].sub) == ["at"]
    dialog.editors["source"].widget.setCurrentText("data")
    offered = [dialog.editors["test"].widget.itemText(i) for i in range(dialog.editors["test"].widget.count())]
    assert {"adf", "kpss", "engle_granger", "phillips_perron", "zivot_andrews", "dfgls"} <= set(offered) and "chow" not in offered
    dialog.editors["test"].widget.setCurrentText("adf")
    options = dialog.editors["options"]
    assert list(options.sub) == ["variable", "lags", "trend", "max_lags", "backend_options", "critical_values", "draws", "seed"]
    assert "backend_options" not in options.get()  # left out while empty


def test_advanced_section_and_settings(app, session):
    from pymetrix.commands import FitVECM
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    assert not hasattr(dialog, "tabs")  # one mechanism: a collapsed section at the bottom of the form, no Advanced tab
    assert dialog.advanced_toggle.text().replace("&", "") == "Advanced settings" and not dialog.advanced_toggle.isChecked()
    assert dialog.in_advanced("hac_lags") and dialog.in_advanced("weights") and not dialog.in_advanced("cov")
    assert not dialog.row_visible("hac_lags")
    dialog.editors["cov"].set("hac")
    assert dialog.row_visible("hac_lags")
    vecm = command_dialog(FitVECM, session)
    assert vecm.editors["signif"].get() == 0.05  # typed value, not the label "0.05"
    test = command_dialog(RunTest, session)
    test.editors["source"].widget.setCurrentText("data")
    test.editors["test"].widget.setCurrentText("adf")
    sub = test.editors["options"]
    assert not hasattr(sub, "advanced_toggle")  # the test's advanced settings: in the dialog's one section, not their own
    assert sub.advanced_form.rowCount() > 0 and test.advanced_toggle is not None
    sub.sub["max_lags"].widget.setText("6")
    assert sub.get() == "max_lags=6"  # typed text that is a number stays a number


def test_jupyter_forms_follow_the_same_rules(session):
    pytest.importorskip("ipywidgets")
    from pymetrix.ui.jupyter import form

    f = form(FitEquation, session)
    assert type(f.inputs["method"]).__name__ == "Dropdown" and "Random effects" in f.inputs["method"].options
    assert f.inputs["endog"].layout.display == "none"
    f.inputs["method"].value = "IV"
    assert f.inputs["endog"].layout.display is None and "x" in f.inputs["endog"].options
    t = form(RunTest, session)
    t.inputs["source"].value = "fit"
    t.inputs["test"].value = "breusch_godfrey"
    assert t.values()["options"] == ""
    v = form(FitVAR, session)
    v.inputs["variables"].value = ("y", "x")
    assert v.values()["variables"] == "y, x"


def test_options_round_trip(session):
    session.run(RunTest(source="fit", test="breusch_godfrey", options="lags=6"))
    assert "test = fit.test('breusch_godfrey', lags=6)" in session.script()
    specs = {s.name: s for s in form_spec(FitEquation)}
    field, allowed = specs["endog"].show_if
    assert field == "method" and set(allowed()) == {"IV", "LIML", "IV-GMM", "GMM"} and specs["method"].kind == "extension" and specs["missing"].radio


def _accept(dialog):
    dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()


def test_optional_dropdowns_start_blank(app, session):
    """A dropdown for an optional field must not quietly pick its first option (it once made every fit weighted)."""
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    assert dialog.editors["weights"].get() == ""
    dialog.editors["formula"].set("y ~ x")
    _accept(dialog)
    assert dialog.command.weights == "" and "weights" not in session.script().splitlines()[-1]


def test_setting_several_items_keeps_them_all(app, session):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitVAR, session)
    dialog.editors["variables"].set("y, x, z")  # after the dialog listens for changes, as a test or macro would
    assert dialog.editors["variables"].get() == "y, x, z"


def test_codes_show_as_words(app, session):
    from pymetrix.ui.qt import command_dialog

    var = command_dialog(FitVAR, session)
    assert [b.text() for b in var.editors["trend"].group.buttons()] == ["Constant", "Constant and trend", "None"]
    var.editors["trend"].set("ct")
    assert var.editors["trend"].get() == "ct"
    eq = command_dialog(FitEquation, session)
    cov = eq.editors["cov"].widget
    assert cov.currentText() == "Conventional" and "HAC (Newey-West)" in [cov.itemText(i) for i in range(cov.count())]
    eq.editors["cov"].set("hac")
    assert eq.editors["cov"].get() == "hac" and eq.row_visible("hac_lags")
    labels = {s.name: s.label for s in form_spec(FitEquation)}
    assert labels["cov"] == "Standard errors" and labels["name"] == "Save as"


def test_example_data_scenarios_and_report_files(app, tmp_path):
    from pymetrix.commands import DefineEquation, ExportReport, LoadExample, SolveModel
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.run(LoadExample())
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    s.run(FitEquation("i ~ y[-1]", name="inv"))
    s.run(DefineEquation("y = c + i + g", name="identity"))
    s.run(SolveModel("cons, inv, identity", "2020Q1", "2024Q4"))
    s.run(SolveModel("cons, inv, identity", "2020Q1", "2024Q4", shift="g=5", name="more_g"))
    multiplier = (s["more_g"]["y"] - s["solution"]["y"]).loc["2020Q1":"2024Q4"] / 5
    assert 1 < multiplier.iloc[0] < multiplier.iloc[-1]  # Lab 2's conclusion, from the dialogs
    script = s.script()
    assert "data = Dataset.example('sample_macro')" in script and "scenario=Scenario(shift={'g': 5.0})" in script
    assert SolveModel("cons", "2020Q1", "2024Q4", shift="g 5").validate(s)
    report = command_dialog(ExportReport, s)
    assert report.editors["path"].spec.kind == "savefile" and report.editors["path"].browse.text() == "Browse…"


def test_the_start_page_offers_a_way_in(app):
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    assert window.start_page is not None and set(window.start_page.buttons) >= {"Load example data…", "Open a project…"}
    window.close()
