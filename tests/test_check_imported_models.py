"""Model > Check model… takes a model however it was made: one imported from an IRIS or Dynare file (Model > Import
IRIS file…, Import Dynare file…) is offered by the dialog's model picker, and checking it there gives what the API's
own check gives (the model's shocks and steady-state guesses included), with a script that replays."""

import os
import warnings
from pathlib import Path

import pytest

from pymetrix import Model, read_iris, read_mod
from pymetrix.commands import CheckModel, Session
from pymetrix.commands.builtin import ImportDynare, ImportIris
from pymetrix.forms import form_spec

from test_dynare import RBC
from test_iris import IRIS_NK


@pytest.fixture()
def imported(tmp_path):
    path = tmp_path / "nk.model"
    path.write_text(IRIS_NK, encoding="utf-8")
    (tmp_path / "rbc.mod").write_text(RBC, encoding="utf-8")
    s = Session()
    s.run(ImportIris(str(path), params="phi=1.5"))
    s.run(ImportDynare(str(tmp_path / "rbc.mod")))
    return s, path


def _replay(s, name):
    ns: dict = {}
    exec(s.script(), ns)
    return ns[name]


def test_the_picker_lists_imported_models(imported):
    s, _ = imported
    spec = {f.name: f for f in form_spec(CheckModel)}["equations"]
    assert {"iris", "dynare"} <= set(s.choices(spec, {}))  # before the fix: [] (only equations were offered)


def test_checking_an_imported_model_equals_its_own_check(imported):
    s, path = imported
    s.run(CheckModel("iris", name="iris_check"))
    s.run(CheckModel("dynare", name="dynare_check"))
    iris, dynare = read_iris(path, params={"phi": 1.5}), read_mod(path.parent / "rbc.mod")
    # the API: the template's own check, and the model it holds checked with its shocks and guesses
    assert s["iris_check"].rows == iris.check().rows
    assert any(check == "shocks" for check, _, _ in s["iris_check"].rows)  # before the fix the shocks were left out
    direct = Model([dynare]).check(shocks=dynare.exogenous, guess=dynare.guess)  # with the file's shocks and initval
    assert s["dynare_check"].rows == direct.rows and s["dynare_check"].ok
    script = s.script()
    assert "iris_check = Model([iris]).check(shocks=('ey', 'epi', 'ei'))" in script
    assert "dynare_check = Model([dynare]).check(shocks=('e',), guess={'k': 30.0, 'c': 2.0, 'y': 3.0})" in script
    assert _replay(s, "iris_check").rows == s["iris_check"].rows
    assert _replay(s, "dynare_check").rows == s["dynare_check"].rows


def test_shocks_given_in_the_dialog_win(imported):
    s, _ = imported
    s.run(CheckModel("iris", shocks="ey", name="chk"))
    assert "chk = Model([iris]).check(shocks=['ey'])" in s.script()


def test_the_dialog_offers_and_checks_an_imported_model(imported):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    s, path = imported
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = command_dialog(CheckModel, s)
    try:
        picker = dialog.editors["equations"].widget
        listed = {picker.item(i).text() for i in range(picker.count())}
        assert {"iris", "dynare"} <= listed
        dialog.editors["equations"].set("iris")
        QtWidgets.QApplication.processEvents()
        ok = dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert ok.isEnabled()
        ok.click()
        QtWidgets.QApplication.processEvents()
        assert dialog.command == CheckModel("iris", data="")  # no dataset in the session
        assert s["model_check"].rows == read_iris(path, params={"phi": 1.5}).check().rows
        assert _replay(s, "model_check").rows == s["model_check"].rows
    finally:
        dialog.close()


def _code_shown(s, initial):
    """What Show the code shows in the Check model dialog opened on ``initial``, before the step runs."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = command_dialog(CheckModel, s, initial=initial)
    try:
        QtWidgets.QApplication.processEvents()
        return dialog.code_text(), dialog.command if hasattr(dialog, "command") else None
    finally:
        dialog.close()


def test_the_code_shown_before_the_step_runs_is_the_code_it_records(imported):
    """Before the fix the dialog showed ``check(data)`` with no dataset in the session, ``check(, span=...)`` (not
    Python) with a span, and left out the model's own shocks, which the recorded step has."""
    import ast

    s, _ = imported
    for initial in (CheckModel("dynare"), CheckModel("iris", start="2000Q1", end="2001Q4")):
        shown, _ = _code_shown(s, initial)
        assert "data" not in shown and ", span" not in shown.replace("check(span", "") and "(," not in shown
        ast.parse(shown)
        s.run(CheckModel(initial.equations, data="", start=initial.start, end=initial.end))
        assert shown == s.history[-1].code  # what the dialog showed is what was recorded
    assert s.history[-1].code == "model_check = Model([iris]).check(span=('2000Q1', '2001Q4'), shocks=('ey', 'epi', 'ei'))"


def test_with_a_dataset_the_code_shown_names_it(imported):
    from pymetrix.commands import LoadExample

    s, _ = imported
    s.run(LoadExample("sample_macro", name="macro"))
    shown, _ = _code_shown(s, CheckModel("dynare", data="macro"))
    s.run(CheckModel("dynare", data="macro"))
    assert shown == s.history[-1].code and shown.startswith("model_check = Model([dynare]).check(macro, shocks=('e',)")


def test_to_code_without_a_session_leaves_out_a_dataset_not_chosen():
    import ast

    code = CheckModel("cons, inv", data="", start="2015Q1", end="2024Q4").to_code()
    assert code == "model_check = Model([cons, inv]).check(span=('2015Q1', '2024Q4'))"  # before: check(, span=...)
    ast.parse(code)
    assert CheckModel("cons, inv").to_code() == "model_check = Model([cons, inv]).check(data)"  # the default: data


def test_a_notebook_form_shows_the_code_it_recorded(imported):
    pytest.importorskip("ipywidgets")
    import html

    from pymetrix.ui.jupyter import form

    s, _ = imported
    f = form(CheckModel, s)
    f.inputs["equations"].value = ("dynare",)
    assert f.inputs["data"].options == ()  # no dataset in the session
    run = next(c for c in f.children[-4].children if getattr(c, "description", "") == "Run")
    run.click()
    shown = html.unescape(f.children[-3].value)
    assert shown == f"<pre>{s.history[-1].code}</pre>" and "data" not in shown  # before: check(data)
