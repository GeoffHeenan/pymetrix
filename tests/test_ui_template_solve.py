"""After Model → Template model…, Rational expectations takes the template as it is (picked for you, recorded as
``re = nk.solve()``, replayed by the script), and Solve, which cannot, says where a template is solved."""

import os
import warnings

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import LoadTemplate, Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _ok(dialog):
    return dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)


def _dialog(cls, session):
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return command_dialog(cls, session)


def test_rational_expectations_takes_the_template_just_made(app):
    from pymetrix.commands.builtin import SolveRationalExpectations

    session = Session()
    session.run(LoadTemplate("new_keynesian", name="nk"))
    dialog = _dialog(SolveRationalExpectations, session)
    assert dialog.editors["equations"].get() == "nk"  # the newest model, picked
    assert _ok(dialog).isEnabled() and not dialog.pickers["equations"].widget.isVisibleTo(dialog)
    _ok(dialog).click()
    assert type(session["re"]).__name__ == "LinearSolution" and session.history[-1].code.strip() == "re = nk.solve()"
    namespace: dict = {}
    exec(session.script(), namespace)  # the script replays
    assert np.allclose(namespace["re"].irf(4).values, session["re"].irf(4).values)
    dialog.close()


def test_solve_points_a_template_to_rational_expectations(app):
    from pymetrix.commands.builtin import SolveModel

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    session.run(LoadTemplate("new_keynesian", name="nk"))
    dialog = _dialog(SolveModel, session)
    picker = dialog.pickers["equations"]
    assert picker.widget.isVisibleTo(dialog)
    assert picker.label.text().endswith("nk is a template model: solve it with Model → Solution → Rational expectations…")  # no "…."
    assert picker.target.__name__ == "SolveRationalExpectations"
    assert not _ok(dialog).isEnabled() and "Equations" in dialog.needed_note.text()
    dialog.close()


def test_a_template_with_settings_meant_for_equations_is_refused():
    from pymetrix.commands.builtin import SolveRationalExpectations

    session = Session()
    session.run(LoadTemplate("new_keynesian", name="nk"))
    assert SolveRationalExpectations("nk", shocks="ey").validate(session) == ["nk has its own shocks: leave Shocks empty"]
    assert SolveRationalExpectations("nk").validate(session) == []
    assert SolveRationalExpectations("").validate(session) == ["give at least one shock"]  # equations need shocks, as before


def test_only_pickers_of_equations_point_to_rational_expectations(app):
    from pymetrix.commands.builtin import Forecast

    session = Session()
    session.run(LoadTemplate("new_keynesian", name="nk"))
    dialog = _dialog(Forecast, session)  # no data and no fitted model: nothing about templates there
    assert "template" not in dialog.pickers["fit"].label.text() and "template" not in dialog.pickers["data"].label.text()
    dialog.close()
