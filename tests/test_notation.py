"""Algebraic notation: equations, coefficients and parameters as an economist writes them."""

import os

import numpy as np
import pytest

from pymetrix import Dataset, Equation, OLS
from pymetrix.notation import algebraic, notation, set_notation, symbol


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


@pytest.mark.parametrize("text, params, unicode, latex", [
    ("c ~ y + c[-1]", (), "cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ", r"c_t = \beta_0 + \beta_1 y_t + \beta_2 c_{t-1} + \varepsilon_t"),
    ("y = c + i + g", (), "yₜ = cₜ + iₜ + gₜ", "y_t = c_t + i_t + g_t"),
    ("pi ~ beta*pi[+1] + kappa*y", ("beta", "kappa"), "πₜ = β πₜ₊₁ + κ yₜ + εₜ", r"\pi_t = \beta \pi_{t+1} + \kappa y_t + \varepsilon_t"),
    ("dlog(c) ~ b0 + b1*dlog(y)", ("b0", "b1"), "Δlog cₜ = b₀ + b₁ Δlog yₜ + εₜ", r"\Delta\log c_t = b_0 + b_1 \Delta\log y_t + \varepsilon_t"),
    ("i ~ rho*i[-1] + (1 - rho)*phi_pi*pi", ("rho", "phi_pi"), "iₜ = ρ iₜ₋₁ + (1 − ρ) φ_π πₜ + εₜ",
     r"i_t = \rho i_{t-1} + \left(1 - \rho\right) \phi_{\pi} \pi_t + \varepsilon_t"),
    ("y ~ a*k*2", ("a",), "yₜ = a kₜ·2 + εₜ", r"y_t = a k_t \cdot 2 + \varepsilon_t"),
    ("c ~ α + β*y", ("α", "β"), "cₜ = α + β yₜ + εₜ", r"c_t = \alpha + \beta y_t + \varepsilon_t"),
    ("x ~ a*z^2", ("a",), "xₜ = a zₜ² + εₜ", r"x_t = a z_t^{2} + \varepsilon_t"),
])
def test_equations_in_algebraic_notation(text, params, unicode, latex):
    eq = Equation(text, params=list(params))
    assert eq.algebraic() == unicode and eq.algebraic("latex") == latex


def test_symbols():
    assert symbol("beta1", parameter=True) == "β₁" and symbol("b10", parameter=True) == "b₁₀"
    assert symbol("x2") == "x2"  # a variable's digits are part of its name
    assert symbol("phi_y", "latex") == r"\phi_y" and symbol("yd", "latex") == r"\mathit{yd}"


def test_greek_names_can_be_typed_and_estimate_the_same(data):
    greek = Equation("c ~ α + β*y", params=["α", "β"]).fit(data)
    plain = Equation("c ~ y").fit(data)
    assert np.allclose(greek.params.values, plain.params.values) and list(greek.params.names) == ["α", "β"]


def test_estimates_and_negative_signs(data):
    fit = Equation("c ~ y + c[-1]").fit(data, method=OLS(cov="hac"))
    assert fit.algebraic(estimates=True) == "cₜ = 12.49 + 0.7561 yₜ − 0.009607 cₜ₋₁"
    negative = Equation("c ~ y").with_params(const=1.0, y=-0.5)
    assert negative.algebraic(estimates=True) == "cₜ = 1 − 0.5 yₜ"


def test_the_option_changes_display_not_meaning(data):
    from pymetrix.commands import FitEquation, Session

    fit = Equation("c ~ y + c[-1]").fit(data)
    code = fit.summary()
    with notation("algebraic"):
        shown = fit.summary()
        assert Equation("c ~ y")._repr_latex_() == r"$$c_t = \beta_0 + \beta_1 y_t + \varepsilon_t$$"
        s = Session()
        s.objects["data"] = data
        s.run(FitEquation("c ~ y + c[-1]", name="cons"))
        assert "cons = Equation('c ~ y + c[-1]').fit(data" in s.script()  # scripts keep the code form
    assert shown.title == "cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ" and shown.index[2] == "β₂ (c[-1])"
    assert np.allclose(shown.to_frame()["coef"], code.to_frame()["coef"]) and code.title == "c ~ y + c[-1]"
    assert Equation("c ~ y")._repr_latex_() is None  # code notation: no typesetting
    with pytest.raises(ValueError):
        set_notation("greek")


def test_the_equation_tab(data):
    from pymetrix.views import view_sections

    fit = Equation("c ~ y").fit(data)
    tabs = dict(view_sections(fit))
    frame = tabs["Equation"].to_frame()
    rows = dict(zip(frame.iloc[:, 0], frame["equation"]))
    assert rows["code"] == "c ~ y" and rows["algebraic"] == "cₜ = β₀ + β₁ yₜ + εₜ" and rows["estimated"].startswith("cₜ = 12.35")
    assert [t for t, _ in view_sections(Equation("y = c + i + g"))] == ["Equation"]


def test_the_desktop_toggle():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pymetrix.commands import Session
    from pymetrix.notation import get_notation
    from pymetrix.ui.qt import main_window

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = main_window(Session())
    before = window.algebraic_action.isChecked()
    window.algebraic_action.setChecked(True)
    assert get_notation() == "algebraic"
    window.algebraic_action.setChecked(before)  # leave the saved setting as found
    window.close()


def test_symbols_from_metadata():
    from pymetrix.metadata import Param

    eq = Equation("i ~ rho*i[-1] + phi*pi + gap", params=[Param("rho"), Param("phi", symbol="φ_π")])
    assert eq.algebraic() == "iₜ = ρ iₜ₋₁ + φ_π πₜ + gapₜ + εₜ"
    assert algebraic(eq, "latex", symbols={"gap": "ŷ"}) == r"i_t = \rho i_{t-1} + \phi_{\pi} \pi_t + ŷ_t + \varepsilon_t"
