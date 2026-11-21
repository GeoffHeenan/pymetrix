"""Expressions in model files are checked before they are evaluated (pymetrix.models.safe_eval): Dynare parameter
values, IRIS !if conditions and parameter values, the model editor's Dynare check and simple MATLAB data files. A model
file can come from anyone, so escapes from a restricted ``eval`` (``().__class__.__bases__[0].__subclasses__()``,
``__import__('os')``, attribute access, ``lambda``) are refused with a ModelError naming what was found, and nothing in
them runs.

Independent checks: the accepted expressions' values against the same arithmetic written by hand, and (in the change
that made this) every model-file expression the Dynare, IRIS, EViews and model-editor tests evaluate (2,571 of them)
logged before and after, with the same inputs giving the same results."""

import math

import numpy as np
import pytest

from pymetrix import ModelSource, read_iris, read_mod
from pymetrix.errors import PyMetrixError
from pymetrix.models import ModelError
from pymetrix.models.dynare import read_m_data
from pymetrix.models.safe_eval import check_expression, safe_eval

ESCAPES = {
    "().__class__.__bases__[0].__subclasses__()": "attribute access",
    "__import__('os')": "__import__",
    "alpha.real": "attribute access",
    "(lambda: 1)()": "lambda",
    "[c for c in ()]": "comprehension",
    "alpha[0]": "indexing",
    "max(*[1, 2])": "unpacking|a list",
    "max(alpha, key=abs)": "keyword argument",
    "'a' * 3": "str value",
    "__builtins__": "__builtins__",
    "exp.__globals__": "attribute access",
}


def _mod(value: str) -> str:
    return f"var x; varexo e; parameters alpha rho; alpha = 0.5; rho = {value}; model; x = rho*x(-1) + e; end;"


@pytest.mark.parametrize("payload", sorted(ESCAPES))
def test_dynare_parameter_escapes_are_refused(payload):
    # read_mod renames Python keywords (Dynare allows a parameter called lambda), so a lambda or a comprehension is
    # not even Python by the time it is checked
    keywords = any(w in payload for w in ("lambda", " for "))
    expected = "not an expression" if keywords else f"({ESCAPES[payload]}).*not allowed"
    with pytest.raises(ModelError, match=f"cannot evaluate rho = .*{expected}") as caught:
        read_mod(_mod(payload))
    assert isinstance(caught.value, PyMetrixError)


@pytest.mark.parametrize("payload", sorted(ESCAPES))
def test_the_checker_refuses_before_anything_runs(payload, monkeypatch):
    import pymetrix.models.safe_eval as module

    ran = []
    monkeypatch.setattr(module, "eval", lambda *args: ran.append(args), raising=False)  # shadows the built-in there
    with pytest.raises(ModelError, match=f"({ESCAPES[payload]}).*not allowed"):
        safe_eval(payload, {"alpha": 0.5}, {"exp": math.exp, "max": max, "abs": abs})
    assert ran == []
    safe_eval("alpha + 1", {"alpha": 0.5})
    assert len(ran) == 1  # the same path runs an accepted expression


def test_iris_condition_and_parameter_escapes_are_refused():
    body = """
    !transition_variables x
    !transition_shocks e
    !parameters rho = {rho}
    !transition_equations
    !if {cond}
        x = rho*x{{-1}} + e;
    !else
        x = e;
    !end
    """
    model = read_iris(body.format(rho="0.5", cond="version == 1 && ~fast"), params={"version": 1, "fast": 0})
    assert model.solve().rules().to_frame().set_index("").loc["x", "x[-1]"] == pytest.approx(0.5)
    # every word in a condition must be a given value; one that is (a parameter named real) passed that test, and
    # the attribute access is now refused as well
    with pytest.raises(ModelError, match="cannot evaluate the !if condition.*attribute access is not allowed"):
        read_iris(body.format(rho="0.5", cond="version.real"), params={"version": 1, "real": 1})
    with pytest.raises(ModelError, match="attribute access is not allowed"):
        read_iris(body.format(rho="().__class__", cond="1 == 1"))


def test_the_model_editor_check_runs_nothing_and_shows_no_value():
    checked = ModelSource(_mod("__import__('os').getpid()"), "dynare").check()
    values = {s.name: s.value for s in checked.symbols if s.role == "parameter"}
    assert values == {"alpha": 0.5, "rho": None}
    ok = ModelSource(_mod("2^-1 + alpha/4"), "dynare").check()
    assert {s.name: s.value for s in ok.symbols}["rho"] == pytest.approx(2 ** -1 + 0.5 / 4)


def test_matlab_data_files_allow_columns_only(tmp_path):
    path = tmp_path / "d.m"
    path.write_text("raw=[1 10\n2 20\n4 40];\ng = log(raw(:,2)./raw(:,1));\n", encoding="utf-8")
    assert np.allclose(read_m_data(path)["g"], [math.log(10)] * 3)
    for bad in ("raw.__class__", "raw[0]", "(lambda: raw)()", "__builtins__", "max(raw)"):  # quotes are refused already
        path.write_text(f"raw=[1 2];\nx = {bad};\n", encoding="utf-8")
        with pytest.raises(ModelError, match="not allowed"):
            read_m_data(path)


def test_what_model_files_use_still_evaluates():
    """Each form the readers accept, against the same arithmetic by hand."""
    v = {"a": 0.25, "b": 4.0, "del_": 2.0}
    funcs = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt, "max": max, "min": min, "abs": abs}
    assert safe_eval("-a + b**2/2 - +1", v) == -0.25 + 8.0 - 1.0
    assert safe_eval("exp(log(b)) * sqrt(b) + max(a, 1) - min(a, b) + abs(-a)", v, funcs) == pytest.approx(4 * 2 + 1 - 0.25 + 0.25)
    assert safe_eval("(a < b) + (b >= 4) + (a != a)", v) == 2
    assert safe_eval("a > 0 and not b < 1 or False", v, conditions=True) is True
    assert safe_eval("raw[:, 1] * 2", {"raw": np.array([[1.0, 2.0], [3.0, 4.0]])}, columns=True).tolist() == [4.0, 8.0]
    assert type(safe_eval("7 * 3", {})) is float and safe_eval("7 * 3", {}) == 21  # whole numbers as floats
    assert read_mod(_mod("1/(1 + alpha^2)")).params["rho"] == pytest.approx(1 / 1.25)
    with pytest.raises(ModelError, match="overflow|too large|range"):
        read_mod(_mod("9^9^9^9"))  # floats overflow at once, not a huge integer computed


def test_refusals_name_what_they_found():
    with pytest.raises(ModelError, match="name 'gamma' is not defined"):
        check_expression("gamma + 1", ["alpha"])
    with pytest.raises(ModelError, match="calling open"):
        check_expression("open(1)", ["alpha"])
    with pytest.raises(ModelError, match="and / or is not allowed"):
        check_expression("alpha and alpha", ["alpha"])
    with pytest.raises(ModelError, match="not allowed"):
        check_expression("alpha in alpha", ["alpha"])
