"""The Wald test reads coefficient names as restricted least squares does: term labels with parentheses (``dlog(Y)``,
``log(P)``), lags (``x[-1]``), names such as ``L(x, 1)`` (with commas and spaces), ``const`` and names in backticks.

Checked against: the same hypothesis written with plain names on the same regression with the regressors computed by
hand and named a, b, c (statistic, F form and p-values equal to 1e-12); and the textbook Wald statistic
(R b - r)' (R V R')^-1 (R b - r) computed here with numpy from the fit's coefficients and covariance.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, Dataset, Equation, System
from pymetrix.estimation.restrictions import parse_restrictions
from pymetrix.expr import ExpressionError
from pymetrix.stats import Wald

FORMULA = "q ~ dlog(Y) + x[-1] + log(P)"


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(5)
    n = 80
    frame = pd.DataFrame({"Y": np.exp(np.cumsum(rng.normal(0.01, 0.02, n))), "x": rng.normal(size=n),
                          "P": np.exp(rng.normal(size=n))}, index=pd.period_range("2000Q1", periods=n, freq="Q"))
    a, b, c = np.log(frame["Y"]).diff(), frame["x"].shift(1), np.log(frame["P"])
    frame["q"] = 0.5 + 0.7 * a.fillna(0) + 0.3 * b.fillna(0) - 0.2 * c + rng.normal(scale=0.5, size=n)
    data = Dataset.from_frame(frame)
    named = Equation(FORMULA).fit(data)
    plain = Equation("q ~ a + b + c").fit(Dataset.from_frame(pd.DataFrame({"q": frame["q"], "a": a, "b": b, "c": c})))
    assert list(named.params.names) == ["const", "dlog(Y)", "x[-1]", "log(P)"]
    np.testing.assert_allclose(named.params.values, plain.params.values, rtol=1e-10)  # the same regression
    return named, plain, data


def _same(a, b):
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12, abs=1e-12)
    assert a.p_value == pytest.approx(b.p_value, rel=1e-12, abs=1e-15) and a.df == b.df
    assert a.details["F"] == pytest.approx(b.details["F"], rel=1e-12)


HYPOTHESES = [
    ("dlog(Y) = 0", "a = 0"),  # was "unknown coefficient(s) ['Y']"
    ("dlog(Y) + x[-1] = 1", "a + b = 1"),
    ("x[-1] = 2 * log(P)", "b = 2 * c"),
    ("`dlog(Y)` = `x[-1]`, `log(P)` = 0", "a = b, c = 0"),
    ("const = 0.5, log(P) + dlog(Y) = 0.5", "const = 0.5, c + a = 0.5"),
    ("dlog(Y) / x[-1] = 2", "a / b = 2"),  # nonlinear, by the delta method
]


@pytest.mark.parametrize("named,plain", HYPOTHESES)
def test_term_labels_equal_plain_names(fits, named, plain):
    fit, ref, _ = fits
    _same(fit.test("wald", named), ref.test("wald", plain))


def test_against_the_textbook_formula(fits):
    fit, _, _ = fits
    b, V = np.asarray(fit.params.values), np.asarray(fit.params.cov)
    R = np.array([[0, 1, 1, 0], [0, 0, 0, 1.0]])  # const, dlog(Y), x[-1], log(P)
    r = np.array([1.0, 0.0])
    d = R @ b - r
    assert fit.test("wald", "dlog(Y) + x[-1] = 1, `log(P)` = 0").statistic == pytest.approx(
        float(d @ np.linalg.solve(R @ V @ R.T, d)), rel=1e-12)


def test_restricted_ls_accepts_the_same_hypothesis(fits):
    _, _, data = fits
    restricted = Equation(FORMULA).fit(data, method=OLS(restrictions="dlog(Y) + x[-1] = 1, `log(P)` = 0"))
    p = restricted.params.as_dict()
    assert p["dlog(Y)"] + p["x[-1]"] == pytest.approx(1, abs=1e-10) and p["log(P)"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("named,plain", [("L(x, 1) = 0", "b = 0"), ("L(x,1) + log(P) = 1", "b + c = 1"),
                                         ("`L(x, 1)` = dlog(Y)", "b = a"), ("`L(x,1)` = 0", "b = 0")])
def test_names_with_commas_and_spaces(fits, named, plain):
    """Names such as L(x, 1) (another package's lag label, or a renamed coefficient): a comma inside the name does not
    split the hypothesis, and its spaces are optional. Run on the fit's own coefficients and covariance renamed."""
    _, ref, _ = fits
    p = ref.params
    labels = [{"const": "const", "a": "dlog(Y)", "b": "L(x, 1)", "c": "log(P)"}[n] for n in p.names]
    params = SimpleNamespace(names=labels, cov=p.cov, as_dict=lambda: dict(zip(labels, p.values)))
    shim = SimpleNamespace(params=params, stats=ref.stats)
    _same(Wald(named).run(shim), ref.test("wald", plain))


def test_shared_parser_and_errors(fits):
    fit, _, _ = fits
    names = list(fit.params.names)
    for text in ("dlog(Y) = 0", "`dlog(Y)` = 0", "` dlog(Y) ` = 0"):
        [(_, expr)] = parse_restrictions(text, names)
        assert {str(v) for v in expr.params()} == {"dlog(Y)"}
    with pytest.raises(ExpressionError, match="unknown coefficient"):
        fit.test("wald", "dlog(Z) = 0")
    with pytest.raises(ExpressionError, match="unknown coefficient `nothing`"):
        fit.test("wald", "`nothing` = 0")
    with pytest.raises(ExpressionError, match="needs '='"):
        fit.test("wald", "dlog(Y)")


def test_system_wald_with_term_labels():
    """A system's names (equation.label) take labels with parentheses too; the result equals the same restriction on
    the system's coefficients and covariance, by the textbook formula."""
    data = Dataset.example("sample_macro")
    fit = System([Equation("dlog(c) ~ dlog(y)"), Equation("dlog(i) ~ dlog(y)")]).fit(data)
    names = list(fit.params.names)
    a, b = [n for n in names if n.endswith(".dlog(y)")]
    res = fit.wald(f"{a} = {b}")
    R = np.zeros(len(names))
    R[names.index(a)], R[names.index(b)] = 1, -1
    d = R @ np.asarray(fit.params.values)
    assert res.df == 1 and res.statistic == pytest.approx(d ** 2 / (R @ np.asarray(fit.params.cov) @ R), rel=1e-10)
