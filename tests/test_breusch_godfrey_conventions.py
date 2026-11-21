"""Breusch-Godfrey without a constant: pymetrix's convention, the auxiliary-constant option, and the dialog."""

import numpy as np
import pytest

from pymetrix import Dataset, Equation, ols
from pymetrix.commands import RunTest, Session
from pymetrix.forms import form_spec
from pymetrix.stats.equation_tests import BreuschGodfrey


def _hand(fit, lags, constant):
    """The LM statistic by hand: residuals on the regressors (and a constant if asked) and their own lags, the lags
    before the first observation set to zero; n times R-squared about the residuals' mean."""
    u = fit.resid.to_numpy(float)
    X = fit.design.to_numpy(float)
    n = len(u)
    lagged = np.zeros((n, lags))
    for t in range(n):
        for j in range(1, lags + 1):
            if t - j >= 0:
                lagged[t, j - 1] = u[t - j]
    Z = np.hstack([np.ones((n, 1)), X, lagged] if constant else [X, lagged])
    b = np.linalg.solve(Z.T @ Z, Z.T @ u)
    e = u - Z @ b
    mean = sum(u) / n
    return n * (1 - float(e @ e) / float(sum((v - mean) ** 2 for v in u)))


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


@pytest.mark.parametrize("text", ["c ~ y - 1", "c ~ y + c[-1] - 1"])
@pytest.mark.parametrize("lags", [1, 4])
def test_both_conventions_match_hand_computations(data, text, lags):
    fit = Equation(text).fit(data)
    default = fit.test("breusch_godfrey", lags=lags)
    other = fit.test("breusch_godfrey", lags=lags, auxiliary_constant=True)
    assert default.statistic == pytest.approx(_hand(fit, lags, constant=False), rel=1e-10)
    assert other.statistic == pytest.approx(_hand(fit, lags, constant=True), rel=1e-10)
    assert default.df == other.df == lags
    assert abs(default.statistic - other.statistic) > 1  # without a constant, the conventions differ


def test_the_option_matches_statsmodels_without_a_constant(data):
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey

    for text in ("c ~ y - 1", "c ~ y + c[-1] - 1", "c ~ y"):
        fit = Equation(text).fit(data)
        X = fit.design.to_numpy(float)
        res = sm.OLS(X @ fit.params.values + fit.resid.to_numpy(float), X).fit()
        for lags in (1, 2, 4):
            lm, p_value, _, _ = acorr_breusch_godfrey(res, nlags=lags)
            ours = fit.test("breusch_godfrey", lags=lags, auxiliary_constant=True)
            assert ours.statistic == pytest.approx(lm, rel=1e-9)
            assert ours.p_value == pytest.approx(p_value, rel=1e-7, abs=1e-300)


def test_existing_results_are_unchanged(data):
    """The default is the convention pymetrix always used (values recorded before the option existed); with a constant
    in the equation the option changes nothing."""
    with_constant = Equation("c ~ y").fit(data)
    assert with_constant.test("breusch_godfrey", lags=4).statistic == pytest.approx(10.41778896289749, rel=1e-10)
    assert with_constant.test("breusch_godfrey", lags=4, auxiliary_constant=True).statistic == pytest.approx(10.41778896289749, rel=1e-10)
    assert Equation("c ~ y - 1").fit(data).test("breusch_godfrey", lags=1).statistic == pytest.approx(86.44168548728283, rel=1e-10)
    assert BreuschGodfrey().auxiliary_constant is False


def test_the_option_works_on_a_direct_fit(data):
    frame = data.frame[["c", "y"]].dropna()
    fit = ols(frame["c"], frame[["y"]], constant=False)
    formula = Equation("c ~ y - 1").fit(data)
    assert fit.test("breusch_godfrey", lags=2, auxiliary_constant=True).statistic == pytest.approx(
        formula.test("breusch_godfrey", lags=2, auxiliary_constant=True).statistic, rel=1e-12)


def test_run_test_dialog_sets_the_option_and_replays(data):
    specs = {s.name: s for s in form_spec(BreuschGodfrey)}
    assert specs["auxiliary_constant"].label == "Add a constant to the auxiliary regression"
    assert specs["auxiliary_constant"].kind == "checkbox"
    session = Session()
    session.objects["data"] = data
    session.run_code("fit = Equation('c ~ y - 1').fit(data)")
    result = session.run(RunTest(source="fit", test="breusch_godfrey", options="lags=2, auxiliary_constant=True"))
    api = session["fit"].test("breusch_godfrey", lags=2, auxiliary_constant=True)
    assert result.statistic == api.statistic and result.p_value == api.p_value
    script = session.script()
    assert "test = fit.test('breusch_godfrey', lags=2, auxiliary_constant=True)" in script
    namespace = {"data": data}
    exec(script, namespace)
    assert namespace["test"].statistic == pytest.approx(api.statistic, rel=1e-12)
    assert result.provenance.method["settings"]["auxiliary_constant"] is True
