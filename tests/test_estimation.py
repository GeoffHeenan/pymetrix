import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import acorr_breusch_godfrey  # noqa: E402
from statsmodels.stats.stattools import jarque_bera  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from pymetrix import OLS, Cluster, Equation, Param, compare
from pymetrix.expr import ExpressionError


def _sm(ts, **fit_kw):
    X = sm.add_constant(ts.frame[["x", "z"]])
    return sm.OLS(ts["y"], X).fit(**fit_kw)


def test_ols_matches_statsmodels(ts):
    fit = Equation("y ~ x + z").fit(ts)
    ref = _sm(ts)
    assert np.allclose(fit.params.values, ref.params.to_numpy())
    assert np.allclose(fit.params.se, ref.bse.to_numpy())
    assert np.isclose(fit.stats["r2"], ref.rsquared)
    assert np.isclose(fit.stats["aic"], ref.aic)
    assert fit.params.names == ("const", "x", "z")


@pytest.mark.parametrize(
    "cov, kw",
    [("white", {"cov_type": "HC1"}), ("hac", {"cov_type": "HAC", "cov_kwds": {"maxlags": 4, "use_correction": True}})],
)
def test_robust_covariances(ts, cov, kw):
    fit = Equation("y ~ x + z").fit(ts, method=OLS(cov=cov, hac_lags=4))
    assert np.allclose(fit.params.se, _sm(ts, **kw).bse.to_numpy())


def test_cluster_matches_statsmodels(panel):
    fit = Equation("y ~ x").fit(panel, method=OLS(cov=Cluster("country")))
    groups = pd.factorize(panel.values_of("country"))[0]
    ref = sm.OLS(panel["y"], sm.add_constant(panel["x"])).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert np.allclose(fit.params.se, ref.bse.to_numpy())


def test_fixed_effects_equal_dummy_regression(panel):
    fit = Equation("y ~ x + fe(country)").fit(panel)
    dummies = pd.get_dummies(panel.values_of("country"), dtype=float)
    ref = sm.OLS(panel["y"], pd.concat([panel["x"], dummies], axis=1)).fit()
    assert np.isclose(fit.params["x"], ref.params["x"])
    assert np.isclose(fit.params.se[0], ref.bse["x"])
    assert "r2_within" in fit.stats


def test_categoricals_create_dummies(panel):
    fit = Equation("y ~ x + C(region)").fit(panel)
    assert fit.params.names == ("const", "x", "C(region)[B]")


def test_explicit_form_by_nls_matches_ols(ts):
    nls = Equation("y ~ a + b*x + g*z", params=["a", "b", "g"]).fit(ts)
    ols = Equation("y ~ x + z").fit(ts)
    assert np.allclose(nls.params.values, ols.params.values, atol=1e-7)
    assert np.allclose(nls.params.se, ols.params.se, rtol=1e-5)


def test_nonlinear_equation_with_bounds(ts):
    data = ts.with_series(q="exp(0.2 + 0.4*x)")
    fit = Equation("q ~ exp(a + b*x)", params=[Param("a"), Param("b", bounds=(0, 1))]).fit(data)
    assert np.allclose(fit.params.values, [0.2, 0.4], atol=1e-6)


def test_parameters_must_be_declared(ts):
    with pytest.raises(ExpressionError, match="unknown name"):
        Equation("y ~ a + b*x").fit(ts)
    with pytest.raises(ExpressionError, match="same name as series"):
        Equation("y ~ x + x*z", params=["x"]).fit(ts)
    with pytest.raises(ExpressionError, match="not used"):
        Equation("y ~ a*x", params=["a", "b"])


def test_span_uses_earlier_data_for_lags(ts):
    fit = Equation("y ~ y[-1] + x").fit(ts, span=("2000Q1", "2024Q4"))
    assert fit.provenance.nobs == 99 and fit.provenance.dropped == 1
    fit = Equation("y ~ y[-1] + x").fit(ts, span=("2000Q2", "2024Q4"))
    assert fit.provenance.dropped == 0


def test_missing_raise(ts):
    with pytest.raises(ValueError, match="missing"):
        Equation("y ~ y[-1]").fit(ts, missing="raise")


def test_wald_and_tests_match_statsmodels(ts):
    fit = Equation("y ~ x + z").fit(ts)
    ref = _sm(ts)
    w = fit.wald("x = 0")
    assert np.isclose(w.statistic, ref.tvalues["x"] ** 2)
    joint = fit.wald("x + z = 0.2, const = 1")
    assert joint.df == 2
    bg = fit.test("breusch_godfrey", lags=4)
    assert np.isclose(bg.statistic, acorr_breusch_godfrey(ref, nlags=4)[0])
    assert bg.reject()  # the errors are AR(1)
    jb = fit.test("jarque_bera")
    assert np.isclose(jb.statistic, jarque_bera(ref.resid)[0])
    assert "breusch_godfrey" in fit.tests()
    names = [row[0] for row in fit.diagnostics().rows]
    assert names == ["Breusch-Godfrey LM", "Breusch-Pagan (Koenker)", "Jarque-Bera", "RESET"]


def test_adf_on_data(ts):
    r = ts.test("adf", "yd", lags=2)
    ref = adfuller(ts["yd"].to_numpy(), maxlag=2, autolag=None)
    assert np.isclose(r.statistic, ref[0])
    assert not r.reject()


def test_compare_and_lr(ts):
    small = Equation("y ~ x").fit(ts)
    big = Equation("y ~ x + z").fit(ts)
    assert len(compare(small, big).rows) == 2
    lr = compare(small, big, test="lr")
    assert lr.df == 1 and lr.reject()


def test_summary_and_provenance(ts):
    fit = Equation("y ~ x + z").fit(ts, span=("2005Q1", "2020Q4"))
    assert fit.provenance.sample == ("2005Q1", "2020Q4")
    assert fit.provenance.method["cov"] == "classic"
    text = str(fit.summary())
    assert "const" in text and "R-squared" in text


def test_static_predict(ts):
    fit = Equation("y ~ x + z").fit(ts)
    assert np.allclose(fit.predict(ts), fit.fitted)
