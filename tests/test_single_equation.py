"""IV, residual diagnostics, ML and forecasts, checked against statsmodels."""

import warnings

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.sandbox.regression.gmm import IV2SLS  # noqa: E402
from statsmodels.stats.diagnostic import het_arch, het_breuschpagan, het_white, linear_reset  # noqa: E402

from pymetrix import IV, ML, OLS, Dataset, Equation, Logit, Poisson, Probit, forecast_accuracy
from pymetrix.commands import FitEquation, Session


@pytest.fixture
def ivdata() -> Dataset:
    """y = 1 + 2x + 0.5w + u, with x correlated with u; z1, z2 valid instruments, bad invalid."""
    rng = np.random.default_rng(3)
    n = 2000
    z1, z2, w = rng.normal(size=(3, n))
    u = rng.normal(size=n)
    v = 0.8 * u + rng.normal(scale=0.6, size=n)
    x = 0.6 * z1 + 0.4 * z2 + 0.3 * w + v
    y = 1 + 2 * x + 0.5 * w + u
    bad = z1 + 0.5 * u
    df = pd.DataFrame({"id": range(n), "y": y, "x": x, "w": w, "z1": z1, "z2": z2, "bad": bad})
    return Dataset.from_frame(df, entity="id")


# IV


def test_iv_matches_statsmodels(ivdata):
    fit = Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x"], instruments=["z1", "z2"]))
    f = ivdata.frame
    ref = IV2SLS(f["y"], sm.add_constant(f[["x", "w"]]), sm.add_constant(f[["w", "z1", "z2"]])).fit()
    assert np.allclose(fit.params.values, ref.params.to_numpy())
    assert np.allclose(fit.params.se, ref.bse.to_numpy())
    ols = Equation("y ~ x + w").fit(ivdata)
    assert abs(fit.params["x"] - 2) < abs(ols.params["x"] - 2)  # OLS is biased upwards


def test_nonlinear_iv_matches_linear(ivdata):
    linear = Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x"], instruments=["z1", "z2"]))
    explicit = Equation("y ~ a + b*x + c*w", params=["a", "b", "c"]).fit(ivdata, method=IV(endog=["x"], instruments=["z1", "z2"]))
    assert np.allclose(explicit.params.values, linear.params.values, atol=1e-8)
    assert np.allclose(explicit.params.se, linear.params.se, rtol=1e-6)


def test_iv_diagnostics(ivdata):
    good = Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x"], instruments=["z1", "z2"]))
    f = ivdata.frame
    restricted = sm.OLS(f["x"], sm.add_constant(f[["w"]])).fit()
    unrestricted = sm.OLS(f["x"], sm.add_constant(f[["w", "z1", "z2"]])).fit()
    F = ((restricted.ssr - unrestricted.ssr) / 2) / (unrestricted.ssr / unrestricted.df_resid)
    assert np.isclose(good.test("first_stage").statistic, F)
    assert not good.test("sargan").reject()
    assert good.test("endogeneity").reject()
    invalid = Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x"], instruments=["z2", "bad"]))
    assert invalid.test("sargan").reject()
    exact = Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x"], instruments=["z1"]))
    with pytest.raises(ValueError, match="exactly identified"):
        exact.test("sargan")
    assert len(good.diagnostics().rows) == 4
    assert "First-stage F (x)" in str(good.summary())


def test_iv_errors(ivdata):
    with pytest.raises(ValueError, match="must name regressors"):
        Equation("y ~ x + w").fit(ivdata, method=IV(endog=["q"], instruments=["z1"]))
    with pytest.raises(ValueError, match="underidentified"):
        Equation("y ~ x + w").fit(ivdata, method=IV(endog=["x", "w"], instruments=["z1"]))
    with pytest.raises(ValueError, match="applies to IV"):
        Equation("y ~ x + w").fit(ivdata).test("sargan")


# least-squares diagnostics


def test_heteroskedasticity_and_specification_tests(ts):
    fit = Equation("y ~ x + z").fit(ts)
    X = sm.add_constant(ts.frame[["x", "z"]])
    ref = sm.OLS(ts["y"], X).fit()
    assert np.isclose(fit.test("white").statistic, het_white(ref.resid, X)[0])
    assert np.isclose(fit.test("breusch_pagan").statistic, het_breuschpagan(ref.resid, X)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arch_ref = het_arch(ref.resid, nlags=4)[0]
        reset_ref = linear_reset(sm.OLS(ts["y"].to_numpy(), X.to_numpy()).fit(), power=3, use_f=True)  # arrays: statsmodels 0.14 with pandas 2.2
    assert np.isclose(fit.test("arch", lags=4).statistic, arch_ref)
    assert np.isclose(fit.test("reset").statistic, reset_ref.fvalue)


def test_chow(ts):
    fit = Equation("y ~ x + z").fit(ts)
    r = fit.test("chow", at="2012Q1")
    f = ts.frame
    before = f.index < pd.Period("2012Q1", "Q")

    def ssr(rows):
        return sm.OLS(f["y"][rows], sm.add_constant(f[["x", "z"]][rows])).fit().ssr

    F = ((ssr(slice(None)) - ssr(before) - ssr(~before)) / 3) / ((ssr(before) + ssr(~before)) / (100 - 6))
    assert np.isclose(r.statistic, F)
    assert r.df == (3, 94)


# ML


@pytest.fixture
def mldata() -> Dataset:
    rng = np.random.default_rng(5)
    n = 1500
    x1, x2 = rng.normal(size=(2, n))
    eta = -0.3 + 0.8 * x1 - 0.5 * x2
    df = pd.DataFrame({
        "id": range(n), "x1": x1, "x2": x2,
        "b": (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
        "count": rng.poisson(np.exp(0.2 + 0.3 * x1 - 0.2 * x2)).astype(float),
        "region": np.where(rng.uniform(size=n) < 0.5, "A", "B"),
    })
    return Dataset.from_frame(df, entity="id")


@pytest.mark.parametrize("family, sm_model, dep", [(Logit(), sm.Logit, "b"), (Probit(), sm.Probit, "b"), (Poisson(), sm.Poisson, "count")])
def test_ml_matches_statsmodels(mldata, family, sm_model, dep):
    fit = Equation(f"{dep} ~ x1 + x2", family=family).fit(mldata, method=ML())
    X = sm.add_constant(mldata.frame[["x1", "x2"]])
    ref = sm_model(mldata[dep], X).fit(disp=0)
    assert np.allclose(fit.params.values, ref.params.to_numpy(), atol=1e-7)
    assert np.allclose(fit.params.se, ref.bse.to_numpy(), rtol=1e-6)
    assert np.isclose(fit.stats["loglike"], ref.llf)
    assert np.isclose(fit.stats["ll_null"], ref.llnull)
    assert np.isclose(fit.stats["pseudo_r2"], ref.prsquared)
    robust = Equation(f"{dep} ~ x1 + x2", family=family).fit(mldata, method=ML(cov="white"))
    assert np.allclose(robust.params.se, sm_model(mldata[dep], X).fit(disp=0, cov_type="HC0").bse.to_numpy(), rtol=1e-6)
    assert np.allclose(fit.predict(mldata), ref.predict(X))
    assert "z" in fit.summary().columns


def test_ml_checks(mldata):
    with pytest.raises(ValueError, match="0/1"):
        Equation("count ~ x1", family=Logit()).fit(mldata, method=ML())
    with pytest.raises(ValueError, match="method=ML"):
        Equation("b ~ x1", family=Logit()).fit(mldata)
    with pytest.raises(ValueError, match="use OLS"):
        Equation("b ~ x1").fit(mldata, method=ML())


# forecasts


def test_static_forecast_matches_statsmodels(ts):
    fit = Equation("y ~ x + z").fit(ts, span=("2000Q1", "2019Q4"))
    fc = fit.forecast(ts, span=("2020Q1", "2024Q4"))
    insample = ts.frame.index <= pd.Period("2019Q4", "Q")
    X = sm.add_constant(ts.frame[["x", "z"]])
    ref = sm.OLS(ts["y"][insample], X[insample]).fit().get_prediction(X[~insample]).summary_frame(alpha=0.05)
    assert np.allclose(fc["forecast"], ref["mean"])
    assert np.allclose(fc["lower"], ref["obs_ci_lower"])
    assert np.allclose(fc["actual"], ts["y"][~insample])
    acc = forecast_accuracy(fc["actual"], fc["forecast"])
    rmse = np.sqrt(np.mean((fc["actual"] - fc["forecast"]) ** 2))
    assert np.isclose(acc.rows[0][1], rmse)


def test_dynamic_forecast():
    data = Dataset.read("examples/sample_macro.parquet")
    inv = Equation("i ~ y[-1]").fit(data)
    fc = inv.forecast(data, span=("2023Q1", "2024Q4"), dynamic=True)
    assert len(fc) == 8 and fc["se"].isna().all()


# the Estimate dialog's command


def test_fit_command_methods(ivdata, mldata):
    session = Session()
    session.objects["data"] = ivdata
    iv = session.run(FitEquation("y ~ x + w", method="IV", endog="x", instruments="z1, z2", name="iv"))
    session.objects["d2"] = mldata
    session.run(FitEquation("b ~ x1 + x2", data="d2", method="ML", family="probit", name="pr"))
    script = session.script()
    assert "method=IV(endog=['x'], instruments=['z1', 'z2']))" in script
    assert "family=Probit()" in script
    ns = {"data": ivdata, "d2": mldata}
    exec(script, ns)
    assert np.allclose(ns["iv"].params.values, iv.params.values)
    errors = FitEquation("b ~ x1", method="ML").validate(session)
    assert any("family" in e for e in errors)
