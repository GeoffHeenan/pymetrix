"""Estimating linearised models: maximum likelihood with the Kalman filter, or the posterior mode with priors."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.estimation.priors import Gamma, Normal
from pymetrix.metadata import Role
from pymetrix.models import FittedModel, ModelError


def _quarters(n: int, start: str = "1950Q1") -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=_quarters(300)))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model, data, x


@pytest.fixture(scope="module")
def nk_data():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(400, nk.shock_sd, seed=11)
    truth.index = _quarters(400, "1925Q1")
    return nk.measurement.apply(Dataset.from_frame(truth))


def test_an_ar1_matches_statsmodels(ar1):
    sm = pytest.importorskip("statsmodels.tsa.statespace.sarimax")
    model, data, x = ar1
    fit = model.estimate(data, ["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.SARIMAX(x, order=(1, 0, 0), trend="n").fit(disp=False)
    assert isinstance(fit, FittedModel) and fit.params.names == ("rho", "sd_e")
    np.testing.assert_allclose(fit.params.values, [ref.params[0], np.sqrt(ref.params[1])], rtol=1e-4)
    np.testing.assert_allclose(fit.stats["loglike"], ref.llf, rtol=1e-7)
    np.testing.assert_allclose(fit.params.se[0], ref.bse[0], rtol=0.05)  # numerical Hessians, different parameterisation of the variance
    assert fit.parameters["rho"].status == "estimated" and fit.shock_sd["e"] == pytest.approx(fit.params.values[1])
    assert fit.model.parameters["rho"].value == pytest.approx(fit.params.values[0])  # the model, rebuilt at the estimates
    assert fit.stats["aic"] == pytest.approx(-2 * fit.stats["loglike"] + 4)


def test_new_keynesian_recovers_its_parameters(nk_data):
    fit = templates.new_keynesian(rho=0.6).estimate(nk_data, estimate=["rho", "sd_ey", "sd_epi", "sd_ei"])
    est = dict(zip(fit.params.names, fit.params.values))
    se = dict(zip(fit.params.names, fit.params.se))
    for name, truth in {"rho": 0.8, "sd_ey": 0.5, "sd_epi": 0.3, "sd_ei": 0.2}.items():
        assert abs(est[name] - truth) < 3 * se[name], name  # within three standard errors of the values that made the data
    assert fit.provenance.converged and fit.method == "maximum likelihood"
    assert fit.parameters["kappa"].status == "calibrated" and fit.parameters["kappa"].value == 0.1  # not estimated, kept
    assert set(fit.smoothed.names) >= {"y", "pi", "i"} and len(fit.smoothed.frame) == 400


def test_priors_give_the_posterior_mode(nk_data):
    start = templates.new_keynesian(rho=0.6)
    ml = start.estimate(nk_data, estimate=["kappa", "rho"])
    post = start.estimate(nk_data, estimate=["kappa", "rho"], priors={"kappa": Gamma(0.3, 0.01), "rho": Normal(0.8, 0.1)})
    kappa = dict(zip(post.params.names, post.params.values))["kappa"]
    assert abs(kappa - 0.3) < 0.05 and abs(kappa - 0.3) < abs(ml.params.values[0] - 0.3)  # a tight prior pulls the estimate to it
    assert post.method == "posterior mode" and "logpost" in post.stats and "logpost" not in ml.stats
    assert post.stats["loglike"] <= ml.stats["loglike"] + 1e-6  # the likelihood is highest at its own maximum
    assert "Gamma" in str(post.summary()) and "Log posterior: " in str(post.summary())


def test_provenance_and_views(ar1):
    from pymetrix.views import titles

    model, data, _ = ar1
    fit = model.estimate(data, ["x"], span=("1960Q1", None))
    prov = fit.provenance
    assert prov.sample == ("1960Q1", "2024Q4") and prov.nobs == 260 and prov.converged is True
    assert prov.method["estimated"] == ["rho", "sd_e"] and prov.method["filter"] == "Kalman" and len(prov.data_fingerprint) == 16
    assert titles(fit) == ["Summary", "Parameters", "Moments", "Variance decomposition", "Shock sizes", "Smoothed variables",
                           "Measurement", "Decision rules", "Provenance"]
    assert fit.moments().std["x"] == pytest.approx(fit.shock_sd["e"] / np.sqrt(1 - fit.params.values[0] ** 2))  # AR(1) at the estimates
    assert "260 observations" in str(fit.summary()) and "estimated by maximum likelihood" in repr(fit)


def test_errors(ar1):
    model, data, _ = ar1
    with pytest.raises(ModelError, match="zz are not parameters"):
        model.estimate(data, ["x"], estimate=["zz"])
    with pytest.raises(ModelError, match="are fixed"):
        Model(model.equations, params=model.parameters.fix("rho"), roles=model.roles).estimate(data, ["x"], estimate=["rho"])
    with pytest.raises(ModelError, match="name the model's shocks"):
        Model(model.equations, params={"rho": 0.3}).estimate(data, ["x"])
    with pytest.raises(ModelError, match="cannot be solved"):
        Model(model.equations, params={"rho": 1.5}, roles=model.roles).estimate(data, ["x"], bounds={"rho": (-5.0, 5.0)})
    fixed = Model(model.equations, params=model.parameters.fix("rho"), roles=model.roles).estimate(data, ["x"])
    assert fixed.params.names == ("sd_e",) and fixed.parameters["rho"].value == 0.3  # fixed parameters are held


def test_the_dialog_and_project(nk_data, tmp_path):
    from pymetrix.commands import EstimateModel, LoadTemplate, Session
    from pymetrix.testing import check_project

    s = Session()
    s.objects["data"] = nk_data
    s.run(LoadTemplate("new_keynesian", name="nk"))
    check_project(s, EstimateModel("nk", estimate="rho, sd_ei", start="1950Q1"), tmp_path)
    assert "estimated = nk.estimate(data, estimate=['rho', 'sd_ei'], span=('1950Q1', None))" in s.script()
    assert s["estimated"].params.names == ("rho", "sd_ei") and s["estimated"].provenance.sample[0] == "1950Q1"


def test_forecasts_from_the_smoothed_state(nk_data):
    from pymetrix import Scenario

    fit = templates.new_keynesian(rho=0.6).estimate(nk_data, estimate=["rho", "sd_ey", "sd_epi", "sd_ei"])
    span = (str(nk_data.frame.index[-1] + 1), str(nk_data.frame.index[-1] + 8))
    fc = fit.forecast(span)
    first, last = pd.Period(span[0]), nk_data.frame.index[-1]
    # by hand: the smoothed state at the end of the data, carried forward by the decision rules
    sol = fit.solution
    x = np.array([fit.smoothed[v].loc[last] - sol.model.steady.values[v] for v in sol.variables])
    for h in range(8):
        x = sol.P @ x
        p = first + h
        for j, v in enumerate(sol.model.observed):
            assert fc[v].loc[p] == pytest.approx(x[j] + sol.model.steady.values[v], abs=1e-10)
    # observed series in data units, through the measurement equations, exactly
    expected = fit.measurement.apply(fc).frame.loc[span[0]:span[1]]
    for name in fit.measurement.observables:
        np.testing.assert_allclose(fc[name].loc[span[0]:span[1]], expected[name])
    held = fit.forecast(span, Scenario(hold={"i": {span[0]: 0.0}}, free=["ei"]))
    assert held["i"].loc[span[0]] == pytest.approx(0.0, abs=1e-12)
    later = fit.forecast(span, data=nk_data)  # filtering the same data again gives the same forecast
    np.testing.assert_allclose(later.frame.loc[span[0]:, ["y", "pi", "i"]], fc.frame.loc[span[0]:, ["y", "pi", "i"]], atol=1e-8)


def test_tunes_impose_judgement_in_the_filter(nk_data):
    fit = templates.new_keynesian(rho=0.6).estimate(nk_data, estimate=["rho", "sd_ey", "sd_epi", "sd_ei"])
    last = str(nk_data.frame.index[-1])
    plain = fit.system.filter(nk_data)
    tuned = fit.system.filter(nk_data, tunes={"y": {last: plain["y"].loc[last] + 1.0}})  # a view on unobserved output
    assert tuned["y"].loc[last] == pytest.approx(plain["y"].loc[last] + 1.0, abs=1e-8)
    for name in fit.measurement.observables:  # still consistent with every observed figure
        both = pd.concat([tuned[f"{name}_fitted"], nk_data[name]], axis=1).dropna()
        np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-6)
    # output is observed only through its growth, so its level is known up to a constant: the tune sets it, and the
    # whole path moves by the same amount; inflation and the rate, observed exactly, cannot move
    np.testing.assert_allclose(tuned["y"] - plain["y"], 1.0, atol=1e-6)
    np.testing.assert_allclose(tuned["pi"], plain["pi"], atol=1e-8)
    span = (str(nk_data.frame.index[-1] + 1), str(nk_data.frame.index[-1] + 4))
    with_view = fit.forecast(span, data=nk_data, tunes={"y": {last: plain["y"].loc[last] + 1.0}})
    # growth is measured from last quarter's output, so the view on output changes forecast growth (by 4 x 1.0)
    assert with_view["growth"].loc[span[0]] == pytest.approx(fit.forecast(span)["growth"].loc[span[0]] - 4.0, abs=1e-6)
    with pytest.raises(ValueError, match="give the data"):
        fit.forecast(span, tunes={"y": {last: 2.0}})
    with pytest.raises(ValueError, match="not variables of the model"):
        fit.system.filter(nk_data, tunes={"gap": {last: 1.0}})


def test_the_reported_posterior_is_the_one_optimised_with_the_filter_options():
    """start= and presample= (Dynare's lik_init and presample) apply to the log posterior that is maximised and to the
    one reported, and are recorded."""
    from pymetrix import Dataset, templates
    from pymetrix.estimation.priors import Beta

    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=6)
    truth.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    fit = nk.estimate(data, estimate=["rho"], priors={"rho": Beta(0.7, 0.1)}, start="scaled", presample=4)
    assert fit.stats["logpost"] == pytest.approx(fit.log_posterior(np.asarray(fit.params.values, dtype=float)), abs=1e-8)
    assert fit.provenance.method["filter_start"] == "scaled" and fit.provenance.method["presample"] == 4
    plain = nk.estimate(data, estimate=["rho"], priors={"rho": Beta(0.7, 0.1)})
    assert abs(plain.stats["loglike"] - fit.stats["loglike"]) > 1.0  # four periods fewer: a different likelihood
