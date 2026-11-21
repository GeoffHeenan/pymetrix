"""The diffuse log likelihood's constant terms: pymetrix's exact diffuse likelihood (Durbin and Koopman 2012, section
7.2.2, the diffuse periods' terms included) against the convention that leaves them out (``diffuse_terms="exclude"``,
what statsmodels' UnobservedComponents and SARIMAX report by default).

Checks: the difference equals the analytic constant ``d/2 log 2 pi + 1/2 log det(X X')`` with ``X`` the d x d matrix
whose rows are ``Z T^(t-1)`` on the diffuse states (the product of the diffuse variances F_inf is det(X X'): they are
its Cholesky pivots), for d = 1, 2, 3 and 5; with the option, the log likelihood equals statsmodels' (computed here) to
1e-8 on a local level, a local linear trend, a seasonal UC model and an ARIMA(1,1,1); the default is unchanged;
estimates and standard errors agree; AIC, BIC and HQ move by twice the constant."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pymetrix.statespace as ss
from pymetrix import Dataset, Equation, Model, StateSpace, UnobservedComponents
from pymetrix.errors import SpecificationError
from pymetrix.metadata import Role

sm = pytest.importorskip("statsmodels.api")

LOG2PI = np.log(2 * np.pi)
N = 160


def _data(y: np.ndarray) -> Dataset:
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1970Q1", periods=len(y), freq="Q", name="time")))


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(7)
    trend = np.cumsum(np.cumsum(rng.normal(0, 0.1, N)) + rng.normal(0, 0.5, N))
    return trend + np.tile([1.0, -2.0, 0.5, 0.5], N // 4) * 2 + rng.normal(0, 1, N) + 50


def _sm_exact(model_cls, y, params, **kw):
    """statsmodels' exact diffuse filter, and with the diffuse periods left out of its likelihood (its burn-in)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = model_cls(y, use_exact_diffuse=True, **kw).filter(params)
        burned = model_cls(y, use_exact_diffuse=True, loglikelihood_burn=int(full.nobs_diffuse), **kw).filter(params)
        default = model_cls(y, **kw).filter(params)  # statsmodels' default: approximately diffuse, the first d left out
    return full, burned, default


LEVEL = StateSpace(["level"], ["level = level[-1] + e"], ["y = level + u"], {"e": "se", "u": "su"}, ["se", "su"])
TREND = StateSpace(["level", "slope"], ["level = level[-1] + slope[-1] + e", "slope = slope[-1] + w"], ["y = level + u"],
                   {"e": "se", "w": "sw", "u": "su"}, ["se", "sw", "su"])
SEASONAL = StateSpace(["level", "slope", "seas"],
                      ["level = level[-1] + slope[-1] + e", "slope = slope[-1] + w", "seas = -seas[-1] - seas[-2] - seas[-3] + g"],
                      ["y = level + seas + u"], {"e": "se", "w": "sw", "g": "sg", "u": "su"}, ["se", "sw", "sg", "su"])
# d = 3: a local level with a dummy seasonal of period 3, and a loading of 2 (F_inf is not 1)
SEASON3 = StateSpace(["level", "seas"], ["level = level[-1] + e", "seas = -seas[-1] - seas[-2] + g"], ["y = 2*level + seas + u"],
                     {"e": "se", "g": "sg", "u": "su"}, ["se", "sg", "su"])


def _analytic(model: StateSpace, values: dict) -> tuple[int, float]:
    """d and the constant d/2 log 2 pi + 1/2 log det(X X'), X with rows Z T^(t-1) (one series, all diffuse)."""
    m = model.system(values)
    d = len(m.states)
    X = np.vstack([m.Z @ np.linalg.matrix_power(m.T, t) for t in range(d)])
    return d, d / 2 * LOG2PI + 0.5 * np.linalg.slogdet(X @ X.T)[1]


@pytest.mark.parametrize("model, values, d", [
    (LEVEL, dict(se=0.7, su=1.0), 1),
    (TREND, dict(se=0.5, sw=0.1, su=1.0), 2),
    (SEASON3, dict(se=0.5, sg=0.3, su=1.0), 3),
    (SEASONAL, dict(se=0.5, sw=0.1, sg=0.3, su=1.0), 5),
])
@pytest.mark.parametrize("engine", ["statsmodels", "native"])
def test_offset_is_the_analytic_constant_at_every_parameter_value(series, model, values, d, engine, monkeypatch):
    monkeypatch.setattr(ss, "ENGINE", engine)
    data = _data(series)
    d_, constant = _analytic(model, values)
    assert d_ == d
    for scale in (1.0, 0.3, 2.5):  # a constant: the same at other parameter values
        vals = {k: v * scale for k, v in values.items()}
        gap = model.loglike(data, vals) - model.loglike(data, vals, diffuse_terms="exclude")
        assert gap == pytest.approx(-constant, abs=1e-9)
    if model is SEASONAL:  # the reported case: 5/2 log 2 pi + log 16 = 7.367
        assert constant == pytest.approx(2.5 * LOG2PI + np.log(16), abs=1e-12) and round(constant, 3) == 7.367


@pytest.mark.parametrize("model, values, sm_kw, sm_params", [
    (LEVEL, dict(se=0.7, su=1.0), dict(level="llevel"), [1.0, 0.49]),
    (TREND, dict(se=0.5, sw=0.1, su=1.0), dict(level="lltrend"), [1.0, 0.25, 0.01]),
    (SEASONAL, dict(se=0.5, sw=0.1, sg=0.3, su=1.0), dict(level="lltrend", seasonal=4), [1.0, 0.25, 0.01, 0.09]),
])
def test_matches_statsmodels_unobserved_components(series, model, values, sm_kw, sm_params):
    data = _data(series)
    full, burned, default = _sm_exact(sm.tsa.UnobservedComponents, series, sm_params, **sm_kw)
    assert model.loglike(data, values) == pytest.approx(full.llf, abs=1e-8)  # the default: the exact diffuse likelihood
    excluded = model.loglike(data, values, diffuse_terms="exclude")
    assert excluded == pytest.approx(burned.llf, abs=1e-8)
    assert excluded == pytest.approx(float(np.sum(full.llf_obs[full.nobs_diffuse:])), abs=1e-8)
    # statsmodels' default (a large initial variance, the first d periods left out) is the same up to its approximation
    assert default.loglikelihood_burn == full.nobs_diffuse and excluded == pytest.approx(default.llf, abs=1e-3)


def test_arima_with_a_diffuse_start_matches_sarimax():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 0.8, N + 1)
    x = np.zeros(N + 1)
    for t in range(1, N + 1):
        x[t] = 0.5 * x[t - 1] + e[t] + 0.3 * e[t - 1]
    y = np.cumsum(x[1:]) + 10
    arima = StateSpace(["lvl", "x", "a"], ["lvl = lvl[-1] + phi*x[-1] + theta*a[-1] + eps", "x = phi*x[-1] + theta*a[-1] + eps", "a = eps"],
                       ["y = lvl"], {"eps": "sigma"}, ["phi", "theta", "sigma"], stationary=["x", "a"])
    values = dict(phi=0.5, theta=0.3, sigma=0.8)
    full, burned, default = _sm_exact(sm.tsa.SARIMAX, y, [0.5, 0.3, 0.64], order=(1, 1, 1))
    data = _data(y)
    assert full.nobs_diffuse == 1
    assert arima.loglike(data, values) == pytest.approx(full.llf, abs=1e-8)
    assert arima.loglike(data, values, diffuse_terms="exclude") == pytest.approx(burned.llf, abs=1e-8)
    assert arima.loglike(data, values) - arima.loglike(data, values, diffuse_terms="exclude") == pytest.approx(-LOG2PI / 2, abs=1e-10)
    assert arima.loglike(data, values, diffuse_terms="exclude") == pytest.approx(default.llf, abs=1e-3)


def test_fit_estimates_and_criteria(series):
    data = _data(series)
    uc = UnobservedComponents("y", trend="local linear trend", cycle=0)
    default, other = uc.fit(data), uc.fit(data, diffuse_terms="exclude")
    for name in default.values:  # the same estimates and standard errors: the likelihoods differ by a constant
        assert other.values[name] == pytest.approx(default.values[name], rel=1e-5, abs=1e-6)
    assert np.allclose(other.params.se, default.params.se, rtol=1e-4, equal_nan=True)
    gap = other.stats["loglike"] - default.stats["loglike"]
    assert gap == pytest.approx(LOG2PI, abs=1e-5)  # d = 2, F_inf = 1, 1
    assert other.stats["aic"] - default.stats["aic"] == pytest.approx(-2 * gap)
    assert other.stats["bic"] - default.stats["bic"] == pytest.approx(-2 * gap)
    assert other.stats["diffuse periods"] == default.stats["diffuse periods"] == 2
    assert "diffuse_terms" not in default.provenance.method and other.provenance.method["diffuse_terms"] == "exclude"
    # criteria: either convention from either fit, consistent with stats
    mine, theirs = default.criteria(), default.criteria("exclude")
    assert mine["loglike"] == default.stats["loglike"] and mine["aic"] == pytest.approx(default.stats["aic"])
    assert mine["bic"] == pytest.approx(default.stats["bic"])
    assert theirs["loglike"] - mine["loglike"] == pytest.approx(LOG2PI, abs=1e-10)
    for key in ("aic", "bic", "hq"):
        assert theirs[key] - mine[key] == pytest.approx(-2 * LOG2PI, abs=1e-9)
    n, k = default.stats["nobs"], default.stats["k"]
    assert mine["hq"] == pytest.approx(-2 * mine["loglike"] + 2 * k * np.log(np.log(n)))
    assert other.criteria()["loglike"] == other.stats["loglike"]
    with pytest.raises(SpecificationError, match="diffuse_terms"):
        uc.fit(data, diffuse_terms="statsmodels")


def test_no_diffuse_states_no_difference(series):
    stationary = StateSpace(["x"], ["x = rho*x[-1] + e"], ["y = x + u"], {"e": "sd", "u": 0.5}, ["rho", "sd"])
    data = _data(series - series.mean())
    values = dict(rho=0.9, sd=1.0)
    assert stationary.loglike(data, values) == stationary.loglike(data, values, diffuse_terms="exclude")
    fit = stationary.filter(data, values)
    assert fit.criteria("exclude") == fit.criteria("include")


def test_linearised_model_estimation():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(0, 0.6, 120))
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1960Q1", periods=120, freq="Q", name="time")))
    walk = Model([Equation("x ~ x[-1] + a*e", params=["a"])], params={"a": 1.0}, roles={"e": Role.SHOCK})  # a unit root: diffuse
    default = walk.estimate(data, ["x"], estimate=["sd_e"], shock_sd={"e": 1.0})
    other = walk.estimate(data, ["x"], estimate=["sd_e"], shock_sd={"e": 1.0}, diffuse_terms="exclude")
    assert other.stats["loglike"] - default.stats["loglike"] == pytest.approx(LOG2PI / 2, abs=1e-6)  # d = 1, F_inf = 1
    assert other.stats["aic"] - default.stats["aic"] == pytest.approx(-LOG2PI, abs=1e-5)
    assert other.stats["bic"] - default.stats["bic"] == pytest.approx(-LOG2PI, abs=1e-5)
    # the ML estimate of a random walk's shock sd is the root mean square of the differences, either way
    rms = np.sqrt(np.mean(np.diff(x) ** 2))
    assert other.shock_sd["e"] == pytest.approx(rms, rel=1e-4) and default.shock_sd["e"] == pytest.approx(rms, rel=1e-4)
    # and the log likelihood is statsmodels' (a random-walk level, exact diffuse, its first period left out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rw = sm.tsa.UnobservedComponents(x, level="rwalk", use_exact_diffuse=True, loglikelihood_burn=1)
        ref = rw.filter([other.shock_sd["e"] ** 2])
    assert other.stats["loglike"] == pytest.approx(ref.llf, abs=1e-8)
    assert other.provenance.method["diffuse_terms"] == "exclude" and "diffuse_terms" not in default.provenance.method
    with pytest.raises(SpecificationError, match="Kalman filter"):
        walk.estimate(data, ["x"], estimate=["sd_e"], shock_sd={"e": 1.0}, diffuse_terms="exclude", occbin=True)


def test_dialogs_record_the_option_and_replay(series):
    from pymetrix.commands import FitStateSpace, FitUnobservedComponents, Session

    s = Session()
    s.objects["data"] = _data(series)
    cmd = FitUnobservedComponents("y", trend="local linear trend", cycle=0, diffuse_terms="exclude")
    fit = s.run(cmd)
    direct = UnobservedComponents("y", trend="local linear trend", cycle=0).fit(_data(series), diffuse_terms="exclude")
    assert fit.stats["loglike"] == pytest.approx(direct.stats["loglike"], abs=1e-9)
    assert "diffuse_terms='exclude'" in cmd.to_code()
    assert "diffuse_terms" not in FitUnobservedComponents("y", cycle=0).to_code()  # the default is not written
    ssm = FitStateSpace("level", "level = level[-1] + e", "y = level + u", "e = se, u = su", "se, su", diffuse_terms="exclude")
    assert "diffuse_terms='exclude'" in ssm.to_code()
    got = s.run(ssm)
    assert got.stats["loglike"] == pytest.approx(LEVEL.fit(_data(series), diffuse_terms="exclude").stats["loglike"], abs=1e-9)
    space = {"data": _data(series)}
    exec(s.script(), space)  # the generated script replays both dialogs
    assert space[cmd.name].stats["loglike"] == pytest.approx(fit.stats["loglike"], abs=1e-9)
    assert space[ssm.name].stats["loglike"] == pytest.approx(got.stats["loglike"], abs=1e-9)


def test_model_estimation_dialog_records_the_option():
    from pymetrix import templates
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateModel
    from pymetrix.forms import form_spec

    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=11)
    truth.index = pd.period_range("1990Q1", periods=80, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    s = Session()
    s.objects.update({"nk": nk, "data": data})
    s.run(EstimateModel("nk", "data", estimate="kappa", initial="diffuse", diffuse_terms="exclude", name="fit"))
    assert "start='diffuse'" in s.history[-1].code and "diffuse_terms='exclude'" in s.history[-1].code
    api = templates.new_keynesian().estimate(data, ["kappa"], start="diffuse", diffuse_terms="exclude")
    assert np.allclose(s["fit"].params.values, api.params.values) and s["fit"].stats["loglike"] == pytest.approx(api.stats["loglike"])
    included = templates.new_keynesian().estimate(data, ["kappa"], start="diffuse")
    assert api.stats["loglike"] != pytest.approx(included.stats["loglike"])  # the diffuse periods' terms are left out
    namespace = dict(s.namespace) | {"nk": templates.new_keynesian(), "data": data}
    exec(s.history[-1].code, namespace)  # noqa: S102 - the session's own script
    assert np.allclose(namespace["fit"].params.values, s["fit"].params.values)
    assert "diffuse_terms" in {f.name for f in form_spec(EstimateModel)}
