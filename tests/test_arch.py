"""arch: cross-check of our native GARCH, arch-backed volatility models and forecasts, unit-root tests."""

import warnings

import numpy as np
import pandas as pd
import pytest

arch = pytest.importorskip("arch")
from arch import arch_model  # noqa: E402
from arch.unitroot import DFGLS, PhillipsPerron, VarianceRatio, ZivotAndrews  # noqa: E402

from pymetrix import APARCH, EGARCH, FIGARCH, GARCH, ML, TARCH, Dataset, Equation  # noqa: E402
from pymetrix.commands import FitEquation, Session  # noqa: E402


def _simulate(n, omega, alpha, gamma, beta, nu=None, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu) if nu else rng.normal(size=n)
    e, s2 = np.zeros(n), np.full(n, omega / (1 - alpha - gamma / 2 - beta))
    for t in range(n):
        if t:
            s2[t] = omega + (alpha + gamma * (e[t - 1] < 0)) * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    return 0.2 + e


def _data(r) -> Dataset:
    return Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2000-01-01", periods=len(r), freq="D", name="time")))


def _ref(r, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(r, mean="Constant", **kw).fit(disp="off", cov_type="classic")


@pytest.mark.parametrize("spec, kw, nu, gamma", [
    (GARCH(), dict(vol="GARCH", p=1, q=1), None, 0.0),
    (GARCH(asym=1, dist="t"), dict(vol="GARCH", p=1, o=1, q=1, dist="t"), 8, 0.1),
])
def test_native_garch_agrees_with_arch(spec, kw, nu, gamma):
    r = _simulate(3000, 0.05, 0.03 if gamma else 0.08, gamma, 0.88, nu=nu, seed=2 if nu else 1)
    ours = Equation("r ~ 1", variance=spec).fit(_data(r))
    ref = _ref(r, **kw)
    assert np.allclose(ours.params.values, ref.params, atol=5e-4)
    assert abs(ours.stats["loglike"] - ref.loglikelihood) < 1e-5
    assert np.allclose(ours.params.se, ref.std_err, rtol=1e-3)


@pytest.fixture(scope="module")
def returns():
    return _simulate(2000, 0.05, 0.05, 0.1, 0.88, nu=6, seed=4)


@pytest.mark.parametrize("spec, kw", [
    (EGARCH(dist="t"), dict(vol="EGARCH", p=1, o=1, q=1, dist="t")),
    (TARCH(dist="skewt"), dict(vol="GARCH", p=1, o=1, q=1, power=1.0, dist="skewt")),
    (FIGARCH(dist="ged"), dict(vol="FIGARCH", p=1, q=1, dist="ged")),
    (APARCH(), dict(vol="APARCH", p=1, o=1, q=1)),
    (GARCH(dist="skewt"), dict(vol="GARCH", p=1, q=1, dist="skewt")),
    (GARCH(backend="arch"), dict(vol="GARCH", p=1, q=1)),
])
def test_arch_backed_models_match_arch(returns, spec, kw):
    fit = Equation("r ~ 1", variance=spec).fit(_data(returns), method=ML(cov="classic"))
    ref = _ref(returns, **kw)
    assert np.allclose(fit.params.values, ref.params) and np.allclose(fit.params.se, ref.std_err)
    assert np.isclose(fit.stats["loglike"], ref.loglikelihood)
    assert fit.params.names[0] == "const" and fit.provenance.method["backend"].startswith("arch")
    assert np.allclose(fit.extra["volatility"], ref.conditional_volatility)


def test_arch_backed_forecasts(returns):
    data = _data(returns)
    idx = data.frame.index
    fit = Equation("r ~ 1", variance=GARCH(dist="skewt")).fit(data, span=(str(idx[0]), str(idx[1799])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = arch_model(returns[:1800], mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fix(fit.params.values)
        fixed_full = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fix(fit.params.values)
    dynamic = fit.forecast(data, (str(idx[1800]), str(idx[1819])), dynamic=True)  # analytic
    assert np.allclose(dynamic["variance"], ref.forecast(horizon=20, reindex=False).variance.iloc[-1])
    static = fit.forecast(data, (str(idx[1800]), str(idx[1999])))
    assert np.allclose(static["variance"], fixed_full.forecast(horizon=1, start=1799, reindex=False).variance.iloc[:-1, 0])
    tarch = Equation("r ~ 1", variance=TARCH(dist="t")).fit(data, span=(str(idx[0]), str(idx[1799])))
    sim = tarch.forecast(data, (str(idx[1800]), str(idx[1809])), dynamic=True)  # simulated beyond step one
    one = tarch.forecast(data, (str(idx[1800]), str(idx[1800])))
    assert np.isclose(sim["variance"].iloc[0], one["variance"].iloc[0]) and "simulation" in sim.meta.description
    assert (sim["upper"] > sim["forecast"]).all()


@pytest.fixture(scope="module")
def levels():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=300))
    broken = rng.normal(size=300) + np.where(np.arange(300) < 150, 0.0, 4.0)  # stationary with a level shift
    index = pd.period_range("1950Q1", periods=300, freq="Q", name="time")
    return walk, broken, Dataset.from_frame(pd.DataFrame({"walk": walk, "broken": broken}, index=index))


def test_unit_root_tests_match_arch(levels):
    walk, broken, data = levels
    pairs = [
        (data.test("phillips_perron", "walk"), PhillipsPerron(walk)),
        (data.test("dfgls", "walk", trend="ct"), DFGLS(walk, trend="ct")),
        (data.test("variance_ratio", "walk", lags=4), VarianceRatio(walk, lags=4)),
        (data.test("zivot_andrews", "broken"), ZivotAndrews(broken)),
    ]
    for ours, ref in pairs:
        assert np.isclose(ours.statistic, ref.stat) and np.isclose(ours.p_value, ref.pvalue)
    assert not pairs[0][0].reject() and pairs[3][0].reject()
    found = pd.Period(pairs[3][0].details["break period"], freq="Q")
    assert abs((found - pd.Period("1987Q3", "Q")).n) <= 3  # the shift is at observation 150


def test_dialog_builds_arch_models(returns):
    session = Session()
    session.objects["data"] = _data(returns)
    session.run(FitEquation("r ~ 1", method="ML", variance="EGARCH", dist="t", name="eg"))
    session.run(FitEquation("r ~ 1", method="ML", variance="FIGARCH", variance_settings="power=1.0", name="fig"))
    script = session.script()
    assert "variance=EGARCH(dist='t')" in script and "variance=FIGARCH(power=1.0)" in script
    ns = {"data": session["data"]}
    exec(script, ns)
    assert np.allclose(ns["eg"].params.values, session["eg"].params.values)
