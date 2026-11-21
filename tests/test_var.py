"""VARs against statsmodels."""

import warnings

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
from statsmodels.tsa.api import VAR as SMVAR  # noqa: E402

from pymetrix import VAR, Cholesky, Dataset, LongRun
from pymetrix.commands import FitVAR, ImpulseResponse, RunTest, Session, VARLagOrder, VarianceDecomposition


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    T = 200
    A1 = np.array([[0.5, 0.1, 0], [0.2, 0.4, 0.1], [0, 0.3, 0.6]])
    A2 = np.array([[-0.2, 0, 0], [0, 0.1, 0], [0.1, 0, -0.1]])
    cov = [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]]
    y = np.zeros((T, 3))
    for t in range(2, T):
        y[t] = 0.1 + A1 @ y[t - 1] + A2 @ y[t - 2] + rng.multivariate_normal(np.zeros(3), cov)
    return pd.DataFrame(y, columns=["a", "b", "c"], index=pd.period_range("1970Q1", periods=T, freq="Q", name="time"))


@pytest.fixture(scope="module")
def pair(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SMVAR(frame).fit(2)
    return VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame)), ref


def test_estimates_match_statsmodels(pair):
    ours, ref = pair
    assert np.allclose(ours.estimate.B, ref.params.to_numpy())
    assert np.allclose(ours.sigma, ref.sigma_u)
    assert np.allclose(ours.params.se, np.sqrt(np.diag(ref.cov_params())).reshape(-1, 3).T.reshape(-1))
    assert np.isclose(ours.stats["loglike"], ref.llf)
    for ours_key, ref_key in (("aic", "aic"), ("bic", "bic"), ("hq", "hqic"), ("fpe", "fpe")):
        assert np.isclose(ours.stats[ours_key], getattr(ref, ref_key))
    assert ours.is_stable and np.allclose(np.sort(ours.roots), np.sort(1 / np.abs(ref.roots)))
    assert ours.params.names[1] == "a: a[-1]"
    assert "VAR(2)" in str(ours.summary())


def test_lag_selection_matches_statsmodels(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SMVAR(frame).select_order(6)
    data = Dataset.from_frame(frame)
    sel = VAR(["a", "b", "c"], lags="bic").select_order(data, 6)
    table = sel.table.to_frame()
    for ours_key, ref_key in (("AIC", "aic"), ("BIC", "bic"), ("HQ", "hqic"), ("FPE", "fpe")):
        assert np.allclose(table[ours_key], [ref.ics[ref_key][i] for i in range(7)])
    assert sel.best == {"aic": ref.aic, "bic": ref.bic, "hq": ref.hqic, "fpe": ref.fpe}
    chosen = VAR(["a", "b", "c"], lags="bic", max_lags=6).fit(data)
    assert chosen.lags == max(sel.best["bic"], 1)


def test_impulse_responses_and_fevd(pair):
    ours, ref = pair
    r = ref.irf(10)
    assert np.allclose(ours.irf(10, draws=0).values, r.orth_irfs)
    assert np.allclose(ours.irf(10, orthogonal=False, draws=0).values, r.irfs)
    assert np.allclose(ours.irf(10, cumulative=True, draws=0).values, r.cum_effects @ np.linalg.cholesky(ref.sigma_u.to_numpy()))
    assert np.allclose(ours.fevd(10).values, ref.fevd(10).decomp.transpose(1, 0, 2))
    banded = ours.irf(8, draws=200, seed=1)
    assert np.all(banded.lower <= banded.values + 1e-12) and np.all(banded.values <= banded.upper + 1e-12)
    assert banded["b", "a"].index[-1] == 8


def test_identification(pair, frame):
    ours, _ = pair
    reordered = ours.identify(Cholesky(["c", "b", "a"]))
    impact = reordered.irf(0, draws=0).values[0]
    assert impact[1, 0] == pytest.approx(0) and impact[2, 0] == pytest.approx(0) and impact[2, 1] == pytest.approx(0)  # c ordered first
    assert np.allclose(impact @ impact.T, ours.estimate.sigma)
    long_run = ours.identify(LongRun()).irf(400, cumulative=True, draws=0).values[-1]
    assert np.allclose(np.triu(long_run, 1), 0, atol=1e-8)  # lower-triangular long-run effects
    with pytest.raises(ValueError, match="once"):
        ours.identify(Cholesky(["a", "b"]))


def test_forecast_matches_statsmodels(pair, frame):
    ours, ref = pair
    fc = ours.forecast(Dataset.from_frame(frame), ("2020Q1", "2021Q4"))
    mean, lower, upper = ref.forecast_interval(frame.to_numpy()[-2:], 8)
    assert np.allclose(fc.frame[["a", "b", "c"]], mean)
    assert np.allclose(fc.frame[["a_lower", "b_lower", "c_lower"]], lower)


def test_var_tests_match_statsmodels(pair):
    ours, ref = pair
    g = ours.test("granger", cause=["b", "c"], effect="a")
    rg = ref.test_causality("a", ["b", "c"], kind="f")
    assert np.isclose(g.statistic, rg.test_statistic) and g.df == rg.df
    assert np.isclose(ours.test("portmanteau", lags=8).statistic, ref.test_whiteness(8).test_statistic)
    assert np.isclose(ours.test("portmanteau", lags=8, adjusted=True).statistic, ref.test_whiteness(8, adjusted=True).test_statistic)
    assert np.isclose(ours.test("normality").statistic, ref.test_normality().test_statistic)


def test_var_commands(frame):
    data = Dataset.from_frame(frame)
    session = Session()
    session.objects["data"] = data
    session.run(VARLagOrder("a, b, c", max_lags=4))
    session.run(FitVAR("a, b, c", lags="2"))
    session.run(ImpulseResponse("var", periods=12, ordering="c, b, a", draws=0))
    session.run(VarianceDecomposition("var", periods=12))
    session.run(RunTest(source="var", test="granger", options="cause='b', effect='a'"))
    script = session.script()
    assert "var = VAR(['a', 'b', 'c'], lags=2).fit(data)" in script
    assert "irf = var.identify(Cholesky(['c', 'b', 'a'])).irf(periods=12, draws=0)" in script
    assert "lag_order = VAR(['a', 'b', 'c'], lags='aic', max_lags=4).select_order(data)" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["irf"].values, session["irf"].values)
