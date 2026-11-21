"""VECM, Johansen, KPSS and Engle-Granger: our interface over statsmodels, checked against direct calls."""

import warnings

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
from statsmodels.tsa.stattools import coint, kpss  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import VECM as SMVECM  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import select_order as sm_select_order  # noqa: E402

from pymetrix import VECM, Dataset
from pymetrix.commands import FitVECM, Forecast, ImpulseResponse, Johansen, RunTest, Session, VECMLagOrder


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    T = 250
    trend = np.cumsum(rng.normal(size=T))
    return pd.DataFrame(
        {"a": trend + rng.normal(scale=0.5, size=T), "b": 0.8 * trend + rng.normal(scale=0.5, size=T), "c": np.cumsum(rng.normal(size=T))},
        index=pd.period_range("1960Q1", periods=T, freq="Q", name="time"),
    )


@pytest.fixture(scope="module")
def data(frame) -> Dataset:
    return Dataset.from_frame(frame)


def test_johansen_matches_statsmodels(frame, data):
    result = data.johansen(["a", "b", "c"], lags=1)
    ref = coint_johansen(frame.to_numpy(), 0, 1)
    table = result.table.to_frame()
    assert np.allclose(table["trace"], ref.lr1) and np.allclose(table["max-eig"], ref.lr2)
    assert np.allclose(table[["trace 10%", "trace 5%", "trace 1%"]], ref.cvt)
    assert result.rank_trace == select_coint_rank(frame.to_numpy(), 0, 1).rank == 1
    assert result.rank_maxeig == select_coint_rank(frame.to_numpy(), 0, 1, method="maxeig").rank
    prov = result.provenance  # through the statsmodels adapter: its arguments, and the rows the lagged differences take
    assert prov.backend["wraps"] == "statsmodels.tsa.vector_ar.vecm.coint_johansen"
    assert prov.options["stages"]["test"] == {"det_order": 0, "k_ar_diff": 1}
    assert prov.nobs == len(frame) - 2 and prov.presample == ("1960Q1", "1960Q2")


def test_vecm_matches_statsmodels(frame, data):
    fit = VECM(["a", "b", "c"], lags=1, rank="trace").fit(data)
    ref = SMVECM(frame.to_numpy(), k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    assert fit.rank == 1
    assert np.allclose(fit.alpha, ref.alpha) and np.allclose(fit.beta, ref.beta)
    assert np.allclose(fit.coefs, ref.var_rep)
    assert np.isclose(fit.stats["loglike"], ref.llf)
    assert np.allclose(fit.irf(10).values, ref.irf(10).orth_irfs)
    shares = fit.fevd(8).values
    assert np.allclose(shares.sum(axis=2), 1)
    assert fit.provenance.method["backend"].startswith("statsmodels")
    assert "beta" in str(fit.summary()) and "da: db[-1]" in str(fit.short_run())


def test_vecm_forecast_and_tests(frame, data):
    fit = VECM(["a", "b", "c"], lags=1, rank=1).fit(data)
    fc = fit.forecast(data, ("2022Q3", "2023Q4"))
    ref = SMVECM(frame.to_numpy(), k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    mean, lower, _ = ref.predict(steps=6, alpha=0.05)
    assert np.allclose(fc.frame[["a", "b", "c"]], mean) and np.allclose(fc["a_lower"], lower[:, 0])
    with pytest.raises(ValueError, match="right after the estimation sample"):
        fit.forecast(data, ("2023Q1", "2023Q4"))
    g = fit.test("granger", cause="c", effect="a")
    assert np.isclose(g.statistic, ref.test_granger_causality(caused=[0], causing=[2]).test_statistic)
    assert np.isclose(fit.test("portmanteau", lags=10).statistic, ref.test_whiteness(10).test_statistic)
    assert np.isclose(fit.test("normality").statistic, ref.test_normality().test_statistic)


def test_vecm_lag_selection(frame, data):
    sel = VECM(["a", "b", "c"]).select_order(data, max_lags=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm_select_order(frame.to_numpy(), 4, deterministic="co")
    assert sel.best["bic"] == ref.bic and np.allclose(sel.table.to_frame()["AIC"], [ref.ics["aic"][p] for p in range(5)])
    assert sel.provenance.backend["wraps"] == "statsmodels.tsa.vector_ar.vecm.select_order"
    assert sel.provenance.nobs == len(frame) - 5  # every order on the same rows: max_lags + 1 presample


def test_unit_root_and_engle_granger(frame, data):
    k = data.test("kpss", "c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = kpss(frame["c"].to_numpy(), regression="c", nlags="auto")
    assert np.isclose(k.statistic, ref[0]) and k.details["lags used"] == ref[2]
    assert k.critical_values["5%"] == pytest.approx(ref[3]["5%"])
    eg = data.test("engle_granger", "a, b")
    ref_eg = coint(frame["a"], frame["b"])
    assert np.isclose(eg.statistic, ref_eg[0]) and eg.reject()
    assert not data.test("engle_granger", "a, c").reject()
    assert data.test("adf", "c", lags="bic").p_value > 0.05


def test_vecm_commands(data):
    session = Session()
    session.objects["data"] = data
    session.run(Johansen("a, b, c"))
    session.run(VECMLagOrder("a, b, c", max_lags=3))
    session.run(FitVECM("a, b, c", lags=1, rank="1"))
    session.run(ImpulseResponse("vecm", periods=8, draws=100))
    session.run(Forecast("vecm", "2022Q3", "2022Q4"))
    session.run(RunTest(source="vecm", test="granger", options="cause='c', effect='a'"))
    script = session.script()
    assert "johansen_result = data.johansen(['a', 'b', 'c'])" in script
    assert "vecm = VECM(['a', 'b', 'c'], rank=1).fit(data)" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["irf"].values, session["irf"].values)
    assert "irf(periods=8, draws=100)" in script and np.allclose(ns["irf"].lower, session["irf"].lower)  # bootstrap bands, replayed


def test_phillips_ouliaris_by_hand_and_against_arch():
    """Zt and Za from the formulas of Phillips and Ouliaris (1990), every variance divided by T, and arch's output."""
    arch = pytest.importorskip("arch")
    from arch.unitroot.cointegration import phillips_ouliaris

    rng = np.random.default_rng(0)
    T, bw = 200, 5
    x = np.cumsum(rng.normal(size=T))
    frame = pd.DataFrame({"y": 1 + 0.8 * x + rng.normal(size=T), "x": x, "w": np.cumsum(rng.normal(size=T))},
                         index=pd.period_range("1970Q1", periods=T, freq="Q", name="time"))
    data = Dataset.from_frame(frame)
    zt = data.test("phillips_ouliaris", "y, x", bandwidth=bw)
    za = data.test("phillips_ouliaris", "y, x", test_type="Za", bandwidth=bw)
    X = np.column_stack([np.ones(T), x])
    u = frame["y"].to_numpy() - X @ np.linalg.lstsq(X, frame["y"].to_numpy(), rcond=None)[0]
    S = u[:-1] @ u[:-1]
    a = (u[1:] @ u[:-1]) / S
    k = u[1:] - a * u[:-1]
    s2 = k @ k / T
    lr = s2 + 2 * sum((1 - j / (bw + 1)) * (k[j:] @ k[:-j]) / T for j in range(1, bw + 1))
    assert za.statistic == pytest.approx(T * (a - 1) - 0.5 * T**2 * (lr - s2) / S, rel=1e-10)
    assert zt.statistic == pytest.approx(np.sqrt(S) * (a - 1) / np.sqrt(lr) - 0.5 * (lr - s2) * T / np.sqrt(lr * S), rel=1e-10)
    for kind in ("Zt", "Za", "Pu", "Pz"):
        ours = data.test("phillips_ouliaris", "y, x", test_type=kind, trend="ct")
        ref = phillips_ouliaris(frame["y"], frame[["x"]], trend="ct", test_type=kind)
        assert ours.statistic == pytest.approx(ref.stat) and ours.p_value == pytest.approx(ref.pvalue)
    assert zt.reject() and zt.critical_values["1%"] < zt.critical_values["10%"]
    # arch 9.0 changed the automatic bandwidth, so the details say which arch computed the test
    assert zt.details["arch version"] == arch.__version__ and ours.details["bandwidth"] == pytest.approx(ref.bandwidth)
    assert not data.test("phillips_ouliaris", "y, w").reject()  # independent random walks
