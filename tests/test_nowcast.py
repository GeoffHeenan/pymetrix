"""Nowcasting tools: the ragged edge, data as known on a day, and filling or realigning the edge."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.nowcast import fill_edge, realign


@pytest.fixture(scope="module")
def monthly():
    return Dataset.example("sample_monthly")


def test_the_example_data_tracks_gdp(monthly):
    quarterly = Dataset.example("sample_macro")
    assert monthly.frame.index[0] == pd.Period("2000-01", "M") and monthly.frame.index[-1] == pd.Period("2025-03", "M")
    assert {n: monthly.variables[n].release_lag for n in monthly.names} == {"ip": 45, "retail": 15, "emp": 5, "pmi": 1}
    assert quarterly.variables["y"].release_lag == 30
    gdp = 400 * np.log(quarterly["y"]).diff()
    assert gdp.abs().max() < 30  # no start-up transient (it reached 160 before the burn-in)
    by_quarter = monthly.frame.groupby(monthly.frame.index.asfreq("Q")).mean()
    growth = 400 * np.log(by_quarter[["ip", "retail"]]).diff()
    both = pd.concat([gdp, growth], axis=1).dropna()
    assert (both.corr().loc["y", ["ip", "retail"]] > 0.5).all()  # the indicators carry news about GDP


def test_as_of_rebuilds_what_was_known(monthly):
    known = monthly.as_of("2025-02-15")
    last = {n: str(known.frame[n].last_valid_index()) for n in known.names}
    assert last == {"ip": "2024-12", "retail": "2025-01", "emp": "2025-01", "pmi": "2025-01"}
    assert monthly.as_of("2025-02-14")["retail"].last_valid_index() == pd.Period("2024-12", "M")  # Jan 31 + 15 days
    assert monthly.as_of("2025-02-14")["ip"].last_valid_index() == pd.Period("2024-12", "M")  # Dec 31 + 45 = Feb 14
    early = monthly.as_of("2025-02-15", lags={"ip": 10})
    assert early["ip"].last_valid_index() == pd.Period("2025-01", "M")
    np.testing.assert_array_equal(known.frame.loc[:"2024-12"].to_numpy(), monthly.frame.loc[:"2024-12"].to_numpy())  # nothing else touched
    table = known.ragged_edge("2025-02-15").to_frame()
    assert table.loc["ip", "periods behind"] == 1 and table.loc["pmi", "periods behind"] == 0
    assert table.loc["ip", "published"] == "2025-02-14"


def test_filling_and_realigning_the_edge(monthly):
    known = monthly.as_of("2025-02-15")
    filled = fill_edge(known, lags=2, through="2025-03")
    assert filled.frame.loc["2025-03"].notna().all()
    np.testing.assert_array_equal(filled["retail"].loc[:"2025-01"], known["retail"].loc[:"2025-01"])  # data kept
    # the AR(2) forecast by hand, for one series
    y = known["pmi"].dropna().to_numpy()
    X = np.column_stack([np.ones(len(y) - 2), y[1:-1], y[:-2]])
    b = np.linalg.lstsq(X, y[2:], rcond=None)[0]
    feb = b[0] + b[1] * y[-1] + b[2] * y[-2]
    mar = b[0] + b[1] * feb + b[2] * y[-1]
    assert filled["pmi"].loc["2025-02"] == pytest.approx(feb) and filled["pmi"].loc["2025-03"] == pytest.approx(mar)
    shifted = realign(known)
    assert shifted["ip"].loc["2025-01"] == known["ip"].loc["2024-12"]  # moved one month on
    assert shifted["pmi"].loc["2025-01"] == known["pmi"].loc["2025-01"]  # already at the edge
    quarterly = Dataset.example("sample_macro").as_of("2025-01-29")  # GDP for 2024Q4 comes out on 30 January
    assert quarterly["y"].last_valid_index() == pd.Period("2024Q3", "Q") and quarterly["c"].last_valid_index() == pd.Period("2024Q3", "Q")


@pytest.fixture(scope="module")
def growth(monthly):
    quarterly = Dataset.example("sample_macro").with_series(gy="400*dlog(y)")
    indicators = monthly.with_series(gr="1200*dlog(retail)", gip="1200*dlog(ip)", gemp="1200*dlog(emp)")
    return quarterly, indicators


def test_bridge_equations(growth):
    from pymetrix import OLS, Bridge, Equation

    quarterly, monthly = growth
    bridge = Bridge("gy ~ gr + pmi").fit(quarterly, monthly)
    # the same regression by hand: quarterly means of the monthly indicators
    means = monthly.frame[["gr", "pmi"]].groupby(monthly.frame.index.asfreq("Q")).mean()
    by_hand = Dataset.from_frame(pd.concat([quarterly.frame[["gy"]], means], axis=1).rename_axis("time"))
    direct = Equation("gy ~ gr + pmi").fit(by_hand, method=OLS())
    np.testing.assert_allclose(bridge.params.values, direct.params.values, rtol=1e-10)
    # nowcast with the edge on 15 February: February and March forecast by AR(2), then averaged
    known = monthly.as_of("2025-02-15")
    filled = fill_edge(known, lags=2, through="2025-03")
    q1 = filled.frame.loc["2025-01":"2025-03", ["gr", "pmi"]].mean()
    expected = direct.params.values @ np.r_[1.0, q1["gr"], q1["pmi"]]
    nowcast = bridge.nowcast(quarterly, known)
    assert list(nowcast.index.astype(str)) == ["2025Q1"] and nowcast.iloc[0] == pytest.approx(expected, rel=1e-10)
    assert "averaged within quarters" in str(bridge.summary())
    with pytest.raises(ValueError, match="nothing to nowcast"):
        bridge.nowcast(quarterly, monthly.as_of("2024-12-15"), through="2024Q3")


def test_midas_with_a_ragged_edge_and_combination(growth):
    from pymetrix import MIDAS
    from pymetrix.nowcast import combine

    quarterly, monthly = growth
    known = realign(monthly.as_of("2025-02-15"))  # every indicator now ends in January
    midas = MIDAS("gy ~ gr + pmi", lags=6, high_lag=2).fit(quarterly, monthly)  # a model for the first month of a quarter
    from_midas = midas.nowcast(quarterly, known).loc[["2025Q1"]]
    from_bridge = __import__("pymetrix").Bridge("gy ~ gr + pmi").fit(quarterly, monthly).nowcast(quarterly, monthly.as_of("2025-02-15"))
    both = combine({"midas": from_midas, "bridge": from_bridge})
    assert both.loc["2025Q1"] == pytest.approx((from_midas.loc["2025Q1"] + from_bridge.loc["2025Q1"]) / 2)
    weighted = combine({"midas": from_midas, "bridge": from_bridge}, {"midas": 3, "bridge": 1})
    assert weighted.loc["2025Q1"] == pytest.approx(0.75 * from_midas.loc["2025Q1"] + 0.25 * from_bridge.loc["2025Q1"])
    with pytest.raises(ValueError, match="non-negative"):
        combine({"midas": from_midas}, {"midas": -1})


def test_the_dynamic_factor_model_is_statsmodels(growth):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    from pymetrix import DynamicFactor

    quarterly, monthly = growth
    names = ["gr", "gip", "gemp", "pmi"]
    known = monthly.as_of("2025-02-15")
    fit = DynamicFactor(monthly=names, quarterly=["gy"]).fit(quarterly, known)
    month = known.frame[names].loc["2000-02":]  # from the first month every series has data (growth rates start a month after the survey)
    direct = DynamicFactorMQ(month, endog_quarterly=quarterly.frame[["gy"]], factors=1, factor_orders=2,
                             idiosyncratic_ar1=True).fit(disp=False, maxiter=500)
    np.testing.assert_allclose(fit.result.params.to_numpy(), direct.params.to_numpy(), rtol=1e-8)
    assert fit.result.llf == pytest.approx(direct.llf) and fit.provenance.converged
    nowcast = fit.nowcast("gy")
    assert list(nowcast.index.astype(str)) == ["2025Q1"]
    smoothed = direct.get_prediction(end="2025-03", information_set="smoothed").predicted_mean["gy"]
    assert nowcast.iloc[0] == pytest.approx(float(smoothed.iloc[-1]))
    assert fit.loadings.shape == (5, 1) and fit.factors.shape[1] == 1 and "Factor loadings" in str(fit.summary())


def test_news_adds_up_to_the_revision(growth):
    pytest.importorskip("statsmodels")
    from pymetrix import DynamicFactor

    quarterly, monthly = growth
    february = DynamicFactor(monthly=["gr", "gip", "gemp", "pmi"], quarterly=["gy"]).fit(quarterly, monthly.as_of("2025-02-15"))
    april = february.update(quarterly, monthly.as_of("2025-04-15"))  # same parameters, two more months of figures
    news = april.news(february, "gy")
    assert news.before == pytest.approx(february.nowcast("gy").loc["2025Q1"])
    assert news.after == pytest.approx(april.nowcast("gy").loc["2025Q1"])
    assert news.after - news.before == pytest.approx(news.details["impact"].sum() + news.revisions, rel=1e-9, abs=1e-9)
    assert abs(news.after) < 50 and abs(news.before) < 50  # sensible growth rates, not a diverged estimate
    assert set(news.by_series.index) <= {"gr", "gip", "gemp", "pmi"} and news.period == "2025Q1"
    assert "before" in str(news.summary())
    # the update's provenance is of the newer figures, through the statsmodels adapter; its parameters are February's
    assert april.provenance.data_fingerprint != february.provenance.data_fingerprint
    assert april.provenance.method["parameters"].startswith(f"estimated on {february.provenance.sample[0]}")
    assert list(april.provenance.options["stages"]) == ["apply"]  # the fitted model applied, not a new fit
    assert february.provenance.backend["wraps"].endswith("DynamicFactorMQ")
    assert february.provenance.options["stages"]["fit"] == {"disp": False, "maxiter": 500}


def test_principal_components_by_hand(growth):
    from pymetrix import PCA

    _, monthly = growth
    names = ["gr", "gip", "gemp", "pmi"]
    pca = PCA(names, n=2).fit(monthly)
    frame = monthly.frame[names].dropna()
    z = (frame - frame.mean()) / frame.std(ddof=1)
    values, vectors = np.linalg.eig(np.corrcoef(z.to_numpy(), rowvar=False))
    order = np.argsort(values)[::-1]
    np.testing.assert_allclose(pca.eigenvalues.to_numpy(), values[order], rtol=1e-10)
    for j in range(2):  # the same directions (up to sign) and the same scores
        assert abs(abs(np.dot(pca.loadings.iloc[:, j], vectors[:, order[j]])) - 1) < 1e-10
    assert abs(pca.scores["pc1"].corr(pd.Series(z.to_numpy() @ vectors[:, order[0]], index=frame.index))) == pytest.approx(1)
    assert pca.explained.sum() == pytest.approx(1) and (pca.loadings.abs().idxmax().map(lambda n: pca.loadings.loc[n]) is not None)
    np.testing.assert_allclose(pca.transform(monthly).dropna().to_numpy(), pca.scores.to_numpy(), atol=1e-10)


def test_dialogs_and_views(growth, tmp_path):
    pytest.importorskip("statsmodels")
    from pymetrix.commands import FitBridge, FitDynamicFactor, Session
    from pymetrix.testing import check_project
    from pymetrix.views import titles

    quarterly, monthly = growth
    s = Session()
    s.objects["quarterly"], s.objects["monthly"] = quarterly, monthly
    check_project(s, FitBridge("gy ~ gr + pmi", low="quarterly", high="monthly"), tmp_path)
    assert "bridge = Bridge(formula='gy ~ gr + pmi').fit(quarterly, monthly)" in s.script()
    check_project(s, FitDynamicFactor("gr, gip, gemp, pmi", high="monthly", quarterly="gy", low="quarterly"), tmp_path)
    graph = importlib.util.find_spec("matplotlib") is not None  # no Graph tab without matplotlib
    assert titles(s["bridge"])[0] == "Summary" and titles(s["dfm"])[:2] == (["Summary", "Graph"] if graph else ["Summary", "Factors"])
    assert FitDynamicFactor("gr", high="monthly", quarterly="gy").validate(s) == ["choose the quarterly data for the quarterly series"]
