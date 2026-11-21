"""MIDAS regressions: lag polynomials, U-MIDAS against OLS, nowcasts."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import MIDAS, Dataset
from pymetrix.midas import lag_weights


@pytest.fixture(scope="module")
def data():
    """Quarterly gdp = 1 + 2 * (exp-Almon weighted 12 months of ip) + noise; ip to December 2024, gdp to 2024Q3."""
    rng = np.random.default_rng(1)
    months = pd.period_range("2000-01", "2024-12", freq="M", name="time")
    ip = np.zeros(len(months))
    for t in range(1, len(months)):
        ip[t] = 0.5 * ip[t - 1] + rng.normal()
    quarters = pd.period_range("2000Q1", "2024Q3", freq="Q", name="time")
    w = lag_weights("exp_almon", (0.2, -0.05), 12)
    x = pd.Series(ip, index=months)
    gdp = [1 + 2 * np.array([x.get(q.asfreq("M", how="end") - j, np.nan) for j in range(12)]) @ w + rng.normal(0, 0.3) for q in quarters]
    return Dataset.from_frame(pd.DataFrame({"gdp": gdp}, index=quarters)), Dataset.from_frame(pd.DataFrame({"ip": ip}, index=months)), w


def test_lag_weights():
    for kind, theta in (("exp_almon", (0.2, -0.05)), ("beta", (0.0, 1.0))):
        w = lag_weights(kind, theta, 12)
        assert w.sum() == pytest.approx(1) and np.all(w >= 0)
    assert np.allclose(lag_weights("exp_almon", (0, 0), 6), 1 / 6)


def test_parametric_midas_recovers_the_weights(data):
    low, high, true_w = data
    fit = MIDAS("gdp ~ ip", lags=12).fit(low, high)
    assert fit.params["ip"] == pytest.approx(2, abs=0.2) and fit.params["const"] == pytest.approx(1, abs=0.1)
    assert np.corrcoef(fit.weights["ip"], true_w)[0, 1] > 0.99 and fit.weights["ip"].sum() == pytest.approx(1)
    assert fit.stats["nobs"] == 96  # the first three quarters lack twelve months of history
    beta = MIDAS("gdp ~ ip", lags=12, weights="beta").fit(low, high)
    assert np.corrcoef(beta.weights["ip"], true_w)[0, 1] > 0.95
    assert "exp_almon weights, 12 high-frequency lags" in str(fit.summary())


def test_unrestricted_midas_is_least_squares(data):
    sm = pytest.importorskip("statsmodels.api")
    low, high, _ = data
    u = MIDAS("gdp ~ ip", lags=6, weights="unrestricted").fit(low, high)
    _, X = u.spec.design(low, high)
    y = low["gdp"].to_numpy()
    ok = np.all(np.isfinite(X["ip"]), axis=1) & np.isfinite(y)
    ref = sm.OLS(y[ok], sm.add_constant(X["ip"][ok])).fit()
    assert np.allclose(u.params.values, ref.params) and np.allclose(u.params.se, ref.bse)
    assert list(u.params.names)[:3] == ["const", "ip[0]", "ip[-1]"]


def test_nowcasts_and_data_short_at_the_end(data):
    low, high, _ = data
    fit = MIDAS("gdp ~ ip", lags=12).fit(low, high)
    now = fit.nowcast(low, high)
    assert str(now.index[-1]) == "2024Q4"  # beyond the quarterly data: the months of 2024Q4 are known
    assert np.allclose(now.loc[fit.fitted.index], fit.fitted)
    early = MIDAS("gdp ~ ip", lags=12, high_lag=2).fit(low, high)  # as if only each quarter's first month were known
    october = high.select(span=(None, "2024-10"))
    assert str(fit.nowcast(low, october).index[-1]) == "2024Q3"  # the full-quarter model needs December for 2024Q4
    assert str(early.nowcast(low, october).index[-1]) == "2024Q4"  # this one needs only October
    assert early.stats["r2"] < fit.stats["r2"]  # less information, a worse fit
    with_ar = MIDAS("gdp ~ ip", lags=12, ar=1).fit(low, high)
    assert "gdp[-1]" in with_ar.params.names and str(with_ar.nowcast(low, high).index[-1]) == "2024Q4"


def test_errors_and_dialog(data, tmp_path):
    from pymetrix.commands import FitMIDAS, Session, load_project, save_project

    low, high, _ = data
    with pytest.raises(KeyError, match="no indicator"):
        MIDAS("gdp ~ sales").fit(low, high)
    with pytest.raises(ValueError, match="series names"):
        MIDAS("gdp ~ log(ip)").fit(low, high)
    with pytest.raises(ValueError, match="2 or more"):
        MIDAS("gdp ~ ip", lags=1)
    s = Session()
    s.objects["quarterly"], s.objects["data"] = low, high
    s.run(FitMIDAS("gdp ~ ip", "quarterly", lags=12))
    assert "midas = MIDAS(formula='gdp ~ ip').fit(quarterly, data)" in s.script()
    reopened = load_project(save_project(s, tmp_path / "midas.pymx"))
    assert np.allclose(reopened["midas"].params.values, s["midas"].params.values)
