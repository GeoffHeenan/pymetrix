import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.stattools import acf as sm_acf  # noqa: E402
from statsmodels.tsa.stattools import pacf as sm_pacf  # noqa: E402

from pymetrix import ARMA, Dataset, Equation
from pymetrix.commands import Correlogram, SelectOrder, Session
from pymetrix.stats import acf, pacf


@pytest.fixture
def ar_data() -> Dataset:
    rng = np.random.default_rng(11)
    n = 300
    eps = rng.normal(size=n + 1)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + eps[t + 1] + 0.3 * eps[t]
    x = rng.normal(size=n)
    index = pd.period_range("1950Q1", periods=n, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": 2 + 0.5 * x + u, "x": x}, index=index))


def test_acf_pacf_match_statsmodels(ar_data):
    y = ar_data["y"].to_numpy()
    assert np.allclose(acf(y, 12), sm_acf(y, nlags=12, fft=False))
    assert np.allclose(pacf(y, 12), sm_pacf(y, nlags=12, method="ldb"))


def test_correlogram_table(ar_data):
    table = ar_data.correlogram("y", lags=10)
    frame = table.to_frame()
    assert list(frame.columns) == ["lag", "AC", "PAC", "Q-stat", "p-value"]
    ref = acorr_ljungbox(ar_data["y"], lags=10)
    assert np.allclose(frame["Q-stat"], ref["lb_stat"])
    assert frame["AC"].iloc[0] > 0.5


def test_correlogram_band_is_1_96_over_root_t(ar_data):
    notes = " ".join(ar_data.correlogram("y", lags=10).notes)
    assert f"±{1.96 / np.sqrt(300):.3f}" in notes
    cross = " ".join(ar_data.crosscorrelogram("y", "x", lags=4).notes)
    assert f"±{1.96 / np.sqrt(300):.3f}" in cross


def test_residual_correlogram_adjusts_degrees_of_freedom(ar_data):
    fit = Equation("y ~ x", errors=ARMA(ar=1, ma=1)).fit(ar_data)
    frame = fit.correlogram(lags=8).to_frame()
    assert np.isnan(frame["p-value"].iloc[:2]).all()  # lags 1-2 have no degrees of freedom left
    assert frame["p-value"].iloc[-1] > 0.05
    ols = Equation("y ~ x").fit(ar_data)
    assert ols.correlogram(lags=8).to_frame()["p-value"].iloc[0] < 0.01  # the ARMA structure shows up


def test_select_order(ar_data):
    sel = ar_data.select_order("y", max_ar=2, max_ma=2, exog=["x"])
    frame = sel.table.to_frame()
    assert len(frame) == 9
    ref = sm.tsa.SARIMAX(ar_data["y"], exog=sm.add_constant(ar_data["x"]), order=(1, 0, 1)).fit(disp=0)
    assert np.isclose(frame.loc["ARMA(1,1)", "AIC"], ref.aic, rtol=1e-6)
    assert np.isclose(frame.loc["ARMA(1,1)", "BIC"], ref.bic, rtol=1e-6)
    ols = Equation("y ~ x").fit(ar_data)
    assert np.isclose(frame.loc["ARMA(0,0)", "loglike"], ols.stats["loglike"])
    assert sel.best["BIC"] in {(1, 1), (2, 0)}


def test_commands(ar_data):
    session = Session()
    session.objects["data"] = ar_data
    session.run(Correlogram(source="data", variable="y", lags=6))
    session.run(SelectOrder("y", max_ar=1, max_ma=1))
    script = session.script()
    assert "correlogram = data.correlogram('y', lags=6)" in script
    assert "orders = data.select_order('y', max_ar=1, max_ma=1)" in script
    exec(script, {"data": ar_data})
