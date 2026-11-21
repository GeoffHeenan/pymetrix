"""The equation result window's correlogram tabs, and the Hannan-Quinn criterion among an equation's statistics.

- HQ = -2 log L + 2k ln ln n: least squares against statsmodels' log likelihood by the formula; regression with AR(1)
  errors against statsmodels ARIMA's hqic (its own computation), to 1e-8; shown in the summary and Statistics.
- A fit to a time series gets a Correlogram tab (its residuals', standardised after ARMA errors, as fit.correlogram)
  and a chart; with a GARCH variance, a tab for the squared residuals too; cross-section fits get none."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.estimation.timeseries import ARMA, GARCH
from pymetrix.views import sections


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    x, e = rng.normal(size=n), rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + e[t]
    return Dataset.from_frame(pd.DataFrame({"y": 1 + 0.5 * x + u, "x": x},
                                           index=pd.period_range("1950Q1", periods=n, freq="Q", name="time")))


def test_hannan_quinn_against_statsmodels(data):
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA

    y, x = data.frame.y.to_numpy(), data.frame.x.to_numpy()
    n = len(y)
    fit = Equation("y ~ x").fit(data)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    assert fit.stats["hq"] == pytest.approx(-2 * ols.llf + 2 * 2 * np.log(np.log(n)), rel=1e-10)
    arma = Equation("y ~ x", errors=ARMA(1, 0)).fit(data)
    ref = ARIMA(y, exog=x, order=(1, 0, 0), trend="c").fit()
    assert arma.stats["hq"] == pytest.approx(ref.hqic, rel=1e-8)
    assert "Hannan-Quinn criterion" in str(fit.summary())
    statistics = dict(sections(fit))["Statistics"]
    assert "hq" in str(statistics) and "ln ln n (HQ)" in " ".join(statistics.notes)


def test_correlogram_tabs(data):
    fit = Equation("y ~ x").fit(data)
    tabs = dict(sections(fit))
    assert np.allclose(np.array(tabs["Correlogram"].rows, float), np.array(fit.correlogram(20).rows, float), equal_nan=True)
    assert "Correlogram graph" in tabs and "Correlogram of squares" not in tabs
    arma = Equation("y ~ x", errors=ARMA(1, 0)).fit(data)
    shown = np.array(dict(sections(arma))["Correlogram"].rows, float)  # standardised, ARMA degrees of freedom
    assert np.allclose(shown, np.array(arma.correlogram(20).rows, float), equal_nan=True)
    garch = dict(sections(Equation("y ~ x", variance=GARCH()).fit(data)))
    assert "Correlogram of squares" in garch and "squared residuals" in garch["Correlogram of squares"].title


def test_short_samples_and_cross_sections(data):
    short = Equation("y ~ x").fit(Dataset.from_frame(data.frame.iloc[:24]))
    assert len(dict(sections(short))["Correlogram"].rows) == 6  # a quarter of the observations
    cross = Dataset.from_frame(data.frame.reset_index(drop=True).rename_axis("id").reset_index(), entity="id")
    assert "Correlogram" not in dict(sections(Equation("y ~ x").fit(cross)))
