"""ARDL models and the bounds test (statsmodels), with long-run multipliers checked by hand."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pymetrix import ARDL, Dataset  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(lc="log(c)", ly="log(y)", lg="log(g)")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_matches_statsmodels_and_the_long_run_by_hand(data):
    from statsmodels.tsa.ardl import ARDL as SMARDL

    fit = ARDL("lc ~ ly + lg", lags=2, order=2).fit(data)
    f = data.frame
    ref = SMARDL(f["lc"].set_axis(f.index.to_timestamp()), 2, f[["ly", "lg"]].set_axis(f.index.to_timestamp()), 2, trend="c").fit()
    assert np.allclose(fit.params.values, ref.params.to_numpy()) and np.allclose(fit.params.se, ref.bse.to_numpy())
    p = ref.params
    phi = sum(p[k] for k in p.index if k.startswith("lc.L"))
    long_run = fit.long_run().to_frame()
    for v in ("ly", "lg"):
        beta = sum(p[k] for k in p.index if k.startswith(f"{v}.L"))
        assert long_run.loc[v, "coefficient"] == pytest.approx(beta / (1 - phi), rel=1e-8)
    assert fit.order == (2, {"ly": 2, "lg": 2}) and "ARDL(2, ly 2, lg 2)" in str(fit.summary())


def test_the_bounds_test(data):
    fit = ARDL("lc ~ ly + lg", lags=2, order=2).fit(data)
    test = fit.bounds_test()
    assert test.statistic > test.critical_values["95% I(1)"] > test.critical_values["95% I(0)"]
    assert test.details["at 5%"].startswith("a level relationship") and test.details["case"] == 3
    rng = np.random.default_rng(0)
    walks = pd.DataFrame({"a": np.cumsum(rng.normal(size=200)), "b": np.cumsum(rng.normal(size=200))},
                         index=pd.period_range("1970Q1", periods=200, freq="Q", name="time"))
    unrelated = ARDL("a ~ b", lags=1, order=1).fit(Dataset.from_frame(walks)).bounds_test()
    assert unrelated.p_value > 0.05  # two independent random walks: no level relationship
    trend = ARDL("lc ~ ly", lags=1, order=1, trend="ct").fit(data).bounds_test()
    assert trend.details["case"] == 5


def test_order_selection_and_sample(data):
    chosen = ARDL("lc ~ ly", select="bic", max_lags=4, max_order=4).fit(data)
    lags, orders = chosen.order
    assert 0 <= lags <= 4 and 0 <= orders["ly"] <= 4 and "chosen by BIC" in str(chosen.summary())  # 0 lags of lc is a valid choice
    short = ARDL("lc ~ ly", lags=1, order=1).fit(data, span=("2010Q1", None))
    assert str(short.resid.index[0]) >= "2010Q1"
    with pytest.raises(ValueError, match="order names"):
        ARDL("lc ~ ly", order={"lw": 2}).fit(data)


def test_dialog_and_view(data):
    from pymetrix.commands import FitARDL, Session
    from pymetrix.ui.render import view_sections

    s = Session()
    s.objects["data"] = data
    s.run(FitARDL("lc ~ ly + lg", lags=2, order=2))
    assert "ardl = ARDL(formula='lc ~ ly + lg', lags=2, order=2).fit(data)" in s.script()
    titles = [t for t, _ in view_sections(s["ardl"])]
    assert titles[0] == "Summary" and {"Long run", "Bounds test", "Error correction"} <= set(titles)


def test_r_squared_is_centred(data):
    """statsmodels' ARDLResults.rsquared (0.15) is uncentred; the summary reports the usual centred R-squared."""
    import re

    import statsmodels.api as sm

    growth = data.with_series(gc="dlog(c)", gy="dlog(y)")  # growth rates: centred and uncentred R-squared differ
    fit = ARDL("gc ~ gy", lags=1, order=1).fit(growth)
    X = np.asarray(fit.model._x, float)  # the ARDL regressors, constant included
    y = np.asarray(fit.model._y, float)
    ols = sm.OLS(y, X).fit()
    uncentred = 1 - ols.ssr / (y @ y)
    shown = float(re.search(r"R-squared: ([0-9.]+)", str(fit.summary())).group(1))
    assert shown == pytest.approx(ols.rsquared, abs=1e-4) and abs(uncentred - ols.rsquared) > 1e-3  # far outside the tolerance
