import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.filters.bk_filter import bkfilter  # noqa: E402
from statsmodels.tsa.filters.cf_filter import cffilter  # noqa: E402
from statsmodels.tsa.filters.hp_filter import hpfilter  # noqa: E402

from pymetrix import HP, BaxterKing, ChristianoFitzgerald, Dataset, Hamilton, MovingAverage
from pymetrix.commands import Filter, Session
from pymetrix.filters import hp_trend


@pytest.fixture
def gdp() -> Dataset:
    rng = np.random.default_rng(0)
    n = 160
    y = np.cumsum(0.005 + rng.normal(0, 0.01, n)) + 0.02 * np.sin(np.arange(n) / 5)
    return Dataset.from_frame(pd.DataFrame({"ly": y}, index=pd.period_range("1985Q1", periods=n, freq="Q", name="time")))


def test_match_statsmodels(gdp):
    y = gdp["ly"].to_numpy()
    hp = gdp.filter("ly", HP())
    cycle, trend = hpfilter(y, 1600)
    assert np.allclose(hp["ly_cycle"], cycle) and np.allclose(hp["ly_trend"], trend)
    assert np.allclose(hp["ly_trend"] + hp["ly_cycle"], gdp["ly"])
    bk = gdp.filter("ly", BaxterKing())["ly_cycle"].to_numpy()
    assert np.allclose(bk[12:-12], bkfilter(y, 6, 32, 12)) and np.isnan(bk[:12]).all() and np.isnan(bk[-12:]).all()
    cf = gdp.filter("ly", ChristianoFitzgerald())["ly_cycle"]
    assert np.allclose(cf, cffilter(y, 6, 32, True)[0])


def test_one_sided_hp_is_real_time(gdp):
    one = gdp.filter("ly", HP(one_sided=True))["ly_trend"].to_numpy()
    y = gdp["ly"].to_numpy()
    for t in (10, 80, 159):
        assert np.isclose(one[t], hp_trend(y[: t + 1], 1600)[-1])
    two = gdp.filter("ly", HP())["ly_trend"].to_numpy()
    assert np.isclose(one[-1], two[-1]) and not np.isclose(one[80], two[80])


def test_hamilton_is_a_regression(gdp):
    out = gdp.filter("ly", Hamilton())
    y = gdp["ly"].to_numpy()
    rows = np.arange(11, len(y))
    X = sm.add_constant(np.column_stack([y[rows - 8 - i] for i in range(4)]))
    assert np.allclose(out["ly_cycle"].to_numpy()[rows], sm.OLS(y[rows], X).fit().resid)
    assert out["ly_cycle"].isna().sum() == 11


def test_defaults_follow_frequency():
    annual = Dataset.from_frame(pd.DataFrame({"y": np.arange(40.0) ** 1.5}, index=pd.period_range("1980", periods=40, freq="Y", name="time")))
    out = annual.filter("y", HP())
    assert out.variables["y_trend"].derived_from == "HP(lambda=100) trend of y"
    assert np.allclose(out["y_trend"], hp_trend(annual["y"].to_numpy(), 100))
    assert annual.filter("y", HP(power=4)).variables["y_cycle"].derived_from.startswith("HP(lambda=6.25)")
    assert annual.filter("y", MovingAverage(3))["y_trend"].iloc[1] == pytest.approx(np.mean(annual["y"].iloc[:3]))


def test_panels_filter_within_entity(panel):
    out = panel.filter("x", HP())
    for country in ("C00", "C05"):
        own = panel.select(country=country)["x"].to_numpy()
        assert np.allclose(out.select(country=country)["x_trend"], hp_trend(own, 100))


def test_missing_values(gdp):
    padded = gdp.with_series(ly=gdp["ly"].where(gdp["ly"].index > pd.Period("1985Q4", "Q")))
    out = padded.filter("ly", HP())
    assert out["ly_trend"].isna().sum() == 4  # leading gap stays missing
    holed = gdp.with_series(ly=gdp["ly"].where(gdp["ly"].index != pd.Period("2000Q1", "Q")))
    with pytest.raises(ValueError, match="unbroken"):
        holed.filter("ly", HP())


def test_filter_command(gdp):
    session = Session()
    session.objects["data"] = gdp
    session.run(Filter("ly", method="HP (one-sided)", settings="lamb=400.0"))
    session.run(Filter("ly", method="Hamilton", keep_updated=False, name="static"))
    script = session.script()
    assert "data = data.filter('ly', HP(lamb=400.0, one_sided=True))" in script
    assert "static = data.filter('ly', Hamilton(), derived=False)" in script
    assert "ly_trend" in session["data"].derived and not session["static"].derived
    ns = {"data": gdp}
    exec(script, ns)
    assert np.allclose(ns["data"]["ly_trend"], session["data"]["ly_trend"])
