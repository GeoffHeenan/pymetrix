import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import recursive_olsresiduals  # noqa: E402

from pymetrix import Dataset, Equation, forecast_accuracy
from pymetrix.commands import FitRolling, Session


def _series(y, x) -> Dataset:
    index = pd.period_range("1980Q1", periods=len(y), freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=index))


@pytest.fixture
def stable() -> Dataset:
    rng = np.random.default_rng(21)
    x = rng.normal(size=120)
    return _series(1 + 0.5 * x + rng.normal(scale=0.5, size=120), x)


@pytest.fixture
def broken() -> Dataset:
    """The slope jumps from 0.5 to 1.5 two-thirds of the way through."""
    rng = np.random.default_rng(22)
    x = rng.normal(size=120)
    slope = np.where(np.arange(120) < 80, 0.5, 1.5)
    return _series(1 + slope * x + rng.normal(scale=0.5, size=120), x)


def test_rolling_windows_match_direct_fits(stable):
    eq = Equation("y ~ x")
    roll = eq.fit_rolling(stable, window=40)
    assert len(roll.fits) == 81 and not roll.failed
    end = pd.Period("1995Q4", "Q")
    direct = eq.fit(stable, span=("1986Q1", "1995Q4"))
    assert np.allclose(roll.params.frame.loc[end].to_numpy(), direct.params.values)
    assert np.allclose(roll.se.frame.loc[end].to_numpy(), direct.params.se)
    assert roll.stats.frame.loc[end, "nobs"] == 40


def test_recursive_ends_at_the_full_sample(stable):
    eq = Equation("y ~ x")
    rec = eq.fit_recursive(stable, min_obs=20)
    assert len(rec.fits) == 101
    assert np.allclose(rec.params.frame.iloc[-1].to_numpy(), eq.fit(stable).params.values)
    assert rec.stats.frame["nobs"].iloc[0] == 20
    assert "Recursive" in str(rec.summary())
    assert {"±2 s.e. lower", "±2 s.e. upper"} <= set(rec.plot(["x"]).data().columns)


def test_pseudo_out_of_sample_forecasts(stable):
    roll = Equation("y ~ x").fit_rolling(stable, window=40)
    fc = roll.forecast(stable, horizon=1)
    assert len(fc) == 80  # the last window has nothing left to forecast
    first = fc.frame.iloc[0]
    assert first["origin"] == "1989Q4"
    own = roll.fits[pd.Period("1989Q4", "Q")].forecast(stable, span=("1990Q1", "1990Q1")).frame.iloc[0]
    assert np.isclose(first["forecast"], own["forecast"])
    assert np.allclose(fc["error"], fc["actual"] - fc["forecast"])
    assert forecast_accuracy(fc["actual"], fc["forecast"]).rows[0][0] == 80


def test_recursive_residuals_match_statsmodels(stable):
    fit = Equation("y ~ x").fit(stable)
    rr = fit.recursive_residuals()
    X = fit.design.to_numpy()
    y = X @ fit.params.values + fit.resid.to_numpy()
    ref = recursive_olsresiduals(sm.OLS(y, X).fit())
    m = len(rr)
    assert m == 118
    assert np.allclose(rr["recursive_resid"], np.asarray(ref[4])[-m:])
    assert np.allclose(rr["cusum"], np.asarray(ref[5])[-m:])
    assert np.allclose(rr["cusum_upper"], 0.948 * (np.sqrt(m) + 2 * np.arange(1, m + 1) / np.sqrt(m)))


def test_cusum_tests_detect_breaks(stable, broken):
    assert not Equation("y ~ x").fit(stable).test("cusum").reject()
    assert not Equation("y ~ x").fit(stable).test("cusumsq").reject()
    rng = np.random.default_rng(23)
    x = rng.normal(size=120)
    drift = _series(np.r_[np.zeros(60), np.linspace(0, 6, 60)] + 0.5 * x + rng.normal(scale=0.3, size=120), x)
    assert Equation("y ~ x").fit(drift).test("cusum").reject()
    scale = np.where(np.arange(120) < 60, 0.2, 1.5)
    noisy = _series(1 + 0.5 * x + scale * rng.normal(size=120), x)
    sq = Equation("y ~ x").fit(noisy).test("cusumsq")
    # Durbin's (1969) two-sided 5% value for about 59 pairs is close to 0.17
    assert sq.reject() and 0.15 < sq.critical_values["5%"] < 0.19
    roll = Equation("y ~ x").fit_rolling(broken, window=30)  # the slope break shows in rolling estimates
    assert roll.params.frame["x"].iloc[0] < 0.8 < 1.2 < roll.params.frame["x"].iloc[-1]


def test_rolling_command(stable):
    session = Session()
    session.objects["data"] = stable
    session.run(FitRolling("y ~ x", window=40, name="roll"))
    session.run(FitRolling("y ~ x", min_obs=30, name="rec"))
    script = session.script()
    assert "roll = Equation('y ~ x').fit_rolling(data, 40, method=OLS())" in script
    assert "rec = Equation('y ~ x').fit_recursive(data, min_obs=30, method=OLS())" in script
    ns = {"data": stable}
    exec(script, ns)
    assert len(ns["rec"].fits) == 91
