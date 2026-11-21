"""Results keep their numbers as named fields, and know how they were made: warnings, convergence, data, seed."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, Dataset, Equation
from pymetrix._capture import capture, fingerprint


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


def test_hac_lags_used_are_a_statistic(data):
    auto = Equation("c ~ y + c[-1]").fit(data, method=OLS(cov="hac"))
    assert auto.stats["hac_lags"] == 3  # floor(4 (99/100)^(2/9)) for 99 observations
    given = Equation("c ~ y + c[-1]").fit(data, method=OLS(cov="hac", hac_lags=6))
    assert given.stats["hac_lags"] == 6 and "6 lags" in str(given.summary())
    assert "hac_lags" not in Equation("c ~ y").fit(data).stats


@pytest.mark.filterwarnings("ignore:an old argument:DeprecationWarning")  # re-issued, as intended
def test_warnings_are_kept_on_the_result(data):
    @dataclass(frozen=True)
    class Noisy:
        label = "noisy"

        def estimate(self, eq, data, **kw):
            warnings.warn("the optimiser stopped early", RuntimeWarning)
            warnings.warn("an old argument", DeprecationWarning)
            return OLS().estimate(eq, data, **kw)

    with pytest.warns(RuntimeWarning, match="stopped early"):  # still shown as usual
        fit = Equation("c ~ y").fit(data, method=Noisy())
    assert fit.provenance.warnings == ("RuntimeWarning: the optimiser stopped early",)  # deprecations are about libraries, not the result
    assert "Warnings: RuntimeWarning" in str(fit.summary())
    assert Equation("c ~ y").fit(data).provenance.warnings == ()


@pytest.mark.filterwarnings("ignore::UserWarning")  # the captured warning is re-issued, as intended
def test_data_fingerprints(data):
    a, b = Equation("c ~ y").fit(data), Equation("c ~ y").fit(data)
    assert a.provenance.data_fingerprint == b.provenance.data_fingerprint and len(a.provenance.data_fingerprint) == 16
    shorter = Equation("c ~ y").fit(data, span=("2010Q1", None))
    changed = Equation("c ~ y").fit(data.with_series(c="c * 1.0001"))
    assert len({a.provenance.data_fingerprint, shorter.provenance.data_fingerprint, changed.provenance.data_fingerprint}) == 3
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    assert fingerprint(frame) == fingerprint(frame.copy()) != fingerprint(frame.assign(x=[1.0, 2.5]))
    with capture() as caught:
        warnings.warn("x", UserWarning)
    assert caught == ["UserWarning: x"]


def test_ardl_numbers_are_fields(data):
    pytest.importorskip("statsmodels")
    from pymetrix import ARDL

    fit = ARDL("lc ~ ly", lags=2, order=2).fit(data.with_series(lc="log(c)", ly="log(y)"))
    lr = fit.long_run_coefficients
    assert list(lr.columns) == ["coefficient", "std err"] and np.allclose(lr.to_numpy(), fit.long_run().to_frame().to_numpy(dtype=float))
    assert "lc[-1]" in fit.ecm_params.names and np.all(np.isfinite(fit.ecm_params.se))  # statsmodels' lc.L1, in pymetrix's syntax
    with pytest.warns(DeprecationWarning, match="now called 'lc\\[-1\\]'"):
        assert fit.ecm_params["lc.L1"] == fit.ecm_params["lc[-1]"]
    assert f"R-squared: {fit.stats['r2']:.4f}" in str(fit.summary()) and fit.stats["nobs"] == fit.backend_result.nobs
    prov = fit.provenance
    assert prov.sample[0] == str(fit.resid.index[0]) and prov.nobs == len(fit.resid) and prov.method["backend"].startswith("statsmodels")
    assert prov.data_fingerprint and prov.converged is None


def test_markov_and_midas_record_convergence(data):
    pytest.importorskip("statsmodels")
    from pymetrix import MIDAS, MarkovSwitching

    growth = data.with_series(gy="100*dlog(y)")
    ms = MarkovSwitching("gy ~ 1", search=2, seed=3).fit(growth)
    assert ms.provenance.converged in (True, False) and ms.provenance.seed == 3 and ms.provenance.sample[0] == "2000Q2"
    months = pd.period_range("2000-01", "2024-12", freq="M", name="time")
    rng = np.random.default_rng(0)
    high = Dataset.from_frame(pd.DataFrame({"ip": rng.normal(size=len(months))}, index=months))
    quarters = pd.period_range("2000Q1", "2024Q4", freq="Q", name="time")
    low = Dataset.from_frame(pd.DataFrame({"gdp": rng.normal(size=len(quarters))}, index=quarters))
    assert MIDAS("gdp ~ ip", lags=6).fit(low, high).provenance.converged is True
    assert MIDAS("gdp ~ ip", lags=6, weights="unrestricted").fit(low, high).provenance.converged is None


def test_forecast_tables_have_structured_results(tmp_path, data):
    from pymetrix import VAR, ForecastRound, ForecastStore, backtest

    d = data.with_series(ly="log(y)", lc="log(c)")
    var = VAR(["ly", "lc"], lags=1).fit(d)
    june = d.select(span=(None, "2024Q2"))
    store = ForecastStore(tmp_path / "f")
    store.save(ForecastRound.from_forecast("a", VAR(["ly", "lc"], lags=1).fit(june).forecast(june, ("2024Q3", "2025Q4")), data=june))
    store.save(ForecastRound.from_forecast("b", var.forecast(d, ("2025Q1", "2025Q4")), data=d))
    rev = store.revisions("a", "b", ["ly"])
    assert isinstance(rev.index, pd.PeriodIndex) and "ly revision" in rev.columns
    shown = store.compare("a", "b", ["ly"]).to_frame()["ly revision"].astype(float).to_numpy()
    assert np.allclose(shown, rev["ly revision"].to_numpy(), equal_nan=True)
    assert list(store.scores("ly", d).columns) == ["n", "RMSE", "MAE", "mean error"]
    bt = backtest(var, d, ("2020Q1", "2022Q4"), horizon=2)
    scores = bt.scores(["ly"])
    assert list(scores.index.names) == ["variable", "horizon"]
    assert np.allclose(scores["RMSE"].to_numpy(), bt.accuracy(["ly"]).to_frame()["RMSE"].astype(float).to_numpy())


def test_views_by_name_and_the_producing_step(data):
    from pymetrix.commands import FitEquation, Session
    from pymetrix.views import titles, view

    fit = Equation("c ~ y").fit(data)
    assert titles(fit)[0] == "Summary" and view(fit, "Coefficients").to_frame().shape[0] == 2
    with pytest.raises(KeyError, match="no view"):
        view(fit, "Nope")
    frame = view(fit, "Provenance").to_frame()
    prov = dict(zip(frame.iloc[:, 0], frame.iloc[:, 1]))
    assert prov["warnings"] == "none" and "data fingerprint" in prov
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y", name="cons"))
    assert s.producer("cons").code.startswith("cons = Equation('c ~ y')")
