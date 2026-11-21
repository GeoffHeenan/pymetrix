"""One time-series abstraction (plan item 6) and results as their own record (plan item 7, part 1): pymetrix's
axes and names on every user-facing field; a library's own object only in ``backend_result``, never saved."""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import ARDL, VECM, Dataset, Equation, MarkovSwitching
from pymetrix.adapters.base import to_axes
from pymetrix.errors import PyMetrixError, SpecificationError
from pymetrix.testing import check_axes

sys.path.insert(0, str(Path(__file__).parent))
from _gallery import BUILDERS  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return Dataset.example().with_series(lc="log(c)", ly="log(y)")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_result_uses_pymetrix_axes(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = BUILDERS[name]()
    check_axes(obj)


def test_wrapped_results_use_pymetrix_axes(data):
    pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    returns = Dataset.from_frame(pd.DataFrame({"r": rng.standard_t(6, size=400)}, index=pd.period_range("1990-01", periods=400, freq="M")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fit in (Equation("r ~ 1", variance=pymetrix.EGARCH()).fit(returns), ARDL("lc ~ ly", lags=1, order=1).fit(data),
                    VECM(["lc", "ly"], lags=1, rank=1).fit(data)):
            check_axes(fit)


def test_the_axis_check_catches_library_dates():
    from pymetrix.estimation.params import point

    resid = pd.Series([0.1, -0.1, 0.2], index=pd.date_range("2000-01-01", periods=3, freq="QS"))
    bad = pymetrix.FittedEquation(Equation("y ~ 1"), point(["const"], [1.0]), pymetrix.OLS(), resid=resid)
    with pytest.raises(AssertionError, match="DatetimeIndex"):
        check_axes(bad)
    positions = pymetrix.FittedEquation(Equation("y ~ 1"), point(["const"], [1.0]), pymetrix.OLS(), resid=resid.reset_index(drop=True))
    with pytest.raises(AssertionError, match="unnamed RangeIndex"):
        check_axes(positions)


def test_to_axes_aligns_dates_by_label_and_positions_by_length():
    periods = pd.period_range("2000Q1", periods=4, freq="Q", name="time")
    dated = pd.Series([1.0, 2.0, 3.0], index=pd.PeriodIndex(periods[1:]).to_timestamp())
    placed = to_axes(dated, periods)
    assert isinstance(placed.index, pd.PeriodIndex) and np.isnan(placed.iloc[0]) and placed.iloc[1:].tolist() == [1.0, 2.0, 3.0]
    assert to_axes(np.arange(4.0), periods, name="x").index.equals(periods)
    with pytest.raises(SpecificationError, match="3 rows for 4 periods"):
        to_axes(np.arange(3.0), periods)


def test_ardl_names_are_pymetrix_expressions_and_old_names_still_work(data):
    fit = ARDL("lc ~ ly", lags=2, order=1).fit(data)
    assert fit.params.names == ("const", "lc[-1]", "lc[-2]", "ly", "ly[-1]")
    assert {"lc[-1]", "d(lc)[-1]", "d(ly)"} <= set(fit.ecm_params.names)
    with pytest.warns(DeprecationWarning, match="now called 'lc\\[-1\\]'"):
        assert fit.params["lc.L1"] == fit.params["lc[-1]"]
    assert list(fit.summary().index) == list(fit.params.names)


def test_ardl_and_markov_are_their_own_record(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ardl = ARDL("lc ~ ly", lags=2, order=1).fit(data)
        markov = MarkovSwitching("g ~ 1").fit(data.with_series(g="100*dlog(y)"))
    for fit in (ardl, markov):
        back = pickle.loads(pickle.dumps(fit))
        assert back.backend_result is None and fit.backend_result is not None  # the library object is never saved
        assert np.allclose(back.params.values, fit.params.values)
        assert str(back.summary()) == str(fit.summary())
        back.plot()
    back = pickle.loads(pickle.dumps(ardl))
    assert back.bounds_test().statistic == ardl.bounds_test().statistic  # the default case is part of the record
    with pytest.raises(PyMetrixError, match="estimate it again"):
        back.bounds_test(case=2)
    assert markov.stats.keys() >= {"nobs", "loglike", "aic"}
    with pytest.warns(DeprecationWarning, match="use .backend_result"):
        assert markov.result is markov.backend_result


def test_equations_keep_the_library_object_in_backend_result(data):
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(3)
    n = 300
    z1, z2, u = rng.normal(size=(3, n))
    x = z1 + z2 + u + rng.normal(size=n)
    iv_data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": 1 + 2 * x + u, "x": x, "z1": z1, "z2": z2}), entity="id")
    fit = Equation("y ~ x").fit(iv_data, method=pymetrix.LIML(endog=["x"], instruments=["z1", "z2"]))
    assert type(fit.backend_result).__module__.startswith("linearmodels") and "linearmodels" not in fit.extra
    with pytest.warns(DeprecationWarning, match="use .backend_result"):
        assert fit.extra["linearmodels"] is fit.backend_result
    back = pickle.loads(pickle.dumps(fit))
    assert back.backend_result is None and "first_stage_F" in back.extra
    with pytest.raises(KeyError):
        back.extra["linearmodels"]


def test_arch_forecasts_work_from_the_record(data):
    pytest.importorskip("arch")
    rng = np.random.default_rng(1)
    returns = Dataset.from_frame(pd.DataFrame({"r": rng.standard_t(6, size=500)}, index=pd.period_range("1980-01", periods=500, freq="M")))
    fit = Equation("r ~ 1", variance=pymetrix.GARCH(dist="t")).fit(returns, span=("1980-01", "2019-12"))
    back = pickle.loads(pickle.dumps(fit))
    assert back.backend_result is None
    a = fit.forecast(returns, ("2020-01", "2020-06"), dynamic=True)
    b = back.forecast(returns, ("2020-01", "2020-06"), dynamic=True)
    assert np.allclose(a["variance"], b["variance"])


def test_results_that_need_the_library_keep_it_and_are_not_cached_in_projects(tmp_path, data, monkeypatch):
    from pymetrix.commands import FitVECM, LoadExample, Session, save_project
    from pymetrix.commands import project as project_module

    vecm = VECM(["lc", "ly"], lags=1, rank=1).fit(data)
    assert vecm.needs_backend and pickle.loads(pickle.dumps(vecm)).backend_result is not None
    monkeypatch.setattr(project_module, "EXPENSIVE", 0.0)  # every step counts as slow, so would be cached
    s = Session()
    s.run(LoadExample())
    s.run_code("data = data.with_series(lc='log(c)', ly='log(y)')")
    s.run(FitVECM("lc, ly", lags=1, rank="1", name="vecm"))
    path = save_project(s, tmp_path / "vecm.pymx")
    import zipfile

    with zipfile.ZipFile(path) as z:
        cached = [n for n in z.namelist() if n.startswith("cache/")]
    steps = [i for i, r in enumerate(s.history) if "vecm" in r.writes]
    assert steps and not any(n.startswith(f"cache/{steps[0]}") for n in cached)


def test_from_statsmodels_places_results_on_periods(data):
    sm = pytest.importorskip("statsmodels.api")
    from pymetrix.interop import from_statsmodels

    frame = data.to_frame(["c", "y"])
    res = sm.OLS(frame["c"].to_numpy(), sm.add_constant(frame["y"].to_numpy())).fit()
    with pytest.warns(UserWarning, match="pass index="):
        from_statsmodels(res)
    fit = from_statsmodels(res, data, index=frame.index)
    assert isinstance(fit.resid.index, pd.PeriodIndex) and fit.provenance.sample == ("2000Q1", "2024Q4")
