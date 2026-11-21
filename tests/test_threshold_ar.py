"""Threshold autoregressions and linearity tests (Elliott and Timmermann 2016, chapter 8): estimates against a brute-force
search with statsmodels, recovery of a simulated threshold, Hansen's sup-F and Terasvirta's test against statsmodels'
regressions, simulated forecasts against an exact one-step forecast and an independent simulation, backtests, and the
dialog."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, ThresholdAR, backtest, terasvirta
from pymetrix.errors import SpecificationError

sm = pytest.importorskip("statsmodels.api")


def _setar(n=400, seed=1, threshold=1.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(2, n):
        if y[t - 1] <= threshold:
            y[t] = 0.5 + 0.6 * y[t - 1] - 0.2 * y[t - 2]
        else:
            y[t] = -0.4 + 0.3 * y[t - 1] + 0.1 * y[t - 2]
        y[t] += rng.normal(scale=0.5)
    idx = pd.period_range("1900Q1", periods=n, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=idx))


@pytest.fixture(scope="module")
def data():
    return _setar()


def test_least_squares_over_thresholds_and_delays_by_brute_force(data):
    fit = ThresholdAR("y", lags=2).fit(data)
    y_all = data["y"].to_numpy()
    y, X = y_all[2:], np.column_stack([np.ones(len(y_all) - 2), y_all[1:-1], y_all[:-2]])
    best = (np.inf, None, None)
    n = len(y)
    least = int(np.ceil(0.15 * n))
    for d in (1, 2):
        q = y_all[2 - d:len(y_all) - d]
        for gamma in np.unique(np.sort(q)[least - 1:n - least]):
            low = q <= gamma
            if low.sum() < least or (~low).sum() < least:
                continue
            ssr = sm.OLS(y[low], X[low]).fit().ssr + sm.OLS(y[~low], X[~low]).fit().ssr
            if ssr < best[0]:
                best = (ssr, d, gamma)
    assert fit.ssr == pytest.approx(best[0], rel=1e-10) and fit.delay == best[1] and fit.threshold == best[2]
    low = y_all[2 - fit.delay:len(y_all) - fit.delay] <= fit.threshold
    assert fit.low == pytest.approx(sm.OLS(y[low], X[low]).fit().params, rel=1e-9)
    assert fit.high == pytest.approx(sm.OLS(y[~low], X[~low]).fit().params, rel=1e-9)
    assert fit.ssr_linear == pytest.approx(sm.OLS(y, X).fit().ssr, rel=1e-10)


def test_a_simulated_threshold_is_recovered():
    fit = ThresholdAR("y", lags=2).fit(_setar(n=3000, seed=5))
    assert fit.delay == 1 and abs(fit.threshold - 1.0) < 0.1
    assert fit.low == pytest.approx([0.5, 0.6, -0.2], abs=0.08) and fit.sigma2 == pytest.approx(0.25, rel=0.1)


def test_linearity_tests(data):
    fit = ThresholdAR("y", lags=2).fit(data)
    hansen = fit.linearity_test(reps=99, seed=1)
    n = fit.n_low + fit.n_high
    assert hansen.statistic == pytest.approx(n * (fit.ssr_linear - fit.ssr) / fit.ssr) and hansen.p_value < 0.05
    rng = np.random.default_rng(4)
    linear = np.zeros(300)
    for t in range(1, 300):
        linear[t] = 0.5 * linear[t - 1] + rng.normal()
    lin = Dataset.from_frame(pd.DataFrame({"y": linear}, index=pd.period_range("1950Q1", periods=300, freq="Q", name="time")))
    assert ThresholdAR("y", lags=1).fit(lin).linearity_test(reps=99, seed=1).p_value > 0.05
    # Terasvirta: the F test of the auxiliary regression, against statsmodels' nested-model F test
    result = terasvirta(data, "y", lags=2, delay=1)
    y_all = data["y"].to_numpy()
    y, X = y_all[2:], np.column_stack([np.ones(len(y_all) - 2), y_all[1:-1], y_all[:-2]])
    e = sm.OLS(y, X).fit().resid
    q = y_all[1:-1]
    Z = np.column_stack([X] + [X[:, 1:] * q[:, None] ** j for j in (1, 2, 3)])
    f, p, df = sm.OLS(e, Z).fit().compare_f_test(sm.OLS(e, X).fit())
    assert result.statistic == pytest.approx(f, rel=1e-9) and result.p_value == pytest.approx(p, rel=1e-6) and result.p_value < 0.01


def test_simulated_forecasts(data):
    fit = ThresholdAR("y", lags=2).fit(data, span=(None, "1990Q4"))
    fc = fit.forecast(data, ("1991Q1", "1992Q4"), draws=40_000, seed=2).frame
    y = data["y"]
    x = np.array([1.0, y["1990Q4"], y["1990Q3"]])
    regime = fit.low if y[pd.Period("1990Q4", "Q") - (fit.delay - 1)] <= fit.threshold else fit.high
    sd = np.sqrt(fit.sigma2)
    assert fc["forecast"].iloc[0] == pytest.approx(float(x @ regime), abs=4 * sd / np.sqrt(40_000))  # exact one step ahead
    assert fc["se"].iloc[0] == pytest.approx(sd, rel=0.02)
    rng = np.random.default_rng(99)  # an independent simulation, written out here
    paths = np.empty((40_000, 8))
    h1, h2 = np.full(40_000, y["1990Q4"]), np.full(40_000, y["1990Q3"])
    for i in range(8):
        q = h1 if fit.delay == 1 else h2
        mean = np.where(q <= fit.threshold, fit.low[0] + fit.low[1] * h1 + fit.low[2] * h2, fit.high[0] + fit.high[1] * h1 + fit.high[2] * h2)
        new = mean + sd * rng.standard_normal(40_000)
        paths[:, i] = new
        h2, h1 = h1, new
    mc = 4 * paths.std(axis=0) / np.sqrt(40_000) * np.sqrt(2)
    assert np.all(np.abs(fc["forecast"].to_numpy() - paths.mean(axis=0)) < mc)


def test_backtests_score_the_simulated_draws(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThresholdAR("y", lags=2).fit(data, span=("1902Q1", "1990Q4"))
        bt = backtest(fit, data, ("1980Q1", "1981Q4"), horizon=2, label="tar")
    row = bt.errors.iloc[2]
    own = ThresholdAR("y", lags=2).fit(data, span=("1902Q1", row["origin"]))
    periods, paths = own.predictive_draws(data, (str(row["target"] - int(row["horizon"]) + 1), str(row["target"])))  # from the origin on
    assert row["density"] == "draws" and row["forecast"] == pytest.approx(paths[:, -1, 0].mean())


def test_errors_and_the_dialog(data):
    with pytest.raises(SpecificationError, match="lags"):
        ThresholdAR("y", lags=0)
    with pytest.raises(SpecificationError, match="no series"):
        ThresholdAR("z").fit(data)
    from pymetrix.commands import FitThresholdAR, Session

    session = Session()
    session.objects["data"] = data
    session.run(FitThresholdAR("y", lags=2))
    namespace = dict(session.namespace) | {"data": data}
    exec(session.history[-1].code, namespace)
    assert namespace["tar"].threshold == session["tar"].threshold and namespace["tar"].delay == session["tar"].delay
