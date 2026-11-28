"""Conformal prediction intervals: split conformal (Vovk, Gammerman and Shafer 2005; Lei et al. 2018) by hand, with the
finite-sample rank ceil((n + 1)(1 - alpha)) in exact arithmetic; its coverage in a simulation against its exact
expectation k / (n + 1); conformalised quantile regression (Romano, Patterson and Candès 2019) by hand and in a
simulation; adaptive conformal inference (Gibbs and Candès 2021) by hand and within its deterministic bound on the
long-run miss rate; the refit on the training periods against the library called directly; and the dialog."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, conformal
from pymetrix.errors import SpecificationError
from pymetrix.ml_conformal import conformal_quantile

FORMULA = "y ~ x1 + x2"


def _data(n, seed, scale=None):
    """y = 1 + x1 - 0.5 x2 + e, e ~ N(0, scale_t^2), on quarterly periods."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    s = np.ones(n) if scale is None else scale
    y = 1.0 + X[:, 0] - 0.5 * X[:, 1] + s * rng.normal(size=n)
    idx = pd.period_range("1960Q1", periods=n, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y}, index=idx))


def _exact_rank(n, alpha):
    return math.ceil((n + 1) * (1 - Fraction(alpha)))


def test_the_rank_is_ceil_n_plus_1_times_1_minus_alpha():
    scores = np.arange(19, 0, -1) * 0.1  # 1.9, 1.8, ..., 0.1
    q, k = conformal_quantile(scores, 0.1)  # (19 + 1)(0.9) = 18 exactly; in floating point 18.000000000000004
    assert k == _exact_rank(19, "0.1") == 18 and q == pytest.approx(1.8)
    for n, alpha in ((5, "0.1"), (40, "0.05"), (99, "0.2"), (39, "0.025"), (100, "0.01")):
        q, k = conformal_quantile(np.arange(1, n + 1, dtype=float), float(alpha))
        assert k == _exact_rank(n, alpha)
        assert q == (k if k <= n else math.inf)  # k > n: an unbounded interval
    assert conformal_quantile([1.0, 2.0], 1.0)[0] == -math.inf


def test_split_conformal_by_hand():
    data = _data(120, 0)
    fit = Equation(FORMULA).fit(data, span=(None, "1984Q4"))  # 100 periods: 75 to train, 25 to calibrate
    ci = conformal(fit, data, ("1985Q1", "1989Q4"), alpha=0.2)
    frame = data.frame
    A = np.column_stack([np.ones(120), frame["x1"], frame["x2"]])
    y = frame["y"].to_numpy()
    b = np.linalg.lstsq(A[:75], y[:75], rcond=None)[0]
    scores = np.abs(y[75:100] - A[75:100] @ b)
    k = _exact_rank(25, "0.2")  # ceil(26 * 0.8) = ceil(20.8) = 21
    q = np.sort(scores)[k - 1]
    assert ci.rank == k == 21 and ci.quantile == pytest.approx(q, rel=1e-12)
    np.testing.assert_allclose(ci.scores.to_numpy(), scores, rtol=1e-12)
    np.testing.assert_allclose(ci.frame["forecast"].to_numpy(), A[100:] @ b, rtol=1e-12)
    np.testing.assert_allclose(ci.frame["lower"].to_numpy(), A[100:] @ b - q, rtol=1e-12)
    np.testing.assert_allclose(ci.frame["upper"].to_numpy(), A[100:] @ b + q, rtol=1e-12)
    covered = (np.abs(y[100:] - A[100:] @ b) <= q).astype(float)
    np.testing.assert_array_equal(ci.frame["covered"].to_numpy(), covered)
    assert ci.coverage == pytest.approx(covered.mean()) and ci.training == ("1960Q1", "1978Q3") and ci.overlap == 0
    assert "exchangeable" in " ".join(ci.summary().notes)
    few = conformal(fit, data, ("1985Q1", "1985Q4"), alpha=0.02, calibration=0.1)  # 10 calibration periods: k = 11 > 10
    assert math.isinf(few.quantile) and "unbounded" in " ".join(few.summary().notes)


def test_split_conformal_covers_at_its_exact_rate():
    """Over R samples, each with 20 calibration periods and 50 new ones, the coverage averages k / (n + 1) = 19/21.
    Its variance per sample is that of the calibration-conditional coverage, Beta(k, n + 1 - k), plus the binomial
    noise of 50 draws; the tolerance is 3.5 standard errors of the mean over R samples."""
    R, n_fit, m, alpha = 200, 80, 50, 0.1
    n_cal = 20
    k = _exact_rank(n_cal, "0.1")
    mean = k / (n_cal + 1)
    beta_var = k * (n_cal + 1 - k) / ((n_cal + 1) ** 2 * (n_cal + 2))
    var = beta_var + (mean * (1 - mean) - beta_var) / m
    tolerance = 3.5 * math.sqrt(var / R)
    rates = []
    for r in range(R):
        data = _data(n_fit + m, 100 + r)
        fit = Equation(FORMULA).fit(data, span=(None, str(data.frame.index[n_fit - 1])))
        ci = conformal(fit, data, (str(data.frame.index[n_fit]), str(data.frame.index[-1])), alpha=alpha)
        assert ci.rank == k and len(ci.scores) == n_cal
        rates.append(ci.coverage)
    assert abs(np.mean(rates) - mean) < tolerance  # 0.9048 +/- about 0.019


def test_cqr_by_hand_and_its_coverage():
    sklearn = pytest.importorskip("sklearn")
    from pymetrix import GradientBoosting

    method = GradientBoosting(trees=30, depth=2, seed=1, folds=3)

    def gbr(**extra):
        return sklearn.ensemble.GradientBoostingRegressor(n_estimators=30, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1,
                                                          **extra)

    rng_scale = lambda n, seed: 0.3 + np.abs(np.random.default_rng(seed + 1).normal(size=n))  # noqa: E731 - heteroskedastic
    data = _data(130, 7, rng_scale(130, 7))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Equation(FORMULA).fit(data, span=(None, "1984Q4"), method=method)
        ci = conformal(fit, data, ("1985Q1", "1992Q2"), alpha=0.2, method="cqr")
    frame = data.frame
    X, y = frame[["x1", "x2"]].to_numpy(), frame["y"].to_numpy()
    lo = gbr(loss="quantile", alpha=0.1).fit(X[:75], y[:75])
    hi = gbr(loss="quantile", alpha=0.9).fit(X[:75], y[:75])
    scores = np.maximum(lo.predict(X[75:100]) - y[75:100], y[75:100] - hi.predict(X[75:100]))
    Q = np.sort(scores)[_exact_rank(25, "0.2") - 1]
    np.testing.assert_allclose(ci.scores.to_numpy(), scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ci.frame["lower"].to_numpy(), lo.predict(X[100:]) - Q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ci.frame["upper"].to_numpy(), hi.predict(X[100:]) + Q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ci.frame["forecast"].to_numpy(), gbr().fit(X[:75], y[:75]).predict(X[100:]), rtol=1e-12)
    assert (ci.frame["upper"] - ci.frame["lower"]).std() > 0  # the width follows the quantile learners
    # coverage over 60 samples (20 calibration periods, 40 new ones): k / (n + 1) = 0.8095, 3.5 standard errors
    R, m, n_cal = 60, 40, 20
    k = _exact_rank(n_cal, "0.2")
    mean = k / (n_cal + 1)
    beta_var = k * (n_cal + 1 - k) / ((n_cal + 1) ** 2 * (n_cal + 2))
    tolerance = 3.5 * math.sqrt((beta_var + (mean * (1 - mean) - beta_var) / m) / R)
    rates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(R):
            sim = _data(80 + m, 300 + r, rng_scale(80 + m, 300 + r))
            f = Equation(FORMULA).fit(sim, span=(None, str(sim.frame.index[79])), method=method)
            rates.append(conformal(f, sim, (str(sim.frame.index[80]), str(sim.frame.index[-1])), alpha=0.2, method="cqr").coverage)
    assert abs(np.mean(rates) - mean) < tolerance


def test_adaptive_conformal_inference_by_hand_and_within_its_bound():
    """After the calibration periods the errors' standard deviation doubles. Split conformal's coverage falls well
    short; ACI's recursion, coded here, keeps the miss rate within (max(alpha, 1 - alpha) + gamma) / (gamma T) of
    alpha (Gibbs and Candès 2021, Proposition 4.1), a bound that holds for any data."""
    n_fit, T, alpha, gamma = 100, 400, 0.1, 0.05
    scale = np.r_[np.ones(n_fit), np.full(T, 2.0)]
    data = _data(n_fit + T, 5, scale)
    fit = Equation(FORMULA).fit(data, span=(None, str(data.frame.index[n_fit - 1])))
    span = (str(data.frame.index[n_fit]), str(data.frame.index[-1]))
    split = conformal(fit, data, span, alpha=alpha)
    aci = conformal(fit, data, span, alpha=alpha, method="aci", gamma=gamma)
    assert split.coverage < 0.75
    miss = 1 - aci.frame["covered"].to_numpy()
    assert abs(miss.mean() - alpha) <= (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(aci.coverage - 0.9) < 0.05
    # the recursion by hand, from the split's scores and predictions
    pool, level = list(split.scores.to_numpy()), alpha
    yhat, y = split.frame["forecast"].to_numpy(), split.frame["actual"].to_numpy()
    size = len(pool)
    for t in range(T):
        recent = np.sort(pool[-size:])
        k = math.ceil((size + 1) * (1 - level))
        q = math.inf if k > size else recent[k - 1]
        assert aci.frame["alpha_t"].iloc[t] == pytest.approx(level, abs=1e-12)
        assert aci.frame["upper"].iloc[t] == pytest.approx(yhat[t] + q, rel=1e-12)
        err = float(abs(y[t] - yhat[t]) > q)
        assert miss[t] == err
        level += gamma * (alpha - err)
        pool.append(abs(y[t] - yhat[t]))
    window = conformal(fit, data, span, alpha=alpha, method="aci", gamma=gamma, window=10)
    assert window.window == 10 and not np.allclose(window.frame["upper"], aci.frame["upper"])


def test_periods_without_an_outcome_keep_the_last_level():
    data = _data(60, 2)
    frame = data.frame.copy()
    frame.loc[frame.index[50]:, "y"] = np.nan  # true forecasts: the regressors are known, the outcome is not
    data = Dataset.from_frame(frame)
    fit = Equation(FORMULA).fit(data, span=(None, str(frame.index[39])))
    ci = conformal(fit, data, (str(frame.index[40]), str(frame.index[-1])), method="aci", gamma=0.05)
    assert ci.frame["covered"].iloc[10:].isna().all() and ci.frame["alpha_t"].iloc[10:].nunique() == 1
    assert ci.frame["actual"].iloc[:10].notna().all()


def test_learners_are_estimated_again_on_the_training_periods():
    lgb = pytest.importorskip("lightgbm")
    from pymetrix import LightGBM

    data = _data(120, 3)
    fit = Equation(FORMULA).fit(data, span=(None, "1984Q4"), method=LightGBM(trees=40, seed=2))
    ci = fit.conformal(data, ("1985Q1", "1989Q4"), 0.1)
    X, y = data.frame[["x1", "x2"]].to_numpy(), data.frame["y"].to_numpy()
    direct = lgb.LGBMRegressor(n_jobs=1, deterministic=True, force_row_wise=True, verbosity=-1, n_estimators=40, learning_rate=0.05,
                               num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=2).fit(X[:75], y[:75])
    np.testing.assert_array_equal(ci.frame["forecast"].to_numpy(), direct.predict(X[100:]))
    np.testing.assert_allclose(ci.scores.to_numpy(), np.abs(y[75:100] - direct.predict(X[75:100])), rtol=1e-12)
    cqr = fit.conformal(data, ("1985Q1", "1989Q4"), 0.1, method="cqr")
    lo = lgb.LGBMRegressor(n_jobs=1, deterministic=True, force_row_wise=True, verbosity=-1, n_estimators=40, learning_rate=0.05,
                           num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=2, objective="quantile",
                           alpha=0.05).fit(X[:75], y[:75])
    hi = lgb.LGBMRegressor(n_jobs=1, deterministic=True, force_row_wise=True, verbosity=-1, n_estimators=40, learning_rate=0.05,
                           num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=2, objective="quantile",
                           alpha=0.95).fit(X[:75], y[:75])
    scores = np.maximum(lo.predict(X[75:100]) - y[75:100], y[75:100] - hi.predict(X[75:100]))
    np.testing.assert_allclose(cqr.scores.to_numpy(), scores, rtol=1e-12, atol=1e-12)


def test_against_mapie():
    """MAPIE's split conformal regressor (1.x), given the same model trained on the same periods and the same
    calibration periods, gives the same interval."""
    mapie = pytest.importorskip("mapie.regression")
    sklearn = pytest.importorskip("sklearn")
    if not hasattr(mapie, "SplitConformalRegressor"):
        pytest.skip("MAPIE before 1.0")
    data = _data(120, 0)
    fit = Equation(FORMULA).fit(data, span=(None, "1984Q4"))
    ci = conformal(fit, data, ("1985Q1", "1989Q4"), alpha=0.2)
    X, y = data.frame[["x1", "x2"]].to_numpy(), data.frame["y"].to_numpy()
    model = sklearn.linear_model.LinearRegression().fit(X[:75], y[:75])
    scr = mapie.SplitConformalRegressor(model, confidence_level=0.8, prefit=True).conformalize(X[75:100], y[75:100])
    _, bounds = scr.predict_interval(X[100:])
    np.testing.assert_allclose(ci.frame["lower"].to_numpy(), bounds[:, 0, 0], rtol=1e-10)
    np.testing.assert_allclose(ci.frame["upper"].to_numpy(), bounds[:, 1, 0], rtol=1e-10)


def test_errors():
    data = _data(60, 1)
    fit = Equation(FORMULA).fit(data, span=(None, "1969Q4"))
    with pytest.raises(SpecificationError, match="quantile loss"):
        conformal(fit, data, ("1970Q1", "1970Q4"), method="cqr")
    with pytest.raises(SpecificationError, match="alpha"):
        conformal(fit, data, ("1970Q1", "1970Q4"), alpha=1.2)
    with pytest.raises(SpecificationError, match="calibration"):
        conformal(fit, data, ("1970Q1", "1970Q4"), calibration=1.0)
    with pytest.raises(SpecificationError, match="estimated equations"):
        conformal(data, data, ("1970Q1", "1970Q4"))


def test_the_dialog_records_a_call():
    from pymetrix.commands import ConformalForecast, Session
    from pymetrix.views import sections

    data = _data(80, 4)
    session = Session()
    session.objects["data"] = data
    session.objects["fit"] = Equation(FORMULA).fit(data, span=(None, "1974Q4"))
    session.run(ConformalForecast("fit", "1975Q1", "1979Q4", method="aci", gamma=0.01))
    record = session.history[-1]
    assert "conformal(fit, data, ('1975Q1', '1979Q4'), method='aci', gamma=0.01)" in record.code
    result = session[record.outputs[0]]
    assert result.method == "aci" and [title for title, _ in sections(result)][:3] == ["Summary", "Graph", "Intervals"]
