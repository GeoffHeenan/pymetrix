"""The scikit-learn plugin (Elliott and Timmermann 2016, chapters 5, 6 and 11): ridge against its closed form, the lasso
and elastic net against their optimality conditions, cross-validation and bagging redone by hand, the learners against
scikit-learn called directly, dynamic forecasts (including a transformed left-hand side) by hand, backtests and the
Estimate dialog."""

import warnings

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")
sm = pytest.importorskip("statsmodels.api")

from pymetrix import Bagging, Dataset, ElasticNet, Equation, GradientBoosting, Lasso, NearestNeighbours, NeuralNetwork, RandomForest, Ridge, backtest  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402
from pymetrix.ext.sklearn import time_series_folds  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(6)
    n = 160
    idx = pd.period_range("1980Q1", periods=n, freq="Q", name="time")
    X = rng.normal(size=(n, 6)) * [1, 2, 0.5, 1, 3, 1]
    y = 1.0 + X @ [0.8, -0.3, 0.0, 0.0, 0.1, 0.0] + rng.normal(scale=0.7, size=n)
    frame = pd.DataFrame(X, index=idx, columns=[f"x{i}" for i in range(1, 7)])
    frame["y"] = y
    return Dataset.from_frame(frame)


FORMULA = "y ~ x1 + x2 + x3 + x4 + x5 + x6"


def _arrays(data):
    frame = data.frame
    return frame[[f"x{i}" for i in range(1, 7)]].to_numpy(), frame["y"].to_numpy()


def _standardized(X, y):
    mean, sd = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / sd, y - y.mean(), mean, sd


def test_ridge_is_its_closed_form(data):
    X, y = _arrays(data)
    alpha = 25.0
    Z, yc, mean, sd = _standardized(X, y)
    bz = np.linalg.solve(Z.T @ Z + alpha * np.eye(6), Z.T @ yc)  # min |yc - Z b|^2 + alpha |b|^2
    beta = bz / sd
    const = y.mean() - beta @ mean
    fit = Equation(FORMULA).fit(data, method=Ridge(alpha=alpha))
    assert fit.params.values == pytest.approx(np.concatenate([[const], beta]), rel=1e-8)
    assert fit.forecast(data, ("2010Q1", "2010Q2")).frame["forecast"].to_numpy() == pytest.approx(
        const + data.frame.loc["2010Q1":"2010Q2", [f"x{i}" for i in range(1, 7)]].to_numpy() @ beta)


@pytest.mark.parametrize("method, l1", [(Lasso(alpha=0.05), 1.0), (ElasticNet(alpha=0.08, l1_ratio=0.4), 0.4)])
def test_lasso_and_elastic_net_meet_their_optimality_conditions(data, method, l1):
    X, y = _arrays(data)
    fit = Equation(FORMULA).fit(data, method=method)
    Z, yc, mean, sd = _standardized(X, y)
    bz = fit.params.values[1:] * sd  # back to the standardized scale
    r = yc - Z @ bz
    grad = Z.T @ r / len(y)  # objective: |r|^2 / 2n + alpha (l1 |b|_1 + (1 - l1) |b|^2 / 2)
    a = method.alpha
    on = bz != 0
    assert grad[on] == pytest.approx(a * (l1 * np.sign(bz[on]) + (1 - l1) * bz[on]), abs=1e-7)
    assert np.all(np.abs(grad[~on]) <= a * l1 + 1e-7)
    if l1 == 1.0:
        assert on.sum() < 6  # the lasso sets some coefficients exactly to zero
    assert fit.params.values[0] == pytest.approx(y.mean() - fit.params.values[1:] @ mean, rel=1e-9)


def test_the_penalty_is_chosen_by_time_series_cross_validation(data):
    X, y = _arrays(data)
    fit = Equation(FORMULA).fit(data, method=Ridge(folds=4))
    n = len(y)
    best = (np.inf, None)
    for alpha in n * np.logspace(-5, 2, 36):
        errors = []
        for k in range(4):  # expanding windows: train on the first part, validate on the next block
            cut = n - (4 - k) * (n // 5)
            train, test = np.arange(cut), np.arange(cut, cut + n // 5)
            Zt, yt, mean, sd = _standardized(X[train], y[train])
            beta = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(6), Zt.T @ yt) / sd
            errors.append(y[test] - (y[train].mean() - beta @ mean + X[test] @ beta))
        mse = float(np.mean(np.concatenate(errors) ** 2))
        if mse < best[0]:
            best = (mse, alpha)
    assert fit.stats["penalty"] == pytest.approx(best[1]) and fit.stats["cv_rmse"] == pytest.approx(np.sqrt(best[0]), rel=1e-9)
    assert [len(t) for _, t in time_series_folds(n, 4)] == [n // 5] * 4
    lasso = Equation(FORMULA).fit(data, method=Lasso())
    assert lasso.params.values[3] == 0 and lasso.params.values[1] > 0.6  # x3 dropped, x1 kept


def test_bagging_by_hand(data):
    X, y = _arrays(data)
    Xc = np.column_stack([np.ones(len(y)), X])
    method = Bagging(critical=1.96, reps=50, block=4, seed=3)
    fit = Equation(FORMULA).fit(data, method=method)
    rng = np.random.default_rng(3)
    n = len(y)
    total = np.zeros(7)
    for _ in range(50):
        starts = rng.integers(0, n - 4 + 1, size=int(np.ceil(n / 4)))
        rows = np.concatenate([np.arange(s, s + 4) for s in starts])[:n]
        ols = sm.OLS(y[rows], Xc[rows]).fit()
        keep = np.abs(ols.tvalues) > 1.96
        keep[0] = True
        b = np.zeros(7)
        b[keep] = sm.OLS(y[rows], Xc[rows][:, keep]).fit().params
        total += b
    assert fit.params.values == pytest.approx(total / 50, rel=1e-8)
    assert fit.extra["inclusion"]["x1"] == 1.0 and fit.extra["inclusion"]["x3"] < 0.5


def test_learners_are_scikit_learn_with_cross_validated_errors(data):
    X, y = _arrays(data)
    method = GradientBoosting(trees=60, learning_rate=0.1, depth=2, subsample=0.8, folds=4, seed=1)
    fit = Equation(FORMULA).fit(data, method=method)
    direct = sklearn.ensemble.GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=1).fit(X, y)
    assert fit.fitted.to_numpy() == pytest.approx(direct.predict(X))
    errors = []
    for train, test in sklearn.model_selection.TimeSeriesSplit(n_splits=4).split(X):
        model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=1)
        errors.append(y[test] - model.fit(X[train], y[train]).predict(X[test]))
    assert fit.cv_rmse == pytest.approx(float(np.sqrt(np.mean(np.concatenate(errors) ** 2))))
    assert fit.importance().index[0] == "x1"  # the strongest regressor matters most
    for m in (RandomForest(trees=50), NeuralNetwork(hidden=(4,)), NearestNeighbours(k=5)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = Equation(FORMULA).fit(data, method=m)
        assert np.isfinite(other.cv_rmse) and other.summary().rows[3][1] == pytest.approx(other.cv_rmse)


def test_every_learner_setting_reaches_scikit_learn(data):
    """Each new setting, changed from its default, gives what scikit-learn gives with that argument (the regressors
    standardized for the network and neighbours, the network's target too); left alone, it is not passed at all."""
    X, y = _arrays(data)
    pipe, prep, compose = sklearn.pipeline, sklearn.preprocessing, sklearn.compose
    cases = [
        (RandomForest(trees=40, seed=2, max_depth=4, min_split=6, criterion="absolute_error", max_samples=0.7, pruning=0.01),
         lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=40, min_samples_leaf=5, max_features=0.5, random_state=2, max_depth=4,
                                                         min_samples_split=6, criterion="absolute_error", max_samples=0.7, ccp_alpha=0.01)),
        (RandomForest(trees=30, seed=2, bootstrap=False),
         lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.5, random_state=2, bootstrap=False)),
        (GradientBoosting(trees=40, seed=1, loss="quantile", quantile=0.3, min_leaf=3, min_split=5, max_features=0.5, pruning=0.001),
         lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=40, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1,
                                                             loss="quantile", alpha=0.3, min_samples_leaf=3, min_samples_split=5,
                                                             max_features=0.5, ccp_alpha=0.001)),
        (GradientBoosting(trees=200, seed=1, loss="huber", stop_after=5, held_out=0.2),
         lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1,
                                                             loss="huber", n_iter_no_change=5, validation_fraction=0.2)),
        (NeuralNetwork(hidden=(5,), seed=3, activation="tanh", solver="lbfgs", max_iter=400),
         lambda: compose.TransformedTargetRegressor(regressor=pipe.make_pipeline(prep.StandardScaler(), sklearn.neural_network.MLPRegressor(
             hidden_layer_sizes=(5,), alpha=0.001, random_state=3, activation="tanh", solver="lbfgs", max_iter=400)),
             transformer=prep.StandardScaler())),
        (NeuralNetwork(hidden=(4,), seed=3, learning_rate=0.01, batch=32, max_iter=300, early_stopping=True),
         lambda: compose.TransformedTargetRegressor(regressor=pipe.make_pipeline(prep.StandardScaler(), sklearn.neural_network.MLPRegressor(
             hidden_layer_sizes=(4,), alpha=0.001, random_state=3, learning_rate_init=0.01, batch_size=32, max_iter=300,
             early_stopping=True)), transformer=prep.StandardScaler())),
        (NearestNeighbours(k=7, power=1),
         lambda: pipe.make_pipeline(prep.StandardScaler(), sklearn.neighbors.KNeighborsRegressor(n_neighbors=7, weights="uniform", p=1))),
    ]
    for method, direct in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on a small sample
            fit = Equation(FORMULA).fit(data, method=method)
            ref = direct().fit(X, y)
        assert fit.fitted.to_numpy() == pytest.approx(ref.predict(X), rel=1e-9, abs=1e-9), type(method).__name__
    plain = Equation(FORMULA).fit(data, method=RandomForest(trees=20)).provenance.options["stages"]["model"]
    assert set(plain) == {"n_estimators", "min_samples_leaf", "max_features", "random_state"}  # defaults are not passed


def test_dynamic_forecasts_by_hand():
    macro = Dataset.example("sample_macro")
    fit = Equation("c ~ c[-1] + y").fit(macro, span=(None, "2020Q4"), method=NearestNeighbours(k=3))
    fc = fit.forecast(macro, ("2021Q1", "2021Q3"), dynamic=True).frame
    c, y = macro["c"], macro["y"]
    previous = c["2020Q4"]
    for period in ("2021Q1", "2021Q2", "2021Q3"):
        value = float(fit.model.predict(np.array([[previous, y[period]]]))[0])
        assert fc.loc[period, "forecast"] == pytest.approx(value)
        previous = value
    assert np.isfinite(fc["se"].iloc[0]) and fc["se"].iloc[1:].isna().all()  # it feeds on itself after one step
    # a transformed left-hand side: dlog(c) is solved for c, c_t = c_(t-1) exp(prediction)
    growth = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]").fit(macro, span=(None, "2020Q4"), method=NearestNeighbours(k=3))
    path = growth.forecast(macro, ("2021Q1", "2021Q3"), dynamic=True).frame
    level, lag_growth = c["2020Q4"], np.log(c["2020Q4"]) - np.log(c["2020Q3"])
    for period in ("2021Q1", "2021Q2", "2021Q3"):
        dly = np.log(y[period]) - np.log(y[pd.Period(period, "Q") - 1])
        g = float(growth.model.predict(np.array([[dly, lag_growth]]))[0])
        level, lag_growth = level * np.exp(g), g
        assert path.loc[period, "forecast"] == pytest.approx(level, rel=1e-9)
    assert path["se"].iloc[0] == pytest.approx(growth.cv_rmse * path["forecast"].iloc[0], rel=1e-4)  # d c / d dlog(c) = c
    static = growth.forecast(macro, ("2021Q1", "2021Q1")).frame
    assert static["actual"].iloc[0] == pytest.approx(np.log(c["2021Q1"]) - np.log(c["2020Q4"]))  # static: the left-hand side


def test_backtests_and_the_dialog():
    from pymetrix.commands import FitEquation, Session

    macro = Dataset.example("sample_macro")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Equation("c ~ y + c[-1]").fit(macro, span=(None, "2020Q4"), method=Lasso(folds=3))
        bt = backtest(fit, macro, ("2016Q1", "2016Q4"), horizon=1, label="lasso")
    row = bt.errors.iloc[2]
    own = Equation("c ~ y + c[-1]").fit(macro, span=(fit.provenance.sample[0], row["origin"]), method=Lasso(folds=3))
    assert row["forecast"] == pytest.approx(float(own.forecast(macro, (str(row["target"]), str(row["target"])), dynamic=True).frame["forecast"].iloc[0]))
    session = Session()
    session.objects["data"] = macro
    session.run(FitEquation("c ~ y + c[-1]", method="Lasso", method_settings="folds=3"))
    namespace = dict(session.namespace) | {"data": macro}
    record = session.history[-1]
    exec(record.code, namespace)
    name = record.outputs[0]
    assert "method=Lasso(folds=3)" in record.code and type(session[name].method).__name__ == "Lasso"
    assert namespace[name].params.values == pytest.approx(session[name].params.values)


def test_errors(data):
    with pytest.raises(SpecificationError, match="weights"):
        Equation(FORMULA).fit(data, method=Ridge(), weights="x1")
    with pytest.raises(SpecificationError, match="l1_ratio"):
        ElasticNet(l1_ratio=0.0)
    with pytest.raises(SpecificationError, match="hidden"):
        NeuralNetwork(hidden=())
