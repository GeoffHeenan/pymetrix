"""LightGBM and XGBoost as equation methods: the libraries called directly with the same data and settings give the
same predictions and cross-validated errors (exactly: one thread, deterministic modes), every setting reaches the
library, forecasts are the model's own, the Estimate dialog records them, and double machine learning takes them as
nuisance learners with the same cross-fitted estimate as the libraries' own estimators on the same folds."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from pymetrix import Dataset, Equation, double_ml  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402

FIXED_LGBM = {"n_jobs": 1, "deterministic": True, "force_row_wise": True, "verbosity": -1}
FIXED_XGB = {"n_jobs": 1, "tree_method": "hist"}
FORMULA = "y ~ x1 + x2 + x3 + x4"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 150
    idx = pd.period_range("1985Q1", periods=n, freq="Q", name="time")
    X = rng.normal(size=(n, 4))
    y = 0.5 + X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 2] + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame(X, index=idx, columns=["x1", "x2", "x3", "x4"])
    frame["y"] = y
    return Dataset.from_frame(frame)


def _arrays(data, end=None):
    frame = data.frame.loc[:end] if end else data.frame
    return frame[["x1", "x2", "x3", "x4"]].to_numpy(), frame["y"].to_numpy()


def _cv_rmse(make, X, y, folds):
    from sklearn.model_selection import TimeSeriesSplit

    errors = [y[test] - make().fit(X[train], y[train]).predict(X[test]) for train, test in TimeSeriesSplit(n_splits=folds).split(X)]
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def test_lightgbm_is_lightgbm(data):
    lgb = pytest.importorskip("lightgbm")
    from pymetrix import LightGBM

    X, y = _arrays(data)
    cases = [
        (LightGBM(trees=80, learning_rate=0.1, leaves=7, min_leaf=8, subsample=0.7, folds=4, seed=3),
         dict(n_estimators=80, learning_rate=0.1, num_leaves=7, min_child_samples=8, subsample=0.7, subsample_freq=1, random_state=3)),
        (LightGBM(trees=60, subsample=1.0, loss="quantile", quantile=0.25, max_features=0.5, depth=3, l1=0.5, l2=2.0, min_gain=0.01, seed=1),
         dict(n_estimators=60, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=1.0, random_state=1, objective="quantile",
              alpha=0.25, colsample_bytree=0.5, max_depth=3, reg_alpha=0.5, reg_lambda=2.0, min_split_gain=0.01)),
        (LightGBM(trees=50, loss="absolute_error"),
         dict(n_estimators=50, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=0,
              objective="regression_l1")),
    ]
    for method, args in cases:
        fit = Equation(FORMULA).fit(data, method=method)
        make = lambda args=args: lgb.LGBMRegressor(**FIXED_LGBM, **args)  # noqa: E731
        direct = make().fit(X, y)
        np.testing.assert_array_equal(fit.fitted.to_numpy(), direct.predict(X))
        assert fit.cv_rmse == pytest.approx(_cv_rmse(make, X, y, method.folds), rel=1e-12)
        assert fit.provenance.options["stages"]["model"] == {**FIXED_LGBM, **args}
        assert fit.provenance.seed == method.seed and fit.provenance.backend["name"] == "lightgbm"
    again = Equation(FORMULA).fit(data, method=cases[0][0])
    np.testing.assert_array_equal(again.fitted.to_numpy(), Equation(FORMULA).fit(data, method=cases[0][0]).fitted.to_numpy())
    other = Equation(FORMULA).fit(data, method=LightGBM(trees=80, learning_rate=0.1, leaves=7, min_leaf=8, subsample=0.7, folds=4, seed=4))
    assert not np.array_equal(other.fitted.to_numpy(), again.fitted.to_numpy())  # the seed draws the row samples


def test_xgboost_is_xgboost(data):
    xgb = pytest.importorskip("xgboost")
    from pymetrix import XGBoost

    X, y = _arrays(data)
    cases = [
        (XGBoost(trees=80, learning_rate=0.1, depth=3, subsample=0.7, folds=4, seed=3),
         dict(n_estimators=80, learning_rate=0.1, max_depth=3, subsample=0.7, random_state=3)),
        (XGBoost(trees=60, loss="quantile", quantile=0.75, max_features=0.5, min_child_weight=3, l1=0.5, l2=4.0, min_gain=0.01, seed=1),
         dict(n_estimators=60, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1, objective="reg:quantileerror",
              quantile_alpha=0.75, colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.5, reg_lambda=4.0, gamma=0.01)),
        (XGBoost(trees=50, loss="absolute_error"),
         dict(n_estimators=50, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0, objective="reg:absoluteerror")),
    ]
    for method, args in cases:
        fit = Equation(FORMULA).fit(data, method=method)
        make = lambda args=args: xgb.XGBRegressor(**FIXED_XGB, **args)  # noqa: E731
        direct = make().fit(X, y)
        np.testing.assert_array_equal(fit.fitted.to_numpy(), direct.predict(X))
        assert fit.cv_rmse == pytest.approx(_cv_rmse(make, X, y, method.folds), rel=1e-12)
        assert fit.provenance.options["stages"]["model"] == {**FIXED_XGB, **args}
        assert fit.provenance.seed == method.seed and fit.provenance.backend["name"] == "xgboost"


def test_forecasts_are_the_models_own(data):
    pytest.importorskip("lightgbm")
    from pymetrix import LightGBM

    fit = Equation(FORMULA).fit(data, span=(None, "2017Q4"), method=LightGBM(trees=40))
    fc = fit.forecast(data, ("2018Q1", "2019Q4")).frame
    rows = data.frame.loc["2018Q1":"2019Q4", ["x1", "x2", "x3", "x4"]].to_numpy()
    np.testing.assert_array_equal(fc["forecast"].to_numpy(), fit.model.predict(rows))
    assert fc["se"].to_numpy() == pytest.approx(np.full(8, fit.cv_rmse))
    assert fit.importance().index[0] == "x1"


def test_the_estimate_dialog_and_the_script(data):
    pytest.importorskip("xgboost")
    from pymetrix.commands import FitEquation, Session

    session = Session()
    session.objects["data"] = data
    session.run(FitEquation(FORMULA, method="XGBoost", method_settings="trees=40, depth=3"))
    record = session.history[-1]
    namespace = dict(session.namespace) | {"data": data}
    exec(record.code, namespace)
    name = record.outputs[0]
    assert "method=XGBoost(trees=40, depth=3)" in record.code and type(session[name].method).__name__ == "XGBoost"
    np.testing.assert_array_equal(namespace[name].fitted.to_numpy(), session[name].fitted.to_numpy())


def test_settings_are_checked():
    pytest.importorskip("lightgbm")
    from pymetrix import LightGBM, XGBoost

    with pytest.raises(SpecificationError, match="quantile"):
        LightGBM(loss="quantile", quantile=1.5)
    with pytest.raises(SpecificationError, match="subsample"):
        XGBoost(subsample=0.0)


def _plr(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    d = np.tanh(X[:, 0]) + 0.3 * X[:, 1] + rng.normal(size=n)
    y = 1.5 * d + np.cos(X[:, 0]) + X[:, 2] + rng.normal(size=n)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(d=d, y=y, id=range(n))
    return Dataset.from_frame(frame, entity="id"), X, d, y


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_double_ml_nuisances_are_the_librarys_own(name):
    """The cross-fitted PLR estimate with the library's estimator fitted by hand on the result's folds, with the seed
    double ML derives from its own (the seed plus a draw from the split's generator)."""
    lib = pytest.importorskip(name)
    from pymetrix import LightGBM, XGBoost

    data, X, d, y = _plr()
    method = LightGBM(trees=60, seed=2) if name == "lightgbm" else XGBoost(trees=60, seed=2)
    fit = double_ml(data, "y", "d", ["x1", "x2", "x3"], learner=method, folds=3, seed=5)
    sub = int(np.random.default_rng(5).integers(0, 2**31 - 1))
    if name == "lightgbm":
        make = lambda: lib.LGBMRegressor(**FIXED_LGBM, n_estimators=60, learning_rate=0.05, num_leaves=15, min_child_samples=10,  # noqa: E731
                                         subsample=0.8, subsample_freq=1, random_state=(2 + sub) % (2**31 - 1))
    else:
        make = lambda: lib.XGBRegressor(**FIXED_XGB, n_estimators=60, learning_rate=0.05, max_depth=2, subsample=0.8,  # noqa: E731
                                        random_state=(2 + sub) % (2**31 - 1))
    folds = fit.splits[:, 0]
    l_hat, m_hat = np.empty(len(y)), np.empty(len(y))
    for f in np.unique(folds):
        test = folds == f
        l_hat[test] = make().fit(X[~test], y[~test]).predict(X[test])
        m_hat[test] = make().fit(X[~test], d[~test]).predict(X[test])
    u, v = y - l_hat, d - m_hat
    assert fit.theta == pytest.approx(float(u @ v / (v @ v)), rel=1e-10)
    assert method.label in fit.learners["l(X) = E[Y|X]"]
    by_name = double_ml(data, "y", "d", ["x1", "x2", "x3"], learner=name, folds=3)
    assert abs(by_name.theta - 1.5) < 4 * by_name.se


def test_double_ml_propensity_from_a_boosted_classifier():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(4)
    n = 400
    X = rng.normal(size=(n, 3))
    d = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    y = 1.0 * d + X[:, 0] + rng.normal(size=n)
    data = Dataset.from_frame(pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(d=d, y=y, id=range(n)), entity="id")
    fit = double_ml(data, "y", "d", ["x1", "x2", "x3"], model="irm", learner="lightgbm", folds=3)
    assert "LightGBM" in fit.learners["m(X) = P(D=1|X)"] and abs(fit.theta - 1.0) < 0.5
