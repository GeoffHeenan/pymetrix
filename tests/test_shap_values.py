"""SHAP values of learners (Lundberg and Lee 2017; TreeSHAP, Lundberg et al. 2020): against shap called directly, against
LightGBM's and XGBoost's own TreeSHAP (pred_contrib), additivity (contributions plus the base value give the
prediction), the hand formula b_j (x_j - mean x_j) for a linear model, the seeded background sample, and the dialog."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
shap = pytest.importorskip("shap")

from pymetrix import Dataset, Equation, GradientBoosting, NeuralNetwork, RandomForest, shap_values  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402

FORMULA = "y ~ x1 + x2 + x3 + x4"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    n = 120
    idx = pd.period_range("1990Q1", periods=n, freq="Q", name="time")
    X = rng.normal(size=(n, 4))
    y = 1.0 + X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 2] + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame(X, index=idx, columns=["x1", "x2", "x3", "x4"])
    frame["y"] = y
    return Dataset.from_frame(frame)


def _check_additive(sv, tol):
    assert sv.additivity_error < tol
    assert np.abs(sv.values.sum(axis=1) + sv.base - sv.prediction).max() < tol


@pytest.mark.parametrize("method", [RandomForest(trees=30, seed=1), GradientBoosting(trees=40, seed=1)], ids=["forest", "boosting"])
def test_tree_shap_for_scikit_learn_trees(data, method):
    fit = Equation(FORMULA).fit(data, method=method)
    sv = shap_values(fit)
    assert sv.explainer == "tree"
    direct = shap.TreeExplainer(fit.model)
    X = fit.X.to_numpy(float)
    np.testing.assert_allclose(sv.values.to_numpy(), direct.shap_values(X), rtol=0, atol=1e-12)
    assert sv.base == pytest.approx(float(np.ravel(direct.expected_value)[0]), abs=1e-12)
    _check_additive(sv, 1e-10)
    assert sv.summary().index[0] == "x1" and sv.mean_abs.iloc[0] == pytest.approx(np.abs(sv.values["x1"]).mean())


def test_tree_shap_is_lightgbms_own(data):
    pytest.importorskip("lightgbm")
    from pymetrix import LightGBM

    fit = Equation(FORMULA).fit(data, span=(None, "2015Q4"), method=LightGBM(trees=60))
    sv = shap_values(fit)
    own = fit.model.predict(fit.X.to_numpy(float), pred_contrib=True)  # LightGBM's TreeSHAP; the last column is the base value
    np.testing.assert_allclose(sv.values.to_numpy(), own[:, :-1], rtol=0, atol=1e-10)
    assert sv.base == pytest.approx(own[0, -1], abs=1e-10)
    _check_additive(sv, 1e-10)
    later = shap_values(fit, data, ("2016Q1", "2019Q4"))  # periods outside the sample
    rows = data.frame.loc["2016Q1":"2019Q4", ["x1", "x2", "x3", "x4"]].to_numpy()
    np.testing.assert_allclose(later.values.to_numpy(), fit.model.predict(rows, pred_contrib=True)[:, :-1], rtol=0, atol=1e-10)
    assert list(later.values.index.astype(str)) == [str(p) for p in pd.period_range("2016Q1", "2019Q4", freq="Q")]


def test_tree_shap_is_xgboosts_own(data):
    xgb = pytest.importorskip("xgboost")
    from pymetrix import XGBoost

    fit = Equation(FORMULA).fit(data, method=XGBoost(trees=60, depth=3))
    sv = fit.shap()
    own = fit.model.get_booster().predict(xgb.DMatrix(fit.X.to_numpy(float)), pred_contribs=True)
    # XGBoost keeps its trees and predictions in single precision: agreement to its rounding
    np.testing.assert_allclose(sv.values.to_numpy(), own[:, :-1], rtol=0, atol=1e-5)
    assert sv.base == pytest.approx(float(own[0, -1]), abs=1e-5)
    _check_additive(sv, 1e-5)


def _linear(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Equation(FORMULA).fit(data, method=NeuralNetwork(hidden=(3,), activation="identity", solver="lbfgs", max_iter=200))
    X = fit.X.to_numpy(float)
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(A, fit.model.predict(X), rcond=None)[0]  # the network is affine: its coefficients, exactly
    assert np.abs(A @ b - fit.model.predict(X)).max() < 1e-10
    return fit, X, b


@pytest.mark.parametrize("explainer", ["exact", "permutation"])
def test_a_linear_models_shap_values_are_b_times_x_minus_its_mean(data, explainer):
    """With the whole sample as background, the contribution of x_j is b_j (x_j - mean x_j) and the base value the
    mean prediction (Lundberg and Lee 2017, linear models); both explainers give it to rounding."""
    fit, X, b = _linear(data)
    sv = shap_values(fit, explainer=explainer, background=len(X), evaluations=60)
    np.testing.assert_allclose(sv.values.to_numpy(), b[1:] * (X - X.mean(axis=0)), rtol=0, atol=1e-8)
    assert sv.base == pytest.approx(float(b[0] + b[1:] @ X.mean(axis=0)), abs=1e-10)
    _check_additive(sv, 1e-10)
    assert sv.explainer == explainer and len(sv.background) == len(X)


def test_the_background_sample_and_the_orderings_are_seeded(data):
    fit, X, b = _linear(data)
    state = np.random.get_state()[1].copy()
    first = shap_values(fit, explainer="permutation", background=30, seed=7)
    assert np.array_equal(np.random.get_state()[1], state)  # NumPy's global generator is left as it was
    rows = np.sort(np.random.default_rng(7).choice(len(X), size=30, replace=False))
    assert list(first.background) == list(fit.X.index[rows]) and first.seed == 7 and first.evaluations == 500
    np.testing.assert_allclose(first.values.to_numpy(), b[1:] * (X - X[rows].mean(axis=0)), rtol=0, atol=1e-8)
    again = shap_values(fit, explainer="permutation", background=30, seed=7)
    np.testing.assert_array_equal(first.values.to_numpy(), again.values.to_numpy())
    assert "seed 7" in first.summary().notes[1]


def test_nonlinear_learners_are_explained_exactly_from_their_predictions(data):
    """A network with a nonlinear activation: the exact explainer against shap's own Exact explainer on the same
    background, and additivity; auto chooses it for four regressors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Equation(FORMULA).fit(data, method=NeuralNetwork(hidden=(4,), solver="lbfgs", max_iter=300))
    sv = shap_values(fit, background=40, seed=3)
    assert sv.explainer == "exact"
    rows = np.sort(np.random.default_rng(3).choice(len(fit.X), size=40, replace=False))
    masker = shap.maskers.Independent(fit.X.to_numpy(float)[rows], max_samples=40)
    direct = shap.explainers.Exact(fit.model.predict, masker)(fit.X.to_numpy(float), silent=True)
    np.testing.assert_allclose(sv.values.to_numpy(), direct.values, rtol=0, atol=1e-12)
    _check_additive(sv, 1e-10)
    figure = sv.plot()
    assert len(figure.panels) == 2


def test_errors(data):
    fit = Equation(FORMULA).fit(data)
    with pytest.raises(SpecificationError, match="machine-learning"):
        shap_values(fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn = Equation(FORMULA).fit(data, method=NeuralNetwork(hidden=(2,), solver="lbfgs", max_iter=50))
    with pytest.raises(SpecificationError, match="TreeSHAP"):
        shap_values(nn, explainer="tree")
    with pytest.raises(SpecificationError, match="2p"):
        shap_values(nn, explainer="permutation", evaluations=5)


def test_the_dialog_records_a_call(data):
    from pymetrix.commands import ExplainLearner, Session
    from pymetrix.views import sections

    session = Session()
    session.objects["data"] = data
    session.objects["forest"] = Equation(FORMULA).fit(data, method=RandomForest(trees=20))
    session.run(ExplainLearner("forest", start="1995Q1", end="2000Q4"))
    record = session.history[-1]
    assert "shap_values(forest, span=('1995Q1', '2000Q4'))" in record.code
    result = session[record.outputs[0]]
    assert len(result.values) == 24 and [title for title, _ in sections(result)][:3] == ["Summary", "Graph", "Values"]
