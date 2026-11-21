"""Spatial econometrics, checked independently: k-nearest-neighbour weights equal to libpysal's; the spatial lag and
error models' estimates, standard errors and log likelihoods equal to spreg's ML_Lag and ML_Error; Moran's I of the
residuals and the four LM tests equal to spreg's OLS diagnostics; and, without spreg, the lag model's log likelihood
by hand (with the log-determinant from numpy) and a simulated rho recovered."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, knn_weights, spatial_diagnostics, spatial_regression
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def spatial():
    rng = np.random.default_rng(0)
    n = 200
    xy = rng.uniform(0, 10, (n, 2))
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    base = Dataset.from_frame(pd.DataFrame({"id": range(n), "lon": xy[:, 0], "lat": xy[:, 1]}), entity="id")
    W = knn_weights(base, ["lon", "lat"], k=5)
    Wr = W / W.sum(axis=1, keepdims=True)
    y = np.linalg.solve(np.eye(n) - 0.5 * Wr, 1 + 2 * x1 - x2 + rng.normal(size=n))
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x1": x1, "x2": x2, "lon": xy[:, 0], "lat": xy[:, 1]}), entity="id")
    return data, W, Wr, xy, y, np.column_stack([x1, x2])


def test_the_lag_likelihood_by_hand(spatial):
    data, W, Wr, _, y, X = spatial
    fit = spatial_regression(data, "y", ["x1", "x2"], W, "lag")
    b, rho = fit.params[:3], fit.params[3]
    n = len(y)
    e = y - rho * Wr @ y - np.column_stack([np.ones(n), X]) @ b
    s2 = e @ e / n
    ll = -n / 2 * (np.log(2 * np.pi * s2) + 1) + np.linalg.slogdet(np.eye(n) - rho * Wr)[1]
    assert fit.loglike == pytest.approx(ll, rel=1e-10) and rho == pytest.approx(0.5, abs=3 * fit.se[3])
    with pytest.raises(SpecificationError, match="model"):
        spatial_regression(data, "y", ["x1"], W, "durbin")
    with pytest.raises(SpecificationError, match="200 x 200"):
        spatial_regression(data, "y", ["x1"], W[:10, :10])


@pytest.mark.parametrize("model", ["lag", "error"])
def test_against_spreg(spatial, model):
    spreg = pytest.importorskip("spreg")
    libpysal = pytest.importorskip("libpysal")
    data, W, Wr, xy, y, X = spatial
    w = libpysal.weights.KNN(xy, k=5)
    w.transform = "r"
    assert w.full()[0] == pytest.approx(Wr, abs=1e-15)
    ref = (spreg.ML_Lag if model == "lag" else spreg.ML_Error)(y[:, None], X, w=w, method="full")
    fit = spatial_regression(data, "y", ["x1", "x2"], W, model)
    assert fit.params == pytest.approx(np.asarray(ref.betas).ravel(), abs=1e-6)
    assert fit.se == pytest.approx(np.asarray(ref.std_err).ravel(), rel=1e-5)
    assert fit.loglike == pytest.approx(float(ref.logll), rel=1e-10)


def test_diagnostics_against_spreg(spatial):
    spreg = pytest.importorskip("spreg")
    libpysal = pytest.importorskip("libpysal")
    data, W, _, xy, y, X = spatial
    w = libpysal.weights.KNN(xy, k=5)
    w.transform = "r"
    ref = spreg.OLS(y[:, None], X, w=w, spat_diag=True, moran=True)
    t = spatial_diagnostics(data, "y", ["x1", "x2"], W).to_frame().set_index("test")
    assert t.loc["Moran's I (residuals)", "statistic"] == pytest.approx(ref.moran_res[0], rel=1e-10)
    assert t.loc["Moran's I (residuals)", "z"] == pytest.approx(ref.moran_res[1], rel=1e-8)
    for name, value in [("LM lag", ref.lm_lag), ("robust LM lag", ref.rlm_lag), ("LM error", ref.lm_error), ("robust LM error", ref.rlm_error)]:
        assert t.loc[name, "statistic"] == pytest.approx(value[0], rel=1e-8)
