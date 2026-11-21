"""The Black-Litterman result: the same numbers as the function's old (mean, covariance) tuple, exactly; the closed
form worked by hand on two assets; Theil's mixed-estimation form of the same posterior; labels by asset and view;
and tuple unpacking kept for old code."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix_finance import BlackLittermanPosterior, black_litterman


def _old_black_litterman(cov, market_weights, views=None, q=None, omega=None, delta=2.5, tau=0.05):
    """The function as it was before it returned a result (commit a5cacf1), less the pymetrix-table branch."""
    S = np.asarray(cov, float)
    w = np.asarray(market_weights, float)
    pi = delta * S @ w
    if views is None:
        return pi, S + tau * S
    P = np.atleast_2d(np.asarray(views, float))
    q = np.atleast_1d(np.asarray(q, float))
    tS = tau * S
    Om = np.diag(np.diag(P @ tS @ P.T)) if omega is None else np.atleast_2d(np.asarray(omega, float))
    K = tS @ P.T @ np.linalg.inv(P @ tS @ P.T + Om)
    return pi + K @ (q - P @ pi), S + tS - K @ P @ tS


S3 = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.0625]])


@pytest.mark.parametrize("args", [
    dict(),
    dict(views=[[1, -1, 0], [0.5, 0, 0.5]], q=[0.002, 0.006]),
    dict(views=[[1, -1, 0], [0.5, 0, 0.5]], q=[0.002, 0.006], omega=np.diag([1e-4, 2e-4]), delta=3.0, tau=0.1),
    dict(views=[0, 0, 1], q=0.05, delta=2.0),
])
def test_the_same_numbers_as_the_old_tuple_exactly(args):
    new = black_litterman(S3, [0.5, 0.3, 0.2], **args)
    old_mu, old_sigma = _old_black_litterman(S3, [0.5, 0.3, 0.2], **args)
    mu, sigma = new  # unpacks as before
    assert np.array_equal(mu, old_mu) and np.array_equal(sigma, old_sigma)
    assert np.array_equal(new[0], old_mu) and np.array_equal(new[1], old_sigma) and len(new) == 2
    assert np.array_equal(new.prior.to_numpy(), args.get("delta", 2.5) * S3 @ np.array([0.5, 0.3, 0.2]))


def test_two_assets_by_hand():
    """S = [[0.04, 0.006], [0.006, 0.01]], w = (0.6, 0.4), delta = 2, tau = 0.1, one view a - b = 0.03 with variance
    0.0012. By hand: pi = 2 S w = (0.0528, 0.0152); tau S P' = (0.0034, -0.0004); P tau S P' = 0.0038, plus omega 0.005;
    K = (0.68, -0.08); q - P pi = 0.03 - 0.0376 = -0.0076; mu = pi + K (-0.0076) = (0.047632, 0.015808);
    Sigma = S (1 + tau) - K (P tau S) = [[0.044 - 0.002312, 0.0066 + 0.000272], [., 0.011 - 0.000032]]."""
    S = pd.DataFrame([[0.04, 0.006], [0.006, 0.01]], index=["a", "b"], columns=["a", "b"])
    bl = black_litterman(S, [0.6, 0.4], [[1, -1]], [0.03], omega=[[0.0012]], delta=2.0, tau=0.1)
    assert np.allclose(bl.prior, [0.0528, 0.0152], atol=1e-15)
    assert np.allclose(bl.posterior_mean, [0.047632, 0.015808], atol=1e-15)
    assert np.allclose(bl.posterior_cov, [[0.041688, 0.006872], [0.006872, 0.010968]], atol=1e-15)
    # Theil's mixed estimation: the precision-weighted mean of the prior and the views
    tS, P, Om = 0.1 * S.to_numpy(), np.array([[1.0, -1.0]]), np.array([[0.0012]])
    A = np.linalg.inv(tS) + P.T @ np.linalg.inv(Om) @ P
    theil = np.linalg.solve(A, np.linalg.inv(tS) @ bl.prior.to_numpy() + P.T @ np.linalg.inv(Om) @ [0.03])
    assert np.allclose(bl.posterior_mean, theil, atol=1e-14)
    assert np.allclose(bl.posterior_cov, S.to_numpy() + np.linalg.inv(A), atol=1e-14)
    # the optimal weights: (delta Sigma)^-1 mu; with no views, the market weights over 1 + tau
    assert np.allclose(bl.weights, np.linalg.solve(2.0 * bl.posterior_cov.to_numpy(), bl.posterior_mean.to_numpy()))
    assert np.allclose(black_litterman(S, [0.6, 0.4], delta=2.0, tau=0.1).weights, np.array([0.6, 0.4]) / 1.1)


def test_labels_views_and_confidences():
    data = Dataset.from_frame(pd.DataFrame(np.random.default_rng(0).multivariate_normal([0, 0, 0], S3 / 100, size=300), columns=["a", "b", "c"],
                                           index=pd.period_range("2000-01", periods=300, freq="M", name="time")))
    bl = black_litterman(data.cov(["a", "b", "c"]), [0.5, 0.3, 0.2], [[1, -1, 0], [0.5, 0, 0.5], [1, -1, 0]], [0.002, 0.006, 0.001])
    assert isinstance(bl, BlackLittermanPosterior) and bl.assets == ("a", "b", "c")
    assert list(bl.posterior_mean.index) == ["a", "b", "c"] and list(bl.posterior_cov.columns) == ["a", "b", "c"]
    assert list(bl.views.index) == ["a - b", "0.5 a + 0.5 c", "a - b (2)"] and list(bl.q) == [0.002, 0.006, 0.001]
    P = bl.views.to_numpy()
    tS = 0.05 * data.frame[["a", "b", "c"]].cov().to_numpy()
    assert np.allclose(bl.confidences, np.diag(P @ tS @ P.T)) and bl.omega_default
    vf = bl.views_frame()
    assert np.allclose(vf["prior"], P @ bl.prior.to_numpy()) and np.allclose(vf["posterior"], P @ bl.posterior_mean.to_numpy())
    summary = bl.summary()
    assert summary.index == ["a", "b", "c"] and "prior return" in summary.columns and "He and Litterman" in " ".join(summary.notes)
    assert bl.to_frame().loc["b", "posterior return"] == bl.posterior_mean["b"]
    fig = bl.plot()
    assert [type(x).__name__ for x in fig.panels[0].layers] == ["Bars", "Bars"]
    assert bl.provenance.spec["q"] == [0.002, 0.006, 0.001] and len(bl.provenance.data_fingerprint) == 16
    assert black_litterman(S3, [0.5, 0.3, 0.2]).views is None
    assert black_litterman(S3, pd.Series([0.5, 0.3, 0.2], index=["x", "y", "z"])).assets == ("x", "y", "z")
    from pymetrix.errors import SpecificationError

    with pytest.raises(SpecificationError, match="square"):
        black_litterman(S3, [0.5, 0.5])
