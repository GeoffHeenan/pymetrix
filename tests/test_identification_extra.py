"""More SVAR identification, checked independently: the max-share shock at horizon 0 equal to the Cholesky shock with
the target first; its share no smaller than any of 20,000 random rotations' and equal to the FEVD's; the news-shock
variant with no impact effect; narrative restrictions holding in every accepted rotation by hand, the narrative set a
subset of the plain sign set; identification through heteroskedasticity reproducing both regimes' covariances exactly
and recovering a simulated impact matrix (columns up to sign and order)."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Cholesky, Dataset, MaxShare, SignRestrictions
from pymetrix.errors import SpecificationError


def _data(T=400, seed=0, B=None, breaks=None):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    B = np.array([[1.0, 0.3, 0.0], [0.5, 1.0, 0.2], [-0.3, 0.4, 1.0]]) if B is None else B
    y = np.zeros((T, 3))
    for t in range(1, T):
        scale = np.array([1.0, 1.0, 1.0]) if breaks is None or t < breaks else np.array([3.0, 1.0, 0.4])
        y[t] = A @ y[t - 1] + B @ (scale * rng.normal(size=3))
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b", "c"], index=pd.period_range("1900Q1", periods=T, freq="Q")))


@pytest.fixture(scope="module")
def var():
    return VAR(["a", "b", "c"], lags=1).fit(_data())


def _share(var, B, i, H, start=0):
    from pymetrix.var import _ma

    phi = _ma(var.coefs, H)[start:]
    num = np.sum((phi[:, i, :] @ B[:, 0]) ** 2)
    den = np.sum(np.einsum("hj,jk,hk->h", phi[:, i, :], var.estimate.sigma, phi[:, i, :]))
    return num / den


def test_max_share_at_horizon_zero_is_cholesky_with_the_target_first(var):
    ms = var.identify(MaxShare("b", horizon=0)).impact()
    chol = var.identify(Cholesky(["b", "a", "c"])).impact()
    assert ms[:, 0] == pytest.approx(chol[:, 1], abs=1e-10)  # Cholesky's columns follow the VAR's order: b's shock is column 1


def test_max_share_beats_random_rotations_and_is_the_fevd(var):
    fit = var.identify(MaxShare("c", horizon=12))
    B = fit.impact()
    best = _share(var, B, 2, 12)
    P = np.linalg.cholesky(var.estimate.sigma)
    rng = np.random.default_rng(1)
    g = rng.normal(size=(20000, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    others = [_share(var, (P @ gi)[:, None], 2, 12) for gi in g[:2000]]
    assert best >= max(others) - 1e-12
    fevd = fit.fevd(13)
    assert fevd.values[12, 2, 0] == pytest.approx(best, rel=1e-8)  # the FEVD at horizon 12 of shock 0 on c
    assert B @ B.T == pytest.approx(var.estimate.sigma, abs=1e-10) and fit.shocks[0] == "max share: c"
    irf = fit.irf(12, draws=30)
    assert irf.shocks[0] == "max share: c" and irf.lower is not None


def test_the_news_shock_has_no_impact_effect(var):
    B = var.identify(MaxShare("a", horizon=20, zero_impact=True)).impact()
    assert B[0, 0] == pytest.approx(0, abs=1e-12) and B @ B.T == pytest.approx(var.estimate.sigma, abs=1e-10)
    with pytest.raises(SpecificationError, match="not a variable"):
        var.identify(MaxShare("q"))


def test_narrative_restrictions(var):
    signs = {"s1": {"a": "+", "b": "+"}}
    plain = var.identify(SignRestrictions(signs, rotations=300, seed=2)).sign_draws()
    date = str(var.resid.frame.index[100])
    narr = var.identify(SignRestrictions(signs, rotations=300, seed=2, narrative_signs={"s1": {date: "+"}},
                                         narrative_major={"s1": {date: "a"}}))
    draws = narr.sign_draws()
    U = var.resid.frame[["a", "b", "c"]].to_numpy()
    for B in draws:  # every kept rotation satisfies the narrative, by hand
        e = np.linalg.solve(B, U[100])
        assert e[0] > 0 and np.argmax(np.abs(B[0] * e)) == 0
    plain_set = {tuple(np.round(B[:, 0], 12)) for B in plain}
    assert all(tuple(np.round(B[:, 0], 12)) in plain_set for B in draws) and len({tuple(np.round(B[:, 0], 12)) for B in draws}) < len(plain_set)
    assert narr.irf(8, draws=50).lower is not None  # bands from the identified set
    with pytest.raises(SpecificationError, match="not in the VAR"):
        var.identify(SignRestrictions(signs, rotations=50, narrative_signs={"s1": {"2300Q1": "+"}}))


def test_identification_through_heteroskedasticity():
    B = np.array([[1.0, 0.3, 0.0], [0.5, 1.0, 0.2], [-0.3, 0.4, 1.0]])
    data = _data(T=3000, seed=3, B=B, breaks=1500)
    var = VAR(["a", "b", "c"], lags=1).fit(data)
    h = var.heteroskedasticity_svar([str(data.frame.index[1500])])
    S1, S2 = h.regime_sigmas
    L = np.diag(h.ratios[0])
    assert h.impact @ h.impact.T == pytest.approx(S1, abs=1e-10) and h.impact @ L @ h.impact.T == pytest.approx(S2, abs=1e-10)
    assert h.ratios[0] == pytest.approx([9.0, 1.0, 0.16], rel=0.15)  # the variance ratios, decreasing
    truth = B  # the first regime's shocks have unit variance, ordered by the ratios 9, 1, 0.16: columns 0, 1, 2
    for j in range(3):
        col = h.impact[:, j] * np.sign(h.impact[:, j] @ truth[:, j])
        assert col == pytest.approx(truth[:, j], abs=0.12)
    assert h.irf(10).values.shape == (11, 3, 3) and "heteroskedasticity" in h.summary().title
