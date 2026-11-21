"""Zero and sign restrictions (Arias, Rubio-Ramírez and Waggoner 2018).

- Every draw meets its zeros to 1e-10 (on impact, and in the long run through (I - A(1))^-1), has its signs and factors
  the error covariance; at the estimates, for bootstrap draws and for each posterior draw of a Bayesian VAR.
- Zeros alone that just identify the VAR (recursive zeros) give the Cholesky impact matrix, up to the columns' signs and
  order, on every draw; the responses then match the Cholesky responses.
- The importance weights: constant without zeros (where ARW's algorithm is the uniform one) and with recursive zeros
  (Q is fixed up to signs); the impact-only shortcut equals the general formula. Against brute force: Sigma drawn
  from an inverse Wishart and Q uniform, then kept when the structural A0 = (Sigma's Cholesky factor' )^-1 Q lies
  within a thin tube around the manifold of one zero (distance |g| / |grad g| in A0's coordinates), which tends to the
  target ARW define (the normal-generalised-normal posterior restricted to the zeros, by volume); the weighted draws
  of ARW's algorithm match the tube's quantiles of impact responses, and the unweighted draws do not.
- Variance and historical decompositions under zero and sign restrictions; refusals; the dialogs record the code."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import invwishart

from pymetrix import VAR, Bayes, Dataset, Minnesota, SignRestrictions
from pymetrix.errors import SpecificationError
from pymetrix.var import _ma, _rotations_for_draws
from pymetrix.zerosign import _scheme_rows, draw_rotation, generation_order, log_weight, zero_rows

NAMES = ["gy", "pi", "r"]
B_TRUE = np.array([[0.8, 0.0, 0.3], [0.3, 0.6, -0.2], [0.2, 0.4, 0.7]])
A1 = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.2, 0.6]])


def _simulate(T=500, seed=3):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A1 @ y[t - 1] + B_TRUE @ rng.standard_normal(3)
    idx = pd.period_range("1900Q1", periods=T, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame(y, columns=NAMES, index=idx))


@pytest.fixture(scope="module")
def data():
    return _simulate()


@pytest.fixture(scope="module")
def var(data):
    return VAR(NAMES, lags=1).fit(data)


MONETARY = SignRestrictions({"monetary": {"r": "+", "pi": "-"}}, zeros={"monetary": ["gy"]}, horizon=1, rotations=200)


def _check_draw(scheme, B, sigma, A, tol=1e-10):
    index = {v: i for i, v in enumerate(NAMES)}
    names = scheme.shock_names(NAMES)
    assert B @ B.T == pytest.approx(sigma, rel=1e-9, abs=1e-12)
    linf = np.linalg.solve(np.eye(3) - A.sum(axis=0), B)
    for k, shock in enumerate(names):
        for v in scheme.zeros.get(shock, ()):
            assert abs(B[index[v], k]) < tol
        for v in scheme.long_run_zeros.get(shock, ()):
            assert abs(linf[index[v], k]) < tol
    R = _ma(A, scheme.horizon) @ B
    for k, shock in enumerate(names):
        for v, s in scheme.signs.get(shock, {}).items():
            assert np.all(s * R[:, index[v], k] > 0)


def test_every_draw_meets_its_zeros_and_signs_at_the_estimates(var):
    sigma = np.asarray(var.sigma, float)
    sr = var.identify(MONETARY)
    draws = sr.sign_draws()
    assert len(draws) == 200
    for B in draws:
        _check_draw(MONETARY, B, sigma, var.coefs)
    long_run = SignRestrictions({"supply": {"gy": "+"}, "demand": {"gy": "+", "pi": "+"}}, long_run_zeros={"demand": ["gy"]},
                                rotations=100)
    for B in var.identify(long_run).sign_draws():
        _check_draw(long_run, B, sigma, var.coefs)
    both = SignRestrictions({"monetary": {"r": "+"}}, zeros={"monetary": ["gy"]}, long_run_zeros={"monetary": ["pi"]}, rotations=50)
    for B in var.identify(both).sign_draws():
        _check_draw(both, B, sigma, var.coefs)
    irf = sr.irf(6, draws=30)  # bands from bootstrap draws, each draw's rotation meeting its own zeros
    assert np.allclose(irf.values[0, 0, 0], 0, atol=1e-10) and np.allclose(irf.lower[0, 0, 0], 0, atol=1e-10)
    assert "Zero and sign restrictions" in irf.identification


def test_every_posterior_draw_meets_its_zeros_and_signs(data):
    bvar = VAR(NAMES, lags=1).fit(data, method=Bayes(prior=Minnesota(), draws=60))
    pairs = bvar._draw_coefs()
    chosen = _rotations_for_draws(MONETARY, pairs, NAMES, np.random.default_rng(0))
    assert len(chosen) == len(pairs)
    for i, B in chosen:
        A, S = pairs[i]
        _check_draw(MONETARY, B, S, A)
    irf = bvar.identify(MONETARY).irf(6, draws=60)
    assert np.allclose(irf.lower[0, 0, 0], 0, atol=1e-10) and np.allclose(irf.upper[0, 0, 0], 0, atol=1e-10)
    assert np.all(irf.lower[0, 2, 0] > 0)  # r rises on impact in every draw


def test_recursive_zeros_alone_give_cholesky(var):
    sigma = np.asarray(var.sigma, float)
    chol = np.linalg.cholesky(sigma)
    scheme = SignRestrictions(zeros={"pi shock": ["gy"], "r shock": ["gy", "pi"]}, rotations=50)
    sr = var.identify(scheme)
    assert sr.shocks == ["pi shock", "r shock", "unidentified 1"]
    target = chol[:, [1, 2, 0]]
    for B in sr.sign_draws():
        assert np.allclose(np.abs(B), np.abs(target), atol=1e-10)
    signed = SignRestrictions({"gy shock": {"gy": "+"}, "pi shock": {"pi": "+"}, "r shock": {"r": "+"}},
                              zeros={"pi shock": ["gy"], "r shock": ["gy", "pi"]}, rotations=50)
    irf = var.identify(signed).irf(8, draws=0)
    assert np.allclose(irf.values, var.irf(8, draws=0).values, atol=1e-10)
    assert np.allclose(var.identify(signed).fevd(8).values, var.fevd(8).values, atol=1e-10)


def _weights(scheme, sigma, A, count, rng):
    rows = zero_rows(scheme, sigma, A, NAMES)
    order = generation_order([len(r) for r in rows])
    srows = _scheme_rows(scheme, NAMES)
    out = []
    for _ in range(count):
        Q = draw_rotation(rows, order, rng)
        out.append(log_weight(rows, srows, order, sigma, A, Q))
    return np.array(out), rows, srows, order


def test_weights_are_constant_without_zeros_and_when_just_identified(var):
    sigma, A, rng = np.asarray(var.sigma, float), var.coefs, np.random.default_rng(1)
    w, *_ = _weights(SignRestrictions({"s": {"gy": "+"}}), sigma, A, 30, rng)
    assert np.ptp(w) < 1e-7
    w, *_ = _weights(SignRestrictions(zeros={"b": ["gy"], "c": ["gy", "pi"]}), sigma, A, 30, rng)
    assert np.ptp(w) < 1e-7
    w, rows, srows, order = _weights(MONETARY, sigma, A, 30, rng)
    assert np.ptp(w) > 1e-3  # one zero: the weights vary
    Q = draw_rotation(rows, order, rng)  # the impact-only shortcut is the general formula
    assert log_weight(rows, srows, order, sigma, A, Q) == pytest.approx(log_weight(rows, srows, order, sigma, A, Q, _long_run=True), abs=1e-6)


def test_importance_weights_against_brute_force():
    n, N, M = 3, 1_000_000, 2500
    psi = np.array([[1.0, 0.6, 0.3], [0.6, 1.5, 0.5], [0.3, 0.5, 0.8]]) * 4
    nu = n + 3
    A = np.zeros((1, n, n))
    scheme = SignRestrictions(zeros={"s1": ["r"]})  # shock 1 does not move the third variable on impact
    rng = np.random.default_rng(0)
    # brute force: the unconditional draws within a thin tube around the zero's manifold, in A0's coordinates
    G = rng.standard_normal((N, nu, n)) @ np.linalg.cholesky(np.linalg.inv(psi)).T  # rows N(0, psi^-1)
    S = np.linalg.inv(np.swapaxes(G, 1, 2) @ G)  # inverse Wishart(nu, psi)
    X = rng.standard_normal((N, n, n))
    Qh, R = np.linalg.qr(X)
    Qh = Qh * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]  # Haar
    L0 = np.linalg.cholesky(S) @ Qh
    a = L0[:, 2, :]  # A0^-1 e_3 = L0' e_3: the third row of L0
    dist = np.abs(L0[:, 2, 0]) / (np.linalg.norm(a, axis=1) * np.linalg.norm(L0[:, :, 0], axis=1))
    tube = L0[dist < np.quantile(dist, 0.004)]
    # ARW: Sigma from the same inverse Wishart, Q column by column, importance weights
    S2 = invwishart.rvs(nu, psi, size=M, random_state=2)
    draws, logw = [], []
    for sigma in S2:
        rows = zero_rows(scheme, sigma, A, NAMES)
        order = generation_order([len(r) for r in rows])
        Q = draw_rotation(rows, order, rng)
        draws.append(np.linalg.cholesky(sigma) @ Q)
        logw.append(log_weight(rows, _scheme_rows(scheme, NAMES), order, sigma, A, Q))
    draws = np.array(draws)
    w = np.exp(np.array(logw) - max(logw))
    w /= w.sum()

    def weighted_quantile(x, q):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return np.interp(q, c - w[o] / 2, x[o])

    for i, j in ((0, 0), (1, 0), (0, 1)):  # squared impact responses (the columns' signs are arbitrary)
        brute, arw = tube[:, i, j] ** 2, draws[:, i, j] ** 2
        for q in (0.25, 0.5, 0.75):
            b, a_w, a_u = np.quantile(brute, q), weighted_quantile(arw, q), np.quantile(arw, q)
            assert abs(a_w - b) < 0.12 * b + 0.01, (i, j, q, b, a_w)
    b, a_w, a_u = np.quantile(tube[:, 1, 0] ** 2, 0.5), weighted_quantile(draws[:, 1, 0] ** 2, 0.5), np.quantile(draws[:, 1, 0] ** 2, 0.5)
    assert abs(a_u - b) > 2 * abs(a_w - b)  # without the weights the distribution is off


def test_decompositions_under_zero_and_sign_restrictions(var):
    sr = var.identify(MONETARY)
    fevd = sr.fevd(8)
    assert fevd.values[0, 0, 0] == pytest.approx(0, abs=1e-12)  # no share of gy's one-step variance
    hd = sr.historical_decomposition(draws=50)
    assert np.allclose(hd.contributions["gy"]["monetary"].iloc[0], 0, atol=1e-10)
    total = hd.baseline + pd.DataFrame({v: hd.contributions[v].sum(axis=1) for v in NAMES})
    assert np.allclose(total.to_numpy(), var.arrays.Y, atol=1e-10)
    assert np.allclose(hd.lower["gy"]["monetary"].iloc[0], 0, atol=1e-10)


def test_restrictions_that_cannot_hold_are_refused(var):
    with pytest.raises(SpecificationError, match="too many zero restrictions"):
        var.identify(SignRestrictions(zeros={"a": ["gy", "pi"], "b": ["gy", "pi"]}))
    with pytest.raises(SpecificationError, match="both a sign and a zero"):
        var.identify(SignRestrictions({"a": {"gy": "+"}}, zeros={"a": ["gy"]}))
    with pytest.raises(SpecificationError, match="does not have"):
        var.identify(SignRestrictions({"a": {"gy": "+"}}, zeros={"a": ["output"]}))
    with pytest.raises(SpecificationError, match="at least one shock"):
        SignRestrictions()


def test_the_dialogs_record_the_code(data):
    from pymetrix.commands import FitVAR, ImpulseResponse, Session, VarianceDecomposition

    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("gy, pi, r", lags="1", name="var"))
    s.run(ImpulseResponse("var", periods=6, scheme="SignRestrictions", signs="monetary: r+, pi-", zeros="monetary: gy",
                          rotations=50, draws=0))
    s.run(VarianceDecomposition("var", periods=6, scheme="SignRestrictions", zeros="b: gy; c: gy, pi", rotations=20))
    assert "zeros={'monetary': ['gy']}" in s.history[1].code
    assert list(s.objects["fevd"].shocks) == ["b", "c", "unidentified 1"]
    assert np.allclose(s.objects["irf"].values[0, 0, 0], 0, atol=1e-10)
    namespace = {"data": s.objects["data"]}
    exec(s.script(), namespace)  # noqa: S102 - the script identifies the same way
    assert np.allclose(namespace["irf"].values, s.objects["irf"].values)
