"""The Kalman filter's loops compiled by numba (pymetrix.models._kalman_compiled), an optional extra: checked against
the NumPy loops they replace (the log likelihood and every adjoint to 1e-10 relative) on a random system whose every
matrix is dense, with missing observations, a presample and both starts; the plain log likelihood against the native
filter (the same arithmetic, to 1e-10) and statsmodels' (an implementation apart, to its convergence tolerance); the
New Keynesian template's analytic gradient with and without them. Skipped without numba, except the check that
everything runs without it."""

import sys

import numpy as np
import pytest

import pymetrix.models
from pymetrix import statespace
from pymetrix.models import _kalman_compiled as kc
from pymetrix.models import kalman_score as ks


@pytest.fixture
def switch():
    yield
    kc.COMPILED = "auto"


def test_without_numba_the_numpy_loops_run(monkeypatch, switch):
    monkeypatch.setitem(sys.modules, "numba", None)  # import numba fails
    monkeypatch.setitem(sys.modules, "pymetrix.models._kalman_kernels", None)
    # `from . import _kalman_kernels` reads the package attribute before sys.modules, so once another test has loaded
    # the compiled kernels the import still succeeds; remove the attribute too
    monkeypatch.delattr(pymetrix.models, "_kalman_kernels", raising=False)
    monkeypatch.setattr(kc, "_TRIED", False)
    monkeypatch.setattr(kc, "_KERNELS", None)
    assert kc.kernels() is None and not kc.available() and ks._kernels() is None
    y, T, RQR, Z, d, H, P0 = _system(3)
    assert np.isfinite(ks._filter(y, T, RQR, Z, d, H, P0, 0)["loglike"])


def _system(seed, n=6, p=3, n_t=90):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(n, n))
    T *= 0.9 / np.max(np.abs(np.linalg.eigvals(T)))
    R = rng.normal(size=(n, 2))
    RQR = R @ np.diag([1.0, 0.5]) @ R.T
    Z = rng.normal(size=(p, n))
    H = np.diag(rng.uniform(0.1, 0.3, p))
    d = rng.normal(size=p)
    y = rng.normal(size=(n_t, p))
    y[5, 1] = y[40, :2] = np.nan
    y[60] = np.nan
    return y, T, RQR, Z, d, H, ks._lyapunov(T, RQR)


def _close(a, b, tol=1e-10):
    return np.max(np.abs(a - b)) <= tol * max(1.0, float(np.max(np.abs(b))))


@pytest.mark.parametrize("presample, steady", [(0, False), (4, False), (0, True)])
def test_the_compiled_filter_and_adjoint_are_the_numpy_ones(switch, presample, steady):
    pytest.importorskip("numba")
    y, T, RQR, Z, d, H, P0 = _system(1, n_t=400 if steady else 90)
    if steady:  # long enough for the variance to converge: the converged state serves the later periods
        y[300:] = np.random.default_rng(2).normal(size=(100, 3))
    out = {}
    for mode in ("off", "auto"):
        kc.COMPILED = mode
        f = ks._filter(y, T, RQR, Z, d, H, P0, presample)
        out[mode] = (f, ks._adjoint(f, T, Z))
    (f0, b0), (f1, b1) = out["off"], out["auto"]
    assert len(f1["P"]) == len(f0["P"]) and np.array_equal(f1["state"], f0["state"])
    if steady:
        assert len(f0["P"]) < 300
    assert f1["loglike"] == pytest.approx(f0["loglike"], rel=1e-10, abs=0)
    for key in ("P", "G", "Finv", "Pf", "L", "a", "v", "e"):
        assert _close(f1[key], f0[key]), key
    for key in b0:
        assert _close(b1[key], b0[key]), key


def _models():
    from pymetrix.statespace import SystemMatrices

    y, T, _, Z, d, H, _ = _system(4)
    rng = np.random.default_rng(5)
    R = rng.normal(size=(6, 2))
    c = 0.1 * rng.normal(size=6)
    return y, SystemMatrices(tuple("abcdef"), ("x", "y", "z"), ("u", "v"), T, c, R, np.diag([1.0, 0.5]), Z, d, H)


@pytest.mark.parametrize("start, presample", [("auto", 0), ("auto", 6), ("scaled", 0)])
def test_the_compiled_log_likelihood(switch, start, presample):
    pytest.importorskip("numba")
    y, m = _models()
    kc.COMPILED = "auto"
    fast = statespace.loglike(y, m, start, presample)
    kc.COMPILED = "off"
    reference = statespace.loglike(y, m, start, presample)  # statsmodels' filter
    try:
        statespace.ENGINE = "native"
        native = statespace.loglike(y, m, start, presample)
    finally:
        statespace.ENGINE = "auto"
    assert fast == pytest.approx(native, rel=1e-10, abs=0)
    assert fast == pytest.approx(reference, rel=1e-9)  # statsmodels keeps updating the converged variance


def test_degenerate_systems_have_no_likelihood(switch):
    pytest.importorskip("numba")
    from dataclasses import replace

    y, m = _models()
    kc.COMPILED = "auto"
    flat = replace(m, Q=np.zeros((2, 2)), H=np.zeros((3, 3)))  # every observation predicted with zero variance
    assert statespace.loglike(y, flat) == -np.inf


def test_the_gradient_of_the_new_keynesian_template(switch):
    pytest.importorskip("numba")
    import pandas as pd

    from pymetrix import Dataset, templates
    from pymetrix.estimation.priors import Beta, Gamma, InvGamma1, Normal

    nk = templates.new_keynesian()
    m = nk.measurement.with_errors({o: 0.2 for o in nk.measurement.observables})
    path = nk.solve().simulate(80, nk.shock_sd, seed=3)
    path.index = pd.period_range("1990Q1", periods=80, freq="Q", name="time")
    frame = m.apply(Dataset.from_frame(path)).frame
    frame.iloc[10, 0] = np.nan
    pri = {"kappa": Gamma(0.1, 0.05), "rho": Beta(0.7, 0.1), "phi_pi": Normal(1.5, 0.25), "sd_ey": InvGamma1(0.5, 1.0)}
    fit = nk.estimate(Dataset.from_frame(frame), list(pri), priors=pri, measurement=m, cov="none", max_iter=3)
    lik = fit.log_posterior.__self__
    point = dict(zip(lik.free, fit.params.values))
    out = {}
    for mode in ("off", "auto"):
        kc.COMPILED = mode
        out[mode] = (ks.loglike_score(lik, point), lik.loglike(point))
    (ll0, g0), plain0 = out["off"]
    (ll1, g1), plain1 = out["auto"]
    assert ll1 == pytest.approx(ll0, rel=1e-10, abs=0) and _close(g1, g0)
    assert plain1 == pytest.approx(ll1, rel=1e-10, abs=0)  # the compiled plain filter is the gradient's filter
    assert plain0 == pytest.approx(plain1, rel=1e-9)  # and statsmodels' to its convergence tolerance
