"""GMM flavours, checked independently: the continuously updated estimator with homoskedastic S equal to LIML
(linearmodels' IVLIML) and its objective by hand; the C test by hand (Hayashi's form) and near linearmodels' IVGMM c_stat; the Anderson-Rubin
F by hand, its size under weak instruments (where the Wald test over-rejects) and its confidence set; the Stock-Wright
S statistic as the CUE objective by hand; and minimum distance in the linear case equal to generalised least squares."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from pymetrix import GMM, IV, Dataset, Equation
from pymetrix.errors import SpecificationError
from pymetrix.estimation.gmm import anderson_rubin, anderson_rubin_set, c_test, minimum_distance, stock_wright


def _iv(n=800, strength=0.8, seed=3, hetero=True):
    rng = np.random.default_rng(seed)
    z1, z2, w = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    v = rng.normal(size=n)
    x = strength * z1 + strength * 0.6 * z2 + 0.3 * w + v
    u = (0.6 * v + rng.normal(size=n)) * ((1 + 0.5 * np.abs(z1)) if hetero else 1)
    y = 1.0 + 2.0 * x - 0.5 * w + u
    return Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "w": w, "z1": z1, "z2": z2}), entity="id")


@pytest.fixture(scope="module")
def iv_data():
    return _iv()


def test_cue_with_classic_s_is_liml(iv_data):
    pytest.importorskip("linearmodels")
    from linearmodels.iv import IVLIML

    f = iv_data.frame.reset_index()
    ref = IVLIML(f["y"], pd.DataFrame({"const": 1.0, "w": f["w"]}), f[["x"]], f[["z1", "z2"]]).fit(cov_type="unadjusted")
    cue = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"], steps="cue", cov="classic"))
    assert cue.params.values == pytest.approx(ref.params.reindex(["const", "x", "w"]).to_numpy(), rel=1e-6)


def test_the_cue_minimises_its_objective(iv_data):
    m = GMM(instruments=["w", "z1", "z2"], steps="cue")
    cue = Equation("y ~ x + w").fit(iv_data, method=m)
    two = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"]))
    Z = cue.extra["instruments"].to_numpy(float)

    def objective(u):  # n gbar' S(theta)^-1 gbar, S recomputed at theta
        g = Z * u[:, None]
        return len(u) * g.mean(axis=0) @ np.linalg.solve(g.T @ g / len(u), g.mean(axis=0))

    assert objective(cue.resid.to_numpy(float)) <= objective(two.resid.to_numpy(float)) + 1e-10
    assert cue.stats["J"] == pytest.approx(objective(cue.resid.to_numpy(float)), rel=1e-6)
    assert cue.params["x"] == pytest.approx(two.params["x"], abs=0.05)
    with pytest.raises(SpecificationError, match="optimal"):
        Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"], steps="cue", weights="identity"))


def test_c_test_against_linearmodels():
    pytest.importorskip("linearmodels")
    from linearmodels.iv import IVGMM

    rng = np.random.default_rng(5)
    n = 1000
    z1, z2, w, v = (rng.normal(size=n) for _ in range(4))
    x = 0.8 * z1 + 0.5 * z2 + 0.3 * w + v
    u = 0.3 * v + rng.normal(size=n)  # x mildly endogenous
    y = 1.0 + 2.0 * x - 0.5 * w + u
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "w": w, "z1": z1, "z2": z2}), entity="id")
    f = data.frame.reset_index()
    ref = IVGMM(f["y"], pd.DataFrame({"const": 1.0, "w": f["w"]}), f[["x"]], f[["z1", "z2"]]).fit(cov_type="robust")
    theirs = ref.c_stat(["x"])  # linearmodels' C tests that an endogenous regressor is exogenous: x as its own instrument
    ours = c_test(Equation("y ~ x + w"), data, GMM(instruments=["w", "z1", "z2", "x"]), ["x"])
    assert ours.statistic == pytest.approx(theirs.stat, rel=5e-3) and ours.df == 1  # linearmodels weights the subset by a block of S^-1 (0.2% apart in 7.1)

    Z = np.column_stack([np.ones(n), w, z1, z2, x])  # by hand (Hayashi 2000, 3.6): S from the two-step residuals, then
    X = np.column_stack([np.ones(n), x, w])          # both J statistics minimised with it (the block for the subset)

    def efficient(Zm, Sinv):
        A = X.T @ Zm @ Sinv @ Zm.T
        b = np.linalg.solve(A @ X, A @ y)
        g = Zm.T @ (y - X @ b) / n
        return b, n * g @ Sinv @ g

    def S_at(b):
        g = Z * (y - X @ b)[:, None]
        return g.T @ g / n

    b, _ = efficient(Z, np.linalg.inv(Z.T @ Z / n))
    b, _ = efficient(Z, np.linalg.inv(S_at(b)))
    S = S_at(b)
    _, J_all = efficient(Z, np.linalg.inv(S))
    _, J_sub = efficient(Z[:, :4], np.linalg.inv(S[:4, :4]))
    assert ours.statistic == pytest.approx(J_all - J_sub, rel=1e-8)
    with pytest.raises(SpecificationError, match="not among"):
        c_test(Equation("y ~ x + w"), data, GMM(instruments=["w", "z1", "z2"]), ["q"])


def test_c_test_finds_an_invalid_instrument():
    rng = np.random.default_rng(6)
    n = 1000
    z1, z2, z3, w, v, e = (rng.normal(size=n) for _ in range(6))
    x = 0.8 * z1 + 0.5 * z2 + 0.4 * z3 + 0.3 * w + v
    u = 0.6 * v + e
    z3 = z3 + 0.3 * e  # correlated with the error: invalid
    y = 1.0 + 2.0 * x - 0.5 * w + u
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "w": w, "z1": z1, "z2": z2, "z3": z3}), entity="id")
    m = GMM(instruments=["w", "z1", "z2", "z3"])
    assert c_test(Equation("y ~ x + w"), data, m, ["z3"]).p_value < 0.01
    assert c_test(Equation("y ~ x + w"), data, m, ["z2"]).statistic >= 0  # same S, so never negative


def test_anderson_rubin_by_hand(iv_data):
    eq = Equation("y ~ x + w")
    ar = anderson_rubin(eq, iv_data, "x", ["z1", "z2"], 2.0)
    f = iv_data.frame
    yt = f["y"].to_numpy() - 2.0 * f["x"].to_numpy()
    X1 = np.column_stack([np.ones(len(f)), f["w"]])
    R = np.column_stack([X1, f[["z1", "z2"]]])
    e0 = yt - X1 @ np.linalg.lstsq(X1, yt, rcond=None)[0]
    e = yt - R @ np.linalg.lstsq(R, yt, rcond=None)[0]
    F = ((e0 @ e0 - e @ e) / 2) / (e @ e / (len(f) - 4))
    assert ar.statistic == pytest.approx(F, rel=1e-10) and ar.p_value == pytest.approx(sps.f.sf(F, 2, len(f) - 4), rel=1e-10)
    assert anderson_rubin(eq, iv_data, "x", ["z1", "z2"], 2.0, cov="white").p_value > 0.01
    assert anderson_rubin(eq, iv_data, "x", ["z1", "z2"], 2.5).p_value < 1e-6


def test_anderson_rubin_keeps_its_size_under_weak_instruments():
    ar_rejects = wald_rejects = 0
    trials = 200
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        n = 200
        z1, z2, v = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        x = 0.05 * z1 + 0.05 * z2 + v  # nearly irrelevant instruments
        y = 1.0 + 2.0 * x + 0.9 * v + 0.45 * rng.normal(size=n)  # strongly endogenous
        data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "z1": z1, "z2": z2}), entity="id")
        ar_rejects += anderson_rubin(Equation("y ~ x"), data, "x", ["z1", "z2"], 2.0).p_value < 0.05
        fit = Equation("y ~ x").fit(data, method=IV(endog=["x"], instruments=["z1", "z2"]))
        wald_rejects += abs((fit.params["x"] - 2.0) / fit.params.se[1]) > 1.96
    assert ar_rejects / trials == pytest.approx(0.05, abs=0.04)  # exact size
    assert wald_rejects / trials > 0.2  # the usual t test is badly over-sized


def test_the_anderson_rubin_set_and_stock_wright(iv_data):
    eq = Equation("y ~ x + w")
    grid = np.linspace(1.5, 2.5, 101)
    s = anderson_rubin_set(eq, iv_data, "x", ["z1", "z2"], grid)
    inside = s.loc[s["in set"], "value"]
    assert inside.min() < 2.0 < inside.max() and inside.max() - inside.min() < 0.6
    m = GMM(instruments=["w", "z1", "z2"])
    sw = stock_wright(eq, iv_data, m, {"const": 1.0, "x": 2.0, "w": -0.5})
    f = iv_data.frame
    Z = np.column_stack([np.ones(len(f)), f[["w", "z1", "z2"]]])
    u = f["y"].to_numpy() - 1.0 - 2.0 * f["x"].to_numpy() + 0.5 * f["w"].to_numpy()
    g = Z * u[:, None]
    assert sw.statistic == pytest.approx(len(u) * g.mean(0) @ np.linalg.solve(g.T @ g / len(u), g.mean(0)), rel=1e-10)
    assert sw.df == 4 and sw.p_value > 0.01
    with pytest.raises(SpecificationError, match="every parameter"):
        stock_wright(eq, iv_data, m, {"x": 2.0})


def test_minimum_distance_in_the_linear_case_is_gls():
    rng = np.random.default_rng(7)
    H = rng.normal(size=(6, 2))
    A = rng.normal(size=(6, 6))
    V = A @ A.T / 6 + 0.5 * np.eye(6)
    pi = H @ np.array([1.0, -0.5]) + np.linalg.cholesky(V) @ rng.normal(size=6)
    md = minimum_distance(pi, V, lambda th: H @ th, [0.0, 0.0], names=["a", "b"])
    Vi = np.linalg.inv(V)
    gls = np.linalg.solve(H.T @ Vi @ H, H.T @ Vi @ pi)
    d = pi - H @ gls
    assert md.values == pytest.approx(gls, rel=1e-8) and md.cov == pytest.approx(np.linalg.inv(H.T @ Vi @ H), rel=1e-6)
    assert md.distance == pytest.approx(d @ Vi @ d, rel=1e-8) and md.df == 4
    assert md.summary().title == "Minimum distance"
    nonlinear = minimum_distance([0.5, 0.25], np.eye(2) * 0.01, lambda th: [th[0], th[0] ** 2], [0.1])  # pi2 = pi1^2 holds
    assert nonlinear.values[0] == pytest.approx(0.5, abs=1e-6) and nonlinear.distance == pytest.approx(0, abs=1e-10)
    with pytest.raises(SpecificationError, match="not identified"):
        minimum_distance([1.0], [[1.0]], lambda th: [th[0] + th[1]], [0.0, 0.0])
