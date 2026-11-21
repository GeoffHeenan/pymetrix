"""GMM with user moments, checked independently: one-step GMM is two-stage least squares (pymetrix's IV), two-step
efficient GMM matches linearmodels' IVGMM, J by hand, and a simulated consumption Euler equation's beta and gamma
are recovered, the over-identifying restrictions not rejected."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import GMM, IV, Dataset, Equation
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def iv_data():
    rng = np.random.default_rng(3)
    n = 800
    z1, z2, w = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.3 * w + v
    u = (0.6 * v + rng.normal(size=n)) * (1 + 0.5 * np.abs(z1))  # endogenous and heteroskedastic
    y = 1.0 + 2.0 * x - 0.5 * w + u
    return Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "w": w, "z1": z1, "z2": z2}), entity="id")


def test_one_step_gmm_is_two_stage_least_squares(iv_data):
    gmm = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"], weights="optimal", steps="two"))
    first = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"], weights="identity"))
    tsls = Equation("y ~ x + w").fit(iv_data, method=IV(endog=["x"], instruments=["z1", "z2"]))
    from pymetrix.estimation.gmm import _minimise, _moments

    m = GMM(instruments=["w", "z1", "z2"])
    mo = _moments(m, Equation("y ~ x + w"), iv_data, None, None, None, "drop")
    one = _minimise(mo, np.linalg.inv(mo.Z.T @ mo.Z / len(mo.index)), mo.start, True)
    assert one == pytest.approx(tsls.params.values, rel=1e-10)  # the first step is 2SLS
    assert gmm.params["x"] == pytest.approx(2.0, abs=0.15) and first.stats["steps"] == 1


def test_two_step_gmm_matches_linearmodels(iv_data):
    pytest.importorskip("linearmodels")
    from linearmodels.iv import IVGMM

    f = iv_data.frame.reset_index()
    ref = IVGMM(f["y"], pd.DataFrame({"const": 1.0, "w": f["w"]}), f[["x"]], f[["z1", "z2"]]).fit(cov_type="robust")
    ours = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"]))
    theirs = ref.params.reindex(["const", "x", "w"]).to_numpy()
    assert ours.params.values == pytest.approx(theirs, rel=1e-8)
    assert ours.params.se == pytest.approx(ref.std_errors.reindex(["const", "x", "w"]).to_numpy(), rel=1e-6)
    assert ours.stats["J"] == pytest.approx(ref.j_stat.stat, rel=1e-6) and ours.stats["J_df"] == 1


def test_j_by_hand_and_iterated_weighting(iv_data):
    fit = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"], steps="iterated"))  # W settles at S^-1
    Z = fit.extra["instruments"].to_numpy(float)
    u = fit.resid.to_numpy(float)
    n = len(u)
    g = Z * u[:, None]
    S = g.T @ g / n
    gbar = g.mean(axis=0)
    assert fit.stats["J"] == pytest.approx(n * gbar @ np.linalg.solve(S, gbar), rel=1e-6)  # n gbar' S^-1 gbar
    two = Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w", "z1", "z2"]))
    assert fit.stats["steps"] > 2 and fit.params["x"] == pytest.approx(two.params["x"], abs=0.02)


def test_a_consumption_euler_equation():
    # E[beta (c_t / c_t-1)^-gamma R_t - 1 | z_t-1] = 0 by construction: R_t = g_t^gamma / beta * exp(e_t - s^2 / 2)
    rng = np.random.default_rng(8)
    T, beta, gamma, s = 4000, 0.98, 2.0, 0.05
    lg = np.zeros(T)
    for t in range(1, T):
        lg[t] = 0.005 + 0.4 * lg[t - 1] + 0.01 * rng.normal()  # predictable consumption growth
    g = np.exp(lg)
    R = g**gamma / beta * np.exp(s * rng.normal(size=T) - s**2 / 2)
    c = np.cumprod(g)
    data = Dataset.from_frame(pd.DataFrame({"c": c, "R": R}, index=pd.period_range("1000Q1", periods=T, freq="Q", name="time")))
    euler = Equation("1 ~ beta * (c / c[-1])^(-gamma) * R", params=["beta", "gamma"])  # ~: estimated (= is an identity)
    fit = euler.fit(data, method=GMM(instruments=["c[-1]/c[-2]", "R[-1]", "c[-2]/c[-3]"], cov="hac", start={"beta": 0.95, "gamma": 1.0}))
    assert fit.params["beta"] == pytest.approx(beta, abs=0.01) and fit.params["gamma"] == pytest.approx(gamma, abs=0.5)
    assert fit.stats["J_pvalue"] > 0.01 and fit.stats["J_df"] == 2  # the restrictions hold
    assert any("Hansen's J" in note for note in fit.summary().notes)


def test_refusals(iv_data):
    with pytest.raises(SpecificationError, match="underidentified"):
        Equation("y ~ x + w").fit(iv_data, method=GMM(instruments=["w"]))
    with pytest.raises(SpecificationError, match="needs instruments"):
        Equation("y ~ x").fit(iv_data, method=GMM())
    with pytest.raises(SpecificationError, match="time series"):
        Equation("y ~ x").fit(iv_data, method=GMM(instruments=["z1"], cov="hac"))
