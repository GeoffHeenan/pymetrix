"""Correlated measurement errors: ``corr y, c = r`` (or ``var y, c = cov``) on two observed variables in a Dynare shocks
block, ``corr y, c, ...`` in estimated_params (estimated as ``corr_y_c``), and ``MeasurementMap(error_corr=...)``.

Checked against: Dynare 6.0's log likelihood on the same file and data (tests/fixtures/dynare-me-corr-priors.json, fixed
and estimated correlations), a Kalman filter written out here with the covariance ``D C D`` built by hand, the analytic
gradient against five-point differences, and a correlation estimated from simulated data against the one that made them.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, read_mod
from pymetrix.metadata import Role
from pymetrix.models import MeasurementMap, ModelError

FIXTURES = Path(__file__).parent / "fixtures"
MOD = FIXTURES / "dynare" / "me_corr_priors.mod"
RECORDED = json.loads((FIXTURES / "dynare-me-corr-priors.json").read_text(encoding="utf-8"))
ESTIMATED = MOD.read_text(encoding="utf-8").split("estimated_params;")[1].split("end;")[0]
FIXED = MOD.read_text(encoding="utf-8").replace("var c; stderr 0.2;", "var c; stderr 0.2;\ncorr y, c = 0.4;").replace(
    ESTIMATED, "\nrho, 0.65;\nalpha, 0.55;\n")


def _kalman(y, rho, alpha, sd_e, H):
    """The log likelihood of x = rho x[-1] + e, (y, c) = (x, alpha x) + errors with covariance H, from the stationary start."""
    Z = np.array([[1.0], [alpha]])
    a, P, out = np.zeros(1), np.array([[sd_e**2 / (1 - rho**2)]]), 0.0
    for row in y:
        v = row - Z @ a
        F = Z @ P @ Z.T + H
        out -= 0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(F)[1] + v @ np.linalg.solve(F, v))
        K = P @ Z.T @ np.linalg.inv(F)
        a, P = rho * (a + K @ v), rho * (P - K @ Z @ P) * rho + sd_e**2
    return out


def _cov(sd_y, sd_c, r):
    return np.array([[sd_y**2, r * sd_y * sd_c], [r * sd_y * sd_c, sd_c**2]])


def test_the_shocks_block_reads_correlations_of_measurement_errors():
    dyn = read_mod(FIXED)
    assert dyn.measurement_corr == {("y", "c"): 0.4} and dyn.shock_corr == {}
    assert dyn.measurement_sd == {"y": 0.3, "c": 0.2}
    assert read_mod(FIXED.replace("corr y, c", "corr c, y")).measurement_corr == {("y", "c"): 0.4}  # varobs order
    cov = read_mod(FIXED.replace("corr y, c = 0.4;", "var y, c = 0.024;"))
    assert cov.measurement_corr == {("y", "c"): pytest.approx(0.4)}  # covariance / (sd_y sd_c)
    m = dyn._measurement(dyn.start)
    assert m.error_corr == {("y", "c"): 0.4}
    np.testing.assert_allclose(m.error_covariance(), _cov(0.3, 0.2, 0.4))


def test_fixed_correlation_matches_dynare_and_the_hand_filter():
    dyn = read_mod(FIXED)
    data = dyn.data(MOD.parent / "me_corr_data.m")
    assert dyn.log_posterior(data=data) == pytest.approx(RECORDED["fixed_corr_loglike"], abs=1e-6)  # Dynare switches to the steady-state filter
    y = data.frame[["y", "c"]].to_numpy()
    assert dyn.log_posterior(data=data) == pytest.approx(_kalman(y, 0.65, 0.55, 0.5, _cov(0.3, 0.2, 0.4)), abs=1e-9)
    uncorrelated = read_mod(FIXED.replace("corr y, c = 0.4;", ""))
    assert uncorrelated.log_posterior(data=data) == pytest.approx(_kalman(y, 0.65, 0.55, 0.5, _cov(0.3, 0.2, 0.0)), abs=1e-9)


def test_estimated_correlation_matches_dynare():
    """Dynare's log posterior less its log priors is the log likelihood at the initial values, corr_y_c = 0.35 among them.
    (The priors are compared in test_dynare_generalised_priors.)"""
    dyn = read_mod(MOD)
    assert dyn.estimated == ("rho", "alpha", "sd_e", "sd_y", "corr_y_c") and dyn.bounds["corr_y_c"] == (-1.0, 1.0)
    loglike = dyn.log_posterior() - sum(p.logpdf(dyn.start[n]) for n, p in dyn.priors.items())
    dynare = RECORDED["logpost"] - sum(RECORDED["log_prior"].values())
    assert loglike == pytest.approx(dynare, abs=1e-6)
    y = dyn.data().frame[["y", "c"]].to_numpy()
    assert loglike == pytest.approx(_kalman(y, 0.65, 0.55, 0.45, _cov(0.35, 0.2, 0.35)), abs=1e-8)


def _simulated(n, r, seed=3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    errors = rng.standard_normal((n, 2)) @ np.linalg.cholesky(_cov(0.3, 0.2, r)).T
    frame = pd.DataFrame({"y": x + errors[:, 0], "c": 0.5 * x + errors[:, 1]},
                         index=pd.period_range("1950Q1", periods=n, freq="Q", name="time"))
    return Dataset.from_frame(frame)


def _model():
    return Model([Equation("x ~ rho*x[-1] + e", params=["rho"]), Equation("y = x"), Equation("c ~ alpha*x", params=["alpha"])],
                 params={"rho": 0.6, "alpha": 0.4}, roles={"e": Role.SHOCK})


def test_a_correlation_estimated_from_simulated_data():
    data = _simulated(1500, -0.5)
    m = MeasurementMap.direct(["y", "c"], {"y": 0.3, "c": 0.2})
    fit = _model().estimate(data, m, shocks=["e"], shock_sd={"e": 0.5}, estimate=["rho", "alpha", "sd_e", "sd_y", "sd_c", "corr_y_c"])
    est, se = fit.params.as_dict()["corr_y_c"], dict(zip(fit.params.names, fit.params.se))["corr_y_c"]
    assert abs(est - (-0.5)) < 3 * se
    assert fit.measurement.error_corr == {("y", "c"): pytest.approx(est)}  # the fit's map carries the estimate
    numerical = _model().estimate(data, m, shocks=["e"], shock_sd={"e": 0.5}, gradient="numerical",
                                  estimate=["rho", "alpha", "sd_e", "sd_y", "sd_c", "corr_y_c"])
    assert numerical.stats["loglike"] == pytest.approx(fit.stats["loglike"], abs=1e-5)


def test_the_analytic_gradient_with_correlated_errors():
    from pymetrix.models.kalman_score import loglike_score

    data = _simulated(200, 0.4)
    m = MeasurementMap.direct(["y", "c"], {"y": 0.3, "c": 0.2}, {("y", "c"): 0.4})
    fit = _model().estimate(data, m, shocks=["e"], shock_sd={"e": 0.5}, estimate=["rho", "alpha", "sd_e", "sd_y", "sd_c", "corr_y_c"])
    lik = fit.log_posterior.__self__
    point = {"rho": 0.65, "alpha": 0.45, "sd_e": 0.55, "sd_y": 0.35, "sd_c": 0.25, "corr_y_c": 0.3}
    ll, g = loglike_score(lik, point)
    assert ll == pytest.approx(lik.loglike(point), rel=1e-9)
    ref = []
    for n in lik.free:  # five-point differences
        h = 1e-4 * abs(point[n])
        f = [lik.loglike({**point, n: point[n] + k * h}) for k in (-2, -1, 1, 2)]
        ref.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
    np.testing.assert_allclose(g, ref, rtol=1e-6)
    fixed = {k: v for k, v in point.items() if k != "corr_y_c"}  # sd_<series> with a fixed correlation: H's row and column move
    fit2 = _model().estimate(data, m, shocks=["e"], shock_sd={"e": 0.5}, estimate=list(fixed))
    lik2 = fit2.log_posterior.__self__
    _, g2 = loglike_score(lik2, fixed)
    i = list(lik2.free).index("sd_y")
    h = 1e-4
    num = (lik2.loglike({**fixed, "sd_y": 0.35 + h}) - lik2.loglike({**fixed, "sd_y": 0.35 - h})) / (2 * h)
    assert g2[i] == pytest.approx(num, rel=1e-6)


def test_refusals():
    with pytest.raises(ModelError, match="not in varobs"):
        read_mod(FIXED.replace("corr y, c = 0.4;", "corr y, x = 0.4;"))
    with pytest.raises(ModelError, match="outside"):
        read_mod(FIXED.replace("corr y, c = 0.4;", "corr y, c = 1.4;"))
    with pytest.raises(ModelError, match="needs their variances"):
        read_mod(FIXED.replace("corr y, c = 0.4;", "var y, c = 0.01;").replace("var c; stderr 0.2;", ""))
    with pytest.raises(ModelError, match="not an observed series"):
        MeasurementMap.direct(["y", "c"], error_corr={("y", "z"): 0.2})
    with pytest.raises(ModelError, match="not a valid correlation matrix"):
        MeasurementMap.direct(["a", "b", "c"], {"a": 1, "b": 1, "c": 1},
                              {("a", "b"): 0.9, ("a", "c"): 0.9, ("b", "c"): -0.9}).error_covariance()
    m = MeasurementMap.direct(["y", "c"], {"y": 0.3, "c": 0.2}, {("y", "c"): 0.4})
    sol = _model().linearize(shocks=["e"]).solve()
    with pytest.raises(ModelError, match="correlated measurement errors"):
        sol.particle_filter(_simulated(50, 0.4), m, {"e": 0.5}, particles=100)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
