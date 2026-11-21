"""ARMA errors with GARCH variance, checked against other implementations.

- AR(1) errors with GARCH(1,1): the same conditional likelihood as the arch package's AR mean (log likelihood to 1e-6,
  estimates to 1e-4; arch's intercept c is our mean times 1 - phi), and the same multi-step forecasts.
- ARMA(1,1) errors with GARCH(1,1): R's fGarch 4032.91 (garchFit(~arma(1,1)+garch(1,1)), recorded below), which starts
  its recursions differently: estimates within 0.2 of a standard error, standard errors within 5%.
- Multi-step forecast variances with MA terms: the fitted process simulated forward."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.errors import SpecificationError
from pymetrix.estimation.timeseries import ARMA, GARCH


def _series(n=2000):
    rng = np.random.default_rng(5)
    e, h, u = np.zeros(n), np.full(n, 0.5), np.zeros(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.88 * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
        u[t] = 0.5 * u[t - 1] + e[t] + 0.3 * e[t - 1]
    idx = pd.period_range("1900-01", periods=n, freq="M", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": 0.2 + u}, index=idx)), idx


@pytest.fixture(scope="module")
def series():
    return _series()


def test_ar_errors_with_garch_match_arch(series):
    arch = pytest.importorskip("arch")
    data, idx = series
    y = data.frame["y"].to_numpy()
    held = Dataset.from_frame(data.frame.iloc[:-10])
    fit = Equation("y ~ 1", errors=ARMA(ar=1), variance=GARCH()).fit(held)
    theirs = arch.arch_model(y[:-10], mean="AR", lags=1, vol="GARCH", p=1, q=1).fit(disp="off", options={"ftol": 1e-12})
    ours = fit.params.as_dict()
    c, phi = theirs.params["Const"], theirs.params["y[1]"]
    assert abs(fit.stats["loglike"] - theirs.loglikelihood) < 1e-6
    assert np.allclose([ours["const"], ours["ar[1]"], ours["omega"], ours["alpha[1]"], ours["beta[1]"]],
                       [c / (1 - phi), phi, theirs.params["omega"], theirs.params["alpha[1]"], theirs.params["beta[1]"]], atol=1e-4)
    out = fit.forecast(data, span=(str(idx[-10]), str(idx[-1])), dynamic=True)
    expected = theirs.forecast(horizon=10, reindex=False)
    assert np.allclose(out.frame["forecast"].to_numpy(), expected.mean.to_numpy()[0], atol=1e-4)
    assert np.allclose(out.frame["se"].to_numpy() ** 2, expected.variance.to_numpy()[0], rtol=1e-4)


FGARCH = {  # garchFit(~arma(1,1)+garch(1,1), data=y): coefficients and standard errors (mu is the intercept)
    "mu": (0.145625100, 0.03162869), "ar1": (0.456894734, 0.02887314), "ma1": (0.395332888, 0.03006465),
    "omega": (0.034913792, 0.01178647), "alpha1": (0.061126212, 0.01139552), "beta1": (0.907220754, 0.01824493)}


def test_arma_errors_with_garch_match_fgarch(series):
    data, _ = series
    fit = Equation("y ~ 1", errors=ARMA(ar=1, ma=1), variance=GARCH()).fit(data)
    ours, se = fit.params.as_dict(), dict(zip(fit.params.names, fit.params.se))
    for name, theirs in (("ar[1]", "ar1"), ("ma[1]", "ma1"), ("omega", "omega"), ("alpha[1]", "alpha1"), ("beta[1]", "beta1")):
        value, error = FGARCH[theirs]
        assert abs(ours[name] - value) < 0.2 * error, name
        assert abs(se[name] / error - 1) < 0.05, name
    intercept = ours["const"] * (1 - ours["ar[1]"])
    assert abs(intercept - FGARCH["mu"][0]) < 0.2 * FGARCH["mu"][1]
    assert fit.extra["conditioning"] == 1 and len(fit.resid) == len(data.frame) - 1


def test_multi_step_variances_match_simulation(series):
    data, idx = series
    held = Dataset.from_frame(data.frame.iloc[:-12])
    fit = Equation("y ~ 1", errors=ARMA(ar=1, ma=1), variance=GARCH()).fit(held)
    out = fit.forecast(data, span=(str(idx[-12]), str(idx[-1])), dynamic=True)
    v = fit.params.as_dict()
    rng = np.random.default_rng(1)
    paths = 40_000
    e_last, h_last = fit.resid.iloc[-1], fit.extra["volatility"].iloc[-1] ** 2
    u_last = fit.extra["structural_resid"].iloc[-1]
    e, h, u = np.full(paths, e_last), np.full(paths, h_last), np.full(paths, u_last)
    sims = np.empty((paths, 12))
    for j in range(12):
        h = v["omega"] + v["alpha[1]"] * e ** 2 + v["beta[1]"] * h
        new = np.sqrt(h) * rng.standard_normal(paths)
        u = v["ar[1]"] * u + new + v["ma[1]"] * e
        e = new
        sims[:, j] = v["const"] + u
    assert np.allclose(out.frame["forecast"].to_numpy(), sims.mean(axis=0), atol=4 * out.frame["se"].to_numpy() / np.sqrt(paths))
    assert np.allclose(out.frame["se"].to_numpy() ** 2, sims.var(axis=0), rtol=0.05)  # GARCH's fat tails: a wider margin
    static = fit.forecast(data, span=(str(idx[-12]), str(idx[-1])))
    assert np.isclose(static.frame["se"].iloc[0], out.frame["se"].iloc[0])  # the first period: one step ahead either way
    assert np.isclose(static.frame["forecast"].iloc[0], out.frame["forecast"].iloc[0])


def test_variants_and_refusals(series):
    data, _ = series
    rng = np.random.default_rng(2)
    with_x = Dataset.from_frame(data.frame.assign(x=rng.normal(size=len(data.frame))))
    fit = Equation("y ~ x", errors=ARMA(ar=[1, 3]), variance=GARCH(asym=1, dist="t")).fit(with_x)
    assert fit.extra["converged"] and fit.extra["conditioning"] == 3
    assert {"ar[1]", "ar[3]", "gamma[1]", "nu"} <= set(fit.params.names)
    with pytest.raises(SpecificationError, match="normal or t errors"):
        Equation("y ~ 1", errors=ARMA(ar=1), variance=GARCH(dist="ged")).fit(data)
    with pytest.raises(SpecificationError, match="too few"):
        Equation("y ~ 1", errors=ARMA(ar=1), variance=GARCH()).fit(Dataset.from_frame(data.frame.iloc[:20]))
