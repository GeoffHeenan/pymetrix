"""Nonparametric estimation, checked independently: densities and Silverman's and Scott's bandwidths against
statsmodels' KDEUnivariate and bandwidth rules, the least-squares cross-validated bandwidth against KDEMultivariate's
cv_ls; Nadaraya-Watson and local-linear regressions in two regressors, their gradients and R^2 against KernelReg at
the same bandwidths, and the cross-validated bandwidths against KernelReg's cv_ls; Robinson's partially linear
estimator by hand from KernelReg's conditional means, recovering a simulated slope; a nonparametric autoregression
equal to kernel regression on the lags, tracking a nonlinear AR's conditional mean; and the semiparametric GARCH's QML
step against arch, with its kernel step nearer the truth than QML (root mean squared error of omega and beta) under t shocks."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, kde, kernel_regression, nonparametric_autoregression, partially_linear, semiparametric_garch
from pymetrix.errors import SpecificationError

pytest.importorskip("statsmodels")
from statsmodels.nonparametric.bandwidths import bw_scott, bw_silverman  # noqa: E402
from statsmodels.nonparametric.kde import KDEUnivariate  # noqa: E402
from statsmodels.nonparametric.kernel_density import KDEMultivariate  # noqa: E402
from statsmodels.nonparametric.kernel_regression import KernelReg  # noqa: E402


def _frame(**cols):
    n = len(next(iter(cols.values())))
    return Dataset.from_frame(pd.DataFrame({"id": range(n), **cols}), entity="id")


@pytest.fixture(scope="module")
def bimodal():
    rng = np.random.default_rng(0)
    return np.r_[rng.normal(size=200), rng.normal(3, 0.5, 100)]


def test_densities_against_statsmodels(bimodal):
    data = _frame(x=bimodal)
    k = kde(data, "x")
    ref = KDEUnivariate(bimodal)
    ref.fit(bw="silverman", fft=False)
    assert k.bandwidth == pytest.approx(bw_silverman(bimodal), rel=1e-12)
    assert kde(data, "x", bandwidth="scott").bandwidth == pytest.approx(bw_scott(bimodal), rel=1e-12)
    points = np.linspace(-3, 5, 9)
    assert k(points) == pytest.approx(ref.evaluate(points), abs=1e-12)
    assert np.sum(np.diff(k.grid) * (k.density[1:] + k.density[:-1]) / 2) == pytest.approx(1, abs=1e-4)
    cv = kde(data, "x", bandwidth="cv")
    assert cv.bandwidth == pytest.approx(KDEMultivariate(bimodal, var_type="c", bw="cv_ls").bw[0], rel=0.01)
    epa = kde(data, "x", bandwidth=0.5, kernel="epanechnikov")
    u = (1.0 - bimodal) / 0.5
    assert epa(1.0)[0] == pytest.approx(np.mean(np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)) / 0.5, rel=1e-12)


@pytest.fixture(scope="module")
def surface():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (200, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=200)
    return _frame(y=y, a=X[:, 0], b=X[:, 1]), X, y


@pytest.mark.parametrize("kind", ["lc", "ll"])
def test_kernel_regression_against_kernelreg(surface, kind):
    data, X, y = surface
    ours = kernel_regression(data, "y", ["a", "b"], kind=kind, bandwidth=[0.4, 0.5])
    ref = KernelReg(y, X, "cc", reg_type=kind, bw=[0.4, 0.5])
    mean, mfx = ref.fit()
    assert ours.fitted.to_numpy() == pytest.approx(mean, abs=1e-12) and ours.r2 == pytest.approx(ref.r_squared(), rel=1e-10)
    if kind == "ll":
        assert ours.gradient.to_numpy() == pytest.approx(mfx, abs=1e-10)
    at = np.array([[0.3, -0.7]])
    assert ours.predict(at)["mean"].iloc[0] == pytest.approx(ref.fit(at)[0][0], abs=1e-12)


def test_cross_validated_bandwidths_against_kernelreg(surface):
    data, X, y = surface
    ours = kernel_regression(data, "y", ["a", "b"], kind="ll")
    ref = KernelReg(y, X, "cc", reg_type="ll", bw="cv_ls")
    assert ours.bandwidth == pytest.approx(ref.bw, rel=0.01)


def test_partially_linear():
    rng = np.random.default_rng(2)
    n = 400
    z = rng.uniform(-2, 2, n)
    x = 0.5 * z + rng.normal(size=n)
    y = 1.5 * x + np.sin(2 * z) + 0.5 * rng.normal(size=n)
    data = _frame(y=y, x=x, z=z)
    fit = partially_linear(data, "y", ["x"], ["z"], bandwidth=0.2)
    my = KernelReg(y, z, "c", reg_type="lc", bw=[0.2]).fit()[0]
    mx = KernelReg(x, z, "c", reg_type="lc", bw=[0.2]).fit()[0]
    assert fit.params[0] == pytest.approx(np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2), rel=1e-12)
    cv = partially_linear(data, "y", ["x"], ["z"])
    assert cv.params[0] == pytest.approx(1.5, abs=3 * cv.se[0]) and cv.summary().title.startswith("Partially linear")
    assert np.corrcoef(cv.g, np.sin(2 * z))[0, 1] > 0.9


def test_nonparametric_autoregression():
    rng = np.random.default_rng(3)
    T = 500
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * np.tanh(2 * y[t - 1]) + 0.4 * rng.normal()  # a nonlinear AR(1)
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1900Q1", periods=T, freq="Q")))
    ar = nonparametric_autoregression(data, "y", lags=1, bandwidth=0.3)
    by_hand = KernelReg(y[1:], y[:-1], "c", reg_type="ll", bw=[0.3]).fit()[0]
    assert ar.fitted.to_numpy() == pytest.approx(by_hand, abs=1e-12)
    cv = nonparametric_autoregression(data, "y", lags=1)
    assert np.corrcoef(cv.fitted, 0.8 * np.tanh(2 * y[:-1]))[0, 1] > 0.98
    with pytest.raises(SpecificationError, match="lags"):
        nonparametric_autoregression(data, "y", lags=0)


def _garch_returns(seed, T=1500):
    rng = np.random.default_rng(seed)
    e = rng.standard_t(5, T) / np.sqrt(5 / 3)
    h, r = np.empty(T), np.empty(T)
    h[0] = 1
    for t in range(T):
        if t:
            h[t] = 0.05 + 0.1 * r[t - 1] ** 2 + 0.85 * h[t - 1]
        r[t] = np.sqrt(h[t]) * e[t]
    return Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2000-01-01", periods=T, freq="D"))), r


def test_semiparametric_garch():
    arch = pytest.importorskip("arch")
    data, r = _garch_returns(0, T=2000)
    fit = semiparametric_garch(data, "r")
    ref = arch.arch_model(r, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
    assert fit.qml == pytest.approx(ref.params.to_numpy(), abs=2e-3)
    assert fit.qml_se == pytest.approx(ref.std_err.to_numpy(), rel=0.1)
    assert np.all(np.isnan(fit.se)) and "draws=" in fit.summary().notes[0]
    semi, qml = [], []
    for seed in range(20):  # the kernel step uses the fat tails: fewer wild estimates of omega and beta than QML
        f = semiparametric_garch(_garch_returns(900 + seed)[0], "r")
        semi.append(f.params[[1, 3]])
        qml.append(f.qml[[1, 3]])
    true = np.array([0.05, 0.85])
    rmse = lambda est: np.sqrt(np.mean((np.array(est) - true) ** 2, axis=0))  # noqa: E731
    assert np.all(rmse(semi) < rmse(qml))
    boot = semiparametric_garch(_garch_returns(7)[0], "r", draws=20)
    assert 0.01 < boot.se[2] < 0.05  # the robust spread of alpha over repeated samples (1.48 x median absolute error) is about 0.023


def test_the_dialogs(surface):
    from pymetrix.commands import COMMANDS, Session

    data, _, _ = surface
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitKernelDensity"].cls("a", bandwidth="0.3"))
    assert s.history[-1].code == "density = kde(data, 'a', bandwidth=0.3)" and s.objects["density"].bandwidth == 0.3
    s.run(COMMANDS["FitKernelRegression"].cls("y", "a, b", bandwidth="0.4, 0.5"))
    assert s.objects["kreg"].bandwidth == pytest.approx([0.4, 0.5]) and "kernel_regression(data, 'y', ['a', 'b']" in s.history[-1].code
