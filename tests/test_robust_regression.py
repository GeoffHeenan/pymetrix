"""Robust regression (M-estimation), checked against statsmodels' RLM: coefficients, Huber's H1 standard errors and the
MAD scale with HuberT() and TukeyBiweight(), default and other tuning constants, to 1e-8 (the reference iterated to
convergence: maxiter=1000, tol=1e-14 on the coefficients); with the scale held fixed, the estimating equations and the
H1 formula (and RLM where it takes a starting scale); a closed form (a location model whose observations all lie
within the tuning constant is the sample mean); and the Estimate dialog's step against the API call and its script."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

import statsmodels.api as sm  # noqa: E402

from pymetrix import Dataset, Equation, RobustLS  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    """y = 1 + 0.5 x1 - 2 x2 + t(2) errors, the first six observations shifted by 15 (outliers)."""
    rng = np.random.default_rng(3)
    n = 150
    x1, x2 = rng.normal(size=n), rng.uniform(size=n)
    y = 1 + 0.5 * x1 - 2 * x2 + rng.standard_t(2, size=n)
    y[:6] += 15
    data = Dataset.from_frame(pd.DataFrame({"i": range(n), "x1": x1, "x2": x2, "y": y}), entity="i")
    return data, y, sm.add_constant(np.column_stack([x1, x2]))


NORMS = {"huber": sm.robust.norms.HuberT(), "bisquare": sm.robust.norms.TukeyBiweight()}


@pytest.mark.parametrize("norm", ["huber", "bisquare"])
def test_against_statsmodels_rlm(sample, norm):
    data, y, X = sample
    fit = Equation("y ~ x1 + x2").fit(data, method=RobustLS(norm=norm))
    ref = sm.RLM(y, X, M=NORMS[norm]).fit(maxiter=1000, tol=1e-14, conv="coefs")  # scale by the MAD, H1 covariance
    np.testing.assert_allclose(fit.params.values, ref.params, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fit.params.se, ref.bse, rtol=0, atol=1e-8)
    assert fit.stats["scale"] == pytest.approx(ref.scale, abs=1e-8)
    np.testing.assert_allclose(fit.extra["robust_weights"], ref.weights, atol=1e-8)
    assert fit.extra["converged"] and fit.provenance.converged
    assert fit.extra["robust_weights"].iloc[:6].max() < 0.3  # the outliers are weighted down
    assert abs(fit.params["x1"] - 0.5) < abs(Equation("y ~ x1 + x2").fit(data).params["x1"] - 0.5)


@pytest.mark.parametrize("norm, tuning", [("huber", 2.0), ("bisquare", 6.0)])
def test_other_tuning_constants(sample, norm, tuning):
    data, y, X = sample
    M = sm.robust.norms.HuberT(t=tuning) if norm == "huber" else sm.robust.norms.TukeyBiweight(c=tuning)
    fit = Equation("y ~ x1 + x2").fit(data, method=RobustLS(norm=norm, tuning=tuning))
    ref = sm.RLM(y, X, M=M).fit(maxiter=1000, tol=1e-14, conv="coefs")
    np.testing.assert_allclose(fit.params.values, ref.params, atol=1e-8)
    np.testing.assert_allclose(fit.params.se, ref.bse, atol=1e-8)


@pytest.mark.parametrize("norm", ["huber", "bisquare"])
def test_a_fixed_scale(sample, norm):
    """With the scale held at 1.5 the estimate solves sum psi((y - X b) / 1.5) x = 0, and H1 is its formula at that
    scale; statsmodels' RLM(start_scale=1.5, update_scale=False) where it takes a starting scale (0.14.2 on)."""
    import inspect

    data, y, X = sample
    method = RobustLS(norm=norm, scale=1.5)
    fit = Equation("y ~ x1 + x2").fit(data, method=method)
    z = (y - X @ fit.params.values) / 1.5
    np.testing.assert_allclose(X.T @ method._psi(z), 0, atol=1e-9)
    dpsi = method._psi_deriv(z)
    K = 1 + 3 / 150 * dpsi.var() / dpsi.mean() ** 2
    h1 = K**2 * np.sum(method._psi(z) ** 2) / 147 * 1.5**2 / dpsi.mean() ** 2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(fit.params.cov, h1, rtol=1e-12)
    assert fit.stats["scale"] == 1.5
    if "start_scale" in inspect.signature(sm.RLM.fit).parameters:
        ref = sm.RLM(y, X, M=NORMS[norm]).fit(maxiter=1000, tol=1e-14, conv="coefs", start_scale=1.5, update_scale=False)
        np.testing.assert_allclose(fit.params.values, ref.params, atol=1e-8)
        np.testing.assert_allclose(fit.params.se, ref.bse, atol=1e-8)


def test_a_location_model_within_the_tuning_constant_is_the_mean():
    """Every scaled residual inside the constant: Huber's weights are all 1, so the estimate is least squares (the mean),
    and H1 is s^2 sum(z^2) / (n - 1) / n = (sum u^2 / (n - 1)) / n, the mean's usual variance."""
    y = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
    data = Dataset.from_frame(pd.DataFrame({"i": range(7), "y": y}), entity="i")
    fit = Equation("y ~ 1").fit(data, method=RobustLS(tuning=10.0))
    assert fit.params["const"] == pytest.approx(y.mean(), abs=1e-14)
    assert fit.params.se[0] == pytest.approx(y.std(ddof=1) / np.sqrt(7), rel=1e-12)


def test_requirements(sample):
    data, _, _ = sample
    with pytest.raises(SpecificationError, match="norm"):
        RobustLS(norm="cauchy")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="positive"):
        RobustLS(tuning=0)
    with pytest.raises(SpecificationError, match="weights"):
        Equation("y ~ x1").fit(data, method=RobustLS(), weights="x2")
    with pytest.raises(SpecificationError, match="linear-form"):
        Equation("y ~ b0 + b1 * exp(x1)", params=["b0", "b1"]).fit(data, method=RobustLS())
    assert RobustLS().constant == 1.345 and RobustLS(norm="bisquare").constant == 4.685


def test_the_estimate_dialog_offers_it(sample):
    from pymetrix.commands import FitEquation, Session
    from pymetrix.estimation.registry import method_settings

    data, _, _ = sample
    settings = {f.name: f for f in method_settings("Robust regression (M)")}
    assert not settings["norm"].advanced and all(settings[k].advanced for k in ("tuning", "scale", "max_iter", "tol"))
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("y ~ x1 + x2", method="Robust regression (M)", method_settings="norm='bisquare', tuning=4.0", name="rob"))
    by_api = Equation("y ~ x1 + x2").fit(data, method=RobustLS(norm="bisquare", tuning=4.0))
    np.testing.assert_array_equal(s["rob"].params.values, by_api.params.values)
    np.testing.assert_array_equal(s["rob"].params.cov, by_api.params.cov)
    script = s.script()
    assert "method=RobustLS(norm='bisquare', tuning=4.0)" in script and "Tukey bisquare" in str(s["rob"].summary())
    ns = dict(s.objects)
    exec(script, ns)
    np.testing.assert_array_equal(ns["rob"].params.cov, by_api.params.cov)
    assert FitEquation("y ~ x1", method="Robust regression (M)", cov="white").validate(s)  # its own standard errors only
