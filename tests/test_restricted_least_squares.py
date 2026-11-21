"""Restricted least squares: linear restrictions R b = r imposed on OLS and IV (``restrictions=[...]``), checked
independently against the substitution estimate. Substituting the restrictions into the equation gives an unrestricted
regression in fewer coefficients; its estimates, mapped back to the full set of coefficients, and its covariance by the
delta method (J V J', J the derivative of the full coefficients with respect to the free ones; exact, as the map is
linear) equal the restricted estimates and covariance, for the classic, White, HAC and clustered standard errors.
The parser is the one system estimation uses (its restriction tests run unchanged); without restrictions the results
are least squares' exactly."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import IV, OLS, Dataset, Equation
from pymetrix.estimation import Cluster
from pymetrix.estimation.restrictions import linear_restrictions
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def data():
    """y = 1 + 0.3 x1 + 0.7 x2 + 0.5 x3 + u, x1 endogenous (correlated with u), w and v instruments."""
    rng = np.random.default_rng(42)
    n = 300
    w, v, x2, x3 = rng.normal(size=(4, n))
    u = rng.normal(size=n)
    x1 = 0.8 * w + 0.5 * v + 0.6 * u + rng.normal(size=n)
    y = 1 + 0.3 * x1 + 0.7 * x2 + 0.5 * x3 + u
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "w": w, "v": v, "g": np.arange(n) % 15},
                         index=pd.period_range("1950Q1", periods=n, freq="Q"))
    return Dataset.from_frame(frame).derive(yy="y - x2 - 0.5 * x3", z="x1 - x2", yd="y - 2 * x1", zd="x2 + 2 * x1")


def _delta(sub, J, c):
    """The full coefficients and their covariance from a substitution fit: b = J theta + c, V = J V_theta J'."""
    return J @ sub.params.values + c, J @ sub.params.cov @ J.T


# b = (const, x1, x2, x3); the restrictions x1 + x2 = 1 and x3 = 0.5 leave theta = (const, x1)
J_SUM = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
C_SUM = np.array([0.0, 0.0, 1.0, 0.5])


@pytest.mark.parametrize("cov", ["classic", "white", "hac", "cluster"])
def test_ols_equals_the_substitution_estimate(data, cov):
    method = (lambda **kw: OLS(cov=Cluster("g") if cov == "cluster" else cov, **kw))
    fit = Equation("y ~ x1 + x2 + x3").fit(data, method=method(restrictions=["x1 + x2 = 1", "x3 = 0.5"]))
    sub = Equation("yy ~ z").fit(data, method=method())  # y - x2 - 0.5 x3 = const + b1 (x1 - x2)
    b, V = _delta(sub, J_SUM, C_SUM)
    np.testing.assert_allclose(fit.params.values, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.params.cov, V, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(fit.resid, sub.resid, atol=1e-12)
    assert fit.params["x1"] + fit.params["x2"] == pytest.approx(1, abs=1e-14) and fit.params["x3"] == pytest.approx(0.5, abs=1e-14)
    assert fit.stats["df_resid"] == sub.stats["df_resid"] == 300 - 2
    assert fit.extra["restrictions"] == ["x1 + x2 = 1", "x3 = 0.5"]
    assert "Restrictions imposed: x1 + x2 = 1, x3 = 0.5" in str(fit.summary())


@pytest.mark.parametrize("cov", ["classic", "white"])
def test_iv_equals_the_substitution_estimate(data, cov):
    """x1 = 2 x2 in y = a + b1 x1 + b2 x2 + u, x1 endogenous: y = a + b2 (2 x1 + x2) + u, by 2SLS with the same instruments
    (the constant, x2 and the excluded w and v); b1 = 2 b2, so se(b1) = 2 se(b2)."""
    fit = Equation("y ~ x1 + x2").fit(data, method=IV(endog=["x1"], instruments=["w", "v"], cov=cov, restrictions="x1 = 2 * x2"))
    sub = Equation("y ~ zd").fit(data, method=IV(endog=["zd"], instruments=["w", "v", "x2"], cov=cov))
    J = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    b, V = _delta(sub, J, np.zeros(3))
    np.testing.assert_allclose(fit.params.values, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.params.cov, V, rtol=1e-10, atol=1e-14)
    assert fit.params.se[1] == pytest.approx(2 * fit.params.se[2], rel=1e-12)
    unrestricted = Equation("y ~ x1 + x2").fit(data, method=IV(endog=["x1"], instruments=["w", "v"], cov=cov))
    assert fit.extra["first_stage_F"] == unrestricted.extra["first_stage_F"]  # the first stage does not change


def test_iv_with_a_sum_restriction(data):
    fit = Equation("y ~ x1 + x2 + x3").fit(data, method=IV(endog=["x1"], instruments=["w", "v"], restrictions="x1 + x2 = 1, x3 = 0.5"))
    sub = Equation("yy ~ z").fit(data, method=IV(endog=["z"], instruments=["w", "v", "x2", "x3"]))
    b, V = _delta(sub, J_SUM, C_SUM)
    np.testing.assert_allclose(fit.params.values, b, atol=1e-12)
    np.testing.assert_allclose(fit.params.cov, V, rtol=1e-10, atol=1e-14)


def test_without_restrictions_least_squares_is_unchanged(data):
    """The default leaves today's results exactly: least squares' closed form (X'X)^-1 X'y and s^2 (X'X)^-1."""
    assert OLS() == OLS(restrictions=None) == OLS(restrictions="") == OLS(restrictions=[])
    assert OLS().restrictions is None and IV(endog=["x1"], instruments=["w"]).restrictions is None
    fit = Equation("y ~ x1 + x2").fit(data)
    X = np.column_stack([np.ones(300), data["x1"], data["x2"]])
    y = data["y"].to_numpy()
    b = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ b
    np.testing.assert_allclose(fit.params.values, b, rtol=1e-12)
    np.testing.assert_allclose(fit.params.cov, u @ u / 297 * np.linalg.inv(X.T @ X), rtol=1e-10)
    assert "restrictions" not in fit.extra and fit.stats["df_resid"] == 297


def test_the_restriction_matrices():
    R, r, texts = linear_restrictions("a + b = 1, 2*c - d[-1] = a - 3, const = 0", ["const", "a", "b", "c", "d[-1]"])
    np.testing.assert_array_equal(R, [[0, 1, 1, 0, 0], [0, -1, 0, 2, -1], [1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(r, [1, -3, 0])
    assert texts == ["a + b = 1", "2*c - d[-1] = a - 3", "const = 0"]
    R, r, _ = linear_restrictions(["dlog(y) = 0.5", "c.y = i.y"], ["dlog(y)", "c.y", "i.y"])  # term labels, system names
    np.testing.assert_array_equal(R, [[1, 0, 0], [0, 1, -1]])


@pytest.mark.parametrize("text, message", [
    ("x1 + x9 = 1", r"unknown coefficient\(s\) \['x9'\] in restriction 'x1 \+ x9 = 1'"),
    ("x1 * x2 = 1", "the restriction 'x1 \\* x2 = 1' is not linear"),
    ("x1 + x2", "could not read the restriction 'x1 \\+ x2'"),
    ("x1 ~ 1", "the restriction 'x1 ~ 1' needs '='"),
    ("x1 = 1, x1 = 2", "repeat or contradict"),
    ("1 = 1", "the restriction '1 = 1' involves no coefficient"),
])
def test_a_bad_restriction_is_named(data, text, message):
    with pytest.raises(SpecificationError, match=message):
        Equation("y ~ x1 + x2").fit(data, method=OLS(restrictions=text))


def test_explicit_form_equations_are_refused(data):
    with pytest.raises(SpecificationError, match="linear-form"):
        Equation("y ~ b0 + b1 * exp(x1)", params=["b0", "b1"]).fit(data, method=OLS(restrictions="b1 = 1"))


def test_the_estimate_dialog_and_its_script(data):
    from pymetrix.commands import FitEquation, Session
    from pymetrix.estimation.registry import method_settings

    assert {"OLS": "text", "IV": "text"} == {m: {f.name: f.kind for f in method_settings(m)}["restrictions"] for m in ("OLS", "IV")}
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("y ~ x1 + x2 + x3", method="OLS", cov="white", method_settings="restrictions='x1 + x2 = 1, x3 = 0.5'", name="rls"))
    s.run(FitEquation("y ~ x1 + x2", method="IV", endog="x1", instruments="w, v", method_settings="restrictions='x1 = 2 * x2'",
                      name="riv"))
    by_api = Equation("y ~ x1 + x2 + x3").fit(data, method=OLS(cov="white", restrictions=["x1 + x2 = 1", "x3 = 0.5"]))
    iv_api = Equation("y ~ x1 + x2").fit(data, method=IV(endog=["x1"], instruments=["w", "v"], restrictions=["x1 = 2 * x2"]))
    np.testing.assert_array_equal(s["rls"].params.values, by_api.params.values)
    np.testing.assert_array_equal(s["rls"].params.cov, by_api.params.cov)
    np.testing.assert_array_equal(s["riv"].params.cov, iv_api.params.cov)
    script = s.script()
    assert "restrictions=['x1 + x2 = 1', 'x3 = 0.5']" in script and "restrictions=['x1 = 2 * x2']" in script
    ns = dict(s.objects)
    exec(script, ns)
    np.testing.assert_array_equal(ns["rls"].params.cov, by_api.params.cov)
    np.testing.assert_array_equal(ns["riv"].params.values, iv_api.params.values)
    with pytest.raises(SpecificationError, match="x9"):
        s.run(FitEquation("y ~ x1 + x2", method="OLS", method_settings="restrictions='x9 = 1'", name="bad"))
