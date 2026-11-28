"""HC0, HC2 and HC3 heteroskedasticity-robust covariances beside White's HC1.

Checks: statsmodels' ``cov_type="HC0"``, ``"HC2"`` and ``"HC3"`` called directly (OLS, WLS, and restricted least squares
as the regression with the restriction substituted), to 1e-12; the hand formula with the leverages h_ii on a regression
of six observations; R's ``sandwich::vcovHC`` on Mroz's hours equation, recorded below; for IV, HC0 against
linearmodels' ``IV2SLS`` with ``cov_type="robust", debiased=False``.

R 4.3.3, sandwich 3.1.0 (``vcovHC``'s default type is HC3), Mroz's data as ``Dataset.example("mroz")``::

    m <- lm(hours ~ educ + exper + kidslt6 + nwifeinc, data = d)
    sqrt(diag(vcovHC(m, type = "HC0")))   # and "HC1", "HC2", "HC3"
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402

from pymetrix import IV, OLS, Dataset, Equation  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402

TOL = 1e-12
KINDS = ("hc0", "hc2", "hc3")

R_SANDWICH = {  # R 4.3.3, sandwich 3.1.0: const, educ, exper, kidslt6, nwifeinc
    "hc0": [149.114608604541, 12.7212091445168, 4.0910662886273, 47.7839648751548, 2.23568186862224],
    "white": [149.612156228679, 12.7636557394228, 4.10471686472526, 47.9434046404317, 2.24314162197879],
    "hc2": [149.838466930088, 12.7831920199849, 4.11492714448478, 48.1279282356767, 2.25195536410835],
    "hc3": [150.568356907449, 12.8457175225928, 4.13907887079204, 48.4788930859105, 2.26859542472651],
}


def _close(a, b, rtol=TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.abs(b).max())


@pytest.mark.parametrize("kind", KINDS)
def test_ols_matches_statsmodels(ts, kind):
    fit = Equation("y ~ x + z").fit(ts, method=OLS(cov=kind))
    ref = sm.OLS(ts["y"], sm.add_constant(ts.frame[["x", "z"]])).fit(cov_type=kind.upper())
    _close(fit.params.cov, ref.cov_params().to_numpy())
    _close(fit.params.values, ref.params.to_numpy())


@pytest.mark.parametrize("kind", KINDS)
def test_weighted_least_squares_matches_statsmodels(ts, kind):
    data = ts.with_series(w="1 + x^2")
    fit = Equation("y ~ x + z").fit(data, method=OLS(cov=kind), weights="w")
    ref = sm.WLS(data["y"], sm.add_constant(data.frame[["x", "z"]]), weights=data["w"]).fit(cov_type=kind.upper())
    _close(fit.params.cov, ref.cov_params().to_numpy())


@pytest.mark.parametrize("kind", KINDS)
def test_restricted_least_squares_is_the_substituted_regression(ts, kind):
    """x + z = 1: y - z = const + b_x (x - z), and b_z = 1 - b_x; the leverages are the substituted regression's."""
    fit = Equation("y ~ x + z").fit(ts, method=OLS(cov=kind, restrictions="x + z = 1"))
    frame = ts.frame
    ref = sm.OLS(frame["y"] - frame["z"], sm.add_constant(frame["x"] - frame["z"])).fit(cov_type=kind.upper())
    V = ref.cov_params().to_numpy()
    _close(fit.params.se[:2], np.sqrt(np.diag(V)))
    _close(fit.params.se[2], np.sqrt(V[1, 1]))


def test_hand_formula_with_leverages():
    """Six observations: h_ii = x_i' (X'X)^-1 x_i; HC2 weights u_i^2 by 1 / (1 - h_ii), HC3 by 1 / (1 - h_ii)^2."""
    x = np.array([1.0, 2.0, 4.0, 5.0, 7.0, 12.0])
    y = np.array([2.1, 2.9, 5.2, 5.8, 8.9, 11.0])
    data = Dataset.from_frame(pd.DataFrame({"y": y, "x": x}), start="2000", freq="Y")
    X = np.column_stack([np.ones(6), x])
    B = np.linalg.inv(X.T @ X)
    u = y - X @ (B @ X.T @ y)
    h = np.array([X[i] @ B @ X[i] for i in range(6)])
    assert np.isclose(h.sum(), 2.0) and h[-1] > 0.6  # the trace is k; the last point has high leverage
    for kind, weight in (("hc0", u**2), ("hc2", u**2 / (1 - h)), ("hc3", u**2 / (1 - h) ** 2)):
        meat = sum(weight[i] * np.outer(X[i], X[i]) for i in range(6))
        fit = Equation("y ~ x").fit(data, method=OLS(cov=kind))
        _close(fit.params.cov, B @ meat @ B)


@pytest.mark.parametrize("kind", ["hc0", "white", "hc2", "hc3"])
def test_mroz_matches_r_sandwich(kind):
    mroz = Dataset.example("mroz")
    fit = Equation("hours ~ educ + exper + kidslt6 + nwifeinc").fit(mroz, method=OLS(cov=kind))
    _close(fit.params.se, R_SANDWICH[kind], rtol=1e-11)  # R printed 15 significant digits


def test_hc0_with_fixed_effects_equals_the_dummy_regression(panel):
    fit = Equation("y ~ x + fe(country)").fit(panel, method=OLS(cov="hc0"))
    dummies = pd.get_dummies(panel.values_of("country"), dtype=float)
    ref = sm.OLS(panel["y"], pd.concat([panel["x"], dummies], axis=1)).fit(cov_type="HC0")
    _close(fit.params.se[0], ref.bse["x"])


def test_iv_hc0_matches_linearmodels():
    linearmodels = pytest.importorskip("linearmodels")
    card = Dataset.example("card")
    fit = Equation("lwage ~ educ + exper + black").fit(card, method=IV(endog=["educ"], instruments=["nearc4"], cov="hc0"))
    frame = card.frame.loc[fit.resid.index]
    ref = linearmodels.IV2SLS(frame["lwage"], sm.add_constant(frame[["exper", "black"]]), frame[["educ"]],
                              frame[["nearc4"]]).fit(cov_type="robust", debiased=False)
    _close(fit.params.cov, ref.cov.loc[list(fit.params.names), list(fit.params.names)].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("kind", ["hc2", "hc3"])
def test_hc2_hc3_refused_where_leverage_is_not_standard(ts, panel, kind):
    with pytest.raises(SpecificationError, match="IV"):
        Equation("y ~ x + z").fit(ts, method=IV(endog=["x"], instruments=["x[-1]"], cov=kind))
    with pytest.raises(SpecificationError, match="explicit-form"):
        Equation("y ~ a + b*x", params=["a", "b"]).fit(ts, method=OLS(cov=kind))
    with pytest.raises(SpecificationError, match="fixed effects"):
        Equation("y ~ x + fe(country)").fit(panel, method=OLS(cov=kind))


def test_leverage_one_is_refused(ts):
    data = Dataset.from_frame(ts.frame.assign(d=(ts.frame.index == pd.Period("2010Q1")).astype(float)))
    with pytest.raises(SpecificationError, match="leverage 1"):
        Equation("y ~ x + d").fit(data, method=OLS(cov="hc3"))
    assert np.isfinite(Equation("y ~ x + d").fit(data, method=OLS(cov="hc0")).params.se).all()


def test_direct_ols_and_summary_note(ts):
    from pymetrix import ols

    fit = Equation("y ~ x + z").fit(ts, method=OLS(cov="hc3"))
    direct = ols(ts["y"], ts.frame[["x", "z"]], cov="hc3")
    _close(direct.params.cov, fit.params.cov)
    assert "heteroskedasticity-robust (HC3) standard errors" in str(fit.summary())


def test_existing_default_and_white_unchanged(ts):
    assert OLS().cov == "classic" and IV(endog=["x"], instruments=["z"]).cov == "classic"
    white = Equation("y ~ x + z").fit(ts, method=OLS(cov="white"))
    hc0 = Equation("y ~ x + z").fit(ts, method=OLS(cov="hc0"))
    _close(white.params.cov, hc0.params.cov * 100 / 97)  # HC1 = n / (n - k) HC0


def test_estimate_dialog(ts):
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["data"] = ts
    step = FitEquation("y ~ x + z", data="data", cov="hc3")
    assert not step.validate(s)
    assert 'OLS(cov="hc3")' in step.to_code().replace("'", '"')
    s.run(step)
    assert s["fit"].method.cov == "hc3"
    assert FitEquation("y ~ x + z", data="data", method="IV", endog="x", instruments="z[-1]", cov="hc0").validate(s) == []
    assert any("OLS" in e for e in FitEquation("y ~ x + z", data="data", method="IV", endog="x", instruments="z[-1]",
                                               cov="hc3").validate(s))


def test_statsmodels_results_name_their_hc_type():
    from pymetrix.interop import Statsmodels

    assert [Statsmodels(cov_type=t).cov for t in ("HC0", "HC1", "HC2", "HC3")] == ["hc0", "white", "hc2", "hc3"]
