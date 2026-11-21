"""Dynamic panel GMM: Arellano-Bond and Blundell-Bond, checked against theory, linearmodels and simulation."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, Dataset, Equation
from pymetrix.estimation import ArellanoBond


def simulate(N=400, T=8, rho=0.5, seed=0):
    """y = rho y[-1] + x + a_i + e, started far enough back to be stationary."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        a = rng.normal()
        y = a / (1 - rho) + rng.normal()
        for t in range(-20, T):
            x = rng.normal()
            y = rho * y + x + a + rng.normal()
            if t >= 0:
                rows.append({"id": i, "t": 2000 + t, "y": y, "x": x})
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="t", freq="Y")


@pytest.fixture(scope="module")
def panel():
    return simulate()


def test_gmm_removes_the_dynamic_panel_biases(panel):
    eq = Equation("y ~ y[-1] + x")
    ols = eq.fit(panel, method=OLS()).params["y[-1]"]
    within = Equation("y ~ y[-1] + x + fe(id)").fit(panel).params["y[-1]"]
    ab = eq.fit(panel, method=ArellanoBond())
    system = eq.fit(panel, method=ArellanoBond(system=True))
    assert ols > 0.7 and within < 0.42  # biased up and down
    assert abs(ab.params["y[-1]"] - 0.5) < 0.06 and abs(system.params["y[-1]"] - 0.5) < 0.05
    assert abs(ab.params["x"] - 1) < 0.08
    se = dict(zip(system.params.names, system.params.se))
    assert se["y[-1]"] < dict(zip(ab.params.names, ab.params.se))["y[-1]"]  # system GMM is more precise here
    assert "const" in system.params.names and "const" not in ab.params.names


def test_specification_tests(panel):
    fit = Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond())
    s = fit.stats
    assert s["ar1_pvalue"] < 0.01 and s["ar1"] < 0  # differenced iid errors are MA(1)
    assert s["ar2_pvalue"] > 0.05  # but not AR(2): the instruments are valid
    assert s["hansen_j_pvalue"] > 0.05 and s["instruments"] == 29 and s["groups"] == 400
    text = str(fit.summary())
    assert "Windmeijer-corrected" in text and "Hansen J" in text and "AR(2)" in text
    collapsed = Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond(collapse=True))
    assert collapsed.stats["instruments"] < s["instruments"]
    limited = Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond(lags="2:3"))
    assert limited.stats["instruments"] < s["instruments"]


def test_exactly_identified_gmm_is_two_stage_least_squares(panel):
    """With one instrument (y[-2]) for the one endogenous regressor, GMM is IV on the differenced data."""
    iv = pytest.importorskip("linearmodels.iv")
    ab = Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond(lags="2:2", collapse=True, time_effects=False, two_step=False))
    f = panel.frame.reset_index().sort_values(["id", "t"])
    g = f.groupby("id")
    f["dy"], f["dx"], f["y2"] = g["y"].diff(), g["x"].diff(), g["y"].shift(2)
    f["dy1"] = f.groupby("id")["dy"].shift()
    ok = f[["dy", "dy1", "dx", "y2"]].notna().all(axis=1)
    ref = iv.IV2SLS(f.loc[ok, "dy"], f.loc[ok, ["dx"]], f.loc[ok, ["dy1"]], f.loc[ok, ["y2"]]).fit()
    assert ab.params["y[-1]"] == pytest.approx(ref.params["dy1"], rel=1e-10)
    assert ab.params["x"] == pytest.approx(ref.params["dx"], rel=1e-10)
    assert np.isnan(ab.stats["hansen_j_pvalue"])


def test_windmeijer_standard_errors_match_the_spread_of_estimates():
    estimates, ses = [], []
    for r in range(40):
        fit = Equation("y ~ y[-1] + x").fit(simulate(100, 7, seed=100 + r), method=ArellanoBond(time_effects=False))
        estimates.append(fit.params["y[-1]"])
        ses.append(fit.params.se[0])
    assert abs(np.mean(estimates) - 0.5) < 0.03
    assert 0.7 < np.mean(ses) / np.std(estimates, ddof=1) < 1.3


def test_requirements_and_errors(panel):
    with pytest.raises(ValueError, match="panel"):
        Equation("c ~ y[-1]").fit(Dataset.example("sample_macro"), method=ArellanoBond())
    with pytest.raises(ValueError, match="fe\\(\\)"):
        Equation("y ~ y[-1] + fe(id)").fit(panel, method=ArellanoBond())
    with pytest.raises(ValueError, match="first:last"):
        ArellanoBond(lags="3:2")
    with pytest.raises(ValueError, match="not series"):
        Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond(gmm="z"))
    fit = Equation("y ~ y[-1] + x").fit(panel, method=ArellanoBond(gmm="y, x"))  # x treated as endogenous
    assert fit.stats["instruments"] > 29


def test_unbalanced_panels(panel):
    frame = panel.frame.reset_index()
    rng = np.random.default_rng(9)
    gappy = frame.drop(index=rng.choice(len(frame), 300, replace=False))
    data = Dataset.from_frame(gappy, entity="id", time="t", freq="Y")
    fit = Equation("y ~ y[-1] + x").fit(data, method=ArellanoBond(system=True))
    assert abs(fit.params["y[-1]"] - 0.5) < 0.08 and fit.stats["groups"] <= 400


def test_the_estimate_dialog_offers_them(panel):
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["data"] = panel
    s.run(FitEquation("y ~ y[-1] + x", method="Blundell-Bond system GMM", method_settings="collapse=True"))
    assert "method=ArellanoBond(system=True, collapse=True)" in s.script()
    assert abs(s["fit"].params["y[-1]"] - 0.5) < 0.06
