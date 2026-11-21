"""Local projections: the regressions, agreement with VARs, options, panels, and the dialog."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, VAR, Cluster, Dataset, Equation, LocalProjection

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def var_data():
    """A recursive VAR(1): m (the policy variable) affects y on impact."""
    rng = np.random.default_rng(7)
    n = 4000
    A = np.array([[0.5, 0.0, 0.0], [0.2, 0.5, 0.0], [0.3, 0.2, 0.6]])
    B = np.array([[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.5, 0.3, 1.0]])
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t] = A @ x[t - 1] + B @ rng.normal(size=3)
    frame = pd.DataFrame(x, columns=["c", "m", "y"], index=pd.period_range("1000Q1", periods=n, freq="Q", name="time"))
    return Dataset.from_frame(frame), A, B


def test_each_horizon_is_an_ols_regression():
    data = Dataset.read(str(DATA)).with_series(ly="100*log(y)", lg="100*log(g)")
    lp = LocalProjection("ly", "lg", horizons=3, lags=2, controls=["i"]).fit(data)
    for h in range(4):
        manual = Equation(lp.spec.formula(h)).fit(data, method=OLS(cov="hac", hac_lags=h + 1))
        assert np.isclose(lp.irf.loc[h, "response"], manual.params["lg"])
        assert np.isclose(lp.irf.loc[h, "se"], manual.params.se[list(manual.params.names).index("lg")])
    assert lp.spec.formula(2) == "ly[+2] ~ lg + ly[-1] + ly[-2] + lg[-1] + lg[-2] + i[-1] + i[-2]"
    assert list(lp.irf["nobs"]) == [98, 97, 96, 95] and len(lp.fits) == 4
    z = 1.6448536269514722
    assert np.allclose(lp.irf["upper"] - lp.irf["response"], z * lp.irf["se"])



def test_function_call_shock_with_and_without_state():
    """A shock written as a call (dlog) is labelled by its expansion in the regressions; the irf finds it."""
    data = Dataset.example()
    data = data.with_series(z=(data["c"].diff() > data["c"].diff().median()).astype(float))
    lp = LocalProjection(response="100*log(y)", shock="dlog(g)", horizons=4, lags=2, controls=["c"]).fit(data)
    for h in range(5):
        manual = Equation(lp.spec.formula(h)).fit(data, method=OLS(cov="hac", hac_lags=h + 1))
        i = list(manual.params.names).index("log(g) - log(g[-1])")
        assert np.isclose(lp.irf.loc[h, "response"], manual.params.values[i])
        assert np.isclose(lp.irf.loc[h, "se"], manual.params.se[i])
    by_state = LocalProjection(response="100*log(y)", shock="dlog(g)", horizons=2, lags=1, state="z").fit(data)
    for h in range(3):
        manual = Equation(by_state.spec.formula(h)).fit(data, method=OLS(cov="hac", hac_lags=h + 1))
        names = list(manual.params.names)
        for regime, name in [("state", "(log(g) - log(g[-1]))*z[-1]"), ("other", "(log(g) - log(g[-1]))*(1 - z[-1])")]:
            assert np.isclose(by_state.irf.loc[h, f"{regime} response"], manual.params.values[names.index(name)])
            assert np.isclose(by_state.irf.loc[h, f"{regime} se"], manual.params.se[names.index(name)])


def test_comparison_state_is_one_term():
    """A state written as a comparison is bracketed, so it interacts with every term as a whole."""
    data = Dataset.example()
    lp = LocalProjection(response="100*log(y)", shock="dlog(g)", horizons=2, lags=1, state="c > c[-1]")
    assert "(dlog(g))*(c[-1] > c[-2])" in lp.formula(0) and "(1 - (c[-1] > c[-2]))" in lp.formula(0)
    fitted = lp.fit(data)
    by_series = LocalProjection(response="100*log(y)", shock="dlog(g)", horizons=2, lags=1, state="z").fit(
        data.with_series(z=(data["c"] > data["c"].shift(1)).astype(float)))
    assert np.allclose(fitted.irf["state response"], by_series.irf["state response"])
    assert np.allclose(fitted.irf["other se"], by_series.irf["other se"])

def test_agrees_with_the_var(var_data):
    data, A, B = var_data
    var = VAR(["c", "m", "y"], lags=1).fit(data)
    irf = var.irf(6).values
    unit = irf[:, 2, 1] / irf[0, 1, 1]  # y's response to a unit m shock, recursive ordering c, m, y
    lp = LocalProjection("y", "m", horizons=6, lags=1, contemporaneous=["c"]).fit(data)
    assert np.isclose(lp.irf.loc[0, "response"], unit[0])  # exactly at impact
    assert np.allclose(lp.irf["response"], unit, atol=0.05)  # and close after, in a long sample
    truth = [(np.linalg.matrix_power(A, h) @ B)[2, 1] for h in range(7)]
    assert np.all(np.abs(lp.irf["response"] - truth) < 3 * lp.irf["se"] + 0.02)


def test_options_long_differences_lag_augmentation_and_scale(var_data):
    data, _, _ = var_data
    base = LocalProjection("y", "m", horizons=2, lags=1)
    long = LocalProjection("y", "m", horizons=2, lags=1, long_difference=True)
    assert long.formula(1) == "(y[+1]) - (y[-1]) ~ m + y[-1] + m[-1]"
    # with y[-1] a regressor, the long difference shifts only that coefficient: the response is the same
    assert np.allclose(long.fit(data).irf["response"], base.fit(data).irf["response"])
    augmented = LocalProjection("y", "m", horizons=2, lags=1, cov="white")
    assert "y[-2]" in augmented.formula(0) and augmented.fit(data).fits["h=0"].method == OLS(cov="white")
    scaled = LocalProjection("y", "m", horizons=2, lags=1, scale=0.25).fit(data)
    assert np.allclose(scaled.irf["response"], 0.25 * base.fit(data).irf["response"])


def test_state_dependent_responses():
    rng = np.random.default_rng(3)
    n = 3000
    state = (rng.random(n) < 0.4).astype(float)
    shock = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        effect = 2.0 if state[t - 1] else 0.5
        y[t] = 0.5 * y[t - 1] + effect * shock[t] + rng.normal()
    frame = pd.DataFrame({"y": y, "s": shock, "z": state}, index=pd.period_range("1000Q1", periods=n, freq="Q", name="time"))
    lp = LocalProjection("y", "s", horizons=1, lags=1, state="z").fit(Dataset.from_frame(frame))
    assert abs(lp.irf.loc[0, "state response"] - 2.0) < 0.1 and abs(lp.irf.loc[0, "other response"] - 0.5) < 0.1
    assert lp.irf.loc[0, "difference p-value"] < 0.001
    assert "- 1" in lp.spec.formula(0) and "s*z[-1]" in lp.spec.formula(0)


def test_panels_use_fixed_effects_and_clusters():
    rng = np.random.default_rng(5)
    rows = []
    for k in range(30):
        y, alpha = 0.0, rng.normal(0, 2)
        for t, p in enumerate(pd.period_range("2000Q1", periods=40, freq="Q")):
            s = rng.normal()
            y = alpha * 0.5 + 0.5 * y + 1.5 * s + rng.normal()
            rows.append({"id": k, "t": p, "y": y, "s": s})
    panel = Dataset.from_frame(pd.DataFrame(rows), time="t", entity="id")
    lp = LocalProjection("y", "s", horizons=2, lags=1).fit(panel)
    assert lp.fits["h=0"].method == OLS(cov=Cluster("id")) and "fe(id)" in lp.fits["h=0"].equation.text
    assert abs(lp.irf.loc[0, "response"] - 1.5) < 0.1 and abs(lp.irf.loc[1, "response"] - 0.75) < 0.15


def test_dialog_records_replays_and_shows(tmp_path):
    from pymetrix.commands import FitLocalProjection, Session, load_project, save_project
    from pymetrix.ui.render import sections

    data = Dataset.read(str(DATA)).with_series(ly="100*log(y)", lg="100*log(g)")
    s = Session()
    s.objects["data"] = data
    s.run(FitLocalProjection("ly", "lg", horizons=4, lags=2, controls="i", long_difference=True))
    script = s.script()
    assert "lp = LocalProjection(response='ly', shock='lg', horizons=4, lags=2, controls=['i'], long_difference=True).fit(data)" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["lp"].irf["response"], s["lp"].irf["response"])
    graph = ["Graph"] if importlib.util.find_spec("matplotlib") else []  # graphs need matplotlib
    assert [t for t, _ in sections(s["lp"])] == ["Responses", *graph, "Regressions", "Statistics"]
    reopened = load_project(save_project(s, tmp_path / "lp.pymx"))
    assert np.allclose(reopened["lp"].irf["se"], s["lp"].irf["se"])
