import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, GaussSeidel, Model, ModelError, Newton, Scenario


def _economy():
    """c = 10 + 0.6*y + 0.2*c[-1];  i = 5 + 0.1*y[-1];  y = c + i + g."""
    cons = Equation("c ~ a + b*y + d*c[-1]", params=["a", "b", "d"]).with_params(a=10, b=0.6, d=0.2)
    inv = Equation("i ~ e + f*y[-1]", params=["e", "f"]).with_params(e=5, f=0.1)
    gdp = Equation("y = c + i + g")
    model = Model([cons, inv, gdp])

    periods = pd.period_range("2000Q1", "2010Q4", freq="Q", name="time")
    g = 20 + np.arange(len(periods), dtype=float)
    c, i, y = np.zeros_like(g), np.zeros_like(g), np.zeros_like(g)
    c[0], i[0], y[0] = 60.0, 15.0, 95.0
    for t in range(1, len(g)):
        i[t] = 5 + 0.1 * y[t - 1]
        # c = 10 + 0.6*(c + i + g) + 0.2*c[-1]
        c[t] = (10 + 0.6 * (i[t] + g[t]) + 0.2 * c[t - 1]) / 0.4
        y[t] = c[t] + i[t] + g[t]
    data = Dataset.from_frame(pd.DataFrame({"c": c, "i": i, "y": y, "g": g}, index=periods))
    return model, data


def test_structure():
    model, _ = _economy()
    assert model.endogenous == ("c", "i", "y")
    assert model.exogenous == ("g",)
    blocks = model.blocks()
    assert [b.variables for b in blocks] == [("i",), ("c", "y")]
    assert blocks[1].simultaneous


@pytest.mark.parametrize("solver", [Newton(), GaussSeidel()])
def test_solve_reproduces_the_data(solver):
    model, data = _economy()
    blank = data.with_series(c=data["c"].where(data["c"].index < pd.Period("2005Q1", "Q")))
    sol = model.solve(blank, span=("2005Q1", "2010Q4"), solver=solver)
    assert np.allclose(sol["c"], data["c"])
    assert np.allclose(sol["y"], sol["c"] + sol["i"] + sol["g"])
    assert sol.provenance.sample == ("2005Q1", "2010Q4")


def test_scenarios():
    model, data = _economy()
    span = ("2005Q1", "2010Q4")
    base = model.solve(data, span=span)
    shocked = model.solve(data, span=span, scenario=Scenario(shift={"g": 1.0}))
    diff = (shocked - base).select(span=span)
    assert np.isclose(diff["y"].iloc[0], 1 / 0.4)  # impact multiplier 1/(1-0.6)

    target = base["y"].copy()
    target.loc[pd.Period("2005Q1", "Q"):] += 1.0
    swapped = model.solve(data, span=span, scenario=Scenario(hold={"y": target}, free=["g"]))
    assert np.allclose(swapped["y"], target)
    assert np.isclose((swapped["g"] - base["g"]).loc[pd.Period("2005Q1", "Q")], 0.4)

    held = model.solve(data, span=span, scenario=Scenario(fix={"c": None}, shift={"g": 1.0}))
    assert np.allclose(held["c"], data["c"])

    adjusted = model.solve(data, span=span, scenario=Scenario(adjust={"c": 2.0}))
    assert np.isclose((adjusted["c"] - base["c"]).loc[pd.Period("2005Q1", "Q")], 2 / 0.4)


def test_forecast_beyond_the_data():
    model, data = _economy()
    g_path = pd.Series(50.0, index=pd.period_range("2011Q1", "2012Q4", freq="Q"))
    sol = model.solve(data, span=("2011Q1", "2012Q4"), scenario=Scenario(set={"g": g_path}))
    assert len(sol) == len(data) + 8
    assert np.isfinite(sol["y"]).all()


def test_errors():
    model, data = _economy()
    with pytest.raises(ModelError, match="fit it"):
        Model([Equation("c ~ a + b*y", params=["a", "b"])])
    with pytest.raises(ModelError, match="Gauss-Seidel"):
        model.solve(data, span=("2005Q1", "2006Q4"), solver=GaussSeidel(), scenario=Scenario(hold={"y": data["y"] + 1}, free=["g"]))
    with pytest.raises(ModelError, match="missing values"):
        model.solve(data.with_series(g=data["g"].where(data["g"] < 30)), span=("2005Q1", "2010Q4"))
    with pytest.raises(ModelError, match="leads"):
        Model([Equation("y = y[+1]")]).solve(data, span=("2005Q1", "2006Q4"))


def test_fitted_equation_in_a_model(ts):
    fit = Equation("y ~ x + z").fit(ts)
    sol = Model([fit]).solve(ts, span=("2010Q1", "2010Q4"))
    expected = fit.params["const"] + fit.params["x"] * ts["x"] + fit.params["z"] * ts["z"]
    window = sol.sample_mask(("2010Q1", "2010Q4"))
    assert np.allclose(sol["y"][window], expected[window])
