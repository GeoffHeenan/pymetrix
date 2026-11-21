"""Models solved for each entity of a panel (Model(..., over="country")).

- Each country's solution against the recursion written out by hand, c(t) = a + b y(t) + d c(t-1) from the data's
  c before the span: the same parameters, a country's own (entity_params), a country's own scenario.
- Draws: each country's simulation, with seeds that differ by country; a missing input named with its country;
  refusals."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Scenario
from pymetrix.models.solvers import ModelError
from pymetrix.errors import SpecificationError

SPAN = ("2008Q1", "2009Q4")


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    rows = []
    for k in ("DE", "FR", "UK"):
        for i, p in enumerate(pd.period_range("2000Q1", "2009Q4", freq="Q")):
            rows.append({"country": k, "time": str(p), "y": 100 + i + rng.normal(), "c": 80.0 + i * 0.5 + rng.normal()})
    return Dataset.from_frame(pd.DataFrame(rows), entity="country", time="time", freq="Q")


@pytest.fixture(scope="module")
def model():
    return Model([Equation("c ~ y + c[-1]", name="cons").with_params(const=5.0, y=0.3, **{"c[-1]": 0.5})], over="country")


def _by_hand(panel, country, b, y_override=None):
    f = panel.entity(country, by="country").frame
    periods = pd.period_range(*SPAN, freq="Q")
    c = f["c"].loc[periods[0] - 1]
    out = []
    for p in periods:
        y = f["y"].loc[p] if y_override is None else y_override
        c = 5.0 + b * y + 0.5 * c
        out.append(c)
    return np.array(out)


def test_each_country_by_hand(panel, model):
    sol = model.solve(panel, SPAN, entity_params={"UK": {"cons.y": 0.6}}, scenario={"FR": Scenario(name="high", set={"y": 120.0})})
    solved = sol.solved().frame
    assert np.allclose(solved.loc["DE", "c"].to_numpy(), _by_hand(panel, "DE", 0.3), atol=1e-9)
    assert np.allclose(solved.loc["UK", "c"].to_numpy(), _by_hand(panel, "UK", 0.6), atol=1e-9)
    assert np.allclose(solved.loc["FR", "c"].to_numpy(), _by_hand(panel, "FR", 0.3, y_override=120.0), atol=1e-9)
    assert sol.solution("FR").scenario.name == "high" and sol.solution("DE").scenario.name == "baseline"
    assert sol.converged and sol.span == SPAN and [a.name for a in sol.axes] == ["country", "time"]
    plain = model.solve(panel, SPAN).solved().frame
    assert np.allclose(plain.loc["UK", "c"].to_numpy(), _by_hand(panel, "UK", 0.3), atol=1e-9)


def test_draws_missing_inputs_and_refusals(panel, model):
    fitted = Equation("c ~ y + c[-1]", name="cons").fit(panel.entity("DE", by="country"))
    sol = Model([fitted], over="country").solve(panel, SPAN, draws=50, seed=1)
    a, b = sol.solution("DE").simulation.values, sol.solution("FR").simulation.values
    assert a.shape == b.shape == (50, 8, 1) and not np.allclose(a - a.mean(0), b - b.mean(0))
    gap = panel.frame.copy()
    gap.loc[("UK", pd.Period("2008Q3", freq="Q")), "y"] = np.nan
    with pytest.raises(ModelError, match="country UK"):
        model.solve(Dataset(gap, panel.axes, panel.variables), SPAN)
    with pytest.raises(ModelError, match="entity_params names IT"):
        model.solve(panel, SPAN, entity_params={"IT": {"cons.y": 0.6}})
    with pytest.raises(ModelError, match="needs a panel"):
        model.solve(panel.entity("DE", by="country"), SPAN)
    with pytest.raises(SpecificationError, match="entity_params are for a model solved per entity"):
        Model(model.equations).solve(panel.entity("DE", by="country"), SPAN, entity_params={"DE": {}})
    with pytest.raises(ModelError, match="Model\\(..., over='country'\\)"):
        Model(model.equations).solve(panel, SPAN)


def test_the_solve_dialog(panel, model):
    from pymetrix.commands import Session, SolveModel

    s = Session()
    s.objects["data"] = panel
    s.objects["cons"] = model.equations[0]
    s.run(SolveModel("cons", "2008Q1", "2009Q4", over="country"))
    script = s.script()
    assert "over='country'" in script
    assert np.allclose(s["solution"].solved().frame.loc["DE", "c"].to_numpy(), _by_hand(panel, "DE", 0.3), atol=1e-9)
