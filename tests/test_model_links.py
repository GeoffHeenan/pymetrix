"""Links between the entities of a model solved per entity (pymetrix.models.links): foreign variables and aggregates.

Checked against the same multi-country model written out by hand as one stacked model (y_DE, y_FR, ... with the
foreign variables as weighted sums in the equations) and solved by Newton: backward-looking with lags of the links,
each country's own parameters and scenario, and two links at once; forward-looking by perfect foresight. A link made
from data is the weighted data in one round. The weights' forms give the same solution; refusals name what is wrong.
Stochastic simulation: each draw is checked against the stacked model solved with that draw's errors (drawn by hand
with the same seeds) as add factors.
"""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Link, Matrix, Model, Scenario
from pymetrix.errors import SpecificationError
from pymetrix.models.solvers import ModelError, PerfectForesight

C = ["DE", "FR", "UK"]
W = np.array([[0, .6, .4], [.5, 0, .5], [.3, .7, 0]])
SHARES = {"DE": .4, "FR": .35, "UK": .25}
SPAN = ("2008Q1", "2009Q4")
PARAMS = {"a": 5.0, "b": 0.5, "c": 0.3, "d": 0.1, "e": 0.05}


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    rows = []
    for k in C:
        for i, p in enumerate(pd.period_range("2000Q1", "2009Q4", freq="Q")):
            rows.append({"country": k, "time": str(p), "y": 100 + i + rng.normal(), "g": 20 + rng.normal(), "oil": 50 + rng.normal()})
    return Dataset.from_frame(pd.DataFrame(rows), entity="country", time="time", freq="Q")


def _wide(panel):
    wide = panel.frame.unstack("country")
    wide.columns = [f"{v}_{k}" for v, k in wide.columns]
    return Dataset.from_frame(wide.reset_index(), time="time", freq="Q")


def _sum(weights, var, lag=""):
    return "(" + " + ".join(f"{w}*{var}_{k}{lag}" for w, k in zip(weights, C) if w) + ")"


def _solved(sol, k):
    return sol.solved().frame.loc[k, "y"].to_numpy()


def test_backward_links_against_the_stacked_model(panel):
    eq = Equation("y ~ a + b*y[-1] + c*ys + d*ys[-1] + e*world + g", params=list(PARAMS), name="y")
    model = Model([eq], over="country", params=PARAMS,
                  links=[Link("ys", "y", Matrix(W, rows=C, cols=C)), Link("world", "y", SHARES)])
    sol = model.solve(panel, SPAN, entity_params={"UK": {"b": 0.6}}, scenario={"FR": Scenario(shift={"g": 5})})
    shares = [SHARES[k] for k in C]
    stacked = Model([Equation(f"y_{k} ~ a + {'b_uk' if k == 'UK' else 'b'}*y_{k}[-1] + c*{_sum(W[i], 'y')} + d*{_sum(W[i], 'y', '[-1]')}"
                              f" + e*{_sum(shares, 'y')} + g_{k}", params=["a", "b_uk" if k == "UK" else "b", "c", "d", "e"], name=f"y_{k}")
                     for i, k in enumerate(C)],
                    params={**PARAMS, "b_uk": 0.6})
    ref = stacked.solve(_wide(panel), SPAN, scenario=Scenario(shift={"g_FR": 5}))
    periods = pd.period_range(*SPAN, freq="Q")
    for k in C:
        assert np.max(np.abs(_solved(sol, k) - ref.frame[f"y_{k}"].reindex(periods).to_numpy())) < 1e-8
    # the links in the solution are the weighted solutions
    solved = sol.solved().frame
    for i, k in enumerate(C):
        ys = sum(W[i, j] * solved.loc[C[j], "y"].to_numpy() for j in range(3))
        assert np.allclose(solved.loc[k, "ys"].to_numpy(), ys, atol=1e-9)
        assert np.allclose(solved.loc[k, "world"].to_numpy(), sum(SHARES[j] * solved.loc[j, "y"].to_numpy() for j in C), atol=1e-9)
    assert sol.links == ("ys", "world") and 1 < sol.rounds < 60 and "linked by ys, world" in repr(sol)
    # before the span: the links from the data (what the lags use)
    history = sol.frame.loc["DE"].loc[pd.Period("2007Q4", freq="Q")]
    data = panel.frame.xs(pd.Period("2007Q4", freq="Q"), level="time")["y"]
    assert np.isclose(history["ys"], 0.6 * data["FR"] + 0.4 * data["UK"])


def test_forward_looking_links_against_the_stacked_model(panel):
    eq = Equation("y ~ a + b*y[+1] + c*ys + g", params=["a", "b", "c"], name="y")
    params = {"a": 2.0, "b": 0.4, "c": 0.3}
    model = Model([eq], over="country", params=params, links=[Link("ys", "y", {k: dict(zip(C, W[i])) for i, k in enumerate(C)})])
    sol = model.solve(panel, SPAN, solver=PerfectForesight(), scenario={"DE": Scenario(shift={"g": {"2009Q1": 4}})})
    stacked = Model([Equation(f"y_{k} ~ a + b*y_{k}[+1] + c*{_sum(W[i], 'y')} + g_{k}", params=list(params), name=f"y_{k}")
                     for i, k in enumerate(C)], params=params)
    ref = stacked.solve(_wide(panel), SPAN, solver=PerfectForesight(), scenario=Scenario(shift={"g_DE": {"2009Q1": 4}}))
    periods = pd.period_range(*SPAN, freq="Q")
    for k in C:
        assert np.max(np.abs(_solved(sol, k) - ref.frame[f"y_{k}"].reindex(periods).to_numpy())) < 1e-8
    before = sol.solved().frame.loc["FR", "y"].loc[pd.Period("2008Q4", freq="Q")]
    alone = Model([eq], over="country", params=params).solve(panel.with_series(ys="y * 0 + 100"), SPAN, solver=PerfectForesight())
    assert before != alone.solved().frame.loc["FR", "y"].loc[pd.Period("2008Q4", freq="Q")]  # DE's anticipated shock reaches FR


def test_a_link_from_data_and_the_weights_forms(panel):
    eq = Equation("y ~ a + b*y[-1] + c*ys + e*oils + g", params=["a", "b", "c", "e"], name="y")
    forms = [Matrix(W, rows=C, cols=C), pd.DataFrame(W, index=C, columns=C), {k: dict(zip(C, W[i])) for i, k in enumerate(C)}]
    results = []
    for weights in forms:
        model = Model([eq], over="country", params=PARAMS, links=[Link("ys", "y", weights), Link("oils", "oil", SHARES)])
        results.append(model.solve(panel, SPAN))
    for other in results[1:]:
        assert np.allclose(other.solved().frame["y"].to_numpy(), results[0].solved().frame["y"].to_numpy(), atol=1e-12)
    oil = panel.frame["oil"].unstack("country")
    expected = sum(SHARES[k] * oil[k] for k in C)
    assert np.allclose(results[0].frame.loc["UK", "oils"].to_numpy(), expected.to_numpy(), atol=1e-12)
    single_row = Matrix(np.array([[SHARES[k] for k in C]]), rows=["all"], cols=C)
    same = Model([eq], over="country", params=PARAMS, links=[Link("ys", "y", forms[0]), Link("oils", "oil", single_row)]).solve(panel, SPAN)
    assert np.allclose(same.solved().frame["y"].to_numpy(), results[0].solved().frame["y"].to_numpy(), atol=1e-12)
    # only data: one round
    data_only = Model([Equation("y ~ a + b*y[-1] + e*oils + g", params=["a", "b", "e"], name="y")], over="country",
                      params={"a": 5.0, "b": 0.5, "e": 0.05}, links=[Link("oils", "oil", SHARES)]).solve(panel, SPAN)
    assert data_only.rounds == 1


def test_refusals(panel):
    eq = Equation("y ~ a + b*y[-1] + c*ys + g", params=["a", "b", "c"], name="y")
    params = {"a": 5.0, "b": 0.5, "c": 0.3}
    trade = Matrix(W, rows=C, cols=C)

    def solve(links, **kw):
        return Model([eq], over="country", params=params, links=links).solve(panel, SPAN, **kw)

    with pytest.raises(SpecificationError, match="give over="):
        Model([eq], params=params, links=[Link("ys", "y", trade)])
    with pytest.raises(SpecificationError, match="cannot be made from itself"):
        Link("y", "y", trade)
    with pytest.raises(ModelError, match="IT, not entities"):
        solve([Link("ys", "y", {"DE": {"IT": 1.0}, "FR": {"DE": 1.0}, "UK": {"DE": 1.0}})])
    with pytest.raises(ModelError, match="no weights for UK"):
        solve([Link("ys", "y", {"DE": {"FR": 1.0}, "FR": {"DE": 1.0}})])
    with pytest.raises(ModelError, match="no equation uses it"):
        solve([Link("ys", "y", trade), Link("zs", "y", trade)])
    with pytest.raises(ModelError, match="neither a variable"):
        solve([Link("ys", "q", trade)])
    with pytest.raises(ModelError, match="cannot set the link ys"):
        solve([Link("ys", "y", trade)], scenario=Scenario(set={"ys": 1.0}))
    with pytest.raises(SpecificationError, match="coefficient draws with links"):
        solve([Link("ys", "y", trade)], draws=10, shock_sd={"y": 1.0}, coefficients=True)
    with pytest.raises(SpecificationError, match="backward-looking models"):
        solve([Link("ys", "y", trade)], draws=10, shock_sd={"y": 1.0}, solver=PerfectForesight())
    with pytest.raises(SpecificationError, match="not stochastic equations"):
        solve([Link("ys", "y", trade)], draws=10, shock_sd={"q": 1.0})
    determined = Model([eq, Equation("ys = 2*y", name="ys")], over="country", params=params, links=[Link("ys", "y", trade)])
    with pytest.raises(ModelError, match="the model determines ys"):
        determined.solve(panel, SPAN)
    # two countries that are each other's whole economy, with no own dynamics: the joint system has no solution
    rows = panel.frame.reset_index()
    pair = Dataset.from_frame(rows[rows["country"] != "UK"], entity="country", time="time", freq="Q")
    loop = Model([Equation("y = ys + g", name="y")], over="country", links=[Link("ys", "y", {"DE": {"FR": 1.0}, "FR": {"DE": 1.0}})])
    with pytest.raises(ModelError, match="did not settle"):
        loop.solve(pair, ("2008Q1", "2008Q1"))


def test_the_solve_dialog_with_links(panel, tmp_path):
    from pymetrix.commands import Session, SolveModel
    from pymetrix.testing import check_project

    eq = Equation("y ~ a + b*y[-1] + c*ys + g", params=["a", "b", "c"], name="y")
    s = Session()
    s.objects["data"] = panel
    s.objects["fy"] = eq
    s.objects["trade"] = Matrix(W, rows=C, cols=C)
    command = SolveModel("fy", "2008Q1", "2009Q4", params="a=5, b=0.5, c=0.3", over="country", links="ys = trade @ y")
    assert command.validate(s) == []
    assert SolveModel("fy", "2008Q1", "2009Q4", over="country", links="ys = y").validate(s)[0].startswith("links: 'ys = y'")
    assert SolveModel("fy", "2008Q1", "2009Q4", over="country", links="ys = shares @ y").validate(s) == [
        "links: no shares in the session (a matrix of weights)"]
    assert SolveModel("fy", "2008Q1", "2009Q4", links="ys = trade @ y").validate(s) == ["links: give the panel's entity axis (over) too"]
    s.run(command)
    script = s.script()
    assert "links=[Link('ys', variable='y', weights=trade)]" in script
    direct = Model([eq], over="country", params={"a": 5.0, "b": 0.5, "c": 0.3}, links=[Link("ys", "y", s.objects["trade"])]).solve(panel, SPAN)
    assert np.allclose(s["solution"].solved().frame["y"].to_numpy(), direct.solved().frame["y"].to_numpy(), atol=1e-12)
    from pymetrix.commands import DefineEquation

    again = Session()  # every input a step or data, so the saved project can rebuild them
    again.objects["data"] = panel
    again.run(DefineEquation("y ~ a + b*y[-1] + c*ys + g", params="a, b, c", name="fy"))
    again.run_code("trade = Matrix([[0, .6, .4], [.5, 0, .5], [.3, .7, 0]], rows=['DE', 'FR', 'UK'], cols=['DE', 'FR', 'UK'])")
    check_project(again, command, tmp_path)  # saved and reopened: the same solution and script


def test_draws_of_linked_entities_against_the_stacked_model(panel):
    """Each draw is the joint solution of every country with that draw's errors: the stacked model, solved with the
    same errors (each country's drawn with seed + its position, as for entities solved alone) as add factors, and the
    scenario's own add factor and shift, gives the same paths. The seed makes the draws reproducible."""
    eq = Equation("y ~ a + b*y[-1] + c*ys + d*ys[-1] + e*world + g", params=list(PARAMS), name="y")
    model = Model([eq], over="country", params=PARAMS,
                  links=[Link("ys", "y", Matrix(W, rows=C, cols=C)), Link("world", "y", SHARES)])
    D, seed, sd = 4, 3, 0.7
    scenario = {"FR": Scenario(shift={"g": 5}, adjust={"y": 1.0})}
    sol = model.solve(panel, SPAN, draws=D, seed=seed, shock_sd={"y": sd}, scenario=scenario)
    periods = pd.period_range(*SPAN, freq="Q")
    shares = [SHARES[k] for k in C]
    stacked = Model([Equation(f"y_{k} ~ a + b*y_{k}[-1] + c*{_sum(W[i], 'y')} + d*{_sum(W[i], 'y', '[-1]')}"
                              f" + e*{_sum(shares, 'y')} + g_{k}", params=list(PARAMS), name=f"y_{k}") for i, k in enumerate(C)],
                    params=PARAMS)
    errors = {k: np.random.default_rng(seed + i).standard_normal((D, len(periods))) * sd for i, k in enumerate(C)}
    for d in range(D):
        adjust = {f"y_{k}": pd.Series(errors[k][d] + (1.0 if k == "FR" else 0.0), index=periods) for k in C}
        ref = stacked.solve(_wide(panel), SPAN, scenario=Scenario(shift={"g_FR": 5}, adjust=adjust)).frame
        for i, k in enumerate(C):
            sims = sol.solution(k).simulation
            assert np.max(np.abs(sims["y"][d].to_numpy() - ref[f"y_{k}"].reindex(periods).to_numpy())) < 1e-8
            ys = sum(W[i, j] * ref[f"y_{C[j]}"].reindex(periods).to_numpy() for j in range(3))
            assert np.max(np.abs(sims["ys"][d].to_numpy() - ys)) < 1e-8  # the links of the draw, from the draw
    first = sol.solution("DE").simulation
    assert first.names == ("y", "ys", "world") and first.values.shape == (D, len(periods), 3)
    assert first.info["draws"] == D and first.info["failed"] == 0 and first.info["seed"] == seed
    assert sol.solution("UK").simulation.info["seed"] == seed + 2
    # the deterministic solution is the one without draws, and the seed reproduces the draws
    plain = model.solve(panel, SPAN, scenario=scenario)
    assert np.allclose(_solved(sol, "DE"), _solved(plain, "DE"), atol=1e-10)
    again = model.solve(panel, SPAN, draws=D, seed=seed, shock_sd={"y": sd}, scenario=scenario)
    assert np.array_equal(again.solution("FR").simulation.values, sol.solution("FR").simulation.values)
