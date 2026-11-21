"""Three solver features: fixing or holding a variable in some periods only (``Scenario(partial=...)``), the inputs
each equation lacks before solving (``Model.missing_inputs``, ``ModelCheck.missing``), and convergence by period
(``Solution.convergence``)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Scenario
from pymetrix.models import fast
from pymetrix.models.solvers import GaussSeidel, ModelError, Newton, PerfectForesight

SPAN = ("2022Q1", "2023Q4")  # eight quarters
FIXED = {"2022Q1": 1.2, "2022Q2": 1.3}  # the first two


def _model():
    return Model([Equation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", name="cons"),  # a simultaneous block (c, i, y)
                  Equation("i = 0.2*y - 0.5*r + 0.4*i[-1]", name="inv"),
                  Equation("y = c + i + g", name="gdp"),
                  Equation("w = 0.9*w[-1] + 0.1*y", name="wage")])


def _data(blank=True):
    index = pd.period_range("2019Q1", "2024Q4", freq="Q", name="time")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({v: 1.0 + 0.1 * rng.standard_normal(len(index)) for v in ("c", "i", "y", "w", "g", "r")}, index=index)
    if blank:  # the solved periods have no endogenous data, as in a projection
        frame.loc["2022Q1":, ["c", "i", "y", "w"]] = np.nan
    return Dataset.from_frame(frame)


SOLVERS = [Newton(), GaussSeidel(tol=1e-13)]
IDS = ["newton", "gauss_seidel"]


@pytest.fixture(params=["compiled", "interpreted"])
def compiled(request, monkeypatch):
    if request.param == "interpreted":
        monkeypatch.setattr(fast, "ENABLED", False)
    return request.param


# 1. partial fixes and holds


@pytest.mark.parametrize("solver", SOLVERS, ids=IDS)
def test_a_fix_in_some_periods_is_the_two_spans_solved_one_after_the_other(solver, compiled):
    model, data = _model(), _data()
    both = model.solve(data, SPAN, Scenario(fix={"c": FIXED}, partial=True), solver=solver)
    first = model.solve(data, ("2022Q1", "2022Q2"), Scenario(fix={"c": FIXED}), solver=solver)
    rest = model.solve(first, ("2022Q3", "2023Q4"), solver=solver)
    np.testing.assert_allclose(both.frame.to_numpy(float), rest.frame.to_numpy(float), atol=1e-12, rtol=0)
    assert both["c"].loc["2022Q1"] == 1.2 and both["c"].loc["2022Q2"] == 1.3
    assert both.iterations == first.iterations + rest.iterations
    # NaN in the path is the same as no value
    nan_path = pd.Series(np.nan, index=pd.period_range(*SPAN, freq="Q"))
    nan_path.loc["2022Q1"], nan_path.loc["2022Q2"] = 1.2, 1.3
    again = model.solve(data, SPAN, Scenario(fix={"c": nan_path}, partial=["c"]), solver=solver)
    np.testing.assert_array_equal(again.frame.to_numpy(float), both.frame.to_numpy(float))


@pytest.mark.parametrize("solver", SOLVERS, ids=IDS)
def test_a_fix_over_the_whole_span_is_unchanged_by_partial(solver, compiled):
    model, data = _model(), _data()
    path = {str(p): 1.0 + 0.01 * k for k, p in enumerate(pd.period_range(*SPAN, freq="Q"))}
    before = model.solve(data, SPAN, Scenario(fix={"c": path}), solver=solver)
    after = model.solve(data, SPAN, Scenario(fix={"c": path}, partial=True), solver=solver)
    assert np.array_equal(before.frame.to_numpy(float), after.frame.to_numpy(float), equal_nan=True)
    assert before.iterations == after.iterations
    pd.testing.assert_frame_equal(before.convergence, after.convergence)


def test_without_partial_a_fix_with_gaps_is_still_an_error():
    model, data = _model(), _data()
    for given in (FIXED, pd.Series({"2022Q1": 1.2, "2022Q2": np.nan, "2022Q3": 1.1})):
        with pytest.raises(ModelError, match="'c' is fixed or targeted but has no value in some simulation periods"):
            model.solve(data, SPAN, Scenario(fix={"c": given}))
    with pytest.raises(ModelError, match="partial= names"):
        model.solve(data, SPAN, Scenario(fix={"c": FIXED}, partial=["i"]))


def test_a_fix_at_the_data_applies_where_the_data_have_values():
    model, data = _model(), _data()
    frame = data.frame.copy()
    frame.loc["2022Q1":"2022Q2", "c"] = [1.2, 1.3]
    with_data = Dataset.from_frame(frame)
    at_data = model.solve(with_data, SPAN, Scenario(fix={"c": None}, partial=True))
    given = model.solve(data, SPAN, Scenario(fix={"c": FIXED}, partial=True))
    np.testing.assert_array_equal(at_data.frame.to_numpy(float), given.frame.to_numpy(float))


def test_a_hold_in_some_periods_is_the_two_spans_solved_one_after_the_other(compiled):
    model, data = _model(), _data()
    held = {"2022Q1": 1.5, "2022Q2": 1.6}
    both = model.solve(data, SPAN, Scenario(hold={"y": held}, free=["g"], partial=True))
    first = model.solve(data, ("2022Q1", "2022Q2"), Scenario(hold={"y": held}, free=["g"]))
    rest = model.solve(first, ("2022Q3", "2023Q4"))  # g at its data from 2022Q3
    np.testing.assert_allclose(both.frame.to_numpy(float), rest.frame.to_numpy(float), atol=1e-12, rtol=0)
    np.testing.assert_allclose(both["y"].loc["2022Q1":"2022Q2"], [1.5, 1.6], atol=1e-12)
    assert both["g"].loc["2022Q3"] == data["g"].loc["2022Q3"]


def test_perfect_foresight_takes_partial_fixes_but_refuses_partial_holds(compiled):
    model, data = _model(), _data()
    newton = model.solve(data, SPAN, Scenario(fix={"c": FIXED}, partial=True))
    stacked = model.solve(data, SPAN, Scenario(fix={"c": FIXED}, partial=True), solver=PerfectForesight(tol=1e-12))
    np.testing.assert_allclose(stacked.frame.to_numpy(float), newton.frame.to_numpy(float), atol=1e-10, rtol=0)
    assert stacked["c"].loc["2022Q1"] == 1.2
    with pytest.raises(ModelError, match="not partial holds"):
        model.solve(data, SPAN, Scenario(hold={"y": {"2022Q1": 1.5}}, free=["g"], partial=True), solver=PerfectForesight())


def test_draws_with_a_partial_fix():
    model, data = _model(), _data()
    sc = Scenario(fix={"c": FIXED}, partial=True)
    errors = pd.DataFrame({"cons": [0.01, -0.02, 0.0, 0.015], "inv": [0.0, 0.01, -0.01, 0.005]})
    solved = model.solve(data, SPAN, sc, draws=20, seed=1, errors=errors)
    j = solved.simulation.names.index("c")
    assert np.all(solved.simulation.values[:, :2, j] == [1.2, 1.3])  # fixed in every draw
    assert np.std(solved.simulation.values[:, 2:, j]) > 0  # solved, with errors, after
    # the convergence table is the deterministic solution's; the draws are summarised in simulation.info
    plain = model.solve(data, SPAN, sc)
    pd.testing.assert_frame_equal(solved.convergence, plain.convergence)
    assert solved.simulation.info["failed"] == 0


# 2. missing inputs


def _two():
    return Model([Equation("c = 0.5*x + 0.3*c[-1]", name="cons"), Equation("y = c + g", name="gdp")])


def _complete():
    index = pd.period_range("2020Q1", periods=12, freq="Q", name="time")
    return pd.DataFrame({v: 1.0 for v in "cxyg"}, index=index)


def test_missing_inputs_names_the_period_equation_and_input():
    model, frame = _two(), _complete()
    frame.loc["2021Q4", "g"] = np.nan
    table = model.missing_inputs(Dataset.from_frame(frame), ("2021Q1", "2022Q2"))
    assert table.to_dict("records") == [{"period": pd.Period("2021Q4", "Q"), "equation": "gdp", "input": "g"}]
    # what the solve then says, one period at a time
    with pytest.raises(ModelError, match="2021Q4: missing values") as info:
        model.solve(Dataset.from_frame(frame), ("2021Q1", "2022Q2"))
    assert list(info.value.convergence["converged"]) == [True] * 3 + [False]


def test_a_missing_lag_before_the_span_is_reported_and_solved_values_are_not():
    model, frame = _two(), _complete()
    frame.loc["2020Q4", "c"] = np.nan
    frame.loc["2021Q1":, ["c", "y"]] = np.nan  # solved: not inputs
    table = model.missing_inputs(Dataset.from_frame(frame), ("2021Q1", "2022Q2"))
    assert table.to_dict("records") == [{"period": pd.Period("2021Q1", "Q"), "equation": "cons", "input": "c[-1]"}]


def test_complete_data_and_the_scenario():
    model, frame = _two(), _complete()
    data = Dataset.from_frame(frame)
    empty = model.missing_inputs(data, ("2021Q1", "2022Q2"))
    assert empty.empty and list(empty.columns) == ["period", "equation", "input"]
    frame.loc["2021Q4", "g"] = np.nan
    frame.loc["2021Q1":, ["c", "y"]] = np.nan
    gap = Dataset.from_frame(frame)
    assert model.missing_inputs(gap, ("2021Q1", "2022Q2"), Scenario(set={"g": {"2021Q4": 2.0}})).empty
    # y fixed where given: gdp is not solved there, so g is not needed; a fix without a value is reported
    assert model.missing_inputs(gap, ("2021Q1", "2022Q2"), Scenario(fix={"y": {"2021Q4": 2.0}}, partial=True)).empty
    table = model.missing_inputs(gap, ("2021Q1", "2022Q2"), Scenario(fix={"y": {"2021Q3": 2.0}}))
    assert ("2021Q4", "gdp", "y") in {(str(p), e, i) for p, e, i in table.itertuples(index=False)}
    # an exogenous series absent from the data: every period
    assert len(model.missing_inputs(Dataset.from_frame(_complete().drop(columns="x")), ("2021Q1", "2022Q2"))) == 6


def test_model_check_shows_them_and_its_rows_are_unchanged():
    model, frame = _two(), _complete()
    frame.loc["2021Q4", "g"] = np.nan
    data = Dataset.from_frame(frame)
    check = model.check(data, ("2021Q1", "2022Q2"))
    assert len(check.missing) == 1 and check.missing.iloc[0]["input"] == "g"
    from pymetrix.models.system import ModelCheck

    assert check.rows == ModelCheck(check.rows, check.system).rows and check.ok  # the warning row, as before
    assert [r for r in check.rows if r[0] == "exogenous paths"][0][1] == "warning"
    assert model.check().missing is None and model.check(data).missing is None
    from pymetrix.views import view_sections

    assert "Missing inputs" in [t for t, _ in view_sections(check)]
    assert "Missing inputs" not in [t for t, _ in view_sections(model.check(Dataset.from_frame(_complete()), ("2021Q1", "2022Q2")))]


# 3. convergence by period


@pytest.mark.parametrize("solver", SOLVERS, ids=IDS)
def test_convergence_by_period(solver, compiled):
    model, data = _model(), _data()
    solved = model.solve(data, SPAN, solver=solver)
    table = solved.convergence
    assert list(table.index) == list(pd.period_range(*SPAN, freq="Q"))
    assert list(table.columns) == ["iterations", "residual", "equation", "converged"]
    assert table["converged"].all() and (table["residual"] < solver.tol).all()
    assert int(table["iterations"].sum()) == solved.iterations and solved.converged
    assert set(table["equation"]) <= {e.name for e in model.model_equations}


@pytest.mark.parametrize("solver", [Newton(max_iter=10), GaussSeidel(max_iter=10)], ids=IDS)
def test_the_period_that_does_not_converge_is_marked(solver, compiled):
    model = Model([Equation("y = 0.5*y[-1] + x - 0.1*y*y*y", name="cubic"), Equation("z = 0.5*z[-1] + y", name="sum")])
    index = pd.period_range("2021Q1", "2024Q4", freq="Q", name="time")
    frame = pd.DataFrame({"y": 1.0, "z": 1.0, "x": 1.0}, index=index)
    frame.loc["2022Q1":, ["y", "z"]] = np.nan
    data = Dataset.from_frame(frame)
    easy = model.solve(data, SPAN, solver=solver)  # each period within a few iterations
    assert easy.convergence["iterations"].max() < 10
    hard = Scenario(set={"x": {"2022Q3": 1000.0}})  # a period far from where the solver starts: 14 iterations
    with pytest.raises(ModelError, match="did not converge") as info:
        model.solve(data, SPAN, hard, solver=solver)
    table = info.value.convergence
    assert list(table.index) == list(pd.period_range("2022Q1", "2022Q3", freq="Q"))  # it stopped there
    assert not table["converged"].iloc[-1] and table["converged"].iloc[:-1].all()
    assert table["iterations"].iloc[-1] == 10
    assert table["residual"].iloc[-1] >= solver.tol and table["equation"].iloc[-1] == "cubic"


def test_perfect_foresight_has_a_row_per_period_with_the_stacked_iterations():
    model, data = _model(), _data()
    solved = model.solve(data, SPAN, solver=PerfectForesight())
    table = solved.convergence
    assert len(table) == 8 and (table["iterations"] == solved.iterations).all()
    assert table["converged"].all() and (table["residual"] < 1e-10).all()
    with pytest.raises(ModelError, match="did not converge") as info:
        model.solve(data, SPAN, solver=PerfectForesight(max_iter=1, tol=1e-300))
    assert len(info.value.convergence) == 8 and not info.value.convergence["converged"].all()


def test_convergence_is_kept_when_saved_and_shown():
    model, data = _model(), _data()
    solved = model.solve(data, SPAN)
    again = pickle.loads(pickle.dumps(solved))
    pd.testing.assert_frame_equal(again.convergence, solved.convergence)
    old = pickle.loads(pickle.dumps(solved))
    del old.__dict__["convergence"]  # as a solution saved before it was recorded
    assert old.convergence is None
    from pymetrix.views import view_sections

    assert "Convergence" in [t for t, _ in view_sections(solved)]
    assert "Convergence" not in [t for t, _ in view_sections(old)]
    first_order = Model([Equation("y = 0.5*y[-1] + e", name="ar")]).solve(
        Dataset.from_frame(pd.DataFrame({"y": 0.0, "e": 0.0}, index=pd.period_range("2000Q1", periods=8, freq="Q", name="time"))),
        ("2000Q2", "2001Q4"), solver=__import__("pymetrix").models.solvers.FirstOrder())
    assert first_order.convergence is None


# the Solve dialog


def test_the_solve_dialog_holds_in_some_periods():
    from pymetrix.commands import Session, SolveModel

    s = Session()
    model, data = _model(), _data()
    s.objects["m"], s.objects["data"] = model, data
    s.run(SolveModel("m", *SPAN, hold="y[2022Q1:2022Q2]=1.5", free="g", partial=True, name="held"))
    sc = Scenario(hold={"y": pd.Series(1.5, index=pd.period_range("2022Q1", "2022Q2", freq="Q"))}, free=["g"], partial=True)
    direct = model.solve(data, SPAN, sc)
    np.testing.assert_allclose(s["held"].frame.to_numpy(float), direct.frame.to_numpy(float), atol=1e-12)
    assert "partial=True" in s.script()
    ns = {**dict(s.objects)}
    exec(s.script(), ns)
    np.testing.assert_allclose(ns["held"].frame.to_numpy(float), direct.frame.to_numpy(float), atol=1e-12)
    with pytest.raises(ModelError, match="no value in some simulation periods"):
        s.run(SolveModel("m", *SPAN, hold="y[2022Q1:2022Q2]=1.5", free="g", name="whole"))
