"""Options of the period-by-period solvers: the convergence test (``convergence="residual" | "step" | "both"``),
retrying a failed period with other solvers (``Fallback``), Newton's backtracking line search (``line_search``), a
Jacobian kept for the period (``refresh="period"``), and block-recursive solution (``Model.solve(ordering="blocks")``).

Independent checks: today's solvers, copied below as they were before these options, give exactly (to 0) what the
defaults give on the existing models (the test models of the solver tests, the FRB/US excerpt, the Solve dialog's
example); closed forms (a saturating equation, a cube root, a linear simultaneous pair); the blocks against
``Model.blocks()`` (Tarjan on the dependency graph, written separately); evaluations counted by wrapping the period
system's own methods, not wall-clock time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Fallback, GaussSeidel, Model, ModelError, Newton, Scenario
from pymetrix.errors import SpecificationError
from pymetrix.models import fast
from pymetrix.models import model as model_module
from pymetrix.models.eviews import read_eviews_xml
from pymetrix.models.ordering import plan
from pymetrix.models.solvers import FirstOrder, PerfectForesight, _finite_jacobian, _stopped, _worst

SPAN = ("2022Q1", "2023Q4")
BASE_COLUMNS = ["iterations", "residual", "equation", "converged"]


# today's solvers, copied as they were before the options, run through the same solve


class _TodayNewton:
    tol, max_iter = 1e-10, 50

    def solve_period(self, system):
        x = system.x0.copy()
        for it in range(self.max_iter + 1):
            F = system.residuals(x)
            if not np.all(np.isfinite(F)):
                raise _stopped(f"non-finite residual in {_worst(system, F)}", it)
            if np.max(np.abs(F), initial=0.0) < self.tol:
                return x, it
            if it == self.max_iter:
                break
            try:
                x = x + np.linalg.solve(_finite_jacobian(system, x, F), -F)
            except np.linalg.LinAlgError:
                raise _stopped("the Jacobian is singular; check that each equation determines its variable", it) from None
        raise _stopped(f"Newton did not converge in {self.max_iter} iterations; worst is {_worst(system, F)}", self.max_iter)


class _TodayGaussSeidel:
    def __init__(self, tol=1e-10, max_iter=500, damping=1.0):
        self.tol, self.max_iter, self.damping = tol, max_iter, damping

    def solve_period(self, system):
        x = system.x0.copy()
        n = len(x)
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                f, d, j = system.own(i, x)
                if j < 0:
                    raise ModelError("Gauss-Seidel needs every equation to solve for its own variable; use Newton for target/free scenarios")
                if d == 0 or not np.isfinite(d):
                    raise ModelError(f"equation {system.equation_names[i]!r} does not depend on its own variable here")
                x[j] -= self.damping * f / d
            F = system.residuals(x)
            if np.max(np.abs(F), initial=0.0) < self.tol:
                return x, it
        raise _stopped(f"Gauss-Seidel did not converge in {self.max_iter} iterations; worst is {_worst(system, F)}", self.max_iter)


# models


def _model(scale: float = 1.0) -> Model:
    """A simultaneous block (c, i, y, w) with a log; ``scale`` multiplies the identity, changing only its units."""
    gdp = "y = c + i + g" if scale == 1.0 else f"{scale!r}*y = {scale!r}*(c + i + g)"
    return Model([Equation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", name="cons"),
                  Equation("i = 0.2*y - 0.5*r + 0.4*i[-1]", name="inv"),
                  Equation(gdp, name="gdp"),
                  Equation("w = 0.9*w[-1] + 0.1*y", name="wage")])


def _data() -> Dataset:
    index = pd.period_range("2019Q1", "2024Q4", freq="Q", name="time")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({v: 1.0 + 0.1 * rng.standard_normal(len(index)) for v in ("c", "i", "y", "w", "g", "r")}, index=index)
    frame.loc["2022Q1":, ["c", "i", "y", "w"]] = np.nan
    return Dataset.from_frame(frame)


def _economy():
    cons = Equation("c ~ a + b*y + d*c[-1]", params=["a", "b", "d"]).with_params(a=10, b=0.6, d=0.2)
    inv = Equation("i ~ e + f*y[-1]", params=["e", "f"]).with_params(e=5, f=0.1)
    model = Model([cons, inv, Equation("y = c + i + g")])
    periods = pd.period_range("2000Q1", "2010Q4", freq="Q", name="time")
    g = 20 + np.arange(len(periods), dtype=float)
    frame = pd.DataFrame({"c": 60.0, "i": 15.0, "y": 95.0, "g": g}, index=periods)
    frame.loc["2005Q1":, ["c", "i", "y"]] = np.nan
    return model, Dataset.from_frame(frame), ("2005Q1", "2010Q4")


def _excerpt():
    fixtures = Path(__file__).parent / "fixtures" / "eviews"
    model = read_eviews_xml(fixtures / "frbus_excerpt.xml").model
    frame = pd.read_csv(fixtures / "frbus_longbase_excerpt.csv", comment="#")
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q", name="time")
    missing = [v for v in model.exogenous if v.endswith("_aerr") and v not in frame.columns]
    frame = pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1)
    span = ("2026Q2", "2027Q2")
    factors = model.add_factors(Dataset.from_frame(frame), span)
    blank = frame.copy()
    blank.loc["2026Q2":"2027Q2", list(model.endogenous)] = np.nan
    return model, Dataset.from_frame(blank), span, Scenario(adjust=factors)


def _example():
    """The Solve dialog's example: two fitted equations and an identity on the sample data."""
    data = Dataset.example("sample_macro")
    cons = Equation("c ~ y + c[-1]").fit(data)
    inv = Equation("i ~ y[-1] + i[-1]").fit(data)
    return Model([cons, inv, Equation("y = c + i + g")]), data, ("2020Q1", "2024Q4")


def _cases():
    model, data = _model(), _data()
    held = Scenario(hold={"y": {"2022Q2": 1.4, "2022Q3": 1.5}}, free=["g"], partial=True)
    fixed = Scenario(fix={"c": {"2022Q1": 1.2, "2022Q2": 1.3}}, partial=True, shift={"g": 0.2})
    economy, economy_data, economy_span = _economy()
    excerpt, excerpt_data, excerpt_span, adjust = _excerpt()
    example, example_data, example_span = _example()
    return {
        "convergence_model": (model, data, SPAN, None),
        "partial_hold": (model, data, SPAN, held),
        "partial_fix": (model, data, SPAN, fixed),
        "economy": (economy, economy_data, economy_span, Scenario(shift={"g": 1.0})),
        "frbus_excerpt": (excerpt, excerpt_data, excerpt_span, adjust),
        "dialog_example": (example, example_data, example_span, Scenario(shift={"g": 5})),
    }


def _same(a, b):
    assert np.array_equal(a.frame.to_numpy(float), b.frame.to_numpy(float), equal_nan=True)
    assert a.iterations == b.iterations
    pd.testing.assert_frame_equal(a.convergence, b.convergence, check_exact=True)


# defaults: today's results exactly


@pytest.mark.parametrize("case", ["convergence_model", "partial_hold", "partial_fix", "economy", "frbus_excerpt", "dialog_example"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_the_defaults_give_todays_results_exactly(case, compiled, monkeypatch):
    if not compiled:
        monkeypatch.setattr(fast, "ENABLED", False)
    model, data, span, scenario = _cases()[case]
    today = model.solve(data, span, scenario, solver=_TodayNewton())
    for solver in (None, Newton(), Newton(convergence="residual", line_search=False, refresh="iteration")):
        solved = model.solve(data, span, scenario, solver=solver)
        _same(solved, today)
        assert list(solved.convergence.columns) == BASE_COLUMNS  # no new columns unless an option asks for them
    if case not in ("partial_hold",):  # Gauss-Seidel cannot hold a variable with a free one
        today = model.solve(data, span, scenario, solver=_TodayGaussSeidel(tol=1e-12))
        _same(model.solve(data, span, scenario, solver=GaussSeidel(tol=1e-12)), today)
        _same(model.solve(data, span, scenario, solver=GaussSeidel(tol=1e-12), ordering="joint"), today)


@pytest.mark.parametrize("solver,today", [(Newton(max_iter=10), _TodayNewton()), (GaussSeidel(max_iter=10), _TodayGaussSeidel(max_iter=10))],
                         ids=["newton", "gauss_seidel"])
def test_a_failing_period_fails_as_today(solver, today):
    if isinstance(today, _TodayNewton):
        today.max_iter = 10
    model = Model([Equation("y = 0.5*y[-1] + x - 0.1*y*y*y", name="cubic"), Equation("z = 0.5*z[-1] + y", name="sum")])
    index = pd.period_range("2021Q1", "2024Q4", freq="Q", name="time")
    frame = pd.DataFrame({"y": 1.0, "z": 1.0, "x": 1.0}, index=index)
    frame.loc["2022Q1":, ["y", "z"]] = np.nan
    data = Dataset.from_frame(frame)
    hard = Scenario(set={"x": {"2022Q3": 1000.0}})
    with pytest.raises(ModelError) as now:
        model.solve(data, SPAN, hard, solver=solver)
    with pytest.raises(ModelError) as before:
        model.solve(data, SPAN, hard, solver=today)
    assert str(now.value) == str(before.value)
    pd.testing.assert_frame_equal(now.value.convergence, before.value.convergence, check_exact=True)


# A1: the convergence test


@pytest.mark.parametrize("solver", [Newton(convergence="step"), GaussSeidel(convergence="step")], ids=["newton", "gauss_seidel"])
def test_the_step_test_does_not_depend_on_an_equations_units(solver):
    data = _data()
    plain = _model().solve(data, SPAN, solver=solver)
    scaled = _model(1e6).solve(data, SPAN, solver=solver)  # the identity times a million: the same steps
    assert list(plain.convergence["iterations"]) == list(scaled.convergence["iterations"])
    np.testing.assert_allclose(scaled.frame.to_numpy(float), plain.frame.to_numpy(float), atol=1e-10, equal_nan=True)
    assert (plain.convergence["test"] == "step").all()
    reference = _model().solve(data, SPAN, solver=Newton(tol=1e-13))
    np.testing.assert_allclose(plain.frame.to_numpy(float), reference.frame.to_numpy(float), atol=1e-9, equal_nan=True)


def test_the_residual_test_depends_on_the_units():
    """With the identity a million times larger its residual is a million times larger at the same point: Newton's
    steps are the same (the Jacobian's row scales too), so the residual test needs at least as many of them, and at
    1e-10 rounding (1e6 * 1e-16) can stop it converging at all; the step test needs the same number either way."""
    data = _data()
    step = _model(1e6).solve(data, SPAN, solver=Newton(convergence="step"))
    try:
        residual = _model(1e6).solve(data, SPAN, solver=Newton())
    except ModelError as exc:
        assert "did not converge" in str(exc)
    else:
        assert (residual.convergence["iterations"] >= _model().solve(data, SPAN).convergence["iterations"]).all()
    assert step.convergence["converged"].all()


def test_both_tests_and_their_record():
    """A linear period: Newton lands on the solution in one step (the residual test stops there, iteration 1); the
    step test needs a second, zero step to see it has stopped moving (iteration 2), and "both" needs both."""
    model = Model([Equation("y = 0.5*y[-1] + x", name="ar")])
    index = pd.period_range("2000Q1", periods=6, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"y": 1.0, "x": 1.0}, index=index).assign(y=[1.0] + [np.nan] * 5))
    counts = {test: model.solve(data, ("2000Q2", "2001Q2"), solver=Newton(convergence=test)).convergence for test in ("residual", "step", "both")}
    assert list(counts["residual"].columns) == BASE_COLUMNS and (counts["residual"]["iterations"] == 1).all()
    assert (counts["step"]["iterations"] == 2).all() and (counts["step"]["test"] == "step").all()
    assert (counts["both"]["iterations"] == 2).all() and (counts["both"]["test"] == "both").all()
    y = 1.0
    for value in model.solve(data, ("2000Q2", "2001Q2"), solver=Newton(convergence="both"))["y"].iloc[1:]:
        y = 0.5 * y + 1.0
        assert value == pytest.approx(y, abs=1e-14)
    with pytest.raises(ModelError, match="convergence is"):
        model.solve(data, ("2000Q2", "2001Q2"), solver=Newton(convergence="steps"))


# A2: Fallback


def _feedback():
    """y = x - k z, z = w + 0.9 y: Gauss-Seidel's sweep multiplies the error by -0.9 k, so it diverges where k = 1.5
    (one period) and converges elsewhere (k = 0.5). No lags, and the data give each period's starting values, so
    every period is solved on its own. Closed form: y = (x - k w) / (1 + 0.9 k)."""
    index = pd.period_range("2000Q1", periods=8, freq="Q", name="time")
    k = np.full(len(index), 0.5)
    k[3] = 1.5
    frame = pd.DataFrame({"y": 1.0, "z": 1.0, "x": 1.0 + 0.1 * np.arange(len(index)), "w": 0.5, "k": k}, index=index)
    model = Model([Equation("y = x - k*z", name="ye"), Equation("z = w + 0.9*y", name="ze")])
    return model, Dataset.from_frame(frame), frame


def test_fallback_retries_only_the_failed_period():
    model, data, frame = _feedback()
    span = ("2000Q2", "2001Q4")
    with pytest.raises(ModelError, match="did not converge"):
        model.solve(data, span, solver=GaussSeidel())
    solver = Fallback(GaussSeidel(), GaussSeidel(damping=0.5, max_iter=400), Newton())
    solved = model.solve(data, span, solver=solver)
    table = solved.convergence
    assert list(table["solver"]) == ["GaussSeidel()"] * 2 + ["GaussSeidel(max_iter=400, damping=0.5)"] + ["GaussSeidel()"] * 4
    assert table.loc[pd.Period("2000Q4", "Q"), "iterations"] > 500  # the failed attempt's 500 sweeps, then the retry's
    exact = (frame["x"] - frame["k"] * frame["w"]) / (1 + 0.9 * frame["k"])
    np.testing.assert_allclose(solved["y"].loc["2000Q2":], exact.loc["2000Q2":], atol=1e-9)
    for period in ("2000Q2", "2000Q3", "2001Q1", "2001Q2", "2001Q3", "2001Q4"):  # first time: as plain Gauss-Seidel
        alone = model.solve(data, (period, period), solver=GaussSeidel())
        for v in ("y", "z"):
            assert abs(solved[v].loc[period] - alone[v].loc[period]) <= 1e-15


def test_when_every_solver_fails_each_attempt_is_listed():
    model, data, _ = _feedback()
    with pytest.raises(ModelError) as info:
        model.solve(data, ("2000Q2", "2001Q4"), solver=Fallback(GaussSeidel(max_iter=30), Newton(max_iter=0)))
    message = str(info.value)
    assert message.startswith("every solver failed: GaussSeidel(max_iter=30): ")
    assert "; Newton(max_iter=0): " in message and message.count("worst is equation") == 2
    table = info.value.convergence
    assert not table["converged"].iloc[-1] and table.index[-1] == pd.Period("2000Q4", "Q")
    assert table["iterations"].iloc[-1] == 30


def test_fallback_settings_are_saved_and_written_as_code():
    from pymetrix.calls import python_code
    from pymetrix.codec import decode, encode

    solver = Fallback(GaussSeidel(convergence="step"), Newton(line_search=True, refresh="period"))
    assert decode(encode(solver)) == solver
    code = python_code(solver)
    assert code == "Fallback([GaussSeidel(convergence='step'), Newton(line_search=True, refresh='period')])"
    assert eval(code, {"Fallback": Fallback, "GaussSeidel": GaussSeidel, "Newton": Newton}) == solver
    assert Fallback([GaussSeidel(), Newton()]) == Fallback(GaussSeidel(), Newton()) == Fallback(solvers=(GaussSeidel(), Newton()))
    with pytest.raises(ModelError, match="period-by-period"):
        Fallback(Newton(), PerfectForesight())
    with pytest.raises(ModelError, match="at least one"):
        Fallback()


# A3: line search


def test_the_line_search_converges_where_newton_diverges():
    """y / sqrt(1 + y^2) = x saturates: from y = 3 the full Newton step overshoots to -11, and the next to thousands
    (the derivative (1 + y^2)^-1.5 vanishes). Backtracking reaches the root x / sqrt(1 - x^2)."""
    model = Model([Equation("y/sqrt(1 + y^2) = x", name="sat")])
    index = pd.period_range("2000Q1", periods=4, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"y": 3.0, "x": 0.5}, index=index))
    with pytest.raises(ModelError):
        model.solve(data, ("2000Q2", "2000Q3"))
    solved = model.solve(data, ("2000Q2", "2000Q3"), solver=Newton(line_search=True))
    np.testing.assert_allclose(solved["y"].loc["2000Q2":"2000Q3"], 0.5 / np.sqrt(0.75), atol=1e-12)
    assert solved.convergence["converged"].all()
    off = model.solve(Dataset.from_frame(pd.DataFrame({"y": 0.6, "x": 0.5}, index=index)), ("2000Q2", "2000Q3"),
                      solver=Newton(line_search=False))
    _same(off, model.solve(Dataset.from_frame(pd.DataFrame({"y": 0.6, "x": 0.5}, index=index)), ("2000Q2", "2000Q3"),
                           solver=_TodayNewton()))


def test_the_line_search_keeps_newtons_steps_near_the_solution():
    """Where the full step reduces the residuals (near the solution), the line search takes it: the same iterations
    and solution as plain Newton."""
    data = _data()
    plain = _model().solve(data, SPAN)
    searched = _model().solve(data, SPAN, solver=Newton(line_search=True))
    assert list(searched.convergence["iterations"]) == list(plain.convergence["iterations"])
    np.testing.assert_allclose(searched.frame.to_numpy(float), plain.frame.to_numpy(float), atol=1e-14, equal_nan=True)


# A5: the Jacobian kept for the period


def _count_jacobians(monkeypatch):
    calls = {"n": 0}
    original = model_module._PeriodSystem.jacobian

    def counted(self, x):
        calls["n"] += 1
        return original(self, x)

    monkeypatch.setattr(model_module._PeriodSystem, "jacobian", counted)
    return calls


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_a_jacobian_kept_for_the_period(compiled, monkeypatch):
    if not compiled:
        monkeypatch.setattr(fast, "ENABLED", False)
    data = _data()
    calls = _count_jacobians(monkeypatch)
    every = _model().solve(data, SPAN)
    full_count = calls["n"]
    assert full_count == int(every.convergence["iterations"].sum())  # one Jacobian per Newton iteration
    calls["n"] = 0
    kept = _model().solve(data, SPAN, solver=Newton(refresh="period"))
    np.testing.assert_allclose(kept.frame.to_numpy(float), every.frame.to_numpy(float), atol=1e-10, equal_nan=True)
    assert calls["n"] == int(kept.convergence["jacobians"].sum()) < full_count
    assert (kept.convergence["jacobians"] == 1).all()  # mild nonlinearity: one per period is enough
    with pytest.raises(ModelError, match="refresh is"):
        _model().solve(data, SPAN, solver=Newton(refresh="never"))


def test_a_kept_jacobian_is_evaluated_again_when_convergence_slows():
    """A strongly nonlinear period: the step from the starting value's Jacobian leaves most of the residual, so a new
    Jacobian is taken; the solution is Newton's (closed form: y^3 = x)."""
    model = Model([Equation("y^3 = x", name="cube")])
    index = pd.period_range("2000Q1", periods=3, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"y": 10.0, "x": 2.0}, index=index))
    kept = model.solve(data, ("2000Q2", "2000Q3"), solver=Newton(refresh="period"))
    np.testing.assert_allclose(kept["y"].loc["2000Q2":"2000Q3"], 2.0 ** (1 / 3), atol=1e-12)
    every = model.solve(data, ("2000Q2", "2000Q3"))
    assert 1 < kept.convergence["jacobians"].iloc[0] < every.convergence["iterations"].iloc[0]


def test_options_compose():
    data = _data()
    reference = _model().solve(data, SPAN, solver=Newton(tol=1e-13))
    solver = Fallback(GaussSeidel(convergence="step"), Newton(line_search=True, refresh="period"))
    for ordering in ("joint", "blocks"):
        solved = _model().solve(data, SPAN, solver=solver, ordering=ordering)
        np.testing.assert_allclose(solved.frame.to_numpy(float), reference.frame.to_numpy(float), atol=1e-8, equal_nan=True)
        assert (solved.convergence["solver"] == "GaussSeidel(convergence='step')").all()
        assert (solved.convergence["test"] == "step").all()
    assert solved.provenance.method["ordering"] == "blocks" and solved.provenance.method["type"] == "Fallback"
    newton = _model().solve(data, SPAN, solver=Fallback(GaussSeidel(max_iter=2), Newton(line_search=True, refresh="period")),
                            ordering="blocks")
    assert newton.convergence["solver"].str.startswith("Newton(").all() and (newton.convergence["jacobians"] >= 1).all()


# A4: block-recursive solution


def _mixed():
    """Recursive equations before and after a simultaneous block (c, y with a log), one equation nonlinear in its own
    variable (solved by the solver, not once), a self-referencing linear one, and an exogenous chain."""
    model = Model([Equation("a = 0.5*a[-1] + x", name="ae"),
                   Equation("b = 2*a + 0.1*b[-1]", name="be"),
                   Equation("c = 0.4*y + 0.2*log(b) + 0.3*c[-1]", name="ce"),
                   Equation("y = c + 0.5*b + g", name="ye"),
                   Equation("log(q) = 0.1*y + 0.5*log(q[-1])", name="qe"),
                   Equation("s = 0.5*s + 0.2*q + 0.1*s[-1]", name="se"),
                   Equation("t = s + q - y", name="te")])
    index = pd.period_range("2000Q1", periods=16, freq="Q", name="time")
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({v: 1.0 + 0.1 * rng.standard_normal(len(index)) for v in ("a", "b", "c", "y", "q", "s", "t", "x", "g")},
                         index=index)
    frame.loc["2001Q1":, ["a", "b", "c", "y", "q", "s", "t"]] = np.nan
    return model, Dataset.from_frame(frame)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
@pytest.mark.parametrize("solver", [Newton(), GaussSeidel(tol=1e-13), Fallback(GaussSeidel(max_iter=3), Newton())],
                         ids=["newton", "gauss_seidel", "fallback"])
def test_blocks_give_the_joint_solution(solver, compiled, monkeypatch):
    if not compiled:
        monkeypatch.setattr(fast, "ENABLED", False)
    model, data = _mixed()
    span = ("2001Q1", "2003Q4")
    joint = model.solve(data, span, solver=Newton(tol=1e-13))
    blocks = model.solve(data, span, solver=solver, ordering="blocks")
    np.testing.assert_allclose(blocks.frame.to_numpy(float), joint.frame.to_numpy(float), atol=1e-10, equal_nan=True)
    table = blocks.convergence
    assert list(table.columns) == [*BASE_COLUMNS, "test", "solver", "jacobians", "blocks", "recursive"]
    assert (table["blocks"] == 2).all() and (table["recursive"] == 4).all()  # (c, y) and log(q); a, b, s, t once
    assert table.attrs["blocks"] == [("ae",), ("be",), ("ce", "ye"), ("qe",), ("se",), ("te",)]
    for name in ("ae", "be", "se", "te"):  # solved once, exactly (to rounding)
        e = next(e for e in model.model_equations if e.name == name)
        from pymetrix.expr import evaluate

        frame = blocks.frame
        for pos in range(4, 16):
            def resolve(n, lag, pos=pos):
                return float(frame[n].iloc[pos + lag])
            assert abs(evaluate(e.residual, resolve)) < 1e-13


def test_the_blocks_are_model_blocks():
    """Without holds the blocks are the strongly connected components Model.blocks() finds (its own Tarjan, on the
    variables), in an order where each needs only those before it."""
    model, _ = _mixed()
    steps = plan(list(model.model_equations), list(model.endogenous))
    found = [frozenset(model.model_equations[i].dependent for i in s.equations) for s in steps]
    assert sorted(found, key=sorted) == sorted((frozenset(b.variables) for b in model.blocks()), key=sorted)
    position = {v: k for k, block in enumerate(found) for v in block}
    for k, block in enumerate(found):
        for v in block:
            e = next(e for e in model.model_equations if e.dependent == v)
            assert all(position[n.id] <= k for n in e.residual.refs() if n.lag == 0 and n.id in position)


def test_blocks_with_holds_and_partial_fixes():
    """A hold pairs its equation with the free variable (found by matching), so that equation becomes a block solved by
    the solver; partial fixes give each period its own blocks."""
    model, data = _mixed()
    span = ("2001Q1", "2003Q4")
    held = Scenario(hold={"y": {"2001Q2": 1.5, "2001Q3": 1.6}}, free=["g"], fix={"q": {"2002Q1": 1.1}}, partial=True)
    joint = model.solve(data, span, held, solver=Newton(tol=1e-13))
    blocks = model.solve(data, span, held, ordering="blocks")
    np.testing.assert_allclose(blocks.frame.to_numpy(float), joint.frame.to_numpy(float), atol=1e-10, equal_nan=True)
    assert blocks["y"].loc["2001Q2"] == pytest.approx(1.5) and isinstance(blocks.convergence.attrs["blocks"], dict)


def _chain(n: int = 200) -> tuple[Model, Dataset]:
    """A mostly recursive model: a small nonlinear simultaneous block (a, b) and a chain of ``n`` identities."""
    eqs = [Equation("a = 0.5*b + 0.2*log(x) + 0.3*a[-1]", name="ea"),
           Equation("b = 0.4*a + 0.1*exp(0.5*a) + z", name="eb"),
           Equation("v0 = b + 0.5*v0[-1]", name="e0")]
    eqs += [Equation(f"v{k} = 0.5*v{k - 1} + 0.1*a + 0.2*v{k}[-1]", name=f"e{k}") for k in range(1, n)]
    names = ["a", "b"] + [f"v{k}" for k in range(n)]
    index = pd.period_range("2000Q1", periods=12, freq="Q", name="time")
    frame = pd.DataFrame(1.0, index=index, columns=[*names, "x", "z"])
    frame["x"] = 1.5
    frame.loc["2000Q3":, names] = np.nan
    return Model(eqs), Dataset.from_frame(frame)


def _count_equations(monkeypatch):
    """Equations evaluated, counted at the period system: all of its equations for each residuals or Jacobian call,
    one for each equation evaluated on its own."""
    count = {"n": 0}
    cls = model_module._PeriodSystem
    residuals, jacobian, own = cls.residuals, cls.jacobian, cls.own

    def counted_residuals(self, x):
        count["n"] += len(self.eqs)
        return residuals(self, x)

    def counted_jacobian(self, x):
        count["n"] += len(self.eqs)
        return jacobian(self, x)

    def counted_own(self, i, x):
        count["n"] += 1
        return own(self, i, x)

    monkeypatch.setattr(cls, "residuals", counted_residuals)
    monkeypatch.setattr(cls, "jacobian", counted_jacobian)
    monkeypatch.setattr(cls, "own", counted_own)
    return count


@pytest.mark.parametrize("solver", [Newton(), GaussSeidel()], ids=["newton", "gauss_seidel"])
def test_blocks_evaluate_far_fewer_equations_on_a_mostly_recursive_model(solver, monkeypatch):
    model, data = _chain()
    span = ("2000Q3", "2002Q4")
    count = _count_equations(monkeypatch)
    joint = model.solve(data, span, solver=solver)
    by_joint = count["n"]
    count["n"] = 0
    blocks = model.solve(data, span, solver=solver, ordering="blocks")
    by_blocks = count["n"]
    np.testing.assert_allclose(blocks.frame.to_numpy(float), joint.frame.to_numpy(float), atol=1e-9, equal_nan=True)
    periods = len(blocks.convergence)
    assert (blocks.convergence["blocks"] == 1).all() and (blocks.convergence["recursive"] == 200).all()
    # once each: 200 a period, plus the two-equation block's iterations
    assert by_blocks <= periods * (200 + 2 * 2 * 60)
    assert by_blocks * 3 < by_joint, (by_blocks, by_joint)


def test_ordering_is_refused_where_it_does_not_apply():
    model, data = _mixed()
    with pytest.raises(SpecificationError, match="ordering is"):
        model.solve(data, ("2001Q1", "2001Q4"), ordering="recursive")
    with pytest.raises(SpecificationError, match="period by period"):
        model.solve(data, ("2001Q1", "2001Q4"), solver=PerfectForesight(), ordering="blocks")
    with pytest.raises(SpecificationError, match="period by period"):
        model.solve(data, ("2001Q1", "2001Q4"), solver=FirstOrder(), ordering="blocks")


def test_a_failing_block_is_recorded():
    model, data = _mixed()
    with pytest.raises(ModelError, match="did not converge") as info:
        model.solve(data, ("2001Q1", "2001Q4"), solver=Newton(max_iter=1), ordering="blocks")
    table = info.value.convergence
    assert not table["converged"].iloc[-1] and table["equation"].iloc[-1] == "qe"  # (c, y) is linear: one step


# the Solve dialog


def test_the_solve_dialog_takes_the_options():
    from pymetrix.commands import Session, SolveModel

    s = Session()
    model, data = _model(), _data()
    s.objects["m"], s.objects["data"] = model, data
    s.run(SolveModel("m", *SPAN, solver_settings="convergence='step', line_search=True, refresh='period'",
                     ordering="blocks", fallback=True, name="solved"))
    solver = Fallback(Newton(convergence="step", line_search=True, refresh="period"), GaussSeidel(damping=0.5, max_iter=400),
                      Newton(line_search=True))
    direct = model.solve(data, SPAN, solver=solver, ordering="blocks")
    np.testing.assert_array_equal(s["solved"].frame.to_numpy(float), direct.frame.to_numpy(float))
    pd.testing.assert_frame_equal(s["solved"].convergence, direct.convergence)
    script = s.script()
    assert "ordering='blocks'" in script and "Fallback([" in script
    ns = {**dict(s.objects)}
    exec(script, ns)
    np.testing.assert_array_equal(ns["solved"].frame.to_numpy(float), direct.frame.to_numpy(float))
    s.run(SolveModel("m", *SPAN, name="plain"))  # the defaults: today's call
    assert "ordering" not in s.script().split("plain =")[1] and "Fallback" not in s.script().split("plain =")[1]
    assert SolveModel("m", *SPAN, solver="perfect_foresight", fallback=True).validate(s)
    assert SolveModel("m", *SPAN, solver="perfect_foresight", ordering="blocks").validate(s)
