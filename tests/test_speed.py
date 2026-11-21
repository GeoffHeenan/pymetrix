"""Compiled equations: the same results as the tree interpreter, and fast enough for a FINEX-sized model."""

import re
import time

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, FirstOrder, Model, PerfectForesight, Scenario, templates
from pymetrix.expr import parse_expression
from pymetrix.expr.compile import compilable, compile_exprs
from pymetrix.expr.evaluate import evaluate
from pymetrix.metadata import Role
from pymetrix.models import fast


@pytest.fixture
def interpreter(monkeypatch):
    """Run with the tree interpreter only (the reference)."""
    monkeypatch.setattr(fast, "ENABLED", False)


def big_model(blocks: int = 90) -> Model:
    """Gap, trend-growth and policy-rate equations in blocks of three, chained: 270 equations by default."""
    eqs, params, roles = [], {}, {}
    for i in range(blocks):
        prev = f"x{i - 1}" if i else "x0"
        eqs.append(Equation(f"x{i} ~ a{i}*x{i}[-1] + b{i}*x{i}[+1] + 0.1*{prev} - 0.2*(exp(0.5*r{i}) - 1) + ex{i}", params=[f"a{i}", f"b{i}"]))
        eqs.append(Equation(f"g{i} ~ rho{i}*g{i}[-1] + (1-rho{i})*2 + 0.05*x{i} + eg{i}", params=[f"rho{i}"]))
        eqs.append(Equation(f"r{i} ~ max(c{i}*r{i}[-1] + (1-c{i})*(1.5*x{i}[+1] + 0.5*x{i}) + er{i}, -5)", params=[f"c{i}"]))
        params.update({f"a{i}": 0.5, f"b{i}": 0.3, f"rho{i}": 0.8, f"c{i}": 0.7})
        roles.update({f"ex{i}": Role.SHOCK, f"eg{i}": Role.SHOCK, f"er{i}": Role.SHOCK})
    return Model(eqs, params=params, roles=roles)


def _flat(model: Model, years: int) -> Dataset:
    idx = pd.period_range("1990", periods=years + 5, freq="Y", name="time")
    values = {**{v: 0.0 for v in model.exogenous}, **{v: (2.0 if v.startswith("g") else 0.0) for v in model.endogenous}}
    return Dataset.from_frame(pd.DataFrame({k: np.full(len(idx), v) for k, v in values.items()}, index=idx))


def test_compiled_expressions_agree_with_the_interpreter():
    texts = ["b*x[-1] + log(y) - max(x, 0)^2 / z", "exp(-x) + step(y - 1) + abs(x) + sign(x) + sqrt(y)", "min(x, y, 0) - 3"]
    exprs = [parse_expression(t, params=["b"]) for t in texts]
    f = compile_exprs(exprs, ["b"])
    rng = np.random.default_rng(0)
    values = {("x", -1): rng.normal(size=50), ("x", 0): rng.normal(size=50), ("y", 0): rng.uniform(0.1, 2, size=50),
              ("z", 0): np.r_[0.0, rng.uniform(0.5, 2, size=49)]}
    for e, got in zip(exprs, f([values[r] for r in f.refs], [0.5])):
        np.testing.assert_array_equal(np.broadcast_to(got, (50,)), evaluate(e, lambda n, lag: values[(n, lag)], {"b": 0.5}))
    assert not re.search(r"[xyzb]", f.source)  # slots v[k] and parameters p[j] only: no names in the code
    assert not compilable(parse_expression("mean(y, country)")) and compilable(parse_expression("max(a, b)"))


@pytest.mark.parametrize("make", [templates.new_keynesian, templates.rbc, templates.gap_model])
def test_solutions_are_the_same_either_way(make, monkeypatch):
    template = make()
    compiled = template.model.linearize(template.steady_state()).solve()
    monkeypatch.setattr(fast, "ENABLED", False)
    interpreted = template.model.linearize(template.steady_state()).solve()
    np.testing.assert_allclose(compiled.P, interpreted.P, atol=1e-12)
    np.testing.assert_allclose(compiled.Q, interpreted.Q, atol=1e-12)


def test_perfect_foresight_is_the_same_either_way(monkeypatch):
    model = big_model(4)
    data = _flat(model, 40)
    scenario = Scenario(set={"ex0": {"1996": 1.0}, "er1": {"1997": -8.0}})  # the second one hits the max(..., -5)
    compiled = model.solve(data, ("1995", "2034"), scenario, solver=PerfectForesight(terminal="steady_state"))
    monkeypatch.setattr(fast, "ENABLED", False)
    interpreted = model.solve(data, ("1995", "2034"), scenario, solver=PerfectForesight(terminal="steady_state"))
    np.testing.assert_allclose(compiled.frame.to_numpy(float), interpreted.frame.to_numpy(float), atol=1e-10)
    assert (compiled["r1"] >= -5 - 1e-12).all() and compiled["r1"].min() == pytest.approx(-5)


def test_a_finex_sized_model_is_fast():
    model = big_model(90)  # 270 equations, annual
    data = _flat(model, 60)
    scenario = Scenario(set={"ex0": {"1996": 1.0}})
    span = ("1995", "2054")
    model.solve(data, span, scenario, solver=PerfectForesight(terminal="steady_state"))  # compiles, takes derivatives
    start = time.perf_counter()
    solved = model.solve(data, span, scenario, solver=PerfectForesight(terminal="steady_state"))
    simulate = time.perf_counter() - start
    start = time.perf_counter()
    rebuilt = Model(model.equations, params=model.parameters.with_values({"a0": 0.55}), roles=model.roles)  # a step of an estimation
    rebuilt.linearize().solve()
    estimation_step = time.perf_counter() - start
    assert solved.iterations <= 6
    assert simulate < 3.0, f"60 years of perfect foresight took {simulate:.2f}s (0.13s when written)"
    assert estimation_step < 20.0, f"rebuild, steady state, linearise and solve took {estimation_step:.2f}s (0.6s when written)"
    first = model.solve(data, span, scenario, solver=FirstOrder())
    assert np.isfinite(first.frame.to_numpy(float)).all()


def test_compiling_pays_on_any_machine(monkeypatch):
    """Machine-independent: on the same machine, compiled linearisation and simulation beat the interpreter by far."""
    model = big_model(60)  # 180 equations: a compiled linearisation of some milliseconds, not one, so that a busy CI
    data = _flat(model, 30)  # machine (tests run in parallel) cannot swamp it; the ratio here is about 17 for it
    span, scenario = ("1995", "2024"), Scenario(set={"ex0": {"1996": 1.0}})

    def timed() -> tuple[float, float]:
        """The fastest of ten runs of each: the least disturbed by whatever else the machine is doing."""
        ss = model.steady_state()
        model.linearize(ss)
        model.solve(data, span, scenario, solver=PerfectForesight(terminal="steady_state"))  # warm
        linear, stacked = [], []
        for _ in range(10):
            start = time.perf_counter()
            model.linearize(ss)
            middle = time.perf_counter()
            model.solve(data, span, scenario, solver=PerfectForesight(terminal="steady_state"))
            linear.append(middle - start)
            stacked.append(time.perf_counter() - middle)
        return min(linear), min(stacked)

    linearise, simulate = timed()
    monkeypatch.setattr(fast, "ENABLED", False)
    linearise_slow, simulate_slow = timed()
    assert linearise_slow > 3 * linearise, (linearise_slow, linearise)  # about 6.6 times when written
    assert simulate_slow > 3 * simulate, (simulate_slow, simulate)  # about 8 times when written
