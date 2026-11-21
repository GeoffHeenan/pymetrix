"""The seconds each period's solve takes, in ``Solution.convergence.attrs["seconds"]``: one a period, not negative,
outside the table's columns (which, and the solution, are as without them), and kept when the solution is saved.

Independent checks: a solver that sleeps a known time in one period (``time.sleep``) shows it in that period's
seconds and not in the others'; the seconds add up to no more than the wall time around the whole solve."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Fallback, GaussSeidel, Model, ModelError, Newton

SPAN = ("2001Q1", "2002Q4")  # eight quarters
BASE_COLUMNS = ["iterations", "residual", "equation", "converged"]


def _model() -> Model:
    return Model([Equation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", name="cons"),  # a simultaneous pair (c, y)
                  Equation("y = c + g", name="gdp"),
                  Equation("v = 0.5*y + 0.2*v[-1]", name="link")])  # recursive


def _data() -> Dataset:
    index = pd.period_range("2000Q1", periods=12, freq="Q", name="time")
    frame = pd.DataFrame({"c": 1.0, "y": 2.0, "v": 1.0, "g": 1.0, "w": 1.5}, index=index)
    frame.loc["2001Q1":, ["c", "y", "v"]] = np.nan
    return Dataset.from_frame(frame)


def _seconds(solution) -> list[float]:
    return solution.convergence.attrs["seconds"]


@pytest.mark.parametrize("options", [{}, {"solver": GaussSeidel()}, {"solver": Newton(refresh="period", convergence="step")},
                                     {"ordering": "blocks"}, {"solver": Fallback(GaussSeidel(), Newton()), "ordering": "blocks"}],
                         ids=["default", "gauss_seidel", "newton_options", "blocks", "fallback_blocks"])
def test_one_non_negative_value_a_period(options):
    solution = _model().solve(_data(), SPAN, **options)
    seconds = _seconds(solution)
    assert isinstance(seconds, list) and len(seconds) == len(solution.convergence) == 8
    assert all(isinstance(s, float) and s >= 0 for s in seconds)


def test_the_columns_and_the_solution_are_as_without_them():
    model, data = _model(), _data()
    start = time.perf_counter()
    solution = model.solve(data, SPAN)
    wall = time.perf_counter() - start
    table = solution.convergence
    assert list(table.columns) == BASE_COLUMNS and "seconds" not in table.columns
    assert 0 <= sum(_seconds(solution)) <= wall
    again = model.solve(data, SPAN)  # the same numbers every time: only the seconds differ
    pd.testing.assert_frame_equal(again.convergence, table)
    np.testing.assert_array_equal(again.frame.to_numpy(float), solution.frame.to_numpy(float))
    with_options = model.solve(data, SPAN, solver=Newton(line_search=True))
    assert list(with_options.convergence.columns) == [*BASE_COLUMNS, "test", "solver", "jacobians"]
    assert len(_seconds(with_options)) == len(with_options.convergence)


class _Sleepy:
    """Newton, but sleeping ``pause`` seconds in the period it solves ``which``-th."""

    def __init__(self, which: int, pause: float) -> None:
        self.which, self.pause, self.calls = which, pause, 0

    def solve_period(self, system):
        if self.calls == self.which:
            time.sleep(self.pause)
        self.calls += 1
        return Newton().solve_period(system)


def test_the_seconds_are_each_periods_own():
    solution = _model().solve(_data(), SPAN, solver=_Sleepy(3, 0.2))
    seconds = _seconds(solution)
    assert seconds[3] >= 0.2
    assert max(seconds[:3] + seconds[4:]) < 0.2
    assert int(np.argmax(seconds)) == 3


def test_a_failed_solve_carries_the_seconds_of_the_periods_it_reached():
    with pytest.raises(ModelError) as info:
        _model().solve(_data(), SPAN, solver=Newton(max_iter=0))
    table = info.value.convergence
    assert len(table) == 1 and not table["converged"].any()
    assert len(table.attrs["seconds"]) == 1 and table.attrs["seconds"][0] >= 0


def test_kept_when_the_solution_is_saved(tmp_path):
    from pymetrix import load_result, save_result

    solution = _model().solve(_data(), SPAN, ordering="blocks")
    again = pickle.loads(pickle.dumps(solution))  # how a project keeps a result it does not compute again
    assert again.convergence.attrs["seconds"] == _seconds(solution)
    assert again.convergence.attrs["blocks"] == solution.convergence.attrs["blocks"]
    pd.testing.assert_frame_equal(again.convergence, solution.convergence)
    saved = load_result(save_result(solution, tmp_path / "solution"))  # a .pymxr keeps the table, not the seconds
    table = dict(saved.views)["Convergence"].to_frame()
    assert "seconds" not in [str(c).lower() for c in table.columns] and len(table) == 8


def test_a_project_saved_and_opened_has_them(tmp_path):
    from pymetrix.commands import Session, SolveModel, load_project, save_project
    from pymetrix.commands.builtin import DefineEquation

    s = Session()
    s.objects["data"] = _data()
    s.run(DefineEquation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", "", name="cons"))
    s.run(DefineEquation("y = c + g", "", name="gdp"))
    s.run(SolveModel("cons, gdp", *SPAN, name="sol"))
    opened = load_project(save_project(s, tmp_path / "seconds.pymx"))
    table = opened["sol"].convergence
    assert list(table.columns) == BASE_COLUMNS
    assert len(table.attrs["seconds"]) == len(table) == 8 and min(table.attrs["seconds"]) >= 0
    pd.testing.assert_frame_equal(table, s["sol"].convergence)
