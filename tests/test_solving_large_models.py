"""The how-to page on solving large backward-looking models (docs/solving-large-models.md) runs as written, its
recommended configuration is the benchmark's, and the benchmark (benchmarks/solve_options.py) runs at a small size.

Independent checks: every configuration gives the default's solution; the counts the benchmark reports against what
Newton must do (one Jacobian of every equation an iteration, jointly) and what the blocks must do (each recursive
equation evaluated once a period, Jacobians only of the simultaneous blocks)."""

from __future__ import annotations

import importlib.util
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "solving-large-models.md"


def _benchmark():
    spec = importlib.util.spec_from_file_location("solve_options_benchmark", ROOT / "benchmarks" / "solve_options.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page() -> dict:
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) >= 8
    scope: dict = {"__name__": "solving_large_models"}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f"{DOC.name}, block {i}", "exec"), scope)
    return scope


def test_the_page_runs_as_written(page):
    base, solution = page["base"], page["solution"]
    assert len(page["model"].equations) == 100 and len(base.convergence) == 40
    for name in ("blocks", "kept", "step"):  # each option gives the default's solution
        np.testing.assert_allclose(page[name].solved().frame.to_numpy(float), base.solved().frame.to_numpy(float), rtol=1e-9)
    blocks = page["blocks"].convergence
    assert (blocks["blocks"] == 2).all() and (blocks["recursive"] == 95).all()  # what the page says
    assert blocks.attrs["blocks"][:3] == [("cons", "invest", "output"), ("prices", "slack"), ("link0",)]
    assert (page["kept"].convergence["jacobians"] == 1).all()
    assert list(page["step"].convergence["test"].unique()) == ["step"]
    table = page["table"]
    assert table.columns.tolist() == ["iterations", "residual", "equation", "converged", "test", "solver", "jacobians",
                                      "blocks", "recursive"]
    assert len(table.attrs["seconds"]) == 40 and list(base.convergence.columns) == ["iterations", "residual", "equation",
                                                                                     "converged"]


def test_the_recommended_configuration_is_the_benchmarks(page):
    bench = _benchmark()
    assert page["recommended"] == bench.RECOMMENDED["solver"] and bench.RECOMMENDED["ordering"] == "blocks"


def test_the_pages_options_on_the_generated_model():
    """The page's options and its recommendation on the benchmark's model (with its own simultaneous blocks), small."""
    bench = _benchmark()
    model, data, span = bench.generate(40, 6)
    base = model.solve(data, span).solved().frame.to_numpy(float)
    for _, options in bench.configurations():
        solved = model.solve(data, span, **options)
        np.testing.assert_allclose(solved.solved().frame.to_numpy(float), base, rtol=1e-8)
        assert len(solved.convergence.attrs["seconds"]) == 6


def test_the_benchmark_counts():
    bench = _benchmark()
    model, data, span = bench.generate(30, 4)
    rows = {r["configuration"]: r for r in bench.measure(model, data, span, repeats=1)}
    default, blocks, kept, best = rows.values()
    n, periods = 30, 4
    assert default["jacobians"] == default["iterations"] and default["rows"] == n * default["jacobians"]  # Newton, jointly
    assert blocks["equations"] >= periods * (n - 5) and blocks["rows"] <= 3 * blocks["jacobians"]  # blocks of 3 and 2
    assert blocks["equations"] < default["equations"] and kept["jacobians"] < default["jacobians"]
    assert kept["jacobians"] == periods  # one a period: this model converges fast enough to keep it
    for r in rows.values():
        assert r["difference"] < 1e-8 and r["seconds"] >= r["solving"] >= 0
    with bench.counting() as count:  # the counting is undone afterwards
        model.solve(data, span)
    assert count["jacobians"] == default["jacobians"]
    model.solve(data, span)
    assert count["jacobians"] == default["jacobians"]


def test_the_benchmark_runs_small():
    out = io.StringIO()
    with redirect_stdout(out):
        _benchmark().main(["--equations", "12", "--quarters", "3", "--repeats", "1"])
    text = out.getvalue()
    assert "Generated model: 12 equations, 2001Q1 to 2001Q3" in text
    lines = [line for line in text.splitlines() if " | " in line]
    assert len(lines) == 5 and lines[0].startswith("configuration") and "recommended" in lines[-1]


def test_the_benchmark_times_a_model_file(tmp_path):
    (tmp_path / "small.model").write_text("""
!transition_variables
    c, y, v
!exogenous_variables
    g
!transition_equations
    c = 0.5*y + 0.3*c{-1};
    y = c + g;
    v = 0.5*y + 0.2*v{-1};
""", encoding="utf-8")
    index = pd.period_range("2000Q1", periods=8, freq="Q")
    frame = pd.DataFrame({"time": index.astype(str), "c": 1.0, "y": 2.0, "v": 1.0, "g": 1.0})
    frame.loc[2:, ["c", "y", "v"]] = np.nan
    frame.to_csv(tmp_path / "small.csv", index=False)
    out = io.StringIO()
    with redirect_stdout(out):
        _benchmark().main(["--model", str(tmp_path / "small.model"), "--data", str(tmp_path / "small.csv"), "--time", "time",
                           "--freq", "Q", "--span", "2000Q3", "2001Q4", "--repeats", "1"])
    text = out.getvalue()
    assert "small.model: 3 equations, 2000Q3 to 2001Q4" in text and len([x for x in text.splitlines() if " | " in x]) == 5
