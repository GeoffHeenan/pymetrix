"""The example course's notebooks run from start to finish, from any folder."""

import json
from pathlib import Path

import pytest

COURSE = Path(__file__).parents[1] / "examples" / "course"
SOLUTIONS = Path(__file__).parents[1] / "examples" / "solutions"


@pytest.fixture(autouse=True)
def _registries_as_found():
    """Lab 4 registers a plugin; put the registries back so no notebook changes other tests."""
    from pymetrix.plugins import GROUPS, registry

    saved = {g: dict(registry(g)._entries) for g in GROUPS}
    yield
    for g, entries in saved.items():
        registry(g)._entries.clear()
        registry(g)._entries.update(entries)


def _run(notebook: Path, folder: Path, monkeypatch) -> dict:
    """Run every code cell, in a fresh namespace, from `folder`; return the namespace."""
    monkeypatch.chdir(folder)  # not the notebook's own folder: the data must not depend on it
    ns: dict = {"__name__": "__main__"}
    for i, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                exec(compile(source, f"{notebook.name}[{i}]", "exec"), ns)  # noqa: S102 - our own example
            except Exception as exc:  # pragma: no cover - shows which cell failed
                raise AssertionError(f"{notebook.name}, cell {i} failed: {exc}\n{source}") from exc
    return ns


@pytest.fixture(scope="module")
def _ran(tmp_path_factory):
    """Each notebook runs once per module, each in its own folder; the tests below read what that run left.

    Lab 3 takes some 16 s, and three tests look at it. A run's namespace and files are only read afterwards."""
    done: dict = {}

    def run(notebook: Path, monkeypatch) -> dict:
        if notebook not in done:
            done[notebook] = _run(notebook, tmp_path_factory.mktemp(notebook.stem), monkeypatch)
        return done[notebook]

    return run


@pytest.mark.parametrize("path", sorted([*COURSE.glob("*.ipynb"), *SOLUTIONS.glob("*.ipynb")]), ids=lambda p: p.name)
def test_notebook_runs(path, _ran, monkeypatch):
    pytest.importorskip("matplotlib")  # the notebooks draw figures, and Lab 3's report saves them (and an Excel file)
    pytest.importorskip("openpyxl")
    _ran(path, monkeypatch)


def test_the_lab_reaches_its_conclusions(_ran, monkeypatch):
    pytest.importorskip("statsmodels")
    ns = _ran(SOLUTIONS / "02_lab_solution.ipynb", monkeypatch)
    assert ns["same"]  # the reopened project reproduces the solution
    m = ns["multiplier"]
    assert 1 < m.iloc[0] < m.iloc[-1]  # a multiplier above one that builds over time
    assert ns["cons"].params["y"] > 0


def test_the_forecast_lab_reaches_its_conclusions(_ran, monkeypatch):
    """Lab 3 is also the demo's smoke test: round, judgement, revisions, backtest, report, project."""
    pytest.importorskip("statsmodels")
    pytest.importorskip("openpyxl")
    ns = _ran(COURSE / "03_forecast_round.ipynb", monkeypatch)
    assert ns["same"]  # the reopened project rebuilds the adjusted round
    december = ns["december"]
    assert len(december.adjustments) == 2 and december.data_end == "2024Q4"
    assert december.path("ly")["2025Q1"] == pytest.approx(december.base["ly"]["2025Q1"] - 0.004)
    assert {p.suffix for p in ns["saved"]} == {".html", ".xlsx"} and all(p.stat().st_size > 0 for p in ns["saved"])
    assert "Where the numbers come from" in ns["saved"][0].read_text(encoding="utf-8")  # the provenance appendix
    assert 0 <= ns["dm"].p_value <= 1
    assert "ForecastRound.from_forecast('2024Q4 round'" in ns["s"].script()


def test_the_student_lab_leaves_the_work_to_students(_ran, monkeypatch):
    """The exercise notebook runs, but estimates nothing: that is for the student (the solution does it)."""
    pytest.importorskip("statsmodels")
    ns = _ran(COURSE / "02_lab.ipynb", monkeypatch)
    assert "data" in ns and not {"cons", "inv", "model", "multiplier"} & set(ns)


def test_the_forecast_lab_includes_a_structural_scenario(_ran, monkeypatch):
    pytest.importorskip("statsmodels")
    pytest.importorskip("openpyxl")
    ns = _ran(COURSE / "03_forecast_round.ipynb", monkeypatch)
    revision = ns["scenario"].to_frame()["y revision"].astype(float)
    assert 2 < revision.iloc[0] / 5 < revision.iloc[-1] / 5  # an impact multiplier above two, building over time
    assert "A fiscal scenario" in ns["saved"][0].read_text(encoding="utf-8")


def test_the_engine_lab_checks_itself(_ran, monkeypatch):
    """Lab 4: cross-checks pass, the plugin passes conformance, and dialog code and project reproduce the fit."""
    pytest.importorskip("statsmodels")
    ns = _ran(COURSE / "04_engine.ipynb", monkeypatch)
    assert ns["same_code"] and ns["same_project"]
    assert ns["fits"]["IV"].params["y"] < ns["fits"]["OLS"].params["y"]  # simultaneity bias removed
    assert ns["ridge_fits"][1000.0] < ns["ridge_fits"][0.0]  # shrinkage
