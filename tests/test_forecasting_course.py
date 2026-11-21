"""The forecasting course pack: each lab runs from start to finish, from any folder, and reaches its conclusions.

The labs simulate their own data with fixed seeds, so these are the lessons a student should draw from them.
``python tests/test_forecasting_course.py`` rebuilds 00_start_here.ipynb from the README and each lab's list of the
pymetrix it uses."""

import json
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).parents[1] / "examples" / "forecasting"
LABS = sorted(PACK.glob("F*.ipynb"))
START = PACK / "00_start_here.ipynb"


def start_here() -> str:
    """00_start_here.ipynb as it should be: the README, a cell per section, with its front matter as metadata (Jupyter
    opens notebooks formatted, Markdown files as text)."""
    from pymetrix.help import _front_matter

    meta, body = _front_matter((PACK / "README.md").read_text(encoding="utf-8"))
    meta["title"] = "Start here: " + meta["title"].lower().replace("forecasting", "the forecasting", 1)
    sections, current = [], []
    for line in body.strip("\n").split("\n"):
        if line.startswith("## ") and current:
            sections.append(current)
            current = []
        current.append(line)
    sections.append(current)
    cells = []
    for lines in sections:
        while lines and not lines[-1].strip():
            lines.pop()
        source = [line + "\n" for line in lines]
        source[-1] = source[-1].rstrip("\n")
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "pymetrix": meta},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def _run(name: str, folder: Path, monkeypatch) -> dict:
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(folder)
    ns: dict = {"__name__": "__main__"}
    notebook = PACK / name
    for i, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                exec(compile(source, f"{notebook.name}[{i}]", "exec"), ns)  # noqa: S102 - our own example
            except Exception as exc:  # pragma: no cover - shows which cell failed
                raise AssertionError(f"{notebook.name}, cell {i} failed: {exc}\n{source}") from exc
    return ns


def test_the_pack_passes_the_course_check():
    from pymetrix import help as hp
    from pymetrix.testing import check_course

    course = check_course(PACK)
    assert course.name == "Forecasting course pack (pymetrix)"
    assert sum(n.kind == "notebook" for n in course.notes) == 7 == len(LABS) + 1
    assert all(not n.text.startswith("---") for n in course.notes if n.kind == "notebook")  # front matter in the metadata
    hp.unload_course(PACK)


def test_the_readme_lists_every_lab():
    readme = (PACK / "README.md").read_text(encoding="utf-8")
    for notebook in LABS:
        assert f"]({notebook.name})" in readme


def test_the_start_here_notebook_is_the_readme():
    assert START.read_text(encoding="utf-8") == start_here(), "README.md changed: run python tests/test_forecasting_course.py"


@pytest.mark.parametrize("path", LABS, ids=lambda p: p.name)
def test_each_lab_lists_the_pymetrix_it_uses(path):
    from pymetrix.testing import check_api_reference

    check_api_reference(path)  # current with the docstrings, and nothing the lab imports is missing


def test_the_reference_traces_calls_through_renamed_arguments(tmp_path):
    # @renamed_arguments wraps above @api with a closure of the same name; the tracer must name what it wraps,
    # for functions and for classes made through a renaming __init__
    from pymetrix.testing import _run_traced, check_api_reference, write_api_reference

    lab = tmp_path / "lab.ipynb"
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["import pymetrix.stats as stats\n", "from pymetrix import Scenario\n",
                       "Scenario()\n", "stats.available_tests(1)\n", "stats.run_test\n"]}
    lab.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    used = set(_run_traced(lab))
    assert {"pymetrix.models.scenario:Scenario", "pymetrix.stats.base:available_tests"} <= used
    assert write_api_reference(lab)
    check_api_reference(lab)  # each has a docstring to describe it


def test_the_reference_helpers_notice_when_a_lab_changes(tmp_path):
    from pymetrix.testing import REFERENCE_TAG, check_api_reference, write_api_reference

    lab = tmp_path / "lab.ipynb"
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["from pymetrix import Dataset\n", "data = Dataset.example('sample_macro').with_series(ly='log(y)')"]}
    lab.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    with pytest.raises(AssertionError, match="no reference cell"):
        check_api_reference(lab)
    assert write_api_reference(lab) and not write_api_reference(lab)  # written once, then current
    check_api_reference(lab)
    nb = json.loads(lab.read_text(encoding="utf-8"))
    reference = nb["cells"][-1]
    assert REFERENCE_TAG in reference["metadata"]["tags"]
    assert set(reference["metadata"]["pymetrix"]["reference"]) == {"pymetrix.data.dataset:Dataset.example", "pymetrix.data.dataset:Dataset.with_series"}
    assert "`Dataset.example(name)`" in "".join(reference["source"])
    nb["cells"].insert(0, {**cell, "source": ["from pymetrix import backtest"]})  # a new import, not yet listed
    lab.write_text(json.dumps(nb), encoding="utf-8")
    with pytest.raises(AssertionError, match="backtest"):
        check_api_reference(lab)
    reference["metadata"]["pymetrix"]["reference"].append("pymetrix.forecasting:Backtest.gone")  # renamed since
    lab.write_text(json.dumps(nb), encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer has"):
        check_api_reference(lab)


@pytest.mark.parametrize("path", [*LABS, START], ids=lambda p: p.name)
def test_notebooks_are_saved_without_outputs(path):
    """Students run the labs themselves; a notebook saved after a run would also commit its outputs."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(not c["outputs"] and c["execution_count"] is None for c in code), f"{path.name}: clear its outputs before committing"


def test_lab_f1_losses_and_efficiency(tmp_path, monkeypatch):
    ns = _run("F1_loss_and_efficiency.ipynb", tmp_path, monkeypatch)
    table = ns["table"]
    assert table.iloc[0, 0] < table.iloc[1, 0] and table.iloc[1, 1] < table.iloc[0, 1]  # each forecast wins under its own loss
    assert ns["under_squared"].p_value < 0.05 and ns["under_linlin"].p_value < 0.05
    assert abs(ns["best_linlin"] - 0.84) < 0.15 and abs(ns["best_linex"] - 0.5) < 0.15
    assert ns["mz_ar"].p_value > 0.1 and ns["mz_naive"].p_value < 0.01 and ns["mz_naive"].details["slope"] < 1
    assert ns["nw"].details["slope se"] > ns["white"].details["slope se"]  # overlapping errors


def test_lab_f2_comparing_forecasts(tmp_path, monkeypatch):
    ns = _run("F2_comparing_forecasts.ipynb", tmp_path, monkeypatch)
    assert ns["dm"].p_value > 0.1 and ns["cw"].p_value < 0.05  # the nested-model lesson
    assert ns["size"]["Diebold-Mariano (one-sided)"] < ns["size"]["Clark-West"] <= 0.15
    if ns["spa_test"] is not None:
        assert ns["spa_test"].p_value > 0.1  # no model beats the mean once the search is allowed for


def test_lab_f3_combination(tmp_path, monkeypatch):
    ns = _run("F3_combination.ipynb", tmp_path, monkeypatch)
    rmse = {name: bt.scores(["y"])["RMSE"].iloc[0] for name, bt in ns["everything"].items()}
    assert rmse["equal"] < min(rmse["x1"], rmse["x2"], rmse["x3"])  # the average beats every member
    assert rmse["regression"] > rmse["equal"]  # the combination puzzle
    inclusion = dict(zip(ns["names"], ns["bma"].inclusion))
    assert inclusion["x1"] > 0.9 and inclusion["x4"] < 0.2


def test_lab_f4_densities(tmp_path, monkeypatch):
    ns = _run("F4_densities.ipynb", tmp_path, monkeypatch)
    assert ns["expanding"].interval_test("y").p_value < 0.01 and ns["rolling"].interval_test("y").p_value > 0.05
    assert ns["expanding"].pit_test("y").p_value < 0.01 and ns["rolling"].pit_test("y").p_value > 0.05
    assert ns["outside"](ns["before"]) < 0.15 < 0.25 < ns["outside"](ns["after"])
    assert ns["pt"].p_value < 0.05


def test_lab_f5_many_predictors(tmp_path, monkeypatch):
    ns = _run("F5_many_predictors.ipynb", tmp_path, monkeypatch)
    rmse = ns["accuracy"]["RMSE"]
    assert rmse["diffusion index"] < rmse["bagging"] < rmse["kitchen sink"] < rmse["mean"]
    assert min(ns["r2"]) > 0.9
    assert "IC_p2 2" in " ".join(ns["pca"].factor_criteria().notes)


def test_lab_f6_breaks_and_nonlinearity(tmp_path, monkeypatch):
    ns = _run("F6_breaks_and_nonlinearity.ipynb", tmp_path, monkeypatch)
    after = ns["periods"]["first five years after (2005-2009)"]
    assert ns["rmse"](ns["rolling"], *after) < ns["rmse"](ns["expanding"], *after)
    before = ns["periods"]["before the break (1995-2004)"]
    assert ns["rmse"](ns["expanding"], *before) < ns["rmse"](ns["rolling"], *before)
    assert ns["hansen"].p_value < 0.1 and ns["lm"].p_value < 0.1  # nonlinear in sample...
    assert ns["bt_setar"].compare(ns["bt_ar"], "y", 1).p_value > 0.05  # ...but no better out of sample
    assert np.isclose(ns["threshold"], ns["setar"].threshold)


if __name__ == "__main__":  # rebuild the generated parts of the pack
    from pymetrix.testing import write_api_reference

    START.write_text(start_here(), encoding="utf-8", newline="\n")
    for lab in LABS:
        print(lab.name, "updated" if write_api_reference(lab) else "unchanged")
