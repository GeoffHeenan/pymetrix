"""A project with a deferred step reopens, whatever happened after the step was added.

A deferred step never ran, so what it uses is told from the command itself (its call's refs, the objects its fields
name): the steps that make those inputs are not dead ends, and ``compute()`` runs it against its inputs as they were at
its place in the history, even when a later step redefines them. A deferred step whose inputs are missing on opening
stays deferred with a note; opening a project never fails because of a pending step. Checked against the estimate run
directly on the model as first defined (and not as redefined), against projects saved by pymetrix 0.1.0a16 (which
marked the step a deferred step uses as dead: ``tests/fixtures/projects/deferred_dead_a16.pymx``, built with it), and
against Manual mode's own deferral of a slow step on opening."""

import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, ModelSource
from pymetrix.commands import CommandError, FitEquation, Session, load_project, save_project
from pymetrix.commands import project as project_module
from pymetrix.commands.builtin import EstimateModel
from pymetrix.commands.model_source import DefineModel
from pymetrix.errors import NotFoundError

sys.path.insert(0, str(Path(__file__).parent))
from test_model_source import NK  # noqa: E402

OTHER = NK.replace("phi_pi = 1.5", "phi_pi = 2.5")  # the model redefined: another policy rule, same names
OLD = Path(__file__).parent / "fixtures" / "projects" / "deferred_dead_a16.pymx"


def _data():
    nk = ModelSource(NK, name="m").build()
    sim = nk.solve().simulate(60, nk.shock_sd, seed=3)
    sim.index = pd.period_range("2000Q1", periods=60, freq="Q", name="time")
    return nk.measurement.apply(Dataset.from_frame(sim))


@pytest.fixture(scope="module")
def reference():
    """kappa estimated directly, on the model as first defined and as redefined."""
    data = _data()
    first = ModelSource(NK, name="m").build().estimate(data, estimate=["kappa"]).params.values
    other = ModelSource(OTHER, name="m").build().estimate(data, estimate=["kappa"]).params.values
    assert not np.allclose(first, other, rtol=1e-4)  # the two models give different estimates
    return {"first": first, "other": other}


def _session(again=NK):
    s = Session()
    s.objects["data"] = _data()
    s.run(DefineModel(NK, name="m"))
    s.add(EstimateModel("m", estimate="kappa", name="est"), deferred=True)
    s.run(DefineModel(again, name="m"))
    return s


def _entries(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("commands.json"))


@pytest.mark.parametrize("again", [NK, OTHER], ids=["same-text", "other-text"])
def test_a_step_a_deferred_step_uses_is_kept_and_the_project_reopens(tmp_path, reference, again):
    s = _session(again)
    assert s.history[1].reads == {"m": s.history[0].writes["m"], "data": s.history[1].reads["data"]}  # from its command
    assert s.needed() == [True, True, True] and s.dead_ends() == []
    path = save_project(s, tmp_path / "p.pymx")
    assert [(bool(e.get("dead")), bool(e.get("deferred"))) for e in _entries(path)] == [(False, False), (False, True), (False, False)]
    reopened = load_project(path, trust=False)
    assert set(reopened.pending()) == {"est"} and not reopened.history[1].note
    assert [r.outputs for r in reopened.history] == [("m",), ("est",), ("m",)]
    done = reopened.compute()
    assert [r.outputs for r in done] == [("est",)] and reopened.pending() == {}
    np.testing.assert_allclose(reopened["est"].params.values, reference["first"], rtol=1e-8)  # the model at its place
    assert reopened.producer("m") is reopened.history[2]  # the later model stays current, so est is out of date,
    assert list(reopened.stale()) == ["est"]  # as when it runs at once and m is redefined after it


def test_compute_in_the_session_uses_the_inputs_at_the_steps_place(reference):
    s = _session(OTHER)
    current = s["m"]
    s.compute()
    np.testing.assert_allclose(s["est"].params.values, reference["first"], rtol=1e-8)
    assert s["m"] is current and s.producer("m") is s.history[2]  # the later model stays current
    assert s.stale() == {"est": "m has changed since est was computed"}
    assert s.history[1].reads["m"] == s.history[0].writes["m"]  # it read the model as first defined


def test_a_computed_deferred_step_is_saved_and_reopened(tmp_path, reference):
    s = _session(OTHER)
    s.compute()
    path = save_project(s, tmp_path / "p.pymx")
    assert [bool(e.get("dead")) or bool(e.get("deferred")) for e in _entries(path)] == [False, False, False]
    reopened = load_project(path, trust=False)
    assert reopened.pending() == {}
    np.testing.assert_allclose(reopened["est"].params.values, reference["first"], rtol=1e-8)
    assert reopened.producer("m") is reopened.history[2]


def test_automatic_mode_computes_it_on_opening_against_the_first_model(tmp_path, reference):
    s = _session(OTHER)
    s.mode = "automatic"
    reopened = load_project(save_project(s, tmp_path / "p.pymx"), trust=False)
    assert reopened.pending() == {}
    np.testing.assert_allclose(reopened["est"].params.values, reference["first"], rtol=1e-8)


def _without_its_input(tmp_path, mode):
    """A project whose deferred step names an object nothing makes (its field edited in the file)."""
    s = _session(OTHER)
    s.mode = mode
    path = save_project(s, tmp_path / f"{mode}.pymx")
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    entries = json.loads(files["commands.json"])
    entries[1]["fields"]["source"] = "gone"
    entries[1].pop("call", None)
    files["commands.json"] = json.dumps(entries).encode()
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return path


@pytest.mark.parametrize("mode", ["manual", "automatic"])
def test_a_deferred_step_whose_input_is_missing_opens_marked(tmp_path, mode):
    reopened = load_project(_without_its_input(tmp_path, mode), trust=False)  # never an error on opening
    record = reopened.history[1]
    assert set(reopened.pending()) == {"est"} and record.deferred and not record.ok
    assert "gone" in record.note and record.error is None
    assert "m" in reopened.objects and "est" not in reopened.objects
    with pytest.raises(NotFoundError, match="gone"):  # computing it says why it cannot run
        reopened.compute()
    assert set(reopened.pending()) == {"est"}  # and it stays pending
    again = load_project(save_project(reopened, tmp_path / "again.pymx"), trust=False)
    assert set(again.pending()) == {"est"} and "gone" in again.history[1].note


def test_the_note_shows_in_history(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.history_qt import HistoryPanel

    panel = HistoryPanel(print, print, print, print)
    panel.show_session(load_project(_without_its_input(tmp_path, "manual"), trust=False))
    item = panel.topLevelItem(1)
    assert item.text(3) == "not computed" and "It may not compute: " in item.toolTip(3) and "'gone'" in item.toolTip(3)


def test_adding_a_deferred_step_whose_input_is_missing_is_refused():
    s = Session()
    s.objects["data"] = _data()
    with pytest.raises(CommandError, match="no object named 'm'"):  # its call cannot be made: refused, as run() would
        s.add(EstimateModel("m", estimate="kappa", name="est"), deferred=True)
    assert s.history == []


def test_a_project_saved_by_0_1_0a16_with_the_step_marked_dead_opens(tmp_path, reference):
    entries = _entries(OLD)
    assert json.loads(zipfile.ZipFile(OLD).read("manifest.json"))["package_version"] == "0.1.0a16"
    assert [(bool(e.get("dead")), bool(e.get("deferred"))) for e in entries] == [(True, False), (False, True), (False, False)]
    reopened = load_project(OLD, trust=False)
    assert set(reopened.pending()) == {"est"} and not reopened.history[1].note
    assert not reopened.history[0].pruned and reopened.history[0].ok  # the step the deferred one uses is run again
    reopened.compute()
    np.testing.assert_allclose(reopened["est"].params.values, reference["first"], rtol=1e-8)
    saved = save_project(reopened, tmp_path / "resaved.pymx")
    assert not any(e.get("dead") for e in _entries(saved))  # saved again, nothing is marked dead
    with zipfile.ZipFile(OLD) as z:  # the same project in Automatic mode: the step is computed on opening
        files = {n: z.read(n) for n in z.namelist()}
    files["manifest.json"] = json.dumps({**json.loads(files["manifest.json"]), "mode": "automatic"}).encode()
    with zipfile.ZipFile(tmp_path / "auto.pymx", "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    opened = load_project(tmp_path / "auto.pymx", trust=False)
    np.testing.assert_allclose(opened["est"].params.values, reference["first"], rtol=1e-8)


def _fits():
    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    s = Session()
    s.objects["data"] = Dataset.from_frame(pd.DataFrame({"y": x + rng.normal(size=40), "x": x}, index=idx))
    return s


def test_manual_modes_deferral_on_opening_never_runs_a_step_before_what_it_uses(tmp_path, monkeypatch):
    """A slow step deferred on opening still runs in its place when a later step changes its inputs; one whose inputs
    stay the same waits, and a step that uses its result waits with it even when a later step changes its other
    inputs (it used to run in its place without that result, and opening failed), then computes from them as they were."""
    s = _fits()
    s.run(FitEquation("y ~ 1", name="base"))  # replaced later
    s.run_code("changed = float(base.params.values[0])")  # base replaced later: runs in its place
    s.run(FitEquation("y ~ x", name="slow"))  # its inputs stay the same: waits
    s.run_code("both = float(slow.params.values[1]) + float(base.params.values[0])")  # uses slow, and base replaced later
    s.run(FitEquation("y ~ x[-1]", name="base"))
    for i in (1, 2, 3):
        s.history[i] = replace(s.history[i], seconds=project_module.EXPENSIVE + 1)
    path = save_project(s, tmp_path / "p.pymx")
    monkeypatch.setattr(project_module, "__version__", "0.0.0-other")  # its saved results unused
    reopened = load_project(path, trust=True)
    assert set(reopened.pending()) == {"slow", "both"}  # 'both' waits for slow: it cannot run in its place
    assert reopened["changed"] == pytest.approx(s["changed"])
    reopened.compute()
    assert reopened["both"] == pytest.approx(s["both"])
