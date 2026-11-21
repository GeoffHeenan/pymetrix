"""Session.add: a step run now (as Session.run), or recorded deferred, as Manual mode defers a slow step when a project
is opened, and run by compute(). Checked against run() itself and against a project reopened with a deferred step."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.calls import make_call
from pymetrix.codec import Ref
from pymetrix.commands import CommandError, FitEquation, GenerateSeries, RunCode, Session, load_project, save_project
from pymetrix.commands import project as project_module
from pymetrix.errors import SpecificationError


def _data():
    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    return Dataset.from_frame(pd.DataFrame({"y": 0.5 + 2 * x + rng.normal(size=40), "x": x}, index=idx))


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = _data()
    return s


def test_add_runs_the_step_as_run_does_by_default(session):
    other = Session()
    other.objects["data"] = session["data"]
    got = session.add(FitEquation("y ~ x", name="fit"))
    want = other.run(FitEquation("y ~ x", name="fit"))
    assert np.array_equal(got.params.values, want.params.values)
    assert [r.code for r in session.history] == [r.code for r in other.history]
    assert [(r.ok, r.deferred) for r in session.history] == [(True, False)]
    with pytest.raises(SpecificationError):
        session.add(FitEquation("y ~ x", name="fit2"), outputs=["fit2"])  # outputs only for a deferred step


def test_a_deferred_step_waits_and_compute_runs_it_in_its_place(session):
    session.run(GenerateSeries("z", "x * 2"))
    assert session.add(FitEquation("y ~ x", name="slow"), deferred=True) is None
    session.run(FitEquation("y ~ z", name="cheap"))
    assert "slow" not in session.objects and set(session.pending()) == {"slow"}
    record = session.history[1]
    assert (record.ok, record.deferred, record.outputs) == (False, True, ("slow",))
    assert record.code == "slow = Equation('y ~ x').fit(data, method=OLS())"
    script = session.script()
    assert "# not computed yet in the session (deferred): slow\nslow = Equation('y ~ x').fit(data, method=OLS())\n" in script
    assert session.cautions(["slow"]) == ["slow is not computed yet"]

    done = session.compute()
    assert [r.outputs for r in done] == [("slow",)] and session.pending() == {}
    assert [r.outputs for r in session.history] == [("data",), ("slow",), ("cheap",)]  # in its place
    assert session.history[1].ok and not session.history[1].deferred
    reference = Session()
    reference.objects["data"] = _data()
    want = reference.run(FitEquation("y ~ x", name="slow"))
    assert np.array_equal(session["slow"].params.values, want.params.values)  # the same numbers as running it at once
    assert "not computed yet" not in session.script()


def test_deferred_steps_that_use_each_other_and_typed_code(session):
    session.add(FitEquation("y ~ x", name="fit"), deferred=True)
    session.add(RunCode("slope = float(fit.params.values[1])"), deferred=True)  # uses a pending result: not validated now
    assert set(session.pending()) == {"fit", "slope"}
    done = session.compute(["slope"])
    assert [r.outputs for r in done] == [("fit",), ("slope",)]
    assert session["slope"] == pytest.approx(float(session["fit"].params.values[1]))


def test_a_deferred_call_and_its_outputs(session):
    call = make_call("pymetrix:dataset.with_series", Ref("data"), output="data2", w="x + 1")
    session.add(call, deferred=True)
    assert set(session.pending()) == {"data2"}
    session.compute()
    assert np.allclose(session["data2"]["w"].to_numpy(), session["data"]["x"].to_numpy() + 1)
    with pytest.raises(SpecificationError, match="outputs"):
        session.add(RunCode("print(1)"), deferred=True)  # makes nothing it can name
    session.add(RunCode("print(1)"), deferred=True, outputs=["printed"])
    assert "printed" in session.pending()


def test_a_deferred_step_is_checked_as_far_as_it_can_be(session):
    with pytest.raises(CommandError):
        session.add(FitEquation("y ~ x", data="nothing", name="fit"), deferred=True)  # no such dataset
    with pytest.raises(CommandError, match="start"):
        session.add(FitEquation("y ~ x", name="fit", start="2005Q1", end="2001Q1"), deferred=True)
    assert session.history == []


def test_a_deferred_step_can_be_undone(session):
    session.run(FitEquation("y ~ x", name="fit"))
    session.add(FitEquation("y ~ x[-1]", name="later"), deferred=True)
    assert session.undo_label().startswith("later = ")
    session.undo()
    assert len(session.history) == 1 and session.pending() == {}


def test_saved_and_reopened_like_manual_modes_own_deferral(session, tmp_path, monkeypatch):
    session.add(FitEquation("y ~ x", name="slow"), deferred=True)
    session.run(FitEquation("y ~ x[-1]", name="cheap"))
    path = save_project(session, tmp_path / "p.pymx")
    with zipfile.ZipFile(path) as z:
        steps = json.loads(z.read("commands.json"))
        assert "# not computed yet in the session (deferred): slow" in z.read("session.py").decode()
    assert [(e.get("deferred", False), e["outputs"]) for e in steps] == [(True, ["slow"]), (False, ["cheap"])]
    reopened = load_project(path)
    assert set(reopened.pending()) == {"slow"} and "cheap" in reopened.objects
    added, opened = session.history[0], reopened.history[0]
    assert (added.ok, added.deferred, added.outputs, added.code) == (opened.ok, opened.deferred, opened.outputs, opened.code)

    # the same record Manual mode's own deferral of a slow step makes on opening
    plain = Session()
    plain.objects["data"] = _data()
    plain.run(FitEquation("y ~ x", name="slow"))
    plain.run(FitEquation("y ~ x[-1]", name="cheap"))
    plain.history[0] = __import__("dataclasses").replace(plain.history[0], seconds=project_module.EXPENSIVE + 1)
    saved = save_project(plain, tmp_path / "q.pymx")
    monkeypatch.setattr(project_module, "__version__", "0.0.0-other")  # its saved results unused: deferred on opening
    slow = load_project(saved).history[0]
    assert (slow.ok, slow.deferred, slow.outputs, slow.code) == (added.ok, added.deferred, added.outputs, added.code)

    reopened.compute()
    assert np.array_equal(reopened["slow"].params.values, plain["slow"].params.values)


def test_automatic_mode_runs_a_deferred_step_on_opening(session, tmp_path):
    session.add(FitEquation("y ~ x", name="slow"), deferred=True)
    session.set_mode("automatic")
    assert set(session.pending()) == {"slow"}  # recorded deferred in either mode: compute() runs it
    reopened = load_project(save_project(session, tmp_path / "p.pymx"))
    assert reopened.pending() == {} and "slow" in reopened.objects  # Automatic mode computes on opening


def test_observers_hear_of_a_deferred_step(session):
    from pymetrix.extension import SessionObserver
    from pymetrix.plugins import load, register, registry

    load("observers")
    entries = registry("observers")._entries
    saved = dict(entries)
    entries.clear()
    heard = []
    try:
        register("observers", SessionObserver("test-add", heard.append))
        session.add(FitEquation("y ~ x", name="slow"), deferred=True)
    finally:
        entries.clear()
        entries.update(saved)
    assert [(e.kind, e.detail.get("deferred"), e.records[0].deferred) for e in heard] == [("step", True, True)]


def test_history_shows_a_deferred_step_as_not_computed(session):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.history_qt import HistoryPanel

    session.add(FitEquation("y ~ x", name="slow"), deferred=True)
    panel = HistoryPanel(print, print, print, print)
    panel.show_session(session)
    assert panel.topLevelItem(0).text(3) == "not computed"
