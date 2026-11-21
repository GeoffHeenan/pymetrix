"""Calculation: Manual and Automatic modes, frozen results, dead ends, and projects that open without redoing work
(dead ends skipped, slow results taken from the project, or deferred in Manual mode)."""

import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project
from pymetrix.commands import project as project_module


@pytest.fixture
def session():
    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    s = Session()
    s.objects["data"] = Dataset.from_frame(pd.DataFrame({"y": x + rng.normal(size=40), "x": x}, index=idx))
    return s


def _slope(s, name):
    return float(s[name].params.values[1])


def test_dead_ends_are_found_and_pruned(session):
    s = session
    s.run(FitEquation("y ~ x", name="fit"))
    s.run(FitEquation("y ~ x[-1]", name="fit"))  # replaces fit: the first is a dead end
    s.run(FitEquation("y ~ 1", name="tmp"))
    del s.objects["tmp"]  # deleted: a dead end
    s.run(GenerateSeries("z", "x * 2"))
    s.run(FitEquation("y ~ z", name="fz"))
    dead = [r.code for r in s.dead_ends()]
    assert dead == ["fit = Equation('y ~ x').fit(data, method=OLS())", "tmp = Equation('y ~ 1').fit(data, method=OLS())"]
    assert all(r.seconds > 0 for r in s.history)
    removed = s.prune()
    assert len(removed) == 2 and len(s.history) == 3 and "tmp" not in s.script() and s.dead_ends() == []


def test_manual_and_automatic_modes_and_frozen_results(session):
    s = session
    s.run(GenerateSeries("z", "x * 2"))
    s.run(FitEquation("y ~ z", name="fz"))
    assert s.mode == "manual"
    s.run(GenerateSeries("z", "x * 3", replace=True))
    assert set(s.stale()) == {"fz"}  # manual: marked out of date, not recomputed
    s.freeze("fz")
    assert s.stale() == {} and set(s.stale(frozen=True)) == {"fz"}
    assert s.recompute() == []  # frozen: never recomputed
    before = _slope(s, "fz")
    s.set_mode("automatic")
    assert _slope(s, "fz") == before  # still frozen
    s.unfreeze("fz")  # automatic: recomputed at once
    assert s.stale() == {} and _slope(s, "fz") == pytest.approx(before / 1.5)
    s.run(GenerateSeries("z", "x * 4", replace=True))
    assert s.stale() == {} and _slope(s, "fz") == pytest.approx(before / 2)
    with pytest.raises(ValueError, match="automatic"):
        s.set_mode("sometimes")
    with pytest.raises(KeyError):
        s.freeze("nothing")


def test_opening_skips_dead_ends_and_keeps_mode_and_frozen(session, tmp_path):
    s = session
    s.run(FitEquation("y ~ x", name="fit"))
    s.run(FitEquation("y ~ x[-1]", name="fit"))
    s.run(GenerateSeries("z", "x * 2"))
    s.run(FitEquation("y ~ z", name="fz"))
    s.set_mode("automatic")
    path = save_project(s, tmp_path / "p.pymx")
    with zipfile.ZipFile(path) as z:
        commands = json.loads(z.read("commands.json"))
        manifest = json.loads(z.read("manifest.json"))
    assert [bool(c.get("dead")) for c in commands] == [True, False, False, False]
    assert manifest["mode"] == "automatic" and all("seconds" in c and "outputs" in c for c in commands)
    reopened = load_project(path)
    assert reopened.history[0].pruned and not reopened.history[1].pruned and reopened.mode == "automatic"
    assert np.allclose(reopened["fit"].params.values, s["fit"].params.values)
    assert reopened.script() == s.script()  # the dead end stays in the script and the history
    again = load_project(save_project(reopened, tmp_path / "q.pymx"))
    assert again.history[0].pruned  # and stays dead when saved again


def _slow(s, i, seconds=5.0):
    """Pretend step ``i`` took ``seconds``."""
    s.history[i] = replace(s.history[i], seconds=seconds)


def test_slow_results_are_saved_and_frozen_values_kept(session, tmp_path, monkeypatch):
    s = session
    s.run(GenerateSeries("z", "x * 2"))
    s.run(FitEquation("y ~ z", name="fz"))
    s.run(FitEquation("y ~ x", name="cheap"))
    _slow(s, 1)
    frozen_slope = _slope(s, "fz")
    s.run(GenerateSeries("z", "x * 3", replace=True))
    s.freeze("fz")
    path = save_project(s, tmp_path / "p.pymx")
    with zipfile.ZipFile(path) as z:
        assert [n for n in z.namelist() if n.startswith("results/")] == ["results/1.pkl"]
    runs = []
    original = FitEquation.run
    monkeypatch.setattr(FitEquation, "run", lambda self, session: (runs.append(self.name), original(self, session))[1])
    reopened = load_project(path)
    assert runs == ["cheap"]  # fz came from the project, not computed again
    assert _slope(reopened, "fz") == pytest.approx(frozen_slope) and reopened.frozen == {"fz"}
    assert set(reopened.stale(frozen=True)) == {"fz"}
    runs.clear()
    untrusted = load_project(path, trust=False)  # saved results not used; z changes later, so fz cannot wait:
    assert runs == ["fz", "cheap"] and untrusted.pending() == {}  # it runs in its place, from the z it was frozen with
    assert _slope(untrusted, "fz") == pytest.approx(frozen_slope)


def test_manual_mode_defers_slow_steps_without_saved_results(session, tmp_path, monkeypatch):
    s = session
    s.run(GenerateSeries("r", "x * 2"))
    s.run(FitEquation("y ~ x", name="slow"))
    s.run(FitEquation("y ~ r", name="cheap"))
    from pymetrix.commands import RunCode  # noqa: F401 - typed code that uses a deferred result waits too

    s.run_code("slope = float(slow.params.values[1])")
    s.run(FitEquation("y ~ x[-1]", name="other"))
    _slow(s, 1)
    path = save_project(s, tmp_path / "p.pymx")
    assert set(json.loads(zipfile.ZipFile(path).read("manifest.json"))["cache"]["steps"]) == {"1"}
    version = project_module.__version__
    monkeypatch.setattr(project_module, "__version__", "0.0.0-other")  # its saved results cannot be used
    reopened = load_project(path)
    assert set(reopened.pending()) == {"slow", "slope"}  # the slow step waits, and the typed code that uses it
    assert "slow" not in reopened.objects and {"data", "cheap", "other"} <= set(reopened.objects)
    done = reopened.compute(["slope"])
    assert [r.outputs for r in done] == [("slow",), ("slope",)] and reopened.pending() == {}
    assert reopened["slope"] == pytest.approx(_slope(s, "slow"))
    assert [r.outputs for r in reopened.history] == [r.outputs for r in s.history]  # in their places

    monkeypatch.setattr(project_module, "__version__", version)
    trusted = load_project(path)  # saved by this user, same version: the results come from the project
    assert trusted.pending() == {} and np.allclose(trusted["slow"].params.values, s["slow"].params.values)


def test_saved_results_are_not_used_by_another_version(session, tmp_path, monkeypatch):
    s = session
    s.run(FitEquation("y ~ x", name="slow"))
    _slow(s, 0)
    path = save_project(s, tmp_path / "p.pymx")
    monkeypatch.setattr(project_module, "__version__", "0.0.0-other")
    reopened = load_project(path)
    assert reopened.pending() == {"slow": reopened.history[0]}  # manual (as saved): deferred, not unpickled


def test_steps_and_reports_say_when_they_use_results_not_up_to_date(session, tmp_path):
    from pymetrix.commands import ExportReport

    s = session
    s.run(GenerateSeries("z", "x * 2"))
    s.run(FitEquation("y ~ z", name="fz"))
    s.run(ExportReport("fz", str(tmp_path / "fresh.md")))
    assert s.history[-1].cautions == () and "Caution" not in (tmp_path / "fresh.md").read_text(encoding="utf-8")
    s.run(GenerateSeries("z", "x * 3", replace=True))
    s.run(ExportReport("fz", str(tmp_path / "stale.md")))
    assert s.history[-1].cautions == ("fz is out of date: z in data has changed since fz was computed",)
    text = (tmp_path / "stale.md").read_text(encoding="utf-8")
    assert text.index("Caution: fz is out of date") < text.index("## fz")  # at the top, before the results
    s.freeze("fz")
    s.run(ExportReport("fz", str(tmp_path / "frozen.md")))
    assert "Caution: fz is frozen, and its inputs have changed since" in (tmp_path / "frozen.md").read_text(encoding="utf-8")
    s.unfreeze("fz")
    s.recompute()
    s.run(ExportReport("fz", str(tmp_path / "again.md")))
    assert s.history[-1].cautions == ()
