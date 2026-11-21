"""Autosave and recovery: a copy of unsaved work that leaves the session untouched, kept while its app runs, offered
back when the app stopped without closing, and gone once the work is saved, closed or replaced."""

import os
import time

import pytest

from pymetrix import Dataset, recovery
from pymetrix.commands import GenerateSeries, Session, load_project, save_project


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    return tmp_path


def _session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(GenerateSeries("z", "y * 2"))
    return s


def test_a_copy_leaves_the_session_as_it_was(tmp_path):
    s = _session()
    assert s.dirty and s.path is None
    path = recovery.write(s, "abc", title="draft")
    assert s.dirty and s.path is None  # still unsaved, still nowhere
    back = load_project(path)
    assert "z" in back.objects["data"].names and len(back.history) == len(s.history)
    saved = save_project(s, tmp_path / "real.pymx")
    assert not s.dirty and s.path == saved  # an ordinary save still does both


def test_orphans_are_copies_whose_app_stopped():
    s = _session()
    recovery.write(s, "running")
    assert recovery.orphans() == []  # its heartbeat is fresh: its app is running
    later = time.time() + recovery.STALE_SECONDS + 5
    found = recovery.orphans(now=later)
    assert [r.key for r in found] == ["running"] and found[0].original is None
    assert "an unsaved session, autosaved" in found[0].describe()
    assert recovery.orphans(exclude="running", now=later) == []
    alive = recovery.recovery_dir() / "running.alive"
    alive.unlink()  # no heartbeat at all: an orphan at once
    assert [r.key for r in recovery.orphans()] == ["running"]
    recovery.discard("running")
    assert recovery.orphans(now=later) == [] and not any(recovery.recovery_dir().iterdir())


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_app_autosaves_unsaved_work_and_forgets_it_when_saved(app, tmp_path):
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        key = window.autosave_key
        assert not window.autosave_now()  # nothing unsaved
        s.run(GenerateSeries("z", "y"))
        assert not window.autosave_now()  # the interval has not passed
        assert window.autosave_now(force=True)
        copy = recovery.recovery_dir() / f"{key}.pymx"
        assert copy.exists() and s.dirty
        assert not window.autosave_now(force=False)  # nothing changed since the copy
        save_project(s, tmp_path / "p.pymx")
        s.dirty = True
        window.autosave_now(force=True)
        assert copy.exists()

        window.autosave_toggle.setChecked(False)
        assert not window.autosave_now()  # off
        window.autosave_toggle.setChecked(True)
    finally:
        s.dirty = False
        window.close()
    assert not (recovery.recovery_dir() / f"{key}.pymx").exists()  # closed on purpose: nothing to recover


def test_the_app_offers_back_what_a_crash_left(app, tmp_path):
    from pymetrix.ui.qt import main_window

    crashed = _session()
    crashed.path = tmp_path / "work.pymx"
    recovery.write(crashed, "crashed", title="work.pymx")
    (recovery.recovery_dir() / "crashed.alive").unlink()  # its app is gone
    window = main_window(Session())
    try:
        asked = []
        assert window.offer_recovery(answer=lambda rec: asked.append(rec.describe()) or "recover") == 1
        assert asked and asked[0].startswith("work.pymx, autosaved")
        restored = window.state["session"]
        assert "z" in restored.objects["data"].names and restored.dirty and restored.path == tmp_path / "work.pymx"
        assert not (recovery.recovery_dir() / "crashed.pymx").exists()  # it is the open session now
        recovery.write(_session(), "other")
        (recovery.recovery_dir() / "other.alive").unlink()
        assert window.offer_recovery(answer=lambda rec: "discard") == 0 and recovery.orphans() == []
    finally:
        window.state["session"].dirty = False
        window.close()
