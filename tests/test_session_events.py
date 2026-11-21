"""Session events, project attachments and the save log: what a plugin that keeps a record of the work (a lab book)
is given (pymetrix.commands.events, extension API 1.3)."""

import zipfile

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project
from pymetrix.commands.events import SessionObserverWarning
from pymetrix.errors import SpecificationError
from pymetrix.extension import SessionObserver
from pymetrix.plugins import PluginInfo, register, registry
from pymetrix.progress import Cancelled


@pytest.fixture(autouse=True)
def only_these_observers():
    """Installed plugins' observers (such as the lab book) are set aside: these tests register their own."""
    from pymetrix.plugins import load

    load("observers")  # loaded now, so not again while set aside
    entries = registry("observers")._entries
    saved = dict(entries)
    entries.clear()
    yield
    entries.clear()
    entries.update(saved)


@pytest.fixture
def events():
    """A recording observer, as a plugin would register it; removed afterwards."""
    seen = []
    info = PluginInfo("pymetrix-test-labbook", "0.1")
    register("observers", SessionObserver("labbook", seen.append), name="labbook", plugin=info)
    yield seen
    registry("observers")._entries.pop(f"{info.prefix}:labbook")


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def _kinds(events):
    return [(e.kind, e.during) for e in events]


def test_every_change_to_the_history_is_told(session, events):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    with pytest.raises(Exception):
        s.run(FitEquation("c ~ nothing", name="bad"))
    assert _kinds(events) == [("step", "work"), ("failed", "work")]
    assert events[0].records[0].code == "fit = Equation('c ~ y').fit(data, method=OLS())" and events[0].session is s
    assert events[1].records[0].ok is False and events[1].time.tzinfo is not None

    s.set_mode("automatic")
    events.clear()
    s.run(GenerateSeries("y", "y * 1.01", replace=True))  # the fit is recomputed after it
    assert _kinds(events) == [("step", "work"), ("step", "recalculation")]
    events.clear()
    s.undo()  # the revision and its recalculation, removed together
    assert [e.kind for e in events] == ["undone"] and len(events[0].records) == 2

    events.clear()
    with pytest.raises(Cancelled):
        s.run_code("from pymetrix.progress import Cancelled\nraise Cancelled('stopped')")
    assert [e.kind for e in events] == ["cancelled"] and "raise Cancelled" in events[0].code
    assert all(r.ok or "Cancelled" not in r.code for r in s.history)  # still no trace in the history itself


def test_cleaning_up_dead_ends_is_told(session, events):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.run(FitEquation("c ~ y[-1]", name="fit"))  # replaces the first: a dead end
    events.clear()
    removed = s.prune()
    assert [e.kind for e in events] == ["removed"] and events[0].records == tuple(removed) and len(removed) == 1


def test_typed_code_made_of_calls_is_one_event_with_its_calls(session, events):
    s = session
    s.run_code("a = Equation('c ~ y').fit(data)\nb = Equation('c ~ y[-1]').fit(data)")
    steps = [e for e in events if e.kind == "step"]
    assert len(steps) == 1 and [r.call.output for r in steps[0].records] == ["a", "b"]


def test_projects_keep_attachments_and_log_each_save(session, events, tmp_path):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))

    def keep(event):  # what a lab book does when the project is about to be written
        if event.kind == "saving":
            event.session.attachments["labbook"] = {"labbook.jsonl": b'{"seq": 1}\n'}

    events_before = len(events)
    writer = PluginInfo("pymetrix-test-writer", "1.0")
    register("observers", SessionObserver("writer", keep, records=False), name="writer", plugin=writer)
    try:
        path = save_project(s, tmp_path / "audited")
    finally:
        registry("observers")._entries.pop(f"{writer.prefix}:writer")
    assert [e.kind for e in events[events_before:]] == ["saving"] and events[-1].path == str(path)
    with zipfile.ZipFile(path) as z:
        assert z.read("attachments/labbook/labbook.jsonl") == b'{"seq": 1}\n'
    log = s.save_log
    assert len(log) == 1 and log[0]["recorders"] == {"labbook": "pymetrix-test-labbook 0.1"}  # only observers that record

    events.clear()
    opened = load_project(path)
    assert opened.attachments == {"labbook": {"labbook.jsonl": b'{"seq": 1}\n'}} and opened.save_log == log
    assert [e.kind for e in events] == ["step", "opened"] and events[0].during == "opening" and events[0].session is opened
    assert events[-1].records == tuple(opened.history) and events[-1].path == str(path)

    save_project(opened, tmp_path / "copy", as_copy=True)  # an autosave: logged in the copy, not in the session
    assert len(opened.save_log) == 1
    with zipfile.ZipFile(tmp_path / "copy.pymx") as z:
        assert z.read("attachments/labbook/labbook.jsonl") == b'{"seq": 1}\n'  # kept, though nothing wrote it again


def test_a_gap_shows_when_no_recorder_was_active(session, tmp_path):
    path = save_project(session, tmp_path / "unrecorded")
    assert load_project(path).save_log[-1]["recorders"] == {}


def test_attachments_need_plain_names_and_bytes(session, tmp_path):
    session.attachments["../escape"] = {"x": b""}
    with pytest.raises(SpecificationError, match="plain name"):
        save_project(session, tmp_path / "bad")
    session.attachments = {"labbook": {"sub/dir.txt": b""}}
    with pytest.raises(SpecificationError, match="no folders"):
        save_project(session, tmp_path / "bad")
    session.attachments = {"labbook": {"notes.txt": "text"}}
    with pytest.raises(SpecificationError, match="must be bytes"):
        save_project(session, tmp_path / "bad")


def test_an_observer_that_fails_never_stops_the_work(session):
    def broken(event):
        raise RuntimeError("disk full")

    info = PluginInfo("pymetrix-test-broken", "0.1")
    register("observers", SessionObserver("broken", broken), name="broken", plugin=info)
    try:
        with pytest.warns(SessionObserverWarning, match="disk full"):
            session.run(FitEquation("c ~ y", name="fit"))
        assert "fit" in session.objects and session.history[-1].ok
    finally:
        registry("observers")._entries.pop(f"{info.prefix}:broken")
