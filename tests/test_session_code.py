"""Typed code in the session: recording, replay in projects, and out-of-date results."""

from pathlib import Path

import pytest

from pymetrix import Dataset
from pymetrix.commands import AddToDataset, FitEquation, Filter, RunCode, Session, load_project, save_project

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.read(str(DATA))
    return s


def test_typed_code_is_recorded_with_what_it_read_and_wrote(session):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run_code("b = fit.params['y']\ndef double(x):\n    return 2 * x\nb2 = double(b)")
    record = session.history[-1]
    assert record.typed and record.ok and set(record.reads) == {"fit"} and set(record.writes) == {"b", "b2"}
    assert "double" not in session.objects and callable(session.namespace["double"])  # functions stay in the namespace
    assert session.run_code("b2 / b") == pytest.approx(2.0)  # a final expression is the result
    session.run_code("del b")
    assert "b" not in session.objects
    with pytest.raises(ZeroDivisionError):
        session.run_code("1 / 0")
    assert not session.history[-1].ok and "1 / 0" not in session.script()
    assert "b2 = double(b)" in session.script()


def test_ipython_syntax_is_recorded_but_not_replayed(session):
    before = session.begin_code()
    session.namespace["z"] = 3  # as if the console ran it
    record = session.end_code(before, "z = 3\n%timeit z + 1", ok=True)
    assert not record.replayable and session.objects["z"] == 3
    assert "# (IPython only, not replayed) %timeit z + 1" in session.script()
    assert RunCode("%timeit 1").run(session) is None


def test_dialogs_and_code_share_one_namespace(session):
    session.run_code("small = data.select(c=1) if False else data")
    session.run(FitEquation("c ~ y", data="small", name="fit"))
    assert session.namespace["fit"] is session.objects["fit"]  # dialog results reach the code namespace
    assert set(session.history[-1].reads) == {"small"}


def test_out_of_date_results_and_recompute(session):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run(Filter("y", name="cycles", keep_updated=False))
    session.run_code("slope = fit.params['y']")
    session.run(AddToDataset("fit", "resid", series_name="u"))  # adds a series to data: nothing is out of date
    assert session.stale() == {}
    session.run_code("data = data.with_series(c=data['c'] * 1.01)")
    stale = session.stale()
    assert stale["fit"] == "c in data has changed since fit was computed"
    assert "cycles" not in stale  # the filter used y only; dialog steps record the series they use
    assert stale["slope"] == "it depends on fit, which is out of date"
    old_slope = session["slope"]
    session.recompute(["fit"])
    assert "fit" not in session.stale() and session.stale()["slope"] == "fit has changed since slope was computed"
    session.recompute()
    assert session.stale() == {} and session["slope"] != old_slope
    session.run_code("data = data.with_series(z='c - y')")
    assert session.stale() == {}  # only a series was added


def test_stale_through_a_chain(session):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run_code("b = fit.params['y']")
    session.objects["fit"] = session["fit"]  # same object: not a change
    assert session.stale() == {}
    session.run_code("fit = None")
    assert session.stale() == {"b": "fit has changed since b was computed"}
    session.run_code("del fit")
    assert session.stale() == {"b": "fit no longer exists"}


def test_projects_replay_typed_code(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run_code("import math\nlog_slope = math.log(fit.params['y'])")
    before = session.begin_code()
    session.end_code(before, "%who", ok=True)
    path = save_project(session, tmp_path / "mixed.pymx")
    reopened = load_project(path)
    assert reopened["log_slope"] == pytest.approx(session["log_slope"])
    assert [r.typed for r in reopened.history] == [r.typed for r in session.history]
    assert reopened.script() == session.script()
