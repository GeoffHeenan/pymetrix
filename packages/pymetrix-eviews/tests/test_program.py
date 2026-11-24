"""EViews programs translated (phase 1, data work): the parser, the report, the script, steps mode, and the two
giving equal objects. Phase 2 (estimation, tests and models) is test_program_estimation.py's.

The expected values are worked out by hand from the programs' text (fixtures/programs/*.prg), not by running the
translator: each array below says how it was made.
"""

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Matrix, Sample
from pymetrix.commands import Session
from pymetrix_eviews import ProgramTranslation, translate_program
from pymetrix_eviews.program import STATUSES, parse_program

FIXTURES = Path(__file__).parent / "fixtures"
PROGRAMS = FIXTURES / "programs"
NAN = np.nan


def _equal_values(a, b):
    if isinstance(a, Dataset):
        assert isinstance(b, Dataset)
        pd.testing.assert_frame_equal(a.frame, b.frame)
        assert a.axes == b.axes
        assert {k: (v.label, v.units, v.description, v.derived_from) for k, v in a.variables.items()} == \
            {k: (v.label, v.units, v.description, v.derived_from) for k, v in b.variables.items()}
        assert a.sample == b.sample
    elif isinstance(a, Matrix):
        np.testing.assert_array_equal(a.values, b.values)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    elif hasattr(a, "names") and hasattr(a, "data"):  # a Selection
        assert list(a.names) == list(b.names)
    elif hasattr(a, "params") and hasattr(a, "equation"):  # a fitted equation (phase 2)
        np.testing.assert_array_equal(a.params.values, b.params.values)
    else:
        assert a == b


def _both(t: ProgramTranslation, session=None):
    """The script run as Python, and the steps run in a session: every object of one equal to the other's."""
    namespace: dict = {}
    exec(compile(t.script, "<script>", "exec"), namespace)
    out = t.run(session)
    s = out["session"]
    made = {**{k: v for k, v in out.items() if k != "session"}, **s.objects}
    for name, value in made.items():
        assert name in namespace, f"{name} is made by the steps but not by the script"
        _equal_values(value, namespace[name])
    for name in t.pages.values():
        assert name in s.objects
    return namespace, out


# -- the parser --------------------------------------------------------------------------------------------------


def test_statements_lines_and_parts():
    p = parse_program("smpl 1990q1 2000q4 ' a comment\n\nx.label(d) GDP _\n  in 2017 prices\n!i = 3\n%s = \"a = b\"\n"
                      "ls(cov=white) y c x\n")
    kinds = [(s.line, s.last_line, s.kind, s.word) for s in p.statements]
    assert kinds == [(1, 1, "command", "smpl"), (3, 4, "method", "label"), (5, 5, "assign", ""), (6, 6, "assign", ""),
                     (7, 7, "command", "ls")]
    label = p.statements[1]
    assert (label.obj, label.options, " ".join(label.rest.split())) == ("x", "d", "GDP in 2017 prices")
    assert p.statements[2].lhs == "!i" and p.statements[2].rest == "3"
    assert p.statements[3].lhs == "%s" and p.statements[3].rest == '"a = b"'
    assert p.statements[4].options == "cov=white" and p.statements[4].rest == "y c x"
    assert p.comments == {1: "a comment"}
    # an offset points into the program's own text
    st = p.statements[0]
    assert p.source[st.rest_at:st.rest_at + 6] == "1990q1"


# -- a hand-written program: every value worked out by hand ------------------------------------------------------


def test_hand_program_values_by_hand():
    t = translate_program(PROGRAMS / "hand.prg")
    namespace, out = _both(t)
    q = out["session"]["q"]
    f = q.frame
    tq = np.arange(1.0, 9.0)  # t = 1 in 2000Q1, then t(-1) + 1 period by period: 1, 2, ..., 8
    y = 2 * tq + 1  # 3, 5, ..., 17
    z = y * 0.5 - 2
    np.testing.assert_array_equal(f["t"], tq)
    np.testing.assert_array_equal(f["y"], y)
    np.testing.assert_array_equal(f["z"], z)
    # v = 10 within 2000Q3-2001Q2 only (renamed vv); w was missing there and is deleted
    np.testing.assert_array_equal(f["vv"], [NAN, NAN, 10, 10, 10, 10, NAN, NAN])
    assert "w" not in f
    # m and sd over the rows with y > 9: 11, 13, 15, 17
    assert out["m"] == 14.0
    assert out["sd"] == pytest.approx(math.sqrt((9 + 1 + 1 + 9) / 3), abs=1e-14)
    # @pch(y) within @first-2000Q4: y/y(-1) - 1, none before the first period
    np.testing.assert_allclose(f["c"], [NAN, 5 / 3 - 1, 7 / 5 - 1, 9 / 7 - 1, NAN, NAN, NAN, NAN], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(f["dz"], [NAN] + [1.0] * 7)
    np.testing.assert_array_equal(f["f"], y + 14)  # frml f = y + m, a derived series
    assert q.variables["f"].recipe is not None
    np.testing.assert_array_equal(f["qabcd"], y + 1)  # series q{%x}: %x = s + "d" = "abcd"
    assert q.variables["t"].units == "quarters" and q.variables["y"].label == "Why"
    assert list(out["session"]["g"].names) == ["t", "y"]
    np.testing.assert_array_equal(out["b"].values, [[1.0], [2.0]])  # fill by column
    np.testing.assert_array_equal(out["mm"].values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # fill(b=r): by row
    assert (out["s"], out["x"], out["half"], out["two"]) == ("abc", "abcd", 0.5, 2)
    assert q.sample is None  # smpl @all last


def test_hand_program_report():
    t = translate_program(PROGRAMS / "hand.prg")
    lines = (PROGRAMS / "hand.prg").read_text().splitlines()
    rows = {r.line: r for r in t.report}
    assert sorted(rows) == [i for i, text in enumerate(lines, start=1) if text.strip()]  # a row per line with text
    assert all(r.status in STATUSES for r in t.report)
    assert t.report.counts() == {"translated": 30, "translated with a note": 7, "left out": 6, "error": 0}
    assert rows[6].status == "translated with a note" and "period by period" in rows[6].reason
    assert rows[32].reason == "continues line 31" and rows[36].reason == "continues line 35"
    assert rows[37].status == "left out" and "graphs are not translated" in rows[37].reason
    assert rows[38].status == "translated" and rows[38].python == 'eq = Equation("y ~ t", name="eq").fit(q)'
    assert "@year" in rows[39].reason and "gap" in rows[39].reason
    assert rows[40].reason == "a for loop: phase 3 (control flow)"
    assert rows[41].reason == "inside a for loop (line 40): phase 3 (control flow)"
    assert rows[43].status == "left out" and "File → Save" in rows[43].reason
    assert rows[8].python == 'q = q.with_series(y="2*t + 1")'
    assert rows[11].python == 'q = q.with_series(z=f"y*({half}) - ({two})")'
    table = t.report.summary()
    assert list(table.columns) == ["line", "status", "EViews", "Python", "reason"] and len(table.rows) == len(t.report)
    assert "30 translated" in table.notes[0]
    # the script keeps every EViews line as a comment
    assert "# 37: graph gr.line y z" in t.script and "#   left out: graphs are not translated" in t.script
    assert t.script.splitlines()[5] == "from pymetrix import Dataset, Matrix, Sample, Equation"


def test_two_pages_offsets_named_sample_and_fill():
    t = translate_program(PROGRAMS / "annual.prg")
    _, out = _both(t)
    a, q = out["session"]["a"].frame, out["session"]["q"].frame
    x = 2.0 ** np.arange(10)  # 1 in 1990, then x(-1)*2 from 1991: 1, 2, 4, ..., 512
    y = np.where((np.arange(1990, 2000) >= 1992) & (np.arange(1990, 2000) <= 1997), x - 1, NAN)  # @first+2 to @last-2
    x[8:] = [7.0, NAN]  # x.fill(o=1998) 7, na, after y was made
    np.testing.assert_array_equal(a["x"], x)
    np.testing.assert_array_equal(a["y"], y)
    np.testing.assert_array_equal(a["k"], [NAN] * 3 + [5.0] * 4 + [NAN] * 3)  # the named sample mid, 1993-1996
    np.testing.assert_array_equal(a["v"], np.ones(10))  # after pageselect a
    np.testing.assert_array_equal(q["u"], [3.0] * 4)
    assert out["mid"] == Sample(("1993", "1996"))
    assert t.pages == {"a": "a", "q": "q"}
    assert t.report.counts()["error"] == 0


# -- steps mode ----------------------------------------------------------------------------------------------------


def test_steps_are_recorded_and_their_script_replays():
    t = translate_program(PROGRAMS / "hand.prg")
    s = Session()
    out = t.run(s)
    assert out["session"] is s
    codes = [r.code for r in s.history]
    assert codes[0] == "q = Dataset.empty('Q', '2000Q1', '2001Q4')"
    assert any("with_series(z='y*(0.5) - (2)')" in c for c in codes)  # the scalars' values in the step
    assert all(r.ok for r in s.history)
    files = s.attachments["pymetrix-eviews"]
    assert files["hand.py"].decode() == t.script and b"left out" in files["hand.txt"]
    ns: dict = {}
    exec(s.script(), ns)
    _equal_values(ns["q"], s["q"])


def test_a_failing_step_stops_with_its_line():
    t = translate_program("wfcreate q 2000q1 2000q4\nseries y = 1\nsmpl 2001q1 2001q4\nseries x = 2\nseries z = 3\n")
    s = Session()
    with pytest.raises(Exception, match="line 3"):
        t.run(s)
    assert "x" not in s["untitled"].frame


def test_names_python_keeps_and_make_call_takes():
    t = translate_program("wfcreate(page=data) q 2000q1 2000q2\nseries output = 1\nscalar lambda = 2\n%in = \"a\"\n"
                          "series within = 3\nseries y = output*lambda\n")
    rows = {r.line: r for r in t.report}
    assert "data is called data_v" in rows[1].reason
    assert "lambda is called lambda_v" in rows[3].reason
    assert rows[4].python == 'in_v = "a"'
    assert rows[5].status == "left out" and "within" in rows[5].reason
    _, out = _both(t)
    np.testing.assert_array_equal(out["session"]["data_v"].frame["y"], [2.0, 2.0])
    np.testing.assert_array_equal(out["session"]["data_v"].frame["output"], [1.0, 1.0])


# -- workfiles: a bundle, a CSV file, neither -----------------------------------------------------------------------


def test_wfopen_a_bundle(tmp_path):
    shutil.copytree(FIXTURES / "macro", tmp_path / "macro")
    bundle = tmp_path / "macro" / "macro.evbundle"
    t = translate_program("wfopen macro\nseries y = gdp*alpha\nsmpl @all\nseries one = 1\npageselect annual\n"
                          "series z = gdp_a/pop\n", bundle=bundle)
    with pytest.warns(Warning):  # the reader notes the series it renamed in the cross page
        _, out = _both(t)
    s = out["session"]
    assert t.pages == {"quarterly": "quarterly", "Annual": "annual", "panel": "panel", "cross": "cross"}
    gdp = Dataset.read(str(bundle), page="quarterly").frame["gdp"].to_numpy()
    # the page opens with its saved sample, 1980q1 1982q2: y = gdp*alpha (0.3, the workfile's scalar) there only
    expected = np.where(np.arange(12) < 10, gdp * 0.3, NAN)
    np.testing.assert_allclose(s["quarterly"].frame["y"], expected, rtol=1e-15)
    np.testing.assert_array_equal(s["quarterly"].frame["one"], np.ones(12))
    annual = Dataset.read(str(bundle), page="Annual").frame
    np.testing.assert_allclose(s["annual"].frame["z"], annual["gdp_a"] / annual["pop"], rtol=1e-15)
    first = t.report.line(1)
    assert "saved sample" in first.reason and "eviews_objects(quarterly)" in first.reason
    assert "import pymetrix_eviews" in t.script


def test_wfopen_a_csv_file(tmp_path):
    pd.DataFrame({"obs": ["2000Q1", "2000Q2", "2000Q3"], "x": [1.0, 2.0, 4.0]}).to_csv(tmp_path / "m.csv", index=False)
    t = translate_program("wfopen m.wf1\nseries y = dlog(x)\n", bundle=tmp_path / "m.csv", read_options={"time": "obs", "freq": "Q"})
    _, out = _both(t)
    np.testing.assert_allclose(out["session"]["m"].frame["y"], [NAN, math.log(2), math.log(2)], rtol=1e-15)
    assert "import pymetrix_eviews" not in t.script


def test_wfopen_without_data_says_how():
    t = translate_program("wfopen macro\nseries y = 1\n")
    first, second = t.report.line(1), t.report.line(2)
    assert first.status == "error" and "bundle" in first.reason and "CSV or Excel" in first.reason
    assert second.status == "error" and "no workfile is open" in second.reason


def test_unknown_series_is_an_error_until_lines_left_out_may_make_it():
    t = translate_program("wfcreate q 2000q1 2000q4\nseries y = nosuch + 1\nfetch nosuch\nseries z = nosuch + 1\n")
    assert t.report.line(2).status == "error" and "nosuch is not a series" in t.report.line(2).reason
    assert t.report.line(3).status == "left out"
    assert t.report.line(4).status == "translated"  # the fetch may have made it


# -- what is left out, and why --------------------------------------------------------------------------------------


@pytest.mark.parametrize("expression, name", [
    ("@year(x)", "@year"), ("@quarter(x)", "@quarter"), ("@month(x)", "@month"), ("@obsnum", "@obsnum"),
    ("@round(x)", "@round"), ("@floor(x)", "@floor"), ("@ceiling(x)", "@ceiling"), ("@nan(x, 0)", "@nan"),
    ("@isna(x)", "@isna"), ("@iif(x > 1, 1, 0)", "@iif"), ("nrnd", "nrnd"), ("rnd", "rnd"), ("@date", "@date"),
    ("@trend", "@trend"), ("@seas(1)", "@seas"), ("x + na", "na"),
])
def test_the_audit_gaps_are_left_out_by_name(expression, name):
    t = translate_program(f"wfcreate q 2000q1 2000q4\nseries x = 1\nseries y = {expression}\n")
    row = t.report.line(3)
    assert row.status == "left out" and name in row.reason and "gap" in row.reason


@pytest.mark.parametrize("line, words", [
    ("for !i = 1 to 3", ("phase 3",)), ("call f(1)", ("phase 3",)), ("include lib", ("phase 3",)), ("stop", ("phase 3",)),
    ("graph g.line x", ("graphs",)), ("x.line", ("graphs",)), ("freeze(t1) x.stats", ("view",)),
    ("table(2,2) t", ("tables",)), ("spool sp", ("output",)), ("show x", ("display",)), ("logmode all", ("setting",)),
    ("shell dir", ("external",)), ("fetch(d=db) x", ("database",)), ("dbopen db", ("database",)),
    ("copy(c=a) q\\x a\\x", ("to_freq",)), ("import a.csv", ("Dataset.read",)), ("pagestruct(freq=a)", ("page",)),
    ("myaddin(x=1) y", ("add-in",)), ("series s = @mean(x) + 1", ("scalar m = @mean",)),
    ("series s = x.@count", ("phase 3",)), ("series s = @str(x)", ("phase 3",)),
    ("alpha a = \"x\"", ("text",)), ("series s = c(1)*x", ("scalar b = c(1)",)),
])
def test_lines_left_out_give_a_reason(line, words):
    t = translate_program(f"wfcreate q 2000q1 2000q4\nseries x = 1\n{line}\n")
    row = t.report.line(3)
    assert row.status == "left out", (row.status, row.reason)
    for w in words:
        assert w in row.reason


def test_values_set_inside_blocks_or_by_lines_left_out_are_not_guessed():
    t = translate_program("wfcreate q 2000q1 2000q4\n%s = \"a\"\nif 1 then\n  %s = \"b\"\nendif\nseries x{%s} = 1\n"
                          "scalar n = @obssmpl\nscalar k = n + 1\n")
    assert t.report.line(6).status == "left out" and "phase 3" in t.report.line(6).reason
    assert t.report.line(8).reason == "n is set by line 7, which is left out"


def test_statistics_and_samples_with_conditions():
    t = translate_program("wfcreate a 2001 2006\nseries x = 1\nsmpl 2002 @last\nx = x(-1) + 1\nsmpl @all if x >= 3 and not x > 5\n"
                          "scalar s = @sum(x)\nscalar n = @obs(x)\nscalar hi = @max(x)\nscalar lo = @min(x)\nscalar md = @median(x)\n"
                          "smpl @all\nseries big = x > s/4\n")
    _, out = _both(t)
    # x = 1, 2, ..., 6; the rows with 3 <= x <= 5: 3, 4, 5
    assert (out["s"], out["n"], out["hi"], out["lo"], out["md"]) == (12.0, 3.0, 5.0, 3.0, 4.0)
    np.testing.assert_array_equal(out["session"]["untitled"].frame["big"], [0, 0, 0, 1, 1, 1.0])  # x > 3
    assert t.report.line(5).python == 'untitled = untitled.with_sample(Sample(where="x >= 3 and not x > 5"))'


def test_scalars_worked_out_and_lagged_conditions_refused():
    t = translate_program("wfcreate q 2000q1 2000q4\n!a = 2\nscalar b = log(!a)*!a^2\nseries x = 1\nsmpl if x(-1) > 0\n")
    row = t.report.line(3)
    assert row.status == "translated with a note" and row.python == f"b = {math.log(2) * 4!r}"
    assert t.report.line(5).status == "left out" and "lag" in t.report.line(5).reason
