"""EViews programs translated, phase 3 (control flow): loops, if blocks, subroutines, included programs, the program's
arguments, names made from its variables, and the program's own values. Each program here is written for the test,
and every expected value is worked out by hand from its text (the comments say how), not by running the translator.
The readable script and the steps (run_steps) must give equal objects, loops, ifs and subroutines included; the
steps must be flat recorded calls.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import Session
from pymetrix_eviews import translate_program

N = 10  # annual, 2001 to 2010


def _same(a, b):
    if isinstance(a, Dataset):
        pd.testing.assert_frame_equal(a.frame, b.frame)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    elif hasattr(a, "params") and hasattr(a, "stats"):  # a fitted equation
        np.testing.assert_array_equal(a.params.values, b.params.values)
    elif hasattr(a, "names") and hasattr(a, "data"):  # a Selection
        assert list(a.names) == list(b.names)
    else:
        assert a == b, (a, b)


def _both(t, session=None, **run):
    """The script run as Python (stopping at a stop) and the steps run in a session: every object equal."""
    namespace: dict = dict(t.objects)
    try:
        exec(compile(t.script, "<script>", "exec"), namespace)
    except SystemExit:
        pass
    out = t.run(session, **run)
    s = out["session"]
    for name, value in {**{k: v for k, v in out.items() if k != "session"}, **s.objects}.items():
        assert name in namespace, f"{name} is made by the steps but not by the script"
        _same(value, namespace[name])
    return namespace, out


LOOPS = """wfcreate a 2001 2010
series t = 1
smpl 2002 @last
t = t(-1) + 1
smpl @all
!total = 0
for !i = 1 to 10 step 3
  !total = !total + !i
  series p{!i} = t * !i
next
!down = 0
for !j = 5 to 1 step -2
  !down = !down * 10 + !j
next
%names = "a b c"
for %s {%names} d
  !k = @wfind(%names, %s)
  series v_{%s} = t + !k
next
!w = 1
while !w < 100
  !w = !w * 3
wend
!count = 0
for !i = 1 to 100
  if !i > 4 then
    exitloop
  endif
  !count = !count + 1
next
if @wcount(%names) = 3 and @word(%names, 2) = "b" then
  %which = "yes"
else
  %which = "no"
endif
"""


def test_loops_their_steps_and_values_by_hand():
    t = translate_program(LOOPS)
    assert t.report.counts()["error"] == 0 and t.report.counts()["left out"] == 0
    namespace, out = _both(t)
    f = out["session"]["untitled"].frame
    tt = np.arange(1.0, 11.0)  # t = 1, then t(-1) + 1 period by period: 1 ... 10
    assert out["total"] == 1 + 4 + 7 + 10  # for !i = 1 to 10 step 3
    for i in (1, 4, 7, 10):
        np.testing.assert_array_equal(f[f"p{i}"], tt * i)
    assert "p2" not in f
    assert out["down"] == 531  # 5, 3, 1 (step -2): ((0*10 + 5)*10 + 3)*10 + 1
    for s, k in (("a", 1), ("b", 2), ("c", 3), ("d", 0)):  # @wfind: the word's position, 0 when it is not there
        np.testing.assert_array_equal(f[f"v_{s}"], tt + k)
    assert out["w"] == 243  # 1, 3, 9, 27, 81, 243: the first power of 3 not below 100
    assert out["count"] == 4  # exitloop when !i reaches 5
    assert out["which"] == "yes"
    # loops stay loops in the script
    assert "for i in range(1, 11, 3):" in t.script and "for j in range(5, 0, -2):" in t.script
    assert 'for s in ["a", "b", "c", "d"]:' in t.script  # %names is known before the loop: its words and "while w < 100:" in t.script and "break" in t.script
    assert 'untitled = untitled.with_series(**{f"p{i}": f"t*({i})"}, within=untitled.sample)' in t.script
    # the steps are flat: the page, t and its two samples (5), then one recorded call per series a loop makes (4 + 4)
    calls = [r.command for r in out["session"].history]
    assert len(calls) == 5 + 4 + 4 and all(getattr(c, "function", "").startswith("pymetrix:dataset.") for c in calls)
    assert {c.output for c in calls} == {"untitled"}


SUBS = """wfcreate q 2000q1 2001q4
series x = 1
!calls = 0
call fill("u", 5)
call fill("w", 7)
!n = 1
call addone(!n)
call addone(!n)
call addone(4)
call twice(%r, "ab")
call early(3)
subroutine fill(string %series, scalar !value)
  series {%series} = x * !value
  !calls = !calls + 1
endsub
subroutine local addone(scalar !x)
  !x = !x + 1
  series lost = 1
endsub
subroutine twice(string %out, string %in)
  %out = %in + %in
endsub
subroutine early(scalar !v)
  !before = !v
  if !v > 2 then
    return
  endif
  !after = !v
endsub
"""


def test_subroutines_arguments_by_reference_local_and_return():
    t = translate_program(SUBS)
    namespace, out = _both(t)
    f = out["session"]["untitled"].frame
    np.testing.assert_array_equal(f["u"], [5.0] * 8)  # x * 5
    np.testing.assert_array_equal(f["w"], [7.0] * 8)
    assert out["calls"] == 2  # a subroutine that is not local shares the program's !calls
    assert out["n"] == 3  # !x is !n, passed by reference: 1 + 1 + 1
    assert out["r"] == "abab"
    assert out["before"] == 3 and "after" not in out  # return before !after is set
    assert "lost" not in f  # a local subroutine's series is left out (EViews deletes it when it returns)
    row = t.report.line(18)
    assert row.status == "left out" and "local subroutine" in row.reason
    assert t.report.line(16).python == "def addone(x):" and "local" in t.report.line(16).reason
    assert t.report.line(9).python == "addone(4)" and "not a variable" in t.report.line(9).reason
    # functions first, each assigning the program's names declared global; parameters it assigns returned
    script = t.script
    assert script.index("def fill(series, value):") < script.index("untitled = Dataset.empty")
    assert "    global untitled, calls" in script and "    return x" in script and "n = addone(n)" in script
    assert 'r = twice("", "ab")' in script


def test_a_condition_on_a_result_just_made(tmp_path):
    rng = np.random.default_rng(20261124)
    x = rng.standard_normal(N)
    y = 1.0 + 2.0 * x + 0.3 * rng.standard_normal(N)
    z = rng.standard_normal(N)  # unrelated to x
    frame = pd.DataFrame({"year": [str(2001 + i) for i in range(N)], "x": x, "y": y, "z": z})
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    program = """wfopen d
%good = ""
%bad = ""
for %v y z
  equation eq_{%v}.ls {%v} c x
  scalar r2_{%v} = eq_{%v}.@r2
  if eq_{%v}.@r2 > 0.5 then
    %good = %good + " " + %v
  else
    %bad = %bad + " " + %v
  endif
next
"""
    t = translate_program(program, bundle=path, read_options={"time": "year", "freq": "Y"})
    assert t.report.counts()["error"] == 0
    namespace, out = _both(t)

    def r2(v):  # least squares of v on a constant and x, by NumPy
        X = np.column_stack([np.ones(N), x])
        e = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        return 1 - e @ e / ((v - v.mean()) @ (v - v.mean()))

    assert out["r2_y"] == pytest.approx(r2(y), abs=1e-12) and out["r2_z"] == pytest.approx(r2(z), abs=1e-12)
    assert r2(y) > 0.5 > r2(z)
    assert out["good"] == " y" and out["bad"] == " z"  # decided by each equation as it is made
    assert 'globals()[f"eq_{v}"] = Equation(f"{v} ~ x", name=f"eq_{v}").fit(d)' in t.script
    assert 'if globals()[f"eq_{v}"].stats["r2"] > 0.5:' in t.script
    assert {"eq_y", "eq_z"} <= set(out["session"].objects)


def test_an_included_program_and_the_program_arguments(tmp_path):
    (tmp_path / "lib.prg").write_text("' a library\nsubroutine mark(string %name)\n  series {%name} = 1\nendsub\n")
    main = tmp_path / "main.prg"
    main.write_text("wfcreate a 2001 2003\ninclude lib\ncall mark(%0)\nseries y_{%1} = 2\n")
    t = translate_program(main, args="first second")
    rows = [(r.file, r.line) for r in t.report]
    assert rows == [("main.prg", 1), ("main.prg", 2), ("lib.prg", 1), ("lib.prg", 2), ("lib.prg", 3), ("lib.prg", 4),
                    ("main.prg", 3), ("main.prg", 4)]
    assert t.report.line(3, file="lib.prg").python == 'untitled = untitled.with_series(**{f"{name}": "1"}, within=untitled.sample)'
    assert t.arguments == {"arg0": "first", "arg1": "second"}
    assert t.script.count('arg0 = "first"') == 1 and "mark(arg0)" in t.script
    namespace, out = _both(t)
    assert list(out["session"]["untitled"].frame.columns) == ["first", "y_second"]
    again = t.run(args=["one", "two"])  # other arguments: other names, the same translation
    assert list(again["session"]["untitled"].frame.columns) == ["one", "y_two"]
    missing = translate_program("include nothere\n")
    assert missing.report.line(1).status == "error" and "not found" in missing.report.line(1).reason


def test_stop_while_cap_and_values_of_the_data():
    program = """wfcreate a 2001 2004
series p1 = 1
series p2 = 2
series q = 3
%found = @wlookup("p*")
!there = @isobject("q")
!absent = @isobject("nothing")
scalar m = @sum(p2)
if m > 5 then
  stop
endif
series after = 1
"""
    t = translate_program(program)
    namespace, out = _both(t)
    assert out["found"] == "p1 p2" and (out["there"], out["absent"]) == (1.0, 0.0)
    assert out["m"] == 8.0  # 2 in each of the 4 years
    assert "after" not in out["session"]["untitled"].frame  # stopped before
    looping = translate_program("!i = 0\nwhile 1\n  !i = !i + 1\nwend\n")
    with pytest.raises(Exception, match="line 2 .*50 passes"):
        looping.run(while_cap=50)


@pytest.mark.parametrize("program, line, status, words", [
    ("exitloop\n", 1, "error", ("outside a loop",)),
    ("for !i = 1 to 2\n", 1, "error", ("has no next",)),
    ("next\n", 1, "error", ("without its for",)),
    ("call nothing(1)\n", 1, "error", ("no subroutine nothing",)),
    ("subroutine f(series s)\nendsub\ncall f(x)\n", 3, "left out", ("object arguments", "gap")),
    ("%d = @datestr(@dateval(\"2001q1\"))\n", 1, "left out", ("default format", "give the format")),
    ("!d = @dateadd(@dateval(\"2001q1\"), 1, \"bd\")\n", 1, "left out", ("unit",)),
    ("wfcreate q 2001q1 2001q4\n%d = @strdate(\"yyyy\")\n", 2, "left out", ("alpha series",)),
    ("for !i = 1 to 2\n  wfcreate a 2001 2002\nnext\n", 2, "left out", ("inside a loop",)),
    ("wfcreate a 2001 2002\nseries s = {!i+1}\n", 2, "left out", ("expression in braces",)),
    ("wfcreate a 2001 2002\nmatrix _$_m\n", 2, "left out", ("$",)),
])
def test_lines_left_out_or_in_error_say_why(program, line, status, words):
    row = translate_program(program).report.line(line)
    assert row.status == status and all(w in row.reason for w in words), row.reason


def test_steps_through_a_runner_stop_at_the_first_failure_with_its_line():
    t = translate_program("wfcreate a 2001 2003\nfor !i = 1 to 3\n  series s{!i} = q{!i} * 2\nnext\n")  # no series q1
    seen = []
    s = Session()
    with pytest.raises(Exception) as caught:
        t.run(s, run=lambda c: (seen.append(c), s.run(c)))
    assert "line 3" in str(caught.value) and caught.value.line == 3
    assert len(seen) == 2 and list(s["untitled"].frame.columns) == []  # the page made; the first series failed


# -- EViews's date numbers ------------------------------------------------------------------------------------------

# Days from 1 January A.D. 1 (date number 0) to 1 January 2001, by hand: 2000 years of 365 days and the leap days in
# years 1 to 2000 (2000 // 4 - 2000 // 100 + 2000 // 400 = 485); 1 July 2001 is 181 days later (31 + 28 + 31 + 30 +
# 31 + 30, 2001 not a leap year).
JAN_2001 = 365 * 2000 + 485
JUL_2001 = JAN_2001 + 181


def test_date_numbers_by_hand_and_the_eviews_example():
    from pymetrix_eviews import dates

    assert dates.dateval("0001-01-01") == 0 and dates.dateval("1/2/0001") == 1  # the first day is 0
    assert dates.dateval("12/1/1999", "dd/mm/yyyy") == 729765  # EViews's Command Reference: 12 January 1999
    assert dates.dateval("2001") == JAN_2001 and dates.dateval("2001-07-01") == JUL_2001
    assert dates.dateval("2001q3") == dates.dateval("2001Q3") == dates.dateval("2001m7") == JUL_2001  # a period's first day
    assert dates.dateval("2001q3", "yyyyfq") == dates.dateval("2001:3", "yyyy:q") == dates.dateval("01Q3", "yy[Q]q") == JUL_2001
    assert dates.datestr(JUL_2001, "yyyyfq") == "2001q3" and dates.datestr(JUL_2001, "YYYYFQ") == "2001Q3"
    assert dates.datestr(JUL_2001, "yyyy[Q]q") == "2001Q3" and dates.datestr(JUL_2001 + 14, "dd/mm/yyyy") == "15/07/2001"
    assert dates.datestr(JUL_2001, "yyyy-mm-dd") == "2001-07-01" and dates.datestr(JUL_2001, "yyyyfm") == "2001m7"
    # @dateadd: whole periods (the day kept, or the month's last), weeks and days
    assert dates.dateadd(JUL_2001, 2, "q") == JAN_2001 + 365  # 2002q1
    assert dates.dateadd(JUL_2001, -3, "q") == dates.dateval("2000q4") == JAN_2001 - 92  # October to December: 92 days
    assert dates.dateadd(JAN_2001 + 30, 1, "m") == JAN_2001 + 31 + 27  # 31 January + 1 month: 28 February
    assert dates.dateadd(JUL_2001, 1, "a") == JUL_2001 + 365 and dates.dateadd(JUL_2001, 2, "w") == JUL_2001 + 14
    assert dates.datestr(dates.dateadd(dates.dateval("2020q1", "yyyyfq"), 23, "q"), "yyyyfq") == "2025q4"  # FRB/US's dateshift
    # @datediff: how far the first is after the second
    assert dates.datediff(dates.dateval("2001q1"), dates.dateval("1999q3"), "q") == 6
    assert dates.datediff(dates.dateval("1999q3"), dates.dateval("2001q1"), "q") == -6
    assert dates.datediff(JUL_2001, JAN_2001, "d") == 181 and dates.datediff(JUL_2001, JAN_2001, "w") == 25  # 181 // 7
    assert dates.datediff(JUL_2001, JAN_2001, "a") == 0 and dates.datediff(JUL_2001, JAN_2001, "m") == 6
    with pytest.raises(ValueError, match="does not read"):
        dates.dateval("2001-07", "yyyyfq")


DATES = """wfcreate q 2000q1 2030q4
%simstart = "2020q1"
!nsimqtrs = 24
call dateshift(%simstart, %simend, !nsimqtrs - 1)
smpl {%simstart} {%simend}
series x = 1
smpl @all
!d = @dateval("12/1/1999", "dd/mm/yyyy")
%s = @datestr(@dateadd(@dateval(%simstart), 59, "q"), "yyyy[Q]q")
!k = @datediff(@dateval("2001q1"), @dateval("1999q3"), "q")
!lqtr = @dtoo(%simstart) - 1
%o = @otod(3)

subroutine dateshift(string %indate, string %outdate, scalar qtrshift)
  !dddd1 = @dateval(%indate, "yyyyfq")
  !dddd2 = @dateadd(!dddd1, qtrshift, "q")
  %outdate = @datestr(!dddd2, "yyyyfq")
endsub
"""


def test_date_functions_in_a_program_and_frbus_dateshift():
    """FRB/US's dateshift subroutine (subs/master_library.prg, lines 4-13) as the package writes it, called as
    example1.prg calls it; the other functions on values known before the program runs. By hand: 2020q1 and 23
    quarters is 2025q4; 2020q1 and 59 quarters, 2034q4; 1999q3 to 2001q1, 6 quarters; 2020q1 is the 81st quarter
    from 2000q1; the third observation is 2000q3."""
    t = translate_program(DATES)
    assert t.report.counts()["left out"] == 0 and t.report.counts()["error"] == 0, t.report.to_text()
    assert "import pymetrix_eviews.dates as eviews_dates" in t.script
    assert "days since 1 January A.D. 1" in t.report.line(15).reason
    namespace, out = _both(t)
    for values in (namespace, out):
        assert values["simend"] == "2025q4" and values["s"] == "2034Q4" and values["k"] == 6 and values["d"] == 729765
        assert values["lqtr"] == 80 and values["o"] == "2000Q3"
    x = out["session"]["untitled"].frame["x"]
    assert x.notna().sum() == 24 and str(x.first_valid_index()) == "2020Q1" and str(x.last_valid_index()) == "2025Q4"
