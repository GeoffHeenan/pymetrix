"""Exercise packs (``pymetrix.exercises``): the format and its validation, value paths that never call, checks that
pass on the reference solution and fail on a wrong one, packs from plugins' entry points, the command-line checker's
exit codes, and the Exercise panel.

The fixture pack (tests/fixtures/exercise-pack) has two exercises on pymetrix's example datasets: a wage equation on
mroz with White standard errors, and an ADF test on the Nile. Its expected numbers are checked here against other
implementations: statsmodels' OLS with HC1 standard errors, and the ADF regression fitted by hand with NumPy."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pymetrix import Dataset, Equation, OLS, exercises as ex, plugins
from pymetrix.commands import Session
from pymetrix.errors import NotFoundError, PyMetrixError, SpecificationError
from pymetrix.exercises.cli import main, new_pack, run_exercise

PACK = Path(__file__).parent / "fixtures" / "exercise-pack"


@pytest.fixture
def pack():
    return ex.load_pack(PACK)


@pytest.fixture
def copy(tmp_path):
    """A copy of the fixture pack to break."""
    return Path(shutil.copytree(PACK, tmp_path / "pack"))


# -- the format -------------------------------------------------------------------------------------------------------


def test_the_fixture_pack_loads_with_its_exercises_steps_hints_and_checks(pack):
    assert (pack.name, pack.version, pack.requires_pymetrix, pack.audience) == ("test-pack", "1.0", ">=0.1.0a16", "the test suite")
    assert pack.compatible and pack.source == "folder" and pack.data == {} and pack.classification == "public"
    assert sorted(pack.models) == ["ar1", "nk", "rbc"] and pack.exercise(1).models == ()
    assert [e.id for e in pack.exercises] == ["01-ols-mroz", "02-unit-root-nile"]
    ols = pack.exercise("ols-mroz")
    assert pack.exercise(1) is ols and pack.exercise("01-ols-mroz") is ols and ols.pack is pack
    assert (ols.title, ols.level, ols.minutes, ols.data) == ("A wage equation with robust standard errors", "beginner", 15, ("mroz",))
    assert [s.title for s in ols.steps] == ["Fit the wage equation", "Make the standard errors robust"]
    first = ols.steps[0]
    assert first.code == 'fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)'
    assert len(first.hints) == 2 and first.hints[1].startswith("`expersq` is experience squared")
    assert "Estimate → Equation…" in first.text and "```" not in first.text and "Hint" not in first.text
    assert ols.intro.startswith("Married women's wages") and "# A wage" not in ols.intro  # the title line dropped
    assert ols.closing.startswith("## What next")
    assert [c.id for c in ols.checks_for(2)] == ["robust", "robust-se", "robust-p"]
    assert ols.solution == PACK.resolve() / "exercises" / "01-ols-mroz" / "solution.py"
    nile = pack.exercise(2)
    assert nile.solution is None and nile.checks[1].value == "details.lags used" and nile.checks[1].expected == 1
    with pytest.raises(NotFoundError, match="has no exercise 'three'"):
        pack.exercise("three")
    with pytest.raises(NotFoundError, match="no step 3"):
        ols.step(3)


def _problems(path):
    with pytest.raises(SpecificationError) as info:
        ex.load_pack(path)
    assert isinstance(info.value, PyMetrixError) and isinstance(info.value, ValueError)
    return info.value.problems


def test_a_bad_pack_lists_every_problem_with_its_file(copy):
    (copy / "pack.toml").write_text('[pack]\nname = "Bad Name"\ntitle = "T"\nversion = "1"\nrequires_pymetrix = "at least 1"\n'
                                    'colour = "red"\n[data.ghost]\ntime = "t"\n', encoding="utf-8")
    ols = copy / "exercises" / "01-ols-mroz"
    md = (ols / "exercise.md").read_text(encoding="utf-8")
    md = md.replace("data: [mroz]", "data: [mroz, nowhere]\nmodels: [ghost]\nauthor: me").replace("level: beginner", "level: easy").replace("## Step 2: Make", "## Step 3: Make")
    md = md.replace("```python\nfit = Equation", "```python\nx = 1\n```\n\n```python\nfit = Equation")
    (ols / "exercise.md").write_text(md, encoding="utf-8")
    checks = (ols / "checks.toml").read_text(encoding="utf-8")
    checks = checks.replace('id = "r2"', 'id = "educ"').replace('value = "method.cov"', 'value = "method.__class__"')
    checks += ('\n[[check]]\nid = "x"\nstep = 9\nobject = "fit white"\nvalue = "a..b"\nexpected = [1]\nabs_tol = -1\nunit = "%"\n'
               '\n[[check]]\nid = "t"\nstep = 1\nobject = "fit"\nvalue = "nobs"\nexpected = 1\nabs_tol = -1\n'
               '\n[[check]]\nid = "s"\nstep = 1\nobject = "fit"\nvalue = "method.cov"\nexpected = "white"\nrel_tol = 0.1\n')
    (ols / "checks.toml").write_text(checks, encoding="utf-8")
    (copy / "exercises" / "3-bad name").mkdir()
    (copy / "data").mkdir()
    (copy / "data" / "notes.xlsx").write_text("x", encoding="utf-8")
    (copy / "models" / "old.txt").write_text("x", encoding="utf-8")
    found = "\n".join(_problems(copy))
    for expected in ("pack.toml: [pack] needs 'description'", "pack.toml: unknown key 'colour' in [pack]",
                     "pack.toml: [pack] name 'Bad Name' should be lower case", "requires_pymetrix 'at least 1' is not a version specifier",
                     "data/notes.xlsx: is not a data file", "exercises/3-bad name: is not an exercise folder",
                     "01-ols-mroz/exercise.md: unknown front-matter key 'author'", "data names 'nowhere'",
                     "step 3 should be step 2", "step 1 has 2 python blocks",
                     "check educ: two checks have the id 'educ'", "check robust: value 'method.__class__' names '__class__'",
                     "check x: unknown key 'unit'", "check x: step should be the number of a step of the exercise (1 to 2), not 9",
                     "check x: object should be the name of a session object, not 'fit white'", "check x: value 'a..b' has an empty part",
                     "check x: expected should be a number, a string or true/false", "check x: abs_tol is for numbers",
                     "check s: rel_tol is for numbers", "models/old.txt: is not a model file (.mod, .model, .pymodel)",
                     "models names 'ghost', which is not a file of the pack's models/",
                     "level should be beginner, intermediate or advanced, not 'easy'", "pack.toml: [data.ghost] names no file of data/", "check t: abs_tol should be a number at least 0"):
        assert expected in found, expected


@pytest.mark.parametrize("change, expected", [
    (lambda p: (p / "pack.toml").unlink(), "pack.toml: missing"),
    (lambda p: (p / "pack.toml").write_text("[pack\n", encoding="utf-8"), "pack.toml: is not valid TOML"),
    (lambda p: shutil.rmtree(p / "exercises"), "exercises: missing"),
    (lambda p: (p / "exercises" / "02-unit-root-nile" / "exercise.md").write_text("No front matter.\n", encoding="utf-8"),
     "needs front matter"),
    (lambda p: (p / "exercises" / "02-unit-root-nile" / "exercise.md").write_text("---\ntitle: T\n---\n\nText only.\n", encoding="utf-8"),
     "has no steps"),
    (lambda p: (p / "exercises" / "02-unit-root-nile" / "exercise.md").write_text("---\ntitle: T\n---\n\n## Step one\n", encoding="utf-8"),
     "'## Step one' should read '## Step N: <title>'"),
    (lambda p: (p / "exercises" / "02-unit-root-nile" / "checks.toml").write_text('check = 1\n', encoding="utf-8"),
     "checks are written [[check]]"),
    (lambda p: shutil.copytree(p / "exercises" / "02-unit-root-nile", p / "exercises" / "02-again"), "has the number of"),
])
def test_bad_packs_say_what_is_wrong(copy, change, expected):
    change(copy)
    assert any(expected in p for p in _problems(copy)), _problems(copy)


def test_a_pack_that_is_not_there_is_not_found(tmp_path):
    with pytest.raises(NotFoundError, match="no exercise pack at"):
        ex.load_pack(tmp_path / "nowhere")


def test_a_pack_for_a_later_pymetrix_loads_and_says_so(copy):
    text = (copy / "pack.toml").read_text(encoding="utf-8").replace(">=0.1.0a16", ">=99")
    (copy / "pack.toml").write_text(text, encoding="utf-8")
    pack = ex.load_pack(copy)
    assert not pack.compatible and "needs pymetrix >=99" in pack.requirement_problem
    assert main(["check", str(copy)]) == 2


def test_headings_in_code_blocks_are_not_steps(copy):
    md = copy / "exercises" / "02-unit-root-nile" / "exercise.md"
    md.write_text(md.read_text(encoding="utf-8").replace(
        'adf = nile.test("adf", "volume")', 'adf = nile.test("adf", "volume")\n## Step 9: not a heading'), encoding="utf-8")
    step = ex.load_pack(copy).exercise(2).steps[0]
    assert "## Step 9: not a heading" in step.code


# -- data -------------------------------------------------------------------------------------------------------------


def test_data_come_from_the_pack_or_the_example_datasets(copy):
    (copy / "data").mkdir()
    Dataset.example("nile").write(str(copy / "data" / "flows.parquet"))
    import pandas as pd

    pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.5]}).to_csv(copy / "data" / "small.csv", index=False)
    assert any("a CSV file needs [data.small] in pack.toml" in p for p in _problems(copy))
    with open(copy / "pack.toml", "a", encoding="utf-8") as f:
        f.write('\n[data.small]\nentity = "id"\n')
    md = copy / "exercises" / "02-unit-root-nile" / "exercise.md"
    md.write_text(md.read_text(encoding="utf-8").replace("data: [nile]", "data: [flows, small, mroz]"), encoding="utf-8")
    pack = ex.load_pack(copy)
    assert set(pack.data) == {"flows", "small"}
    flows = ex.load_data(pack, "flows")
    assert np.allclose(flows["volume"].to_numpy(float), Dataset.example("nile")["volume"].to_numpy(float))
    assert list(ex.load_data(pack, "small")["y"]) == [2.0, 4.0, 6.5]
    assert ex.load_data(pack, "mroz").names == Dataset.example("mroz").names
    with pytest.raises(NotFoundError, match="no dataset 'other'"):
        ex.load_data(pack, "other")
    s = Session()
    for cmd in ex.data_commands(pack.exercise(2)):
        s.run(cmd)
    assert {"flows", "small", "mroz"} <= set(s.objects)
    assert "dataset.read" in s.script() or "Dataset.read" in s.script()
    assert 'Dataset.example("mroz")' in s.script() or "Dataset.example('mroz')" in s.script()


# -- value paths --------------------------------------------------------------------------------------------------------


class _Thing:
    def __init__(self):
        self.ok = {"a": [10, 20, {"deep": 1.5}]}
        self.ran = []

    def method(self):
        self.ran.append("method")
        return 1

    @property
    def prop(self):
        return np.float64(2.5)


def test_values_are_read_by_attribute_and_key_only():
    t = _Thing()
    assert ex.resolve(t, "ok.a.1") == 20 and ex.resolve(t, "ok.a.2.deep") == 1.5 and ex.resolve(t, "prop") == 2.5
    for path in ("method", "ok.keys", "ok.a.append", "prop.mean"):
        with pytest.raises(SpecificationError, match="is a method: checks read values and never call"):
            ex.resolve(t, path)
    for path in ("__class__", "ok.__class__.__init__", "__dict__", "ok.__getitem__", "_Thing__x.__x"):
        with pytest.raises(SpecificationError, match="names starting with __ are not followed"):
            ex.resolve(t, path)
    for path in ("", "ok..a", ".ok", "ok."):
        with pytest.raises(SpecificationError):
            ex.resolve(t, path)
    with pytest.raises(NotFoundError, match="ok.a has no attribute or key 'b'"):
        ex.resolve(t, "ok.a.b")
    assert t.ran == []  # nothing was called


def test_value_paths_into_estimates_and_tests():
    data = Dataset.example("mroz")
    fit = Equation("lwage ~ educ + exper + expersq").fit(data, method=OLS(cov="white"))
    assert ex.resolve(fit, "params.educ") == pytest.approx(fit.params["educ"])
    assert ex.resolve(fit, "stats.r2") == fit.stats["r2"] and ex.resolve(fit, "nobs") == 428
    table = fit.coefficients().set_index("term")
    assert ex.resolve(fit, "bse.educ") == pytest.approx(table.loc["educ", "std_error"])
    assert ex.resolve(fit, "tvalues.exper") == pytest.approx(table.loc["exper", "statistic"])
    assert ex.resolve(fit, "pvalues.const") == pytest.approx(table.loc["const", "p_value"])
    assert ex.resolve(fit, "method.cov") == "white"
    with pytest.raises(SpecificationError, match="is a method"):
        ex.resolve(fit, "summary")
    with pytest.raises(NotFoundError):
        ex.resolve(fit, "bse.nothing")
    import pandas as pd

    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"])
    assert ex.resolve({"f": frame}, "f.x.b") == 2.0
    with pytest.raises(SpecificationError, match="is a method"):
        ex.resolve({"f": frame}, "f.x.mean")


def test_a_whole_number_part_is_a_string_key_then_an_int_key_then_a_position():
    import pandas as pd

    by_year = pd.Series([1.5, 2.5, 3.5], index=[2000, 2001, 2002])  # an integer index: "2001" is the label 2001
    assert ex.resolve(by_year, "2001") == by_year.loc[2001] == 2.5
    assert ex.resolve({"s": by_year}, "s.2002") == 3.5 and ex.resolve({"s": by_year}, "s.values.0") == 1.5
    with pytest.raises(NotFoundError, match="has no attribute or key '1'"):
        ex.resolve(by_year, "1")  # a label, never a position: 1 is not a year of the index
    horizons = pd.Series([0.0, 0.7, 0.49], index=pd.RangeIndex(3))  # responses by horizon
    assert ex.resolve(horizons, "2") == 0.49 and ex.resolve(pd.DataFrame({0: [1.0, 2.0]}), "0.1") == 2.0
    assert ex.resolve(pd.Series([1.0, 2.0], index=["1", 1]), "1") == 1.0  # the string label comes first
    assert ex.resolve(pd.Series([1.0, 2.0], index=["a", "b"]), "b") == 2.0  # labels as before
    with pytest.raises(NotFoundError):
        ex.resolve(pd.Series([1.0, 2.0], index=["a", "b"]), "1")  # string labels: no position either

    assert ex.resolve([10, 20, 30], "2") == 30 and ex.resolve((10, 20), "1") == 20 and ex.resolve([10, 20], "-1") == 20
    assert ex.resolve(np.array([[1.0, 2.0], [3.0, 4.0]]), "1.0") == 3.0 and ex.resolve(np.arange(5), "4") == 4
    with pytest.raises(NotFoundError, match="has no attribute or key '3'"):
        ex.resolve([10, 20, 30], "3")
    assert ex.resolve({1: "int", 2: "two"}, "1") == "int" and ex.resolve({"1": "str", 1: "int"}, "1") == "str"
    assert ex.resolve({"d": {7: {"x": 1.25}}}, "d.7.x") == 1.25
    # still no calls and no dunders, whatever the keys
    for obj, path in ((by_year, "2001.__class__"), ({1: by_year}, "1.__len__")):
        with pytest.raises(SpecificationError, match="names starting with __"):
            ex.resolve(obj, path)
    with pytest.raises(SpecificationError, match="is a method"):
        ex.resolve({1: by_year}, "1.mean")
    # and through a check, as a pack writes it
    exercise = ex.Exercise("01-years", "Years", (ex.Step(1, "One", ""),),
                           (ex.Check("y", 1, "gdp", "2001", 2.5), ex.Check("r", 1, "resp", "1", 0.7, abs_tol=1e-12)), PACK)
    results = ex.check(exercise, {"gdp": by_year, "resp": horizons})
    assert [r.passed for r in results] == [True, True] and results[0].actual == 2.5


# -- checks -------------------------------------------------------------------------------------------------------------


def test_the_fixture_numbers_agree_with_other_implementations(pack):
    import statsmodels.api as sm

    frame = Dataset.example("mroz").frame[["lwage", "educ", "exper", "expersq"]].dropna()
    X = sm.add_constant(frame[["educ", "exper", "expersq"]])
    ols = sm.OLS(frame["lwage"], X).fit(cov_type="HC1")
    checks = {c.id: c for c in pack.exercise(1).checks}
    assert int(ols.nobs) == checks["sample"].expected
    assert abs(ols.params["educ"] - checks["educ"].expected) <= checks["educ"].abs_tol
    assert abs(ols.rsquared - checks["r2"].expected) <= checks["r2"].abs_tol
    assert ols.bse["educ"] == pytest.approx(checks["robust-se"].expected, rel=checks["robust-se"].rel_tol)
    # the ADF regression by hand: diff(y) on a constant, y[-1] and diff(y)[-1] (AIC chose one lag): t on y[-1]
    y = Dataset.example("nile")["volume"].to_numpy(float)
    dy = np.diff(y)
    for trend, cid in (("c", "statistic"), ("ct", "trend")):
        rows = np.column_stack([np.ones(len(dy) - 1), y[1:-1], dy[:-1]] + ([np.arange(len(dy) - 1)] if trend == "ct" else []))
        lhs = dy[1:]
        beta, *_ = np.linalg.lstsq(rows, lhs, rcond=None)
        resid = lhs - rows @ beta
        s2 = resid @ resid / (len(lhs) - rows.shape[1])
        t = beta[1] / np.sqrt(s2 * np.linalg.inv(rows.T @ rows)[1, 1])
        check_ = next(c for c in pack.exercise(2).checks if c.id == cid)
        assert abs(t - check_.expected) <= check_.abs_tol, (trend, t)


def test_checks_pass_on_the_reference_solution_and_fail_on_a_wrong_one(pack):
    ols = pack.exercise(1)
    session, results, error = run_exercise(ols)
    assert error == "" and results and all(r.passed for r in results), [r.detail for r in results if not r.passed]
    step1 = ex.check(ols, session, step=1)
    assert [r.id for r in step1] == ["sample", "educ", "r2"]
    r = step1[0]
    assert (r.actual, r.expected, r.step) == (428, 428, 1) and isinstance(r.actual, int)
    # a wrong answer: no robust standard errors, and a regressor left out
    wrong = Session()
    for cmd in ex.data_commands(ols):
        wrong.run(cmd)
    wrong.run_code('fit = Equation("lwage ~ educ + exper").fit(mroz)\nfit_white = Equation("lwage ~ educ + exper + expersq").fit(mroz)')
    by_id = {r.id: r for r in ex.check(ols, wrong)}
    assert by_id["sample"].passed and not by_id["educ"].passed and not by_id["r2"].passed
    assert not by_id["robust"].passed and by_id["robust"].actual == "classic"
    assert by_id["robust"].message == "Choose Standard errors: White (HC1)."
    assert not by_id["robust-se"].passed and by_id["robust-se"].actual == pytest.approx(0.014146, rel=1e-3)
    assert by_id["robust-p"].passed  # education is significant either way
    # nothing yet: each check fails, saying why
    empty = ex.check(ols, Session(), step=2)
    assert not any(r.passed for r in empty) and empty[0].detail == "no object named 'fit_white' yet" and empty[0].actual is None
    assert [r.passed for r in ex.check(ols, {"fit": session["fit"]}, step=1)] == [True, True, True]  # a plain mapping too
    with pytest.raises(NotFoundError):
        ex.check(ols, session, step=5)


def test_comparisons_follow_the_expected_type_and_tolerances():
    def one(expected, actual, **tol):
        c = ex.Check("c", 1, "o", "v", expected, tol.get("abs_tol"), tol.get("rel_tol"), "")
        steps = (ex.Step(1, "s", ""),)
        exercise = ex.Exercise("01-x", "X", steps, (c,), Path("."))
        return ex.check(exercise, {"o": {"v": actual}})[0].passed

    assert one(1.0, 1.0000001) and not one(1.0, 1.001)  # the default: 6 significant digits
    assert one(1.0, 1.04, abs_tol=0.05) and not one(1.0, 1.06, abs_tol=0.05)
    assert one(100, 101, rel_tol=0.02) and not one(100, 103, rel_tol=0.02)
    assert one(3, np.int64(3)) and not one(3, float("nan")) and not one(3, "3")
    assert one(True, np.bool_(True)) and not one(True, 1) and not one(False, True)
    assert one("white", "white") and not one("white", "White") and not one("1", 1)


# -- plugins' packs -----------------------------------------------------------------------------------------------------


class _Dist:
    def __init__(self, name, version="1.0"):
        self.name, self.version, self.metadata = name, version, {}


class _EP:
    def __init__(self, name, value, dist, obj):
        self.name, self.value, self.dist, self._obj = name, value, dist, obj
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._obj


@pytest.fixture
def plugin_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugins, "_STATE", None)
    monkeypatch.setattr(plugins, "_SKIPPED", set())
    yield
    plugins._STATE = None


def test_plugins_ship_packs_through_entry_points(monkeypatch, plugin_state, tmp_path):
    good = _EP("tests", "pymetrix_testpack:folder", _Dist("pymetrix-testpack"), lambda: PACK)
    broken = _EP("broken", "pymetrix_broken:folder", _Dist("pymetrix-broken"), lambda: tmp_path / "missing")
    shy = _EP("new", "pymetrix_new:folder", _Dist("pymetrix-new"), lambda: PACK)
    eps = {"pymetrix.exercises": [good, broken, shy]}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: eps.get(group, []))
    plugins.enable("pymetrix-testpack", "1.0")
    plugins.enable("pymetrix-broken", "1.0")
    plugins.set_mode("app")  # the app: a new plugin waits for the user's answer, its code not run
    try:
        found = ex.installed_packs()
        assert [(p.name, p.source) for p in found] == [("test-pack", "pymetrix-testpack")]
        assert found[0].exercises[0].title == "A wage equation with robust standard errors"
        assert "pymetrix-broken: broken" in ex.problems() and "no exercise pack at" in ex.problems()["pymetrix-broken: broken"]
        assert not shy.loaded
        assert [p.name for p in plugins.pending()] == ["pymetrix-new"] and plugins.pending()[0].groups == ("exercises",)
        plugins.disable("pymetrix-testpack")
        assert ex.installed_packs() == []
    finally:
        plugins.set_mode("script")
    with pytest.warns(UserWarning, match="pymetrix-new is new"):
        assert main(["list"]) == 0


def test_the_checker_finds_an_installed_pack_by_name(monkeypatch, plugin_state, capsys):
    eps = {"pymetrix.exercises": [_EP("tests", "pymetrix_testpack:folder", _Dist("pymetrix-testpack"), lambda: str(PACK))]}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: eps.get(group, []))
    plugins.enable("pymetrix-testpack", "1.0")
    assert main(["check", "test-pack", "--exercise", "unit-root-nile"]) == 0
    out = capsys.readouterr().out
    assert "02-unit-root-nile" in out and "01-ols-mroz" not in out and "(the steps' code)" in out


# -- the command line ----------------------------------------------------------------------------------------------------


def test_the_checker_exits_zero_when_all_pass_and_one_when_a_check_fails(copy, capsys):
    assert main(["check", str(copy)]) == 0
    out = capsys.readouterr().out
    assert "All checks passed: 0 problems in 2 exercises with 9 checks." in out and "Classification: public" in out
    (copy / "exercises" / "01-ols-mroz" / "solution.py").write_text(
        'fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)\nfit_white = fit\n', encoding="utf-8")
    assert main(["check", str(copy), "--exercise", "1"]) == 1
    out = capsys.readouterr().out
    assert "FAIL step 2 robust" in out and "Choose Standard errors: White (HC1)." in out
    (copy / "exercises" / "01-ols-mroz" / "solution.py").write_text("raise RuntimeError('drift')\n", encoding="utf-8")
    assert main(["check", str(copy), "--exercise", "01-ols-mroz"]) == 1
    assert "RuntimeError: drift" in capsys.readouterr().out


def test_the_checker_exits_two_when_the_pack_cannot_be_used(copy, tmp_path, capsys):
    assert main(["check", str(tmp_path / "nowhere")]) == 2
    assert main(["check", str(copy), "--exercise", "nine"]) == 2
    (copy / "pack.toml").write_text("[pack]\n", encoding="utf-8")
    assert main(["check", str(copy)]) == 2
    assert "[pack] needs 'name'" in capsys.readouterr().err


def test_new_scaffolds_a_pack_that_passes_its_checks(tmp_path, capsys):
    folder = tmp_path / "my-pack"
    assert main(["new", str(folder), "--title", "My pack"]) == 0
    pack = ex.load_pack(folder)
    assert (pack.name, pack.title) == ("my-pack", "My pack") and len(pack.exercises) == 1
    assert pack.compatible and (folder / "README.md").exists()
    assert main(["check", str(folder)]) == 0
    assert main(["new", str(folder)]) == 2  # not over a pack
    with pytest.raises(FileExistsError):
        new_pack(folder)


def test_the_module_runs_as_a_program(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(ex.__file__).parents[2]), os.environ.get("PYTHONPATH", "")]))
    bad = subprocess.run([sys.executable, "-m", "pymetrix.exercises", "check", str(tmp_path / "nowhere")],
                         capture_output=True, text=True, env=env, timeout=120)
    assert bad.returncode == 2 and "is neither a folder nor the name of an installed exercise pack" in bad.stderr


# -- help --------------------------------------------------------------------------------------------------------------


def test_the_help_has_a_topic_on_exercises():
    from pymetrix import help as hp

    page = hp.page_for_link("pymetrix:topic/exercises")
    assert "Help → Exercises…" in page and "pymetrix-exercises check" in page


def test_dataset_values_by_series_and_period():
    assert ex.resolve(Dataset.example("nile"), "volume.1871") == 1120.0
    assert ex.resolve(Dataset.example("sample_macro"), "y.2000Q1") == pytest.approx(Dataset.example("sample_macro")["y"].iloc[0])


def test_the_guides_example_runs_as_written(tmp_path, monkeypatch):
    guide = (Path(__file__).resolve().parents[1] / "docs" / "guide.md").read_text(encoding="utf-8")
    section = guide.split("## Writing an exercise pack", 1)[1].split("\n## Using pymetrix from an AI", 1)[0]
    code = section.split("In code:\n\n```python\n", 1)[1].split("```", 1)[0]
    new_pack(tmp_path / "my-pack")
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(code.replace("[(r.id", "results = [(r.id"), "guide", "exec"), namespace)  # noqa: S102 - the guide's own code
    assert namespace["results"] == [("sample", True, 100)]
    spec = (Path(__file__).resolve().parents[1] / "docs" / "exercise-packs.md").read_text(encoding="utf-8")
    for key in ("name", "title", "version", "description", "requires_pymetrix", "audience"):
        assert f"\n{key} = " in spec  # every pack.toml key is in the reference
    assert set(ex._CHECK_KEYS) <= {k for k in ex._CHECK_KEYS if f"\n{k} = " in spec}


# -- models, where solutions run, and the documented paths --------------------------------------------------------------


def test_models_load_by_their_extension_as_the_import_dialogs_read_them(pack):
    from pymetrix.commands import DefineModel
    from pymetrix.commands.builtin import ImportDynare, ImportIris
    from pymetrix.models.dynare import read_mod
    from pymetrix.models.iris import read_iris

    rbc, nk, ar1 = (ex.load_model(pack, n) for n in ("rbc", "nk", "ar1"))
    assert type(rbc).__name__ == "DynareModel" and set(rbc.model.endogenous) == {"c", "k", "y", "z"}
    assert type(nk).__name__ == "Template" and type(ar1).__name__ == "Template"
    assert set(read_mod(pack.models["rbc"]).model.endogenous) == set(rbc.model.endogenous)
    assert set(read_iris(pack.models["nk"]).model.endogenous) == set(nk.model.endogenous)
    assert ex.load_model(PACK, "ar1").model.endogenous == ar1.model.endogenous == ("y",)  # a folder works as the pack
    for name, cls, session_name in (("rbc", ImportDynare, "rbc"), ("nk", ImportIris, "nk"), ("ar1", DefineModel, "gap")):
        cmd = ex.model_command(pack, name)  # ar1.pymodel is imported as gap: [models.ar1] name = "gap" in pack.toml
        assert type(cmd) is cls and cmd.name == session_name == pack.model_name(name)
        s = Session()
        s.run(cmd)  # the step the panel's Import… opens: recorded, and it gives the same model
        assert type(s[session_name]).__name__ == type(ex.load_model(pack, name)).__name__
        assert (name in s.objects) == (name == session_name)
    with pytest.raises(NotFoundError, match="no model 'ghost'"):
        ex.load_model(pack, "ghost")
    with pytest.raises(NotFoundError, match="no model 'ghost'"):
        ex.model_command(pack, "ghost")


def test_pack_toml_names_the_session_object_a_model_is_imported_as(copy):
    """[models.<file stem>] name = "...": the panel's Import…, model_command and load_model use it; the stem by default."""
    from pymetrix.commands.builtin import ImportIris

    (copy / "pack.toml").write_text((copy / "pack.toml").read_text(encoding="utf-8")
                                    + '\n[models.nk]\nname = "policy_model"\n', encoding="utf-8")
    md = copy / "exercises" / "02-unit-root-nile" / "exercise.md"
    md.write_text(md.read_text(encoding="utf-8").replace("data: [nile]", "data: [nile]\nmodels: [policy_model, rbc, gap]"),
                  encoding="utf-8")
    pack = ex.load_pack(copy)
    assert pack.model_names == {"ar1": "gap", "nk": "policy_model"}
    assert [pack.model_name(m) for m in ("rbc", "nk", "ar1", "policy_model", "gap")] == ["rbc", "policy_model", "gap", "policy_model", "gap"]
    assert pack.exercise(2).models == ("nk", "rbc", "ar1")  # the front matter takes either name; kept by file stem
    cmd = ex.model_command(pack, "nk")
    assert cmd == ImportIris(path=str(copy.resolve() / "models" / "nk.model"), name="policy_model")
    assert ex.model_command(pack, "policy_model") == cmd  # by its session name too
    s = Session()
    s.run(cmd)
    assert "policy_model" in s.objects and "nk" not in s.objects
    assert s.history[-1].code.startswith("policy_model = read_iris(")
    # the same model as load_model gives, by either name; a .pymodel is built under its session name
    assert set(s["policy_model"].model.endogenous) == set(ex.load_model(pack, "policy_model").model.endogenous) \
        == set(ex.load_model(copy, "nk").model.endogenous)
    gap = ex.load_model(pack, "gap")
    assert gap.name == "gap" and gap.model.endogenous == ("y",)
    s.run(ex.model_command(pack, "ar1"))
    assert type(s["gap"]).__name__ == "Template" and s["gap"].name == gap.name and "ModelSource(" in s.history[-1].code


@pytest.mark.parametrize("tables, expected", [
    ('[models.ghost]\nname = "g"\n', "[models.ghost] names no file of models/"),
    ('[models.rbc]\nname = "r"\nsize = 1\n', "[models.rbc]: unknown key 'size'; the key is name"),
    ('[models.rbc]\nlabel = "r"\n', "[models.rbc] needs name"),
    ('[models.rbc]\nname = "two words"\n', "[models.rbc] name should be a Python name"),
    ('[models.rbc]\nname = "class"\n', "[models.rbc] name should be a Python name"),
    ('[models.rbc]\nname = 3\n', "[models.rbc] name should be a Python name"),
    ('[models.rbc]\nname = "nk"\n', "the models nk and rbc would both be imported as 'nk'"),
    ('[models.rbc]\nname = "gap"\n', "the models ar1 and rbc would both be imported as 'gap'"),
    ('models = "rbc"\n', "a model's session name is written [models.<file stem>], with name"),
], ids=["no-file", "unknown-key", "no-name", "spaces", "keyword", "not-text", "clash-stem", "clash-name", "not-a-table"])
def test_a_bad_models_table_is_a_problem(copy, tables, expected):
    text = (copy / "pack.toml").read_text(encoding="utf-8")
    if tables.startswith("models ="):  # a key before any table, at the top, and without the fixture's [models.ar1]
        text = tables + text.split("\n[models.ar1]")[0]
    else:
        text += "\n" + tables
    (copy / "pack.toml").write_text(text, encoding="utf-8")
    assert any(expected in p for p in _problems(copy)), _problems(copy)


def test_a_solution_runs_in_the_pack_folder_with_pack_dir(copy, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (copy / "exercises" / "02-unit-root-nile" / "solution.py").write_text(
        "from pathlib import Path\n"
        "from pymetrix.exercises import load_model\n"
        "assert Path.cwd() == PACK_DIR and Path('models/rbc.mod').is_file()\n"
        "ar = load_model(PACK_DIR, 'ar1')\n"
        "adf = nile.test('adf', 'volume')\n"
        "adf_trend = nile.test('adf', 'volume', trend='ct')\n", encoding="utf-8")
    session, results, error = run_exercise(ex.load_pack(copy).exercise(2))
    assert error == "" and all(r.passed for r in results)
    assert Path.cwd() == tmp_path and "PACK_DIR" not in session.objects and "PACK_DIR" not in session.namespace
    assert type(session["ar"]).__name__ == "Template" and type(session["nile"]).__name__ == "Dataset"


def test_every_documented_path_resolves_on_real_results():
    import re

    sample_macro, nile = Dataset.example("sample_macro"), Dataset.example("nile")
    fit = Equation("c ~ y + g").fit(sample_macro, method=OLS(cov="white"))
    objects = {"fit": fit, "eg": sample_macro.test("engle_granger", "c, y"), "wald": fit.wald("y = 0.7"),
               "adf": nile.test("adf", "volume"), "sample_macro": sample_macro, "mroz": Dataset.example("mroz")}
    spec = (Path(__file__).resolve().parents[1] / "docs" / "exercise-packs.md").read_text(encoding="utf-8")
    assert 'fit = Equation("c ~ y + g").fit(sample_macro, method=OLS(cov="white"))' in spec  # the objects as made here
    assert 'mroz = Dataset.example("mroz")' in spec
    table = spec.split("<!-- paths:start -->", 1)[1].split("<!-- paths:end -->", 1)[0]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in table.strip().splitlines()[2:]]
    assert len(rows) >= 20
    for obj, path, value, _ in rows:
        path = path.strip("`")
        found = ex._plain(ex.resolve(objects[obj], path))
        if re.fullmatch(r"-?[\d.]+(e-?\d+)?", value):
            assert found == pytest.approx(float(value), rel=1e-4), (obj, path, found)
            assert isinstance(found, (int, float)) and not isinstance(found, bool)
        else:
            assert found == value, (obj, path, found)
    assert ex.resolve(fit, "se.y") == pytest.approx(fit.coefficients().set_index("term").loc["y", "std_error"])
    with pytest.raises(SpecificationError, match="is a method"):
        ex.resolve(objects["adf"], "reject")
