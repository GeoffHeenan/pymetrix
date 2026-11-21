"""Row conditions (``where=``) are checked before pandas evaluates them (``pymetrix.data.conditions``).

- Every condition the repository's docs, help, docstrings and tests use, and every form of the documented syntax,
  selects exactly the rows plain ``DataFrame.eval`` selected before the check (the old code path, as the reference),
  and the main forms match the same selection written by hand in NumPy.
- Escape payloads (``@`` references to Python variables, attribute chains, ``__import__``, unknown calls) are refused
  with a ``SpecificationError`` before anything runs, wherever a condition is used: ``Sample``, ``Dataset.select``,
  estimation. One of them is shown to be live in plain ``DataFrame.eval`` (it writes a file).
- A project from someone else whose steps and saved dataset carry such a condition opens with ``trust=False``: the
  steps are kept, not run, and the payload never runs (on the old code, opening it wrote the file).
"""

from __future__ import annotations

import ast
import io
import json
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, PyMetrixError, Sample, SpecificationError
from pymetrix.data.conditions import FUNCTIONS, ConditionError, check_condition, condition_mask

ROOT = Path(__file__).resolve().parents[1]

# the documented syntax: the guide, the New sample dialog's help, the module docstring, the "Rows where" field
DOCUMENTED = [
    "pop > 1e6", "pop > 1e6 and year != 2020", "year > 2000", "recession == 0", "strike == 0", "urban == 1", "x > 0",
    "keep == 1", "country in ['DEU', 'FRA']", "country not in ('MLT',)", "country == 'DEU' or pop < 10",
    "not (x > 0)", "~(x > 0) & (y < 3)", "(x > 0) | (y <= -1)", "x > 0 and not y == 2", "x + y > 1", "x * 2 - y / 4 >= 0",
    "x ** 2 > 3", "x % 2 == 1", "x // 2 == 0", "-x > 1", "abs(x) > 1", "sqrt(abs(x)) < 2", "log(abs(x) + 1) > 0.5",
    "exp(x) > 1", "arctan2(x, y) > 0", "floor(x) == 1", "x > y", "x == x", "x != 2.5", "`x` > 0", "x in [1, 2, -150]",
]
HAND = {  # the same selection written out in NumPy (an independent check of the main forms)
    "x > 0": lambda f: f["x"] > 0,
    "pop > 1e6 and year != 2020": lambda f: (f["pop"] > 1e6) & (f["year"] != 2020),
    "country in ['DEU', 'FRA']": lambda f: f["country"].isin(["DEU", "FRA"]),
    "country == 'DEU' or pop < 10": lambda f: (f["country"] == "DEU") | (f["pop"] < 10),
    "not (x > 0)": lambda f: ~(f["x"] > 0),
    "~(x > 0) & (y < 3)": lambda f: ~(f["x"] > 0) & (f["y"] < 3),
    "x + y > 1": lambda f: f["x"] + f["y"] > 1,
    "abs(x) > 1": lambda f: np.abs(f["x"]) > 1,
    "x in [1, 2, -150]": lambda f: f["x"].isin([1, 2, -150]),
}
PAYLOADS = [
    "@self", "@where", "@self.frame.to_csv('out.csv')", "@mask.tofile('out.bin')", "y.__class__", "x.__class__.__mro__",
    "x.abs() > 0", "x.values", "().__class__.__bases__[0].__subclasses__()", "__import__('os').system('true')",
    "__import__", "`__class__` > 0", "open('out.txt')", "getattr(x, 'values')", "eval('1')", "exec('1')",
    "Timestamp('2000') < time", "x[0] > 1", "(lambda: 1)()", "abs(x=x) > 0", "abs(*[x]) > 0", "x @ y", "x if y else y",
    "{1: 2}", "f'{x}' == 'a'", "x is None", "10 ** 10 ** 10 > x", "((9 ** 99) ** 99) ** 99 > x", "'a' * 10 ** 9 == x",
    "[1] * 10 ** 9 == x", "x > 0; import os", "[i for i in x]",
]
SKIP = {"y >", "sample", "span", ", ", "Rows where"}  # not conditions: one a test types half way, a label, other "where ="s


def _panel() -> Dataset:
    rng = np.random.default_rng(7)
    values = np.array([-150.0, -1.0, 0.0, 1.0, 2.0, 2.5, 50.0, 2e6])
    rows = [{"country": c, "year": y} for c in ("DEU", "FRA", "MLT", "ITA") for y in range(2010, 2030)]
    frame = pd.DataFrame(rows)
    for name in ("x", "y", "pop", "keep", "strike", "urban", "recession", "c"):
        frame[name] = rng.choice(values, size=len(frame))
    return Dataset.from_frame(frame, entity="country", time="year", freq="Y")


def _repository_conditions() -> set[str]:
    """Every ``where="..."`` (and ``"where": "..."``) in the docs, the in-app help, the source and the tests."""
    pattern = re.compile(r"""(?:\bwhere\s*=\s*|["']where["']\s*:\s*)(["'])((?:(?!\1).)+)\1""")
    files = [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md"), *(ROOT / "src").rglob("*.py"),
             *(ROOT / "tests").rglob("*.py"), *(ROOT / "packages").rglob("*.py"), *(ROOT / "packages").rglob("*.md")]
    found = set()
    for f in files:
        if f.name == Path(__file__).name:
            continue
        found |= {m.group(2) for m in pattern.finditer(f.read_text(encoding="utf-8", errors="replace"))}
    return found - SKIP


def _frame_for(text: str) -> pd.DataFrame:
    """A frame with a column for each name the condition uses, so it can be evaluated whatever data it was for (the
    years as numbers, as in a cross-section)."""
    base = _panel().frame.reset_index()
    base["year"] = base["year"].dt.year
    names = {n.id for n in ast.walk(ast.parse(text.replace("`", ""), mode="eval")) if isinstance(n, ast.Name)} - FUNCTIONS
    extra = {n: base["x"].to_numpy()[::-1] for n in names if n not in base.columns}
    return base.assign(**extra)


def _old(frame: pd.DataFrame, text: str) -> np.ndarray:
    """What the code did before the check: pandas' DataFrame.eval on the rows, axes as columns."""
    return np.asarray(frame.reset_index().eval(text), dtype=bool)


def test_every_condition_in_the_repository_and_the_docs_selects_the_same_rows():
    conditions = sorted(_repository_conditions() | set(DOCUMENTED))
    assert len(conditions) >= 35 and {"pop > 1e6", "keep == 1", "strike == 0", "x > -100"} <= set(conditions)
    proper = 0
    for text in conditions:
        check_condition(text)  # none refused
        frame = _frame_for(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # log and sqrt of negative values: NaN, as before
            expected, got = _old(frame, text), condition_mask(frame, text)
        assert np.array_equal(got, expected), text
        proper += 0 < expected.sum() < len(expected)
    assert proper >= 0.9 * len(conditions)  # nearly all select some rows and not others: the comparison has teeth
    data = _panel()
    for text, hand in HAND.items():
        expected = np.asarray(hand(data.frame.reset_index()), dtype=bool)
        assert np.array_equal(Sample(where=text).mask(data), expected), text
        assert np.array_equal(data.sample_mask(where=text), expected), text
        assert len(data.select(where=text).frame) == expected.sum(), text


def test_periods_and_dates_are_compared_as_text():
    data = Dataset.example("sample_macro")
    periods = data.frame.index
    assert np.array_equal(Sample(where="time >= '2005Q1'").mask(data), periods >= pd.Period("2005Q1"))
    assert np.array_equal(data.sample_mask(where="time < '2003Q1'"), periods < pd.Period("2003Q1"))
    daily = pd.DataFrame({"r": np.arange(10.0)}, index=pd.date_range("2020-01-01", periods=10, name="date"))
    expected = daily.index < pd.Timestamp("2020-01-04")
    assert np.array_equal(condition_mask(daily, "date < '2020-01-04'"), expected) and expected.sum() == 3


def test_estimation_uses_the_rows_the_condition_selects():
    data = _panel()
    fit = Equation("y ~ x").fit(data, where="pop > 1 and country != 'MLT'")
    frame = data.frame.reset_index()
    kept = frame[(frame["pop"] > 1) & (frame["country"] != "MLT")]
    X = np.column_stack([np.ones(len(kept)), kept["x"]])
    beta = np.linalg.lstsq(X, kept["y"].to_numpy(), rcond=None)[0]
    assert fit.provenance.nobs == len(kept)
    assert np.allclose(np.asarray(fit.params.values, dtype=float), beta, rtol=1e-10)


def test_the_check_is_needed(tmp_path):
    """The payloads are not hypothetical: plain DataFrame.eval runs a method of a Python variable named with @."""

    class Box:
        def write(self, path: str) -> int:
            Path(path).write_text("ran")
            return 1

    box = Box()  # noqa: F841 - reached by the condition below, through pandas' @
    marker = tmp_path / "ran.txt"
    pd.DataFrame({"x": [1.0]}).eval(f"@box.write({str(marker)!r})")
    assert marker.read_text() == "ran"


@pytest.mark.parametrize("payload", PAYLOADS)
def test_escape_payloads_are_refused_before_anything_runs(payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(pd.DataFrame, "eval", lambda *a, **k: called.append(a) or pytest.fail("evaluated"))
    data = _panel()
    with pytest.raises(SpecificationError, match="row condition") as info:
        check_condition(payload)
    assert isinstance(info.value, ConditionError) and isinstance(info.value, PyMetrixError)
    for use in (lambda: Sample(where=payload).mask(data), lambda: data.select(where=payload),
                lambda: data.sample_mask(where=payload), lambda: Equation("y ~ x").fit(data, where=payload)):
        with pytest.raises(ConditionError):
            use()
    assert not called and not list(tmp_path.iterdir())


def test_what_the_refusal_says():
    with pytest.raises(ConditionError, match=r"@ \(a reference to a Python variable\)"):
        check_condition("@self")
    with pytest.raises(ConditionError, match=r"attribute access"):
        check_condition("x.__class__")
    with pytest.raises(ConditionError, match=r"a call to system\(\)|a call to __import__"):
        check_condition("__import__('os').system('true')")
    with pytest.raises(ConditionError, match=r"names starting with __"):
        check_condition("__builtins__ > 0")
    with pytest.raises(ConditionError, match=r"cannot be read"):
        check_condition("x >")
    with pytest.raises(ConditionError, match=r"takes 2 arguments"):
        check_condition("arctan2(x) > 0")
    with pytest.raises(ConditionError, match="a row condition is text"):
        check_condition(3)  # type: ignore[arg-type]
    check_condition("name == '@ __import__ x.y'")  # inside quotes, all of it is just text
    assert Sample(where="@self").where == "@self"  # kept as text: refused only when it would be evaluated


def _malicious_project(tmp_path: Path, payload: str) -> Path:
    """A project as someone else might send it: steps with row conditions, and a dataset saved with an active sample
    with one, each replaced by ``payload``."""
    import pyarrow.parquet as pq

    from pymetrix.commands import FitEquation, NewSample, SetSample, Session, save_project

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.objects["placed"] = Dataset.example("sample_macro").with_sample(Sample(where="y > 0"))
    s.objects["other"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y", name="cons", where="y > 0"))
    s.run(NewSample("big", where="y > 0"))
    s.run(SetSample(sample="big"))
    s.run(FitEquation("c ~ y", name="cons2"))
    s.run(FitEquation("c ~ y", data="placed", name="cons3"))
    s.run(FitEquation("c ~ y", data="other", name="plain", where="c > 0"))
    good = save_project(s, tmp_path / "good.pymx")
    bad = tmp_path / "bad.pymx"
    replaced = 0
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            if item.filename == "commands.json":
                replaced += data.count(b'"y > 0"')
                data = data.replace(b'"y > 0"', json.dumps(payload).encode())
            elif item.filename.endswith(".parquet"):
                table = pq.read_table(io.BytesIO(data))
                meta = dict(table.schema.metadata)
                if b'"y > 0"' in meta[b"pymetrix"]:
                    replaced += 1
                    meta[b"pymetrix"] = meta[b"pymetrix"].replace(b'"y > 0"', json.dumps(payload).encode())
                    buffer = io.BytesIO()
                    pq.write_table(table.replace_schema_metadata(meta), buffer)
                    data = buffer.getvalue()
            dst.writestr(item, data)
    assert replaced >= 3  # the fit's condition, the sample's, and the saved dataset's active sample
    return bad


@pytest.mark.parametrize("trust", [False, None])
def test_a_project_with_a_malicious_condition_opens_without_running_it(tmp_path, trust):
    from pymetrix.commands import load_project

    marker = tmp_path / "ran.parquet"
    path = _malicious_project(tmp_path, f"@self.write({str(marker)!r})")
    session = load_project(path, trust=trust)
    assert not marker.exists()  # the old code wrote it here, while opening
    records = {r.command.name if hasattr(r.command, "name") else type(r.command).__name__: r for r in session.history}
    for name in ("cons", "big", "cons3"):
        assert records[name].skipped and "row condition" in records[name].error and "@" in records[name].error, name
    assert records["SetSample"].skipped and records["cons2"].skipped  # they use what the refused steps would have made
    assert records["plain"].ok and "plain" in session.objects  # the rest of the project still runs
    assert "placed" in session.objects and session["placed"].sample.where.startswith("@self")  # kept as it was saved
