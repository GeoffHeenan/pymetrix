"""The public validation programs the model reader already uses (docs/validation.md), translated: every line has a
row (an included program's under its name), and the count of each status is fixed, so a change in the translation
shows.

They are not part of pymetrix: ``PYMETRIX_BRILLET`` names a folder with the reader's own copy of brillet_basic.prg
(J.-L. Brillet's book, chapter 5; copyright, so not in the repository), ``PYMETRIX_FRBUS`` the Federal Reserve's
FRB/US package folder (programs/example1.prg ... and subs/master_library.prg, which they include) and
``PYMETRIX_FRBUS_DATA`` its data package (LONGBASE.TXT). Each test skips without its files.

The FRB/US programs read the Fed's database (``dbopen``, ``fetch``) and make their model with an add-in
(``read_xml_model``): neither is translated. ``test_frbus_example1_against_the_hand_written_simulation`` supplies
both as tests/test_frbus_example.py does (LONGBASE, the model read by ``read_eviews_xml``) and translates the
program's own lines: its included library's subroutines (the policy switches, the ``_aerr`` group set to zero), the
model's solve options, add factors, scenarios and solves, its ``if`` block and the responses. One line is changed
for that: ``wfcreate`` opens LONGBASE instead (the page the database would fill). ``call dateshift``, whose
subroutine uses EViews's date numbers (``@dateval``, ``@dateadd``, ``@datestr``), translates as it is: the end of
the simulation it gives, 2025q4 (2020q1 and 23 quarters), is checked.
"""

import os
from pathlib import Path

import pytest

from pymetrix_eviews import translate_program

BRILLET = os.environ.get("PYMETRIX_BRILLET")
FRBUS = os.environ.get("PYMETRIX_FRBUS")

# translated, translated with a note, left out, error: the program's own lines, then every row (the included programs')
FRBUS_COUNTS = {
    "programs/example1.prg": ((57, 12, 29, 0), (195, 57, 51, 0)),
    "programs/example2.prg": ((72, 9, 45, 0), (1613, 152, 1370, 0)),
    "programs/example3.prg": ((147, 6, 38, 0), (285, 51, 60, 0)),
    "programs/example4.prg": ((108, 10, 55, 0), (1649, 153, 1380, 0)),
    "programs/pings.prg": ((341, 62, 60, 0), (1882, 205, 1385, 0)),
    "programs/stochsim.prg": ((243, 37, 121, 0), (381, 82, 143, 0)),
    "programs/ocpolicy.prg": ((171, 18, 89, 0), (1712, 161, 1414, 0)),
    "subs/master_library.prg": ((118, 27, 60, 0), (118, 27, 60, 0)),
}


def _counts(t, own=False):
    from collections import Counter

    c = Counter(r.status for r in t.report if not own or r.file == t.program)
    return tuple(c.get(s, 0) for s in ("translated", "translated with a note", "left out", "error"))


def _rows_cover(t, path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    assert [r.line for r in t.report if r.file == t.program] == [i for i, text in enumerate(lines, start=1) if text.strip()]
    assert all(r.reason for r in t.report if r.status in ("left out", "error"))


def _brillet(name):
    if not BRILLET:
        pytest.skip("set PYMETRIX_BRILLET to a folder with your copies of Brillet's book files")
    root = Path(BRILLET)
    path = root / name if root.is_dir() else root.parent / name
    if not path.is_file():
        pytest.skip(f"{name} is not in PYMETRIX_BRILLET")
    return path


def test_brillet_basic_model_program():
    path = _brillet("brillet_basic.prg")
    t = translate_program(path)
    _rows_cover(t, path)
    # the comments; scalar f (with a note: the model reader's refusal, at its line); the model and its 17 appends,
    # collected and given to the core reader, which refuses the model: f, "a function of" in the book, has no value
    assert _counts(t) == (4, 1, 18, 0)
    assert all(r.reason.startswith("the model _fra_1 is not read: line 6: the scalar f is declared without a value")
               for r in t.report if r.status == "left out")
    assert "the model reader: the scalar f is declared without a value" in t.report.line(6).reason
    namespace: dict = {}
    exec(compile(t.script, str(path), "exec"), namespace)
    out = t.run()
    assert out["f"] != out["f"] and namespace["f"] != namespace["f"]  # NA both ways


@pytest.mark.skipif(not FRBUS, reason="set PYMETRIX_FRBUS to the FRB/US package folder")
@pytest.mark.parametrize("name", sorted(FRBUS_COUNTS))
def test_frbus_programs(name):
    path = Path(FRBUS) / name
    if not path.is_file():
        pytest.skip(f"{name} is not in PYMETRIX_FRBUS")
    t = translate_program(path)
    _rows_cover(t, path)
    assert (_counts(t, own=True), _counts(t)) == FRBUS_COUNTS[name]
    compile(t.script, str(path), "exec")
    if name.startswith("programs/"):  # each includes the library: its rows follow the include line, under its name
        files = t.report.files()
        assert files[:2] == [path.name, "../subs/master_library.prg"]
        assert "def set_mp(dmpxxx):" in t.script and "set_mp(" in t.script.split("def set_mp(dmpxxx):")[1]
    for r in t.report:  # their models come from the add-in: the model's lines say it is not made
        if r.status == "left out" and r.file == t.program and any(f".{m}" in r.eviews for m in ("solve", "addinit", "scenario")):
            assert "is not made by a line translated so far" in r.reason or "$" in r.reason, r.reason


FRBUS_DATA = os.environ.get("PYMETRIX_FRBUS_DATA")


@pytest.mark.skipif(not (FRBUS and FRBUS_DATA), reason="set PYMETRIX_FRBUS and PYMETRIX_FRBUS_DATA to the FRB/US packages")
def test_frbus_example1_against_the_hand_written_simulation(tmp_path):
    """example1.prg translated and run, both ways, against the same simulation written by hand with
    tests/test_frbus_example.py's means: LONGBASE with the examples' policy settings, the tracking add factors over
    the program's 24 quarters, and the funds-rate shock. Equal to 1e-8 (the same arithmetic: equal to the last bit)."""
    import importlib.util

    import numpy as np
    import pandas as pd

    from pymetrix import Dataset
    from pymetrix.models import Newton, Scenario
    from pymetrix.models.eviews import read_eviews_xml

    spec = importlib.util.spec_from_file_location("frbus_example", Path(__file__).parents[3] / "tests" / "test_frbus_example.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    model = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml").model
    frame = pd.read_csv(Path(FRBUS_DATA) / "LONGBASE.TXT")
    frame.columns = [c.lower() for c in frame.columns]
    data = tmp_path / "longbase.csv"
    frame.to_csv(data, index=False)
    text = (Path(FRBUS) / "programs" / "example1.prg").read_text()
    old = "wfcreate(wf=aaa,page={%mainpage}) q {%wfstart} {%wfend}"
    assert text.count(old) == 1 and text.count("call dateshift(%simstart,%simend,!nsimqtrs-1)") == 1
    text = text.replace(old, "wfopen longbase")
    t = translate_program(text, bundle=data, read_options={"time": "obs", "freq": "Q"}, objects={"stdver": model},
                          folder=Path(FRBUS) / "programs")
    left = {r.eviews.split()[0].split("(")[0].split(".")[-1] for r in t.report if r.file == t.program and r.status == "left out"}
    assert left == {"wfclose", "read_xml_model", "dbopen", "fetch", "statusline", "graph", "addtext", "legend", "align", "show"}  # every model, sample, series and control line translated
    assert t.report.counts()["error"] == 0
    namespace = {"stdver": model}
    exec(compile(t.script, "example1.prg", "exec"), namespace)
    assert namespace["simend"] == "2025q4"  # dateshift: 2020q1 and !nsimqtrs - 1 = 23 quarters, by hand
    got = namespace["longbase"].frame.loc["2020Q1":"2025Q4"]

    hand = frame.copy()
    hand.index = pd.PeriodIndex(hand.pop("obs"), freq="Q", name="time")
    hand = example.with_add_factor_series(example.policy(hand), model)
    span = ("2020Q1", "2025Q4")
    factors = model.add_factors(Dataset.from_frame(hand), span)
    base = model.solve(Dataset.from_frame(hand), span, Scenario(adjust=factors), solver=Newton()).frame.loc["2020Q1":"2025Q4"]
    shocked = model.solve(Dataset.from_frame(hand), span, Scenario(adjust=factors, shift={"rffintay_aerr": {"2020Q1": 1.0}}),
                          solver=Newton()).frame.loc["2020Q1":"2025Q4"]
    for v in ("rff", "rg10", "lur", "pic4"):
        np.testing.assert_allclose(got[f"d_{v}"], (shocked - base)[v], rtol=0, atol=1e-8)
    np.testing.assert_allclose(got["xgap_0"], base["xgap"], rtol=0, atol=1e-8)  # the tracking solution is the baseline
    assert got["d_rff"].iloc[0] == pytest.approx(1.0, abs=0.01)  # test_funds_rate_shock's check of the first quarter
    # steps mode: the same, as recorded steps (about two minutes: each of the 1,169 steps copies the page)
    out = t.run()
    assert out["simend"] == "2025q4"
    steps = out["session"]["longbase"].frame.loc["2020Q1":"2025Q4"]
    for column in ("d_rff", "d_rg10", "d_lur", "d_pic4", "xgap_0", "rff_1"):
        np.testing.assert_array_equal(steps[column], got[column])
