"""The public validation programs the model reader already uses (docs/validation.md), translated: every line has a
row, and the count of each status is fixed, so a change in the translation shows.

They are not part of pymetrix: ``PYMETRIX_BRILLET`` names a folder with the reader's own copy of brillet_basic.prg
(J.-L. Brillet's book, chapter 5; copyright, so not in the repository), ``PYMETRIX_FRBUS`` the Federal Reserve's
FRB/US package folder (programs/example1.prg ... and subs/master_library.prg). Each test skips without its files.
The programs need the Fed's database (``dbopen``, ``fetch``) and EViews's model objects (phase 2), so their steps are
not run here; the scripts are compiled, and the one with no data (Brillet's) is run both ways.
"""

import os
from pathlib import Path

import pytest

from pymetrix_eviews import translate_program

BRILLET = os.environ.get("PYMETRIX_BRILLET")
FRBUS = os.environ.get("PYMETRIX_FRBUS")

# translated, translated with a note, left out, error
FRBUS_COUNTS = {
    "programs/example1.prg": (49, 8, 41, 0),
    "programs/example2.prg": (62, 10, 54, 0),
    "programs/example3.prg": (97, 8, 86, 0),
    "programs/example4.prg": (98, 11, 64, 0),
    "programs/pings.prg": (154, 45, 264, 0),
    "programs/stochsim.prg": (163, 12, 226, 0),
    "programs/ocpolicy.prg": (132, 7, 139, 0),
    "subs/master_library.prg": (55, 0, 150, 0),
}


def _counts(t):
    return tuple(t.report.counts().values())


def _rows_cover(t, path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    assert [r.line for r in t.report] == [i for i, text in enumerate(lines, start=1) if text.strip()]
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
    assert _counts(t) == (4, 1, 18, 0)  # the comments; scalar f (with a note); model and its 17 appends: phase 2
    assert all(r.reason.startswith("phase 2") for r in t.report if r.status == "left out")
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
    assert _counts(t) == FRBUS_COUNTS[name]
    compile(t.script, str(path), "exec")
    reasons = " ".join(r.reason for r in t.report)
    assert "phase 3" in reasons  # their loops and subroutines
