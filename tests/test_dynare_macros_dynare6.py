"""Dynare 6's macro language beyond the common directives: set operations on arrays (``|``, ``&``, ``-``, ``*``, ``^``),
type tests, casts, ``sum``/``gamma``/``erf``/``normcdf``..., the way values are written out, and ``@#line``.

Checked against Dynare 6.0's own preprocessor (``dynare-preprocessor file.mod onlymacro``): 145 expressions and their
expansions, or the preprocessor's refusal, recorded in tests/fixtures/dynare-macro-6.0.json."""

import json
import math
import sys
from pathlib import Path

import pytest

from pymetrix import read_mod
from pymetrix.models.dynare_macros import MacroError, expand_macros

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "dynare-macro-6.0.json").read_text(encoding="utf-8"))
DEFINES = "".join(f"@#define {k} = {v}\n" for k, v in RECORDED["defines"].items())


def _numbers(text):
    try:
        return float(text)
    except ValueError:
        return None


@pytest.mark.parametrize("expr", sorted(RECORDED["cases"]))
def test_as_dynares_preprocessor(expr):
    expected = RECORDED["cases"][expr]
    source = DEFINES + f"R @{{{expr}}}\n"
    if expected is None:
        with pytest.raises(MacroError):
            expand_macros(source)
        return
    got = expand_macros(source)[0].strip()[2:]
    if _numbers(expected) is not None and _numbers(got) is not None and "e" not in expected:
        assert _numbers(got) == pytest.approx(_numbers(expected), rel=1e-14, abs=1e-15)
    assert got == expected


def test_line_directive_is_accepted_and_changes_nothing():
    text = 'a\n@#line "other.mod" 40\nb @{1+1}\n@#define x = [1,2] | [3]\nc @{x}\n'
    out, lines = expand_macros(text)
    assert out.rstrip("\n").split("\n") == ["a", "b 2", "c [1, 2, 3]"] and lines[:3] == [1, 3, 5]  # Dynare 6.0 writes the same text
    with pytest.raises(MacroError, match="@#line reads"):
        expand_macros("@#line 40\n")


def test_unknown_directives_are_still_refused_at_their_line():
    with pytest.raises(MacroError, match=r"line 2: unknown macro directive @#foo") as err:
        expand_macros("a\n@#foo bar\n")
    assert err.value.line == 2


def test_a_model_written_with_set_operations():
    text = """@#define all = ["us", "ea", "jp"]
@#define small = all - ["us"]
@#define pairs = small * ["y", "pi"]
var @#for c in all
  y_@{c}
@#endfor
;
varexo @#for c in all
  e_@{c}
@#endfor
;
parameters rho;
rho = 0.5;
model;
@#for c in all
y_@{c} = rho*y_@{c}(-1) + e_@{c};
@#endfor
end;
@#if isarray(pairs) && length(pairs) == 4 && pairs[1] == ("ea", "y")
@#else
@#error "pairs"
@#endif
"""
    dyn = read_mod(text.replace("var @#for", "var\n@#for").replace("varexo @#for", "varexo\n@#for"))
    assert dyn.endogenous == ("y_us", "y_ea", "y_jp") and len(dyn.equations) == 3
    assert math.isclose(dyn.params["rho"], 0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
