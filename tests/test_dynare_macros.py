"""Dynare's macro processor (``@#`` directives), run before a ``.mod`` file is read (pymetrix.models.dynare_macros).

Checked: each directive against text expanded by hand (``@#define`` of numbers, strings, arrays, ranges and
expressions, ``@{...}``, ``@#if``/``@#ifdef``/``@#ifndef``/``@#elseif``/``@#else``, ``@#for`` over arrays and tuples,
``@#include``, ``@#echo``, ``@#error``, line continuations, macro functions); a three-country model written with
``@#define``, ``@#for`` loops generating each country's equations, ``@#if`` on a define and an ``@#include``d
calibration, against the same model written out by hand: the same equations, and the same solution to 1e-12; the
model editor's check after expansion, its problems at the lines they come from; refusals: unknown directives,
unclosed blocks, includes not found or without a file, undefined variables, ``@#error``."""

import numpy as np
import pytest

from pymetrix import ModelSource, read_mod
from pymetrix.models import ModelError
from pymetrix.models.dynare_macros import MacroError, evaluate, expand_macros


def _expand(text: str, **kw) -> str:
    return expand_macros(text, **kw)[0]


def test_expressions():
    assert evaluate("1:4") == [1.0, 2.0, 3.0, 4.0] and evaluate("1:2:7") == [1.0, 3.0, 5.0, 7.0] and evaluate("3:1") == []
    assert evaluate("2 + 3*4 - 6/2") == 11 and evaluate("2^3^2") == 512 and evaluate("-2^2") == -4
    assert evaluate("1 < 2 && !(3 >= 4) || false") is True and evaluate("1 == 1 && 2 != 2") is False
    assert evaluate('"ab" + "cd"') == "abcd" and evaluate('"a" < "b"') is True
    assert evaluate("[1, 2] + [3]") == [1.0, 2.0, 3.0] and evaluate("[1, 2, 3][2]") == 2 and evaluate("[1, 2, 3][2:3]") == [2.0, 3.0]
    assert evaluate("2 in [1, 2]") is True and evaluate('"x" in ["y"]') is False
    assert evaluate("length(1:5)") == 5 and evaluate("isempty([])") is True and evaluate("max(1, 4, 2)") == 4
    assert evaluate("[i^2 for i in 1:4 when i != 2]") == [1.0, 9.0, 16.0]
    assert evaluate("(1, \"a\")") == (1.0, "a") and evaluate("defined(N)", {"N": 1.0}) is True and evaluate("defined(M)") is False
    assert evaluate("x * 2", {"x": 3.0}) == 6 and evaluate("floor(2.7) + ceil(0.2) + round(2.5) + mod(7, 3)") == 2 + 1 + 3 + 1


def test_define_and_substitution():
    text = '@#define N = 3\n@#define name = "us"\n@#define rho = 0.5 + 0.25\n@#define list = [1, 2.5, "z"]\nx_@{name} = @{rho}*@{N + 1}; @{list}\n'
    assert _expand(text) == "x_us = 0.75*4; [1, 2.5, z]\n"  # directives leave no line; strings unquoted, as Dynare 6.0
    assert _expand("@#define t = 1 > 0\n@{t} @{!t}") == "true false"
    assert _expand("@#define f(a, b) = a*10 + b\n@{f(2, 3)}") == "23"
    assert _expand("@#define long = 1 + \\\n  2\n@{long}") == "3"  # a directive continued on the next line


def test_conditionals():
    def branch(n: int) -> str:
        return _expand(f"@#define N = {n}\n@#if N == 1\none\n@#elseif N == 2\ntwo\n@#else\nmany\n@#endif").strip()

    assert [branch(1), branch(2), branch(5)] == ["one", "two", "many"]
    assert _expand("@#define A = 1\n@#ifdef A\nyes\n@#endif\n@#ifndef A\nno\n@#endif\n@#ifndef B\nb\n@#endif").split() == ["yes", "b"]
    nested = "@#define A = 1\n@#if A\n@#if A > 1\ninner\n@#else\nouter\n@#endif\n@#endif"
    assert _expand(nested).split() == ["outer"]
    assert _expand("@#if 0\nhidden @{undefined_name}\n@#endif\nshown").strip() == "shown"  # a branch not taken is not evaluated


def test_loops():
    text = '@#for c in ["us", "ea"]\ny_@{c} = rho*y_@{c}(-1) + e_@{c};\n@#endfor'
    assert _expand(text).split("\n") == ["y_us = rho*y_us(-1) + e_us;", "y_ea = rho*y_ea(-1) + e_ea;"]
    pairs = '@#for (c, w) in [("us", 0.6), ("ea", 0.4)]\n+ @{w}*y_@{c}\n@#endfor'
    assert _expand(pairs).split("\n") == ["+ 0.6*y_us", "+ 0.4*y_ea"]
    assert _expand("@#for i in 1:5 when i != 3\n@{i}\n@#endfor").split() == ["1", "2", "4", "5"]
    nested = "@#for i in 1:2\n@#for j in 1:2\n@#if i != j\n@{i}@{j}\n@#endif\n@#endfor\n@#endfor"
    assert _expand(nested).split() == ["12", "21"]
    assert _expand("@#define i = 9\n@#for i in 1:2\n@#endfor\n@{i}").strip() == "9"  # the loop variable is the loop's own


def test_lines_map_back():
    text = "a\n@#for i in 1:2\nx@{i}\n@#endfor\n// comment\nb"
    expanded, lines = expand_macros(text)
    assert [x.strip() for x in expanded.split("\n")] == ["a", "x1", "x2", "", "b"] and lines == [1, 3, 3, 5, 6]  # comments blanked


def test_include_echo_and_error(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "part.inc").write_text("@#define K = 7\nk = @{K}; // from the include\n", encoding="utf-8")
    (tmp_path / "other.inc").write_text("other\n", encoding="utf-8")
    text = '@#include "sub/part.inc"\n@#includepath "sub"\n@#include "part.inc"\n@#include "other.inc"\n@{K + 1}'
    expanded, lines = expand_macros(text, base=tmp_path)
    assert [x.strip() for x in expanded.split("\n")] == ["k = 7;", "", "k = 7;", "", "other", "", "8"]
    assert lines == [1, 1, 3, 3, 4, 4, 5]  # an included file's lines at its @#include
    with pytest.warns(UserWarning, match="@#echo \\(line 2\\): N is 3"):
        _expand('@#define N = 3\n@#echo "N is " + "3"')
    with pytest.raises(MacroError, match="line 2: @#error: N must be positive") as caught:
        _expand('@#define N = -1\n@#error "N must be positive"')
    assert caught.value.line == 2


def test_refusals(tmp_path):
    cases = {
        "@#bogus x\n": "unknown macro directive @#bogus",
        "@#if 1\nx\n": "the @#if of line 1 is not closed \\(no @#endif\\)",
        "a\n@#for i in 1:2\nx\n": "the @#for of line 2 is not closed",
        "@#endif\n": "@#endif without an open @#if",
        "@#for i in 1:2\n@#endif\n": "@#endif without an open @#if \\(the @#for of line 1 is still open\\)",
        "@#else\n": "@#else without an open @#if",
        "@{N}\n": "N is not defined",
        "@{1 + }\n": "unexpected",
        "x @#define N = 1\n": "must start its line",
        "@#if \"s\"\n@#endif\n": "not a boolean or a number",
        "@#define 1x = 2\n": "@#define reads",
        "@{[1, 2][3]}": "outside 1 to 2",
    }
    for text, message in cases.items():
        with pytest.raises(MacroError, match=message):
            _expand(text)
    with pytest.raises(MacroError, match="needs the model read from its file"):
        _expand('@#include "x.mod"\n')
    with pytest.raises(MacroError, match="was not found"):
        _expand('@#include "missing.mod"\n', base=tmp_path)
    (tmp_path / "self.mod").write_text('@#include "self.mod"\n', encoding="utf-8")
    with pytest.raises(MacroError, match="nested more than 50"):
        _expand('@#include "self.mod"\n', base=tmp_path)
    assert issubclass(MacroError, ModelError)
    with pytest.raises(ModelError, match="unknown macro directive"):
        read_mod("@#bogus\nvar x;")


MAIN = """// three countries, written with the macro processor
@#define countries = ["us", "ea", "jp"]
@#define persistence = [0.8, 0.7, 0.6]
@#define n = length(countries)
@#define spillover = true
var
@#for c in countries
  y_@{c} pi_@{c} i_@{c}
@#endfor
  ;
varexo
@#for c in countries
  ey_@{c} epi_@{c} ei_@{c}
@#endfor
  ;
parameters sig bet kap phi
@#for c in countries
  rho_@{c}
@#endfor
@#if spillover
  omega
@#endif
  ;
@#include "calibration.inc"
model;
@#for i in 1:n
@#define c = countries[i]
  [name='IS @{c}'] y_@{c} = y_@{c}(+1) - sig*(i_@{c} - pi_@{c}(+1))
@#if spillover
    + omega*(0
@#for d in countries
@#if d != c
    + y_@{d}(-1)/@{n - 1}
@#endif
@#endfor
    )
@#endif
    + ey_@{c};
  pi_@{c} = bet*pi_@{c}(+1) + kap*y_@{c} + epi_@{c};
  i_@{c} = rho_@{c}*i_@{c}(-1) + (1 - rho_@{c})*(phi*pi_@{c} + 0.5*y_@{c}) + ei_@{c};
@#endfor
end;
shocks;
@#for c in countries
  var ey_@{c}; stderr 0.5;
  var epi_@{c}; stderr 0.3;
  var ei_@{c}; stderr 0.2;
@#endfor
@#if n > 1
  corr ey_us, ey_ea = 0.3;
@#endif
end;
"""

CALIBRATION = """sig = 1; bet = 0.99; kap = 0.1; phi = 1.5;
@#for i in 1:n
rho_@{countries[i]} = @{persistence[i]};
@#endfor
@#ifdef spillover
omega = 0.2;
@#endif
"""

BY_HAND = """// the same three countries, written out by hand
var y_us pi_us i_us y_ea pi_ea i_ea y_jp pi_jp i_jp;
varexo ey_us epi_us ei_us ey_ea epi_ea ei_ea ey_jp epi_jp ei_jp;
parameters sig bet kap phi rho_us rho_ea rho_jp omega;
sig = 1; bet = 0.99; kap = 0.1; phi = 1.5;
rho_us = 0.8; rho_ea = 0.7; rho_jp = 0.6;
omega = 0.2;
model;
  [name='IS us'] y_us = y_us(+1) - sig*(i_us - pi_us(+1)) + omega*(0 + y_ea(-1)/2 + y_jp(-1)/2) + ey_us;
  pi_us = bet*pi_us(+1) + kap*y_us + epi_us;
  i_us = rho_us*i_us(-1) + (1 - rho_us)*(phi*pi_us + 0.5*y_us) + ei_us;
  [name='IS ea'] y_ea = y_ea(+1) - sig*(i_ea - pi_ea(+1)) + omega*(0 + y_us(-1)/2 + y_jp(-1)/2) + ey_ea;
  pi_ea = bet*pi_ea(+1) + kap*y_ea + epi_ea;
  i_ea = rho_ea*i_ea(-1) + (1 - rho_ea)*(phi*pi_ea + 0.5*y_ea) + ei_ea;
  [name='IS jp'] y_jp = y_jp(+1) - sig*(i_jp - pi_jp(+1)) + omega*(0 + y_us(-1)/2 + y_ea(-1)/2) + ey_jp;
  pi_jp = bet*pi_jp(+1) + kap*y_jp + epi_jp;
  i_jp = rho_jp*i_jp(-1) + (1 - rho_jp)*(phi*pi_jp + 0.5*y_jp) + ei_jp;
end;
shocks;
  var ey_us; stderr 0.5; var epi_us; stderr 0.3; var ei_us; stderr 0.2;
  var ey_ea; stderr 0.5; var epi_ea; stderr 0.3; var ei_ea; stderr 0.2;
  var ey_jp; stderr 0.5; var epi_jp; stderr 0.3; var ei_jp; stderr 0.2;
  corr ey_us, ey_ea = 0.3;
end;
"""


def test_a_three_country_model_solves_as_its_hand_written_twin(tmp_path):
    (tmp_path / "calibration.inc").write_text(CALIBRATION, encoding="utf-8")
    (tmp_path / "world.mod").write_text(MAIN, encoding="utf-8")
    macro, hand = read_mod(tmp_path / "world.mod"), read_mod(BY_HAND)
    assert macro.endogenous == hand.endogenous and macro.exogenous == hand.exogenous and macro.params == hand.params
    assert [e.replace(" ", "") for e in macro.equations] == [e.replace(" ", "") for e in hand.equations]  # the same, but for spaces
    assert macro.shock_sd == hand.shock_sd and macro.shock_corr == hand.shock_corr == {("ey_us", "ey_ea"): 0.3}
    a, b = macro.solve(), hand.solve()
    np.testing.assert_allclose(a.P, b.P, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.Q, b.Q, rtol=0, atol=1e-12)
    assert np.abs(a.P).max() > 0.1  # a model with dynamics, not a trivial one
    no_spillover = MAIN.replace("@#define spillover = true", "@#define spillover = false")
    (tmp_path / "closed.mod").write_text(no_spillover, encoding="utf-8")
    closed = read_mod(tmp_path / "closed.mod")  # @#if on the define: no omega, no spillovers (ifdef still sees the name)
    assert "omega" not in closed.model.parameters.names and all("omega" not in e for e in closed.equations)
    with pytest.raises(ModelError, match="needs the model read from its file"):
        read_mod(MAIN)  # text without a file: the include cannot be found


def test_the_editor_checks_after_expansion():
    text = ("@#define shocks = [\"ey\", \"ei\"]\nvar y i;\nvarexo\n@#for s in shocks\n  @{s}\n@#endfor\n;\nparameters rho;\nrho = 0.5;\n"
            "model;\ny = rho*y(-1) + ey;\ni = rho*i(-1) + eii;\nend;\n")
    found = ModelSource(text, "dynare").check()
    errors = [d for d in found.diagnostics if d.severity == "error"]
    assert [(d.line, d.column) for d in errors] == [(12, 17)] and "eii" in errors[0].message  # at its own line and column
    good = ModelSource(text.replace("eii", "ei"), "dynare")
    assert good.check().ok and {s.name: s.line for s in good.check().symbols}["ey"] == 5  # the loop's line
    np.testing.assert_allclose(good.build().solve().P, read_mod(text.replace("eii", "ei")).solve().P)
    loop_error = ModelSource(text.replace("eii", "ei").replace("y = rho*y(-1) + ey;", "@#for s in shocks\ny_@{s} = rho*y(-1) + ey;\n@#endfor"), "dynare").check()
    assert {d.line for d in loop_error.diagnostics if d.severity == "error"} == {12}  # generated twice, reported at the loop body
    bad = ModelSource(text.replace("@#endfor", "@#endfor\n@#bogus"), "dynare").check()
    assert [(d.line, d.column) for d in bad.diagnostics] == [(7, 1)] and "unknown macro directive @#bogus" in bad.diagnostics[0].message
    unclosed = ModelSource(text.replace("@#endfor\n", ""), "dynare").check()
    assert unclosed.diagnostics[0].line == 4 and "not closed" in unclosed.diagnostics[0].message
    include = ModelSource('@#include "common.mod"\n' + text, "dynare").check()
    assert include.diagnostics[0].line == 1 and "needs the model read from its file" in include.diagnostics[0].message
