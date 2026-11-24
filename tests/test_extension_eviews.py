"""Extension API 1.10: the core EViews reader's text helpers under public names (pymetrix.extension), as a plugin that
translates EViews programs imports them. Each must be the private helper itself, so it gives the same results."""

from __future__ import annotations

import pytest

PROGRAM = """' a comment, with "quotes"
smpl 1990q1 2007q4  ' the sample
series y = dlog(x) + c(2)*z(-1) _
   + @pch(w)
genr s = "it's" ' a quote inside a string is not a comment
"""


def test_public_names_are_the_private_helpers():
    from pymetrix import extension as ext
    from pymetrix.models import eviews, source
    from pymetrix.plugins import API

    assert API == "1.10"
    pairs = {"eviews_blank_comments": eviews._blank, "eviews_statements": eviews._statements,
             "parse_eviews_expression": eviews._parse, "render_eviews_expression": eviews._render,
             "walk_eviews_expression": eviews._walk, "EViewsExpressionError": eviews._Refused,
             "EViewsNumber": eviews._Num, "EViewsName": eviews._Ser, "EViewsCoefficient": eviews._Coef,
             "EViewsCall": eviews._Call, "EViewsNegative": eviews._Neg, "EViewsBinary": eviews._Bin,
             "EViewsBrackets": eviews._Par, "Diagnostic": source.Diagnostic, "line_and_column": source._place}
    for name, private in pairs.items():
        assert getattr(ext, name) is private, name
        assert name in ext.__all__


def test_statements_keep_offsets():
    from pymetrix.extension import eviews_blank_comments, eviews_statements

    clean = eviews_blank_comments(PROGRAM)
    assert len(clean) == len(PROGRAM)
    statements = eviews_statements(clean)
    assert [t.split()[0] for t, _ in statements] == ["smpl", "series", "genr"]
    for text, at in statements:  # each statement starts where it is in the program
        assert PROGRAM[at:at + 6] == text[:6]
    joined = statements[1][0]
    assert "@pch(w)" in joined and "_" not in joined.replace("@pch", "")
    assert statements[2][0].rstrip().endswith('"it\'s"')  # the quote inside the string is kept


def test_parse_and_render_match_convert_expression():
    from pymetrix.extension import (EViewsBinary, EViewsCall, EViewsCoefficient, EViewsName, parse_eviews_expression,
                                    render_eviews_expression, walk_eviews_expression)
    from pymetrix.models.eviews import convert_expression

    cases = [("dlog(x) + c(2)*z(-1) + @pch(w)", {}), ("d(x, 0, 4)", {}), ("@pchy(y)", {"frequency": "Q"}),
             ("@recode(a > b, a, b)", {}), ("alpha*y(-1)", {"scalars": ("alpha",)}), ("@movav(x, 4)/@inv(z)", {})]
    for text, options in cases:
        tree = parse_eviews_expression(text, 0, {"c"})
        assert render_eviews_expression(tree, **options) == convert_expression(text, **options)
    base = 40  # offsets are the document's: the text's own plus base
    tree = parse_eviews_expression("dlog(x) + c(2)*z(-1)", base, {"c"})
    nodes = walk_eviews_expression(tree)
    assert isinstance(tree, EViewsBinary) and tree.op == "+"
    call = next(n for n in nodes if isinstance(n, EViewsCall))
    assert (call.fn, call.at) == ("dlog", base)
    coef = next(n for n in nodes if isinstance(n, EViewsCoefficient))
    assert (coef.vector, coef.index, coef.at) == ("c", 2, base + 10)
    lagged = next(n for n in nodes if isinstance(n, EViewsName) and n.name == "z")
    assert (lagged.lag, lagged.at, lagged.length) == (-1, base + 15, 5)


def test_refusals_carry_their_place():
    from pymetrix.errors import SpecificationError
    from pymetrix.extension import Diagnostic, EViewsExpressionError, line_and_column, parse_eviews_expression
    from pymetrix.models.eviews import _parse

    text = "y = log(x) + @year(z)"
    with pytest.raises(EViewsExpressionError) as public:
        parse_eviews_expression(text[4:], 4, {"c"})
    with pytest.raises(SpecificationError) as private:
        _parse(text[4:], 4, {"c"})
    assert (public.value.message, public.value.at, public.value.length) == (private.value.message, private.value.at,
                                                                         private.value.length)
    assert text[public.value.at:public.value.at + public.value.length] == "@year"
    program = "smpl @all\n" + text
    line, column = line_and_column(program, 10 + public.value.at)
    d = Diagnostic(line, column, public.value.message, "error", public.value.length)
    assert (d.line, d.column) == (2, 14) and str(d).startswith("line 2, column 14: error: @year")

