"""EViews models as text: a model's text (what its Text view shows, what ``m.append`` takes) and a subset of EViews
programs (``.prg``), read into a pymetrix ``Template``; and the FRB/US-style XML model files.

    nk = read_eviews(text)                          # model text or a program; ModelSource(text, "eviews").build()
    fitted = read_eviews(prg, data=data)            # its "equation eq.ls ..." specifications fitted to data first
    frbus = read_eviews_xml("model.xml")            # an XML model file in the format of the Federal Reserve's FRB/US package

Written without EViews itself; the subset was first written from how EViews models are commonly written, then confirmed
and widened against J.-L. Brillet, "Structural Econometric Modelling: Methodology and Tools with Applications under
EViews" (its programs, model listings and its account of @elem and the time trends) and the Federal Reserve's public
FRB/US packages (model.xml with the Fed's Python twins of its EViews equations, and the supply-side state-space
programs). What neither settles stays refused; every refusal says where and why. What is read:

Model text (one statement a line; ``'`` starts a comment; a line ending in `` _`` continues on the next; names in any
case, read in lower case, as EViews treats them):

- equations in EViews expression syntax: ``cons = c(1) + c(2)*inc + c(3)*cons(-1)``; lags ``x(-1)``, leads ``x(+1)`` or
  ``x(1)``; ``+ - * / ^`` (``a^b^c`` and ``-a^b`` are refused as ambiguous: write the brackets); the variable an equation
  determines is the first series of the current period on its left (EViews's rule): ``dlog(cons) = ...``,
  ``y - y_aerr = ...`` (terms on the left without it are moved to the right); an equation whose left side mixes it
  with other series of the current period, as ``y/x = ...``, is kept as ``y = y - ((y/x) - (rhs))``.
- functions: ``log``/``@log``, ``exp``/``@exp``, ``abs``/``@abs``, ``sqr``/``@sqrt`` (the square root), ``@inv``
  (1/x), ``d(x)``, ``d(x, n)``, ``d(x, n, s)`` (n first differences and one of span s), ``dlog(x[, n[, s]])`` (the
  same of log x), ``@pch(x)`` (x/x(-1) - 1: a fraction, not percent; pymetrix's ``pch`` is 100 times it), the trailing
  windows ``@movav``, ``@movsum``, ``@movmax``, ``@movmin`` (x and the n - 1 periods before it), the normal ``@cnorm``,
  ``@dnorm``, ``@qnorm``, Student's ``@ctdist``/``@dtdist``/``@qtdist``, ``@cchisq``/``@dchisq``/``@qchisq`` and
  ``@cfdist``/``@dfdist``/``@qfdist``; ``@recode(a > b, a, b)`` (and ``>=``, ``<``, ``<=``, either order of the two)
  as ``max`` or ``min``, an occasionally binding constraint, and any other ``@recode(condition, a, b)`` as
  ``iif(condition, a, b)``, as ``@iif``; ``@log10``, ``@floor``, ``@ceiling``, ``@round(x)`` (``round``: halves away
  from zero; EViews's ``@round`` has no number of decimals), ``@isna``, ``@nan(x, v)`` (``fillna``) and ``na`` (the
  missing value); comparisons in brackets, 1 where they
  hold and 0 where not, as the book's ``(t <= 2004)`` (``=`` and ``<>`` inside brackets are equality and inequality);
  ``@pc`` (100*@pch), and with the data's frequency (``read_eviews(frequency=)``, ``wfcreate q``, a dated ``smpl``, or
  the data) ``@pchy`` (x/x(-4) - 1 in quarterly data), ``@pcy`` (100 times it), ``@pcha`` ((x/x(-1))^4 - 1) and
  ``@pca``; ``@trend(base)`` (periods since base: 1960q1, "1978Q2", 62Q2, 60:1 in the data's own frequency) and
  ``@trend`` (from the data's first period), ``@seas(n)`` and ``@elem(x, "2004q4")`` (x's value then, a constant) as
  series named ``trend_1960_1``, ``trend_start``, ``seas_2``, ``x_at_2004q4``, which ``eviews_series`` makes from a
  dataset's dates; ``@elem(@trend(base), "2004q4")`` is the number itself. Refused, at the function: ``@max``/``@min``
  (a sample's maximum, a number: ``@recode`` is the elementwise one), the moving variances and standard deviations
  (their degrees of freedom are not settled by either source), ``and``/``or`` (multiply comparisons instead), dates
  without their frequency in ``@elem``, two-digit years before 30, and every other function. The functions of the
  data's rows are refused in a model, saying so, and read in data work (``convert_expression(data=True)``, the program
  translator): ``@year``, ``@quarter``, ``@month``, ``@day`` (``year()`` ...: of each period's first day), ``@obsnum``
  and ``@obsid`` (``obsnum()``, from 1 within each entity: EViews's ``@obsnum`` counts a panel's rows straight
  through), ``@cumsum(x)`` (``cumsum``: NAs skipped, as EViews's), ``@sumsby``, ``@meansby``, ``@minsby`` and
  ``@maxsby(x, g)`` (``sum(x, by=g)`` ..., g a series' name), and, with a seed, ``nrnd``/``@nrnd``/``@rnorm``
  (``normrnd(seed=s)``), ``rnd``/``@rnd`` and ``@runif(a, b)`` (``unifrnd``): pymetrix's own generator, so the numbers
  are never EViews's. Pymetrix computes ``cumsum`` and the by-group statistics over all the data's rows, EViews over
  the current sample. Inside ``d``, ``dlog``, ``@pch`` and the moving windows they are refused (lagging them is not
  lagging their rows).
- ``@identity y = ...``: an identity (no coefficients; no add factor or innovation).
- coefficients: ``c(1)``, ``c(2)`` ... (EViews's default vector) and named vectors ``beta(1)`` declared with
  ``coef(n) beta`` (or ``vector(n) beta``) or given values as ``beta(1) = ...``, become the parameters ``c_1``,
  ``beta_1`` (the names the FRB/US package's Python twins of its equations use); ``scalar`` objects are parameters by
  their names. Values: ``c(1) = 0.5``, ``param c(1) .5 c(2) .3``, ``scalar rho = 0.9``, ``coef(3) beta = 0.1`` (each
  element), or ``read_eviews(params={"c_1": 0.5})`` (``{"p_x(1)": 1}`` also declares the vector p_x), which wins.
  ``scalar f`` declares a scalar whose value comes later or from ``params``. A coefficient without a value is an error
  at its place (a scalar's, once, where it is declared). A named vector not declared reads as a series with a lead:
  ``p_x(1)`` is ``p_x[+1]``, as it would be in EViews without the coefficient object.
- ``@add y y_a`` (``y_a`` by default): an add factor, a series added to the right of the equation for ``y`` (EViews's
  intercept shift); ``@add(v)`` (a shift of the variable) is read the same where the left side is ``y`` alone, refused
  otherwise (neither source settles how EViews shifts a variable inside a function). Add factors are exogenous series, zero unless the data say otherwise.
- ``@innov y sd``: the equation for ``y`` gets a shock ``y_innov`` of standard deviation ``sd`` (added as the add
  factor is): the template's shocks, for its impulse responses and stochastic simulations.
- ``:eq`` (a linked equation, as model text shows one) and ``m.merge eq``: the equation object ``eq`` specified in the
  same text (below); refused when it is not.

Programs (``.prg``): ``model m``, ``m.append <a line of model text>``, ``m.merge eq``, ``equation eq.ls <spec>`` (by
least squares: a list ``y c y(-1) x``, ``c`` the intercept and the coefficients c(1), c(2) ... in the order listed, or
an expression ``y = c(1) + c(2)*x``; options in brackets are ignored), ``scalar``, ``coef``/``vector``, ``c(1) = ...``,
``param``, ``smpl first last`` (the sample of the specifications after it: ``1990q1``, ``1990m3``, ``1990``,
``@first``, ``@last``, ``@all``) and ``series x = <number>`` or ``x = <number>`` (a constant: an exogenous series'
steady-state value, or an endogenous variable's starting guess). A text with ``model``, ``equation`` or ``.append`` is a
program: its bare ``x = ...`` lines are series assignments, not equations. A specification is fitted to
``read_eviews(data=..., span=...)`` (``Equation.fit``, by least squares, on its ``smpl``); without data, a list
specification is ``c(1) + c(2)*y(-1) + ...`` with the values given to ``c``, as EViews's coefficient vector holds them.
Workfile, database, display and solve commands (``wfcreate``, ``fetch``, ``show``, ``m.solve``, ``m.addassign``,
series generation ...) are ignored with a note; the programming language (``for``/``next``, ``if``/``endif``,
``while``, ``subroutine``, ``call``, ``include``, ``!n`` and ``%s`` variables, ``{%s}`` substitution) and every
other estimation method are refused at their place.

``read_eviews_xml`` reads the XML model format of the Federal Reserve's public FRB/US package (``mods/model.xml``:
``<variable>`` with ``<name>``, ``<equation_type>``, ``<sector>``, ``<definition>``, ``<standard_equation>`` or
``<mce_equation>`` with ``<eviews_equation>`` and ``<coeff>`` values): the equations through the reader above, the
coefficients ``y_x(1)`` as parameters ``y_x_1``, the definitions as labels, and each equation's ``<stochastic_type>``
(``EViewsXmlModel.stochastic``: ``"NO"`` for an equation the Fed's stochastic simulations do not shock).
"""

from __future__ import annotations

import difflib
import itertools
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._api import api
from ..errors import SpecificationError
from .solvers import ModelError

__all__ = ["EViewsXmlModel", "convert_expression", "eviews_series", "eviews_expression", "read_eviews", "read_eviews_sspace", "read_eviews_xml"]

# ---------------------------------------------------------------------------------------------
# expressions

FUNCTIONS: dict[str, tuple[str, str]] = {  # name: (signature, what it does)
    "log": ("log(x)", "the natural logarithm"), "@log": ("@log(x)", "the natural logarithm"),
    "exp": ("exp(x)", "e to the power x"), "@exp": ("@exp(x)", "e to the power x"),
    "abs": ("abs(x)", "the absolute value"), "@abs": ("@abs(x)", "the absolute value"),
    "sqr": ("sqr(x)", "the square root (EViews's sqr is the root, not the square)"), "@sqrt": ("@sqrt(x)", "the square root"),
    "@inv": ("@inv(x)", "1/x"),
    "d": ("d(x[, n[, s]])", "n first differences (1 by default) and one of span s: d(x) = x - x(-1)"),
    "dlog": ("dlog(x[, n[, s]])", "d of log(x): dlog(x) = log(x) - log(x(-1))"),
    "@pch": ("@pch(x)", "the change as a fraction, x/x(-1) - 1"), "@pc": ("@pc(x)", "the change in percent, 100*@pch(x)"),
    "@pchy": ("@pchy(x)", "the change over a year as a fraction, x/x(-4) - 1 in quarterly data (the data's frequency)"),
    "@pcy": ("@pcy(x)", "100*@pchy(x)"),
    "@pcha": ("@pcha(x)", "the change annualised, (x/x(-1))^4 - 1 in quarterly data (the data's frequency)"),
    "@pca": ("@pca(x)", "100*@pcha(x)"),
    "@trend": ("@trend(base)", "the periods since base (1960q1, 60:1, ...): a series pymetrix makes from the data's dates "
               "(eviews_series); @trend alone counts from the data's first period"),
    "@elem": ("@elem(x, \"2004q4\")", "x's value at that date, a constant (of @trend: the number)"),
    "@seas": ("@seas(n)", "1 in the n-th period of the year, else 0 (a series from the data's dates)"),
    "@movav": ("@movav(x, n)", "the average of x and the n - 1 periods before it"),
    "@movsum": ("@movsum(x, n)", "the sum of x and the n - 1 periods before it"),
    "@movmax": ("@movmax(x, n)", "the largest of x and the n - 1 periods before it"),
    "@movmin": ("@movmin(x, n)", "the smallest of x and the n - 1 periods before it"),
    "@recode": ("@recode(a > b, a, b)", "a where the condition holds, else b: read as max(a, b) (or min), its only form here"),
    "@cnorm": ("@cnorm(x)", "the standard normal cumulative distribution function"),
    "@dnorm": ("@dnorm(x)", "the standard normal density"), "@qnorm": ("@qnorm(p)", "the standard normal quantile"),
    "@ctdist": ("@ctdist(x, v)", "Student's t cumulative distribution function"),
    "@dtdist": ("@dtdist(x, v)", "Student's t density"), "@qtdist": ("@qtdist(p, v)", "Student's t quantile"),
    "@cchisq": ("@cchisq(x, v)", "the chi-squared cumulative distribution function"),
    "@dchisq": ("@dchisq(x, v)", "the chi-squared density"), "@qchisq": ("@qchisq(p, v)", "the chi-squared quantile"),
    "@cfdist": ("@cfdist(x, v1, v2)", "the F cumulative distribution function"),
    "@dfdist": ("@dfdist(x, v1, v2)", "the F density"), "@qfdist": ("@qfdist(p, v1, v2)", "the F quantile"),
    "@log10": ("@log10(x)", "the base-10 logarithm"), "@floor": ("@floor(x)", "the largest whole number not above x"),
    "@ceiling": ("@ceiling(x)", "the smallest whole number not below x"),
    "@round": ("@round(x)", "x rounded to the nearest whole number, halves away from zero"),
    "@isna": ("@isna(x)", "1 where x is NA, else 0"), "@nan": ("@nan(x, v)", "v where x is NA, else x"),
    "@iif": ("@iif(condition, a, b)", "a where the condition holds, b where not (NA where the condition is NA)"),
    "na": ("na", "the missing value"),
    "@year": ("@year", "the year of each observation (data only: refused in a model)"),
    "@quarter": ("@quarter", "the quarter of each observation (data only: refused in a model)"),
    "@month": ("@month", "the month of each observation's first day (data only: refused in a model)"),
    "@day": ("@day", "the day of the month of each observation's first day (data only: refused in a model)"),
    "@obsnum": ("@obsnum", "the observation's number, from 1 (data only: refused in a model)"),
    "@obsid": ("@obsid", "the observation's number within its cross-section, from 1 (data only: refused in a model)"),
    "nrnd": ("nrnd", "a standard normal random draw for each observation (data only, seeded: refused in a model)"),
    "@nrnd": ("@nrnd", "as nrnd"), "@rnorm": ("@rnorm", "as nrnd"),
    "rnd": ("rnd", "a uniform random draw on (0, 1) for each observation (data only, seeded: refused in a model)"),
    "@rnd": ("@rnd", "as rnd"), "@runif": ("@runif(a, b)", "a uniform random draw on (a, b) for each observation (data only)"),
    "@cumsum": ("@cumsum(x)", "the sum of x up to each observation, NAs skipped (data only)"),
    "@sumsby": ("@sumsby(x, g)", "the sum of x over the observations with the same value of the series g (data only)"),
    "@meansby": ("@meansby(x, g)", "the mean of x over the observations with the same value of g (data only)"),
    "@minsby": ("@minsby(x, g)", "the smallest x among the observations with the same value of g (data only)"),
    "@maxsby": ("@maxsby(x, g)", "the largest x among the observations with the same value of g (data only)"),
}
# functions written without brackets in EViews (``series yr = @year``, ``e = nrnd``); empty brackets are also read
_NULLARY = frozenset({"na", "@year", "@quarter", "@month", "@day", "@obsnum", "@obsid", "nrnd", "@nrnd", "@rnorm", "rnd", "@rnd"})
# Functions of the data's rows (pymetrix.expr.nodes.DATA_ONLY) and of the sample: the calendar, the observation's
# number, random draws, and the cumulative and by-group statistics. A model has no rows: read in data work
# (render_eviews_expression(..., data=True), as the program translator does), refused in a model's equations.
_CALENDAR = {"@year": "year()", "@quarter": "quarter()", "@month": "month()", "@day": "day()", "@obsnum": "obsnum()",
             "@obsid": "obsnum()"}
_RANDOM = {"nrnd": "normrnd", "@nrnd": "normrnd", "@rnorm": "normrnd", "rnd": "unifrnd", "@rnd": "unifrnd", "@runif": "unifrnd"}
_BY = {"@sumsby": "sum", "@meansby": "mean", "@minsby": "min", "@maxsby": "max"}
DATA_FUNCTIONS = frozenset(_CALENDAR) | frozenset(_RANDOM) | frozenset(_BY) | {"@cumsum"}
# functions whose argument is taken at earlier periods too: a function of the data's rows inside one is refused
_LAGGING = frozenset({"d", "dlog", "@pch", "@pc", "@pchy", "@pcy", "@pcha", "@pca", "@movav", "@movsum", "@movmax", "@movmin"})
_SIMPLE = {"log": "log", "@log": "log", "exp": "exp", "@exp": "exp", "abs": "abs", "@abs": "abs", "sqr": "sqrt", "@sqrt": "sqrt",
           "@movav": "movavg", "@movsum": "movsum", "@movmax": "movmax", "@movmin": "movmin",
           "@cnorm": "normcdf", "@dnorm": "normpdf", "@qnorm": "norminv", "@ctdist": "tcdf", "@dtdist": "tpdf", "@qtdist": "tinv",
           "@cchisq": "chi2cdf", "@dchisq": "chi2pdf", "@qchisq": "chi2inv", "@cfdist": "fcdf", "@dfdist": "fpdf", "@qfdist": "finv",
           "@log10": "log10", "@floor": "floor", "@ceiling": "ceil", "@round": "round", "@isna": "isna", "@nan": "fillna",
           "@iif": "iif"}
_ARITY = {**{f: (1, 1) for f in ("log", "@log", "exp", "@exp", "abs", "@abs", "sqr", "@sqrt", "@inv", "@pch", "@cnorm", "@dnorm",
                                  "@qnorm", "@pc", "@pchy", "@pcy", "@pcha", "@pca", "@log10", "@floor", "@ceiling", "@round",
                                  "@isna", "@cumsum")},
          **{f: (2, 2) for f in ("@movav", "@movsum", "@movmax", "@movmin", "@ctdist", "@dtdist", "@qtdist", "@cchisq", "@dchisq",
                                  "@qchisq", "@nan", "@runif", *_BY)},
          **{f: (3, 3) for f in ("@cfdist", "@dfdist", "@qfdist", "@recode", "@iif")}, "d": (1, 3), "dlog": (1, 3),
          **{f: (0, 0) for f in _NULLARY}}
_WHY = {"@max": "@max is a sample's maximum (a number), not elementwise: write @recode(a > b, a, b)",
        "@min": "@min is a sample's minimum (a number), not elementwise: write @recode(a < b, a, b)",
        "@movstdev": "the degrees of freedom of EViews's moving standard deviations are not settled here: write it out",
        "@movvar": "the degrees of freedom of EViews's moving variances are not settled here: write it out",
        "@obssmpl": "the sample's number of observations is a number: give a scalar its value first (!n = @obssmpl)",
        "@date": "EViews's date numbers are not a pymetrix series: use @year, @quarter, @month and @day"}
RESERVED = frozenset({"abs", "acos", "and", "ar", "asin", "c", "con", "cnorm", "coef", "cos", "d", "dlog", "dnorm", "else", "endif",
                      "exp", "log", "logit", "lpt1", "lpt2", "ma", "na", "not", "nrnd", "or", "pdl", "rnd", "sar", "sin", "sma",
                      "sqr", "then"})


class _Refused(SpecificationError):
    """Text the reader refuses: why, and where (an offset in the document, and how many characters)."""

    def __init__(self, message: str, at: int, length: int = 1) -> None:
        super().__init__(message)
        self.message, self.at, self.length = message, at, length


@dataclass(frozen=True)
class _Num:
    """A number as written (``text``), at offset ``at``."""

    text: str
    at: int


@dataclass(frozen=True)
class _Ser:  # a series (or a scalar), at a lag
    """A series or scalar ``name`` (lower case) at ``lag`` (negative: a lag, positive: a lead), spanning ``length``
    characters from ``at``."""

    name: str
    lag: int
    at: int
    length: int


@dataclass(frozen=True)
class _Coef:
    """A coefficient ``vector(index)``, such as ``c(2)``."""

    vector: str
    index: int
    at: int
    length: int


@dataclass(frozen=True)
class _Call:
    """A function call: ``fn`` as written in lower case (``"dlog"``, ``"@pch"``) and its argument nodes."""

    fn: str
    args: tuple[Any, ...]
    at: int
    length: int


@dataclass(frozen=True)
class _Neg:
    """A minus sign before ``x``."""

    x: Any
    at: int


@dataclass(frozen=True)
class _Bin:
    """``left op right``: an arithmetic operator (``+ - * / ^``) or a comparison (``< > <= >= <> =``)."""

    op: str
    left: Any
    right: Any
    at: int


@dataclass(frozen=True)
class _Par:
    """``x`` in brackets, as written."""

    x: Any
    at: int


_TOKEN = re.compile(r"\s*(?:(?P<str>\"[^\"]*\"?)|(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|(?P<name>@?[A-Za-z_]\w*)"
                    r"|(?P<op><>|<=|>=|[-+*/^(),<>=:])|(?P<bad>\S))")
_PROGRAMMING = {"{": "string substitution ({%s}) is part of EViews's programming language, which is not read: write the text out",
                "%": "%string variables are part of EViews's programming language, which is not read: write the text out",
                "!": "!control variables are part of EViews's programming language, which is not read: write the number"}


class _Parser:
    """An EViews expression, with the positions of its parts (offsets in the document, ``base`` the text's own)."""

    def __init__(self, text: str, base: int, vectors: set[str]) -> None:
        self.tokens: list[tuple[str, str, int]] = []
        pos = 0
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if m is None or m.end() == pos:
                break
            kind = m.lastgroup or ""
            if kind == "bad":
                ch = m.group("bad")
                raise _Refused(_PROGRAMMING.get(ch, f"{ch!r} does not belong in an EViews expression"), base + m.start(kind))
            self.tokens.append((kind, m.group(kind), base + m.start(kind)))
            pos = m.end()
        self.end = base + len(text.rstrip())
        self.i = 0
        self.vectors = vectors

    def peek(self) -> tuple[str, str, int]:
        return self.tokens[self.i] if self.i < len(self.tokens) else ("end", "", self.end)

    def take(self) -> tuple[str, str, int]:
        t = self.peek()
        self.i += 1
        return t

    def expect(self, value: str) -> tuple[str, str, int]:
        t = self.take()
        if t[1] != value:
            raise _Refused(f"expected {value!r}" + (f", not {t[1]!r}" if t[1] else " before the end"), t[2], max(len(t[1]), 1))
        return t

    def whole(self, comparison: bool = False) -> Any:
        node = self.compare() if comparison else self.sum()
        t = self.peek()
        if t[0] != "end":
            if t[1].lower() in ("and", "or", "not"):
                raise _Refused(f"{t[1]!r}: logical operators are not read: multiply comparisons instead, (a > 0)*(b > 0)", t[2],
                               len(t[1]))
            what = "an unmatched ')'" if t[1] == ")" else f"{t[1]!r}"
            raise _Refused(f"{what} does not belong here" + (": a comparison goes in brackets, as (t <= 2004)" if t[1] in "<>=" else ""),
                           t[2], max(len(t[1]), 1))
        return node

    def compare(self) -> Any:
        left = self.sum()
        t = self.peek()
        if t[1] in ("<", ">", "<=", ">=", "<>", "="):
            self.take()
            return _Bin(t[1], left, self.sum(), t[2])
        return left

    def sum(self) -> Any:
        node = self.product()
        while self.peek()[1] in ("+", "-"):
            t = self.take()
            node = _Bin(t[1], node, self.product(), t[2])
        return node

    def product(self) -> Any:
        node = self.unary()
        while self.peek()[1] in ("*", "/"):
            t = self.take()
            node = _Bin(t[1], node, self.unary(), t[2])
        return node

    def unary(self) -> Any:
        t = self.peek()
        if t[1] in ("-", "+"):
            self.take()
            x = self.unary()
            if isinstance(x, _Bin) and x.op == "^":
                raise _Refused(f"{t[1]}a^b reads differently in different languages: write {t[1]}(a^b) or ({t[1]}a)^b", t[2])
            return _Neg(x, t[2]) if t[1] == "-" else x
        return self.power()

    def power(self) -> Any:
        base = self.atom()
        if self.peek()[1] == "^":
            t = self.take()
            sign = self.peek()
            if sign[1] in ("-", "+"):
                self.take()
                exponent = self.atom()
                exponent = _Neg(exponent, sign[2]) if sign[1] == "-" else exponent
            else:
                exponent = self.atom()
            if self.peek()[1] == "^":
                raise _Refused("a^b^c reads differently in different languages: write (a^b)^c or a^(b^c)", self.peek()[2])
            return _Bin("^", base, exponent, t[2])
        return base

    def integer(self) -> tuple[int, int]:
        """A signed integer in brackets after a name: its value and the closing bracket's place."""
        sign = 1
        t = self.take()
        if t[1] in ("-", "+"):
            sign = -1 if t[1] == "-" else 1
            t = self.take()
        if t[0] != "num" or not t[1].isdigit():
            raise _Refused("", t[2], max(len(t[1]), 1))
        close = self.expect(")")
        return sign * int(t[1]), close[2]

    def atom(self) -> Any:
        kind, value, at = self.take()
        if kind == "num":
            return _Num(value, at)
        if value == "(":
            inner = self.compare()
            if self.peek()[1] != ")":
                raise _Refused("this bracket is never closed", at)
            self.take()
            return _Par(inner, at)
        if kind == "str":
            raise _Refused("a string does not belong in a model's equation", at, len(value))
        if kind != "name":
            raise _Refused("expected a number, a name or '('" + (f", not {value!r}" if value else " before the end"), at,
                           max(len(value), 1))
        name = value.lower()
        if name in ("and", "or", "not"):
            raise _Refused(f"{value!r}: logical operators are not read: multiply comparisons instead, (a > 0)*(b > 0)", at, len(value))
        if name in _NULLARY:  # @year, nrnd, na: no arguments (empty brackets read too)
            if self.peek()[1] == "(":
                self.take()
                if self.peek()[1] != ")":
                    raise _Refused(f"{value} takes no arguments: {FUNCTIONS[name][0]}, {FUNCTIONS[name][1]}", at, len(value))
                return _Call(name, (), at, self.take()[2] - at + 1)
            return _Call(name, (), at, len(value))
        if self.peek()[1] != "(":
            if name == "@trend":  # counted from the data's first period
                return _Ser("trend_start", 0, at, len(value))
            if name in _WHY:
                raise _Refused(f"{value} is not read here: {_WHY[name]}", at, len(value))
            if name.startswith("@"):
                raise _Refused(f"{value} needs its arguments in brackets", at, len(value))
            return _Ser(name, 0, at, len(value))
        open_ = self.take()
        if name in self.vectors:
            save = self.i
            try:
                index, close = self.integer()
            except _Refused:
                self.i = save
                raise _Refused(f"{value}(...) is a coefficient: {value}(1), {value}(2) ...", at, len(value)) from None
            if index < 1:
                raise _Refused(f"coefficients count from 1: {value}({index})", at, close - at + 1)
            return _Coef(name, index, at, close - at + 1)
        if name in ("@trend", "@elem", "@seas"):
            return self.dated(name, value, at)
        if name in FUNCTIONS:
            args = [self.compare() if name in ("@recode", "@iif") and not args_ else self.sum() for args_ in [[]]]
            while self.peek()[1] == ",":
                self.take()
                args.append(self.sum())
            if self.peek()[1] != ")":
                raise _Refused("this bracket is never closed" if self.peek()[0] == "end" else
                               f"expected ',' or ')', not {self.peek()[1]!r}", open_[2] if self.peek()[0] == "end" else self.peek()[2])
            close = self.take()
            low, high = _ARITY[name]
            if not low <= len(args) <= high:
                raise _Refused(f"{FUNCTIONS[name][0]} takes {low if low == high else f'{low} to {high}'} argument"
                               f"{'s' if high > 1 else ''}, not {len(args)}", at, close[2] - at + 1)
            return _Call(name, tuple(args), at, close[2] - at + 1)
        if name.startswith("@"):
            why = _WHY.get(name, "")
            raise _Refused(f"{value} is not read here" + (f": {why}" if why else "; the functions read are "
                           + ", ".join(sorted(f for f in FUNCTIONS if f.startswith("@")))), at, len(value))
        save = self.i
        try:
            lag, close = self.integer()
        except _Refused:
            self.i = save
            close_at = next((t[2] for t in self.tokens[self.i:] if t[1] == ")"), at + len(value))
            if name in ("dlog", "d"):
                raise
            known = difflib.get_close_matches(name, list(FUNCTIONS), 1, 0.75)
            raise _Refused(f"{value}(...) is neither a lag or lead, as {value}(-1), nor a function read here"
                           + (f"; did you mean {known[0]}?" if known else ""), at, max(close_at - at + 1, len(value))) from None
        return _Ser(name, lag, at, close - at + 1)


    def date_text(self) -> tuple[str, int]:
        """The tokens up to the next ',' or ')' joined (a date as EViews writes it: 1960q1, "1978Q2", 60:1), and where."""
        start, parts = self.peek()[2], []
        while self.peek()[1] not in (")", ",") and self.peek()[0] != "end":
            parts.append(self.take()[1])
        return "".join(parts).strip('"'), start

    def dated(self, name: str, value: str, at: int) -> Any:
        """``@trend(base)``, ``@elem(x, "date")`` and ``@seas(n)``: series pymetrix makes from the data's dates
        (``eviews_series``), or a number."""
        if name == "@trend":
            if self.peek()[1] == ")":
                close = self.take()
                return _Ser("trend_start", 0, at, close[2] - at + 1)
            text, where = self.date_text()
            date = _date(text)
            if date is None:
                raise _Refused(f"{text!r} is not a date read here: 1960q1, 1960m3, 1960s1, 1960 or 60:1", where, max(len(text), 1))
            close = self.expect(")")
            return _Ser(f"trend_{_date_token(date)}", 0, at, close[2] - at + 1)
        if name == "@seas":
            t = self.take()
            if t[0] != "num" or not t[1].isdigit() or int(t[1]) < 1:
                raise _Refused("@seas takes the period of the year: @seas(1)", t[2], max(len(t[1]), 1))
            close = self.expect(")")
            return _Ser(f"seas_{int(t[1])}", 0, at, close[2] - at + 1)
        first = _strip(self.sum())
        self.expect(",")
        text, where = self.date_text()
        date = _date(text)
        if date is None or date[1] == ":":
            raise _Refused(f"{text!r} is not a date read here: 2004q4, 2004m3, 2004s2 or 2004 (the frequency in the date)", where,
                           max(len(text), 1))
        close = self.expect(")")
        length = close[2] - at + 1
        if isinstance(first, _Ser) and first.name.startswith("trend_") and first.lag == 0:
            number = _trend_at(first.name, date)
            if number is None:
                raise _Refused("@elem of @trend needs the date's frequency in the date itself: 2004q4", where, max(len(text), 1))
            return _Num(str(number), at)
        if isinstance(first, _Ser) and first.lag == 0 and first.name not in self.vectors:
            return _Ser(f"{first.name}_at_{_date_token(date)}", 0, at, length)
        raise _Refused("@elem is read for a series or @trend: @elem(x, \"2004q4\"), the value of x then, a constant", at, length)


def _date(text: str) -> tuple[int, str, int] | None:
    """A date as EViews writes one (1960q1, 1960m3, 1960s1, 1960, 60:1, 62Q2): year, kind (q, m, s, a, or : for the
    workfile's own frequency) and period. Two-digit years from 30 are 19xx; earlier ones are not read (ambiguous)."""
    m = re.fullmatch(r"(\d{2}|\d{4})(?:([qQmMsS:])(\d{1,2}))?", text.strip())
    if m is None:
        return None
    year = int(m.group(1))
    if len(m.group(1)) == 2:
        if year < 30:
            return None
        year += 1900
    kind = (m.group(2) or "a").lower()
    period = int(m.group(3) or 1)
    per_year = {"q": 4, "m": 12, "s": 2}.get(kind)
    if per_year is not None and not 1 <= period <= per_year:
        return None
    return year, kind, period


def _date_token(date: tuple[int, str, int]) -> str:
    year, kind, period = date
    return str(year) if kind == "a" else f"{year}_{period}" if kind == ":" else f"{year}{kind}{period}"


_PER_YEAR = {"a": 1, "s": 2, "q": 4, "m": 12}


def _trend_at(name: str, date: tuple[int, str, int]) -> int | None:
    """``@trend(base)`` at ``date``, when the date says the frequency: periods from the base."""
    m = re.fullmatch(r"trend_(\d{4})(?:([qms_])(\d+))?", name)
    if m is None or date[1] == ":":
        return None
    f = _PER_YEAR[date[1]]
    kind = m.group(2) or "a"
    if kind not in ("_", "a") and _PER_YEAR[kind] != f:
        return None
    base = int(m.group(1)) * f + (int(m.group(3)) - 1 if m.group(3) else 0)
    return date[0] * f + date[2] - 1 - base


def _parse(text: str, base: int, vectors: set[str], comparison: bool = False) -> Any:
    """One EViews expression as a tree of nodes whose offsets are ``base`` plus the text's own; ``vectors`` are the
    coefficient vectors (``{"c"}`` at least); ``comparison`` lets the whole be a comparison. ``_Refused`` where it does
    not read."""
    return _Parser(text, base, vectors).whole(comparison)


def convert_expression(text: str, scalars: tuple[str, ...] = (), vectors: tuple[str, ...] = (), frequency: str | None = None,
                       *, data: bool = False, seed: int | None = None) -> str:
    """One EViews expression as pymetrix's (``d(x, 0, 4)`` as ``(d(x) - d(x[-4]))``...); refused (``ModelError``, with
    the column) when it does not read. ``vectors``: coefficient vectors besides ``c``; ``frequency`` ("Q", "M", "A")
    for ``@pchy`` and the like. With ``data``, an expression for data work, not a model: the functions of the data's
    rows (``@year``, ``@obsnum``, ``@cumsum``, ``@sumsby`` ...) are read too, and with ``seed`` the random draws
    (``nrnd``, ``rnd``, ``@runif``), seeded ``seed``, ``seed + 1`` ... in the order they are written."""
    seeds = None if seed is None else itertools.count(int(seed)).__next__
    try:
        return _Render({s.lower() for s in scalars}, _per_year(frequency), data, seeds)(_parse(text, 0, {"c", *(v.lower() for v in vectors)}))
    except _Refused as exc:
        raise ModelError(f"column {exc.at + 1}: {exc.message}") from None


def _render(node: Any, scalars: tuple[str, ...] = (), frequency: str | None = None, *, data: bool = False,
            seeds: Callable[[], int] | None = None) -> str:
    """A parsed expression (``_parse``) as pymetrix's text, as ``convert_expression`` writes it; ``_Refused`` (with
    the place) where it does not translate. Public as ``pymetrix.extension.render_eviews_expression``. ``data``: an
    expression for data work (the data's calendar, cumulative and by-group statistics read; a model refuses them);
    ``seeds``: called for each random draw's seed (without it, random draws are refused)."""
    return _Render({s.lower() for s in scalars}, _per_year(frequency), data, seeds)(node)


def _walk(node: Any) -> list[Any]:
    """A tree's nodes, the root first, then each child's in order."""
    out = [node]
    for child in _children(node):
        out += _walk(child)
    return out


def _children(node: Any) -> tuple[Any, ...]:
    if isinstance(node, _Call):
        return node.args
    if isinstance(node, (_Neg, _Par)):
        return (node.x,)
    if isinstance(node, _Bin):
        return (node.left, node.right)
    return ()


def _strip(node: Any) -> Any:
    while isinstance(node, _Par):
        node = node.x
    return node


def _integer_arg(node: Any, what: str) -> int:
    node = _strip(node)
    if not isinstance(node, _Num) or not node.text.isdigit():
        raise _Refused(f"{what} is a whole number", getattr(node, "at", 0))
    return int(node.text)


class _Render:
    """EViews expression nodes as pymetrix text, ``k`` periods later (for d, @pch and the like)."""

    def __init__(self, scalars: set[str], per_year: int | None = None, data: bool = False,
                 seeds: Callable[[], int] | None = None) -> None:
        self.scalars = scalars
        self.per_year = per_year  # periods a year, for @pchy and the like
        self.data = data  # data work: the functions of the data's rows are read
        self.seeds = seeds  # the next random draw's seed
        self.window = 0  # > 0 inside a moving window, whose argument pymetrix lags period by period

    def rows(self, node: Any, k: int) -> str:
        """A function of the data's rows (``DATA_FUNCTIONS``) as pymetrix's, in data work; refused in a model, and
        where it would be lagged (inside d, @pch, a moving window): its lag is not the function of the lagged rows."""
        fn, args = node.fn, node.args
        if not self.data:
            if fn in _CALENDAR:
                why = (f"{fn} comes from the data's dates (data only): a model has no dates of its own; make it a series of "
                       f"the data first (series obs = {fn}) and use the series")
            elif fn in _RANDOM:
                why = (f"{fn} draws random numbers (data only): a model's random draws are its shocks (@innov); make the "
                       "series in the data first and use it")
            else:
                why = (f"{fn} is a statistic over the sample (data only): a model solves period by period; make it a series "
                       "of the data first and use the series")
            raise _Refused(why, node.at, node.length)
        if k or self.window:
            raise _Refused(f"{fn} inside d, dlog, @pch or a moving window is not read: make the series first "
                           f"(series s = {fn}{'(...)' if args else ''}), then use s", node.at, node.length)
        if fn in _CALENDAR:
            return _CALENDAR[fn]
        if fn in _RANDOM:
            if self.seeds is None:
                raise _Refused(f"{fn}: pymetrix's random draws are seeded, and no seed is given here (a program's rndseed "
                               "gives one)", node.at, node.length)
            seed = int(self.seeds())
            if fn == "@runif":
                return f"unifrnd({self(args[0])}, {self(args[1])}, seed={seed})"
            return f"{_RANDOM[fn]}(seed={seed})"
        if fn == "@cumsum":
            return f"cumsum({self(args[0])})"
        group = _strip(args[1])
        if not (isinstance(group, _Ser) and group.lag == 0 and group.name not in self.scalars):
            raise _Refused(f"{fn}(x, g) is read with g a series' name: make the series first (series g = ...), then "
                           f"{fn}(x, g)", getattr(group, "at", node.at), getattr(group, "length", node.length))
        return f"{_BY[fn]}({self(args[0])}, by={group.name})"

    def __call__(self, node: Any, k: int = 0) -> str:
        r = self
        if isinstance(node, _Call) and node.fn in DATA_FUNCTIONS:
            return self.rows(node, k)
        if isinstance(node, _Call) and node.fn == "na":
            return "na"
        if isinstance(node, _Call) and node.fn in _LAGGING and not self.window:  # what is inside is lagged too
            self.window += 1
            try:
                return self(node, k)
            finally:
                self.window -= 1
        if isinstance(node, _Num):
            return "0" + node.text if node.text.startswith(".") else node.text
        if isinstance(node, _Ser):
            if node.name in self.scalars:
                if node.lag:
                    raise _Refused(f"{node.name} is a scalar: it has no lags or leads", node.at, node.length)
                return node.name
            lag = node.lag + k
            return f"{node.name}[{lag:+d}]" if lag else node.name
        if isinstance(node, _Coef):
            return f"{node.vector}_{node.index}"
        if isinstance(node, _Par):
            return f"({r(node.x, k)})"
        if isinstance(node, _Neg):
            return f"-{r(node.x, k)}"
        if isinstance(node, _Bin):
            if node.op in ("<", ">", "<=", ">=", "<>", "="):  # 1 where it holds, else 0
                op = {"=": "==", "<>": "!="}.get(node.op, node.op)
                return f"{r(node.left, k)} {op} {r(node.right, k)}"
            space = " " if node.op in "+-" else ""
            return f"{r(node.left, k)}{space}{node.op}{space}{r(node.right, k)}"
        fn, args = node.fn, node.args
        if fn in _SIMPLE:
            if fn.startswith("@mov"):
                return f"{_SIMPLE[fn]}({r(args[0], k)}, {_integer_arg(args[1], 'the window')})"
            return f"{_SIMPLE[fn]}({', '.join(r(a, k) for a in args)})"
        if fn == "@inv":
            return f"(1/({r(args[0], k)}))"
        if fn in ("@pch", "@pc"):
            change = f"({self.wrapped(args[0], k)}/{self.wrapped(args[0], k - 1)} - 1)"
            return change if fn == "@pch" else f"(100*{change})"
        if fn in ("@pchy", "@pcy", "@pcha", "@pca"):
            s = self.per_year
            if s is None:
                raise _Refused(f"{fn} needs the data's frequency: give read_eviews(frequency=\"Q\") or state it in the text "
                               "(wfcreate q ..., or smpl with dated periods)", node.at, node.length)
            if fn in ("@pchy", "@pcy"):
                change = f"({self.wrapped(args[0], k)}/{self.wrapped(args[0], k - s)} - 1)"
            else:
                change = f"(({self.wrapped(args[0], k)}/{self.wrapped(args[0], k - 1)})^{s} - 1)"
            return change if fn in ("@pchy", "@pcha") else f"(100*{change})"
        if fn in ("d", "dlog"):
            inner = args[0] if fn == "d" else _Call("log", (args[0],), node.at, node.length)
            n = _integer_arg(args[1], "d's number of differences") if len(args) > 1 else 1
            s = _integer_arg(args[2], "d's seasonal span") if len(args) > 2 else 0
            firsts = n + (1 if s == 1 else 0)
            if s <= 1 and firsts == 1:  # the common case, as pymetrix writes it
                return f"{fn}({r(args[0], k)})"

            def differenced(shift: int) -> str:
                text = r(inner, k + shift)
                for _ in range(firsts):
                    text = f"d({text})"
                return text
            return f"({differenced(0)} - {differenced(-s)})" if s > 1 else differenced(0)
        if fn == "@recode":
            cond = _strip(args[0])
            if not (isinstance(cond, _Bin) and cond.op in ("<", ">", "<=", ">=", "=", "<>")):
                raise _Refused("@recode's first argument is a comparison: @recode(a > b, a, b)", node.at, node.length)
            x, y, a, b = (r(_strip(e), k) for e in (cond.left, cond.right, args[1], args[2]))
            greater = cond.op in (">", ">=")
            if cond.op not in ("=", "<>") and (x, y) == (a, b):  # max or min: an occasionally binding constraint
                return f"{'max' if greater else 'min'}({a}, {b})"
            if cond.op not in ("=", "<>") and (x, y) == (b, a):
                return f"{'min' if greater else 'max'}({a}, {b})"
            return f"iif({r(cond, k)}, {a}, {b})"  # otherwise a where the condition holds, b where not (as @iif)
        raise _Refused(f"{fn} is not read here", node.at, node.length)

    def wrapped(self, node: Any, k: int) -> str:
        text = self(node, k)
        return text if isinstance(_strip(node), (_Ser, _Coef, _Num, _Call)) and not text.startswith("-") else f"({text})"


def _additive(node: Any, sign: int = 1) -> list[tuple[int, Any]]:
    """The terms of a sum (not looking inside brackets), each with its sign."""
    if isinstance(node, _Bin) and node.op in ("+", "-"):
        return _additive(node.left, sign) + _additive(node.right, sign if node.op == "+" else -sign)
    return [(sign, node)]


def _current(node: Any, scalars: set[str]) -> list[_Ser]:
    return [n for n in _walk(node) if isinstance(n, _Ser) and n.lag == 0 and n.name not in scalars]


# ---------------------------------------------------------------------------------------------
# statements


@dataclass
class _Diag:
    at: int
    message: str
    severity: str = "error"
    length: int = 1


@dataclass
class _Eq:
    kind: str  # "text", "identity" or "link"
    text: str  # the equation (for a link: the equation object's name)
    at: int  # where the text starts
    statement_at: int
    label: str = ""


@dataclass
class _Spec:
    name: str
    form: str  # "list" or "expression"
    text: str
    at: int
    span: tuple[str | None, str | None] | None = None


@dataclass
class _Built:
    """An equation understood: pymetrix text, the parameters it uses, where it came from."""

    label: str
    dependent: str
    lhs: str
    rhs: str
    params: list[str]
    identity: bool
    at: int
    spec: _Spec | None = None
    lhs_alone: bool = True
    coef_places: dict[str, tuple[int, int]] = field(default_factory=dict)
    dependent_at: int = 0  # where the variable it determines is named


@dataclass
class _Read:
    """A text understood: what ``check`` reports and ``read_eviews`` builds from."""

    text: str
    program: bool = False
    model: str = ""
    diagnostics: list[_Diag] = field(default_factory=list)
    equations: list[_Eq] = field(default_factory=list)
    specs: dict[str, _Spec] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    value_at: dict[str, int] = field(default_factory=dict)
    scalars: set[str] = field(default_factory=set)
    vectors: set[str] = field(default_factory=lambda: {"c"})
    adds: list[tuple[str, str, bool, int, int]] = field(default_factory=list)  # variable, factor, variable shift, at, length
    innovs: list[tuple[str, float, int, int]] = field(default_factory=list)  # variable, sd, at, length
    declared_at: dict[str, int] = field(default_factory=dict)  # scalars declared without a value
    per_year: int | None = None  # the data's periods a year (wfcreate, smpl, frequency=, the data): for @pchy and the like
    constants: dict[str, float] = field(default_factory=dict)  # series x = number
    constant_at: dict[str, int] = field(default_factory=dict)
    built: list[_Built] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def _blank(text: str) -> str:
    """Comments (``'`` to the end of the line, outside double quotes) replaced by spaces: positions stay."""
    out = []
    for line in text.split("\n"):
        quoted, cut = False, len(line)
        for i, ch in enumerate(line):
            if ch == '"':
                quoted = not quoted
            elif ch == "'" and not quoted:
                cut = i
                break
        out.append(line[:cut] + " " * (len(line) - cut))
    return "\n".join(out)


def _statements(clean: str) -> list[tuple[str, int]]:
    """Each statement and its offset; a line ending in `` _`` continues on the next (same length: offsets stay)."""
    out: list[tuple[str, int]] = []
    pending: tuple[str, int] | None = None
    offset = 0
    for line in clean.split("\n"):
        text = line
        joined = re.search(r"(^|\s)_\s*$", text)
        if joined:
            cut = text.rfind("_")
            text = text[:cut] + " " + text[cut + 1:]
        if pending is not None:
            pending = (pending[0] + " " + text, pending[1])
        else:
            pending = (text, offset)
        if not joined:
            out.append(pending)
            pending = None
        offset += len(line) + 1
    if pending is not None:
        out.append(pending)
    return [(t, o) for t, o in out if t.strip()]


_PROGRAM_WORDS = ("for", "next", "if", "else", "endif", "while", "wend", "subroutine", "endsub", "call", "include", "exitloop",
                  "return", "stop", "then", "local", "global")
_IGNORED_COMMANDS = ("wfcreate", "wfopen", "wfclose", "wfsave", "pagecreate", "pageselect", "pagestruct", "dbopen", "dbclose",
                     "fetch", "store", "close", "show", "print", "freeze", "delete", "group", "graph", "table", "text", "spool",
                     "series", "genr", "frml", "alpha", "string", "svector", "matrix", "sym", "copy", "rename", "logmode", "mode",
                     "output", "cd", "setmaxerrs", "pon", "poff", "tic", "toc", "statusline", "smpl")
_IGNORED_METHODS = ("solve", "solveopt", "scenario", "addassign", "addinit", "makegroup", "update", "label", "display", "text",
                    "spec", "results", "output", "msg", "ls", "makeresids", "fit", "forecast", "override", "exclude", "trace", "save",
                    "store", "fetch", "displayname", "setattr", "sheet", "line", "graph", "print", "show", "freeze")


def _programming(st: str) -> tuple[int, str] | None:
    """The first piece of EViews's programming language in a statement (outside double quotes): where, and why not."""
    quoted = False
    for i, ch in enumerate(st):
        if ch == '"':
            quoted = not quoted
        elif not quoted and (ch == "{" or ch in "%!" and re.match(r"[A-Za-z_]", st[i + 1:i + 2])):
            return i, _PROGRAMMING[ch]
    return None


def _is_program(statements: list[tuple[str, int]]) -> bool:
    return any(re.match(r"\s*(model|equation)\s+\S", t, re.I) or re.match(r"\s*[A-Za-z_]\w*\.(append|merge)\b", t, re.I)
               for t, _ in statements)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _value(r: _Read, text: str, at: int) -> float | None:
    """A number, or an expression in the scalars and coefficients that have values."""
    from ..expr.evaluate import evaluate
    from ..expr.parser import parse_expression

    try:
        node = _parse(text, at, r.vectors)
        rendered = _Render(set(r.values))(node)
    except _Refused as exc:
        r.diagnostics.append(_Diag(exc.at, exc.message or "not a value", length=exc.length))
        return None
    for n in _walk(node):
        name = f"{n.vector}_{n.index}" if isinstance(n, _Coef) else n.name if isinstance(n, _Ser) else None
        if name is not None and name not in r.values:
            r.diagnostics.append(_Diag(n.at, f"{name if isinstance(n, _Ser) else _eviews_coef(name)} has no value yet: a value may "
                                       "use the scalars and coefficients given before it", length=n.length))
            return None
    try:
        return float(evaluate(parse_expression(rendered, list(r.values)), lambda name, lag: 0.0, r.values))
    except Exception as exc:  # noqa: BLE001 - arithmetic on the text's own numbers
        r.diagnostics.append(_Diag(at, f"cannot evaluate {text.strip()!r}: {exc}", length=len(text.strip())))
        return None


def _eviews_coef(name: str) -> str:
    vector, _, index = name.rpartition("_")
    return f"{vector}({index})" if index.isdigit() else name


def _set(r: _Read, name: str, value: float, at: int) -> None:
    r.values[name] = value
    r.value_at.setdefault(name, at)


def _read(text: str, vectors: set[str] | None = None) -> _Read:
    """The statements of a model text or program, each to its place in ``_Read``."""
    clean = _blank(text)
    r = _Read(text)
    if vectors:
        r.vectors |= set(vectors)
    statements = _statements(clean)
    r.program = _is_program(statements)
    r.per_year = _frequency(statements)
    sample: tuple[str | None, str | None] | None = None
    for body, offset in statements:
        lead = len(body) - len(body.lstrip())
        st, at = body.strip(), offset + lead
        head = re.match(r"[A-Za-z_@:!%{][\w.]*", st)
        word = head.group(0).lower() if head else st[:1]
        bad = _programming(st)
        if bad is not None:
            r.diagnostics.append(_Diag(at + bad[0], bad[1]))
            continue
        if word in _PROGRAM_WORDS:
            r.diagnostics.append(_Diag(at, f"{word!r} is part of EViews's programming language, which is not read: write the "
                                       "model's statements out", length=len(word)))
            continue
        m = re.match(r"model\s+([A-Za-z_]\w*)\s*$", st, re.I)
        if m:
            if r.model and r.model != m.group(1).lower():
                r.diagnostics.append(_Diag(at + m.start(1), f"a second model ({m.group(1)}): one model a text (the first is "
                                           f"{r.model})", length=len(m.group(1))))
                continue
            r.model = m.group(1).lower()
            continue
        m = re.match(r"([A-Za-z_]\w*)\.(\w+)(?:\(([^)]*)\))?\s*(.*)$", st)
        if m and word != "equation" and not re.match(r"[A-Za-z_]\w*\s*\(", st):
            obj, method, rest = m.group(1).lower(), m.group(2).lower(), m.group(4)
            rest_at = at + m.start(4)
            if method in ("append", "merge"):
                if not r.model:
                    r.diagnostics.append(_Diag(at, f"{obj}.{method} before 'model {obj}': declare the model first", length=len(obj)))
                    continue
                if obj != r.model:
                    r.diagnostics.append(_Diag(at, f"{obj} is not the model ({r.model})" + _suggest(obj, [r.model]), length=len(obj)))
                    continue
                if method == "merge":
                    name = rest.strip().lower()
                    if not re.fullmatch(r"[a-z_]\w*", name):
                        r.diagnostics.append(_Diag(rest_at, "merge takes an equation object's name: m.merge eq", length=max(len(rest), 1)))
                        continue
                    r.equations.append(_Eq("link", name, rest_at, at))
                    continue
                _model_line(r, rest, rest_at, at)
                continue
            if method in _IGNORED_METHODS:
                r.ignored.append(f"{obj}.{method}")
                r.diagnostics.append(_Diag(at, f"{obj}.{method} is not part of the model: ignored", "note", len(st)))
                continue
            r.diagnostics.append(_Diag(at + m.start(2), f"{obj}.{m.group(2)} is not read", length=len(m.group(2))))
            continue
        if word == "equation":
            _equation(r, st, at, sample)
            continue
        if word == "scalar":
            m = re.match(r"scalar\s+([A-Za-z_]\w*)\s*$", st, re.I)
            if m:  # declared: its value given later, or with params=
                r.scalars.add(m.group(1).lower())
                r.declared_at[m.group(1).lower()] = at + m.start(1)
                continue
            m = re.match(r"scalar\s+([A-Za-z_]\w*)\s*=\s*(.+)$", st, re.I)
            if not m:
                r.diagnostics.append(_Diag(at, "a scalar is: scalar name = value", length=len(st)))
                continue
            name = m.group(1).lower()
            v = _value(r, m.group(2), at + m.start(2))
            if v is not None:
                if name in r.vectors:
                    r.diagnostics.append(_Diag(at + m.start(1), f"{name} is a coefficient vector", length=len(name)))
                    continue
                r.scalars.add(name)
                _set(r, name, v, at + m.start(1))
            continue
        m = re.match(r"(coef|vector)\s*\(\s*(\d+)\s*\)\s*([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", st, re.I)
        if m:
            name, n = m.group(3).lower(), int(m.group(2))
            if name in r.scalars:
                r.diagnostics.append(_Diag(at + m.start(3), f"{name} is a scalar", length=len(name)))
                continue
            r.vectors.add(name)
            if m.group(4) is not None:
                v = _value(r, m.group(4), at + m.start(4))
                if v is not None:
                    for i in range(1, n + 1):
                        _set(r, f"{name}_{i}", v, at + m.start(3))
            continue
        if word == "param":
            rest = st[5:]
            pairs = list(re.finditer(r"([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", rest))
            covered = re.sub(r"\s+", "", "".join(p.group(0) for p in pairs)) == re.sub(r"\s+", "", rest)
            if not pairs or not covered:
                r.diagnostics.append(_Diag(at, "param is: param c(1) value c(2) value ...", length=len(st)))
                continue
            for p in pairs:
                vec = p.group(1).lower()
                if vec not in r.vectors:
                    r.diagnostics.append(_Diag(at + 5 + p.start(1), f"{p.group(1)} is not a coefficient vector (declare it: "
                                               f"coef(n) {p.group(1)})" + _suggest(vec, sorted(r.vectors)), length=len(vec)))
                    continue
                _set(r, f"{vec}_{int(p.group(2))}", float(p.group(3)), at + 5 + p.start(1))
            continue
        m = re.match(r"([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)\s*=\s*(.+)$", st)
        if m and (m.group(1).lower() in r.vectors or not r.program):
            vec = m.group(1).lower()
            if vec not in r.vectors:
                r.vectors.add(vec)  # model text with a coefficient block: beta(1) = ... declares beta
            v = _value(r, m.group(3), at + m.start(3))
            if v is not None:
                _set(r, f"{vec}_{int(m.group(2))}", v, at)
            continue
        if word == "smpl":
            sample = _sample(r, st, at)
            continue
        if r.program:
            m = re.match(r"(?:(series|genr)\s+)?([A-Za-z_]\w*)\s*=\s*(.+)$", st, re.I)
            if m:
                number = _number(m.group(3).strip())
                if number is not None:
                    name = m.group(2).lower()
                    r.constants[name] = number
                    r.constant_at[name] = at + m.start(2)
                    continue
                r.ignored.append("series generation")
                r.diagnostics.append(_Diag(at, "a series assignment is data, not part of the model: ignored", "note", len(st)))
                continue
            if word in _IGNORED_COMMANDS or word.split("(")[0] in _IGNORED_COMMANDS:
                r.ignored.append(word)
                r.diagnostics.append(_Diag(at, f"{word} is not part of the model: ignored", "note", len(st)))
                continue
            r.diagnostics.append(_Diag(at, f"{st.split()[0]!r} is not read: the program subset read is model, .append, .merge, "
                                       "equation .ls, scalar, coef, param and smpl", length=len(st.split()[0])))
            continue
        if word in ("series", "genr"):
            m = re.match(r"(?:series|genr)\s+([A-Za-z_]\w*)\s*=\s*(.+)$", st, re.I)
            number = _number(m.group(2).strip()) if m else None
            if m and number is not None:
                r.constants[m.group(1).lower()] = number
                r.constant_at[m.group(1).lower()] = at + m.start(1)
            else:
                r.diagnostics.append(_Diag(at, "a series assignment is data, not part of the model: ignored", "note", len(st)))
            continue
        _model_line(r, st, at, at)
    return r


def _frequency(statements: list[tuple[str, int]]) -> int | None:
    """The workfile's periods a year, from ``wfcreate`` (its frequency letter) or ``smpl`` with dated periods."""
    for st, _ in statements:
        m = re.match(r"\s*wfcreate(?:\([^)]*\))?\s+([aqmsAQMSyY])\b", st)
        if m:
            return {"y": 1}.get(m.group(1).lower(), _PER_YEAR.get(m.group(1).lower()))
        m = re.match(r"\s*smpl\s+(\S+)", st, re.I)
        if m:
            date = _date(m.group(1))
            if date is not None and date[1] in ("q", "m", "s"):
                return _PER_YEAR[date[1]]
    return None


def _suggest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


_PERIOD = re.compile(r"^(\d{4})(?:([qQ])([1-4])|([mM])(\d{1,2}))?$")


def _period(text: str) -> str | None:
    m = _PERIOD.match(text)
    if not m:
        return None
    if m.group(2):
        return f"{m.group(1)}Q{m.group(3)}"
    if m.group(4):
        return f"{m.group(1)}-{int(m.group(5)):02d}"
    return m.group(1)


def _sample(r: _Read, st: str, at: int) -> tuple[str | None, str | None] | None:
    words = list(re.finditer(r"\S+", st))[1:]
    if [w.group(0).lower() for w in words] == ["@all"]:
        return None
    if len(words) != 2:
        r.diagnostics.append(_Diag(at, "smpl is read as: smpl first last (1990q1, 1990m3, 1990, @first, @last) or smpl @all; "
                                   "conditions and several ranges are not", length=len(st)))
        return None
    out: list[str | None] = []
    for w in words:
        text = w.group(0).lower()
        if text in ("@first", "@last"):
            out.append(None)
            continue
        p = _period(text)
        if p is None:
            r.diagnostics.append(_Diag(at + w.start(), f"{w.group(0)!r} is not a period read here: 1990q1, 1990m3 or 1990",
                                       length=len(w.group(0))))
            return None
        out.append(p)
    return out[0], out[1]


def _equation(r: _Read, st: str, at: int, sample: tuple[str | None, str | None] | None) -> None:
    m = re.match(r"equation\s+([A-Za-z_]\w*)(?:\.(\w+))?(?:\(([^)]*)\))?\s*(.*)$", st, re.I)
    if not m:
        r.diagnostics.append(_Diag(at, "an equation object is: equation name.ls specification", length=len(st)))
        return
    name, method, spec = m.group(1).lower(), (m.group(2) or "").lower(), m.group(4)
    if method != "ls":
        where = at + (m.start(2) if m.group(2) else m.start(1))
        r.diagnostics.append(_Diag(where, f"equation {name}.{method or '?'}: only least squares (.ls) specifications are read",
                                   length=len(m.group(2) or name)))
        return
    if m.group(3):
        r.diagnostics.append(_Diag(at + m.start(3), f"the options ({m.group(3)}) are ignored: estimated by least squares", "note",
                                   len(m.group(3))))
    if not spec.strip():
        r.diagnostics.append(_Diag(at, f"equation {name}.ls needs a specification", length=len(st)))
        return
    if name in r.specs:
        r.diagnostics.append(_Diag(at + m.start(1), f"the equation {name} is specified twice", length=len(name)))
        return
    form = "expression" if _top_level(spec, "=") else "list"
    r.specs[name] = _Spec(name, form, spec, at + m.start(4), sample)


def _top_level(text: str, ch: str) -> list[int]:
    out, depth = [], 0
    for i, c in enumerate(text):
        depth += (c == "(") - (c == ")")
        if c == ch and depth == 0 and text[i - 1:i] not in ("<", ">") and text[i + 1:i + 2] not in (">",):
            out.append(i)
    return out


def _model_line(r: _Read, st: str, at: int, statement_at: int) -> None:
    """A line of model text: an equation, @identity, @add, @innov or :eq."""
    low = st.lower()
    if st.startswith(":"):
        name = st[1:].strip().lower()
        if not re.fullmatch(r"[a-z_]\w*", name):
            r.diagnostics.append(_Diag(at, "a linked equation is :name", length=len(st)))
            return
        r.equations.append(_Eq("link", name, at + len(st) - len(st[1:].lstrip()), statement_at))
        return
    if low.startswith("@identity"):
        rest = st[9:]
        r.equations.append(_Eq("identity", rest, at + 9, statement_at))
        return
    m = re.match(r"@add(\(\s*v\s*\))?\s+([A-Za-z_]\w*)(?:\s+([A-Za-z_]\w*))?\s*$", st, re.I)
    if m:
        variable = m.group(2).lower()
        r.adds.append((variable, (m.group(3) or f"{variable}_a").lower(), bool(m.group(1)), at + m.start(2), len(variable)))
        return
    if low.startswith("@add"):
        r.diagnostics.append(_Diag(at, "an add factor is: @add variable [factor] (or @add(v) ...)", length=len(st)))
        return
    m = re.match(r"@innov\s+([A-Za-z_]\w*)\s+(\S+)\s*$", st, re.I)
    if m:
        sd = _number(m.group(2))
        if sd is None or sd <= 0:
            r.diagnostics.append(_Diag(at + m.start(2), "an innovation's standard deviation is a positive number", length=len(m.group(2))))
            return
        r.innovs.append((m.group(1).lower(), sd, at + m.start(1), len(m.group(1))))
        return
    if low.startswith("@innov"):
        r.diagnostics.append(_Diag(at, "an innovation is: @innov variable sd", length=len(st)))
        return
    if st.startswith("@"):
        word = st.split()[0]
        r.diagnostics.append(_Diag(at, f"{word} is not read in a model's text: @identity, @add and @innov are", length=len(word)))
        return
    r.equations.append(_Eq("text", st, at, statement_at))


def _understand(r: _Read, eq: _Eq, text: str, at: int, identity: bool, label: str, spec: _Spec | None = None) -> _Built | None:
    """An equation's text as pymetrix's: the variable it determines (EViews's rule) on the left."""
    split = _top_level(text, "=")
    if len(split) != 1:
        r.diagnostics.append(_Diag(at, "an equation needs one '=': variable = expression" if not split else
                                   "more than one '=': comparisons are read only inside @recode", length=max(len(text.strip()), 1)))
        return None
    lhs_text, rhs_text = text[:split[0]], text[split[0] + 1:]
    try:
        lhs = _parse(lhs_text, at, r.vectors)
        rhs = _parse(rhs_text, at + split[0] + 1, r.vectors)
    except _Refused as exc:
        r.diagnostics.append(_Diag(exc.at, exc.message, length=exc.length))
        return None
    return _arrange(r, lhs, rhs, at, len(text.strip()), identity, label, spec)


def _arrange(r: _Read, lhs: Any, rhs: Any, at: int, length: int, identity: bool, label: str, spec: _Spec | None) -> _Built | None:
    render = _Render(r.scalars, r.per_year)
    current = sorted(_current(lhs, r.scalars), key=lambda n: n.at)
    if not current:
        r.diagnostics.append(_Diag(at, "the left-hand side has no series of the current period: the equation determines the first "
                                   "one there", length=length))
        return None
    dependent = current[0].name
    try:
        terms = _additive(lhs)
        keep = [(s, t) for s, t in terms if any(n.name == dependent for n in _current(t, r.scalars))]
        moved = [(s, t) for s, t in terms if (s, t) not in keep]
        alone = len(keep) == 1 and keep[0][0] == 1 and not any(n.name != dependent for n in _current(keep[0][1], r.scalars))
        if alone:
            lhs_out = render(keep[0][1])
            rhs_out = render(rhs) + "".join(f" {'+' if s < 0 else '-'} {render(t)}" for s, t in moved)
        else:
            lhs_out, rhs_out = dependent, f"{dependent} - (({render(lhs)}) - ({render(rhs)}))"
    except _Refused as exc:
        r.diagnostics.append(_Diag(exc.at, exc.message, length=exc.length))
        return None
    coefs: dict[str, tuple[int, int]] = {}
    for n in _walk(lhs) + _walk(rhs):
        if isinstance(n, _Coef):
            coefs.setdefault(f"{n.vector}_{n.index}", (n.at, n.length))
        elif isinstance(n, _Ser) and n.name in r.scalars:
            coefs.setdefault(n.name, (n.at, n.length))
    if identity and coefs:
        first = min(coefs.values())
        r.diagnostics.append(_Diag(first[0], "an identity has no coefficients: drop @identity, or give the number", length=first[1]))
        return None
    return _Built(label or dependent, dependent, lhs_out, rhs_out, list(coefs), identity, at, spec, alone and not moved, coefs,
                  current[0].at)


def _list_spec(r: _Read, spec: _Spec, fitted: bool) -> tuple[Any, list[Any], bool] | None:
    """A list specification's parts: the dependent expression, the regressors, whether ``c`` is among them."""
    parts, depth, start = [], 0, None
    for i, ch in enumerate(spec.text + " "):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if start is not None:
                parts.append((spec.text[start:i], spec.at + start))
                start = None
        elif start is None:
            start = i
    if depth > 0:
        r.diagnostics.append(_Diag(spec.at, "a bracket here is never closed", length=len(spec.text.strip())))
        return None
    if len(parts) < 2:
        r.diagnostics.append(_Diag(spec.at, "a list specification is: dependent regressor ... (c the intercept)", length=len(spec.text)))
        return None
    nodes = []
    for text, at in parts:
        low = text.lower()
        if re.match(r"(ar|ma|sar|sma|pdl)\(", low) or low.startswith("@expand"):
            r.diagnostics.append(_Diag(at, f"{text}: ARMA terms, PDLs and @expand are not read in a specification", length=len(text)))
            return None
        if low == "c":
            nodes.append("c")
            continue
        try:
            nodes.append(_parse(text, at, r.vectors - {"c"}))
        except _Refused as exc:
            r.diagnostics.append(_Diag(exc.at, exc.message, length=exc.length))
            return None
    if nodes[0] == "c":
        r.diagnostics.append(_Diag(parts[0][1], "the first in a list specification is the dependent variable, not c"))
        return None
    return nodes[0], nodes[1:], "c" in nodes[1:]


def _understand_all(r: _Read, data: Any = None) -> None:
    """Every equation of the model understood into ``r.built`` (problems placed)."""
    linked: set[str] = set()
    for eq in r.equations:
        if eq.kind == "link":
            spec = r.specs.get(eq.text)
            if spec is None:
                r.diagnostics.append(_Diag(eq.at, f"the equation {eq.text} is not specified in this text (equation {eq.text}.ls ...): "
                                           "a linked equation is read only with its specification" + _suggest(eq.text, list(r.specs)),
                                           length=len(eq.text)))
                continue
            if eq.text in linked:
                r.diagnostics.append(_Diag(eq.at, f"the equation {eq.text} is in the model twice", length=len(eq.text)))
                continue
            linked.add(eq.text)
            if spec.form == "expression":
                built = _understand(r, eq, spec.text, spec.at, False, spec.name, spec)
            else:
                parts = _list_spec(r, spec, data is not None)
                if parts is None:
                    continue
                dependent, regressors, _ = parts
                node, k = None, 0
                for x in regressors:  # c(1) + c(2)*x + ...: EViews's coefficients, in the order listed
                    k += 1
                    term = _Coef("c", k, spec.at, 1) if x == "c" else _Bin("*", _Coef("c", k, spec.at, 1), _wrap(x), spec.at)
                    node = term if node is None else _Bin("+", node, term, spec.at)
                built = _arrange(r, dependent, node, spec.at, len(spec.text), False, spec.name, spec)
            if built is not None:
                r.built.append(built)
            continue
        built = _understand(r, eq, eq.text, eq.at, eq.kind == "identity", "")
        if built is not None:
            r.built.append(built)
    seen: dict[str, _Built] = {}
    kept = []
    for b in r.built:
        if b.dependent in seen:
            from .source import _place

            first = _place(r.text, seen[b.dependent].at)[0]
            r.diagnostics.append(_Diag(b.dependent_at, f"{b.dependent} is already determined by the equation on line {first}",
                                       length=len(b.dependent)))
            continue
        if b.dependent in r.scalars:
            r.diagnostics.append(_Diag(b.at, f"{b.dependent} is a scalar: an equation determines a series", length=len(b.dependent)))
            continue
        seen[b.dependent] = b
        kept.append(b)
    labels = [b.label for b in kept]
    for b in kept:
        if b.spec is not None and b.label in seen and seen[b.label] is not b:
            r.diagnostics.append(_Diag(b.at, f"the equation {b.label} has the name of the variable another determines: rename it",
                                       length=len(b.label)))
    del labels
    r.built = kept
    by_variable = {b.dependent: b for b in r.built}
    shocks = {}
    for variable, factor, shift, at, length in r.adds:
        b = by_variable.get(variable)
        if b is None:
            r.diagnostics.append(_Diag(at, f"no equation determines {variable}" + _suggest(variable, list(by_variable)), length=length))
            continue
        if b.identity:
            r.diagnostics.append(_Diag(at, f"the equation for {variable} is an identity: it takes no add factor", length=length))
            continue
        if b.spec is not None and data is not None:
            r.diagnostics.append(_Diag(at, f"{variable}'s equation is fitted to the data: an add factor is read only on a model text's "
                                       "equations", length=length))
            continue
        if shift and not b.lhs_alone:
            r.diagnostics.append(_Diag(at, f"@add(v) shifts {variable} itself: read only where the left-hand side is {variable} alone "
                                       "(use @add, which shifts the equation)", length=length))
            continue
        b.rhs = f"{b.rhs} + {factor}"
    for variable, sd, at, length in r.innovs:
        b = by_variable.get(variable)
        if b is None:
            r.diagnostics.append(_Diag(at, f"no equation determines {variable}" + _suggest(variable, list(by_variable)), length=length))
            continue
        if b.identity:
            r.diagnostics.append(_Diag(at, f"the equation for {variable} is an identity: it has no innovation", length=length))
            continue
        if b.spec is not None and data is not None:
            r.diagnostics.append(_Diag(at, f"{variable}'s equation is fitted to the data: its errors are its residuals", length=length))
            continue
        name = f"{variable}_innov"
        if name in shocks:
            r.diagnostics.append(_Diag(at, f"the innovation of {variable} is given twice", length=length))
            continue
        shocks[name] = sd
        b.rhs = f"{b.rhs} + {name}"


def _wrap(node: Any) -> Any:
    return node if isinstance(node, (_Ser, _Coef, _Num, _Call, _Par)) else _Par(node, getattr(node, "at", 0))


def _shocks(r: _Read) -> dict[str, float]:
    determined = {b.dependent for b in r.built}
    out: dict[str, float] = {}
    for variable, sd, _, _ in r.innovs:
        if variable in determined:
            out.setdefault(f"{variable}_innov", sd)
    return out


def _missing(r: _Read, params: Mapping[str, float], data: Any) -> list[_Diag]:
    """Coefficients without values, at their first use (a specification's, fitted to data, need none)."""
    out = []
    told: set[str] = set()
    for b in r.built:
        if b.spec is not None and data is not None:
            continue
        for name, (at, length) in b.coef_places.items():
            if name not in r.values and name not in params and name not in told:
                told.add(name)
                if name in r.declared_at:  # scalar f: said once, where it is declared
                    out.append(_Diag(r.declared_at[name], f"the scalar {name} is declared without a value: give it (scalar {name} = "
                                     "...) or with params=", length=len(name)))
                    continue
                if b.spec is not None:
                    out.append(_Diag(at, f"{b.label} is estimated: give data (read_eviews(text, data=...)) to fit it, or values for "
                                     f"its coefficients ({_eviews_coef(name)} has none)", "warning", length))
                else:
                    out.append(_Diag(at, f"{_eviews_coef(name)} has no value: give it ({_eviews_coef(name)} = ..., param, scalar) "
                                     "or with params=", length=length))
                break
    return out


# ---------------------------------------------------------------------------------------------
# building


def _template(r: _Read, name: str | None, params: Mapping[str, float] | None, data: Any, span: Any) -> Any:
    from ..estimation import Equation
    from ..metadata import Role
    from .model import Model
    from .templates import Template

    values = {**r.values, **{_pymetrix_name(k): float(v) for k, v in (params or {}).items()}}
    shocks = _shocks(r)
    eqs: list[Any] = []
    for b in r.built:
        if b.spec is not None and data is not None:
            eqs.append(_fit(r, b, data, span))
            continue
        used = [q for q in b.params]
        text = f"{b.lhs} {'~' if used else '='} {b.rhs}"
        eqs.append(Equation(text, params=used, name=b.label) if used else Equation(text, name=b.label))
    used = {q for b in r.built if not (b.spec is not None and data is not None) for q in b.params}
    roles = {s: Role.SHOCK for s in shocks}
    model = Model(eqs, roles=roles or None, params={q: values[q] for q in values if q in used} or None)
    guess = {k: v for k, v in r.constants.items() if k in model.endogenous or k in model.exogenous}
    notes = sorted(set(r.ignored))
    description = "Read from EViews syntax" + (f"; ignored: {', '.join(notes)}" if notes else "")
    return Template(name or r.model or "eviews", model, tuple(shocks), shocks, None, guess, description)


def _per_year(frequency: Any) -> int | None:
    """Periods a year of a frequency ("Q", "M", "A", a pandas frequency or period index)."""
    if frequency is None:
        return None
    text = str(getattr(frequency, "freqstr", frequency)).upper()
    for key, n in (("Q", 4), ("M", 12), ("A", 1), ("Y", 1), ("S", 2)):
        if text.startswith(key):
            return n
    raise ModelError(f"frequency {frequency!r}: give 'Q', 'M' or 'A'")


def _pymetrix_name(key: str) -> str:
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)\s*", key)
    return f"{m.group(1).lower()}_{int(m.group(2))}" if m else key


def _fit(r: _Read, b: _Built, data: Any, span: Any) -> Any:
    """A specification fitted by least squares on its sample (``smpl``), or ``span``."""
    from ..estimation import Equation

    spec = b.spec
    assert spec is not None
    sample = span if span is not None else spec.span
    if sample is not None:
        index = data.frame.index
        periods = index.get_level_values(-1) if hasattr(index, "levels") else index
        sample = (sample[0] or str(periods[0]), sample[1] or str(periods[-1]))
    if spec.form == "list":
        dependent, regressors, intercept = _list_spec(r, spec, True)  # type: ignore[misc]
        render = _Render(r.scalars, r.per_year)
        terms = [render(x) for x in regressors if x != "c"]
        text = f"{render(dependent)} ~ {' + '.join(terms) if terms else '1'}{'' if intercept or not terms else ' - 1'}"
        if not intercept and not terms:
            raise ModelError(f"{spec.name}: a specification with nothing to estimate")
        eq = Equation(text, name=b.label)
    else:
        eq = Equation(f"{b.lhs} ~ {b.rhs}", params=b.params, name=b.label)
    return eq.fit(data, span=sample)


def _diagnostics(r: _Read) -> list[Any]:
    from .source import Diagnostic, _place

    out = []
    for d in r.diagnostics:
        line, column = _place(r.text, d.at)
        out.append(Diagnostic(line, column, d.message, d.severity, max(d.length, 1)))  # type: ignore[arg-type]
    order = {"error": 0, "warning": 1, "note": 2}
    seen, unique = set(), []
    for d in sorted(out, key=lambda d: (d.line, d.column, order[d.severity])):
        if (d.line, d.column, d.message) not in seen:
            seen.add((d.line, d.column, d.message))
            unique.append(d)
    return unique


def _prepare(text: str, params: Mapping[str, float] | None = None, data: Any = None, vectors: set[str] | None = None,
             frequency: Any = None) -> _Read:
    vectors = set(vectors or ()) | {m.group(1).lower() for k in (params or {})
                                    if (m := re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(\s*\d+\s*\)\s*", k))}
    r = _read(text, vectors)
    if frequency is not None:
        r.per_year = _per_year(frequency)
    elif data is not None and r.per_year is None:
        index = data.frame.index
        periods = index.get_level_values(-1) if hasattr(index, "levels") else index
        r.per_year = _per_year(getattr(periods, "freq", None)) if hasattr(periods, "freq") else None
    given = {_pymetrix_name(k) for k in (params or {})}
    r.scalars |= {k for k in given if not (re.fullmatch(r"(\w+)_\d+", k) and k.rpartition("_")[0] in r.vectors)}
    _understand_all(r, data)
    r.diagnostics += _missing(r, {_pymetrix_name(k): v for k, v in (params or {}).items()}, data)
    return r


def check_eviews(text: str) -> Any:
    """``ModelSource(text, "eviews").check()``: problems at their places, the symbols and their roles."""
    from ..errors import SpecificationError
    from ..expr.parser import parse_equation
    from .source import Diagnostic, SourceCheck, Symbol, _place

    r = _prepare(text)
    diagnostics = _diagnostics(r)
    determined = {b.dependent: b for b in r.built}
    used: dict[str, set[int]] = {}
    max_lead = max_lag = 0
    exogenous: dict[str, int] = {}
    constraints = []
    for b in r.built:
        line = _place(text, b.at)[0]
        try:
            lhs, _, rhs = parse_equation(f"{b.lhs} = {b.rhs}", b.params)
        except Exception as exc:  # noqa: BLE001 - a conversion pymetrix cannot parse: shown, not raised
            diagnostics.append(Diagnostic(line, 1, f"cannot read this equation: {exc}"))
            continue
        for n in lhs.refs() | rhs.refs():
            used.setdefault(n.id, set()).add(line)
            max_lead, max_lag = max(max_lead, n.lag), max(max_lag, -n.lag)
            if n.id not in determined:
                exogenous.setdefault(n.id, line)
        for q in b.params:
            used.setdefault(q, set()).add(line)
        if rhs.calls("max") or rhs.calls("min"):
            constraints.append(b.label)
    shocks = _shocks(r)
    found_ok = not any(d.severity == "error" for d in diagnostics)
    if found_ok and r.built and not any(d.severity == "warning" and "is estimated" in d.message for d in diagnostics):
        try:
            _template(r, None, None, None, None)
        except (ModelError, SpecificationError, ValueError) as exc:
            diagnostics.append(Diagnostic(1, 1, f"the model does not build: {exc}"))
    for name, line in exogenous.items():  # a series no equation determines, close to one that is: likely a typo
        if name in shocks or len(used.get(name, ())) > 1 or name.endswith(("_a", "_aerr")):
            continue
        close = difflib.get_close_matches(name, list(determined), n=1, cutoff=0.8)
        if close and not (name.startswith(close[0]) or name.endswith(close[0])):  # erx beside er: a name of its own
            at = next(n for b in r.built for n in _equation_names(r, b) if n.name == name)
            ln, col = _place(text, at.at)
            diagnostics.append(Diagnostic(ln, col, f"{name} is exogenous (no equation determines it); did you mean {close[0]!r}?",
                                          "warning", len(name)))
    symbols = [Symbol(b.dependent, "endogenous", r.constants.get(b.dependent), "", _place(text, b.at)[0],
                      tuple(sorted(used.get(b.dependent, ())))) for b in r.built]
    symbols += [Symbol(s, "shock", sd, "", 0, tuple(sorted(used.get(s, ())))) for s, sd in shocks.items()]
    symbols += [Symbol(n, "exogenous", r.constants.get(n), "", 0, tuple(sorted(used.get(n, ()))))
                for n in sorted(exogenous) if n not in shocks]
    names = sorted({q for b in r.built for q in b.params} | set(r.values))
    symbols += [Symbol(q, "parameter", r.values.get(q), _eviews_coef(q) if q not in r.scalars else "scalar",
                       _place(text, r.value_at[q])[0] if q in r.value_at else 0, tuple(sorted(used.get(q, ())))) for q in names]
    order = {"error": 0, "warning": 1, "note": 2}
    diagnostics.sort(key=lambda d: (d.line, d.column, order[d.severity]))
    if not r.built and not any(d.severity == "error" for d in diagnostics):
        diagnostics.append(Diagnostic(1, 1, "no equations yet", "note"))
    return SourceCheck("eviews", tuple(diagnostics), tuple(symbols), len(r.built), max_lead, max_lag, tuple(constraints))


def _equation_names(r: _Read, b: _Built) -> list[_Ser]:
    """The series an equation's text names, with their places."""
    out: list[_Ser] = []
    for e in r.equations:
        if e.kind != "link" and e.at == b.at:
            try:
                for half, at in _halves(e.text, e.at):
                    out += [n for n in _walk(_parse(half, at, r.vectors)) if isinstance(n, _Ser)]
            except _Refused:
                pass
    return out


def _halves(text: str, at: int) -> list[tuple[str, int]]:
    split = _top_level(text, "=")
    return [(text[:split[0]], at), (text[split[0] + 1:], at + split[0] + 1)] if len(split) == 1 else []


@api("pymetrix:read_eviews")
def read_eviews(source: str | Path, params: Mapping[str, float] | None = None, data: Any = None, span: Any = None,
                name: str | None = None, frequency: str | None = None) -> Any:
    """Read an EViews model's text or program (a path to a ``.prg`` or ``.txt`` file, or the text) into a ``Template``:
    see the module docstring. ``params`` gives coefficient and scalar values (``{"c_1": 0.5}`` or ``{"c(1)": 0.5}``),
    winning over the text's; ``data`` (a Dataset) fits the ``equation eq.ls`` specifications by least squares, on their
    ``smpl`` or ``span``. ``frequency`` ("Q", "M", "A") is the data's, for ``@pchy`` and the like, when neither the
    text (``wfcreate``, a dated ``smpl``) nor ``data`` says it. Series that ``@trend``, ``@elem`` and ``@seas`` stand for
    are made from a dataset's dates by ``eviews_series``. Refused, with the first problems and their places, when the
    text has errors."""
    from .source import _place

    path = Path(source) if not isinstance(source, str) or ("\n" not in source and source.strip().lower().endswith((".prg", ".txt"))) else None
    text = path.read_text(encoding="utf-8") if path is not None else str(source)
    r = _prepare(text, params, data, frequency=frequency)
    problems = [d for d in r.diagnostics if d.severity == "error" or "is estimated" in d.message]
    if problems:
        first = sorted(problems, key=lambda d: d.at)[:3]
        raise ModelError("the EViews text has errors: " + "; ".join(
            f"line {_place(text, d.at)[0]}, column {_place(text, d.at)[1]}: {d.message}" for d in first))
    if not r.built:
        raise ModelError("the EViews text has no equations")
    return _template(r, name or (path.stem if path else None), params, data, span)


@api("pymetrix:eviews_series")
def eviews_series(model: Any, data: Any) -> Any:
    """``data`` with the series the EViews reader made of ``@trend``, ``@elem`` and ``@seas`` (the exogenous variables
    of ``model``, a Model or a template, named ``trend_<base>``, ``trend_start``, ``<x>_at_<date>`` and ``seas_<n>``),
    worked out from the data's dates: ``@trend(base)`` the periods since the base (``60:1`` in the data's own
    frequency), ``@trend`` alone from the data's first period, ``@elem(x, date)`` x's value at that date in every
    period, ``@seas(n)`` 1 in the n-th period of the year. Series the data already has are kept as they are."""
    import numpy as np
    import pandas as pd

    from ..data import Dataset

    model = getattr(model, "model", model)
    frame = data.frame.copy()
    periods = frame.index
    if not isinstance(periods, pd.PeriodIndex):
        raise ModelError("eviews_series needs a time series with a period index")
    freq = periods.freq
    for name in model.exogenous:
        if name in frame:
            continue
        m = re.fullmatch(r"trend_(?:(start)|(\d{4})(?:([qms_])(\d+))?)", name)
        if m:
            if m.group(1):
                frame[name] = np.arange(len(periods), dtype=float)
                continue
            year, kind, period = int(m.group(2)), m.group(3) or "a", int(m.group(4) or 1)
            if kind == "_":
                base = pd.Period(year=year, freq=freq, **({"quarter": period} if freq.freqstr.startswith("Q") else
                                                          {"month": period} if freq.freqstr.startswith("M") else {}))
            elif kind == "a":
                base = pd.Period(str(year), "Y").asfreq(freq, "start")
            elif kind == "s":
                raise ModelError(f"{name}: half-years are not a pandas frequency: give the trend's series in the data")
            else:
                base = pd.Period(f"{year}Q{period}" if kind == "q" else f"{year}-{period:02d}").asfreq(freq, "start")
            frame[name] = np.asarray([p.ordinal - base.ordinal for p in periods], dtype=float)
            continue
        m = re.fullmatch(r"seas_(\d+)", name)
        if m:
            frame[name] = np.asarray([float((p.quarter if freq.freqstr.startswith("Q") else p.month) == int(m.group(1)))
                                      for p in periods])
            continue
        m = re.fullmatch(r"(.+)_at_(\d{4})(?:([qms_])(\d+))?", name)
        if m and m.group(1) in frame:
            year, kind, period = int(m.group(2)), m.group(3) or "a", int(m.group(4) or 1)
            if kind == "a":
                when = pd.Period(str(year), "Y").asfreq(freq, "end")
            elif kind in ("q", "m"):
                when = pd.Period(f"{year}Q{period}" if kind == "q" else f"{year}-{period:02d}").asfreq(freq, "end")
            else:
                raise ModelError(f"{name}: give the date with its frequency (2004q4)")
            frame[name] = float(frame[m.group(1)].loc[when])
    return Dataset.from_frame(frame)


# ---------------------------------------------------------------------------------------------
# writing


def eviews_expression(expr: Any, params: set[str] | frozenset[str] = frozenset(), rename: Mapping[str, str] | None = None) -> str:
    """A pymetrix expression in EViews syntax: lags ``x(-1)``, ``d``, ``dlog``, ``100*@pch`` for ``pch``, ``@recode``
    for ``max`` and ``min``, ``c_<i>`` as ``c(i)``, ``iif`` as ``@iif``, ``fillna`` as ``@nan``, ``round(x, n)`` as
    ``@round(x*10^n)/10^n``, the calendar as ``@year`` ..., ``obsnum()`` as ``@obsnum`` (in a panel EViews's
    ``@obsid`` is pymetrix's), ``sum(x, by=g)`` as ``@sumsby(x, g)`` (and mean, min, max), ``normrnd`` and ``unifrnd`` as
    ``nrnd``, ``rnd`` and ``@runif`` (their seed is dropped: EViews draws its own numbers, seeded by ``rndseed``).
    Refused (``ModelError``) for what EViews has no equivalent of."""
    from ..expr.nodes import BinOp, Call, Coef, Level, Name, Neg, Num
    from ..expr.parser import parse_expression

    rename = rename or {}
    functions = {v: k for k, v in _SIMPLE.items() if not k.startswith("@") or k in ("@movav", "@movsum", "@movmax", "@movmin", "@cnorm",
                                                                                      "@dnorm", "@qnorm", "@ctdist", "@dtdist", "@qtdist",
                                                                                      "@cchisq", "@dchisq", "@qchisq", "@cfdist", "@dfdist",
                                                                                      "@qfdist")}
    functions.update({"log": "log", "exp": "exp", "abs": "abs", "sqrt": "@sqrt", "log10": "@log10", "floor": "@floor", "ceil": "@ceiling",
                      "round": "@round", "isna": "@isna", "fillna": "@nan", "iif": "@iif", "cumsum": "@cumsum"})
    rows = {"na": "na", "year": "@year", "quarter": "@quarter", "month": "@month", "day": "@day", "obsnum": "@obsnum"}
    by = {"sum": "@sumsby", "mean": "@meansby", "min": "@minsby", "max": "@maxsby"}

    def param(name: str) -> str:
        m = re.fullmatch(r"c_(\d+)", name)
        return f"c({m.group(1)})" if m else rename.get(name, name)

    def prec(e: Any) -> int:
        if isinstance(e, BinOp) and not e.grouped and not e.source:
            return {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}.get(e.op, 0)
        return 0 if isinstance(e, Neg) else 4

    def go(e: Any) -> str:
        if isinstance(e, Num):
            v = float(e.value)
            return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
        if isinstance(e, Name):
            name = rename.get(e.id, e.id)
            if e.id in params:
                return param(e.id)
            return f"{name}({e.lag:+d})" if e.lag else name
        if isinstance(e, Coef):
            return param(e.id)
        if isinstance(e, Neg):
            inner = go(e.x) if hasattr(e, "x") else go(e.children()[0])
            child = e.children()[0]
            return f"-{inner}" if prec(child) >= 2 and not isinstance(child, Neg) else f"-({inner})"
        if isinstance(e, BinOp):
            if e.source:
                m = re.fullmatch(r"(d|dlog|pch)\((.*)\)(?:\[([+-]\d+)\])?", e.source)
                if m:
                    inner = parse_expression(m.group(2), list(params)).shift(int(m.group(3) or 0))
                    text = f"{'@pch' if m.group(1) == 'pch' else m.group(1)}({go(inner)})"
                    return f"(100*{text})" if m.group(1) == "pch" else text
            if e.op in ("<", ">", "<=", ">=", "==", "!="):  # 1 where it holds: EViews's comparisons, in brackets
                op = {"==": "=", "!=": "<>"}.get(e.op, e.op)
                return f"({go(e.left)} {op} {go(e.right)})"
            if e.op not in ("+", "-", "*", "/", "^"):
                raise ModelError(f"the operator {e.op!r} has no equivalent in an EViews model's equations")
            p = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}[e.op]
            left, right = go(e.left), go(e.right)
            if prec(e.left) < p or (e.op == "^" and prec(e.left) <= 3) or isinstance(e.left, Neg):
                left = f"({left})" if not (prec(e.left) == 4) else left
            if prec(e.right) < p or (prec(e.right) == p and e.op in "-/^") or isinstance(e.right, Neg):
                right = f"({right})" if not (prec(e.right) == 4) else right
            space = " " if e.op in "+-" else ""
            text = f"{left}{space}{e.op}{space}{right}"
            return f"({text})" if e.grouped else text
        if isinstance(e, Call):
            if e.fn in by and len(e.args) == 2 and isinstance(e.args[1], Level):  # sum(x, by=g): @sumsby(x, g)
                return f"{by[e.fn]}({go(e.args[0])}, {rename.get(e.args[1].id, e.args[1].id)})"
            if e.fn in ("max", "min") and len(e.args) == 2:
                a, b = (go(x) for x in e.args)
                return f"@recode(({a}) {'>' if e.fn == 'max' else '<'} ({b}), {a}, {b})"
            if e.fn in rows and not e.args:
                return rows[e.fn]
            if e.fn == "normrnd":  # the seed is pymetrix's: EViews draws its own numbers (rndseed)
                mu, sigma = (go(x) for x in e.args[:2])
                return "nrnd" if (mu, sigma) == ("0", "1") else f"({mu} + {sigma}*nrnd)"
            if e.fn == "unifrnd":
                a, b = (go(x) for x in e.args[:2])
                return "rnd" if (a, b) == ("0", "1") else f"@runif({a}, {b})"
            if e.fn == "round" and len(e.args) == 2:  # n decimals: EViews's @round has none
                n = go(e.args[1])
                return f"(@round({go(e.args[0])}*10^({n}))/10^({n}))" if n != "0" else f"@round({go(e.args[0])})"
            if e.fn in functions and (e.fn not in ("normcdf", "normpdf", "norminv") or len(e.args) == 1):
                return f"{functions[e.fn]}({', '.join(go(x) for x in e.args)})"
            raise ModelError(f"{e.fn}() has no equivalent read in EViews syntax here: write it out in other terms to export")
        raise ModelError(f"{type(e).__name__} has no equivalent in EViews syntax")

    return go(expr)


def eviews_names(names: list[str]) -> dict[str, str]:
    """pymetrix names as EViews names: lower case (EViews ignores case); reserved names (``c``, ``d``, ``log`` ...)
    with ``_v`` after them. Refused when two would be one."""
    out: dict[str, str] = {}
    for n in names:
        new = n.lower()
        if new in RESERVED:
            new = f"{new}_v"
        out[n] = new
    seen: dict[str, str] = {}
    for n, new in out.items():
        if new in seen and seen[new] != n:
            raise ModelError(f"{seen[new]} and {n} are one name in EViews, which ignores case: rename one")
        seen[new] = n
    return {k: v for k, v in out.items() if k != v}


# ---------------------------------------------------------------------------------------------
# FRB/US-style XML model files


@dataclass(frozen=True, eq=False)
class EViewsXmlModel:
    """An XML model file read: the template (equations, coefficients, definitions as labels) and what else the file says
    of each variable: its kind (Behavioral, Identity, Exogenous), sector, description and equation text."""

    template: Any
    kinds: Mapping[str, str]
    sectors: Mapping[str, str]
    descriptions: Mapping[str, str]
    equations: Mapping[str, str]  # the EViews text used, by variable
    mce: Mapping[str, str] = field(default_factory=dict)  # variables with model-consistent-expectations versions: their group
    stochastic: Mapping[str, str] = field(default_factory=dict)  # each equation's stochastic type (FRB/US: "IS", "FN", ..., "NO")

    @property
    def model(self) -> Any:
        return self.template.model

    def __repr__(self) -> str:
        return (f"<EViewsXmlModel {self.template.name}: {len(self.equations)} equations, "
                f"{len(self.template.model.exogenous)} exogenous>")


def _xml_equations(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    raw = Path(path).read_text(encoding="utf-8")
    if re.search(r"<!(DOCTYPE|ENTITY)", raw):
        raise ModelError("the XML file declares a DOCTYPE or entities: not read")
    root = ET.fromstring(raw)
    out = []
    for v in root.findall("variable"):
        item: dict[str, Any] = {"name": (v.findtext("name") or "").strip().lower(), "kind": (v.findtext("equation_type") or "").strip(),
                                "sector": (v.findtext("sector") or "").strip(), "definition": (v.findtext("definition") or "").strip(),
                                "description": (v.findtext("description") or "").strip(),
                                "stochastic": (v.findtext("stochastic_type") or "").strip()}
        for tag in ("standard_equation", "mce_equation"):
            node = v.find(tag)
            if node is not None:
                coeffs = {}
                for c in node.findall("coeff"):
                    coeffs[_pymetrix_name((c.findtext("cf_name") or "").strip())] = float((c.findtext("cf_value") or "nan").strip())
                item[tag] = {"eviews": node.findtext("eviews_equation") or "", "python": node.findtext("python_equation") or "",
                             "coeffs": coeffs, "group": (node.findtext("mce_group") or "").strip()}
        out.append(item)
    return (root.findtext("name") or "model").strip(), out


def xml_equation(text: str, coeffs: Mapping[str, float], vectors: set[str], kind: str = "Behavioral") -> tuple[str, str, list[str]]:
    """One XML file's EViews equation as pymetrix's: (left, right, parameters used). Refused (``ModelError``, with the
    column) when it does not read."""
    flat = " ".join(text.split())
    r = _Read(flat, vectors={"c"} | vectors)
    r.values.update(coeffs)
    built = _understand(r, _Eq("text", flat, 0, 0), flat, 0, False, "")
    errors = [d for d in r.diagnostics if d.severity == "error"]
    if errors or built is None:
        d = errors[0]
        raise ModelError(f"column {d.at + 1}: {d.message} (in {flat!r})")
    del kind
    return built.lhs, built.rhs, built.params


@api("pymetrix:read_eviews_xml")
def read_eviews_xml(path: str | Path, mce: bool | list[str] = False, name: str | None = None) -> EViewsXmlModel:
    """Read an XML model file in the format of the Federal Reserve's FRB/US package (``mods/model.xml``) into a
    ``Template`` (in an ``EViewsXmlModel`` with each variable's kind, sector, description and stochastic type). ``mce`` uses the
    model-consistent-expectations versions of the equations that have them (``True``: all; or a list of their
    groups, e.g. ``["mcap", "mcwp"]``) instead of the standard (VAR-expectations) ones. The equations go through the
    EViews reader (``read_eviews``), their coefficients ``y_x(1)`` becoming parameters ``y_x_1``."""
    from ..estimation import Equation
    from .model import Model
    from .templates import Template

    title, items = _xml_equations(path)
    vectors = {k.rpartition("_")[0] for item in items for tag in ("standard_equation", "mce_equation") if tag in item
               for k in item[tag]["coeffs"]}
    eqs, texts, values, groups = [], {}, {}, {}
    for item in items:
        chosen = item.get("standard_equation")
        alt = item.get("mce_equation")
        if alt is not None:
            groups[item["name"]] = alt["group"]
            if mce is True or (isinstance(mce, list) and alt["group"] in mce):
                chosen = alt
        if chosen is None:
            continue
        try:
            lhs, rhs, used = xml_equation(chosen["eviews"], chosen["coeffs"], vectors, item["kind"])
        except ModelError as exc:
            raise ModelError(f"{item['name']}: {exc}") from None
        values.update({q: chosen["coeffs"][q] for q in used if q in chosen["coeffs"]})
        missing = [q for q in used if q not in values]
        if missing:
            raise ModelError(f"{item['name']}: no value for {', '.join(_eviews_coef(q) for q in missing)}")
        text = f"{lhs} {'~' if used else '='} {rhs}"
        eqs.append(Equation(text, params=used, name=item["name"]) if used else Equation(text, name=item["name"]))
        texts[item["name"]] = " ".join(chosen["eviews"].split())
    model = Model(eqs, params=values or None)
    labels = {i["name"]: i["definition"] for i in items if i["definition"]}
    template = Template(name or title, model, (), {}, None, {}, f"Read from an XML model file ({Path(path).name})", labels)
    return EViewsXmlModel(template, {i["name"]: i["kind"] for i in items}, {i["name"]: i["sector"] for i in items if i["sector"]},
                          {i["name"]: i["description"] for i in items if i["description"]}, texts, groups,
                          {i["name"]: i["stochastic"] for i in items if i["stochastic"] and i["name"] in texts})


# ---------------------------------------------------------------------------------------------
# state-space objects (sspace)


_ERROR_SPEC = re.compile(r"\[\s*var\s*=\s*(.*)\]\s*$", re.I)


def _sspace_read(text: str) -> tuple[dict[str, Any], list[_Diag]]:
    """An ``sspace`` object's text (``@signal``, ``@state``, ``@ename``, ``@evar``; bare, or after ``ss.append``):
    its pieces, and the problems with their places."""
    clean = _blank(text)
    diags: list[_Diag] = []
    out: dict[str, Any] = {"name": "", "signals": [], "states": [], "errors": {}, "variances": {}, "values": {}, "vectors": {"c"},
                           "ignored": []}
    for body, offset in _statements(clean):
        lead = len(body) - len(body.lstrip())
        st, at = body.strip(), offset + lead
        bad = _programming(st)
        if bad is not None:
            diags.append(_Diag(at + bad[0], bad[1]))
            continue
        m = re.match(r"sspace\s+([A-Za-z_]\w*)\s*$", st, re.I)
        if m:
            out["name"] = m.group(1).lower()
            continue
        m = re.match(r"([A-Za-z_]\w*)\.append\s+(.*)$", st, re.I)
        if m:
            if out["name"] and m.group(1).lower() != out["name"]:
                diags.append(_Diag(at, f"{m.group(1)} is not the state-space object ({out['name']})", length=len(m.group(1))))
                continue
            st, at = m.group(2), at + m.start(2)
        elif re.match(r"[A-Za-z_]\w*\.\w+", st):
            out["ignored"].append(st.split("(")[0].split()[0])
            diags.append(_Diag(at, f"{st.split('(')[0].split()[0]} is not part of the model: ignored", "note", len(st)))
            continue
        m = re.match(r"(coef|vector)\s*\(\s*(\d+)\s*\)\s*([A-Za-z_]\w*)\s*$", st, re.I)
        if m:
            out["vectors"].add(m.group(3).lower())
            continue
        m = re.match(r"([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$", st)
        if m:
            out["vectors"].add(m.group(1).lower())
            out["values"][f"{m.group(1).lower()}_{int(m.group(2))}"] = float(m.group(3))
            continue
        m = re.match(r"(@signal|@state)\s+(.*)$", st, re.I)
        if m:
            body_at = at + m.start(2)
            spec = _ERROR_SPEC.search(m.group(2))
            equation = re.sub(r"\+\s*$", " ", m.group(2)[:spec.start()]) if spec else m.group(2)
            kind = "signals" if m.group(1).lower() == "@signal" else "states"
            out[kind].append((equation, body_at, (spec.group(1), body_at + spec.start(1)) if spec else None))
            continue
        m = re.match(r"@ename\s+([A-Za-z_]\w*)\s*$", st, re.I)
        if m:
            out["errors"][m.group(1).lower()] = at + m.start(1)
            continue
        m = re.match(r"@evar\s+var\s*\(\s*([A-Za-z_]\w*)\s*\)\s*=\s*(.+)$", st, re.I)
        if m:
            out["variances"][m.group(1).lower()] = (m.group(2), at + m.start(2), at + m.start(1))
            continue
        m = re.match(r"@evar\s+cov", st, re.I)
        if m:
            diags.append(_Diag(at, "covariances of errors (@evar cov) are not read: pymetrix's StateSpace has independent shocks",
                               length=len(st)))
            continue
        m = re.match(r"(@vprior|@mprior)\b", st, re.I)
        if m:
            out["ignored"].append(m.group(1).lower())
            diags.append(_Diag(at, f"{m.group(1)} (the initial states' prior) is ignored: the StateSpace starts its states as it "
                               "does (stationary where it can, else diffuse)", "note", len(st)))
            continue
        word = st.split()[0]
        if re.split(r"[(\s]", word)[0].lower() in _IGNORED_COMMANDS:
            out["ignored"].append(word)
            diags.append(_Diag(at, f"{word} is not part of the model: ignored", "note", len(st)))
            continue
        diags.append(_Diag(at, f"{word!r} is not read in a state-space object: @signal, @state, @ename and @evar are", length=len(word)))
    return out, diags


def _first_at(node: Any, default: int) -> int:
    return min((n.at for n in _walk(node) if hasattr(n, "at")), default=default)


def _distribute(term: Any, sign: int) -> list[tuple[int, Any]]:
    """A term's additive pieces: ``k*(a - b)``, with ``k`` free of series, as ``k*a`` and ``-k*b`` (brackets opened)."""
    node = term
    if isinstance(node, _Neg):
        return _distribute(node.x, -sign)
    if isinstance(node, _Par):
        inner = _additive(node.x)
        return [p for s, t in inner for p in _distribute(t, sign * s)] if len(inner) > 1 else _distribute(node.x, sign)
    if isinstance(node, _Bin) and node.op == "*" and not any(isinstance(n, _Ser) for n in _walk(node.left)):
        inner = _additive(_strip(node.right))
        if len(inner) > 1:
            return [(s2, _Bin("*", node.left, t2, node.at)) for s, t in inner for s2, t2 in _distribute(t, sign * s)]
    return [(sign, term)]


def _sd_of(text: str, at: int, vectors: set[str], diags: list[_Diag]) -> str | float | None:
    """A variance as pymetrix's shock size: ``beta(3)^2`` -> the parameter ``beta_3``; a number -> its root."""
    try:
        node = _strip(_parse(text, at, vectors))
    except _Refused as exc:
        diags.append(_Diag(exc.at, exc.message, length=exc.length))
        return None
    if isinstance(node, _Num):
        return float(node.text) ** 0.5
    if isinstance(node, _Bin) and node.op == "^" and isinstance(_strip(node.right), _Num) and _strip(node.right).text == "2":
        base = _strip(node.left)
        if isinstance(base, _Coef):
            return f"{base.vector}_{base.index}"
        if isinstance(base, _Num):
            return float(base.text)
    diags.append(_Diag(at, "a variance is read as a coefficient squared, beta(3)^2, or a number: pymetrix's StateSpace takes a "
                       "shock's standard deviation as a parameter or a number", length=len(text.strip())))
    return None


def _sspace(text: str) -> tuple[dict[str, Any], list[_Diag]]:
    """The pieces in pymetrix's terms: states, transition and measurement equations, shocks and their sizes, parameters."""
    out, diags = _sspace_read(text)
    states = []
    for equation, at, _ in out["states"]:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", equation)
        if m:
            states.append(m.group(1).lower())
    vectors = set(out["vectors"])
    # beta(11) in a state-space object: a coefficient (states take lags, never positive indices)
    for equation, at, _ in out["signals"] + out["states"]:
        for m in re.finditer(r"([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)", equation):
            if m.group(1).lower() not in states:
                vectors.add(m.group(1).lower())
    shocks: dict[str, str | float] = {}
    params: set[str] = set()
    errors = set(out["errors"])
    for name, (variance, at, name_at) in out["variances"].items():
        if name not in errors:
            diags.append(_Diag(name_at, f"{name} is not an error named with @ename" + _suggest(name, sorted(errors)), length=len(name)))
            continue
        sd = _sd_of(variance, at, vectors, diags)
        if sd is not None:
            shocks[name] = sd
    for name, at in out["errors"].items():
        if name not in out["variances"]:
            diags.append(_Diag(at, f"{name} has no variance: @evar var({name}) = ...", length=len(name)))
    render = _Render(set())
    transition, measurement = [], []
    for kind in ("states", "signals"):
        for equation, at, spec in out[kind]:
            try:
                node = _parse(equation, at, vectors)
            except _Refused:
                split = _top_level(equation, "=")
                if len(split) != 1:
                    diags.append(_Diag(at, "an equation needs one '='", length=max(len(equation.strip()), 1)))
                    continue
                node = None
            split = _top_level(equation, "=")
            try:
                lhs = _parse(equation[:split[0]], at, vectors)
                rhs = _parse(equation[split[0] + 1:], at + split[0] + 1, vectors) if equation[split[0] + 1:].strip() else None
            except _Refused as exc:
                diags.append(_Diag(exc.at, exc.message, length=exc.length))
                continue
            del node
            lhs = _strip(lhs)
            if not isinstance(lhs, _Ser) or lhs.lag:
                diags.append(_Diag(at, "the left of a state or signal equation is the state or the observed series, alone",
                                   length=len(equation[:split[0]].strip())))
                continue
            name = lhs.name
            terms = _additive(rhs) if rhs is not None else []
            if spec is not None:  # [var = ...]: an error of its own
                error = f"e_{name}"
                sd = _sd_of(spec[0], spec[1], vectors, diags)
                if sd is None:
                    continue
                shocks[error] = sd
                terms.append((1, _Ser(error, 0, spec[1], 1)))
            data_terms, model_terms = [], []
            for sign, term in [piece for sign0, term0 in terms for piece in _distribute(term0, sign0)]:
                names = {n.name for n in _walk(term) if isinstance(n, _Ser)}
                if names & (set(states) | set(shocks) | errors):
                    if names - set(states) - set(shocks) - errors:
                        diags.append(_Diag(_first_at(term, at), "a term mixing data and states is not read "
                                           "(pymetrix's StateSpace has constant coefficients)", length=1))
                        break
                    model_terms.append((sign, term))
                elif names and kind == "states":
                    diags.append(_Diag(_first_at(term, at), f"{sorted(names)[0]}: a state equation takes states, errors and "
                                       "coefficients, not data", length=len(sorted(names)[0])))
                    break
                elif names and any(isinstance(n, _Coef) for n in _walk(term)):
                    diags.append(_Diag(_first_at(term, at), "a data series with an estimated coefficient (a regression in the "
                                       "signal) is not read: pymetrix's StateSpace observes data less known terms", length=1))
                    break
                elif names:
                    data_terms.append((sign, term))
                else:
                    model_terms.append((sign, term))
            else:
                text_rhs = " ".join(f"{'+' if s > 0 else '-'} {render(t)}" for s, t in model_terms).lstrip("+ ") or "0"
                if text_rhs.startswith("- "):
                    text_rhs = "-" + text_rhs[2:]
                if kind == "states":
                    transition.append(f"{name} = {text_rhs}")
                else:
                    moved = "".join(f" {'-' if s > 0 else '+'} {render(_wrap(t))}" for s, t in data_terms)
                    measurement.append(f"{name}{moved} = {text_rhs}")
                for n in _walk(rhs) if rhs is not None else []:
                    if isinstance(n, _Coef):
                        params.add(f"{n.vector}_{n.index}")
    params |= {sd for sd in shocks.values() if isinstance(sd, str)}
    return {"name": out["name"], "states": states, "transition": transition, "measurement": measurement, "shocks": shocks,
            "params": sorted(params), "values": out["values"], "ignored": out["ignored"]}, diags


@api("pymetrix:read_eviews_sspace")
def read_eviews_sspace(source: str | Path, params: Mapping[str, float] | None = None) -> Any:
    """An EViews state-space object (``sspace``: ``@signal``, ``@state``, ``@ename``, ``@evar var(e) = ...``, errors
    as ``[var = ...]``; bare or after ``ss.append``) as pymetrix's ``StateSpace``: signals are measurement equations
    (terms in data series moved to the left: the observed side), states transition equations, an error's variance
    ``beta(3)^2`` the shock's standard deviation, the parameter ``beta_3`` (or a number); coefficients ``beta(3)``
    are parameters ``beta_3``, with starting values from ``beta(3) = 0.5`` lines or ``params``. Refused (with the
    first problems at their places) for what does not map: covariances of errors, variances not of that form, terms
    mixing data and states, data in state equations, and EViews's programming language."""
    from ..metadata import Param
    from ..statespace import StateSpace
    from .source import _place

    path = Path(source) if not isinstance(source, str) or ("\n" not in source and source.strip().lower().endswith(".prg")) else None
    text = path.read_text(encoding="utf-8") if path is not None else str(source)
    pieces, diags = _sspace(text)
    errors = [d for d in diags if d.severity == "error"]
    if errors:
        raise ModelError("the state-space text has errors: " + "; ".join(
            f"line {_place(text, d.at)[0]}, column {_place(text, d.at)[1]}: {d.message}" for d in sorted(errors, key=lambda d: d.at)[:3]))
    values = {**pieces["values"], **{_pymetrix_name(k): float(v) for k, v in (params or {}).items()}}
    specs = [Param(q, start=values[q]) if q in values else Param(q) for q in pieces["params"]]
    return StateSpace(pieces["states"], pieces["transition"], pieces["measurement"], pieces["shocks"], specs)
