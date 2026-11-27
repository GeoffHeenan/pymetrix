"""Parser for expressions and equations.

Grammar (lowest to highest precedence)::

    equation := expr ("~" | "=") expr
    expr     := and_ ("or" and_)*                       # conditions: 1 where true, 0 where false
    and_     := not_ ("and" not_)*
    not_     := "not" not_ | compare
    compare  := sum (("<" | "<=" | ">" | ">=" | "==" | "!=") sum)?
    sum      := term (("+" | "-") term)*
    term     := unary (("*" | "/") unary)*
    unary    := "-" unary | power
    power    := postfix ("^" unary)?          # "**" is accepted for "^"
    postfix  := atom ("[" signed-integer "]")*  # x[-1] lags, x[+1] leads
    atom     := number | name | name "(" args ")" | "(" expr ")"
    args     := arg ("," arg)*        arg := expr | "by" "=" name | "seed" "=" integer   # by= groups; seed= seeds draws

Names listed in ``params`` become parameters (``Coef``); every other name is
a variable. ``d``, ``dlog`` and ``pch`` are expanded into primitive
operations so that all lags are explicit in the tree; moving-window
functions (``movavg(x, 4)``) keep every lag of their window as an argument.
Distribution functions (``normcdf(x)``, ``tinv(p, df)``) take their
parameters as further arguments (see ``pymetrix.expr.functions``). ``na``
without brackets is the missing value; random draws take ``seed=``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .functions import DISTRIBUTIONS, SPECIAL, arity, signature
from .nodes import (CALENDAR, CENTRED, CHOICES, COMPARISONS, ELEMENTWISE, EXPANDING, GROUPWISE, LOGICAL, MARKERS, MOVING, NA,
                    PRIMITIVES, RANDOM, BinOp, Call, Coef, Expr, ExpressionError, Level, Name, Neg, Num, window_lead)

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
  | (?P<name>[^\W\d]\w*)            # letters, digits and _, not starting with a digit: α and β too
  | (?P<op>==|!=|<=|>=|\*\*|[-+*/^()\[\],=~<>])
    """,
    re.VERBOSE,
)

_FUNCTIONS = sorted(PRIMITIVES | MARKERS | GROUPWISE | {"d", "dlog", "pch"} | set(DISTRIBUTIONS) | SPECIAL | set(MOVING) | set(EXPANDING)
                    | set(CHOICES) | CALENDAR | set(RANDOM))

# other packages' names for functions pymetrix spells differently: the hint after "unknown function"
_OTHER_NAMES = {"diff": "d(x) (the first difference; d(x, 2) the second; x - x[-4] over four periods)", "ln": "log(x)",
                "lag": "x[-1] (the lag of x)", "lead": "x[1] (the lead of x)", "pct_change": "pch(x)", "pchange": "pch(x)",
                "growth": "pch(x) or dlog(x)", "std": "sd(x)", "stdev": "sd(x)", "avg": "mean(x)", "average": "mean(x)",
                "mavg": "movavg(x, n)", "movav": "movavg(x, n)", "rolling_mean": "movavg(x, n)", "cumprod": "exp(cumsum(log(x)))",
                "power": "x^n", "pow": "x^n", "sqr": "x^2",
                "ceiling": "ceil(x)", "if": "iif(condition, a, b)", "ifelse": "iif(condition, a, b)", "where": "iif(condition, a, b)",
                "isnan": "isna(x)", "isnull": "isna(x)", "ismissing": "isna(x)", "nan": "na (the missing value)",
                "coalesce": "fillna(x, v)", "nan_to": "fillna(x, v)", "randn": "normrnd(seed=1)", "rand": "unifrnd(seed=1)",
                "rnorm": "normrnd(seed=1)", "runif": "unifrnd(seed=1)", "obsid": "obsnum()"}


def _suggestion(fn: str) -> str:
    """A hint for a function name pymetrix does not know: another package's name for one it has, or a close spelling."""
    if fn.lower() in _OTHER_NAMES:
        return f": did you mean {_OTHER_NAMES[fn.lower()]}?"
    import difflib

    close = difflib.get_close_matches(fn, _FUNCTIONS, n=1, cutoff=0.75)
    return f": did you mean {close[0]}()?" if close else ""

Token = tuple[str, str, int]  # kind, text, position


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ExpressionError(f"unexpected character {text[pos]!r} at position {pos} in {text!r}")
        kind = m.lastgroup or ""
        if kind != "ws":
            value = m.group(kind)
            tokens.append((kind, "^" if value == "**" else value, pos))
        pos = m.end()
    tokens.append(("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str, params: Iterable[str]):
        self.text = text
        self.params = frozenset(params)
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at(self, op: str) -> bool:
        kind, value, _ = self.peek()
        return kind == "op" and value == op

    def expect(self, op: str) -> None:
        tok = self.next()
        if tok[0] != "op" or tok[1] != op:
            self.error(f"expected {op!r}", tok)

    def error(self, msg: str, tok: Token | None = None) -> None:
        tok = tok or self.peek()
        raise ExpressionError(f"{msg} at position {tok[2]} in {self.text!r}")

    def expect_end(self) -> None:
        if self.peek()[0] != "end":
            self.error(f"unexpected {self.peek()[1]!r}")

    # grammar

    def keyword(self, word: str) -> bool:
        kind, value, _ = self.peek()
        return kind == "name" and value == word

    def expression(self) -> Expr:
        node = self.conjunction()
        while self.keyword("or"):
            self.next()
            node = BinOp("or", node, self.conjunction())
        return node

    def conjunction(self) -> Expr:
        node = self.negation()
        while self.keyword("and"):
            self.next()
            node = BinOp("and", node, self.negation())
        return node

    def negation(self) -> Expr:
        if self.keyword("not") and not (self.i + 1 < len(self.tokens) and self.tokens[self.i + 1][:2] == ("op", "(")):
            self.next()
            return Call("not", (self.negation(),))
        return self.comparison()

    def comparison(self) -> Expr:
        node = self.additive()
        kind, value, _ = self.peek()
        if kind == "op" and value in COMPARISONS:
            self.next()
            node = BinOp(value, node, self.additive())
            kind, value, _ = self.peek()
            if kind == "op" and value in COMPARISONS:
                self.error("join comparisons with 'and', as in (a < b) and (b < c)")
        return node

    def additive(self) -> Expr:
        node = self.term()
        while self.at("+") or self.at("-"):
            op = self.next()[1]
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> Expr:
        node = self.unary()
        while self.at("*") or self.at("/"):
            op = self.next()[1]
            node = BinOp(op, node, self.unary())
        return node

    def unary(self) -> Expr:
        if self.at("-"):
            self.next()
            return Neg(self.unary())
        if self.at("+"):
            self.next()
            return self.unary()
        return self.power()

    def power(self) -> Expr:
        base = self.postfix()
        if self.at("^"):
            self.next()
            return BinOp("^", base, self.unary())
        return base

    def postfix(self) -> Expr:
        node = self.atom()
        while self.at("["):
            open_tok = self.next()
            sign = 1
            if self.at("-") or self.at("+"):
                sign = -1 if self.next()[1] == "-" else 1
            tok = self.next()
            if tok[0] != "num" or not tok[1].isdigit():
                self.error("a lag must be an integer, as in x[-1]", tok)
            self.expect("]")
            if node.params():
                self.error("parameters cannot be lagged", open_tok)
            node = node.shift(sign * int(tok[1]))
        return node

    def atom(self) -> Expr:
        tok = self.next()
        kind, value, _ = tok
        if kind == "num":
            return Num(float(value))
        if kind == "name" and value in LOGICAL:
            self.error(f"'{value}' joins two conditions, as in (u > 5) {value} (infl > 2)", tok)
        if kind == "name":
            if self.at("("):
                self.next()
                args: list[Expr] = []
                by: Name | None = None
                seed: int | None = None
                if not self.at(")"):
                    while True:
                        if self.peek_keyword("seed"):
                            self.next()
                            self.next()
                            given = self.next()
                            if given[0] != "num" or not given[1].isdigit():
                                self.error("seed= takes a whole number, e.g. seed=1", given)
                            seed = int(given[1])
                        elif self.peek_keyword("by"):
                            self.next()
                            self.next()
                            group = self.expression()
                            if not isinstance(group, Name) or group.lag != 0:
                                self.error("by= takes an axis or a series name, e.g. by=country", tok)
                            by = group  # type: ignore[assignment]
                        else:
                            args.append(self.expression())
                        if not self.at(","):
                            break
                        self.next()
                self.expect(")")
                if seed is not None and value not in RANDOM:
                    self.error(f"{value}() does not take seed=; seed= seeds random draws, as in normrnd(seed=1)", tok)
                if value in RANDOM:
                    return self.random(value, args, seed, tok)
                return self.call(value, args, tok, by)
            if value == NA and value not in self.params:
                return Call(NA, ())
            return Coef(value) if value in self.params else Name(value)
        if kind == "op" and value == "(":
            node = self.expression()
            self.expect(")")
            return BinOp(node.op, node.left, node.right, grouped=True) if isinstance(node, BinOp) else node
        self.error("expected a number, a name or '('", tok)
        raise AssertionError  # unreachable

    def random(self, fn: str, args: list[Expr], seed: int | None, tok: Token) -> Expr:
        """``normrnd(mu, sigma, seed=s)``, ``unifrnd(a, b, seed=s)``: stored with the defaults filled in and the seed last."""
        names = "mu = 0, sigma = 1" if fn == "normrnd" else "a = 0, b = 1"
        if len(args) > 2:
            self.error(f"{fn}() takes at most two parameters and the seed: {fn}({names}, seed=1)", tok)
        if seed is None:
            self.error(f"{fn}() needs seed=, as in {fn}(seed=1): seeded, a formula gives the same draws each time it is "
                       "computed and when the session is replayed", tok)
        full = list(args) + [Num(d) for d in RANDOM[fn][len(args):]]
        return Call(fn, (*full, Num(float(seed))))

    def peek_keyword(self, word: str) -> bool:
        """The next tokens are ``word =`` (a keyword argument)."""
        here = self.tokens[self.i]
        nxt = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        return bool(nxt and here[0] == "name" and here[1] == word and nxt[0] == "op" and nxt[1] == "=")

    def call(self, fn: str, args: list[Expr], tok: Token, by: Name | None = None) -> Expr:
        if by is not None and fn not in GROUPWISE:
            self.error(f"{fn}() does not take by=; by= groups a group-wise function such as mean(y, by=country)", tok)
        if fn in ELEMENTWISE and len(args) >= 2:
            if by is not None:
                self.error(f"{fn}(a, b, ...) is period by period; for each group's {fn} of a series write {fn}(y, by=country)", tok)
            node = Call(fn, (args[0], args[1]))
            for more in args[2:]:
                node = Call(fn, (node, more))  # max(a, b, c) = max(max(a, b), c)
            return node
        if fn in PRIMITIVES:
            if len(args) != 1:
                self.error(f"{fn}() takes one argument", tok)
            return Call(fn, (args[0],))
        if fn in CHOICES:
            low, high = CHOICES[fn]
            shape = {"round": "round(x) or round(x, n), n decimals", "fillna": "fillna(x, v), v where x is missing",
                     "iif": "iif(condition, a, b), a where the condition holds, b where not"}[fn]
            if not low <= len(args) <= high:
                self.error(f"{fn}() takes {low if low == high else f'{low} or {high}'} arguments: {shape}", tok)
            if fn == "round" and len(args) == 2:
                n = args[1]
                n = Num(-n.operand.value) if isinstance(n, Neg) and isinstance(n.operand, Num) else n
                if not (isinstance(n, Num) and float(n.value).is_integer()):
                    self.error("round(x, n): n, the number of decimals, is a whole number (negative rounds to tens, hundreds...)", tok)
                return Call(fn, (args[0], n))
            return Call(fn, tuple(args))
        if fn in CALENDAR:
            if args:
                self.error(f"{fn}() takes no arguments: it comes from the data's time axis", tok)
            return Call(fn, ())
        if fn in ("d", "dlog"):
            if len(args) not in (1, 2):
                self.error(f"{fn}() takes one or two arguments", tok)
            n = self.order(args, tok)
            x = args[0] if fn == "d" else Call("log", (args[0],))
            for _ in range(n):
                x = BinOp("-", x, x.shift(-1))
            text = f"{fn}({args[0]})" if n == 1 else f"{fn}({args[0]}, {n})"
            return BinOp(x.op, x.left, x.right, grouped=True, source=text)  # type: ignore[attr-defined]
        if fn == "pch":
            if len(args) != 1:
                self.error("pch() takes one argument", tok)
            x = args[0]
            return BinOp("*", Num(100.0), BinOp("-", BinOp("/", x, x.shift(-1)), Num(1.0)), grouped=True, source=f"pch({x})")
        if fn in GROUPWISE:
            if len(args) not in (1, 2) or (by is not None and len(args) != 1):
                self.error(f"{fn}() takes a series and, optionally, what to group by: {fn}(y, by=country)", tok)
            if by is not None:
                return Call(fn, (args[0], Level(by.id)))
            if len(args) == 2:
                if not isinstance(args[1], Name) or args[1].lag != 0:
                    self.error(f"{fn}(): group by an axis or a series name, e.g. {fn}({args[0]}, country)", tok)
                return Call(fn, (args[0], Level(args[1].id)))  # type: ignore[union-attr]
            return Call(fn, (args[0],))
        if fn in DISTRIBUTIONS or fn in SPECIAL:
            low, high = arity(fn)
            if not low <= len(args) <= high:
                self.error(f"{fn}() takes {low if low == high else f'{low} to {high}'} argument{'s' * (high > 1)}: {signature(fn)}", tok)
            return Call(fn, tuple(args))
        if fn in MOVING:
            k = MOVING[fn]
            if len(args) != k + 1:
                self.error(f"{fn}() takes {'two series' if k == 2 else 'a series'} and the window length: "
                           f"{fn}({'x, y' if k == 2 else 'x'}, 4)", tok)
            n = args[-1]
            least = 1 if CENTRED.get(fn, fn) in ("movavg", "movsum", "movmin", "movmax") else 2
            if not (isinstance(n, Num) and float(n.value).is_integer() and n.value >= least):
                self.error(f"{fn}(): the window length must be an integer of at least {least}", tok)
            width = int(n.value)  # type: ignore[union-attr]
            if width > 1000:
                self.error(f"{fn}(): a window of at most 1000 periods", tok)
            lead = window_lead(fn, width)
            return Call(fn, tuple(s.shift(lead - j) for s in args[:k] for j in range(width)))
        if fn in EXPANDING:
            k = MOVING[EXPANDING[fn]]
            if len(args) != k:
                self.error(f"{fn}() takes {'two series' if k == 2 else 'a series'} (every period up to t): "
                           f"{fn}({'x, y' if k == 2 else 'x'})", tok)
            return Call(fn, tuple(args))
        if fn == "seasonal":
            if args:
                self.error("seasonal() takes no arguments: its dummies come from the data's time axis", tok)
            return Call(fn, ())
        if fn in MARKERS:
            if len(args) != 1 or not isinstance(args[0], Name) or args[0].lag != 0:
                self.error(f"{fn}() takes one variable name", tok)
            return Call(fn, (args[0],))
        hint = _suggestion(fn)  # a likely name: that alone (the list of every function buried it in an error box)
        self.error(f"unknown function {fn!r}{hint}" if hint else f"unknown function {fn!r} (known: {', '.join(_FUNCTIONS)})", tok)
        raise AssertionError  # unreachable

    def order(self, args: list[Expr], tok: Token) -> int:
        if len(args) == 1:
            return 1
        n = args[1]
        if not (isinstance(n, Num) and n.value.is_integer() and n.value >= 1):
            self.error("the difference order must be a positive integer", tok)
        return int(n.value)  # type: ignore[union-attr]


def parse_expression(text: str, params: Iterable[str] = ()) -> Expr:
    """Parse an expression such as ``"dlog(y) - 0.5*x[-1]"``."""
    p = _Parser(text, params)
    node = p.expression()
    if p.at("~") or p.at("="):
        p.error("this is an equation; an expression was expected")
    p.expect_end()
    return node


def parse_equation(text: str, params: Iterable[str] = ()) -> tuple[Expr, str, Expr]:
    """Parse ``lhs ~ rhs`` (stochastic) or ``lhs = rhs`` (identity)."""
    p = _Parser(text, params)
    lhs = p.expression()
    tok = p.next()
    if tok[0] != "op" or tok[1] not in ("~", "="):
        raise ExpressionError(
            f"{text!r} is not an equation: separate the two sides with '~' for an estimated "
            "equation (e.g. 'c ~ y + c[-1]') or '=' for an identity (e.g. 'y = c + i + g')"
        )
    rhs = p.expression()
    p.expect_end()
    return lhs, tok[1], rhs
