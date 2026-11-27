"""Algebraic notation: equations, coefficients and parameters as an economist writes them.

    Equation("c ~ y + c[-1]").algebraic()                      # 'cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ'
    Equation("c ~ y + c[-1]").algebraic("latex")               # 'c_t = \\beta_0 + \\beta_1 y_t + ...'
    fit.algebraic(estimates=True)                              # 'cₜ = 9.632 + 0.6425 yₜ + 0.1456 cₜ₋₁'
    Equation("pi ~ beta*pi[+1] + kappa*y", params=["beta", "kappa"]).algebraic()   # 'πₜ = β πₜ₊₁ + κ yₜ + εₜ'
    set_notation("algebraic")                                  # summaries titled and labelled this way

Notation is a way of *showing* equations; scripts, saved projects and generated code always keep the
code form (``c ~ y + c[-1]``), so a project means the same whichever notation is on.

- **Coefficients** of a linear-form equation are numbered: β₀ for the constant, then β₁, β₂, ... in the
  order of the terms. Parameters of an explicit-form equation keep their names.
- **Greek names** display as Greek letters, whether typed as Unicode (``α``) or spelled out (``alpha``,
  ``beta``, ``kappa``, ``phi``, ``Gamma``...), for parameters and variables alike (``pi`` is π). A
  parameter's trailing digits and anything after an underscore become a subscript: ``b1`` is b₁,
  ``phi_pi`` is φ with subscript π.
- **Time** is a subscript: ``c[-1]`` is cₜ₋₁ and ``pi[+1]`` is πₜ₊₁.
- **Functions:** ``log`` and ``exp`` as written, ``d(x)`` as Δx, ``dlog(x)`` as Δlog x, ``pch(x)`` as
  %Δx.
- **The error** εₜ is shown for estimated (stochastic) equations, not identities.

Unicode suits windows, tables, Word and Excel; LaTeX suits papers, slides and notebooks.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from .errors import SpecificationError
from .expr.nodes import MOVING, NA, BinOp, Call, Coef, Expr, Level, Name, Neg, Num, window_lead

if TYPE_CHECKING:
    from .estimation import Equation

__all__ = ["GREEK", "algebraic", "coefficient_symbols", "expression", "get_notation", "notation", "set_notation", "symbol"]

Style = Literal["unicode", "latex"]

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "lam": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}
_LATEX_GREEK = {v: ("\\" + k if k != "lam" else "\\lambda") for k, v in GREEK.items()}
_LATEX_GREEK["ε"] = "\\varepsilon"
_LATEX_GREEK.update({"ϱ": "\\varrho", "ϑ": "\\vartheta", "ϕ": "\\varphi", "ς": "\\varsigma", "ϵ": "\\epsilon"})  # variant letters
_SUBSCRIPTABLE = "0123456789+-=()aehijklmnoprstuvxβγρφχ"
_SUB = str.maketrans(_SUBSCRIPTABLE, "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵦᵧᵨᵩᵪ")
_SUP = str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")

_NOTATION = "code"


def set_notation(value: Literal["code", "algebraic"]) -> None:
    """How summaries show equations: ``"code"`` (as typed, the default) or ``"algebraic"``."""
    global _NOTATION
    if value not in ("code", "algebraic"):
        raise SpecificationError("notation is 'code' or 'algebraic'")
    _NOTATION = value


def get_notation() -> str:
    return _NOTATION


@contextlib.contextmanager
def notation(value: Literal["code", "algebraic"]) -> Iterator[None]:
    """``with notation("algebraic"): ...``: for one block only."""
    before = _NOTATION
    set_notation(value)
    try:
        yield
    finally:
        set_notation(before)


# -- symbols ---------------------------------------------------------------------


def _subscript(text: str, style: Style) -> str:
    if not text:
        return ""
    if style == "latex":
        return f"_{{{text}}}" if len(text) > 1 else f"_{text}"
    if all(ch in _SUBSCRIPTABLE for ch in text):
        return text.translate(_SUB)
    return f"_{text}"  # Unicode has no subscript for some letters: φ_π, not an ambiguous φπ


def _letters(name: str, style: Style) -> str:
    greek = GREEK.get(name, name)
    if style == "unicode":
        return greek
    if greek in _LATEX_GREEK:
        return _LATEX_GREEK[greek]
    if len(greek) == 1 and greek.isascii():
        return greek
    return "".join(_LATEX_GREEK.get(ch, ch) for ch in greek) if all(ch in _LATEX_GREEK for ch in greek) else f"\\mathit{{{greek}}}"


def symbol(name: str, style: Style = "unicode", parameter: bool = False) -> str:
    """One name in algebraic notation: Greek where spelled out, and for parameters, digits and the part
    after an underscore as a subscript (``phi_pi`` is φ with subscript π)."""
    base, _, sub = name.partition("_")
    if parameter and not sub:
        m = re.fullmatch(r"([^\W\d_]+?)(\d+)", name)
        if m:
            base, sub = m.group(1), m.group(2)
    if not sub:
        return _letters(name, style)
    sub_text = "".join(_letters(part, style) if part in GREEK else part for part in re.split(r"(\W)", sub))
    return _letters(base, style) + _subscript(sub_text if style == "latex" else GREEK.get(sub, sub), style)


def _time(lag: int, style: Style) -> str:
    t = "t" if lag == 0 else f"t{lag:+d}"
    return _subscript(t, style) if style == "latex" else t.translate(_SUB)


def _given(text: str, style: Style) -> str:
    """A symbol given in metadata (``Param(symbol="φ_π")``), as Unicode or LaTeX."""
    if style == "unicode":
        return text
    base, _, sub = text.partition("_")
    convert = lambda part: "".join(_LATEX_GREEK.get(ch, ch) for ch in part)  # noqa: E731
    return convert(base) + (_subscript(convert(sub), style) if sub else "")


def from_latex(text: str) -> str:
    """A symbol written in LaTeX (``\\phi_{\\pi}``, as Dynare's ``$...$`` names give it) in the Unicode form
    ``Param.symbol`` takes (``φ_π``)."""
    out = text.strip().strip("$").strip()
    for letter, latex in sorted(_LATEX_GREEK.items(), key=lambda kv: -len(kv[1])):
        out = re.sub(re.escape(latex) + r"(?![A-Za-z])", letter, out)
    out = re.sub(r"\\(?:mathit|mathrm|text|bar|hat|tilde)\{([^{}]*)\}", r"\1", out)
    return re.sub(r"[{}\s]", "", out).replace("\\", "")


def coefficient_symbols(eq: Equation, style: Style = "unicode", letter: str = "beta") -> dict[str, str]:
    """The symbol for each coefficient name of a fit of ``eq``: β₀, β₁, ... for a linear form; for an
    explicit form, each parameter's ``Param.symbol`` if given, else its name in algebraic notation."""
    if eq.form == "explicit":
        return {p.name: _given(p.symbol, style) if p.symbol else symbol(p.name, style, parameter=True) for p in eq.param_specs}
    names = eq.coefficient_names()
    base = _letters(letter, style)
    return {n: base + _subscript(str(i), style) for i, n in enumerate(names)}


# -- expressions -----------------------------------------------------------------

_PREC = {"or": -2, "and": -1, "<": 0, "<=": 0, ">": 0, ">=": 0, "==": 0, "!=": 0, "+": 1, "-": 1, "*": 2, "/": 2, "^": 4}
_RELATIONS = {"<": ("<", "<"), "<=": ("≤", "\\le"), ">": (">", ">"), ">=": ("≥", "\\ge"), "==": ("=", "="), "!=": ("≠", "\\ne"),
              "and": ("and", "\\;\\text{and}\\;"), "or": ("or", "\\;\\text{or}\\;")}
_FUNCS = {"log": ("log ", "\\log "), "exp": ("exp", "\\exp"), "d": ("Δ", "\\Delta "), "dlog": ("Δlog ", "\\Delta\\log "),
          "pch": ("%Δ", "\\%\\Delta "), "abs": ("abs", "\\operatorname{abs}"), "not": ("not ", "\\neg ")}
_NORMAL = {"normcdf": ("Φ", "\\Phi"), "normpdf": ("φ", "\\phi"), "norminv": ("Φ⁻¹", "\\Phi^{-1}")}


def _prec(e: Expr) -> int:
    if isinstance(e, BinOp):
        return _PREC[e.op]
    if isinstance(e, Neg) or (isinstance(e, Num) and e.value < 0):
        return 3
    return 5


def _number(v: float, digits: int) -> str:
    text = f"{v:.{digits}g}" if abs(v) < 1e6 else f"{v:.{digits}e}"
    return text


class _Writer:
    def __init__(self, style: Style, values: Mapping[str, float] | None, coefs: Mapping[str, str], digits: int,
                 symbols: Mapping[str, str] | None = None):
        self.style, self.values, self.coefs, self.digits = style, values or {}, coefs, digits
        self.symbols = symbols or {}

    def __call__(self, e: Expr) -> str:
        s = self.style
        if isinstance(e, Num):
            return _number(float(e.value), self.digits)
        if isinstance(e, Name):
            given = self.symbols.get(e.id)
            return (_given(given, s) if given else symbol(e.id, s)) + _time(e.lag, s)
        if isinstance(e, Level):
            return symbol(e.id, s)
        if isinstance(e, Coef):
            if e.id in self.values:
                return _number(float(self.values[e.id]), self.digits)
            return self.coefs.get(e.id) or symbol(e.id, s, parameter=True)
        if isinstance(e, Neg):
            return "-" + self.wrap(e.operand, 3) if s == "latex" else "−" + self.wrap(e.operand, 3)
        if isinstance(e, Call):
            return self.call(e)
        assert isinstance(e, BinOp)
        shorthand = self.shorthand(e)
        if shorthand is not None:
            return shorthand
        if e.op == "^":
            base = self.wrap(e.left, 5)
            if s == "latex":
                return f"{base}^{{{self(e.right)}}}"
            exponent = self(e.right)
            if exponent and all(ch in "0123456789+-()" for ch in exponent):
                return base + exponent.translate(_SUP)
            return base + "^" + (exponent if isinstance(e.right, (Num, Name, Coef)) else f"({exponent})")
        if e.op == "/" and s == "latex":
            return f"\\frac{{{self(e.left)}}}{{{self(e.right)}}}"
        p = _PREC[e.op]
        if e.op in _RELATIONS:
            sign = _RELATIONS[e.op][1 if s == "latex" else 0]
            return f"{self.wrap(e.left, p + 1)} {sign} {self.wrap(e.right, p + 1)}"
        left = self.wrap(e.left, p)
        right = self.wrap(e.right, p + (e.op in ("-", "/")))
        if e.op == "*":
            if s == "unicode" and isinstance(e.left, BinOp) and e.left.op == "/" and self.shorthand(e.left) is None:
                left = f"({left})"  # (1 / σ) (iₜ − πₜ₊₁): side by side, 1 / σ (iₜ − πₜ₊₁) would read as one fraction
            juxtapose = not isinstance(e.right, Num)  # factors side by side, as in algebra; a dot before a number
            return f"{left} {right}" if juxtapose else (f"{left} \\cdot {right}" if s == "latex" else f"{left}·{right}")
        if e.op == "+" and right[:1] in ("-", "−"):  # a negative estimate: x − 0.5 y, not x + -0.5 y
            return f"{left} {'-' if s == 'latex' else '−'} {right[1:]}"
        if e.op == "-" and s == "unicode":
            return f"{left} − {right}"
        return f"{left} {e.op} {right}"

    def wrap(self, e: Expr, p: int) -> str:
        text = self(e)
        if _prec(e) >= p or (isinstance(e, BinOp) and self.shorthand(e) is not None):  # Δlog y is one term
            return text
        return f"\\left({text}\\right)" if self.style == "latex" else f"({text})"

    def shorthand(self, e: BinOp) -> str | None:
        """d(x), dlog(x) and pch(x) as typed: the parser expands them, keeping the text in ``source``."""
        m = re.fullmatch(r"(d|dlog|pch)\((.*)\)", e.source or "")
        if m is None:
            return None
        from .expr import parse_expression

        try:
            inner = parse_expression(m.group(2), self.coefs)
        except Exception:
            return None
        return self.call(Call(m.group(1), (inner,)))

    def call(self, e: Call) -> str:
        latex = self.style == "latex"
        if e.fn == NA and not e.args:  # the missing value
            return "\\text{NA}" if latex else "NA"
        if e.fn in MOVING:  # the window, not its lags: movavg(x, 4)
            k = MOVING[e.fn]
            n = len(e.args) // k
            at = window_lead(e.fn, n)
            shown = [self(e.args[i * n + at]) for i in range(k)] + [str(n)]
            name = f"\\operatorname{{{e.fn}}}" if latex else e.fn
            return f"{name}\\left({', '.join(shown)}\\right)" if latex else f"{name}({', '.join(shown)})"
        if e.fn in _NORMAL:  # Φ(x), φ(x), Φ⁻¹(p); with a mean and standard deviation, Φ(x; μ, σ)
            name = _NORMAL[e.fn][latex]
            inner = "; ".join([self(e.args[0])] + ([", ".join(self(a) for a in e.args[1:])] if len(e.args) > 1 else []))
            return f"{name}\\left({inner}\\right)" if latex else f"{name}({inner})"
        if e.fn in _FUNCS and len(e.args) == 1:
            prefix = _FUNCS[e.fn][latex]
            arg = e.args[0]
            simple = isinstance(arg, (Name, Coef, Num))
            inner = self(arg)
            if e.fn in ("exp", "abs") or not simple:
                inner = f"\\left({inner}\\right)" if latex else f"({inner})"
            return prefix + inner
        if e.fn == "sqrt" and len(e.args) == 1:
            return f"\\sqrt{{{self(e.args[0])}}}" if latex else f"√({self(e.args[0])})"
        name = (f"\\{e.fn}" if e.fn in ("max", "min") else f"\\operatorname{{{e.fn}}}") if latex else e.fn
        args = ", ".join(self(a) for a in e.args)
        return f"{name}\\left({args}\\right)" if latex else f"{name}({args})"


def algebraic(eq: Equation, style: Style = "unicode", values: Mapping[str, float] | None = None, error: bool | None = None,
              digits: int = 4, letter: str = "beta", symbols: Mapping[str, str] | None = None) -> str:
    """``eq`` in algebraic notation. ``values`` replaces coefficients with numbers (an estimated equation);
    ``error`` adds εₜ (default: for stochastic equations without values); ``symbols`` gives variables their
    own symbols, e.g. ``{"gap": "ŷ"}`` or a dataset's ``Variable.symbol`` values."""
    if style not in ("unicode", "latex"):
        raise SpecificationError("style is 'unicode' or 'latex'")
    coefs = coefficient_symbols(eq, style, letter)
    write = _Writer(style, values, coefs, digits, symbols)
    rhs = eq.rhs if eq.form != "linear" else eq.linear_rhs()
    text = f"{write(eq.lhs)} = {write(rhs)}"
    if error is None:
        error = not eq.is_identity and not values
    if error:
        eps = "\\varepsilon_t" if style == "latex" else "εₜ"
        text += f" + {eps}"
    return text


def expression(e: Expr, style: Style = "unicode", values: Mapping[str, float] | None = None, digits: int = 4,
               symbols: Mapping[str, str] | None = None, parameters: Mapping[str, str] | None = None) -> str:
    """One expression in algebraic notation: its parameters by their own symbols, or those in ``parameters``
    (``{"phi_pi": "φ_π"}``, as ``Param.symbol`` gives them), variables by ``symbols``."""
    coefs = {k: _given(v, style) for k, v in (parameters or {}).items()}
    return _Writer(style, values, coefs, digits, symbols)(e)


def labelled(names: Any, eq: Equation) -> list[str]:
    """Coefficient labels for a summary in algebraic notation: ``β₁ (y)``, or the parameter's symbol."""
    coefs = coefficient_symbols(eq)
    out = []
    for n in names:
        sym = coefs.get(n)
        out.append(n if sym is None else (sym if eq.form == "explicit" or sym == n else f"{sym} ({n})"))
    return out
