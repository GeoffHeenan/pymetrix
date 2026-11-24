"""Phase 2: estimation and tests. Equations (``ls``, ``tsls``, ``gmm``, ``liml``, ``arch``, ``binary``, ``ordered``,
``censored``, ``count``, ``qreg``, ``heckit``, ``ardl``), VARs and VECs, systems, the tests on an equation, unit-root
and cointegration tests, forecasts and residuals added to a page, and the values a program reads from a result
(``eq.@coefs(2)``, ``@r2``, ``c(2)`` after an estimation).

A list specification ``y c x y(-1)`` becomes ``y ~ x + y[-1]`` (``- 1`` without ``c``); an expression specification
``y = c(1) + c(2)*x`` an explicit-form equation with parameters ``c_1``, ``c_2``. Each estimation keeps its
coefficient map: EViews's ``c(i)`` against pymetrix's term (``const``, ``x``, ``y[-1]``) or parameter (``c_2``), which
later lines use (``eq.@coefs(2)``, ``eq.wald c(2)=0``).

Every option is either mapped or the line is left out naming it: an estimator that runs with an option dropped would
give other numbers without saying so. Where EViews's defaults differ from pymetrix's (HAC bandwidths, the ARMA and
GARCH estimators, information criteria per observation), the line is translated with a note saying what to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ir import Arg, Cmd, Code, New, Op, RefTo, SampleSpec, Step, Var, quote

__all__ = ["ESTIMATORS", "Estimation", "Fit", "LEFT_ESTIMATORS"]

# EViews's estimation commands translated here
ESTIMATORS = ("ls", "tsls", "gmm", "liml", "arch", "binary", "logit", "probit", "ordered", "censored", "count", "qreg", "heckit",
              "ardl")
# left out, naming pymetrix's equivalents (design note: phase 2b if wanted)
LEFT_ESTIMATORS = {
    "stepls": "stepwise least squares has no pymetrix equivalent; choose the regressors, then Equation(...).fit(data)",
    "breakls": "least squares with breaks: bai_perron(...) finds the breaks; then fit each regime (Equation(...).fit(data, span=...))",
    "threshold": "threshold regression: ThresholdAR(...).fit(data) (Model → Threshold autoregression)",
    "switchreg": "switching regression: MarkovSwitching(...).fit(data)",
    "glm": "generalised linear models: Equation(..., family=Logit(), Probit(), Poisson() ...).fit(data, method=ML())",
    "nls": "EViews's nls options are not translated: an expression specification with ls is (Equation in explicit form)",
}
_NOTE_HAC = ("HAC (Newey-West) standard errors: EViews's default bandwidth and kernel may differ from pymetrix's "
             "(Bartlett, floor(4(n/100)^(2/9)) lags); check them and give hac_lags= to match")
_NOTE_ARMA = ("ARMA errors are estimated by exact maximum likelihood (method=ML()); EViews's default ARMA estimator differs "
              "by version (verify: conditional least squares, or ML/GLS from EViews 10), so estimates may differ")


@dataclass
class Fit:
    """An estimated object of the program: its Python name, what it is, its page, its coefficient map (pymetrix's names
    in EViews's c(i) order) and what its re-reading by the model reader needs."""

    var: str
    kind: str  # equation, var, vecm, system, ardl
    page: Any
    coefs: list[str] = field(default_factory=list)
    method: str = "ls"
    classic: bool = True  # least squares with classic standard errors
    spec: str = ""  # the EViews specification (for m.merge)
    sample: Any = None  # the page's sample when it was estimated
    dependent: str = ""
    vector: str = "c"


def _split_top(text: str, at: int) -> list[tuple[str, int]]:
    """Words separated by spaces outside brackets, with their offsets."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text + " "):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth <= 0:
            if start is not None:
                out.append((text[start:i], at + start))
                start = None
        elif start is None:
            start = i
    return out


def _commas(text: str | None) -> list[str]:
    """Options split at commas outside brackets."""
    out, depth, word = [], 0, ""
    for ch in text or "":
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(word.strip())
            word = ""
        else:
            word += ch
    if word.strip():
        out.append(word.strip())
    return out


def _options(text: str | None) -> tuple[dict[str, str], list[str]]:
    """``key=value`` options (keys lower case) and the flags, in order."""
    named: dict[str, str] = {}
    flags: list[str] = []
    for part in _commas(text):
        if "=" in part:
            k, _, v = part.partition("=")
            named[k.strip().lower()] = v.strip().strip('"')
        else:
            flags.append(part.strip().lower())
    return named, flags


def _value(text: str, what: str, kind: Any = int) -> Any:
    """An option's number, or the line left out saying it is not one (a scalar's name, say)."""
    from .data import _left

    try:
        return kind(text)
    except (TypeError, ValueError):
        raise _left(f"{what}={text} is not translated: a number is") from None


def _term_name(term: str) -> str:
    """The name pymetrix gives a term's coefficient: its text, without brackets around all of it."""
    t = term.strip()
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        for i, ch in enumerate(t):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(t) - 1:
                return t
        t = t[1:-1].strip()
    return t


def _in_restriction(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_]\w*(\[-?\d+\])?|const|[A-Za-z_]\w*\([^()]*\)", name) else f"({name})"


@dataclass
class _Spec:
    formula: str
    params: list[str]
    coefs: list[str]  # by c(i), from 1
    dependent: str
    terms: list[str]  # regressors as pymetrix writes them (no c)
    intercept: bool
    ar: list[int]
    ma: list[int]
    explicit: bool
    vector: str = "c"


class Estimation:
    """The phase 2 handlers for estimation and tests, mixed into the ``Translator`` (``data.py``)."""

    # -- specifications ---------------------------------------------------------------------------------------------

    def _list_terms(self, words: list[tuple[str, int]], page: Any, arma: bool, what: str) -> tuple[list[str], bool, list[int], list[int]]:
        from .data import _left

        terms: list[str] = []
        intercept = False
        ar: list[int] = []
        ma: list[int] = []
        for word, at in words:
            low = word.lower()
            if low == "c":
                intercept = True
                terms.append("const")
                continue
            m = re.fullmatch(r"(ar|ma)\(\s*(\d+)\s*\)", low)
            if m:
                if not arma:
                    raise _left(f"{word}: ARMA terms are translated for ls only")
                (ar if m.group(1) == "ar" else ma).append(int(m.group(2)))
                terms.append(f"{m.group(1)}[{int(m.group(2))}]")
                continue
            if re.match(r"(sar|sma|ar|ma)\(", low):
                raise _left(f"{word}: seasonal ARMA terms and ARMA ranges are not translated: Equation(..., errors=ARMA(ar=[...], "
                            "ma=[...])) with the lags listed")
            if low.startswith(("pdl(", "@expand(")) or low in ("@expand", "pdl"):
                raise _left(f"{word}: polynomial distributed lags and @expand are not translated: write the terms out")
            text = self.expression(word, at, page)
            if text.names:
                raise _left(f"{word} uses the program's scalars ({', '.join(text.names)}): a {what}'s regressors are series")
            terms.append(str(text))
        return terms, intercept, ar, ma

    def _spec(self, text: str, at: int, page: Any, arma: bool = False, what: str = "specification") -> _Spec:
        """A list or expression specification as pymetrix's formula, with the coefficient map."""
        from .data import _error, _left, _top_level

        if _top_level(text, "="):
            split = _top_level(text, "=")
            if len(split) != 1:
                raise _error("an expression specification has one '=': y = c(1) + c(2)*x")
            lhs = self.expression(text[:split[0]], at, page)
            rhs = self.expression(text[split[0] + 1:], at + split[0] + 1, page, coefs=True)
            if lhs.names or rhs.names:
                raise _left("an expression specification that uses the program's scalars is not translated: put their values in")
            found = sorted({(m.group(1), int(m.group(2))) for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)_(\d+)(?!\w)", str(rhs))
                            if m.group(1) in self._vectors()}, key=lambda p: (p[0], p[1]))
            if not found:
                raise _error("an expression specification needs coefficients: c(1), c(2) ...")
            vectors = {v for v, _ in found}
            if len(vectors) > 1:
                raise _left("an expression specification with two coefficient vectors is not translated")
            vector = vectors.pop()
            params = [f"{vector}_{i}" for _, i in found]
            top = max(i for _, i in found)
            coefs = [f"{vector}_{i}" for i in range(1, top + 1)]
            dependent = _term_name(str(lhs))
            return _Spec(f"{lhs} ~ {rhs}", params, coefs, dependent, [], False, [], [], True, vector)
        words = _split_top(text, at)
        if len(words) < 2:
            raise _error("a specification is: dependent regressor ... (c the constant), or y = c(1) + c(2)*x")
        dep = self.expression(words[0][0], words[0][1], page)
        if dep.names:
            raise _left("a dependent variable that uses the program's scalars is not translated")
        if words[0][0].lower() == "c":
            raise _error("the first in a list specification is the dependent variable, not c")
        terms, intercept, ar, ma = self._list_terms(words[1:], page, arma, what)
        regressors = [t for t in terms if t != "const" and not re.fullmatch(r"(ar|ma)\[\d+\]", t)]
        if not regressors and not intercept:
            raise _error("a specification with nothing to estimate")
        rhs = " + ".join(f"({t})" if t.startswith("-") else t for t in regressors) or "1"
        if not intercept and regressors:
            rhs += " - 1"
        coefs = [_term_name(t) if t not in ("const",) else "const" for t in terms]
        return _Spec(f"{dep} ~ {rhs}", [], coefs, _term_name(str(dep)), regressors, intercept, ar, ma, False)

    def _vectors(self) -> set[str]:
        return {"c"} | {k for k, v in self.types.items() if v == "coef"}

    # -- equations --------------------------------------------------------------------------------------------------

    def _cmd_equation(self, st: Any) -> tuple[list[Op], str]:
        from .data import _error
        from .parse import _options as bracket

        m = re.match(r"\s*([A-Za-z_]\w*)(?:\.(\w+))?", st.rest)
        if m is None:
            raise _error("equation names the equation: equation eq.ls y c x")
        name, method = m.group(1), (m.group(2) or "").lower()
        options, after = bracket(st.rest, m.end())
        spec = st.rest[after:]
        spec_at = st.rest_at + after + (len(spec) - len(spec.lstrip()))
        self.types[name.lower()] = "equation"
        if not method:
            if spec.strip():
                return self._estimate(st, "ls", options, spec.strip(), spec_at, name)
            return [], "the equation is declared; a later line estimates it (no step)"
        return self._estimate(st, method, options, spec.strip(), spec_at, name)

    def _estimator(self, st: Any) -> tuple[list[Op], str]:
        """A bare estimation command (``ls y c x``): an equation EViews calls untitled."""
        return self._estimate(st, st.word, st.options, st.rest, st.rest_at, None)

    _cmd_ls = _cmd_tsls = _cmd_gmm = _cmd_liml = _cmd_arch = _cmd_binary = _cmd_logit = _cmd_probit = _estimator
    _cmd_ordered = _cmd_censored = _cmd_count = _cmd_qreg = _cmd_heckit = _cmd_ardl = _estimator

    def _left_estimator(self, st: Any) -> tuple[list[Op], str]:
        from .data import _left

        raise _left(f"{st.word}: {LEFT_ESTIMATORS[st.word]}")

    _cmd_stepls = _cmd_breakls = _cmd_threshold = _cmd_switchreg = _cmd_glm = _cmd_nls = _left_estimator

    def _estimate(self, st: Any, method: str, options: str | None, spec: str, at: int, name: str | None) -> tuple[list[Op], str]:
        from .data import _error, _left

        if method in LEFT_ESTIMATORS:
            raise _left(f"{method}: {LEFT_ESTIMATORS[method]}")
        if method not in ESTIMATORS:
            raise _left(f"{method} is not an estimation method translated here: ls, tsls, gmm, liml, arch, binary, ordered, "
                        "censored, count, qreg, heckit and ardl are")
        if not spec.strip():
            raise _error(f"{method} needs a specification")
        page = self._page()
        if method == "ardl":
            return self._ardl(st, options, spec, at, name, page)
        named, flags = _options(options)
        notes: list[str] = []
        instruments: list[str] | None = None
        selection: str | None = None
        main, main_at = spec, at
        if method in ("tsls", "gmm", "liml", "heckit"):
            parts = re.split(r"(?:^|\s)@(?:\s|$)", spec)
            if len(parts) != 2:
                raise _error(f"{method} takes the specification, then @ and the {'selection equation' if method == 'heckit' else 'instruments'}: "
                             f"{method} y c x @ c z")
            main = parts[0]
            rest_at = at + spec.index(parts[1], len(parts[0]))
            if method == "heckit":
                selection = parts[1]
                sel_at = rest_at
            else:
                words = _split_top(parts[1], rest_at)
                ins, has_c, _, _ = self._list_terms(words, page, False, "instrument list")
                instruments = [t for t in ins if t != "const"]
                if not has_c:
                    notes.append("the instruments have no c: pymetrix's IV, GMM and LIML add the constant (verify that EViews's "
                                 "estimate has it among its instruments)")
        elif "@" in re.sub(r"@\w+", "", spec):
            raise _left(f"{method} with @ in its specification is not translated")
        s = self._spec(main, main_at, page, arma=(method == "ls"), what="specification")
        if s.ar or s.ma:
            if s.explicit:
                raise _left("ARMA terms in an expression specification are not translated")
        eq_args: list[Arg] = [Arg("text", s.formula, "pos")]
        if s.params:
            eq_args.append(Arg("params", tuple(s.params)))
        method_new: New | None = None
        classic = True
        used: set[str] = set()

        def take(*keys: str) -> None:
            used.update(keys)

        if method == "ls":
            cov = self._cov(named, flags, notes, take)
            classic = cov == "classic"
            if s.ar or s.ma:
                orders = [Arg(k, _order(v)) for k, v in (("ar", s.ar), ("ma", s.ma)) if v]
                eq_args.append(Arg("errors", New("ARMA", tuple(orders))))
                method_new = New("ML", (Arg("cov", cov),) if cov != "classic" else ())
                notes.append(_NOTE_ARMA)
                if cov == "hac":
                    raise _left("HAC standard errors with ARMA terms are not translated: ML gives classic or white (sandwich)")
            elif cov != "classic":
                method_new = New("OLS", (Arg("cov", cov),))
        elif method in ("tsls", "liml"):
            assert instruments is not None
            cov = self._cov(named, flags, notes, take)
            classic = False
            names = set(instruments)
            endog = [t for t in s.terms if t not in names]
            excluded = [t for t in instruments if t not in s.terms]
            if s.explicit:
                raise _left(f"{method} with an expression specification is not translated: GMM(instruments=[...]) takes one")
            if not endog:
                notes.append("every regressor is among the instruments: the estimate is least squares'")
            args = [Arg("endog", tuple(endog)), Arg("instruments", tuple(excluded))]
            if cov != "classic":
                args.append(Arg("cov", cov))
            method_new = New("IV" if method == "tsls" else "LIML", tuple(args))
        elif method == "gmm":
            assert instruments is not None
            classic = False
            take("instwgt", "cov")
            wgt = named.get("instwgt", "white").lower()
            if wgt not in ("white", "hac"):
                raise _left(f"gmm(instwgt={wgt}) is not translated: white and hac weights are")
            args = [Arg("instruments", tuple(instruments))]
            if wgt == "hac":
                args.append(Arg("cov", "hac"))
                notes.append(_NOTE_HAC)
            method_new = New("GMM", tuple(args))
            notes.append("GMM: pymetrix's two-step estimator with White weights; EViews's default iterations of the weighting "
                         "matrix and its covariance options may differ (verify: gmm's instwgt=, w= and iteration options)")
        elif method == "arch":
            classic = False
            numbers = [f for f in flags if re.fullmatch(r"\d+", f)]
            p, q = (int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else ((int(numbers[0]), 1) if numbers else (1, 1))
            take(*numbers)
            dist = "normal"
            if "tdist" in flags or named.get("dist", "").lower() in ("t", "tdist"):
                dist = "t"
                take("tdist", "dist")
            elif "ged" in flags or named.get("dist", "").lower() == "ged":
                dist = "ged"
                take("ged", "dist")
            elif "dist" in named and named["dist"].lower() not in ("n", "normal"):
                raise _left(f"arch(dist={named['dist']}) is not translated: normal, t and ged errors are")
            take("dist")
            var_args = [Arg("arch", p), Arg("garch", q)]
            if "egarch" in flags:
                take("egarch")
                cls = "EGARCH"
            else:
                cls = "GARCH"
                if "thrsh" in named:
                    take("thrsh")
                    var_args.append(Arg("asym", _value(named["thrsh"], "thrsh")))
                    notes.append("thrsh= (TARCH): pymetrix's GARCH(asym=...), the GJR form with the threshold on the variance, "
                                 "as EViews's TARCH (verify)")
            if dist != "normal":
                var_args.append(Arg("dist", dist))
            eq_args.append(Arg("variance", New(cls, tuple(var_args))))
            method_new = New("ML", ())
            notes.append("GARCH by maximum likelihood: EViews backcasts the first conditional variance and uses Marquardt "
                         "steps (verify the presample variance), so estimates may differ in the last digits")
            if s.explicit:
                raise _left("arch with an expression specification is not translated")
        elif method in ("binary", "logit", "probit"):
            classic = False
            d = (named.get("d") or {"logit": "l", "probit": "n"}.get(method, "n")).lower()
            take("d")
            if d == "x":
                raise _left("binary(d=x), the extreme-value (complementary log-log) link, has no pymetrix family: a gap "
                            "(docs/known-gaps.md); Logit() and Probit() are translated")
            if d not in ("l", "n"):
                raise _left(f"binary(d={d}) is not translated: d=l (logit) and d=n (probit) are")
            eq_args.append(Arg("family", New("Logit" if d == "l" else "Probit")))
            method_new = New("ML", ())
            if "d" not in named and method == "binary":
                notes.append("binary without d=: probit, EViews's default (verify)")
        elif method == "ordered":
            classic = False
            d = named.get("d", "n").lower()
            take("d")
            if d not in ("l", "n"):
                raise _left(f"ordered(d={d}) is not translated: d=l (OrderedLogit) and d=n (OrderedProbit) are; the "
                            "extreme-value link is a gap")
            if s.intercept:
                notes.append("the constant is dropped: the cut points stand for it, as in EViews")
                s.coefs = [c for c in s.coefs if c != "const"]
            eq_args.append(Arg("family", New("OrderedLogit" if d == "l" else "OrderedProbit")))
            method_new = New("ML", ())
        elif method == "censored":
            classic = False
            d = named.get("d", "n").lower()
            take("d", "l", "r")
            if d != "n":
                raise _left(f"censored(d={d}) is not translated: normal errors (Tobit) are; logistic and extreme-value errors "
                            "are a gap")
            if any(f in flags for f in ("i", "t")):
                raise _left("censored with limits given as series (i) or truncation (t) is not translated: Tobit(left=, right=) "
                            "takes numbers")
            try:
                left = float(named["l"]) if named.get("l", "") != "" else 0.0
                right = float(named["r"]) if named.get("r", "") != "" else None
            except ValueError:
                raise _left("censoring limits given as series are not translated: Tobit(left=, right=) takes numbers") from None
            tob = [Arg("left", left)] + ([Arg("right", right)] if right is not None else [])
            if "l" not in named:
                notes.append("censored at zero from below, EViews's default (verify)")
            eq_args.append(Arg("family", New("Tobit", tuple(tob))))
            method_new = New("ML", ())
        elif method == "count":
            classic = False
            d = named.get("d", "p").lower()
            take("d")
            if d not in ("p", "n"):
                raise _left(f"count(d={d}) is not translated: d=p (Poisson) and d=n (negative binomial) are")
            eq_args.append(Arg("family", New("Poisson" if d == "p" else "NegativeBinomial")))
            method_new = New("ML", ())
            if d == "n":
                notes.append("the negative binomial: pymetrix's NB2 with its dispersion estimated; check EViews's v= option "
                             "(verify)")
        elif method == "qreg":
            classic = False
            take("quant")
            try:
                q = float(named.get("quant", "0.5"))
            except ValueError:
                raise _left(f"qreg(quant={named.get('quant')}) is not a number") from None
            method_new = New("Quantile", (Arg("q", q),) if q != 0.5 else ())
            notes.append("quantile regression: the estimates are the same; the standard errors use pymetrix's Hall-Sheather "
                         "sandwich, and EViews's default covariance and bandwidth may differ (verify)")
        elif method == "heckit":
            classic = False
            assert selection is not None
            words = _split_top(selection, sel_at)
            if len(words) < 2:
                raise _error("heckit's selection equation is: s c z1 z2 (after @)")
            sel = self._spec(selection, sel_at, page, what="selection equation")
            take("2step", "ml")
            args = [Arg("selection", sel.formula)]
            if "2step" in flags:
                args.append(Arg("estimator", "twostep"))
            else:
                args.append(Arg("estimator", "ml"))
                notes.append("maximum likelihood, EViews's default for heckit (verify; 2step gives Heckman's two steps)")
            method_new = New("Heckman", tuple(args))
        unused = [k for k in named if k not in used] + [f for f in flags if f not in used]
        if unused:
            raise _left(f"{method}'s option{'s' if len(unused) > 1 else ''} {', '.join(unused)} {'are' if len(unused) > 1 else 'is'} "
                        "not translated: the estimate would differ without saying so")
        return self._fit_step(st, name, page, s, eq_args, method_new, method, classic, spec, notes)

    def _cov(self, named: dict[str, str], flags: list[str], notes: list[str], take: Any) -> str:
        from .data import _left

        cov = "classic"
        if "cov" in named:
            take("cov")
            value = named["cov"].lower()
            cov = {"white": "white", "hac": "hac", "ordinary": "classic", "default": "classic"}.get(value, "")
            if not cov:
                raise _left(f"cov={value} is not translated: white, hac and ordinary are")
        if "h" in flags:
            take("h")
            cov = "white"
            notes.append("h, an older letter for White's standard errors (cov=white from EViews 8): verify for your version")
        if "n" in flags:
            take("n")
            cov = "hac"
            notes.append("n, an older letter for Newey-West standard errors (cov=hac from EViews 8): verify for your version")
        if cov == "white":
            notes.append("White's standard errors: HC1, as EViews's (docs/design/definitions.md)")
        if cov == "hac":
            notes.append(_NOTE_HAC)
        return cov

    def _fit_name(self, name: str | None) -> str:
        if name is None:
            var, _ = self._name("untitled", "equation untitled")
            return var
        var, _ = self._name(name, f"equation {name.lower()}")
        return var

    def _fit_step(self, st: Any, name: str | None, page: Any, s: _Spec, eq_args: list[Arg], method_new: New | None, method: str,
                  classic: bool, spec_text: str, notes: list[str]) -> tuple[list[Op], str]:
        var = self._fit_name(name)
        if name is not None:
            eq_args.append(Arg("name", name.lower()))
        equation = New("Equation", tuple(eq_args))
        args = [Arg("data", RefTo(page.var), "pos")]
        if method_new is not None:
            args.append(Arg("method", method_new))
        step = Step("pymetrix:equation.fit", equation, var, f"{equation.code()}.fit", tuple(args))
        fit = Fit(var, "equation", page, s.coefs, method, classic and method == "ls" and not (s.ar or s.ma), spec_text,
                  page.sample, s.dependent, s.vector)
        self.fits[(name or "untitled").lower()] = fit
        self.types[(name or "untitled").lower()] = "equation"
        self._fit_line[(name or "untitled").lower()] = self.line
        if s.vector == "c":
            self.last_fit = fit
        if name is None:
            notes.insert(0, f"an equation EViews calls untitled: kept as {var}")
        return [step], "; ".join(notes)

    def _ardl(self, st: Any, options: str | None, spec: str, at: int, name: str | None, page: Any) -> tuple[list[Op], str]:
        from .data import _error, _left

        named, flags = _options(options)
        parts = re.split(r"(?:^|\s)@(?:\s|$)", spec)
        if len(parts) > 1:
            raise _left("ardl with fixed regressors after @ is not translated: ARDL(...) takes the distributed-lag regressors")
        words = _split_top(spec, at)
        if len(words) < 2:
            raise _error("ardl takes the dependent variable and the regressors: ardl y x1 x2")
        terms, intercept, _, _ = self._list_terms(words, page, False, "ARDL")
        if intercept:
            raise _left("ardl with c in the list is not translated: the constant is trend= (trend=const)")
        dep, regs = terms[0], terms[1:]
        if not regs:
            raise _error("ardl needs at least one regressor")
        trend = {"none": "n", "const": "c", "linear": "ct"}.get(named.pop("trend", "const").lower())
        if trend is None:
            raise _left("ardl(trend=...) is not translated: none, const and linear are")
        try:
            deplags = int(named.pop("deplags", "4"))
            reglags = int(named.pop("reglags", "4"))
        except ValueError:
            raise _left("ardl's lags are not whole numbers") from None
        ic = named.pop("ic", "aic").lower()
        notes = []
        args = [Arg("formula", f"{dep} ~ {' + '.join(regs)}", "pos")]
        if "fixed" in flags:
            flags.remove("fixed")
            args += [Arg("lags", deplags), Arg("order", reglags)]
        else:
            crit = {"aic": "aic", "sic": "bic", "bic": "bic", "hq": "hqic", "hqc": "hqic"}.get(ic)
            if crit is None:
                raise _left(f"ardl(ic={ic}) is not translated: aic, sic and hq are")
            args += [Arg("select", crit), Arg("max_lags", deplags), Arg("max_order", reglags)]
            notes.append(f"the lags chosen by {ic.upper()} up to {deplags} and {reglags}, as EViews's default search (verify the "
                         "default maximum lags and whether EViews searches each regressor's lag separately)")
        if trend != "c":
            args.append(Arg("trend", trend))
        if named or flags:
            raise _left(f"ardl's options {', '.join([*named, *flags])} are not translated")
        var = self._fit_name(name)
        ardl = New("ARDL", tuple(args))
        step = Step("pymetrix:ardl.fit", ardl, var, f"{ardl.code()}.fit", (Arg("data", RefTo(page.var), "pos"),))
        self.fits[(name or "untitled").lower()] = Fit(var, "ardl", page, [], "ardl", False, spec, page.sample, _term_name(dep))
        self.types[(name or "untitled").lower()] = "equation"
        return [step], "; ".join(notes)

    # -- VARs and VECs ----------------------------------------------------------------------------------------------

    def _cmd_var(self, st: Any) -> tuple[list[Op], str]:
        from .data import _error
        from .parse import _options as bracket

        m = re.match(r"\s*([A-Za-z_]\w*)(?:\.(\w+))?", st.rest)
        if m is None:
            raise _error("var names the VAR: var v.ls 1 4 y1 y2 @ c")
        name, method = m.group(1), (m.group(2) or "").lower()
        options, after = bracket(st.rest, m.end())
        self.types[name.lower()] = "var"
        if not method:
            return [], "the VAR is declared; a later line estimates it (no step)"
        rest = st.rest[after:]
        rest_at = st.rest_at + after + (len(rest) - len(rest.lstrip()))
        if method == "ec":
            return self._vec(name, options, rest.strip(), rest_at)
        return self._var(name, method, options, rest.strip(), rest_at)

    def _var(self, name: str, method: str, options: str | None, spec: str, at: int) -> tuple[list[Op], str]:
        from .data import _error, _left

        page = self._page()
        named, flags = _options(options)
        endo_text, _, exo_text = spec.partition("@")
        words = _split_top(endo_text, at)
        lags: list[int] = []
        while words and re.fullmatch(r"\d+", words[0][0]):
            lags.append(int(words.pop(0)[0]))
        if len(lags) % 2 or not lags:
            raise _error("a VAR's lags come in pairs before the variables: v.ls 1 4 y1 y2")
        pairs = list(zip(lags[::2], lags[1::2]))
        if pairs != [(1, pairs[0][1])]:
            raise _left("lags other than one range from 1 (1 4) are not translated: VAR(..., lags=n) takes lags 1 to n")
        n = pairs[0][1]
        endog, has_c, _, _ = self._list_terms(words, page, False, "VAR")
        if has_c:
            raise _error("c goes after @ in a VAR")
        if not all(re.fullmatch(r"[A-Za-z_]\w*", v) for v in endog):
            raise _left("a VAR's endogenous variables must be series: make the transformed series first")
        exo: list[str] = []
        trend = "n"
        notes: list[str] = []
        if exo_text.strip():
            exo_words = _split_top(exo_text, at + len(endo_text) + 1)
            for w, _ in exo_words:
                if w.lower() in ("@trend",):
                    if trend != "c":
                        raise _left("@trend without c in a VAR is not translated")
                    trend = "ct"
                    continue
            exo_words = [(w, a) for w, a in exo_words if w.lower() != "@trend"]
            terms, has_c, _, _ = self._list_terms(exo_words, page, False, "VAR")
            if has_c:
                trend = "ct" if trend == "ct" else "c"
            exo = [t for t in terms if t != "const"]
        else:
            notes.append("no c after @: the VAR has no constant, as EViews writes it (verify)")
        if named or flags:
            raise _left(f"the VAR's options {', '.join([*named, *flags])} are not translated")
        var, _ = self._name(name, f"var {name.lower()}")
        if method == "ls":
            args = [Arg("variables", tuple(endog), "pos"), Arg("lags", n)]
            if exo:
                args.append(Arg("exog", tuple(exo)))
            if trend != "c":
                args.append(Arg("trend", trend))
            model = New("VAR", tuple(args))
            step = Step("pymetrix:var.fit", model, var, f"{model.code()}.fit", (Arg("data", RefTo(page.var), "pos"),))
            self.fits[name.lower()] = Fit(var, "var", page, [], "var", False, spec, page.sample)
            return [step], "; ".join(notes)
        raise _left(f"{name}.{method} is not translated: ls and ec are")

    def _vec(self, name: str, options: str | None, spec: str, at: int) -> tuple[list[Op], str]:
        from .data import _error, _left

        page = self._page()
        parts = _commas(options)
        case = (parts[0].lower() if parts else "c")
        try:
            rank = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            raise _left(f"ec({options}): the number of cointegrating relations is not a number") from None
        if len(parts) > 2:
            raise _left(f"ec's options {', '.join(parts[2:])} are not translated")
        det = {"a": "n", "b": "ci", "c": "co", "d": "coli"}.get(case)
        if det is None:
            raise _left(f"ec({case}): EViews's case {case} is not translated: a, b, c and d are (VECM(deterministic=...))")
        endo_text, _, exo_text = spec.partition("@")
        if exo_text.strip():
            raise _left("exogenous variables in a VEC are not translated: VECM(..., exog=[...])")
        words = _split_top(endo_text, at)
        lags: list[int] = []
        while words and re.fullmatch(r"\d+", words[0][0]):
            lags.append(int(words.pop(0)[0]))
        if len(lags) != 2 or lags[0] != 1:
            raise _error("a VEC's lags are one range from 1 before the variables: v.ec(c, 1) 1 4 y1 y2")
        endog, has_c, _, _ = self._list_terms(words, page, False, "VEC")
        if has_c or not all(re.fullmatch(r"[A-Za-z_]\w*", v) for v in endog):
            raise _left("a VEC's variables must be series")
        var, _ = self._name(name, f"var {name.lower()}")
        args = [Arg("variables", tuple(endog), "pos"), Arg("lags", lags[1]), Arg("rank", rank)]
        if det != "co":
            args.append(Arg("deterministic", det))
        model = New("VECM", tuple(args))
        step = Step("pymetrix:vecm.fit", model, var, f"{model.code()}.fit", (Arg("data", RefTo(page.var), "pos"),))
        self.fits[name.lower()] = Fit(var, "vecm", page, [], "ec", False, spec, page.sample)
        note = (f"EViews's lags 1 {lags[1]} of the VEC are in first differences: VECM(lags={lags[1]}); case {case} is "
                f"deterministic={det!r} (verify the cases against EViews's table of the five)")
        return [step], note

    # -- systems ----------------------------------------------------------------------------------------------------

    def _cmd_system(self, st: Any) -> tuple[list[Op], str]:
        from .data import _error

        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*", st.rest)
        if m is None:
            raise _error("system names the system: system s")
        self.systems[m.group(1).lower()] = {"lines": [], "instruments": [], "name": m.group(1)}
        self.types[m.group(1).lower()] = "system"
        return [], "the system's equations are collected; a later line estimates it (no step)"

    def _system_method(self, st: Any, key: str) -> tuple[list[Op], str]:
        from .data import _error, _left

        sysinfo = self.systems[key]
        page = self._page()
        if st.word == "append":
            text = st.rest.strip()
            if text.lower().startswith("@inst"):
                words = _split_top(text[5:], st.rest_at + 5)
                ins, has_c, _, _ = self._list_terms(words, page, False, "instrument list")
                sysinfo["instruments"] += [t for t in ins if t != "const"]
                return [], "instruments of the system (no step)"
            if text.startswith("@"):
                raise _left(f"{text.split()[0]} in a system is not translated: equations and @inst are")
            s = self._spec(text, st.rest_at, page)
            if not s.explicit:
                raise _left("a system's equations are written with their coefficients: y = c(1) + c(2)*x")
            sysinfo["lines"].append(s)
            return [], "an equation of the system (no step)"
        methods = {"sur": "SUR", "3sls": "ThreeSLS", "fiml": "FIML", "tsls": "TwoSLS"}
        if st.word in methods:
            if not sysinfo["lines"]:
                raise _error(f"the system {sysinfo['name']} has no equations")
            if st.word in ("3sls", "tsls") and not sysinfo["instruments"]:
                raise _error(f"{st.word} needs instruments: {sysinfo['name']}.append @inst z1 z2")
            if st.options:
                raise _left(f"{st.word}({st.options}): the system's estimation options are not translated")
            eqs = tuple(New("Equation", (Arg("text", s.formula, "pos"), Arg("params", tuple(s.params)))) for s in sysinfo["lines"])
            sargs = [Arg("equations", eqs, "pos")]
            if sysinfo["instruments"]:
                sargs.append(Arg("instruments", tuple(dict.fromkeys(sysinfo["instruments"]))))
            system = New("System", tuple(sargs))
            var, _ = self._name(sysinfo["name"], f"system {key}")
            step = Step("pymetrix:system.fit", system, var, f"{system.code()}.fit",
                        (Arg("data", RefTo(page.var), "pos"), Arg("method", New(methods[st.word]))))
            coefs = sorted({c for s in sysinfo["lines"] for c in s.params}, key=lambda c: int(c.rpartition("_")[2]))
            fit = Fit(var, "system", page, [f"c_{i}" for i in range(1, max(int(c.rpartition('_')[2]) for c in coefs) + 1)],
                      st.word, False, "", page.sample)
            self.fits[key] = fit
            self.last_fit = fit
            return [step], ("the coefficients c(i) shared across the equations are the parameters c_i; EViews's default "
                            "iteration of the weights may differ (verify: the system's estimation options)")
        if st.word == "ls":
            raise _left(f"{sysinfo['name']}.ls (each equation by least squares) is not translated: fit each equation, "
                        "Equation(...).fit(data)")
        raise _left(f"{sysinfo['name']}.{st.word} is not translated: append, sur, 3sls, fiml and tsls are")

    # -- methods on fitted objects ----------------------------------------------------------------------------------

    def _fit_method(self, st: Any, key: str) -> tuple[list[Op], str]:
        from .data import _left

        word = st.word
        if word in ESTIMATORS or word in LEFT_ESTIMATORS:
            return self._estimate(st, word, st.options, st.rest, st.rest_at, st.obj)
        fit = self.fits.get(key)
        kind = self.types.get(key)
        if kind == "var":
            if word == "ls":
                return self._var(st.obj, "ls", st.options, st.rest, st.rest_at)
            if word == "ec":
                return self._vec(st.obj, st.options, st.rest, st.rest_at)
        if fit is None:
            raise _left(f"{st.obj} is not estimated by a line translated so far (a loop, a subroutine or a line left out may "
                        "estimate it)" if self.lost else f"{st.obj} is not estimated before this line")
        if fit.kind != "equation":
            if word in ("forecast", "fit") and fit.kind in ("var", "vecm"):
                raise _left(f"{st.obj}.{word}: a VAR's forecast in EViews goes through a model (makemodel); in pymetrix: "
                            f"{fit.var}.forecast(data, span)")
            raise _left(f"{st.obj}.{word} is not translated for a {fit.kind}")
        handler = getattr(self, f"_eq_{word}", None)
        if handler is None:
            raise _left(f"{st.obj}.{word} is not translated" + (": testadd (omitted variables) has no pymetrix equivalent, a gap "
                                                             "(docs/known-gaps.md); add the variables and compare the fits, or "
                                                             "eq.test('wald', ...) on the larger equation" if word == "testadd" else ""))
        if fit.kind == "equation" and fit.method == "ardl" and word not in ("forecast", "fit", "makeresids"):
            raise _left(f"{st.obj}.{word} on an ARDL is not translated")
        return handler(st, fit)

    def _test_step(self, fit: Fit, test: str, what: str, opts: dict[str, Any], args: tuple[Any, ...] = ()) -> Step:
        out, _ = self._name(f"{fit.var}_{what}", f"result {fit.var}_{what}")
        step_args = [Arg("name", test, "pos")]
        if args:
            step_args.append(Arg("args", args, "star"))
        if opts:
            step_args.append(Arg("opts", opts, "varkw"))
        return Step("pymetrix:fitted_equation.test", fit.var, out, f"{fit.var}.test", tuple(step_args))

    def _restriction(self, text: str, fit: Fit) -> str:
        from .data import _error, _left

        def coef(m: re.Match[str]) -> str:
            i = int(m.group(2))
            if m.group(1).lower() != fit.vector:
                raise _left(f"{m.group(0)}: only the equation's own coefficients ({fit.vector}) are translated")
            if not 1 <= i <= len(fit.coefs):
                raise _error(f"{m.group(0)}: the equation has {len(fit.coefs)} coefficients")
            return _in_restriction(fit.coefs[i - 1])

        out = re.sub(r"(?<![\w.])([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)", coef, text)
        return " ".join(out.split())

    def _eq_wald(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _error

        if not st.rest.strip():
            raise _error("wald needs restrictions: eq.wald c(2)=0, c(3)=0")
        restriction = self._restriction(st.rest, fit)
        return [self._test_step(fit, "wald", "wald", {}, (restriction,))], \
            f"EViews's c(i) as pymetrix's coefficients: {restriction}; the statistic is chi-squared, the F form in its details"

    def _eq_testdrop(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _error, _left

        names = [w for w, _ in _split_top(st.rest, st.rest_at)]
        if not names:
            raise _error("testdrop names the regressors: eq.testdrop x z")
        page = fit.page
        terms = []
        for w in names:
            t = "const" if w.lower() == "c" else _term_name(str(self.expression(w, st.rest_at, page)))
            if t not in fit.coefs:
                raise _left(f"{w} is not a regressor of {st.obj}")
            terms.append(t)
        restriction = ", ".join(f"{_in_restriction(t)} = 0" for t in terms)
        why = ("the Wald F statistic equals EViews's redundant-variables F under least squares with classic standard errors"
               if fit.classic else "EViews's redundant-variables test refits the equation; with these standard errors or "
               "this estimator the Wald statistic differs from it (verify)")
        return [self._test_step(fit, "wald", "testdrop", {}, (restriction,))], why

    def _eq_auto(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        lags = (st.options or st.rest).strip()
        if not re.fullmatch(r"\d+", lags):
            raise _left("auto without its order is not translated: give it (eq.auto(4))")
        return [self._test_step(fit, "breusch_godfrey", "auto", {"lags": int(lags)})], \
            "Breusch-Godfrey: pymetrix gives the LM statistic and the F form; EViews's presample residuals are set to zero, as here (verify)"

    def _eq_hettest(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        named, flags = _options(st.options)
        if st.rest.strip():
            raise _left("hettest with its own regressors is not translated: the equation's regressors are")
        kind = named.pop("type", "bpg").lower()
        test = {"bpg": "breusch_pagan", "white": "white"}.get(kind)
        if test is None:
            raise _left(f"hettest(type={kind}) is not translated: bpg and white are")
        if named or flags:
            raise _left(f"hettest's options {', '.join([*named, *flags])} are not translated")
        note = "type=bpg, EViews's default (verify)" if "type" not in (st.options or "") else ""
        if test == "white":
            note = "White's test with cross terms, as pymetrix's (verify EViews's c option for the cross terms)"
        return [self._test_step(fit, test, "hettest", {})], note

    def _eq_archtest(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        lags = (st.options or st.rest or "1").strip()
        if not re.fullmatch(r"\d+", lags):
            raise _left(f"archtest({lags}) is not translated: a number of lags is")
        return [self._test_step(fit, "arch", "archtest", {"lags": int(lags)})], ""

    def _eq_reset(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        n = (st.options or st.rest or "1").strip()
        if not re.fullmatch(r"\d+", n):
            raise _left(f"reset({n}) is not translated: a number of fitted terms is")
        return [self._test_step(fit, "reset", "reset", {"power": int(n) + 1})], \
            f"EViews's {n} fitted term{'s' if int(n) > 1 else ''} are powers 2 to {int(n) + 1}: power={int(n) + 1}"

    def _eq_chow(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        dates = st.rest.split()
        if len(dates) != 1 or st.options:
            raise _left("chow with several dates or options is not translated: one break date is")
        page = fit.page
        at = self._end(dates[0], page, True)
        if at is None:
            raise _left(f"{dates[0]} is not a date read here")
        return [self._test_step(fit, "chow", "chow", {"at": at})], "the breakpoint test (F), the second regime from the date"

    def _eq_hist(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        return [self._test_step(fit, "jarque_bera", "hist", {})], "the residuals' histogram view: its Jarque-Bera statistic"

    def _eq_rls(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from .data import _left

        view = (st.options or "q").strip().lower()
        if view not in ("q", "c", "r"):
            raise _left(f"rls({view}) is not translated: r, c and q (recursive residuals, CUSUM, CUSUM of squares) are")
        out, _ = self._name(f"{fit.var}_rls", f"result {fit.var}_rls")
        return [Code(f"{out} = {fit.var}.recursive_residuals()", (out,))], \
            "recursive residuals with the CUSUM and CUSUM-of-squares paths and their 5% bands (verify the view letters)"

    def _forecast(self, st: Any, fit: Fit, dynamic: bool) -> tuple[list[Op], str]:
        from pymetrix.commands import AddToDataset

        from .data import _error, _left

        named, flags = _options(st.options)
        if named or flags:
            raise _left(f"{st.word}'s options {', '.join([*named, *flags])} are not translated")
        words = st.rest.split()
        if not words or len(words) > 2:
            raise _error(f"{st.word} names the forecast series, and optionally its standard error: eq.{st.word} yf yf_se")
        page = self._page()
        span = self._span(page, st.word)
        out, _ = self._name(f"{fit.var}_{st.word}", f"result {fit.var}_{st.word}")
        fc = Step("pymetrix:fitted_equation.forecast", fit.var, out, f"{fit.var}.forecast",
                  (Arg("data", RefTo(page.var), "pos"), Arg("span", span, "pos"), Arg("dynamic", dynamic)))
        ops: list[Op] = [fc]
        for word, column in zip(words, ("forecast", "se")):
            series = self._series_name(word, page)
            ops.append(Cmd(AddToDataset(source=out, item=f"column:{column}", data=page.var, series_name=series, replace=True), (out,)))
        notes = [f"{'dynamic' if dynamic else 'static (one step ahead)'} forecast over the page's sample, {span[0]} to {span[1]}"]
        if fit.dependent and not re.fullmatch(r"[A-Za-z_]\w*", fit.dependent):
            notes.append(f"the left-hand side is {fit.dependent}: EViews forecasts its first series by default; check which "
                         "pymetrix's forecast column holds")
        if fit.method == "ardl":
            raise _left("an ARDL's forecast is not translated here")
        if any(m in fit.spec.lower() for m in ("ar(", "ma(")):
            notes.append("with ARMA errors (verify how EViews starts the MA terms)")
        return ops, "; ".join(notes)

    def _eq_forecast(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        return self._forecast(st, fit, True)

    def _eq_fit(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        return self._forecast(st, fit, False)

    def _eq_makeresids(self, st: Any, fit: Fit) -> tuple[list[Op], str]:
        from pymetrix.commands import AddToDataset

        from .data import _error, _left

        if st.options:
            raise _left(f"makeresids({st.options}): standardized and generalized residuals are not translated")
        words = st.rest.split()
        if len(words) != 1:
            raise _error("makeresids names the series: eq.makeresids r")
        page = fit.page
        series = self._series_name(words[0], page)
        return [Cmd(AddToDataset(source=fit.var, item="resid", data=page.var, series_name=series, replace=True), (fit.var,))], ""

    def _series_name(self, word: str, page: Any) -> str:
        from ..bundle import _py_name
        from .data import _left

        if not re.fullmatch(r"[A-Za-z_]\w*", word):
            raise _left(f"{word}: a series name is needed")
        series = _py_name(word)
        if page.series is not None:
            page.series.add(series)
        return series

    def _span(self, page: Any, what: str) -> tuple[str, str]:
        from .data import _left

        sample = page.sample
        if isinstance(sample, Var) or (isinstance(sample, SampleSpec) and sample.where is not None):
            raise _left(f"{what} over a named sample or a sample with a condition is not translated: one with two dates is")
        start = sample.start if isinstance(sample, SampleSpec) and sample.start else page.first
        end = sample.end if isinstance(sample, SampleSpec) and sample.end else page.last
        if start is None or end is None:
            raise _left(f"{what}: the page's first or last period is not known here: set the sample with two dates first")
        return (start, end)

    # -- tests on series and groups ---------------------------------------------------------------------------------

    def _uroot(self, st: Any) -> tuple[list[Op], str]:
        from ..bundle import _py_name
        from .data import _error, _left

        page = self._page()
        series = _py_name(st.obj)
        if page.series is not None and series not in page.series:
            raise _error(f"{st.obj} is not a series of page {page.eviews}")
        named, flags = _options(st.options)
        tests = {"adf": "adf", "pp": "phillips_perron", "kpss": "kpss", "dfgls": "dfgls", "ers": "ers_point_optimal",
                 "np": "ng_perron"}
        chosen = [f for f in flags if f in tests]
        if len(chosen) > 1:
            raise _error("uroot takes one test: adf, pp, kpss, dfgls, ers or np")
        test = tests[chosen[0]] if chosen else "adf"
        opts: dict[str, Any] = {}
        trend = [f for f in flags if f in ("const", "trend", "none")]
        if trend:
            opts["trend"] = {"const": "c", "trend": "ct", "none": "n"}[trend[0]]
            if opts["trend"] == "n" and test not in ("adf", "phillips_perron"):
                raise _left(f"{chosen[0] if chosen else 'adf'} without a constant is not translated: pymetrix's has c or ct")
        if "dif" in named or any(f.startswith("dif") for f in flags):
            raise _left("uroot on differences (dif=) is not translated: make the differenced series first, then test it")
        notes = []
        if test in ("adf", "dfgls"):
            if "lag" in named:
                opts["lags"] = _value(named.pop("lag"), "lag")
            else:
                ic = named.pop("info", "sic").lower()
                crit = {"aic": "aic", "sic": "bic", "bic": "bic", "tstat": "t-stat"}.get(ic)
                if crit is None:
                    raise _left(f"uroot(info={ic}) is not translated: aic, sic and tstat are")
                if test == "adf":
                    opts["lags"] = crit
                else:
                    opts["method"] = crit
                notes.append(f"the lags chosen by {ic.upper()}, EViews's default being SIC (verify its default maximum lag)")
                if "maxlag" in named:
                    opts["max_lags"] = _value(named.pop("maxlag"), "maxlag")
        elif test in ("ers_point_optimal", "ng_perron"):
            notes.append("the long-run variance by the modified AIC autoregression (verify EViews's spectral estimator, hac=)")
        else:
            notes.append("the long-run variance's bandwidth: EViews's default Newey-West bandwidth may differ (verify band=)")
        used = {"lag", "info", "maxlag"}
        rest = [k for k in named if k not in used] + [f for f in flags if f not in tests and f not in ("const", "trend", "none")]
        if rest:
            raise _left(f"uroot's options {', '.join(rest)} are not translated")
        out, _ = self._name(f"{series}_uroot", f"result {series}_uroot")
        step = Step("pymetrix:dataset.test", page.var, out, f"{page.var}.test",
                    (Arg("name", test, "pos"), Arg("args", (series,), "star"), *( [Arg("opts", opts, "varkw")] if opts else [])))
        return [step], "; ".join(notes)

    def _coint(self, st: Any, group: Any) -> tuple[list[Op], str]:
        from .data import _left

        page = self._page()
        parts = _commas(st.options)
        named = [p for p in parts if "=" in p]
        if named:
            raise _left(f"coint's options {', '.join(named)} are not translated: the Johansen test with a case letter and lags is")
        case = parts[0].lower() if parts and not parts[0].isdigit() else "c"
        lags_text = [p for p in parts if p.isdigit()]
        words = st.rest.split()
        if len(words) == 2 and words[0] == "1" and words[1].isdigit():
            lags = int(words[1])
        elif not words and lags_text:
            lags = int(lags_text[-1])
        elif not words and not lags_text:
            lags = 1
        else:
            raise _left("coint's lags other than one range from 1 (1 4) are not translated")
        det = {"a": -1, "c": 0}.get(case)
        if det is None:
            raise _left(f"coint({case}): EViews's case {case} is not translated: a (no deterministic terms) and c (a constant "
                        "outside the relations) are, as data.johansen's det_order -1 and 0")
        out, _ = self._name(f"{group.name}_coint", f"result {group.name}_coint")
        step = Step("pymetrix:dataset.johansen", page.var, out, f"{page.var}.johansen",
                    (Arg("variables", tuple(group.members), "pos"), Arg("lags", lags), *([Arg("det_order", det)] if det else [])))
        return [step], (f"Johansen's trace and maximum-eigenvalue tests, {lags} lagged differences (EViews's lags 1 {lags} "
                        "are in differences: verify); case " + case)

    # -- values read from results -------------------------------------------------------------------------------------

    _MEMBERS = {"@r2": "r2", "@rbar2": "r2_adj", "@se": "se_regression", "@ssr": "ssr", "@dw": "durbin_watson",
                "@regobs": "nobs", "@logl": "loglike"}
    _CRITERIA = {"@aic": "aic", "@schwarz": "bic", "@sc": "bic", "@hq": "hq"}

    def member_code(self, obj: str, member: str, index: int | None) -> tuple[str, str]:
        """Python for an equation's data member (``eq.@coefs(2)``) and a note, or left out."""
        from .data import _error, _left

        fit = self.fits.get(obj.lower())
        if fit is None:
            raise _left(f"{obj} is not estimated by a line translated so far" if self.lost or obj.lower() in self.types
                        else f"{obj} is not an object of the program")
        if fit.kind not in ("equation", "system") or fit.method == "ardl":
            raise _left(f"{obj}.{member} is translated for equations")
        v = fit.var
        if member in ("@coefs", "@stderrs", "@tstats"):
            if index is None:
                raise _left(f"{member} without an index (the whole vector) is not translated: {v}.params")
            if not 1 <= index <= len(fit.coefs):
                raise _error(f"{obj} has {len(fit.coefs)} coefficients, not {index}")
            name = quote(fit.coefs[index - 1])
            coef = f"{v}.params[{name}]"
            se = f"{v}.params.se[{v}.params.names.index({name})]"
            return {"@coefs": coef, "@stderrs": se, "@tstats": f"{coef} / {se}"}[member], ""
        if member in self._MEMBERS:
            return f"{v}.stats[{quote(self._MEMBERS[member])}]", ""
        if member == "@ncoef":
            return f"len({v}.params)", ""
        if member in self._CRITERIA:
            return (f"{v}.stats[{quote(self._CRITERIA[member])}] / {v}.stats[\"nobs\"]",
                    f"{member} per observation, as EViews reports it: pymetrix's criteria are totals (docs/design/definitions.md)")
        raise _left(f"{obj}.{member} is not translated: @coefs, @stderrs, @tstats, @r2, @rbar2, @se, @ssr, @dw, @regobs, @logl, "
                    "@ncoef, @aic, @schwarz and @hq are")

    def members_in(self, text: str) -> tuple[str, dict[str, str], list[str]]:
        """``text`` with each result's value (``eq.@coefs(2)``, ``eq.@r2``, ``c(2)`` after an estimation) as a placeholder
        name, the Python for each placeholder, and the notes."""
        from .data import _left

        codes: dict[str, str] = {}
        notes: list[str] = []

        def member(m: re.Match[str]) -> str:
            code, note = self.member_code(m.group(1), m.group(2).lower(), int(m.group(3)) if m.group(3) else None)
            key = f"pmxv{len(codes)}"
            codes[key] = code
            if note and note not in notes:
                notes.append(note)
            return key

        out = re.sub(r"(?<![\w.])([A-Za-z_]\w*)\.(@\w+)(?:\(\s*(\d+)\s*\))?", member, text)

        def coefficient(m: re.Match[str]) -> str:
            if m.group(1).lower() != "c":
                return m.group(0)
            fit = self.last_fit
            if fit is None:
                raise _left(f"{m.group(0)}: no estimation translated so far fills c" + (" (a line left out may)" if self.lost else ""))
            i = int(m.group(2))
            if not 1 <= i <= len(fit.coefs):
                raise _left(f"{m.group(0)}: the last estimation ({fit.var}) has {len(fit.coefs)} coefficients")
            key = f"pmxv{len(codes)}"
            codes[key] = f"{fit.var}.params[{quote(fit.coefs[i - 1])}]"
            note = f"c holds the last estimation's coefficients, {fit.var}'s, as in EViews"
            if note not in notes:
                notes.append(note)
            return key

        if codes or re.search(r"(?<![\w.])c\s*\(\s*\d+\s*\)", out, re.I):
            out = re.sub(r"(?<![\w.@])([A-Za-z_]\w*)\s*\(\s*(\d+)\s*\)", coefficient, out)
        return out, codes, notes


def _order(lags: list[int]) -> Any:
    """ARMA lags as ARMA's argument: the order when they are 1 to n, else the lags."""
    lags = sorted(set(lags))
    return lags[-1] if lags == list(range(1, lags[-1] + 1)) else tuple(lags)
