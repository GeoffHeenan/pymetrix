"""Equation: one specification class for estimation and for models.

``~`` marks a stochastic equation, ``=`` an identity. An equation has one of
three forms:

* linear form, no parameters declared: ``"wage ~ educ + exper + fe(region)"``.
  Coefficients are created per term and named after it; ``const`` is the
  intercept unless ``- 1`` is written.
* explicit form, parameters declared: ``Equation("dlog(c) ~ b0 + b1*dlog(yd)",
  params=["b0", "b1"])``. Parameters are never inferred.
* identity: ``"y = c + i + g"``; no parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .._api import api
from ..expr import BinOp, Call, Coef, Expr, ExpressionError, Num, parse_equation
from ..expr.nodes import add, mul
from ..metadata import Param
from ..results import deprecated_alias
from .families import Family, Gaussian

if TYPE_CHECKING:
    from ..data import Dataset
    from .fitted import FittedEquation


@dataclass(frozen=True)
class Term:
    label: str
    expr: Expr


def _split(node: Expr, sign: int = 1) -> list[tuple[int, Expr]]:
    if isinstance(node, BinOp) and node.op in "+-" and not node.grouped:
        right_sign = sign if node.op == "+" else -sign
        return _split(node.left, sign) + _split(node.right, right_sign)
    return [(sign, node)]


@dataclass(frozen=True, eq=False)
class Equation:
    """An equation written as text: ``"c ~ y + c[-1]"`` to estimate, or ``"y = c + i + g"`` (an identity).

    A ``~`` equation is linear in its coefficients unless ``params`` names them, as in
    ``Equation("x ~ rho*x[-1]", params=["rho"])``. Lags are ``x[-1]``, and functions such as ``log``, ``dlog``
    and ``d`` may appear on either side. ``errors`` (``ARMA(...)``) and ``variance`` (``GARCH(...)``) add an error
    process. Estimate it with ``fit``."""

    code_positional: ClassVar[int] = 1  # generated code gives the first field by position: Equation(...)

    text: str
    params: Sequence[str | Param] = ()
    family: Family = field(default_factory=Gaussian)
    priors: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    errors: Any = None  # ARMA(...): the error process
    variance: Any = None  # GARCH(...): the conditional variance

    lhs: Expr = field(init=False, repr=False)
    rhs: Expr = field(init=False, repr=False)
    form: Literal["linear", "explicit", "identity"] = field(init=False, repr=False)
    param_specs: tuple[Param, ...] = field(init=False, repr=False)
    terms: tuple[Term, ...] = field(init=False, repr=False)
    intercept: bool = field(init=False, repr=False)
    fixed_effects: tuple[str, ...] = field(init=False, repr=False)
    categoricals: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(self.params))  # a list or a tuple: the same equation, recorded the same way
        specs = tuple(p if isinstance(p, Param) else Param(p) for p in self.params)
        names = [p.name for p in specs]
        if len(set(names)) != len(names):
            raise ExpressionError(f"duplicate parameter names in {names}")
        lhs, op, rhs = parse_equation(self.text, names)
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        set_("lhs", lhs)
        set_("rhs", rhs)
        set_("param_specs", specs)
        set_("terms", ())
        set_("intercept", False)
        set_("fixed_effects", ())
        set_("categoricals", ())
        has_markers = any(side.calls(m) for side in (lhs, rhs) for m in ("fe", "C", "seasonal"))
        if lhs.params():
            raise ExpressionError("parameters belong on the right-hand side")
        if op == "=":
            if names:
                raise ExpressionError(f"identity {self.text!r} cannot have parameters")
            if has_markers:
                raise ExpressionError("fe() and C() cannot appear in an identity")
            set_("form", "identity")
        elif names:
            unused = set(names) - rhs.params()
            if unused:
                raise ExpressionError(f"declared parameters not used in {self.text!r}: {sorted(unused)}")
            if has_markers:
                raise ExpressionError("fe() and C() are only allowed in linear-form equations (no params=)")
            set_("form", "explicit")
        else:
            set_("form", "linear")
            self._parse_linear(rhs)
        unknown_priors = set(self.priors) - set(names) if names else set()
        if unknown_priors:
            raise ExpressionError(f"priors given for undeclared parameters: {sorted(unknown_priors)}")
        if self.has_error_model:
            if self.form != "linear":
                raise ExpressionError("ARMA errors and GARCH variance need a stochastic linear-form equation (explicit form is planned)")
            if self.fixed_effects or self.categoricals or not isinstance(self.family, Gaussian):
                raise ExpressionError("ARMA errors and GARCH variance need a Gaussian equation without fe() or C() terms")

    @property
    def has_error_model(self) -> bool:
        return self.errors is not None or self.variance is not None

    def _parse_linear(self, rhs: Expr) -> None:
        intercept = True
        terms: list[Term] = []
        fes: list[str] = []
        cats: list[str] = []
        for sign, e in _split(rhs):
            if isinstance(e, Num) and e.value in (0.0, 1.0):
                intercept = sign > 0 and e.value == 1.0
                continue
            if sign < 0:
                raise ExpressionError(
                    f"'- {e}' in a linear-form equation: only '- 1' (no intercept) may be subtracted; "
                    "write the term inside parentheses or use the explicit form"
                )
            if isinstance(e, Call) and e.fn == "fe":
                fes.append(e.args[0].id)  # type: ignore[attr-defined]
            elif isinstance(e, Call) and e.fn == "C":
                cats.append(e.args[0].id)  # type: ignore[attr-defined]
            elif isinstance(e, Call) and e.fn == "seasonal":
                cats.append("seasonal()")  # dummies of the period of the year, from the time axis
            else:
                if e.calls("fe") or e.calls("C") or e.calls("seasonal"):
                    raise ExpressionError("fe(), C() and seasonal() must be terms on their own")
                terms.append(Term(str(e), e))
        labels = [t.label for t in terms]
        if len(set(labels)) != len(labels):
            raise ExpressionError(f"repeated terms in {self.text!r}")
        object.__setattr__(self, "terms", tuple(terms))
        object.__setattr__(self, "intercept", intercept and not fes)
        object.__setattr__(self, "fixed_effects", tuple(fes))
        object.__setattr__(self, "categoricals", tuple(cats))

    # properties

    @property
    def is_identity(self) -> bool:
        return self.form == "identity"

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.param_specs)

    @property
    def dependent(self) -> str | None:
        current = {n.id for n in self.lhs.refs() if n.lag == 0}
        return next(iter(current)) if len(current) == 1 else None

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    def require_dependent(self) -> str:
        """The dependent variable, or an error when the left-hand side has none or several."""
        dependent = self.dependent
        if dependent is None:
            raise ExpressionError(
                f"the left-hand side of {self.text!r} must contain exactly one current-period variable"
            )
        return dependent

    def require_target(self) -> str:
        """Deprecated: the old name of ``require_dependent``."""
        import warnings

        warnings.warn("Equation.require_target() is deprecated since pymetrix 0.1.0a9; use .require_dependent()",
                      DeprecationWarning, stacklevel=2)
        return self.require_dependent()

    @property
    def label(self) -> str:
        return self.name or self.dependent or self.text

    def variables(self) -> frozenset[str]:
        """Series the equation uses (fe() and C() arguments are grouping names, not series)."""
        if self.form == "linear":
            return self.lhs.variables().union(*(t.expr.variables() for t in self.terms))
        return self.lhs.variables() | self.rhs.variables()

    def check_names(self, data: Dataset) -> None:
        """Every name must be a series in the data or a declared parameter, never both."""
        series = set(data.names)
        clashes = set(self.param_names) & series
        if clashes:
            raise ExpressionError(
                f"parameter(s) {sorted(clashes)} have the same name as series in the data; rename them"
            )
        unknown = sorted(self.variables() - series)
        if unknown:
            raise ExpressionError(
                f"unknown name(s) {unknown} in {self.text!r}: not series in the data"
                + ("" if self.form == "explicit" else " (declare parameters with params=[...])")
            )
        for level in self.fixed_effects + self.categoricals:
            if level not in series and level not in data.frame.index.names:
                from ..data.dataset import CALENDAR_FIELDS

                if level in CALENDAR_FIELDS:
                    try:
                        data.values_of(level)  # quarter, month or seasonal(): from the time axis, or why not
                    except Exception as exc:
                        raise ExpressionError(str(exc.args[0] if exc.args else exc)) from None
                    continue
                raise ExpressionError(f"{level!r} is neither a series nor an axis of the data")

    # estimation

    @api("pymetrix:equation.fit")
    def fit(
        self,
        data: Dataset,
        span: tuple[Any, Any] | None = None,
        where: str | None = None,
        method: Any = None,
        weights: str | None = None,
        missing: Literal["drop", "raise"] = "drop",
    ) -> FittedEquation:
        """Estimate the equation on ``data``: by least squares unless ``method`` says otherwise (``OLS(cov="hac")``,
        ``IV(...)``, ``ML()``, ``Ridge()``, ...).

        ``span`` (first, last) and ``where`` (a condition such as ``"recession == 0"``) choose the sample; ``weights``
        names a weight series; ``missing="raise"`` refuses rows with missing values instead of dropping them.
        Equations with ARMA errors or a GARCH variance are estimated by maximum likelihood by default."""
        if self.is_identity:
            raise ExpressionError(f"{self.text!r} is an identity; there is nothing to estimate")
        from .methods import OLS
        from .ml import ML

        if method is None:
            method = ML() if self.has_error_model else OLS()
        self.check_names(data)
        from dataclasses import replace

        from .._capture import capture

        with capture() as caught:
            fit = method.estimate(self, data, span=span, where=where, weights=weights, missing=missing)
        if caught and getattr(fit, "provenance", None) is not None:  # warnings belong to the result, not just the console
            new = tuple(w for w in caught if w not in fit.provenance.warnings)  # a backend adapter may have recorded some already
            object.__setattr__(fit, "provenance", replace(fit.provenance, warnings=(*fit.provenance.warnings, *new)))
        converged = (getattr(fit, "extra", None) or {}).get("converged")
        if converged is not None and getattr(fit, "provenance", None) is not None:  # iterative estimators say whether they did
            object.__setattr__(fit, "provenance", replace(fit.provenance, converged=bool(converged)))
        if getattr(fit, "provenance", None) is not None:  # how the rows were chosen, and choices made automatically
            source, rule = data.sample_record(span, where)
            resolved = dict(fit.provenance.resolved or {})
            stats = getattr(fit, "stats", None) or {}
            if "hac_lags" in stats:  # the HAC truncation lag used (the rule's choice when none was given)
                resolved["hac_lags"] = int(stats["hac_lags"])
            object.__setattr__(fit, "provenance", replace(fit.provenance, sample_source=source, sample_definition=rule,
                                                          resolved=resolved or None))
            object.__setattr__(fit, "provenance", self._effective(fit, data, span, where, weights))
        return fit

    def _effective(self, fit: Any, data: Dataset, span: Any, where: Any, weights: Any) -> Any:
        """The rows the fit actually used: those of its residuals (a library may take some of pymetrix's rows as
        presample, as first differences do), with the reason each requested row was not used."""
        from .._capture import with_effective_sample

        design = getattr(fit, "design", None)
        rows = design.index if design is not None else None
        resid = getattr(fit, "resid", None)
        used = resid.index if resid is not None and list(resid.index.names) == list(data.frame.index.names) else rows
        if used is None:
            return fit.provenance
        presample = rows.difference(used, sort=False) if rows is not None and len(used) < len(rows) else ()
        variables = [*self.variables(), *([weights] if weights else [])]
        return with_effective_sample(fit.provenance, data, used, span=span, where=where, variables=variables, presample=presample)

    @api("pymetrix:equation.select_lags")
    def select_lags(self, data: Dataset, max_lags: int = 8, of: Sequence[str] | None = None, span: tuple[Any, Any] | None = None,
                    method: Any = None) -> Any:
        """Choose how many lags to add, by AIC, BIC and Hannan-Quinn on the sample the largest model allows: lags 1 to
        p of each variable in ``of`` (by default the left-hand side as written) for p = 0 ... ``max_lags``. Returns
        an ``EquationLagSelection``: its ``table``, ``best`` and ``equation("bic")``."""
        from .lag_search import select_lags

        return select_lags(self, data, max_lags, of, span, method)

    @api("pymetrix:equation.fit_rolling")
    def fit_rolling(self, data: Dataset, window: int, span: tuple[Any, Any] | None = None, method: Any = None, step: int = 1) -> Any:
        """Estimate over a moving window of ``window`` periods; returns a ``RollingFit``."""
        from .rolling import fit_windows

        return fit_windows(self, data, window, None, span, method, step)

    @api("pymetrix:equation.fit_recursive")
    def fit_recursive(self, data: Dataset, min_obs: int | None = None, span: tuple[Any, Any] | None = None, method: Any = None, step: int = 1) -> Any:
        """Estimate over an expanding window starting with ``min_obs`` periods; returns a ``RollingFit``."""
        from .rolling import fit_windows

        return fit_windows(self, data, None, min_obs, span, method, step)

    def coefficient_names(self) -> tuple[str, ...]:
        """Names of the coefficients a linear-form fit will produce (without C() dummies)."""
        if self.form == "explicit":
            return self.param_names
        return (("const",) if self.intercept else ()) + tuple(t.label for t in self.terms)

    def with_params(self, **values: float) -> FittedEquation:
        """Calibrate instead of estimating."""
        from .fitted import FittedEquation
        from .methods import Calibrated
        from .params import point

        if self.is_identity:
            raise ExpressionError("identities have no parameters")
        if self.fixed_effects or self.categoricals:
            raise ExpressionError("equations with fe() or C() terms cannot be calibrated")
        names = self.coefficient_names()
        missing = set(names) - set(values)
        extra = set(values) - set(names)
        if missing or extra:
            raise ExpressionError(f"with_params needs exactly {list(names)}; missing {sorted(missing)}, unexpected {sorted(extra)}")
        return FittedEquation(self, point(names, [values[n] for n in names]), Calibrated())

    def residual_expression(self, values: Mapping[str, float] | None = None) -> Expr:
        """``lhs - rhs`` with parameter values substituted; zero when the equation holds."""
        if self.is_identity:
            return BinOp("-", self.lhs, self.rhs)
        values = values or {}
        if not isinstance(self.family, Gaussian):
            raise ExpressionError(f"{type(self.family).__name__} equations cannot be used in a model")
        if self.fixed_effects or self.categoricals:
            raise ExpressionError("equations with fe() or C() terms cannot be used in a model yet")
        rhs = self.rhs if self.form == "explicit" else self.linear_rhs()
        missing = rhs.params() - set(values)
        if missing:
            raise ExpressionError(f"no values for parameters {sorted(missing)}")
        return BinOp("-", self.lhs, rhs.substitute(values))

    def linear_rhs(self) -> Expr:
        """The right-hand side of a linear-form equation with its coefficients as parameters."""
        rhs: Expr = Num(0.0)
        if self.intercept:
            rhs = add(rhs, Coef("const"))
        for t in self.terms:
            rhs = add(rhs, mul(Coef(t.label), t.expr))
        return rhs

    def algebraic(self, style: Literal["unicode", "latex"] = "unicode", error: bool | None = None) -> str:
        """The equation as an economist writes it: ``cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ`` (see ``pymetrix.notation``)."""
        from ..notation import algebraic

        return algebraic(self, style, error=error)

    def _repr_latex_(self) -> str | None:
        """In notebooks with algebraic notation on, the equation is typeset."""
        from ..notation import get_notation

        return f"$${self.algebraic('latex')}$$" if get_notation() == "algebraic" else None

    def __repr__(self) -> str:
        params = f", params={list(self.param_names)}" if self.param_names else ""
        errors = f", errors={self.errors!r}" if self.errors is not None else ""
        variance = f", variance={self.variance!r}" if self.variance is not None else ""
        return f"Equation({self.text!r}{params}{errors}{variance})"
