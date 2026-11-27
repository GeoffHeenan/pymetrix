"""The compiled model: one indexed representation behind every way of solving a model.

    system = model.compile()
    system.endogenous, system.exogenous, system.shocks      # roles, in order
    system.parameters                                        # values, with status: estimated, calibrated, fixed
    system.incidence                                         # the lags and leads at which each variable appears
    system.residuals()                                       # the equations as residuals, parameters substituted
    model.check(data, span)                                  # diagnostics before solving

A ``Model`` is what the user writes: equations, fitted or calibrated, and identities. ``compile()`` turns it into
a ``ModelSystem``: the equations as residuals with their parameters still symbolic, every variable with its
role, the parameters with their values and status, the lead and lag structure, and the recursive and
simultaneous blocks. Backward simulation, perfect foresight, linearisation and the Dynare importer all work
from it. Estimated coefficients are named by their equation (``cons.y``), so two equations' constants never
collide; calibrated parameters keep their own names, so one parameter can appear in several equations.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import inf, isfinite
from typing import TYPE_CHECKING, Any, Literal

from ..errors import NotFoundError, SpecificationError
from ..expr import BinOp, Coef, Expr, Name, Num, diff
from ..metadata import Param, Role
from ..output import Table, bounds_text
from ..results import deprecated_alias

if TYPE_CHECKING:
    from ..estimation import Equation

__all__ = ["Parameter", "ParameterSet", "SystemEquation", "ModelSystem"]

Status = Literal["estimated", "calibrated", "fixed"]
_ZERO = Num(0.0)


@dataclass(frozen=True)
class Parameter:
    """One parameter of a model: its value, where it came from, and its metadata.

    ``status``: ``estimated`` (from a fit), ``calibrated`` (set by the user), or ``fixed`` (held at its value
    when the model's parameters are estimated). ``spec`` carries the ``Param`` metadata: bounds, transform,
    start, prior and symbol."""

    name: str
    value: float
    status: Status = "calibrated"
    equation: str | None = None  # the equation an estimated coefficient belongs to
    se: float | None = None
    spec: Param | None = None

    @property
    def bounds(self) -> tuple[float, float]:
        return self.spec.limits if self.spec is not None else (-inf, inf)

    @property
    def prior(self) -> Any:
        return self.spec.prior if self.spec is not None else None


def prior_text(prior: Any) -> str:
    """A prior as it is written: ``Beta(0.7, 0.1)``."""
    if prior is None:
        return ""
    values = ", ".join(f"{v:g}" if isinstance(v, (int, float)) else repr(v) for v in vars(prior).values())
    return f"{type(prior).__name__}({values})"


@dataclass(frozen=True)
class ParameterSet:
    """The parameters of a model, in order, looked up by name."""

    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise SpecificationError(f"parameter names must be unique: {dups}")

    @classmethod
    def from_values(cls, values: Mapping[str, float], status: Status = "calibrated", specs: Mapping[str, Param] | None = None) -> ParameterSet:
        return cls(tuple(Parameter(n, float(v), status, spec=(specs or {}).get(n)) for n, v in values.items()))

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def __contains__(self, name: object) -> bool:
        return any(p.name == name for p in self.parameters)

    def __getitem__(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise NotFoundError(f"no parameter {name!r}; parameters: {self.names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def values(self) -> dict[str, float]:
        return {p.name: p.value for p in self.parameters}

    def with_values(self, values: Mapping[str, float], status: Status | None = None) -> ParameterSet:
        """New values for some parameters (and, if given, a new status for them)."""
        unknown = sorted(set(values) - set(self.names))
        if unknown:
            raise NotFoundError(f"no parameters {unknown}; parameters: {list(self.names)}")
        return ParameterSet(tuple(replace(p, value=float(values[p.name]), status=status or p.status) if p.name in values else p
                                  for p in self.parameters))

    def fix(self, *names: str) -> ParameterSet:
        """Hold these parameters at their values when the model is estimated."""
        missing = sorted(set(names) - set(self.names))
        if missing:
            raise NotFoundError(f"no parameters {missing}")
        return ParameterSet(tuple(replace(p, status="fixed") if p.name in names else p for p in self.parameters))

    def by_status(self, status: Status) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.status == status)

    def table(self) -> Table:
        rows = []
        for p in self.parameters:
            low, high = p.bounds
            bounds = bounds_text(low, high)
            rows.append([p.value, p.se if p.se is not None else None, p.status, p.equation or "", bounds, prior_text(p.prior),
                         p.spec.label if p.spec is not None else ""])
        return Table(["value", "std err", "status", "equation", "bounds", "prior", "description"], rows, index=list(self.names),
                     title="Parameters")

    def __repr__(self) -> str:
        counts = ", ".join(f"{len(self.by_status(s))} {s}" for s in ("estimated", "calibrated", "fixed") if self.by_status(s))
        return f"<ParameterSet: {counts or 'empty'}>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


@dataclass(frozen=True)
class SystemEquation:
    """One equation of a compiled model: ``residual`` is zero when it holds, with parameters as symbols."""

    name: str
    dependent: str  # the variable it determines (was target)
    residual: Expr
    stochastic: bool
    source: Any = field(default=None, compare=False, repr=False)  # the Equation it came from

    def __setstate__(self, state: dict[str, Any]) -> None:  # pickled (in a project's cached results) before the rename
        if "target" in state:
            state["dependent"] = state.pop("target")
        self.__dict__.update(state)

    target = deprecated_alias("dependent", "target", why="the variable the equation determines; target is its old name")


@dataclass(frozen=True, eq=False)
class ModelSystem:
    """A model compiled: equations as residuals, variables by role, parameters, lead and lag structure."""

    equations: tuple[SystemEquation, ...]
    parameters: ParameterSet
    endogenous: tuple[str, ...]
    exogenous: tuple[str, ...]
    shocks: tuple[str, ...]
    roles: Mapping[str, Role]

    def residuals(self, values: Mapping[str, float] | None = None) -> list[Expr]:
        """The residuals with parameter values substituted: the model's own, or ``values`` where given."""
        current = {**self.parameters.values(), **(values or {})}
        return [e.residual.substitute(current) for e in self.equations]

    @property
    def incidence(self) -> dict[str, tuple[int, ...]]:
        """For each variable, the lags (negative) and leads (positive) at which it appears."""
        seen: dict[str, set[int]] = {}
        for e in self.equations:
            for ref in e.residual.refs():
                seen.setdefault(ref.id, set()).add(ref.lag)
        order = [*self.endogenous, *self.exogenous]
        return {v: tuple(sorted(seen.get(v, ()))) for v in order}

    @property
    def max_lag(self) -> int:
        return max((-lag for lags in self.incidence.values() for lag in lags if lag < 0), default=0)

    @property
    def max_lead(self) -> int:
        return max((lag for lags in self.incidence.values() for lag in lags if lag > 0), default=0)

    def contemporaneous(self) -> dict[str, list[str]]:
        """For each equation, the endogenous variables it contains in the current period."""
        endo = set(self.endogenous)
        return {e.name: sorted({r.id for r in e.residual.refs() if r.lag == 0 and r.id in endo}) for e in self.equations}

    # -- views -------------------------------------------------------------------

    def equations_table(self) -> Table:
        rows = []
        for e in self.equations:
            algebraic = ""
            if e.source is not None:
                try:
                    algebraic = e.source.algebraic()
                except Exception:  # an expression the notation cannot write
                    algebraic = ""
            rows.append([e.dependent, "stochastic" if e.stochastic else "identity", str(e.source.text) if e.source is not None else "", algebraic])
        return Table(["determines", "kind", "equation", "algebraic"], rows, index=[e.name for e in self.equations], title="Equations")

    def variables_table(self) -> Table:
        determined = {e.dependent: e.name for e in self.equations}
        rows = []
        for v, lags in self.incidence.items():
            role = self.roles.get(v, Role.ENDOGENOUS if v in self.endogenous else Role.EXOGENOUS)
            rows.append([role.value, determined.get(v, ""), ", ".join(_timing(k) for k in lags)])
        return Table(["role", "equation", "appears at"], rows, index=list(self.incidence), title="Variables")

    def incidence_table(self) -> Table:
        """Equations by variables: where each variable enters each equation (t-1, t, t+1...)."""
        names = list(self.incidence)
        rows = []
        for e in self.equations:
            lags: dict[str, set[int]] = {}
            for r in e.residual.refs():
                lags.setdefault(r.id, set()).add(r.lag)
            rows.append([", ".join(_timing(k) for k in sorted(lags[v])) if v in lags else "" for v in names])
        return Table(names, rows, index=[e.name for e in self.equations], title="Incidence: where each variable enters each equation")

    def __repr__(self) -> str:
        return (f"<ModelSystem: {len(self.equations)} equations, {len(self.exogenous)} exogenous, "
                f"{len(self.parameters)} parameters, lags {self.max_lag}, leads {self.max_lead}>")


def _timing(lag: int) -> str:
    return "t" if lag == 0 else f"t{lag:+d}"


def _no_expanding(eq: Any, symbolic: Expr) -> None:
    """Expanding windows (cumavg(x), ...) use every period up to t, not fixed lags: refused in a model."""
    from ..expr import Call
    from ..expr.nodes import EXPANDING
    from .solvers import ModelError

    from ..expr.nodes import DATA_ONLY

    rows = sorted({n.fn for n in symbolic.walk() if isinstance(n, Call) and n.fn in DATA_ONLY})
    if rows:
        raise ModelError(f"{eq.text!r} uses {', '.join(f + '()' for f in rows)}, a function of the data's rows (their dates, "
                         "or random draws): a model has no rows to take them from; make it a series of the data first "
                         "(data.with_series(yr=\"year()\")) and use the series as an exogenous variable (a model's own "
                         "random draws are its shocks)")
    found = sorted({n.fn for n in symbolic.walk() if isinstance(n, Call) and n.fn in EXPANDING})
    if found:
        raise ModelError(f"{eq.text!r} uses {', '.join(f + '()' for f in found)}, an expanding window: it uses every period "
                         "up to t, not fixed lags, so a model cannot solve or linearise it; make it a derived series of the "
                         "data, or use a moving window (movavg(x, n), or cmovavg(x, n), centred, which a model reads as leads)")


def compile_model(equations: Sequence[Any], params: Mapping[str, float] | ParameterSet | None,
                  roles: Mapping[str, Role] | None) -> ModelSystem:
    """The work behind ``Model.compile()``: see the module docstring."""
    from ..estimation import Equation, FittedEquation
    from .solvers import ModelError

    calibrated = params if isinstance(params, ParameterSet) else ParameterSet.from_values(dict(params or {}))
    built: list[SystemEquation] = []
    found: dict[str, Parameter] = {}
    for item in equations:
        if isinstance(item, FittedEquation):
            eq = item.equation
            values = item.params.as_dict()
            se = dict(zip(item.params.names, map(float, item.params.se)))
            specs = {p.name: p for p in eq.param_specs}
            qualify = {n: f"{eq.label}.{n}" for n in values}
            for n, v in values.items():
                found[qualify[n]] = Parameter(qualify[n], v, "estimated", eq.label, se.get(n), specs.get(n))
        elif isinstance(item, Equation):
            eq, qualify = item, {}
            if not item.is_identity:
                names = item.coefficient_names()
                missing = [n for n in names if n not in calibrated]
                if missing:
                    raise ModelError(f"{item.text!r} needs values for {', '.join(missing)}: fit it, or give params={{...}}")
                specs = {p.name: p for p in item.param_specs}
                for n in names:
                    p = calibrated[n]
                    found.setdefault(n, replace(p, spec=p.spec or specs.get(n)))
        else:
            raise TypeError(f"a Model takes Equations, FittedEquations, FittedSystems and Models, not {type(item).__name__}")
        if eq.errors is not None and not isinstance(item, FittedEquation):
            raise ModelError(f"{eq.text!r} has ARMA errors: fit it before using it in a model (its AR and MA coefficients are estimated)")
        values_now = {n: found[qualify.get(n, n)].value for n in (eq.coefficient_names() if not eq.is_identity else ())}
        eq.residual_expression(values_now)  # the same checks as always: Gaussian, no fe()/C(), parameters present
        symbolic = _symbolic(eq, qualify)
        _no_expanding(eq, symbolic)
        if eq.errors is not None:
            built.extend(_with_arma(eq, symbolic))
        else:
            built.append(SystemEquation(eq.label, eq.require_dependent(), symbolic, not eq.is_identity, eq))
    for attr in ("dependent", "name"):
        seen = [getattr(e, attr) for e in built]
        dups = sorted({s for s in seen if seen.count(s) > 1})
        if dups:
            raise ModelError(f"more than one equation has {attr} {dups}")
    endogenous = tuple(e.dependent for e in built)
    names = set().union(*(e.residual.variables() for e in built)) if built else set()
    exogenous = tuple(sorted(names - set(endogenous)))
    roles = dict(roles or {})
    shocks = tuple(v for v in exogenous if roles.get(v) == Role.SHOCK)
    full_roles = {**{v: Role.ENDOGENOUS for v in endogenous}, **{v: roles.get(v, Role.EXOGENOUS) for v in exogenous}}
    return ModelSystem(tuple(built), ParameterSet(tuple(found.values())), endogenous, exogenous, shocks, full_roles)


def innovation_name(label: str) -> str:
    """The model variable holding an equation's ARMA innovations (for its MA terms): ``e_<equation>``."""
    return f"e_{label}"


def _with_arma(eq: Equation, u: Expr) -> list[SystemEquation]:
    """An equation with ARMA errors, in a model: ``u = lhs - rhs`` follows ``u = sum phi u[-i] + sum theta e[-j] + e``, so
    its residual is ``u - sum phi u[-i] - sum theta e[-j]``, the innovation e (zero ahead, or a draw). With MA terms, e
    is a model variable of its own, ``e_<equation>``, defined by that residual, so that its lags are known."""
    spec = eq.errors
    label = eq.label
    residual = u
    for lag in spec.ar_lags:
        residual = BinOp("-", residual, BinOp("*", Coef(f"{label}.ar[{lag}]"), u.shift(-lag)))
    innovation = innovation_name(label)
    for lag in spec.ma_lags:
        residual = BinOp("-", residual, BinOp("*", Coef(f"{label}.ma[{lag}]"), Name(innovation, -lag)))
    out = [SystemEquation(label, eq.require_dependent(), residual, True, eq)]
    if spec.ma_lags:
        out.append(SystemEquation(innovation, innovation, BinOp("-", Name(innovation, 0), residual), False, None))
    return out


def _symbolic(eq: Equation, qualify: Mapping[str, str]) -> Expr:
    """``lhs - rhs`` with parameters as symbols, renamed by ``qualify`` (estimated coefficients by equation)."""
    if eq.is_identity:
        return BinOp("-", eq.lhs, eq.rhs)
    rhs = eq.rhs if eq.form == "explicit" else eq.linear_rhs()
    if qualify:
        rhs = rhs.map(lambda n: Coef(qualify[n.id]) if isinstance(n, Coef) and n.id in qualify else None)
    return BinOp("-", eq.lhs, rhs)


# ---------------------------------------------------------------------------------------------
# checks


@dataclass(frozen=True)
class ModelCheck:
    """The diagnostics of ``Model.check``: one row per check, each ``ok``, ``note``, ``warning`` or ``error``."""

    rows: tuple[tuple[str, str, str], ...]  # check, result, detail
    system: ModelSystem = field(repr=False)
    # with data and a span: the inputs each equation lacks in each period (Model.missing_inputs), else None
    missing: Any = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """No errors (notes and warnings allowed)."""
        return all(result != "error" for _, result, _ in self.rows)

    def errors(self) -> list[str]:
        return [f"{check}: {detail}" for check, result, detail in self.rows if result == "error"]

    def table(self) -> Table:
        status = "passes" if self.ok else f"{len(self.errors())} error(s)"
        return Table(["result", "detail"], [[r, d] for _, r, d in self.rows], index=[c for c, _, _ in self.rows],
                     title=f"Model check: {status}")

    def __repr__(self) -> str:
        return f"<ModelCheck: {'passes' if self.ok else '; '.join(self.errors())}>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


def check_model(model: Any, data: Any = None, span: Any = None, shocks: Sequence[str] = (), steady: bool | None = None,
                guess: Mapping[str, float] | None = None, scenario: Any = None) -> ModelCheck:
    """The diagnostics behind ``Model.check``."""
    import numpy as np
    import pandas as pd

    system = model.compile()
    rows: list[tuple[str, str, str]] = []
    add = lambda check, result, detail: rows.append((check, result, detail))  # noqa: E731

    add("equations", "ok", f"{len(system.equations)} equations for {len(system.endogenous)} endogenous variables; "
        f"{len(system.exogenous)} exogenous: {', '.join(system.exogenous) or 'none'}")

    missing_target = [e.name for e in system.equations if diff(e.residual, Name(e.dependent, 0)) == _ZERO]
    add("targets", "error" if missing_target else "ok",
        f"{', '.join(missing_target)} do not contain their own variable in the current period" if missing_target
        else "every equation contains the variable it determines, in the current period")

    blocks = model.blocks()  # with every target present, each block's equations can be paired with its variables
    simultaneous = [b for b in blocks if b.simultaneous]
    add("simultaneity", "ok", f"{len(blocks) - len(simultaneous)} recursive and {len(simultaneous)} simultaneous block(s)"
        + (f" (largest: {', '.join(max(simultaneous, key=lambda b: len(b.variables)).variables)})" if simultaneous else ""))

    params = system.parameters
    bad = [p.name for p in params if not isfinite(p.value)]
    outside = [p.name for p in params if isfinite(p.value) and not (p.bounds[0] <= p.value <= p.bounds[1])]
    counts = ", ".join(f"{len(params.by_status(s))} {s}" for s in ("estimated", "calibrated", "fixed") if params.by_status(s)) or "none"
    if bad:
        add("parameters", "error", f"no finite value for {', '.join(bad)}")
    elif outside:
        add("parameters", "warning", f"{', '.join(outside)} outside their bounds")
    else:
        add("parameters", "ok", counts)

    timing = f"lags up to {system.max_lag}, leads up to {system.max_lead}"
    add("timing", "note" if system.max_lead else "ok",
        timing + ("; leads need solver=PerfectForesight() or linearize()" if system.max_lead else ""))

    shock_names = tuple(shocks) or system.shocks
    unknown = [s for s in shock_names if s not in system.exogenous]
    timed = sorted({s for s in shock_names for lag in system.incidence.get(s, ()) if lag != 0})
    if unknown or timed:
        add("shocks", "error", "; ".join(filter(None, [f"{_names(unknown)} {_be(unknown)} not exogenous in this model" if unknown else "",
                                                       f"{_names(timed)} {'enters' if len(timed) == 1 else 'enter'} with a lag or lead"
                                                       if timed else ""])))
    elif shock_names:
        add("shocks", "ok", f"{', '.join(shock_names)}, each in the current period only")

    if data is not None:
        absent = [v for v in system.exogenous if v not in data.names and v not in shock_names]
        add("data", "error" if absent else "ok",
            f"exogenous {', '.join(absent)} not in the data" if absent else "every exogenous variable is in the data")
        if span is not None and not absent and data.time_axis is not None:
            freq = data.frame.index.freq
            from ..data.sample import as_span
            span = as_span(span)  # '1990Q1:2019Q4' or a pair
            first_period, last_period = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
            starts, paths = [], []
            for v, lags in system.incidence.items():
                if v in shock_names or v not in data.names or not lags:
                    continue
                if v in system.endogenous:  # starting values: the periods before the span that its lags reach
                    deepest = min((lag for lag in lags if lag < 0), default=0)
                    if not deepest:
                        continue
                    first, last, gaps = first_period + deepest, first_period - 1, starts
                else:  # the exogenous path over the span, and the lags and leads around it
                    first, last, gaps = first_period + min(min(lags), 0), last_period + max(max(lags), 0), paths
                window = data.frame[v].reindex(pd.period_range(first, last, freq=freq))
                if not np.all(np.isfinite(window.to_numpy(float))):
                    gaps.append(v)
            add("starting values", "error" if starts else "ok",
                f"no data before {first_period} for {', '.join(starts)}, which the lags need" if starts
                else f"lagged values available before {first_period}")
            add("exogenous paths", "warning" if paths else "ok",
                f"{_names(paths)} {'has' if len(paths) == 1 else 'have'} no values for part of {first_period} to {last_period}: "
                "set them with a Scenario" if paths
                else f"exogenous values cover {first_period} to {last_period}")

    missing = None
    if data is not None and span is not None and data.kind == "time_series":
        from ..errors import PyMetrixError
        from .model import missing_inputs

        try:
            missing = missing_inputs(model, data, span, scenario)
            if shock_names:  # shocks are drawn or set by a scenario, not read from the data
                missing = missing[~missing["input"].str.split("[").str[0].isin(shock_names)].reset_index(drop=True)
        except (PyMetrixError, ValueError):  # what the checks above report, or a scenario that does not fit the model
            missing = None
    if steady if steady is not None else system.max_lead > 0:
        from .solvers import ModelError

        try:
            ss = model.steady_state(guess=dict(guess) if guess else None)
            add("steady state", "ok", "found")
            try:
                model.linearize(ss, shocks=[s for s in shock_names if s in system.exogenous]).solve()
                add("Blanchard-Kahn", "ok", "a unique stable solution exists")
            except ModelError as exc:
                add("Blanchard-Kahn", "error", str(exc))
        except ModelError as exc:
            add("steady state", "error", str(exc))
    return ModelCheck(tuple(rows), system, missing)


def _names(names: Sequence[str]) -> str:
    return ", ".join(names)


def _be(names: Sequence[str]) -> str:
    return "is" if len(names) == 1 else "are"
