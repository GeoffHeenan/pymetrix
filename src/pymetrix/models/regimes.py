"""Regimes with known dates: parameters that change, and alternative policy rules, from given periods.

    from pymetrix.models import Regime
    post = Regime("2009Q1", {"phi_pi": 1.2}, name="post")              # a new policy response from 2009Q1, for good
    peg = Regime("2025Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2026Q4")    # a temporary peg; the rule before it resumes
    alt = Regime("2027Q1", equations={"policy": "i = 0.9*i[-1] + 0.2*pi + ei"})   # another rule, by equation name

    solution.solve(data, ("2025Q1", "2030Q4"), regimes=[post, peg])    # the path with the regimes known in advance
    fit.forecast(("2025Q1", "2030Q4"), regimes="2025Q1..2026Q4: rho=1, phi_pi=0")   # the same, written as text
    fit = template.estimate(data, regimes=[post])                       # phi_pi@post estimated with phi_pi
    regime_filter(fit, data).regime                                     # the regime of each period

A model's text may hold its regimes (a ``regimes`` section: ``pymetrix.models.source``): they become the template's
``regimes``, which its solution, ``estimate`` (those inside the sample; later ones for the fit's forecasts) and
``regime_filter`` follow unless given others.

Regimes are given in order of their starts; each holds until the next starts, or until its ``end``, after which
the regime before it resumes. A regime changes the parameters and equations it names and keeps the rest as they
were in the regime before it; the base model holds before the first. Equations are replaced by name, with the same
variables (an alternative policy rule); names in them that are not variables are parameters, with values from the
model or the regime.

Agents know the whole path of regimes from the first period, as in DSGE.jl's temporary alternative policies (a
credible announcement). ``Regime(..., expected=False)`` is a surprise instead: unknown until it starts, and then
believed to hold for good.

The regime in force before the first period is the reference: its steady state is where every regime is linearised,
each with the constant its equations leave there (as OccBin does), so a regime with another steady state moves
the path towards it. Given the regime of each period, the decision rules ``x(t) = P(t) x(t-1) + k(t) + Q(t) e(t)``
are found backwards from the last regime, which holds for good: its own stable solution, with the constant that
takes the path to its steady state, so there is no horizon to choose. Regimes in between need no stable solution of
their own (a peg for a while, even from the first period). With nothing changing in the periods solved, the result is
the ordinary solution of the regime in force, exactly.

Filtering and estimation: the Kalman filter and smoother run through the same time-varying rules (the state before
the sample from the reference regime's stationary distribution), and its likelihood is what estimation maximises.
Posterior forecasts solve and filter each draw with its own values of the regimes' parameters; decompositions go
through the time-varying rules, the regimes' constants a part of their own (``RegimeFilter.decompose``,
``LinearSolution.decompose(regimes=)``); moments and variance decompositions are of one regime at a time, held for
good at its own steady state (``FittedModel.moments(regime=)``).

The nonlinear solutions take them too: perfect foresight switches the stacked system's equations at their dates (the
nonlinear model's own, not linearised; ``Model.solve(solver=PerfectForesight(), regimes=)``), and OccBin combines them
with its occasionally binding constraints (``occbin``, ``occbin_filter``, OccBin forecasts and estimation; ``Schedule``),
a regime's equations free to rewrite, remove or bring in a max or min. A surprise is solved again from the state
reached when it starts, with the schedule then believed (``regime_schedules``), as unanticipated shocks are.
Identification of a model estimated with regimes differentiates the moments of every period of the sample through the
time-varying rules (``bayes.regime_moments``).
A regime's values are estimated as ``<parameter>@<regime>`` (the regime's name, or its start: ``phi_pi@2009Q1``),
with the parameter's own bounds and prior unless ``specs``, ``bounds`` or ``priors`` name them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from .._api import api
from ..errors import SpecificationError
from ..output import Table
from .solvers import ModelError

__all__ = ["Regime", "RegimeFilter", "parse_regimes", "regime_filter"]

_BASE = "base"
REGIMES = "regimes"  # the decompositions' part for the regimes' constants
_NAME = re.compile(r"[A-Za-z0-9_.]+")


@dataclass(frozen=True)
class Regime:
    """Parameter values (``params``) and equations replaced by name (``equations``) from period ``start`` on: see the
    module docstring. ``end``: the last period of a temporary regime, after which the regime before it resumes.
    ``expected=False``: a surprise, unknown until it starts and then believed permanent. ``description``: what it is,
    for people (a model text's quoted description of the regime)."""

    start: Any
    params: Mapping[str, float] = field(default_factory=dict)
    equations: Mapping[str, str] = field(default_factory=dict)
    name: str = ""
    end: Any = None
    expected: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", str(self.start).strip())
        if self.end is not None:
            object.__setattr__(self, "end", str(self.end).strip())
        try:
            params = {str(k): float(v) for k, v in dict(self.params or {}).items()}
        except (TypeError, ValueError):
            raise SpecificationError(f"a regime's params are numbers by name, e.g. {{'phi_pi': 1.2}}: {self.params!r}") from None
        equations = {str(k): str(v) for k, v in dict(self.equations or {}).items()}
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "equations", equations)
        if not params and not equations:
            raise SpecificationError(f"the regime from {self.start} changes nothing: give params= or equations=")
        if self.name and not _NAME.fullmatch(self.name):
            raise SpecificationError(f"a regime's name is letters, digits, _ and . (it names estimated values, phi_pi@name): {self.name!r}")

    @property
    def label(self) -> str:
        """The regime's name, or its start when it has none."""
        return self.name or self.start

    def __hash__(self) -> int:
        return hash((self.start, tuple(self.params.items()), tuple(self.equations.items()), self.name, self.end, self.expected))


@api("pymetrix:parse_regimes")
def parse_regimes(text: str) -> tuple[Regime, ...]:
    """Regimes written as text, as the dialogs take them: ``"2009Q1: phi_pi=1.2; 2025Q1..2026Q4: rho=1, phi_pi=0"``.
    Each item is a start (or ``start..end``, a temporary regime), a colon and parameter values; items are separated by
    ``;`` or new lines. ``surprise`` after the period makes it a surprise (``expected=False``):
    ``"2026Q1 surprise: phi_pi=2"``."""
    out = []
    for item in [t.strip() for t in re.split(r"[;\n]", text or "") if t.strip()]:
        when, colon, values = item.partition(":")
        if not colon:
            raise SpecificationError(f"write a regime as period: name=value, ... (e.g. 2009Q1: phi_pi=1.2): {item!r}")
        when, surprise = re.subn(r"\s+surprise\s*$", "", when.strip())
        start, dots, end = when.strip().partition("..")
        params = {}
        for part in [p.strip() for p in values.split(",") if p.strip()]:
            name, eq, value = part.partition("=")
            try:
                params[name.strip()] = float(value)
            except ValueError:
                raise SpecificationError(f"give each parameter a number, as name=value: {part!r} in {item!r}") from None
            if not eq or not name.strip():
                raise SpecificationError(f"give each parameter a number, as name=value: {part!r} in {item!r}")
        if not start.strip() or (dots and not end.strip()):
            raise SpecificationError(f"a regime starts in a period, e.g. 2009Q1 (or 2025Q1..2026Q4 for a while): {item!r}")
        out.append(Regime(start.strip(), params, end=end.strip() if dots else None, expected=not surprise))
    return tuple(out)


def as_regimes(value: Any) -> tuple[Regime, ...]:
    """Regimes from what the entry points take: None, text, a Regime, or a sequence of Regimes."""
    if value is None:
        return ()
    if isinstance(value, str):
        return parse_regimes(value)
    if isinstance(value, Regime):
        return (value,)
    if isinstance(value, Sequence) and all(isinstance(r, Regime) for r in value):
        return tuple(value)
    raise SpecificationError("regimes are Regime objects, in order of their starts, or text such as '2009Q1: phi_pi=1.2'")


def regimes_text(regimes: Sequence[Regime]) -> str:
    """One line per regime, for summaries."""
    parts = []
    for r in regimes:
        when = f"{r.start}..{r.end}" if r.end else r.start
        what = ", ".join([f"{k}={v:g}" for k, v in r.params.items()] + [f"{k}: {v}" for k, v in r.equations.items()])
        parts.append(f"{r.label}: {when}{'' if r.expected else ' (unexpected)'}; {what}")
    return "; ".join(parts)


def with_values(regimes: Sequence[Regime], values: Mapping[str, float]) -> tuple[Regime, ...]:
    """The regimes with values ``<parameter>@<regime>`` from ``values`` (estimation's trial values)."""
    out = []
    for r in regimes:
        params = {k: float(values.get(f"{k}@{r.label}", v)) for k, v in r.params.items()}
        out.append(replace(r, params=params) if params != dict(r.params) else r)
    return tuple(out)


def regime_values(regimes: Sequence[Regime]) -> dict[str, float]:
    """Each regime's parameter values by estimation name, ``<parameter>@<regime>``."""
    return {f"{k}@{r.label}": float(v) for r in regimes for k, v in r.params.items()}


# ---------------------------------------------------------------------------------------------
# the path of regimes


@dataclass(frozen=True)
class _Phase:
    """A stretch of periods with the same model: from ``start`` to the next phase."""

    start: pd.Period
    params: tuple[tuple[str, float], ...]  # cumulative values over the base model
    equations: tuple[tuple[str, str], ...]
    name: str
    expected: bool
    given: bool = True  # a regime's own start (not the resumption after a temporary one)


def _period(value: Any, freq: Any, what: str) -> pd.Period:
    try:
        return pd.Period(str(value), freq=freq)
    except (ValueError, TypeError):
        raise SpecificationError(f"{what} {value!r} is not a period of the data's frequency") from None


def _number(value: Any, what: str) -> int:
    """A period numbered 1, 2, ... (``occbin``'s periods), from a regime's start or end."""
    try:
        return int(str(value))
    except ValueError:
        raise SpecificationError(f"{what} {value!r} is not a period number (1, 2, ...): OccBin's periods are numbered") from None


def _phases(regimes: Sequence[Regime], freq: Any) -> list[_Phase]:
    """The regimes as phases, in order, each with the cumulative parameters and equations; refuses regimes out of
    order, overlapping a temporary one, or named twice. ``freq=None``: periods numbered 1, 2, ... (OccBin's)."""
    out: list[_Phase] = []
    params: dict[str, float] = {}
    equations: dict[str, str] = {}
    name, previous, ends, names = _BASE, None, None, set()
    for r in regimes:
        if freq is None:
            start, end = _number(r.start, "the regime start"), _number(r.end, "the regime end") if r.end is not None else None
        else:
            start = _period(r.start, freq, "the regime start")
            end = _period(r.end, freq, "the regime end") if r.end is not None else None
        if end is not None and end < start:
            raise SpecificationError(f"the regime from {start} ends ({end}) before it starts")
        if previous is not None and start <= previous:
            raise SpecificationError(f"regimes go in order of their starts: {r.label} starts in {start}, not after {previous}")
        if ends is not None and start <= ends:
            raise SpecificationError(f"{r.label} starts in {start}, before the temporary regime before it ends ({ends})")
        if r.label in names or r.label == _BASE:
            raise SpecificationError(f"two regimes are called {r.label!r}: give them different names")
        names.add(r.label)
        before = (dict(params), dict(equations), name)
        params.update(r.params)
        equations.update(r.equations)
        name = r.label
        if out and out[-1].start == start:  # a temporary regime's end, followed at once by this one
            out.pop()
        out.append(_Phase(start, tuple(params.items()), tuple(equations.items()), name, r.expected))
        if end is not None:
            params, equations, name = before
            out.append(_Phase(end + 1, tuple(params.items()), tuple(equations.items()), name, r.expected, given=False))
            ends = end
        previous = start
    return out


def check_starts(regimes: Sequence[Regime], periods: pd.PeriodIndex, where: str) -> None:
    """Refuse regimes that start outside the periods: after the span (``where="span"``), or outside the sample after
    its first period (``where="sample"``)."""
    for r in regimes:
        start = _period(r.start, periods.freq, "the regime start")
        if where == "sample":
            if not periods[0] < start <= periods[-1]:
                raise ModelError(f"the regime {r.label} starts in {start}, outside the data ({periods[0]} to {periods[-1]}): "
                                 "regimes filtered or estimated start after the sample's first period and by its last")
        elif start > periods[-1]:
            raise ModelError(f"the regime {r.label} starts in {start}, after the span ends ({periods[-1]}): extend the span to it")


# ---------------------------------------------------------------------------------------------
# the model of each regime


def _base_model(sol: Any) -> Any:
    model = getattr(sol.model.steady, "model", None)
    if model is None:
        raise ModelError("regimes need the model behind the solution (solve one made by Model.linearize)")
    if any(abs(v) > 0 for v in (getattr(sol.model.steady, "growth", {}) or {}).values()):
        raise ModelError("regimes are for models without growth trends")
    return model


def _key(phase: _Phase | None, model: Any) -> tuple[Any, ...]:
    """What makes a regime's model: its parameter values that differ from the model's, and its equations."""
    if phase is None:
        return ((), ())
    params = model.parameters
    changed = tuple(sorted((k, v) for k, v in phase.params if k not in params or params[k].value != v))
    return changed, tuple(sorted(phase.equations))


def _carried(regimes: Sequence[Regime]) -> frozenset[str]:
    """The names the regimes' equations use: a parameter one of them brings in stays among the values of the regimes
    after it, also where a later regime replaces that equation again."""
    return frozenset(w for r in regimes for text in r.equations.values() for w in re.findall(r"[A-Za-z_]\w*", text))


def regime_model(model: Any, params: Mapping[str, float], equations: Mapping[str, str], carried: frozenset[str] = frozenset()) -> Any:
    """The model with these parameter values and equations replaced by name (see the module docstring). ``carried``:
    parameters an earlier regime's equation brought in, which these equations may no longer use."""
    from ..estimation import Equation
    from ..expr import parse_equation
    from .model import Model

    known = model.parameters
    if not equations:
        unknown = sorted(p for p in params if p not in known)
        if unknown:
            raise ModelError(f"{', '.join(unknown)} are not parameters of the model; its parameters: {', '.join(known.names)}")
        return model.with_parameters(dict(params))
    labels = [getattr(item, "label", None) or getattr(getattr(item, "equation", None), "label", None) for item in model.equations]
    unknown = sorted(set(equations) - set(labels))
    if unknown:
        raise ModelError(f"{', '.join(unknown)} are not equations of the model; its equations: {', '.join(str(n) for n in labels)}")
    variables = set(model.endogenous) | set(model.exogenous)
    values = {p.name: p.value for p in known if p.status != "estimated"}
    values.update(params)
    items = list(model.equations)
    for label, text in equations.items():
        i = labels.index(label)
        old = model.compile().equations[i].dependent if len(model.compile().equations) == len(items) else None
        try:
            lhs, _, rhs = parse_equation(text.replace("~", "=", 1) if "=" not in text else text)
        except Exception as exc:  # noqa: BLE001 - a parse error, reported with the equation
            raise ModelError(f"the regime's equation {label!r} does not parse: {exc}") from None
        names = lhs.variables() | rhs.variables()
        strange = sorted(n for n in names if n not in variables and n not in values)
        if strange:
            raise ModelError(f"the regime's equation {label!r} uses {', '.join(strange)}: neither variables of the model nor "
                             "parameters with values (give them in the regime's params)")
        used = sorted(n for n in names if n not in variables)
        left, _, right = text.replace("~", "=", 1).partition("=") if "=" not in text else text.partition("=")
        eq = Equation(f"{left.strip()} ~ {right.strip()}", params=used, name=label) if used else Equation(f"{left.strip()} = {right.strip()}", name=label)
        if old is not None and eq.require_dependent() != old:
            raise ModelError(f"the regime's equation {label!r} determines {eq.require_dependent()}, not {old}: an alternative "
                             "equation keeps the variable it determines")
        items[i] = eq
    new = Model(items, roles=model.roles, params={k: v for k, v in values.items()})
    extra = sorted(p for p in params if p not in new.parameters and p not in known and p not in carried)  # a replaced equation may drop some
    if extra:
        raise ModelError(f"{', '.join(extra)} are not parameters of the model or of the regime's equations")
    return new


def _linear(model: Any, ss: Any, shocks: tuple[str, ...], variables: tuple[str, ...] | None) -> Any:
    """A regime's model linearised at the reference steady state, with its constant, on the reference's variables and
    shocks (a shock the regime's equations leave out has no effect in it)."""
    from .linear import LinearModel, linearize

    present = tuple(s for s in shocks if s in model.exogenous)
    lin = linearize(model, ss, present, constant=True)
    if variables is not None and lin.variables != variables:
        raise ModelError("a regime's equations use lags or leads beyond one period that the model's do not (or no longer "
                         "use ones they do); write the alternative equation with the same timing")
    B = np.zeros((len(lin.variables), len(shocks)))
    for j, s in enumerate(present):
        B[:, shocks.index(s)] = lin.B[:, j]
    c = lin.c if lin.c is not None else np.zeros(len(lin.variables))
    return LinearModel(lin.A0, lin.A1, lin.A2, B, lin.variables, shocks, ss, lin.observed, c)


# ---------------------------------------------------------------------------------------------
# the time-varying decision rules


@dataclass(frozen=True, eq=False)
class _Rules:
    """``x(t) = P[t] x(t-1) + k[t] + w[t] + Q[t] e(t)`` on a window of periods, ``w`` from anticipated shocks through
    ``G[t] = -(A0 P[t+1] + A1)^-1 A0``."""

    P: np.ndarray
    Q: np.ndarray
    G: np.ndarray
    k: np.ndarray


class RegimePlan:
    """The regimes over the periods solved or filtered: the rule of each period, under what agents know then."""

    def __init__(self, sol: Any, ref_key: tuple[Any, ...], names: list[str], models: dict[tuple[Any, ...], Any],
                 infos: list[tuple[int, list[tuple[Any, ...]]]], periods: pd.PeriodIndex) -> None:
        self.sol, self.periods, self.names, self._ref = sol, periods, names, ref_key
        self.n = len(sol.variables)
        self.linear = {key: sol.model if key == ref_key else _linear(m, sol.model.steady, tuple(sol.shocks), tuple(sol.variables))
                       for key, m in models.items()}
        cache: dict[tuple[Any, ...], _Rules] = {}
        self.segments: list[tuple[int, int, _Rules]] = []
        for j, (start, believed) in enumerate(infos):
            stop = infos[j + 1][0] if j + 1 < len(infos) else len(believed)
            sig = tuple(believed)
            if sig not in cache:
                cache[sig] = self._rules(believed)
            self.segments.append((start, stop, cache[sig]))
        lagged = np.zeros(self.n, dtype=bool)
        for rules in cache.values():
            lagged |= np.any(np.abs(rules.P) > 0, axis=(0, 1))
        self.needed = lagged  # the variables whose previous value some period's rule uses

    def _final(self, key: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
        """The last regime's stable solution and the constant that takes the path to its steady state."""
        if key == self._ref:
            return self.sol.P, np.zeros(self.n)
        lin, name = self.linear[key], self.names[-1]
        try:
            P = lin.solve().P
        except ModelError as exc:
            raise ModelError(f"the last regime ({name}), which holds for good, has no unique stable solution: {exc}") from None
        try:
            d = -np.linalg.solve(lin.A0 @ P + lin.A1 + lin.A0, lin.c)
        except np.linalg.LinAlgError:
            raise ModelError(f"the last regime ({name}) has no steady state to go to") from None
        return P, d

    def _rules(self, believed: list[tuple[Any, ...]]) -> _Rules:
        N, n, m = len(believed), self.n, len(self.sol.shocks)
        P, Q, G, k = np.empty((N, n, n)), np.empty((N, n, m)), np.empty((N, n, n)), np.empty((N, n))
        P_next, k_next = self._final(believed[-1])
        for t in range(N - 1, -1, -1):
            lin = self.linear[believed[t]]
            try:
                lu = linalg.lu_factor(lin.A0 @ P_next + lin.A1, check_finite=True)
                if np.any(np.abs(np.diag(lu[0])) <= 1e-13 * max(1.0, float(np.max(np.abs(lu[0]))))):
                    raise np.linalg.LinAlgError
            except (np.linalg.LinAlgError, ValueError):
                raise ModelError(f"the regime in {self.periods[0] + t} has no solution (A0 P + A1 is singular)") from None
            P[t] = -linalg.lu_solve(lu, lin.A2)
            Q[t] = -linalg.lu_solve(lu, lin.B) if m else np.zeros((n, 0))
            G[t] = -linalg.lu_solve(lu, lin.A0)
            c = lin.c if lin.c is not None else np.zeros(n)
            k[t] = G[t] @ k_next - linalg.lu_solve(lu, c)
            P_next, k_next = P[t], k[t]
        return _Rules(P, Q, G, k)

    def paths(self, x0: np.ndarray, anticipated: np.ndarray, unanticipated: np.ndarray, constant: bool = True) -> np.ndarray:
        """Deviations from the reference steady state over the periods (T x variables): anticipated shocks known from
        the first period, unanticipated ones in their own; with ``constant=False``, without the regimes' constants (the
        responses to the shocks and the starting state alone)."""
        T = len(anticipated)
        out = np.empty((T, self.n))
        x = np.asarray(x0, float)
        for start, stop, R in self.segments:
            if start >= T:
                break
            N = len(R.P)
            w = np.zeros(self.n)
            ws = np.zeros((N, self.n))
            for t in range(N - 1, start - 1, -1):
                w = (R.Q[t] @ anticipated[t] if t < T else 0.0) + R.G[t] @ w
                ws[t] = w
            for t in range(start, min(stop, T)):
                x = R.P[t] @ x + (R.k[t] if constant else 0.0) + ws[t] + R.Q[t] @ unanticipated[t]
                out[t] = x
        return out

    def period_rules(self, T: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(P, k, Q)`` of each of the first ``T`` periods: the rule used in it, with no shocks anticipated."""
        out = []
        for start, stop, R in self.segments:
            for t in range(start, min(stop, T)):
                out.append((R.P[t], R.k[t], R.Q[t]))
        return out

    def spells(self, T: int) -> list[dict[str, str]]:
        """Which regime holds when, over the first ``T`` periods."""
        out: list[dict[str, str]] = []
        for t in range(T):
            name = self.names[t]
            if out and out[-1]["regime"] == name:
                out[-1]["to"] = str(self.periods[0] + t)
            else:
                out.append({"regime": name, "from": str(self.periods[0] + t), "to": str(self.periods[0] + t)})
        return out


def regime_plan(sol: Any, regimes: Any, periods: pd.PeriodIndex, where: str = "span") -> tuple[Any, RegimePlan | None]:
    """The reference solution (that of the regime in force before the first period, where the starting state comes from)
    and the plan of the regimes over the periods, or None when nothing changes in them (the reference solution is then
    the whole answer)."""
    regs = as_regimes(regimes)
    if not regs:
        return sol, None
    model = _base_model(sol)
    phases = _phases(regs, periods.freq)
    check_starts(regs, periods, where)
    first = periods[0]
    before = [p for p in phases if p.start < first]  # the reference: the regime the starting state comes from
    ahead = [p for p in phases if p.start >= first]
    reference = before[-1] if before else None
    ref_key = _key(reference, model)
    # drop changes to what already holds (a regime equal to the one before it)
    kept: list[_Phase] = []
    current = ref_key
    for p in ahead:
        key = _key(p, model)
        if key != current or not p.expected:
            kept.append(p)
            current = key
    if all(_key(p, model) == ref_key for p in kept):
        kept = []
    ref_sol = sol if ref_key == ((), ()) else _reference(sol, model, reference, _carried(regs))
    if not kept:
        return ref_sol, None
    window = max(len(periods), int((kept[-1].start - first).n) + 1)
    models: dict[tuple[Any, ...], Any] = {ref_key: None}
    for p in kept:
        key = _key(p, model)
        if key not in models:
            models[key] = regime_model(model, dict(p.params), dict(p.equations), _carried(regs))
    index = [int((p.start - first).n) for p in kept]

    def believed(known: list[int]) -> list[tuple[Any, ...]]:
        out, current, j = [], ref_key, 0
        chosen = [i for i in range(len(kept)) if i in known]
        for t in range(window):
            while j < len(chosen) and index[chosen[j]] <= t:
                current = _key(kept[chosen[j]], model)
                j += 1
            out.append(current)
        return out

    arrivals = [0] + [index[i] for i, p in enumerate(kept) if not p.expected and index[i] > 0]
    infos = []
    for a in sorted(set(arrivals)):
        known = [i for i, p in enumerate(kept) if p.expected or index[i] <= a]
        infos.append((a, believed(known)))
    names, j, name = [], 0, reference.name if reference is not None else _BASE
    for t in range(window):
        while j < len(kept) and index[j] <= t:
            name = kept[j].name
            j += 1
        names.append(name)
    plan = RegimePlan(ref_sol, ref_key, names, models, infos, pd.period_range(first, periods=window, freq=periods.freq))
    return ref_sol, plan


def _offset(period: Any, first: Any) -> int:
    """Periods from ``first`` to ``period`` (pandas periods, or numbers)."""
    gap = period - first
    return int(getattr(gap, "n", gap))


@dataclass(frozen=True, eq=False)
class Schedule:
    """Known-date regimes over the periods from the first solved, for the nonlinear solutions (perfect foresight,
    OccBin): the model of each period, the last holding for good after them."""

    keys: tuple[Any, ...]  # each period's regime, from the first period solved
    models: Mapping[Any, Any]  # each regime's model (the model with its parameters and equations)
    names: tuple[str, ...]  # each period's regime by name
    reference: Any  # the regime in force before the first period

    def key(self, s: int) -> Any:
        """The regime of period ``s`` (0 the first), the last one after the periods."""
        return self.keys[min(max(s, 0), len(self.keys) - 1)]

    def model(self, s: int) -> Any:
        return self.models[self.key(s)]


def regime_schedule(model: Any, regimes: Any, first: Any, freq: Any, length: int, what: str) -> tuple[Any, Schedule | None]:
    """For the nonlinear solutions (``what`` names them in refusals): the model of the regime in force before
    ``first`` (the reference: the model itself unless a regime starts earlier), and the schedule over ``length``
    periods from ``first`` and up to the last regime's start, or None when no period's model differs from the
    reference. ``freq=None``: periods numbered 1, 2, ... (``first`` a number). The regimes are known in advance here:
    with surprises among them, ``regime_schedules`` gives the schedule believed from each arrival."""
    reference, views = regime_schedules(model, regimes, first, freq, length, what)
    if len(views) > 1:
        surprises = [r.label for r in as_regimes(regimes) if not r.expected]
        raise ModelError(f"{what} takes regimes known in advance here; {', '.join(surprises)} is a surprise (expected=False)")
    return reference, views[0][1]


def regime_schedules(model: Any, regimes: Any, first: Any, freq: Any, length: int,
                     what: str) -> tuple[Any, list[tuple[int, Schedule | None]]]:
    """The reference model (as ``regime_schedule``) and, for each date news of the regimes arrives, the schedule
    believed from then on: ``[(0, schedule), (a, schedule), ...]``, ``a`` the period (0 the first) a surprise
    (``expected=False``) starts in. Each schedule has the regimes known in advance and the surprises that have
    started by its date, each believed to hold for good (a temporary surprise's end is a surprise of its own), as the
    first-order solution's ``regime_plan`` has them; a schedule is None where no period's model differs from the
    reference. With no surprises after the first period there is one."""
    regs = as_regimes(regimes)
    if not regs:
        return model, [(0, None)]
    if any(r.equations for r in regs) and not hasattr(model, "equations"):
        raise ModelError("regimes that replace equations need the model's equations")
    phases = _phases(regs, freq)
    last = first + (length - 1)
    for p in phases:
        if p.given and p.start > last:
            raise ModelError(f"the regime {p.name} starts in {p.start}, after the span ends ({last}): extend the span to it")
    carried = _carried(regs)

    def build(p: _Phase | None) -> Any:
        if p is None or _key(p, model) == ((), ()):
            return model
        return regime_model(model, dict(p.params), dict(p.equations), carried)

    before = [p for p in phases if p.start < first]
    ahead = [p for p in phases if p.start >= first]
    reference = before[-1] if before else None
    ref_key = _key(reference, model)
    models = {ref_key: build(reference)}
    where = [_offset(p.start, first) for p in ahead]
    total = max(length, where[-1] + 1 if where else 0)
    arrivals = sorted({0, *(w for w, p in zip(where, ahead) if not p.expected and w > 0)})
    views: list[tuple[int, Schedule | None]] = []
    for a in arrivals:
        known = [i for i, p in enumerate(ahead) if p.expected or where[i] <= a]
        keys, names, j = [], [], 0
        key, name = ref_key, reference.name if reference is not None else _BASE
        for s in range(total):
            while j < len(known) and where[known[j]] <= s:
                p = ahead[known[j]]
                key, name = _key(p, model), p.name
                if key not in models:
                    models[key] = build(p)
                j += 1
            keys.append(key)
            names.append(name)
        views.append((a, None if all(k == ref_key for k in keys) else Schedule(tuple(keys), models, tuple(names), ref_key)))
    return models[ref_key], views


def regime_solution(sol: Any, regimes: Sequence[Regime], label: str, freq: Any) -> Any:
    """The first-order solution of one regime on its own, held for good at its own steady state: what moments and
    variance decompositions of a model estimated with regimes are of, for that regime."""
    if label == _BASE:
        return sol
    regs = as_regimes(regimes)
    model = _base_model(sol)
    phases = [p for p in _phases(regs, freq) if p.given]
    found = [p for p in phases if p.name == label]
    if not found:
        raise ModelError(f"no regime called {label!r}; the regimes: {', '.join([_BASE, *(p.name for p in phases)])}")
    return _reference(sol, model, found[0], _carried(regs))


def _reference(sol: Any, model: Any, phase: _Phase, carried: frozenset[str] = frozenset()) -> Any:
    """The solution of the regime in force before the first period, at its own steady state."""
    m = regime_model(model, dict(phase.params), dict(phase.equations), carried)
    steady = sol.model.steady
    exogenous = {v: float(steady.values.get(v, 0.0)) for v in m.exogenous}
    ss = m.steady_state(exogenous, guess={v: float(steady.values[v]) for v in m.endogenous if v in steady.values})
    try:
        return _linear(m, ss, tuple(sol.shocks), None).solve()
    except ModelError as exc:
        raise ModelError(f"the regime in force before the first period ({phase.name}) has no unique stable solution: {exc}") from None


# ---------------------------------------------------------------------------------------------
# filtering and the likelihood


def tv_filter(y: np.ndarray, Ts: np.ndarray, cs: np.ndarray, Rs: np.ndarray, Qc: np.ndarray, Z: np.ndarray, d: np.ndarray,
              H: np.ndarray, a0: np.ndarray, P0: np.ndarray, presample: int = 0) -> dict[str, Any]:
    """The Kalman filter with time-varying transitions ``a(t) = c(t) + T(t) a(t-1) + R(t) e(t)`` from the state before
    the first period (mean ``a0``, variance ``P0``); missing observations (NaN) handled exactly. The log likelihood
    leaves out the first ``presample`` periods. Returns what ``occbin_filter.piecewise_smoother`` takes."""
    nT, ns = len(y), len(a0)
    a_f, P_f, a_p, P_p = np.empty((nT, ns)), np.empty((nT, ns, ns)), np.empty((nT, ns)), np.empty((nT, ns, ns))
    a, P = np.asarray(a0, float), np.asarray(P0, float)
    loglike = 0.0
    for t in range(nT):
        a = cs[t] + Ts[t] @ a
        P = Ts[t] @ P @ Ts[t].T + Rs[t] @ Qc @ Rs[t].T
        a_p[t], P_p[t] = a, P
        seen = np.isfinite(y[t])
        if seen.any():
            Zt = Z[seen]
            v = y[t, seen] - d[seen] - Zt @ a
            F = Zt @ P @ Zt.T + H[np.ix_(seen, seen)]
            F = 0.5 * (F + F.T)
            try:
                L = np.linalg.cholesky(F)
            except np.linalg.LinAlgError:
                raise ModelError(f"period {t + 1}: the prediction-error variance is not positive definite (give the observed "
                                 "series measurement errors, or more shocks)") from None
            X = np.linalg.solve(L, np.column_stack((v, Zt @ P)))
            w, W = X[:, 0], X[:, 1:]
            a = a + W.T @ w
            P = P - W.T @ W
            if t >= presample:
                loglike += -0.5 * (len(v) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + w @ w)
        P = 0.5 * (P + P.T)
        a_f[t], P_f[t] = a, P
    return {"loglike": float(loglike), "a_filt": a_f, "P_filt": P_f, "a_pred": a_p, "P_pred": P_p, "T": Ts, "c": cs, "R": Rs,
            "a0": np.asarray(a0, float), "P0": np.asarray(P0, float)}


def _matrices(plan: RegimePlan, system: Any, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The state-space transition of each period: the system's, with the model block from the period's rule."""
    k = plan.n
    ns, m = system.T.shape[0], system.R.shape[1]
    Ts, cs, Rs = np.repeat(system.T[None], T, 0), np.zeros((T, ns)), np.repeat(system.R[None], T, 0)
    for t, (P, c, Q) in enumerate(plan.period_rules(T)):
        Ts[t, :k, :k], cs[t, :k], Rs[t, :k] = P, c, Q
    if m != len(plan.sol.shocks):
        raise ModelError("the state-space form's shocks are not the solution's")
    return Ts, cs, Rs


def run_filter(sol: Any, system: Any, regimes: Any, y: np.ndarray, index: pd.Index, start: str = "auto",
               presample: int = 0) -> tuple[dict[str, Any], RegimePlan | None, Any]:
    """The regime filter on observations ``y`` (in data units, measurement constants still in): the filter's output, the
    plan (None when no regime changes in the sample: one regime throughout) and the reference solution."""
    periods = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)
    ref, plan = regime_plan(sol, regimes, periods, where="sample")
    if system.trend is not None and np.any(system.trend) or getattr(system.measurement, "trend_exprs", None):
        raise ModelError("the regime filter is for models without growth trends or deterministic trends in the measurement")
    a0, P0, D = system.start(start)
    if D.any():
        raise ModelError("the regime filter is for stationary models: the first regime has unit roots")
    T = len(y)
    if plan is None:
        Ts, cs, Rs = np.repeat(system.T[None], T, 0), np.zeros((T, len(a0))), np.repeat(system.R[None], T, 0)
    else:
        Ts, cs, Rs = _matrices(plan, system, T)
    return tv_filter(y, Ts, cs, Rs, system.Q, system.Z, system.d, system.H, a0, P0, presample), plan, ref


def regime_loglike(sol: Any, system: Any, regimes: Any, y: np.ndarray, index: Any, start: str = "auto", presample: int = 0) -> float:
    """The log likelihood with the regimes: the Kalman filter's, exactly, when no regime changes in the sample."""
    periods = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)
    _, plan = regime_plan(sol, regimes, periods, where="sample")
    if plan is None:
        return system.loglike_observed(y, index, start=start, presample=presample)
    return run_filter(sol, system, regimes, y, index, start, presample)[0]["loglike"]


@dataclass(frozen=True, eq=False)
class RegimeFilter:
    """The Kalman filter and smoother through known-date regimes: see ``pymetrix.models.regimes``."""

    smoothed: Any  # Dataset: the model's variables in levels, the observed series as fitted, and the smoothed shocks
    filtered: pd.DataFrame  # the model's variables in levels, from data up to each period
    regime: pd.Series  # the regime of each period
    loglike: float
    observables: tuple[str, ...] = ()
    regimes: tuple[Regime, ...] = ()
    rules: Any = field(default=None, repr=False)  # the time-varying rules and the smoothed states and shocks (decompose)

    def decompose(self, start: str = "first", orthogonal: bool = True) -> Any:
        """The smoothed path split into the shocks' contributions (a ``Decomposition``) through the regimes'
        time-varying rules ``x(t) = k(t) + P(t) x(t-1) + Q(t) e(t)``: each shock's contribution and the starting
        point's go through the rules, and the regimes' constants ``k(t)`` (a regime with another steady state, or a
        rule away from its own) are a part of their own, ``"regimes"``; the parts add up to the path. With no regime
        changing in the sample it is the linear decomposition (and the regimes' part zero). ``start`` as in
        ``LinearSolution.decompose``; with correlated shocks, of the shocks orthogonalised by Cholesky as impulse
        responses are (``orthogonal=False``: of the shocks themselves)."""
        from .decompose import path_decomposition

        r = self.rules
        if r is None:
            raise ModelError("this result has no smoother rules: run regime_filter again")
        return path_decomposition(r["T"], r["c"], r["R"], r["a_s"], r["e_s"], self.regime.index, r["variables"], r["observed"],
                                  r["shocks"], REGIMES, "Shock decomposition (known-date regimes)",
                                  r.get("factor") if orthogonal else None, start)

    def summary(self) -> Table:
        rows, index = [], self.regime.index
        values = list(self.regime)
        start = 0
        for t in range(1, len(values) + 1):
            if t == len(values) or values[t] != values[start]:
                rows.append([values[start], str(index[start]), str(index[t - 1]), t - start])
                start = t
        notes = [f"Log likelihood: {self.loglike:.4f}", f"Observed: {', '.join(self.observables)}",
                 "Regimes known in advance unless marked unexpected: " + (regimes_text(self.regimes) or "none")]
        return Table(["regime", "from", "to", "periods"], rows, title="Kalman filter with regimes", notes=notes)

    def __repr__(self) -> str:
        return f"<RegimeFilter {len(self.regime)} periods, {self.regime.nunique()} regimes, log likelihood {self.loglike:.3f}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def smooth(sol: Any, system: Any, regimes: Any, data: Any, span: Any = None, start: str = "auto", presample: int = 0,
           shocks: bool = True) -> RegimeFilter:
    """The regime filter and smoother of a solution's state-space form on data: see ``regime_filter``."""
    from ..data import Dataset
    from .occbin_filter import piecewise_smoother

    y, index = system.observations(data, span)
    run, plan, ref = run_filter(sol, system, regimes, y, index, start, presample)
    if ref is not sol:  # cannot happen: regimes in the sample start after its first period
        raise ModelError("the regime filter starts in the base regime")
    a_s, e_s = piecewise_smoother(run, system.Q)
    observed = list(system.variables)
    k = len(observed)
    levels = np.array([system.levels.get(v, 0.0) for v in observed])
    frame = pd.DataFrame(a_s[:, :k] + levels, index=index, columns=observed)
    fitted = system.d + a_s @ system.Z.T
    for j, o in enumerate(system.observables):
        frame[f"{o}_fitted"] = fitted[:, j]
    if shocks:
        for j, s in enumerate(system.shocks):
            frame[s] = e_s[:, j]
    if system.measurement is not None:
        frame = system.measurement.with_reported(frame)
    filtered = pd.DataFrame(run["a_filt"][:, :k] + levels, index=index, columns=observed)
    names = plan.names[:len(index)] if plan is not None else [_BASE] * len(index)
    rules = {"T": run["T"][:, :k, :k], "c": run["c"][:, :k], "R": run["R"][:, :k], "a_s": a_s[:, :k], "e_s": e_s,
             "variables": tuple(observed), "observed": tuple(sol.model.observed), "shocks": tuple(system.shocks)}
    corr = getattr(system, "shock_corr", None)
    if corr:
        from .covariance import shock_factor

        rules["factor"] = shock_factor(list(system.shocks), np.array([float(system.shock_sd.get(s, 1.0)) for s in system.shocks]), corr)
    return RegimeFilter(Dataset.from_frame(frame), filtered, pd.Series(names, index=index, name="regime"), float(run["loglike"]),
                        tuple(system.observables), as_regimes(regimes), rules)


@api("pymetrix:regime_filter")
def regime_filter(obj: Any, data: Any, regimes: Any = None, measurement: Any = None, shock_sd: Mapping[str, float] | None = None,
                  span: Any = None, start: str = "auto", presample: int = 0, shocks: Sequence[str] | None = None) -> RegimeFilter:
    """The Kalman filter and smoother of ``obj`` (a fitted model, a template, a Dynare or IRIS model, or a Model with
    ``measurement`` and ``shocks``) on ``data`` with known-date ``regimes`` (default: a fitted model's own, or a template's
    that start by the sample's end), the shocks correlated as the object's ``shock_corr``: the smoothed variables, the
    regime of each period and the log likelihood. See ``pymetrix.models.regimes``."""
    from .estimation import FittedModel
    from .occbin_filter import _setup

    model, names, guess, measurement, sds = _setup(obj, measurement, shock_sd, shocks)
    if isinstance(obj, FittedModel):
        sol = obj.solution
        regimes = obj.regimes if regimes is None else (*obj.regimes, *as_regimes(regimes))
        corr = obj._corr
    else:
        sol = model.linearize(model.steady_state(guess=dict(guess) or None), shocks=names).solve()
        if regimes is None and getattr(obj, "regimes", None):  # a template's own, those that start by the sample's end
            from .templates import _within

            regimes = _within(obj.regimes, data, span)[0] or None
        corr = getattr(obj, "shock_corr", None) or None
    system = sol.system(measurement, sds, shock_corr=corr)
    return smooth(sol, system, regimes, data, span, start, presample)
