"""Links between the entities of a model solved per entity: foreign variables and world aggregates, as in a global
VAR (GVAR) or a multi-country model.

    trade = Matrix([[0, .6, .4], [.5, 0, .5], [.3, .7, 0]], rows=["DE", "FR", "UK"], cols=["DE", "FR", "UK"])
    model = Model([Equation("y ~ y[-1] + ys", ...)], over="country",
                  links=[Link("ys", "y", trade),                      # each country's trade-weighted foreign output
                         Link("world", "y", {"DE": .4, "FR": .3, "UK": .3})])   # one aggregate, the same for every country
    sol = model.solve(panel, ("2025Q1", "2027Q4"))         # every country solved jointly with the others

A link is a variable in each entity's equations, ``name``, made from another variable, ``variable``, of the other
entities: ``name(i, t) = sum over j of w(i, j) variable(j, t)``. ``weights`` is a matrix with a row for each entity
(its partners' weights, usually trade shares summing to one, with zero on the diagonal: a ``Matrix``, a DataFrame or
``{entity: {partner: weight}}``), or a single set of weights for every entity (``{partner: weight}``, or a matrix with
one row): a world aggregate. Partners left out weigh nothing. Weights are fixed over time.

Before the span, links are worked out from the data, so their lags are there. Over the span, each entity is solved
with its links as given paths, the links are worked out again from every entity's solution, and so on until they no
longer change (by 1e-12 of their starting size; at most 500 rounds, each mixing the last twenty as Anderson acceleration
does). That is the joint solution of all the entities, as one stacked model would give: the tests check it against
one. The variable linked may also be exogenous: its links then come from the data, and settle in one round. A
scenario sets the variables links are made from, not the links themselves.

**Stochastic simulation** (``draws=``, backward-looking models). Each entity's errors are drawn as for an entity
solved alone (seed + the entity's position; ``errors``, ``shock_sd``), and each draw is the joint solution of all the
entities with that draw's errors as add factors, solved as above: a shock to one country reaches the others through
the links within the draw. A draw whose links do not settle is dropped (missing) and counted. The draws include the
links. Coefficient draws (``coefficients=True``), GARCH errors and first-order solutions are not drawn with links.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from .solvers import ModelError

__all__ = ["Link"]

_ROUNDS = 500
_TOL = 1e-12
_DEPTH = 20  # rounds the acceleration mixes


@dataclass(frozen=True)
class Link:
    """A variable each entity takes from the others: see the module docstring."""

    code_positional: ClassVar[int] = 1  # generated code gives the name by position: Link('ys', 'y', trade)

    name: str
    variable: str
    weights: Any

    def __post_init__(self) -> None:
        for what, value in (("name", self.name), ("variable", self.variable)):
            if not isinstance(value, str) or not value.isidentifier():
                raise SpecificationError(f"a link's {what} is a variable name, not {value!r}")
        if self.name == self.variable:
            raise SpecificationError(f"a link is a new variable: {self.name!r} cannot be made from itself")

    def matrix(self, entities: Sequence[Any]) -> np.ndarray:
        """The weights as an entities x entities array: row i gives entity i's link from each entity."""
        keys = [str(e) for e in entities]
        rows = _rows(self.weights)
        unknown = sorted({p for row in rows.values() for p in row if p not in keys} | ({r for r in rows if r is not None} - set(keys)))
        if unknown:
            raise ModelError(f"link {self.name}: weights name {', '.join(unknown)}, not entities of the data ({', '.join(keys)})")
        if None not in rows:
            missing = [k for k in keys if k not in rows]
            if missing:
                raise ModelError(f"link {self.name}: no weights for {', '.join(missing)} (give a row for each entity, or one row "
                                 "for all: an aggregate)")
        W = np.zeros((len(keys), len(keys)))
        for i, k in enumerate(keys):
            row = rows[None] if None in rows else rows[k]
            for j, partner in enumerate(keys):
                W[i, j] = float(row.get(partner, 0.0))
        if not np.all(np.isfinite(W)):
            raise ModelError(f"link {self.name}: the weights must be numbers")
        return W


def _rows(weights: Any) -> dict[Any, dict[str, float]]:
    """``{entity: {partner: weight}}``, or ``{None: {partner: weight}}`` for one set of weights for every entity."""
    from ..matrix import Matrix

    if isinstance(weights, Matrix):
        weights = weights.to_frame()
    if isinstance(weights, pd.Series):
        return {None: {str(k): float(v) for k, v in weights.items()}}
    if isinstance(weights, pd.DataFrame):
        if len(weights) == 1:
            return {None: {str(k): float(v) for k, v in weights.iloc[0].items()}}
        return {str(r): {str(c): float(v) for c, v in row.items()} for r, row in weights.iterrows()}
    if isinstance(weights, Mapping) and weights:
        if all(isinstance(v, Mapping) for v in weights.values()):
            return {str(r): {str(c): float(v) for c, v in row.items()} for r, row in weights.items()}
        if not any(isinstance(v, Mapping) for v in weights.values()):
            return {None: {str(k): float(v) for k, v in weights.items()}}
    raise SpecificationError("a link's weights are a Matrix, a DataFrame, {entity: {partner: weight}} or {partner: weight}")


def check_links(links: Sequence[Link], model: Any, data_names: Sequence[str]) -> None:
    """What a model's links need: names that differ, used by the equations and not determined by them, made from
    variables of the model or the data."""
    names = [link.name for link in links]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise SpecificationError(f"two links named {', '.join(twice)}")
    endogenous = set(model.endogenous)
    for link in links:
        if link.name in endogenous:
            raise ModelError(f"link {link.name}: the model determines {link.name}; a link is made from the other entities")
        if link.name not in model.exogenous:
            raise ModelError(f"link {link.name}: no equation uses it")
        if link.variable not in endogenous and link.variable not in data_names:
            raise ModelError(f"link {link.name}: {link.variable} is neither a variable of the model nor a series of the data")


def link_values(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``W @ values`` period by period (``values``: entities x periods); missing where a partner with weight is."""
    weighted = W != 0
    out = np.where(weighted, W, 0.0) @ np.nan_to_num(values)
    gaps = (weighted.astype(float) @ np.isnan(values).astype(float)) > 0
    out[gaps] = np.nan
    return out


def solve_linked(model: Any, links: Sequence[Link], data: Any, span: Any, scenario: Any, solver: Any, entity_params: Any,
                 entities: list[Any], over: str) -> tuple[dict[Any, Any], int]:
    """Each entity's solution of ``model`` (the model for one entity), with the links worked out jointly (see the
    module docstring): the solutions and the rounds taken."""
    for given in ([scenario] if not isinstance(scenario, Mapping) else list(scenario.values())):
        touched = _scenario_names(given) & {link.name for link in links}
        if touched:
            raise ModelError(f"a scenario cannot set the link {', '.join(sorted(touched))}: set the variable it is made from")
    frame = data.frame
    periods = frame.index.get_level_values(-1).unique().sort_values()
    matrices = {link.name: link.matrix(entities) for link in links}

    def table(column: str, frames: Mapping[Any, pd.DataFrame] | None = None) -> np.ndarray:
        """entities x periods of ``column``, from the data or the entities' solutions."""
        out = np.full((len(entities), len(periods)), np.nan)
        for i, key in enumerate(entities):
            source = frames[key] if frames is not None else frame.xs(key, level=over)
            if column in source:
                out[i] = source[column].reindex(periods).to_numpy(float)
        return out

    from ..data.sample import as_span

    ends = as_span(span)
    first, last = (pd.Period(str(p), freq=periods.freq) if p is not None else default for p, default in zip(ends, (periods[0], periods[-1])))
    inside = np.asarray((periods >= first) & (periods <= last))
    # the history from the data; over the span, a first guess carried forward from the last value known
    linked = {link.name: pd.DataFrame(link_values(matrices[link.name], table(link.variable))).ffill(axis=1).to_numpy(copy=True)
              for link in links}
    own = {key: data.entity(key, by=over) for key in entities}
    solutions: dict[Any, Any] = {}

    def solve(x: np.ndarray) -> np.ndarray:
        """Every entity solved with the links' span values ``x``; the links worked out again from the solutions."""
        for k, link in enumerate(links):
            linked[link.name][:, inside] = x[k]
        for key_i, key in enumerate(entities):
            series = {name: pd.Series(values[key_i], index=periods) for name, values in linked.items()}
            entity_model = model.with_parameters(entity_params[key]) if entity_params and key in entity_params else model
            entity_scenario = scenario.get(key) if isinstance(scenario, Mapping) else scenario
            try:
                solutions[key] = entity_model.solve(own[key].with_series(**series), span, entity_scenario, solver)
            except (ModelError, SpecificationError) as exc:
                raise ModelError(f"{over} {key}: {exc}") from exc
        frames = {key: s.frame for key, s in solutions.items()}
        out = np.stack([link_values(matrices[link.name], table(link.variable, frames))[:, inside] for link in links])
        if not np.all(np.isfinite(out)):
            missing = sorted({links[k].variable for k in range(len(links)) if not np.all(np.isfinite(out[k]))})
            raise ModelError(f"the links need {', '.join(missing)} for every entity over the span")
        return out

    # fixed-point rounds, accelerated (Anderson mixing of the last few rounds; a plain round when that misleads)
    x = np.stack([linked[link.name][:, inside] for link in links])
    if not np.all(np.isfinite(x)):
        raise ModelError("the links have no values before the span to start from: the data need the variables they are made of")
    xs: list[np.ndarray] = []
    fs: list[np.ndarray] = []
    best = np.inf
    scale = 1.0 + float(np.max(np.abs(x)))  # the size of the links where they start: changes are measured against it
    for rounds in range(1, _ROUNDS + 1):
        g = solve(x)
        f = (g - x).ravel()
        change = float(np.max(np.abs(f))) / scale
        if change < _TOL:
            return solutions, rounds
        if not np.all(np.isfinite(g)) or float(np.max(np.abs(g))) > 1e8 * scale:
            break  # running away: there is no joint solution to settle on
        if np.linalg.norm(f) > 10 * best:  # the mixing went astray: start it again from a plain round
            xs, fs = [], []
        best = min(best, float(np.linalg.norm(f)))
        xs.append(g.ravel())
        fs.append(f)
        xs, fs = xs[-_DEPTH - 1:], fs[-_DEPTH - 1:]
        step = g.ravel()
        if len(fs) > 1:
            dF = np.column_stack([fs[i + 1] - fs[i] for i in range(len(fs) - 1)])
            dG = np.column_stack([xs[i + 1] - xs[i] for i in range(len(xs) - 1)])
            gamma = np.linalg.lstsq(dF, f, rcond=None)[0]
            step = step - dG @ gamma
        x = step.reshape(x.shape)
    raise ModelError(f"the links did not settle in {rounds} rounds (last change {change:.3g} of their size): the entities' "
                     "contemporaneous effects on each other may be too strong for a joint solution; check the weights and the model")


def draw_linked(model: Any, links: Sequence[Link], data: Any, span: Any, scenario: Any, solver: Any, entity_params: Any,
                entities: list[Any], over: str, solutions: Mapping[Any, Any], draws: int, seed: int | None, errors: str,
                shock_sd: Mapping[str, float] | None) -> dict[Any, Any]:
    """Each entity's draws (``Simulations``) of a stochastic simulation of linked entities: every draw the joint solution
    of all the entities with that draw's errors as add factors (see the module docstring). ``solutions`` are the
    deterministic joint solutions."""
    from dataclasses import replace

    from . import stochastic
    from .model import _prepare
    from .scenario import Scenario

    D = int(draws)
    periods = data.frame.index.get_level_values(-1).unique().sort_values()
    plans: dict[Any, tuple[Any, list[str], np.ndarray, Any, tuple[str, ...], dict[str, Any]]] = {}
    for i, key in enumerate(entities):
        entity_model = model.with_parameters(entity_params[key]) if entity_params and key in entity_params else model
        given = scenario.get(key) if isinstance(scenario, Mapping) else scenario
        sc = given or Scenario(name="baseline")
        frame = solutions[key].frame
        own = data.entity(key, by=over).with_series(**{link.name: frame[link.name].reindex(periods) for link in links})
        prep = _prepare(entity_model, own, span, sc)
        positions = np.flatnonzero(prep.rows)
        fitted = stochastic._fitted(entity_model)
        names = [e.name for e in prep.eqs if e.stochastic]
        garch = [n for n in names if n in fitted and stochastic._garch(fitted[n])]
        if garch:
            raise SpecificationError(f"{', '.join(garch)} have GARCH variance: drawing it with links is not available; "
                                     "solve without links, or without draws")
        E, info, _ = stochastic._errors(entity_model, names, len(positions), D, errors, shock_sd,
                                        stochastic._rng(None if seed is None else seed + i))
        where = {e.name: j for j, e in enumerate(prep.eqs)}
        index = prep.full[positions]
        base = {n: prep.adjust[where[n]][positions] for n in names}
        variables = tuple(dict.fromkeys([*entity_model.endogenous, *sc.free, *(link.name for link in links)]))
        plans[key] = (sc, names, E, (index, base), variables, {**info, "seed": None if seed is None else seed + i})
    values = {key: np.full((D, len(plan[3][0]), len(plan[4])), np.nan) for key, plan in plans.items()}
    failed = np.zeros(D, dtype=bool)
    for d in range(D):
        drawn = {}
        for key, (sc, names, E, (index, base), _, _) in plans.items():
            adjust = {n: pd.Series(base[n] + E[d, :, k], index=index) for k, n in enumerate(names)}
            drawn[key] = replace(sc, adjust={**sc.adjust, **adjust})
        try:
            joint, _ = solve_linked(model, links, data, span, drawn, solver, entity_params, entities, over)
        except ModelError:
            failed[d] = True
            continue
        for key, (_, _, _, (index, _), variables, _) in plans.items():
            values[key][d] = joint[key].frame.reindex(index)[list(variables)].to_numpy(float)
    if failed.all():
        raise ModelError("no draw of the stochastic simulation could be solved jointly: check the deterministic solution, "
                         "and the size of the errors (shock_sd)")
    out = {}
    for key, (_, _, _, (index, _), variables, info) in plans.items():
        info = {**info, "method": "linked entities: errors as add factors, each draw the joint solution of all the entities",
                "draws": D, "failed": int(failed.sum())}
        out[key] = stochastic.Simulations(values[key], variables, index, info)
    return out


def _scenario_names(scenario: Any) -> set[str]:
    if scenario is None:
        return set()
    names: set[str] = set()
    for field_name in ("set", "shift", "fix", "hold"):
        names |= set(getattr(scenario, field_name, None) or {})
    return names | set(getattr(scenario, "free", ()) or ())
