"""Parameter sweeps and scenario lists: many runs of a model as one step, side by side, optionally in parallel.

    s = sweep(nk, {"phi_pi": [1.2, 1.5, 2.0], "kappa": [0.05, 0.1]}, "irf", periods=20)  # every combination
    s = sweep(nk, {"phi_pi": [1.2, 1.5], "kappa": [0.05, 0.1]}, "moments", grid="zip")     # pairwise
    s.table("y", "ei", horizon=4)      # one value per run: the response of y to ei after 4 periods
    s.plot("y", "ei")                  # the responses of all runs, overlaid
    sweep(nk, {"phi_pi": [1.2, 3.0]}, response="y", shock="epi")   # plot() (and the app's graph) shows this pair first
    s[{"phi_pi": 1.5, "kappa": 0.1}]   # one run's result

    c = scenarios(fit, {"baseline": Scenario(), "tight": Scenario(shift={"ei": {"2025Q1": 1.0}})}, ("2025Q1", "2027Q4"))
    c.table("y"); c.difference("tight", "baseline"); c.plot(["y", "pi"])
    c.impact(["y", "pi"], at="average")   # a scenario per row: how far each is from the baseline
    c.plot(["y", "pi"], deviations=True)  # each scenario less the baseline
    Report.from_template("policy_brief", {"Monetary policy options": c})
    c = compare_solutions(baseline=sol1, tight=sol2)    # solutions already computed (Model > Solve model)

``what`` is what each run of a sweep computes: ``"irf"``, ``"moments"``, ``"fevd"``, ``"steady_state"``,
``"solution"``, or a function of the model with the run's values (``lambda m: m.solve().irf(8)``). Parameters are
the model's calibrated parameters, or shock sizes as ``sd_<shock>``. A run that fails (no steady state, no unique
stable solution) is kept with its error rather than stopping the others.

``workers=n`` runs the sweep or scenario list in ``n`` processes. Each process starts by importing pymetrix (about
a second), so this pays for runs that each take a while (estimation, long simulations); templates, models,
solutions and scenarios can all be sent to other processes, but a function given as ``what`` must be defined at
the top level of a module (not a lambda) to run in parallel.
"""

from __future__ import annotations

import itertools
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ._api import api
from .errors import NotFoundError, SpecificationError
from .output import Table

__all__ = ["ScenarioSet", "SweepResult", "compare_solutions", "scenarios", "sweep"]

_WHAT = ("irf", "moments", "fevd", "steady_state", "solution")


# --- running jobs, in this process or several -----------------------------------------------------------

def _run_all(job: Callable[..., Any], args: list[tuple], workers: int | None) -> list[tuple[Any, str | None]]:
    """(result, error) for each job's arguments; in ``workers`` processes when more than one."""
    if not workers or workers <= 1 or len(args) <= 1:
        from .threads import blas_threads

        from .progress import report

        out = []
        with blas_threads():
            for k, a in enumerate(args):
                report(k / len(args), f"run {k + 1} of {len(args)}")  # outside the guard: Cancel stops the sweep
                out.append(_guarded(job, *a))
        return out
    try:
        pickle.dumps((job, args[0]))
    except Exception as exc:  # noqa: BLE001 - say what cannot go to another process
        raise SpecificationError(f"these runs cannot be sent to other processes ({exc}); use workers=1, or define the function "
                         "at the top level of a module rather than as a lambda") from None
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(args))) as pool:
        return list(pool.map(_guarded, [job] * len(args), *zip(*args)))


def _guarded(job: Callable[..., Any], *args: Any) -> tuple[Any, str | None]:
    try:
        return job(*args), None
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop the others
        return None, f"{type(exc).__name__}: {exc}"


# --- parameter sweeps -----------------------------------------------------------------------------------

def _with_values(obj: Any, values: Mapping[str, float]) -> Any:
    """A template, Dynare model or model with some parameters (and shock sizes, sd_<shock>) set to ``values``."""
    from .models.model import Model

    model = obj if isinstance(obj, Model) else getattr(obj, "model", None)
    if not isinstance(model, Model):
        raise TypeError(f"sweep a template, a Dynare or IRIS model, or a Model, not {type(obj).__name__}")
    structural = {k: v for k, v in values.items() if not k.startswith("sd_")}
    sizes = {k[3:]: v for k, v in values.items() if k.startswith("sd_")}
    known = set(model.parameters.names)
    unknown = [k for k in structural if k not in known]
    shock_sd = dict(getattr(obj, "shock_sd", {}) or {})
    unknown += [f"sd_{s}" for s in sizes if s not in shock_sd]
    if unknown:
        raise NotFoundError(f"{', '.join(unknown)} are not parameters or shock sizes of this model; it has "
                       f"{', '.join(sorted(known))}" + (f" and {', '.join('sd_' + s for s in shock_sd)}" if shock_sd else ""))
    new_model = Model(model.equations, params=model.parameters.with_values(structural), roles=model.roles) if structural else model
    if isinstance(obj, Model):
        return new_model
    changes: dict[str, Any] = {"model": new_model}
    if sizes:
        changes["shock_sd"] = {**shock_sd, **sizes}
    return replace(obj, **changes)


def _compute(obj: Any, what: Any, options: Mapping[str, Any]) -> Any:
    if callable(what):
        return what(obj)
    shock_sd = getattr(obj, "shock_sd", None) or None
    if what == "steady_state":
        return obj.steady_state()
    solution = obj.solve()
    if what == "solution":
        return solution
    if what == "irf":
        return solution.irf(int(options.get("periods", 20)), shock_sd=shock_sd)
    if what == "moments":
        return solution.moments(shock_sd, int(options.get("lags", 4)))
    if what == "fevd":
        return solution.fevd(int(options.get("periods", 20)), shock_sd)
    raise SpecificationError(f"what= is one of {', '.join(_WHAT)} or a function, not {what!r}")


def _sweep_job(obj: Any, values: dict[str, float], what: Any, options: dict[str, Any]) -> Any:
    return _compute(_with_values(obj, values), what, options)


@dataclass(frozen=True, eq=False)
class SweepResult:
    """The runs of a parameter sweep: see the module docstring."""

    names: tuple[str, ...]  # the parameters swept
    points: tuple[tuple[float, ...], ...]  # their values in each run
    results: tuple[Any, ...]  # each run's result (None where it failed)
    errors: tuple[str | None, ...]
    what: str = "irf"
    grid: str = "product"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, values: Mapping[str, float] | tuple) -> Any:
        point = tuple(values[n] for n in self.names) if isinstance(values, Mapping) else tuple(values)
        for p, r, e in zip(self.points, self.results, self.errors):
            if np.allclose(p, point):
                if e:
                    raise SpecificationError(f"the run at {dict(zip(self.names, p))} failed: {e}")
                return r
        raise NotFoundError(f"no run at {dict(zip(self.names, point))}")

    def label(self, point: tuple[float, ...]) -> str:
        return ", ".join(f"{n}={v:g}" for n, v in zip(self.names, point))

    def _index(self) -> pd.MultiIndex | pd.Index:
        if len(self.names) == 1:
            return pd.Index([p[0] for p in self.points], name=self.names[0])
        return pd.MultiIndex.from_tuples(self.points, names=list(self.names))

    def values(self, extract: Callable[[Any], Any]) -> pd.DataFrame:
        """``extract(result)`` for every run (a number, or a dict of numbers), one row per run; NaN where a run failed."""
        rows = []
        for r, e in zip(self.results, self.errors):
            out = extract(r) if e is None else np.nan
            rows.append(out if isinstance(out, Mapping) else {"value": out})
        return pd.DataFrame(rows, index=self._index())

    def table(self, *args: Any, horizon: int | None = None, statistic: str = "std. dev.") -> Table:
        """One value per run. For impulse responses: ``table(response, shock, horizon=h)`` (every horizon without
        ``horizon``); steady states: ``table(variable)``; moments: ``table(variable, statistic="std. dev.")`` (or
        "autocorr. 1" ...)."""
        if self.what == "irf":
            response, shock = args
            frame = self.values(lambda r: dict(enumerate(r[response, shock].to_numpy())))
            frame.columns = [f"h={c}" for c in frame.columns]
            if horizon is not None:
                frame = frame[[f"h={horizon}"]]
            title = f"Response of {response} to {shock}"
        elif self.what == "steady_state":
            (variable,) = args
            frame = self.values(lambda r: {variable: float(r[variable])})
            title = f"Steady state of {variable}"
        elif self.what == "moments":
            (variable,) = args
            frame = self.values(lambda r: {f"{statistic} of {variable}": float(r.summary().to_frame().loc[variable, statistic])})
            title = f"{statistic} of {variable}"
        else:
            raise SpecificationError("table() works for sweeps of irf, steady_state or moments; use values(function) for others")
        return _table(frame, title, self)

    def summary(self) -> Table:
        rows = [[self.label(p), "ok" if e is None else e] for p, e in zip(self.points, self.errors)]
        failed = sum(e is not None for e in self.errors)
        return Table(["run", "status"], rows, title=f"Sweep of {', '.join(self.names)} ({self.what})",
                     notes=[f"{len(self)} runs ({self.grid} of the values); {failed} failed" if failed else f"{len(self)} runs ({self.grid} of the values)"])

    def plot(self, response: str | None = None, shock: str | None = None) -> Any:
        """Impulse responses of every run, overlaid (by default the pair the sweep was given, else the first response
        and shock)."""
        from .figures import Figure, Line, Panel

        if self.what != "irf":
            raise SpecificationError("plot() draws sweeps of impulse responses; for others, plot the values in a table")
        first = next((r for r in self.results if r is not None), None)
        if first is None:
            raise SpecificationError("every run failed")
        response = response or self.options.get("response") or list(first.responses)[0]
        shock = shock or self.options.get("shock") or list(first.shocks)[0]
        lines = tuple(Line(r[response, shock], self.label(p)) for p, r in zip(self.points, self.results) if r is not None)
        return Figure((Panel(lines, title=f"{response}, shock to {shock}", xlabel="periods"),), title="Parameter sweep")

    def __repr__(self) -> str:
        return f"<SweepResult: {len(self)} runs of {self.what} over {', '.join(self.names)}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _table(frame: pd.DataFrame, title: str, sweep: SweepResult) -> Table:
    index = [sweep.label(p) for p in sweep.points]
    return Table([str(c) for c in frame.columns], frame.to_numpy().tolist(), index=index, title=title,
                 notes=[f"{sum(e is not None for e in sweep.errors)} runs failed (blank)"] if any(sweep.errors) else [])


def _points(values: Mapping[str, Sequence[float]], grid: str) -> list[tuple[float, ...]]:
    lists = [list(np.atleast_1d(np.asarray(v, dtype=float))) for v in values.values()]
    if grid == "product":
        return [tuple(float(x) for x in p) for p in itertools.product(*lists)]
    if grid == "zip":
        if len({len(v) for v in lists}) != 1:
            raise SpecificationError("grid='zip' pairs the values in order: give each parameter the same number of values")
        return [tuple(float(x) for x in p) for p in zip(*lists)]
    raise SpecificationError("grid is 'product' (every combination) or 'zip' (the values in order, pairwise)")


@api("pymetrix:sweep")
def sweep(obj: Any, values: Mapping[str, Sequence[float]], what: Any = "irf", grid: str = "product",
          workers: int | None = None, response: str | None = None, shock: str | None = None, **options: Any) -> SweepResult:
    """Run ``obj`` (a template, a Dynare or IRIS model, a Model) once for each set of parameter values: see the
    module docstring. ``response`` and ``shock``: the impulse response ``plot()`` draws by default."""
    if not values:
        raise SpecificationError("give the parameters to sweep and their values, e.g. {'phi_pi': [1.2, 1.5, 2.0]}")
    from .models.model import Model

    if isinstance(obj, Model) and not callable(what) and what != "steady_state":
        raise SpecificationError(f"a Model on its own has no shocks to solve with for {what!r}: sweep its template (or Dynare or "
                         "IRIS model), or give what= a function of the model")
    names = tuple(values)
    points = _points(values, grid)
    _with_values(obj, dict(zip(names, points[0])))  # unknown names fail now, not once per run
    runs = _run_all(_sweep_job, [(obj, dict(zip(names, p)), what, dict(options)) for p in points], workers)
    label = what if isinstance(what, str) else getattr(what, "__name__", "function")
    if response or shock:
        if what != "irf":
            raise SpecificationError("response= and shock= choose the impulse response to draw: for what='irf'")
        first = next((r for r, e in runs if e is None), None)
        if first is not None:
            if response and response not in first.responses:
                raise NotFoundError(f"no response {response!r}; the responses are {', '.join(first.responses)}")
            if shock and shock not in first.shocks:
                raise NotFoundError(f"no shock {shock!r}; the shocks are {', '.join(first.shocks)}")
        options.update({k: v for k, v in (("response", response), ("shock", shock)) if v})
    return SweepResult(names, tuple(points), tuple(r for r, _ in runs), tuple(e for _, e in runs), label, grid, dict(options))


# --- scenario lists -------------------------------------------------------------------------------------

def _scenario_job(source: Any, scenario: Any, span: Any, data: Any) -> Any:
    from .models.estimation import FittedModel

    if isinstance(source, FittedModel):
        return source.forecast(span, scenario, data=data)
    return source.solve(data, span, scenario)


@dataclass(frozen=True, eq=False)
class ScenarioSet:
    """Model paths under several named scenarios, compared with one of them, the baseline: see the module docstring.

    ``baseline`` is the scenario the others are measured against: the one given, else one named "baseline", else the
    first. ``deviations``, ``impact`` and ``plot(deviations=True)`` show each scenario less the baseline, which is
    what a policy brief reports (``Report.from_template("policy_brief", {...: set})``). ``with_loss`` sets a
    quadratic loss to rank the scenarios by (``losses``): as optimal control's, the sum over the periods solved and the
    variables weighted of discount^t * weight * (value - desired)^2."""

    solutions: Mapping[str, Any]  # name -> Solution (None where it failed)
    errors: Mapping[str, str | None] = field(default_factory=dict)
    baseline: str = ""  # the scenario the others are compared with ("": one named "baseline", else the first)
    loss: Mapping[str, float] = field(default_factory=dict)  # variable -> weight of its squared gap (with_loss)
    desired: Mapping[str, float] = field(default_factory=dict)  # variable -> the value its gap is measured from (0)
    discount: float = 1.0  # the loss's discount factor per period; 1: none

    def __post_init__(self) -> None:
        if self.baseline and self.baseline not in self.solutions:
            raise NotFoundError(f"no scenario {self.baseline!r} to use as the baseline; the scenarios are {', '.join(self.solutions)}")
        if not self.baseline and self.solutions:
            object.__setattr__(self, "baseline", "baseline" if "baseline" in self.solutions else next(iter(self.solutions)))

    def __getitem__(self, name: str) -> Any:
        if self.errors.get(name):
            raise SpecificationError(f"the scenario {name!r} failed: {self.errors[name]}")
        return self.solutions[name]

    @property
    def names(self) -> list[str]:
        return list(self.solutions)

    @property
    def alternatives(self) -> list[str]:
        """The scenarios other than the baseline that solved."""
        return [n for n, sol in self.solutions.items() if n != self.baseline and sol is not None]

    def with_baseline(self, name: str) -> ScenarioSet:
        """The same scenarios, compared with ``name``."""
        return replace(self, baseline=name)

    @api()
    def with_loss(self, loss: Mapping[str, float], desired: Mapping[str, float] | None = None, discount: float = 1.0) -> ScenarioSet:
        """The same scenarios with a loss to rank them by (``losses``, the window's Loss tab, the policy brief):
        ``loss`` the weight on each variable's squared gap (``{"pi": 1, "ygap": 0.5}``), ``desired`` the value each gap
        is measured from (``{"pi": 2}``; 0 otherwise), ``discount`` per period (1: none), as optimal control's."""
        weights = {str(k): float(v) for k, v in dict(loss).items()}
        goals = {str(k): float(v) for k, v in dict(desired or {}).items()}
        if not weights:
            raise SpecificationError("give the loss's weights, e.g. {'pi': 1, 'ygap': 0.5}")
        if any(w < 0 or not np.isfinite(w) for w in weights.values()):
            raise SpecificationError("the loss's weights are numbers of zero or more")
        if not 0 < float(discount) <= 1:
            raise SpecificationError(f"the discount factor is above 0 and at most 1; got {discount!r}")
        extra = sorted(set(goals) - set(weights))
        if extra:
            raise SpecificationError(f"desired values for {', '.join(extra)}, which are not in the loss")
        missing = sorted(set(weights) - set(self.variables))
        if missing:
            raise NotFoundError(f"no variable {', '.join(missing)} in the scenarios; they have {', '.join(self.variables[:12])}")
        return replace(self, loss=weights, desired=goals, discount=float(discount))

    def losses(self) -> Table:
        """Each scenario's loss (``with_loss``): a row per scenario that solved, a column per variable's part, the
        total, and the total less the baseline's (negative: better than the baseline)."""
        if not self.loss:
            raise SpecificationError("set the loss first: with_loss({'pi': 1, 'ygap': 0.5}, desired={'pi': 2})")
        rows = {}
        for name, sol in self.solutions.items():
            if sol is None:
                continue
            paths = self._paths(sol)
            weights = self.discount ** np.arange(len(paths))
            parts = {v: float(w * np.nansum(weights * (paths[v].to_numpy(float) - self.desired.get(v, 0.0)) ** 2))
                     for v, w in self.loss.items()}
            rows[name] = {**parts, "loss": sum(parts.values())}
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "scenario"
        if self.baseline in frame.index:
            frame["less baseline"] = frame["loss"] - frame.loc[self.baseline, "loss"]
        terms = " + ".join(f"{w:g}·({v}{f' − {self.desired[v]:g}' if self.desired.get(v) else ''})²" for v, w in self.loss.items())
        when = f", discounted by {self.discount:g} a period" if self.discount != 1 else ""
        return Table.from_frame(frame, title=f"Loss: the sum over the periods of {terms}{when}")

    @staticmethod
    def _paths(sol: Any) -> pd.DataFrame:
        return sol.solved().frame if hasattr(sol, "solved") else sol.frame

    @property
    def variables(self) -> list[str]:
        """The variables every scenario that solved has, in the first's order."""
        frames = [self._paths(s) for s in self.solutions.values() if s is not None]
        return [] if not frames else [c for c in frames[0].columns if all(c in f.columns for f in frames[1:])]

    def frame(self, variable: str) -> pd.DataFrame:
        """``variable`` in each scenario, side by side (the periods the scenarios solved)."""
        cols = {}
        for name, sol in self.solutions.items():
            if sol is None:
                continue
            if variable not in sol.frame.columns:
                raise NotFoundError(f"no variable {variable!r} in scenario {name!r}")
            cols[name] = self._paths(sol)[variable]
        return pd.DataFrame(cols)

    def table(self, variable: str) -> Table:
        return Table.from_frame(self.frame(variable), title=f"{variable} by scenario")

    def difference(self, scenario: str, baseline: str | None = None, variables: Sequence[str] | None = None) -> Table:
        """Each variable in ``scenario`` less ``baseline`` (the set's baseline if not given), period by period."""
        baseline = baseline or self.baseline
        a, b = self[scenario], self[baseline]
        part_a, part_b = self._paths(a), self._paths(b)
        names = list(variables) if variables else [c for c in part_a.columns if c in part_b.columns]
        diff = part_a[names] - part_b[names]
        return Table.from_frame(diff, title=f"{scenario} less {baseline}")

    def deviations(self, variable: str, percent: bool = False) -> pd.DataFrame:
        """``variable`` in each scenario less the baseline, side by side; ``percent``: as a percentage of the
        baseline (for levels, not rates)."""
        both = self.frame(variable)
        if self.baseline not in both.columns:
            raise SpecificationError(f"the baseline {self.baseline!r} failed: {self.errors.get(self.baseline)}")
        base = both[self.baseline]
        others = both[[n for n in both.columns if n != self.baseline]]
        return others.sub(base, axis=0).div(base, axis=0) * 100 if percent else others.sub(base, axis=0)

    def impact(self, variables: Sequence[str] | str | None = None, at: Any = "average", percent: bool = False) -> Table:
        """A scenario per row and a variable per column: how far each scenario is from the baseline. ``at``:
        ``"average"`` over the periods solved, ``"end"`` (the last period), ``"peak"`` (the largest departure, with
        its sign) or a period."""
        names = [variables] if isinstance(variables, str) else list(variables or self.variables[:8])
        if not self.alternatives:
            raise SpecificationError("no scenario besides the baseline solved: nothing to compare")
        cols = {}
        for v in names:
            dev = self.deviations(v, percent)
            if at == "average":
                cols[v] = dev.mean()
            elif at == "end":
                cols[v] = dev.iloc[-1]
            elif at == "peak":
                cols[v] = dev.apply(lambda c: c.loc[c.abs().idxmax()] if c.notna().any() else np.nan)
            else:
                key = pd.Period(str(at), freq=dev.index.freq) if isinstance(dev.index, pd.PeriodIndex) else at
                if key not in dev.index:
                    raise NotFoundError(f"no period {at!r} among those solved ({dev.index[0]} to {dev.index[-1]})")
                cols[v] = dev.loc[key]
        frame = pd.DataFrame(cols)
        frame.index.name = "scenario"
        when = {"average": "on average", "end": "at the end", "peak": "at the largest departure"}.get(str(at), f"in {at}")
        unit = "per cent of the baseline" if percent else "difference"
        return Table.from_frame(frame, title=f"Each scenario less {self.baseline}, {when} ({unit})")

    def plot(self, variables: Sequence[str] | str | None = None, deviations: bool = False) -> Any:
        """Each variable in a panel, a line per scenario (the baseline first); ``deviations``: each scenario less the
        baseline, with a zero line."""
        from .figures import Figure, Line, Panel

        names = [variables] if isinstance(variables, str) else list(variables or self.variables[:6])
        panels = []
        for v in names:
            frame = self.deviations(v) if deviations else self.frame(v)
            order = [n for n in frame.columns if n == self.baseline] + [n for n in frame.columns if n != self.baseline]
            panels.append(Panel(tuple(Line(frame[n], n) for n in order), title=f"{v} less {self.baseline}" if deviations else v))
        title = f"Scenarios less {self.baseline}" if deviations else "Scenarios"
        return Figure(tuple(panels), title=title, columns=min(2, len(panels)))

    def summary(self) -> Table:
        rows = [[name, "baseline" if name == self.baseline else "", "ok" if not self.errors.get(name) else self.errors[name]]
                for name in self.solutions]
        if self.loss:
            ranked = self.losses().to_frame()["loss"]
            return Table(["scenario", "role", "status", "loss"],
                         [[*r, float(ranked[r[0]]) if r[0] in ranked.index else float("nan")] for r in rows],
                         title=f"{len(self.solutions)} scenarios, compared with {self.baseline}")
        return Table(["scenario", "role", "status"], rows, title=f"{len(self.solutions)} scenarios, compared with {self.baseline}")

    def __repr__(self) -> str:
        return f"<ScenarioSet: {', '.join(self.solutions)}; baseline {self.baseline}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


@api("pymetrix:scenarios")
def scenarios(source: Any, scenarios: Mapping[str, Any] | Sequence[Any], span: tuple[Any, Any], data: Any = None,
              workers: int | None = None, baseline: str | None = None) -> ScenarioSet:
    """Solve ``source`` (an estimated model, forecast from the end of its data, or a rational-expectations solution
    with ``data``) under each scenario over ``span``: see the module docstring. ``baseline``: the scenario the others
    are compared with (default: one named "baseline", else the first)."""
    items = list(scenarios.items()) if isinstance(scenarios, Mapping) else [(getattr(s, "name", f"scenario {i + 1}"), s)
                                                                             for i, s in enumerate(scenarios)]
    if not items:
        raise SpecificationError("give the scenarios to compare")
    if len({n for n, _ in items}) != len(items):
        raise SpecificationError("give each scenario its own name")
    runs = _run_all(_scenario_job, [(source, s, span, data) for _, s in items], workers)
    return ScenarioSet({n: r for (n, _), (r, _) in zip(items, runs)}, {n: e for (n, _), (_, e) in zip(items, runs)},
                       baseline=baseline or "")


@api("pymetrix:compare_solutions")
def compare_solutions(**solutions: Any) -> ScenarioSet:
    """Solutions already computed (e.g. by Model > Solve model, one per scenario), side by side; the baseline is the
    one named "baseline", else the first (``.with_baseline(name)`` for another)."""
    if len(solutions) < 2:
        raise SpecificationError("give at least two solutions to compare")
    return ScenarioSet(dict(solutions), {n: None for n in solutions})
