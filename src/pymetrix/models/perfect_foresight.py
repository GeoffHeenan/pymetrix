"""Perfect-foresight simulation: all periods solved at once (stacked-time Newton).

With leads in the model (``x[+1]``), period t depends on later periods, so the
periods cannot be solved one after another. The unknowns of every period are
stacked into one vector and solved jointly by Newton's method with a sparse
Jacobian (as in Dynare's and IRIS's deterministic simulations). Before the span,
variables take their data values; beyond it, leads take terminal values: the
last simulated value (``constant``), the data, or the steady state.

Known-date regimes (``regimes=``, see ``pymetrix.models.regimes``) switch the
parameters and equations of the stacked system at their dates: each period's
equations are those of the regime in force then, the nonlinear model itself (not
linearised); the terminal steady state is the last regime's. A surprise regime (``expected=False``) is unknown until
it starts: the path believed before it is kept until then, and the model is solved again from the state reached, with
the surprise believed to hold for good (several, one after another), as unanticipated shocks are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..expr import Name, diff, evaluate
from .solvers import ModelError

if TYPE_CHECKING:
    from ..data import Dataset
    from .model import Model
    from .scenario import Scenario
    from .solution import Solution

_ZERO_TEXT = "0"


def simulate_stacked(model: Model, data: Dataset, span: tuple[Any, Any], sc: Scenario, solver: Any, regimes: Any = None) -> Solution:
    if not regimes:
        return _stacked(model, data, span, sc, solver, None)
    import pandas as pd

    from ..data.sample import as_span
    from .regimes import regime_schedules

    freq = data.frame.index.freq
    first_period, last_period = (pd.Period(str(p), freq=freq) for p in as_span(span))
    reference, views = regime_schedules(model, regimes, first_period, freq, int((last_period - first_period).n) + 1,
                                        "Perfect foresight")
    if len(views) > 1:  # surprises: solved again from each one's start
        return _surprise_regimes(model, reference, views, data, first_period, last_period, sc, solver, regimes)
    schedule = views[0][1]
    if schedule is None:  # one regime throughout: that regime's model, solved as any other
        return _with_spells(_stacked(reference, data, span, sc, solver, None), None, first_period, regimes)
    return _with_spells(_stacked(model, data, span, sc, solver, schedule), schedule, first_period, regimes)


def _surprise_regimes(model: Model, reference: Any, views: list[tuple[int, Any]], data: Dataset, first: Any, last: Any,
                      sc: Scenario, solver: Any, regimes: Any) -> Solution:
    """Perfect foresight with surprise regimes (``expected=False``): the path believed at the first period, kept until
    the first surprise starts; from the state reached then, the path believed with that surprise (held for good), kept
    until the next; and so on, as ``simulate_with_surprises`` does for unanticipated shocks."""
    from dataclasses import replace

    import pandas as pd

    from .regimes import Schedule

    periods = pd.period_range(first, last, freq=first.freq)
    working, solution, iterations, names, kept = data, None, 0, [], []
    for k, (a, schedule) in enumerate(views):
        stop = views[k + 1][0] if k + 1 < len(views) else len(periods)
        segment_model, shifted = reference, None
        if schedule is not None:
            keys = schedule.keys[a:] if a < len(schedule.keys) else (schedule.key(a),)
            if all(key == keys[0] for key in keys):
                segment_model = schedule.models[keys[0]]
            else:
                segment_model = model
                shifted = Schedule(tuple(keys), schedule.models, tuple(schedule.names[a:]) or (schedule.names[-1],), schedule.key(a - 1))
        told = schedule if schedule is not None else next((v for _, v in views[k + 1:] if v is not None), None)  # the same before the next
        names += [told.names[min(t, len(told.names) - 1)] if told is not None else "base" for t in range(a, stop)]
        solution = _stacked(segment_model, working, (str(periods[a]), str(periods[-1])), sc, solver, shifted)
        iterations += solution.iterations
        kept.append((solution, periods[a], periods[stop - 1]))
        if k + 1 < len(views):
            working = _keep(working, solution, periods[a], periods[stop - 1], data)
    assert solution is not None
    provenance = solution.provenance
    if provenance is not None:
        from .regimes import as_regimes, regimes_text

        method = dict(provenance.method) if isinstance(provenance.method, dict) else {"solver": provenance.method}
        spells: list[dict[str, str]] = []
        for t, name in enumerate(names):
            if spells and spells[-1]["regime"] == name:
                spells[-1]["to"] = str(periods[t])
            else:
                spells.append({"regime": name, "from": str(periods[t]), "to": str(periods[t])})
        method["regimes"] = spells
        method["known regimes"] = regimes_text(as_regimes(regimes))
        method["surprise regimes at"] = [str(periods[a]) for a, _ in views[1:]]
        provenance = replace(provenance, sample=(str(periods[0]), str(periods[-1])), nobs=len(periods), method=method)
    return replace(solution, scenario=sc, iterations=iterations, provenance=provenance, convergence=_kept_convergence(kept))


def _keep(working: Dataset, solution: Solution, start: Any, end: Any, data: Dataset) -> Dataset:
    """``working`` with the solution's values from ``start`` to ``end``: the history the next solve starts from."""
    from ..data import Dataset as _Dataset

    frame = working.frame.reindex(solution.frame.index)
    keep = (solution.frame.index >= start) & (solution.frame.index <= end)
    for c in solution.frame.columns:
        if c not in frame.columns:
            frame[c] = np.nan
        frame.loc[keep, c] = solution.frame.loc[keep, c]
    frame.index.name = data.frame.index.name
    return _Dataset.from_frame(frame)


def _stacked(model: Model, data: Dataset, span: tuple[Any, Any], sc: Scenario, solver: Any, schedule: Any) -> Solution:
    """The stacked-time Newton solution over ``span``; ``schedule``: each period's regime (``regimes.Schedule``), or
    None for the model's own equations throughout."""
    from .model import ModelEquation, _prepare, _solution

    prep = _prepare(model, data, span, sc)
    arrays, eqs, unknowns = prep.arrays, prep.eqs, prep.unknowns
    positions = np.flatnonzero(prep.rows)
    first, last = int(positions[0]), int(positions[-1])
    T, n = len(positions), len(unknowns)
    uidx = {u: j for j, u in enumerate(unknowns)}
    nrows = len(prep.full)

    # terminal values for leads beyond the span
    terminal: dict[str, float] = {}
    growth: dict[str, float] = {}  # terminal="growth": beyond the span, the last value moving on the balanced growth path
    if solver.terminal in ("steady_state", "growth"):
        final = model if schedule is None else schedule.model(T - 1)  # with regimes, the last one's
        exo = {v: float(arrays[v][last]) for v in final.exogenous if v in arrays and np.isfinite(arrays[v][last])}
        guess = {u: float(arrays[u][last]) for u in unknowns if np.isfinite(arrays[u][last])}
        if solver.terminal == "steady_state":
            terminal = dict(final.steady_state(exo, guess).values)
        else:
            growth = {k: float(v) for k, v in (final.steady_state(exo, guess, growth=True).growth or {}).items()}

    # starting values: the data where there are any, else carried forward
    X = np.empty((T, n))
    for j, u in enumerate(unknowns):
        values = arrays[u][positions].copy()
        before = arrays[u][first - 1] if first > 0 and np.isfinite(arrays[u][first - 1]) else np.nan
        carry = before if np.isfinite(before) else 1.0
        for k in range(T):
            if np.isfinite(values[k]):
                carry = values[k]
            else:
                values[k] = carry
        X[:, j] = values

    # derivative of each equation with respect to each unknown at each lag or lead
    def derivatives(equations: Any) -> list[Any]:
        out = []
        for e in equations:
            terms = []
            for ref in sorted(e.residual.refs(), key=lambda r: (r.id, r.lag)):
                if ref.id in uidx:
                    d = diff(e.residual, Name(ref.id, ref.lag))
                    if str(d) != _ZERO_TEXT:
                        terms.append((ref.id, ref.lag, d))
            out.append(terms)
        return out

    if prep.varying and _partial_holds(sc):
        raise ModelError("perfect foresight takes partial fixes (Scenario(partial=...)) but not partial holds: hold the "
                         "variable over the whole span, or solve a backward-looking model period by period (Newton)")
    fast = None if schedule is not None or prep.varying else _compiled(model, prep, positions, first, last, T, n, uidx, nrows, terminal, solver, growth)
    if schedule is None:
        derivs = derivatives(eqs) if fast is None else []  # compiled: the derivatives are taken there
        eqs_at, derivs_at = [eqs] * T, [derivs] * T
    else:  # each period's equations, those of its regime
        by_key = {}
        for key, m in schedule.models.items():
            own = {e.name: e for e in m.model_equations}
            chosen = [own.get(e.name) for e in eqs]
            if any(c is None or c.dependent != e.dependent for c, e in zip(chosen, eqs)):
                raise ModelError("a regime's equations do not match the model's by name and the variable each determines")
            by_key[key] = (chosen, derivatives(chosen))
        eqs_at = [by_key[schedule.key(k)][0] for k in range(T)]
        derivs_at = [by_key[schedule.key(k)][1] for k in range(T)]
    adjust = prep.adjust
    if prep.varying:  # partial fixes: in a period a variable is fixed, its equation is "x - value = 0" (value as add factor)
        adjust = [a.copy() for a in prep.adjust]
        eqs_at, derivs_at = list(eqs_at), list(derivs_at)
        pins: dict[str, tuple[Any, Any]] = {}
        for k, t in enumerate(positions):
            solved = {e.name for e in prep.at(int(t))[0]}
            if len(solved) == len(eqs):
                continue
            eqs_at[k], derivs_at[k] = list(eqs_at[k]), list(derivs_at[k])
            for i, e in enumerate(eqs):
                if e.name in solved:
                    continue
                if e.name not in pins:
                    pin = ModelEquation(e.name, e.dependent, Name(e.dependent, 0), False)
                    pins[e.name] = (pin, derivatives([pin])[0])
                eqs_at[k][i], derivs_at[k][i] = pins[e.name]
                adjust[i][t] = arrays[e.dependent][t]

    def locate(name: str, pos: int) -> tuple[str, Any]:
        """Where a value comes from: ("x", (k, j)) an unknown, ("value", v) a number, ("missing", None)."""
        j = uidx.get(name)
        if j is not None and first <= pos <= last:
            return "x", (pos - first, j)
        if j is not None and pos > last:
            if solver.terminal == "constant":
                return "x", (T - 1, j)
            if solver.terminal == "growth":
                return "x+", ((T - 1, j), growth.get(name, 0.0) * (pos - last))
            if solver.terminal == "steady_state":
                return ("value", terminal[name]) if name in terminal else ("missing", None)
            value = arrays[name][pos] if pos < nrows else np.nan
            return ("value", value) if np.isfinite(value) else ("missing", None)
        if 0 <= pos < nrows:
            value = arrays[name][pos]
            return ("value", value) if np.isfinite(value) else ("missing", None)
        if solver.terminal == "steady_state" and name in terminal:
            return "value", terminal[name]
        return "missing", None

    def resolver(t: int, Xc: np.ndarray):
        def resolve(name: str, lag: int) -> float:
            where, what = locate(name, t + lag)
            if where == "x":
                return Xc[what]
            if where == "x+":
                return Xc[what[0]] + what[1]
            if where == "value":
                return float(what)
            if solver.terminal == "data" and name in uidx and t + lag > last:
                raise ModelError(f"{name}[{lag:+d}] beyond {prep.full[last]} is not in the data; use terminal='constant' or 'steady_state'")
            raise ModelError(f"{prep.full[t] if t < nrows else t}: no value for {name}[{lag:+d}]")
        return resolve

    def residuals(Xc: np.ndarray) -> np.ndarray:
        F = np.empty(T * len(eqs))
        for k, t in enumerate(positions):
            resolve = resolver(int(t), Xc)
            for i, e in enumerate(eqs_at[k]):
                F[k * len(eqs) + i] = evaluate(e.residual, resolve) - adjust[i][t]
        return F

    def jacobian(Xc: np.ndarray) -> sparse.csr_matrix:
        rows_, cols_, vals_ = [], [], []
        for k, t in enumerate(positions):
            resolve = resolver(int(t), Xc)
            for i, terms in enumerate(derivs_at[k]):
                for name, lag, d in terms:
                    where, what = locate(name, int(t) + lag)
                    if where not in ("x", "x+"):
                        continue
                    kk, j = what if where == "x" else what[0]
                    rows_.append(k * len(eqs) + i)
                    cols_.append(kk * n + j)
                    vals_.append(float(evaluate(d, resolve)))
        return sparse.csr_matrix((vals_, (rows_, cols_)), shape=(T * len(eqs), T * n))

    if fast is not None:  # the same residuals and Jacobian, compiled and vectorised over the periods
        residuals, jacobian = fast  # noqa: F811

    # The line search and the chord steps judge a step by each equation's residual relative to the size of its terms
    # (the largest |derivative x value| in its row of the Jacobian, at least 1). In a model with equations in levels
    # of thousands (FRB/US's nominal incomes and wealth) a Newton step leaves second-order residuals that are large in
    # absolute terms but tiny relative to those levels, and judging by the raw norm would cut such steps to nothing.
    # Convergence is still judged on the residuals as written.
    def scales(J: sparse.spmatrix, Xc: np.ndarray) -> np.ndarray:
        terms = abs(sparse.csr_matrix(J).multiply(np.abs(Xc.ravel())[None, :]))
        out = terms.max(axis=1).toarray().ravel()
        return np.where(np.isfinite(out), np.maximum(1.0, out), 1.0)

    F = residuals(X)
    iterations = 0
    for iterations in range(1, solver.max_iter + 1):
        if not np.all(np.isfinite(F)):
            raise ModelError("a residual is not finite; check the starting values and the data")
        if np.max(np.abs(F)) < solver.tol:
            break
        J = jacobian(X)
        lu, scale = _factorise(J), scales(J, X)

        step = lu.solve(-F).reshape(T, n)
        if not np.all(np.isfinite(step)):
            raise ModelError("the stacked system is singular; does each equation determine its variable?")
        size = np.linalg.norm(F / scale)
        for _ in range(30):  # backtrack until the residuals shrink
            trial = X + step
            F_trial = residuals(trial)
            if np.all(np.isfinite(F_trial)) and np.linalg.norm(F_trial / scale) < size:
                break
            step /= 2
        X, F = trial, F_trial
        for _ in range(_CHORD_STEPS):  # more steps with the same factorisation, while each cuts the residuals tenfold
            size = np.linalg.norm(F / scale)
            if not np.all(np.isfinite(F)) or np.max(np.abs(F)) < solver.tol:
                break
            trial = X + lu.solve(-F).reshape(T, n)
            F_trial = residuals(trial)
            if not (np.all(np.isfinite(F_trial)) and np.linalg.norm(F_trial / scale) < _CHORD * size):
                break
            X, F = trial, F_trial
    else:
        if np.max(np.abs(F)) >= solver.tol:
            exc = ModelError(f"perfect foresight did not converge in {solver.max_iter} iterations (largest residual {np.max(np.abs(F)):.3g})")
            exc.convergence = _convergence(prep, positions, eqs_at, F, iterations, solver.tol)  # type: ignore[attr-defined]
            raise exc
    for j, u in enumerate(unknowns):
        arrays[u][positions] = X[:, j]
    return _solution(model, data, sc, solver, prep, iterations, _convergence(prep, positions, eqs_at, F, iterations, solver.tol))


def _partial_holds(sc: Scenario) -> bool:
    from .model import _partial_names

    return any(v in sc.hold for v in _partial_names(sc))


def _convergence(prep: Any, positions: np.ndarray, eqs_at: Any, F: np.ndarray, iterations: int, tol: float) -> Any:
    """``Solution.convergence`` of the stacked system: each period's largest absolute residual and its equation; the
    iterations are the whole system's, the same in every row."""
    from .model import convergence_frame

    T = len(positions)
    size = np.abs(F).reshape(T, -1) if T else np.zeros((0, 0))
    size = np.where(np.isfinite(size), size, np.inf)
    worst = size.argmax(axis=1) if size.shape[1] else np.zeros(T, dtype=int)
    largest = size[np.arange(T), worst] if size.shape[1] else np.zeros(T)
    names = [eqs_at[k][int(i)].name if size.shape[1] else "" for k, i in enumerate(worst)]
    return convergence_frame(prep.full[positions], [iterations] * T, largest, names, largest < tol)


_CHORD = 0.1  # a step with an iteration's factorised Jacobian (a chord step) is kept when it cuts the residuals' norm tenfold
_CHORD_STEPS = 10  # at most this many in an iteration


def _factorise(J: sparse.spmatrix) -> Any:
    """The sparse LU of the stacked Jacobian, with the unknowns kept in their order: period by period. The Jacobian is
    block-banded (each period's equations involve the periods a few lags and leads away), and eliminating the periods
    in time order keeps the fill-in within that band, growing with the number of periods, not faster: a general
    fill-reducing ordering (COLAMD) scatters it across the whole matrix, which at the size of a large model over a
    long horizon (FRB/US: 284 equations by 240 quarters) costs minutes and gigabytes instead of seconds."""
    try:
        return splu(J.tocsc(), permc_spec="NATURAL")
    except RuntimeError as exc:  # exactly singular
        raise ModelError(f"the stacked system is singular ({exc}); does each equation determine its variable?") from None


def _with_spells(solution: Any, schedule: Any, first: Any, regimes: Any) -> Any:
    """The solution with the regimes in its provenance: which holds when (as the first-order solution's)."""
    from dataclasses import replace

    provenance = solution.provenance
    if provenance is None:
        return solution
    from .regimes import as_regimes, regimes_text

    method = dict(provenance.method) if isinstance(provenance.method, dict) else {"solver": provenance.method}
    if schedule is not None:
        spells: list[dict[str, str]] = []
        for k, name in enumerate(schedule.names):
            if spells and spells[-1]["regime"] == name:
                spells[-1]["to"] = str(first + k)
            else:
                spells.append({"regime": name, "from": str(first + k), "to": str(first + k)})
        method["regimes"] = spells
    method["known regimes"] = regimes_text(as_regimes(regimes))
    return replace(solution, provenance=replace(provenance, method=method))


def _compiled(model: Any, prep: Any, positions: np.ndarray, first: int, last: int, T: int, n: int, uidx: dict[str, int],
              nrows: int, terminal: dict[str, float], solver: Any, growth: dict[str, float] | None = None) -> tuple[Any, Any] | None:
    """Residuals and the sparse Jacobian of the stacked system from the compiled equations: every equation is evaluated
    for all periods at once. Where each variable's value comes from (an unknown, the data, the terminal value) is
    worked out once, as an index into the unknowns or a fixed value."""
    from .fast import functions, parameter_values

    fns = functions(model, prep.eqs)
    if fns is None:
        return None
    p = parameter_values(model, fns)
    neq = len(prep.eqs)
    ks = np.arange(T)

    def gather(name: str, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = positions + lag
        index = np.full(T, -1)
        fixed = np.full(T, np.nan)
        add = np.zeros(T)  # terminal="growth": added to the last value
        j = uidx.get(name)
        for k in range(T):
            ps = int(pos[k])
            if j is not None and first <= ps <= last:
                index[k] = (ps - first) * n + j
            elif j is not None and ps > last:
                if solver.terminal in ("constant", "growth"):
                    index[k] = (T - 1) * n + j
                    if solver.terminal == "growth":
                        add[k] = (growth or {}).get(name, 0.0) * (ps - last)
                elif solver.terminal == "steady_state":
                    fixed[k] = terminal.get(name, np.nan)
                else:
                    value = prep.arrays[name][ps] if ps < nrows else np.nan
                    if not np.isfinite(value):
                        raise ModelError(f"{name}[{lag:+d}] beyond {prep.full[last]} is not in the data; use terminal='constant' or 'steady_state'")
                    fixed[k] = value
            elif 0 <= ps < nrows and np.isfinite(prep.arrays[name][ps]):
                fixed[k] = prep.arrays[name][ps]
            elif solver.terminal == "steady_state" and name in terminal:
                fixed[k] = terminal[name]
            if index[k] < 0 and not np.isfinite(fixed[k]):
                raise ModelError(f"{prep.full[int(positions[k])]}: no value for {name}[{lag:+d}]")
        return index, fixed, add

    plans = {ref: gather(*ref) for ref in {*fns.residuals.refs, *fns.derivatives.refs}}

    def slots(compiled: Any, Xc: np.ndarray) -> list[np.ndarray]:
        flat = Xc.ravel()
        out = []
        for ref in compiled.refs:
            index, fixed, add = plans[ref]
            values = fixed.copy()
            known = index >= 0
            values[known] = flat[index[known]] + add[known]
            out.append(values)
        return out

    adjust = np.array([a[positions] for a in prep.adjust]) if neq else np.zeros((0, T))

    def residuals(Xc: np.ndarray) -> np.ndarray:
        values = fns.residuals(slots(fns.residuals, Xc), p)
        F = np.array([np.broadcast_to(np.asarray(v, dtype=float), (T,)) for v in values]) - adjust  # equations x periods
        return F.T.ravel()  # period by period, as the unknowns

    # each non-zero derivative with respect to an unknown, at every period where it is one: rows and columns, once
    wanted = [(c, i, name, lag) for c, (i, name, lag) in enumerate(fns.pairs) if name in uidx]
    rows_, cols_, which, where = [], [], [], []
    for c, i, name, lag in wanted:
        index, _, _ = plans[(name, lag)]
        known = index >= 0
        rows_.append(ks[known] * neq + i)
        cols_.append(index[known])
        which.append(c)
        where.append(known)
    rows = np.concatenate(rows_) if rows_ else np.zeros(0, int)
    cols = np.concatenate(cols_) if cols_ else np.zeros(0, int)

    def jacobian(Xc: np.ndarray) -> sparse.csr_matrix:
        values = fns.derivatives(slots(fns.derivatives, Xc), p)
        parts = [np.broadcast_to(np.asarray(values[c], dtype=float), (T,))[known] for c, known in zip(which, where)]
        vals = np.concatenate(parts) if parts else np.zeros(0)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(T * neq, T * n))

    return residuals, jacobian


def simulate_with_surprises(model: Model, data: Dataset, span: tuple[Any, Any], sc: Scenario, solver: Any, regimes: Any = None) -> Solution:
    """Perfect foresight with expectation errors: agents know the anticipated paths from the start, and each
    unanticipated shock only when it happens. The model is solved again at each such date, from the path so far,
    with the surprises known by then; each solve is kept until the next surprise (as Dynare's perfect foresight
    with expectation errors)."""
    from dataclasses import replace

    import pandas as pd

    from .model import _path

    freq = data.frame.index.freq
    from ..data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    periods = pd.period_range(pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq), freq=freq)
    surprises = list(sc.unanticipated)
    news = {0}
    for u in surprises:
        for kind in (sc.set, sc.shift):
            if u in kind:
                p = _path(kind[u], periods)
                news |= {int(t) for t in np.flatnonzero(np.isfinite(p) & ((p != 0) | (kind is sc.set)))}
    dates = sorted(news)

    def known(values: Any, upto: int, fill: float) -> pd.Series:
        p = np.array(_path(values, periods), dtype=float)  # a copy: the path may be a read-only view
        p[upto + 1:] = fill  # not yet known: the baseline (set: the data's path; shift: nothing)
        return pd.Series(p, index=periods)

    working = data
    solution, iterations, kept = None, 0, []
    for k, d in enumerate(dates):
        start = periods[d]
        segment = replace(sc, unanticipated=(),
                          set={**{n: v for n, v in sc.set.items() if n not in surprises},
                               **{u: known(sc.set[u], d, np.nan) for u in surprises if u in sc.set}},
                          shift={**{n: v for n, v in sc.shift.items() if n not in surprises},
                                 **{u: known(sc.shift[u], d, 0.0) for u in surprises if u in sc.shift}})
        solution = simulate_stacked(model, working, (str(start), str(periods[-1])), segment, solver, regimes)
        iterations += solution.iterations
        kept.append((solution, start, periods[dates[k + 1] - 1] if k + 1 < len(dates) else periods[-1]))
        if k + 1 < len(dates):  # keep this solve until the next surprise; it becomes history for the next one
            working = _keep(working, solution, start, periods[dates[k + 1] - 1], data)
    assert solution is not None
    provenance = solution.provenance
    if provenance is not None:
        method = dict(provenance.method) if isinstance(provenance.method, dict) else {"solver": provenance.method}
        method["expectation errors at"] = [str(periods[d]) for d in dates[1:]]
        provenance = replace(provenance, sample=(str(periods[0]), str(periods[-1])), nobs=len(periods), method=method)
    return replace(solution, scenario=sc, iterations=iterations, provenance=provenance, convergence=_kept_convergence(kept))


def _kept_convergence(kept: list[tuple[Any, Any, Any]]) -> Any:
    """The convergence of solves each kept from one period to another (surprises): each period's row from the solve kept
    there, its iterations that solve's."""
    import pandas as pd

    parts = [s.convergence.loc[a:b] for s, a, b in kept if getattr(s, "convergence", None) is not None]
    return pd.concat(parts) if parts and len(parts) == len(kept) else None
