# Solving large backward-looking models

This page is for large models solved period by period: many identities and estimated equations, mostly
recursive (each equation needs only variables found before it), with a few simultaneous blocks. It says which of
the solver options to use, how they combine, and how to read what `Solution.convergence` records. The
[user guide](guide.md#solver-options-convergence-test-fallback-line-search-jacobian-reuse-and-blocks) has the
reference for each option.

The code blocks run in order, as one script; a test runs them as written, so they stay correct. The model is
generated, so nothing needs downloading.

## A model to try them on

```python
import numpy as np
import pandas as pd
from pymetrix import Dataset, Equation, Fallback, GaussSeidel, Model, Newton

eqs = [Equation("c = 0.55*y + 0.25*c[-1] + 0.1*log(w)", name="cons"),          # a simultaneous block: c, inv, y
       Equation("inv = 0.12*y + 0.05*exp(0.2*(y - y[-1])) + 0.4*inv[-1]", name="invest"),
       Equation("y = c + inv + g", name="output"),
       Equation("p = 0.6*p[-1] + 0.3*log(1 + u) + 0.01*y", name="prices"),   # another: p, u
       Equation("u = 0.5*u[-1] + 0.2*p/(1 + p) + 0.1*z", name="slack"),
       Equation("v0 = 0.1*y + 0.3*v0[-1]", name="link0")]
eqs += [Equation(f"v{k} = 0.5*v{k - 1} + 0.1*p + 0.3*v{k}[-1] + 0.01*z", name=f"link{k}") for k in range(1, 95)]
model = Model(eqs)                                   # 100 equations, 95 of them a recursive chain

index = pd.period_range("2000Q1", periods=44, freq="Q", name="time")
frame = pd.DataFrame(1.0, index=index, columns=list(model.endogenous))
frame["g"], frame["w"], frame["z"] = 1.0 + 0.05 * np.sin(np.arange(44) / 3), 1.5, 0.5
frame.loc["2001Q1":, list(model.endogenous)] = np.nan   # the solved periods have no values, as in a projection
data = Dataset.from_frame(frame)
span = ("2001Q1", "2010Q4")                            # 40 quarters; the lags come from 2000

base = model.solve(data, span)                         # the default: Newton() on all 100 equations at once
```

`python benchmarks/solve_options.py` times the options below on a model like this one, with 200 equations, and
counts the equations and Jacobians each evaluates; `--model`, `--data` and `--span` time your own model.

## Solve the recursive equations once: `ordering="blocks"`

```python
blocks = model.solve(data, span, ordering="blocks")
blocks.convergence[["iterations", "blocks", "recursive"]].head(2)   # 2 blocks iterated on, 95 equations solved once
blocks.convergence.attrs["blocks"][:3]                              # [('cons', 'invest', 'output'), ('prices', 'slack'), ('link0',)]
```

By default (`ordering="joint"`) the solver takes every equation in every iteration: Newton evaluates all of them
and a Jacobian of all of them, and factorises it. With `ordering="blocks"` each period's equations are split into
blocks (the strongly connected components of "this equation needs that variable in the same period") and taken in
an order where each block needs only those before it. An equation that is a block of its own, and linear in its
own variable, as an identity is, is solved once, by one evaluation; the solver iterates only within the
simultaneous blocks, each on its own. The solution is the joint solution (to the solver's tolerance).

Use it when most equations are recursive. The `recursive` column says how many are; if it is small, the model is
mostly one simultaneous block and ordering changes little. It needs a backward-looking model solved period by
period (`Newton`, `GaussSeidel` or a `Fallback` of them), without regimes or links.

The work saved is in evaluations. In wall time, a joint Newton step on a model of a few hundred equations is
already fast (its residuals and Jacobian are compiled and evaluated at once), while each equation solved once is a
call of its own: on a small model the block solve can take as long, though it evaluates several times fewer
equations. The joint Jacobian grows with the square of the number of equations and its factorisation with the
cube; the block solve grows about in proportion. The larger the model, and the more costly its equations, the more
the blocks save.

## Keep the Jacobian: `Newton(refresh="period")`

```python
kept = model.solve(data, span, solver=Newton(refresh="period"))
kept.convergence[["iterations", "jacobians"]].head(2)              # one Jacobian a period, a few iterations
```

Newton evaluates and factorises the Jacobian at every iteration (`refresh="iteration"`, the default).
`refresh="period"` keeps the period's first Jacobian, factorised once, while each step cuts the largest residual
to a quarter or less, and evaluates it again when convergence slows (a modified Newton). It takes more iterations,
each cheaper: use it when the Jacobian is the costly part, a large simultaneous block or equations with costly
derivatives. With `ordering="blocks"` it applies to each block.

## Stop on the step, whatever the units: `convergence="step"`

```python
step = model.solve(data, span, solver=Newton(convergence="step"))
step.convergence["test"].unique()                                   # ['step']
```

By default a period has converged when the largest residual is below `tol` (`convergence="residual"`), a test in
the equations' units: an equation in billions needs its residual far smaller, relative to its terms, than one in
percentage points, and may never get there in floating point. `convergence="step"` stops instead when the largest
scaled step, `|x_new - x_old| / (1 + |x_old|)`, is below `tol`: the same iterations however the equations are
scaled. `"both"` asks for both. The step test needs one step past the last that moved, so it takes one more
iteration than the residual's where the residual's would stop; with `refresh="period"` that iteration reuses the
Jacobian.

Use it for models whose variables are in mixed units (levels, ratios, rates), which large models usually are.

## Starts far from the solution: `Newton(line_search=True)`

```python
far = model.solve(data, span, solver=Newton(line_search=True))
```

A period starts from the previous period's solution. After a large shock, or in the first period of a projection
with poor starting values, Newton's full step can overshoot and diverge. `line_search=True` halves the step until
the residuals fall enough (Armijo's sufficient decrease), and near the solution takes Newton's full steps, so it
gives the same solution where plain Newton converges.

## Retry a failed period: `Fallback(...)`

```python
retry = Fallback(Newton(),                                          # tried first, on every period
                 GaussSeidel(damping=0.5, max_iter=400),            # where it fails, from the same starting values
                 Newton(line_search=True))                          # and then this
model.solve(data, span, solver=retry).convergence["solver"].value_counts()   # the solver that solved each period
```

Without a fallback, a period the solver cannot solve stops the solve with an error. `Fallback(a, b, c)` tries `a`
on every period and, where it fails, `b` and then `c`, each from the period's starting values; the periods `a`
solves are exactly as `a` alone solves them. When every solver fails, the error lists each attempt with its worst
equation. With `ordering="blocks"` the solvers are tried block by block, so one hard block does not re-solve the
rest.

A sequence that works well: first the fast solver for the model (Newton with the options above); then Newton with a
line search, for a start too far away; last, damped Gauss-Seidel, which needs no Jacobian, for a period where the
Jacobian is singular or nearly so. Give the last a generous `max_iter`: damping slows it down.

## A recommended starting configuration

For a model of mostly identities with a few simultaneous blocks:

```python
recommended = Fallback(Newton(convergence="step", refresh="period"),
                       Newton(convergence="step", line_search=True),
                       GaussSeidel(convergence="step", damping=0.5, max_iter=1000))
solution = model.solve(data, span, solver=recommended, ordering="blocks")
assert np.allclose(solution.solved().frame, base.solved().frame, rtol=1e-8)
```

The blocks evaluate each identity once; the step test does not depend on the equations' units; the kept Jacobian
makes each block's iterations cheap; and a period that fails is retried, first with a line search, then by damped
Gauss-Seidel, rather than stopping the solve. Check the convergence table of the first solve (below); then drop
what the model does not need. If `recursive` is small, solve jointly; if the `solver` column never names a retry,
the fallback costs nothing and can stay.

It is chosen for evaluations and reliability, and its wall time depends on the model's size: on the benchmark's
generated model it evaluated a fifth of the default's equations and took longer at 200 equations, less at 600.
Time it on your own model with `benchmarks/solve_options.py`, which also shows the time in the periods' solves
apart from setting the solve up.

In the app: Model → Solution → Solve model, with Solver Newton and, under Solver settings, Convergence: step and
Refresh: period; under Advanced, Ordering: blocks and Fallback. The app's Fallback retries with
`GaussSeidel(damping=0.5, max_iter=400)` and then `Newton(line_search=True)`, as the Script window shows; for the
step test on the retries too, solve with the code above.

## Reading `Solution.convergence`

```python
table = solution.convergence
table.columns.tolist()          # ['iterations', 'residual', 'equation', 'converged', 'test', 'solver', 'jacobians', 'blocks', 'recursive']
seconds = table.attrs["seconds"]           # each period's solve, in seconds
assert len(seconds) == len(table) and min(seconds) >= 0
```

One row a period. Always there:

- `iterations`: the solver's iterations, summed over the blocks iterated on, and over every attempt of a
  `Fallback`;
- `residual` and `equation`: the largest absolute residual left at the end, and its equation (with blocks, the
  largest in the blocks iterated on: equations solved once hold to rounding);
- `converged`: whether the period converged (a solve that fails stops there; the error's `convergence` has the
  table up to that period).

With any of the options (a convergence test other than the residual's, a line search, `refresh="period"`, a
`Fallback`, or `ordering="blocks"`), also:

- `test`: the test that stopped the period (`residual`, `step` or `both`);
- `solver`: the solver that solved it, as code, such as `Newton(convergence='step', refresh='period')`. With a
  `Fallback`, a period that names a later solver failed with the earlier ones: look at its row, its starting values
  and the shocks in that period;
- `jacobians`: the Jacobians evaluated. Close to `iterations` without `refresh="period"`; much smaller with it,
  unless convergence keeps slowing.

With `ordering="blocks"`, also:

- `blocks`: the simultaneous blocks iterated on;
- `recursive`: the equations solved once.

And, outside the columns:

- `attrs["blocks"]` (with `ordering="blocks"`): the blocks by equation name, in solution order; the large blocks
  are where the iterations go (a dict, by layout, when a partial fix or hold changes the equations solved);
- `attrs["seconds"]`: the seconds each period's solve took, one a row, measured by `time.perf_counter` around the
  solver's call for that period, so setting the solve up (preparing the data, compiling the equations, finding the
  blocks) is not in it. A period that takes far longer than the others is the one to look at. It is recorded for
  every solve period by period, with or without options; perfect foresight, which solves all periods at once, does
  not have it.

The seconds are metadata, not a column, so the table and the default results are as before. They are kept when a
solution is kept in a project, and when the solve is run again on opening, they are its own; a result saved to a
`.pymxr` file keeps the Convergence table without them.
