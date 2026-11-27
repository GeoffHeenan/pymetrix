---
summary: choose the solver's options for a large model of identities and estimated equations (blocks, the step test, a kept Jacobian, a line search, fallbacks) and read the convergence table.
audience: forecasters with large models, and EViews users of model solve
order: 13
---

# Solve a large backward-looking model

Solve a large model period by period, quickly and without failing: many identities and estimated equations, mostly
recursive (each equation needs only variables found before it), with a few simultaneous blocks. This page says which
of the solver's options to use, how they combine, and how to read what `Solution.convergence` records of each
period's solve.

## Before you start

- The model and its data. The page generates them, so nothing needs downloading: 100 equations, of which two small
  blocks are simultaneous (consumption, investment and output; prices and slack) and 95 are a recursive chain, and
  quarterly data from 2000 with the solved periods, 2001Q1 to 2010Q4, empty, as in a projection. The first block of
  In code makes them.
- What makes the model hard, if anything. The options answer different problems:
  - most equations recursive: `ordering="blocks"` solves each of them once;
  - a costly Jacobian (a large simultaneous block, or equations with costly derivatives): `Newton(refresh="period")`;
  - variables in mixed units (levels, ratios, rates): `convergence="step"`;
  - starts far from the solution (a large shock, poor starting values): `Newton(line_search=True)`;
  - periods that fail now and then: `Fallback(...)`, which retries them with other solvers.

`python benchmarks/solve_options.py` times the options on a model like this one, with 200 equations, and counts the
equations and Jacobians each evaluates; `--model`, `--data` and `--span` time your own model.

## In the app

1. File → New script, paste the first block of In code below, and ▶ Run (F5): `model` and `data` appear in Objects.
   A model of your own comes from the model editor or an import instead.
2. Model → Check model…, with Equations: model, Data: data, Start: 2001Q1 and End: 2010Q4: every equation
   determines its variable, and the data have every input the span needs.
3. Model → Solution → Solve model…, with Equations: model, Start: 2001Q1, End: 2010Q4, Solver: Newton and Save as:
   solution. Under Advanced settings, in Solver settings: Ordering: Blocks and Fallback ticked; and under Newton,
   Convergence: Step and Refresh: Period.
4. The solution's Convergence view has a row a period: the iterations, the largest residual left and its equation,
   and with these options the test that stopped the period, the solver that solved it, the Jacobians evaluated, the
   blocks iterated on and the equations solved once.

The app's Fallback retries a period with `GaussSeidel(damping=0.5, max_iter=400)` and then
`Newton(line_search=True)`, as View → Script shows; for the step test on the retries too, solve with the code below.

## In code

The code blocks run in order, as one script.

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

### Solve the recursive equations once: `ordering="blocks"`

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
mostly one simultaneous block and ordering changes little.

### Keep the Jacobian: `Newton(refresh="period")`

```python
kept = model.solve(data, span, solver=Newton(refresh="period"))
kept.convergence[["iterations", "jacobians"]].head(2)              # one Jacobian a period, a few iterations
```

Newton evaluates and factorises the Jacobian at every iteration (`refresh="iteration"`, the default).
`refresh="period"` keeps the period's first Jacobian, factorised once, while each step cuts the largest residual
to a quarter or less, and evaluates it again when convergence slows (a modified Newton). It takes more iterations,
each cheaper: use it when the Jacobian is the costly part, a large simultaneous block or equations with costly
derivatives. With `ordering="blocks"` it applies to each block.

### Stop on the step, whatever the units: `convergence="step"`

```python
step = model.solve(data, span, solver=Newton(convergence="step"))
step.convergence["test"].unique()                                   # ['step']
```

By default a period has converged when the largest residual is below `tol` (`convergence="residual"`), a test in
the equations' units: an equation in billions needs its residual far smaller, relative to its terms, than one in
percentage points, and may never get there in floating point. `convergence="step"` stops instead when the largest
scaled step, `|x_new - x_old| / (1 + |x_old|)`, is below `tol`: the same iterations however the equations are
scaled. `"both"` asks for both. Use it for models whose variables are in mixed units, which large models usually
are.

### Starts far from the solution: `Newton(line_search=True)`

```python
far = model.solve(data, span, solver=Newton(line_search=True))
```

A period starts from the previous period's solution. After a large shock, or in the first period of a projection
with poor starting values, Newton's full step can overshoot and diverge. `line_search=True` halves the step until
the residuals fall enough (Armijo's sufficient decrease), and near the solution takes Newton's full steps, so it
gives the same solution where plain Newton converges.

### Retry a failed period: `Fallback(...)`

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
rest. A sequence that works well: first the fast solver for the model; then Newton with a line search, for a start
too far away; last, damped Gauss-Seidel, which needs no Jacobian, for a period where the Jacobian is singular or
nearly so, with a generous `max_iter`, as damping slows it down.

### A recommended starting configuration

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

## Check the result

```python
table = solution.convergence
table.columns.tolist()          # ['iterations', 'residual', 'equation', 'converged', 'test', 'solver', 'jacobians', 'blocks', 'recursive']
seconds = table.attrs["seconds"]           # each period's solve, in seconds
assert len(seconds) == len(table) and min(seconds) >= 0
```

- Every option gives the default's solution, to the solver's tolerance: compare `solved().frame` with `base`'s, as
  the recommended configuration's block does.
- `solution.convergence`, one row a period. Always there: `iterations` (summed over the blocks iterated on, and over
  every attempt of a `Fallback`); `residual` and `equation`, the largest absolute residual left and its equation
  (with blocks, the largest in the blocks iterated on: equations solved once hold to rounding); `converged`.
- With any of the options, also: `test`, the test that stopped the period (`residual`, `step` or `both`); `solver`,
  the solver that solved it, as code, such as `Newton(convergence='step', refresh='period')` (with a `Fallback`, a
  period that names a later solver failed with the earlier ones: look at its row, its starting values and the shocks
  in that period); `jacobians`, close to `iterations` without `refresh="period"`, much smaller with it.
- With `ordering="blocks"`, also `blocks` (the simultaneous blocks iterated on, 2 here) and `recursive` (the
  equations solved once, 95 here), and `attrs["blocks"]`, the blocks by equation name in solution order: the large
  blocks are where the iterations go (a dict, by layout, when a partial fix or hold changes the equations solved).
- `attrs["seconds"]`: the seconds each period's solve took, measured by `time.perf_counter` around the solver's call
  for that period, so setting the solve up (preparing the data, compiling the equations, finding the blocks) is not
  in it. A period that takes far longer than the others is the one to look at.

## Pitfalls

- **Fewer evaluations is not always less time.** A joint Newton step on a model of a few hundred compiled equations
  is already fast, while each equation solved once is a call of its own: on a small model the block solve can take
  as long, though it evaluates several times fewer equations. The joint Jacobian grows with the square of the number
  of equations and its factorisation with the cube; the block solve about in proportion. On the benchmark's
  generated model the recommended configuration evaluated a fifth of the default's equations and took longer at 200
  equations, less at 600. Time it on your own model with `benchmarks/solve_options.py`, which also shows the time
  in the periods' solves apart from setting the solve up.
- **The step test takes one iteration more.** It needs a step past the last that moved, so it stops one iteration
  after the residual test would; with `refresh="period"` that iteration reuses the Jacobian.
- **Blocks need a model solved period by period.** `ordering="blocks"` takes `Newton`, `GaussSeidel` or a `Fallback`
  of them, without regimes or links; perfect foresight solves all periods at once and has no blocks, nor the
  seconds by period.
- **A Fallback can hide a problem.** A period that only a retry solves has something wrong: a large shock, poor
  starting values, a nearly singular Jacobian. Read the `solver` column, not only the solution.
- **The seconds are metadata, not a column.** They are kept when a solution is kept in a project, and when the solve
  is run again on opening they are its own; a result saved to a `.pymxr` file keeps the Convergence table without
  them.

## See also

- [Solving models](pymetrix:topic/model-solution), in the help.
- [Bring an EViews model across, then check and solve it](bring-in-an-eviews-model.md): a small model, from EViews,
  solved and shocked.
- [The user guide's Solver options](../../../docs/guide.md#solver-options-convergence-test-fallback-line-search-jacobian-reuse-and-blocks):
  the reference for each option, and
  [Fixing in some periods, missing inputs and convergence by period](../../../docs/guide.md#fixing-in-some-periods-missing-inputs-and-convergence-by-period).
