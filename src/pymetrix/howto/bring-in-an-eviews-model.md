---
summary: read an EViews model's text in the model editor, check it against the data, solve it over a span, and run a scenario.
audience: EViews users with backward-looking models
order: 9
---

# Bring an EViews model across, then check and solve it

Read a model written for EViews (its `model` object's equations, as its Text view or a program shows them) into
pymetrix, check that it reads as meant and that the data supply what it needs, solve it period by period over a span,
and run a scenario against the baseline. EViews is not needed: the text is read as it is.

## Before you start

- The data the model runs on. This page uses the example `sample_macro` (quarterly, 2000Q1 to 2024Q4), with
  consumption and investment copied to `cons` and `inv`: EViews keeps the name `c` for its coefficient vector, so an
  EViews model cannot have a series called `c`.
- The model's text, with values for its coefficients. A model whose equations EViews estimated has them in the
  `c` vector; write them out, as EViews's model text does, as `macro.txt`:

  ```text macro.txt
  ' The model, with the coefficients EViews estimated
  model macro
  macro.append cons = c(1) + c(2)*y
  macro.append inv = c(3) + c(4)*y(-1) + c(5)*inv(-1)
  macro.append @identity y = cons + inv + g
  c(1) = 11.468
  c(2) = 0.75145
  c(3) = 4.9437
  c(4) = 0.10745
  c(5) = -0.06499
  ```

- Or the program that estimates the equations and builds the model, `macro.prg`. Its equations are estimated
  again, in pymetrix, on the same sample:

  ```text macro.prg
  ' A small model of the example economy, estimated 2001-2019
  smpl 2001q1 2019q4
  equation eq_cons.ls cons c y
  equation eq_inv.ls inv c y(-1) inv(-1)
  model macro
  macro.merge eq_cons
  macro.merge eq_inv
  macro.append @identity y = cons + inv + g
  ```

- The span to solve over, and its starting values: the model is solved 2020Q1 to 2024Q4, after the estimation
  sample, with the lags of 2019 from the data.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data. Data → Derived series…, with
   Data: data, New series name: cons and Formula: `c`; again with New series name: inv and Formula: `i`.
2. Model → Model editor…, Syntax: EViews (.prg / model text), Name: macro, and paste the text of `macro.txt`. The
   Problems tab lists anything that does not read, at its line and column; Symbols lists `c_1` to `c_5` (EViews's
   `c(1)` to `c(5)`) with their values, and the series as endogenous or exogenous. Apply builds the model into the
   session.
3. Model → Check model…, with Equations: macro, Data: data, Start: 2020Q1 and End: 2024Q4. It checks that each
   equation determines its variable, the blocks, and that the data have every exogenous series and starting value
   the span needs; here it passes.
4. Model → Solution → Solve model…, with Equations: macro, Start: 2020Q1, End: 2024Q4 and Save as: base: a dynamic
   simulation, each period from the model's own earlier values. The Convergence view gives each period's iterations
   and largest residual; Graph and Data the solved series.
5. A scenario: Solve model… again, with Add: g=5 (government spending 5 higher in every period) and Save as: more_g.
   The difference between `more_g` and `base` is the scenario's effect.

A program such as `macro.prg`, whose equations are estimated, needs the data to read: in code, `read_eviews` with
`data=` estimates them; in the app, File → Translate EViews program… translates the whole program, with the
`pymetrix-eviews` plugin ([Translate an EViews program](translate-an-eviews-program.md)). In the model editor alone
its equations are flagged "is estimated: give data", as it has no data.

## In code

```python
from pymetrix import Dataset, Scenario, read_eviews

data = Dataset.example("sample_macro").derive(cons="c", inv="i")
macro = read_eviews("macro.txt")                       # a template: three equations, c_1 to c_5
model = macro.model
model.check(data, ("2020Q1", "2024Q4"))                # passes
base = model.solve(data, ("2020Q1", "2024Q4"))
more_g = model.solve(data, ("2020Q1", "2024Q4"), scenario=Scenario(shift={"g": 5}))
effect = more_g.solved()["y"] - base.solved()["y"]
effect["2020Q1"]                                       # 20.1: 5 times 1 / (1 - 0.75145)
```

The program, its equations estimated on its `smpl`:

```python
fitted = read_eviews("macro.prg", data=data)
[eq.params.values for eq in fitted.model.equations[:2]]   # 11.468 and 0.75145; 4.9437, 0.10745 and -0.06499
```

## Check the result

- The impact of the scenario. Investment depends on last quarter's output only, so in 2020Q1 output rises by 5 times
  the multiplier 1 / (1 - 0.75145) = 4.02: 20.1, worked out by hand from the consumption equation.
- The coefficients: `read_eviews("macro.prg", data=data)` estimates 11.468 and 0.75145 for consumption and 4.9437,
  0.10745 and -0.06499 for investment, the values in `macro.txt`, as EViews's own estimates on 2001Q1 to 2019Q4
  should.
- The simulation against the data: output in 2024Q4 is 593.3 in `base` against 584.3 in the data. A dynamic
  simulation drifts from the data, as the equations' errors are left out; EViews's deterministic solve does the same.
- The Check model view: no equation without its variable, no missing series.

## Pitfalls

- **EViews's names are its own.** `c` is the coefficient vector, so a series named `c` must be renamed; names are
  read in lower case, as EViews treats them; `c(1)` becomes the parameter `c_1`.
- **A coefficient without a value is an error at its line.** The model editor names it; give it in the text
  (`c(1) = ...`, `param`), or estimate the equations with `read_eviews(program, data=data)`.
- **Some EViews functions are refused at their place.** `@max` over a sample, moving variances, `and`/`or`, and the
  programming language (`for`, `!n`, `%s`) in a model's text: the guide lists what reads.
- **The default solver differs.** EViews solves by Gauss-Seidel unless told otherwise; pymetrix by Newton. Both give
  the same solution where both converge; choose Gauss-Seidel under Solver to match EViews's iterations.
- **Add factors.** EViews's `@add` (`macro.append @add cons cons_a`, an intercept shift) comes across; with tracking
  add factors (Model → Solution → Add factors…) the baseline reproduces the data and a scenario reads as a deviation
  from it.
- **An EViews state-space object is another route.** Model → Import → EViews state-space object… reads one as a
  state-space model.

## See also

- [Solving models](pymetrix:topic/model-solution) and
  [Coming from EViews](pymetrix:topic/coming-from-eviews), in the help.
- [Translate an EViews program](translate-an-eviews-program.md): a whole program, data work and estimation
  included.
- [Solve a large backward-looking model](solve-a-large-backward-looking-model.md): the solver's options for large
  models.
- [The user guide's EViews's syntax: model text and programs](../../../docs/guide.md#eviewss-syntax-model-text-and-programs):
  every function read, and what is refused.
