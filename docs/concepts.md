# Concepts

The objects and verbs pymetrix is built from, how they relate, and which name each has in the code.
This is the reference for design work: a new feature should fit one of these concepts, or make the
case for a new one here first. Status: **exists**, **partial** (exists but incomplete), or
**planned**.

## Principles

1. **One vocabulary.** The engine, the desktop app, notebooks, reports, saved projects and any
   future assistant use the same objects and the same names. A concept has one name; we do not add
   a second name for something that exists (no `ModelSpec` beside `Model`, no `estimate` beside
   `fit`).
2. **Specifications, settings, data and results are separate objects.**
   - *What* is estimated or solved: an equation, a model.
   - *How*: method or solver settings.
   - *On what*: a dataset, a sample.
   - *What came out*: a result with its provenance.

   No object mixes them.
3. **Results are values.** A fitted equation, a solution or a forecast round is immutable. It
   knows where it came from (`Provenance`) and can be compared, reported and saved.
4. **The code form is canonical.** Formulas such as `c ~ y + c[-1]` are what scripts, projects and
   generated code contain. Algebraic notation (`cₜ = β₀ + β₁ yₜ + β₂ cₜ₋₁ + εₜ`) is a way of
   showing them, never a different meaning.
5. **The engine is headless.** Nothing in the engine imports a user interface. What a result shows
   is defined once, in `pymetrix.views`, and every frontend renders that. A test checks that the
   engine, reports and the test kit never import `pymetrix.ui`.
6. **Commands are the only way a session changes.** Dialogs, notebooks driving the session, project
   replay and any future AI assistant all run commands, so all of it is recorded, replayable and
   visible as Python.

## How the pieces fit

```text
 data                    specification              settings
 Dataset, Variable,      Equation, Param,           OLS, IV, ML, Bayes ...  (methods)
 Vintage, axes,          Model, Scenario,           Newton, PerfectForesight (solvers)
 derived recipes         MeasurementMap             filters, tests
        \                      |                       /
         \___________ fit / solve / forecast ________/
                               |
 results                       v
 FittedEquation, FittedVAR, FittedSystem, FittedStateSpace, Solution,
 LinearSolution, StateSpaceSystem, TestResult, ForecastRound, Backtest    (+ Provenance)
                               |
 views (pymetrix.views)        v
 [(title, Table | Figure), ...]  ->  desktop windows, Jupyter, reports, fingerprints, AI*
                               |
 workflow                      v
 Command -> Session (CommandRecord history) -> project file (.pymx) -> replay
 ForecastStore (rounds over time), Report (Word, Excel, HTML)

 * planned
```

## Objects

### Data

| Concept | pymetrix | Status | Notes |
| --- | --- | --- | --- |
| Dataset | `Dataset` | exists | Series on time and entity axes (`TimeAxis`, `EntityAxis`), with metadata; panels and cross-sections too. |
| Series metadata | `Variable` | exists | Label, units, scale, measure, prices, base period, seasonal adjustment, source, vintage, symbol, recipe. |
| Vintage | `Vintage` | exists | When a series was released (the release date, never the load date), from FRED/ALFRED, SDMX and the World Bank. |
| Transformation | expressions; `with_series`, `derive`; recipes `Formula`, `Filtered`, `SeasonallyAdjusted` | exists | Lags and leads, `log`, `d`, `dlog`, `pch`, `max`/`min` of values (period by period), `step`, group-wise functions, filters, seasonal adjustment; recipes recompute when inputs change. Frequency conversion: `to_freq`, `aggregate`. |
| Role | `Role` | exists | Endogenous, exogenous, instrument, shock. The compiled model gives every variable a role; `Model(roles=...)` marks shocks and others. |

### Specifications

| Concept | pymetrix | Status | Notes |
| --- | --- | --- | --- |
| Equation | `Equation` | exists | Three forms: linear (`c ~ y + c[-1]`), explicit with named parameters (`c ~ a + b*y`, `params=[...]`), and identity (`y = c + i + g`). Also measurement equations, once measurement maps exist. |
| Parameter | `Param` (metadata), `Parameter` and `ParameterSet` (values) | exists | `Param`: name, label, symbol, bounds, transform, start, prior. `Parameter`: a value with its status (estimated, calibrated, fixed), standard error, equation and `Param`. `Model(params=...)` takes a dictionary or a `ParameterSet`; `model.parameters` gives the set, estimated coefficients named by equation (`cons.y`). |
| Shock | shock names in `LinearModel`, `Role.SHOCK` | partial | Named in rational-expectations models; residuals play the part in backward models. |
| Model | `Model` | exists | Fitted equations, calibrated equations and identities, solved together. |
| Template model | `templates.new_keynesian`, `rbc`, `gap_model` returning a `Template` | exists | A small standard model with calibrated parameters, shocks, shock sizes, a steady-state guess and measurement equations; `solve()`, `check()`, `system()`; **Model → Template model**. Examples to copy, not classes to inherit. |
| Compiled model | `ModelSystem`, from `Model.compile()` | exists | Equations as residuals with symbolic parameters, every variable with its role, the `ParameterSet`, lead/lag incidence; views of equations (with algebraic notation), variables, parameters and incidence. Every solution path and the Dynare importer use it. |
| Scenario | `Scenario` | exists | Set, shift or fix paths, add factors, targets with free shocks (exogenize/endogenize), and unanticipated shocks for first-order solutions: one object for scenarios and simulation plans. |
| Measurement map | `MeasurementMap` | exists | Observed series as equations in model variables and their lags, with their own parameters (`growth = g + 4*(y - y[-1])`), measurement errors, data-series names, units and labels. `apply(solution)` reports model paths in data units (exact); for the Kalman filter they are linearised at the steady state (exact when linear). |

### Settings

| Concept | pymetrix | Status | Notes |
| --- | --- | --- | --- |
| Estimation settings | method dataclasses: `OLS`, `IV`, `ML`, `Bayes`, plugin methods | exists | Frozen dataclasses; the same objects drive forms, scripts and projects. |
| Solution settings | solver dataclasses: `Newton`, `GaussSeidel`, `PerfectForesight` | exists | |
| Simulation settings | span, `Scenario`, solver; `LinearSolution.simulate(periods, shock_sd, seed)` | partial | No single `SimulationSpec`: the pieces are explicit arguments, which is enough for now. |

### Results

| Concept | pymetrix | Status | Notes |
| --- | --- | --- | --- |
| Estimates | `FittedEquation`, `FittedSystem`, `FittedVAR`, `FittedVECM`, `FittedStateSpace`, ... | exists | With `params` (`PointEstimate` or `Draws`), statistics, residuals and `provenance`. Every estimate has `summary()`, `params`, `resid`, `fitted`, `predict()` and `plot()` (or a `NotSupportedError` naming what to use instead), the tidy `coefficients()`, `statistics()` and `observations()`, and `to_record()` (`pymetrix.results`). |
| Test result | `TestResult` | exists | Statistic, p-value, critical values, details. |
| Solution | `Solution` (a `Dataset`), `LinearSolution`, `SteadyState` | exists | |
| State-space system | `StateSpaceSystem` (a `SystemMatrices`) | exists | From `LinearSolution.system(measurement, shock_sd)`: named states (with lagged copies where measurement needs them), the measurement map, shock sizes; `filter(data)` gives the model's variables in levels and fitted observables; `loglike(data)`; views of every matrix; **Model → Analysis → State-space form** and **Model → Analysis → Kalman filter**. |
| Forecast | forecast datasets; `ForecastRound` | exists | A round is a published forecast: paths, data end, judgement with reasons, notes, source. |
| Track record | `ForecastStore`, `Backtest` | exists | |
| Summary objects | `Table`, `Figure` | exists | What views and reports are made of. Draws are summarised (means, bands) before reporting. |
| Provenance | `Provenance` | exists | Data vintages, sample, method, backend, pymetrix version. |

### Views, workflow and assistance

| Concept | pymetrix | Status | Notes |
| --- | --- | --- | --- |
| View | `pymetrix.views`: `register_view`, `view_sections` | exists | A result as `[(title, Table or Figure), ...]`; frontends only render it. `pymetrix.testing.fingerprint` compares results through their views. |
| Command | `Command` subclasses, `@command` | exists | Serializable, stable ID, schema version, migrations, `validate`, `run`, `to_code`. |
| Call | `Call`, `@api` (`pymetrix.calls`) | partial | A public function's call as a command: typed code and dialogs record the same step, with every argument; see [design/commands.md](design/commands.md). |
| History | `CommandRecord` | exists | Each step with the objects it read and wrote, and its code. |
| Session | `Session` | exists | The workspace: objects, history, script. |
| Project | `.pymx` file: `save_project`, `load_project` | exists | Data, steps and course material; replayed on opening; typed code runs only if trusted. |
| Report | `Report` | exists | Word, Excel, HTML, with a provenance appendix. |
| Packet | none | planned | A forecast round with its charts, tables and report bundled, or a replication or review package. |
| Context bundle | none | planned | A privacy-filtered summary of schemas, help and project state for an AI assistant ([note](notes/ai-assistance-in-the-ui.md)). |

## Verbs

| Verb | pymetrix | Status |
| --- | --- | --- |
| import | `Dataset.read` (files, `fred://`, `worldbank://`, `sdmx://`), `Dataset.example`; Load data, Load example data | exists |
| validate data | `Dataset.coverage`, `describe`, `describe_panel`; checks on loading | partial |
| transform | expressions, `with_series`, `derive`, `to_freq`, `aggregate`, filters, seasonal adjustment | exists |
| vintage | `Vintage` metadata; ALFRED vintages; `backtest(..., vintages=...)` | partial |
| describe | `describe`, `corr`, `correlogram`, `plot`, the Data and Variables views | exists |
| define | `Equation`, `Model`, `Scenario`; Define equation | exists |
| compile | `Model.compile()` returning a `ModelSystem` | exists |
| check | `Model.check(data, span, shocks)` returning a `ModelCheck`; **Model → Check model** | exists |
| estimate | `fit` (equations, systems, VARs, state space); `fit_rolling`, `fit_recursive`; `Model.estimate` for linearised models | exists |
| calibrate | `Equation.with_params`, `Model(params=...)` | exists |
| solve | `Model.solve`, `Model.linearize(...).solve()`, `steady_state` | exists |
| simulate | `LinearSolution.simulate`, `Model.solve` with a `Scenario`; a Monte Carlo helper is planned | exists |
| forecast | `forecast` on fitted results; `ForecastRound` | exists |
| decompose | `FEVD` (VARs), impulse responses; historical shock decompositions | partial |
| compare | `compare(...)`, `ForecastStore.compare`, `Backtest.compare`, side-by-side tables | exists |
| record | `CommandRecord`, `Session.script()` | exists |
| replay | `load_project`, `Session.recompute` | exists |
| export | `Report.save`, File → Export script, table and figure export | exists |
| explain | help pages, the defaults page, algebraic notation | exists |
| package | forecast packets, replication packages | planned |
| suggest, preview, approve, apply, audit, redact | the AI layer; *approve* exists for typed code in projects (trust) | planned |

## Results

**Rule: nothing important exists only in a summary, a table's notes, a chart or a report.** Every
estimator, solver, forecast, test and decomposition returns a result object, and its numbers are
named fields; tables, figures, reports and the desktop windows are views of those fields. The test
for any number shown anywhere: *could a script get the same number without reading display text?*

- **Payload:** parameters as `PointEstimate` or `Draws` (values, standard errors, covariance),
  `stats` (a dictionary of statistics, including choices made automatically such as `hac_lags`),
  residuals and fitted values, forecasts, states, and method-specific fields (for example
  `FittedARDL.long_run_coefficients`, `ecm_params`). Tests return `TestResult` fields.
- **Provenance** (`result.provenance`): data vintages, specification, method and backend, the
  sample actually estimated, observations, the seed, the pymetrix version, a `data_fingerprint`
  of the data used, the `warnings` raised while estimating (recorded, and still shown), and
  `converged` for iterative estimators (`None` when not iterative or not known).
  `Session.producer(name)` gives the command step that made an object.
- **Views:** `pymetrix.views.titles(result)` lists them and `view(result, "Summary")` returns
  one. A `Table` counts as structured, since it holds typed numbers and has `to_frame()`, but key
  quantities also get named fields. Some tables have a DataFrame method behind them, for example
  `ForecastStore.revisions` behind `compare`, `ForecastStore.scores` behind `accuracy`, and
  `Backtest.scores` behind `accuracy`.
- **Exports** go through views and `Report`; no result needs its own `to_excel` or `to_markdown`.

Result types with provenance today: fitted equations, systems, VARs and VECMs, state-space fits,
ARDL, Markov switching, MIDAS and model solutions. Not yet: temporal disaggregation, quantile
processes, rolling fits, local projections, impulse responses and decompositions, linearised
solutions and state-space systems, backtests and test results.

## Notation

Equations can be shown in **algebraic notation** as an option (`pymetrix.notation`):

- **Methods:** `Equation.algebraic()` and `FittedEquation.algebraic(estimates=True)` give Unicode
  (for windows, tables, Word and Excel) or LaTeX (for papers and slides).
- **The option:** `set_notation("algebraic")`, or **View → Algebraic notation** in the desktop app,
  titles summaries with the equation and labels coefficients β₀, β₁ and so on.
- **In notebooks,** the option also typesets equations.
- **The Equation tab:** every fitted equation's window has one, showing the code, algebraic,
  estimated and LaTeX forms.
- **Greek names:** written as Unicode in formulas (`c ~ α + β*y`) or spelled out (`alpha`, `beta`,
  `kappa`, `phi`), they display as Greek letters. A parameter's digits and anything after an
  underscore become subscripts (`b1` is b₁, `phi_pi` is φ_π).
- **Own symbols:** `Param.symbol` and `Variable.symbol` override the default symbol.
- **Time** is a subscript (`c[-1]` is cₜ₋₁), and `d`, `dlog` and `pch` show as Δ, Δlog and %Δ.

Notation is display only (principle 4): scripts and projects are the same whichever notation is on.

## The model layer: what comes next

These are the planned pieces above, in the order they depend on each other.

1. ~~`Model.compile()` and `Model.check()`~~ **Done** (day 2): one `ModelSystem` used by every
   solution path, and a diagnostics table before solving.
2. ~~Parameter status and `ParameterSet`~~ **Done** (day 2): values with status, standard
   errors and `Param` metadata; `fix(...)` for estimation.
3. ~~Measurement maps~~ **Done** (day 3): equations in the model's language, with errors,
   sources, units and labels, linearised for the filter and exact for reporting.
4. ~~`StateSpaceSystem` as a result~~ **Done** (day 3): from `solution.system(measurement)`,
   with views, dialogs and project support.
5. ~~Template models~~ **Done** (day 3): New Keynesian, RBC and a small gap model.
6. ~~Estimation of linearised models~~ **Done** (day 4): `Model.estimate` and
   `Template.estimate`, by maximum likelihood or, with priors, the posterior mode. It is built
   from steps 1 to 4 without new machinery: parameter status says what is free, the measurement map
   says what is observed, and the state-space system gives the likelihood. Estimating a model
   gives a `FittedModel` with provenance. Full Bayesian sampling (MCMC) is not built.

## What we deliberately do not do

- Rename or duplicate user-facing names.
- Make users subclass large model classes. Models are equations; templates are examples to copy,
  not classes to inherit.
- Compile for speed without a benchmark. (Done once a FINEX-sized benchmark showed the need: model equations
  are compiled to NumPy; the interpreter remains the reference. JAX and Numba are not used.)
- Add regime-switching model systems before a user needs them.
- Give any assistant a way to change a session other than running commands the user approves.
