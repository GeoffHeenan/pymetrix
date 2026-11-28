# Heterogeneous-agent macro models: the `pymetrix-ha` plugin

Design note, 28 November 2026 · a proposal for the owner to approve or change; nothing here is built · builds on
the modelling side as it is (`src/pymetrix/models/`), [plugin-split.md](plugin-split.md) ("Heterogeneous-agent
(HANK) models", which first proposed wrapping sequence-jacobian), the extension API ([plugins.md](plugins.md),
`pymetrix.plugins.API` 1.10) and the adapters ([adapters.md](adapters.md))

## The question

pymetrix solves and estimates representative-agent models: linear and nonlinear rational-expectations models,
perfect foresight, OccBin, second-order perturbation, Bayesian estimation. Research departments increasingly use
models in which households differ in income and wealth: Krusell and Smith (1998) for capital, and HANK models
(Kaplan, Moll and Violante 2018) for monetary and fiscal policy, because who holds debt or receives a transfer
changes the aggregate response. How should pymetrix add them, so that a user gets a steady state with its
distribution, impulse responses, decompositions, nonlinear transitions and estimates as pymetrix results, with
provenance and replay?

## Short answer

- **Wrap sequence-jacobian in a plugin, `pymetrix-ha`.** The package (shade-econ, MIT licence) implements the
  sequence-space Jacobian method of Auclert, Bardóczy, Rognlie and Straub (2021), the standard method for these
  models. It is small, validated by its authors, and still works with today's Python and NumPy (checked below).
  It is dormant, not dead: one release, January 2022. The plugin pins it, and keeps a fork under its MIT licence as
  the fallback if it ever stops working. A native rewrite is not justified while it works.
- **A plugin, not core.** It brings numba, a heavy dependency that supports new Python versions late. Its audience
  is research departments, not forecasting desks. Its tests are slow. These are the three reasons
  [plugin-split.md](plugin-split.md) gives for a plugin.
- **The same results as the DSGE models.** Impulse responses are pymetrix `IRF` objects. Paths are Datasets.
  Estimates are fits whose `mcmc()` is pymetrix's own sampler. So the result windows, figures, reports, scenario
  sets, provenance and the lab book work unchanged.
- **Models written as Python blocks first**, as sequence-jacobian's users write them. Model text for the aggregate
  equations, with a library of standard household blocks, is a later phase.
- **First deliverable:** Krusell–Smith's steady state, general-equilibrium Jacobians, impulse responses and a
  nonlinear transition, matched to the package's published notebook to the printed digits.

## Conventions in this note

- *(verify)* marks a fact about sequence-jacobian or another program that must be checked against its code or
  documentation before it is relied on.
- **Gap** marks something this plan does not provide. Each becomes a `docs/known-gaps.md` item when the plugin is
  built.
- References give no DOIs yet. Crossref could not be reached when this note was written, and CONTRIBUTING.md says
  never to type one from memory. `scripts/check_references.py --metadata` checks each before it goes into a help
  topic.

## 1. What pymetrix's modelling side does today

| Area | Where | What it does |
| --- | --- | --- |
| Model text | `models/source.py` (`ModelSource`) | pymetrix's syntax (sections `parameters`, `shocks`, `exogenous`, `variables`, `equations`, `measurement`, `reported`, `regimes`), Dynare's, IRIS's and EViews's; checked as typed; built into a `Template` or `DynareModel` |
| The compiled model | `models/system.py`, `models/fast.py` | equations as residuals with symbolic parameters, the lead and lag structure, blocks; residuals and derivatives compiled to NumPy once per form |
| Steady state and first order | `models/linear.py` | `A0 E x(t+1) + A1 x(t) + A2 x(t-1) + B e(t) = 0`, solved by QZ (Klein 2000) for `x(t) = P x(t-1) + Q e(t)`; Blanchard-Kahn checked; `irf`, `simulate`, `moments`, `fevd` |
| Second order | `models/perturbation.py` | Schmitt-Grohé and Uribe's second-order terms, pruned simulation, the risk correction |
| Backward-looking models | `models/model.py`, `models/solvers.py` | period by period, Newton or Gauss-Seidel; stochastic simulation |
| Perfect foresight | `models/perfect_foresight.py` | every period stacked, sparse Newton; known-date and surprise regimes |
| Occasionally binding constraints | `models/occbin.py`, `models/occbin_filter.py` | OccBin's piecewise-linear solution, its filter and smoother |
| State space | `models/measurement.py`, `linear.py`'s `system` | measurement maps from data series to model variables; the state-space form; the Kalman filter and smoother |
| Estimation | `models/estimation.py`, `bayes.py`, `bayes_nuts.py`, `kalman_score.py` | maximum likelihood or posterior mode through the Kalman filter, with an analytic gradient; Metropolis-Hastings, NUTS, SMC; Iskrev identification; DSGE-VAR |
| Results | `pymetrix.var.IRF`, `models/decompose.py`, `Solution` | responses `values[h, response, shock]`; shock decompositions; paths as Datasets with provenance |

`docs/known-gaps.md` lists the modelling gaps: no third-order perturbation, moments from the linear solution only
under OccBin, regimes only without growth trends, NUTS slow on large models. **It has no heterogeneous-agent item,
because nothing exists.** The plugin-split note records the same: "PyMetrix has none today."

What carries over: the IRF and figure classes, measurement maps, the samplers (`bayes.mcmc` needs only
`fit.log_posterior`, `fit.params` and `fit.priors`), provenance, scenario sets and reports. What does not: the QZ
solution and the Kalman filter assume a finite state vector. A heterogeneous-agent model's state includes a
distribution over thousands of grid points, so it needs a different solution method.

## 2. The method

A heterogeneous-agent model has three parts. The sequence-space method treats each as a block of a directed
acyclic graph (DAG) that maps sequences of inputs to sequences of outputs.

### The steady state

- **The household problem by the endogenous grid method** (Carroll 2006). Given prices, iterate the Euler
  equation backwards on a grid of assets and income states. Each step inverts the Euler equation at next period's
  grid points instead of solving a root-finding problem at each point. It stops when the marginal value of assets
  converges.
- **The stationary distribution by Young's histogram method** (Young 2010). Each agent's chosen assets fall between
  two grid points. The mass is split between them in proportion to the distance (a lottery), then moved by the
  income process's Markov matrix. Iterating this map converges to the stationary distribution, with no simulation
  noise.
- **Calibration.** An outer solver finds the unknowns (a discount factor, say) that hit the targets (an interest
  rate, market clearing).

### Linearised dynamics: sequence-space Jacobians

Auclert, Bardóczy, Rognlie and Straub (2021):

- **Blocks.** A *simple block* is a few aggregate equations, with lags and leads (`K(-1)`): a firm, a Taylor rule,
  market clearing. A *het block* is the household problem: a backward step (EGM), a policy, the distribution's law
  of motion, and the aggregates it reports. A *solved block* is a small set of equations solved for its own
  unknowns (a price-setting block). The model is the DAG of blocks, from exogenous shocks and the equilibrium's
  unknowns to its targets.
- **Each block's Jacobian.** For a simple block, its derivatives at the steady state, which are sparse. For a het
  block, the *fake-news algorithm*: one backward iteration per input gives every column of the T×T Jacobian at a
  cost of about one simulation, instead of T. Jacobians are `T × T` matrices: the response at date s of an output
  to a change at date t of an input.
- **General-equilibrium Jacobians.** Chaining the block Jacobians along the DAG gives `H_U` (targets on unknowns)
  and `H_Z` (targets on shocks). The equilibrium is `H_U dU + H_Z dZ = 0`, so `G = -H_U^-1 H_Z` maps any shock path
  to the unknowns' paths, and every other output follows by the chain rule.
- **Impulse responses.** `dY = G dZ` for any path of the shock: an AR(1), news at date 10, or a path from a
  scenario. Once `G` is known, each response is a matrix product.
- **Estimation.** In the linearised model each observable is an MA(T−1) process in the shocks, with the impulse
  responses as its coefficients. The autocovariances follow by FFT. The exact Gaussian likelihood of the data
  follows from them (the paper's appendix B.5; `sequence_jacobian.estimation`). When only the shock processes and
  measurement errors change, `G` is reused, and one likelihood evaluation costs milliseconds.
- **Nonlinear perfect-foresight transitions.** The same DAG gives the nonlinear transition after an unexpected
  shock (an "MIT shock", Boppart, Krusell and Mitman 2018). Newton's method on the unknowns' paths uses `H_U` as
  its Jacobian, so it converges in a few iterations.

This is first order in aggregate shocks, and exact in the distribution's response to them. It is the method of
choice in the literature since 2021, and the one central banks' HANK models use *(verify which, before the help
says so)*.

## 3. Library first: options

| Option | What it is | For | Against |
| --- | --- | --- | --- |
| **A. Wrap sequence-jacobian** (recommended) | `pip install sequence-jacobian`; MIT; the paper's authors' code | the reference implementation of the method; steady state, Jacobians, IRFs, nonlinear transitions, likelihood; standard household blocks; tested by its authors | dormant since 2022; numba; some rough edges (below) |
| B. A native core | EGM, Young's method, the fake-news algorithm and the GE solve written in pymetrix, in NumPy | no numba; pymetrix's own API, provenance and defaults throughout | about 3,500 lines to write, validate and maintain; it would reimplement a mature library, which CONTRIBUTING.md allows only when none fits |
| C. econpizza | Boehl's package: JAX, MIT, active (0.6.10, June 2026); YAML model files | nonlinear HANK, and models with occasionally binding constraints; automatic differentiation | JAX as a dependency; 0.x and changing; built around nonlinear transitions, not the sequence-space Jacobian workflow |
| D. Econ-ARK (HARK) | household-problem toolkits; active (0.17.2, May 2026) | strong on micro consumption models | not a general-equilibrium workflow |
| E. Dynare | recent versions add heterogeneous-agent features *(verify their state)* | pymetrix already checks against Dynare | needs Octave or MATLAB; not a Python library to wrap |

### sequence-jacobian, checked (28 November 2026)

- **Releases.** One on PyPI: 1.0.0, 10 January 2022. The repository's `master` has 47 commits in 2022, the last
  release-worthy one on 21 February 2022 (fixes to stage blocks and solver options), then two in 2024 (the
  notebooks' zip, and CI on Python 3.9 to 3.11, 9 September 2024). Nothing since. **Dormant**, not archived.
- **Licence.** MIT. A fork, if ever needed, keeps the licence and attribution.
- **Metadata.** The PyPI release declares no dependencies (`Requires:` is empty). Its `setup.cfg` asks for
  `numpy >= 1.19.2`, `scipy >= 1.2` and `numba >= 0.49`. The plugin declares them itself.
- **Works today.** In a fresh environment with Python 3.13, NumPy 2.5.5, SciPy 1.18.1 and numba 0.68.1, the
  repository's own test suite (40 tests) passes against the PyPI 1.0.0 release, with deprecation warnings only.
  The Krusell–Smith notebook's printed household Jacobian (the 5×5 corner of `J['C']['r']`) is reproduced to all 8
  printed digits by both 1.0.0 and `master`.
- **The API since 1.0.** Stable in the way that matters: `simple`, `het`, `solved`, `create_model`,
  `solve_steady_state`, `solve_jacobian`, `solve_impulse_linear`, `solve_impulse_nonlinear`, `jacobian`,
  `ImpulseDict`, `JacobianDict`, `SteadyStateDict`, and `estimation.all_covariances` and `log_likelihood`. Older
  names (`sj.agrid`, `sj.markov_rouwenhorst`) still work, with a warning.
- **Rough edges the wrapper must handle.**
  - Output names are read from the source text of each block's `return` line (`inspect.getsource` and a regular
    expression). A block defined in text run by `exec`, as the app's Code window runs it, fails with "could not get
    source code". A block that returns an expression rather than a name (`return (1 + r) * c ** -1`) is silently
    misread: its output is not found later (a `KeyError` in the steady state). The plugin checks each block when it
    is added and says what to write.
  - The grid helper matters for the published numbers: the notebook uses `grids.agrid`; the newer
    `grids.asset_grid` gives entries of the same Jacobian up to 0.3% apart. Examples state their grid.
  - Steady-state and transition solvers print their iterations by default. The wrapper turns this off and records
    the iterations in provenance.
  - Library defaults (tolerances 1e-8 and 1e-10 for the backward and forward iterations, 1e-12 for the targets,
    1e-8 and 30 iterations for transitions) are passed explicitly from pymetrix's own spec, as
    [adapters.md](adapters.md) does for every library, so a new library release cannot change a result silently.

### Recommendation

**Option A, as a plugin.** The method is the library's; reimplementing it would be the larger risk. The plugin
depends on `sequence-jacobian>=1.0,<2` and pins a tested version in CI. If a future NumPy, numba or Python breaks
the release and nobody upstream fixes it, the plugin vendors a fork limited to compatibility fixes (MIT allows it).
Option B stays the escape hatch, and the validation plan below would carry over to it unchanged. Option C is a
candidate second engine later, for nonlinear models with occasionally binding constraints (section 9).

## 4. How a user writes a model

### Phase 1 to 3: Python blocks

The blocks are sequence-jacobian's own decorators. The plugin adds a model object that holds the calibration, the
steady-state problem, the equilibrium and the shocks, so that pymetrix's verbs apply:

```python
import sequence_jacobian as sj
from pymetrix_ha import HAModel, AR1, households

@sj.simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@sj.simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt

ks = HAModel(
    [households.one_asset(income="w * e"), firm, mkt_clearing],
    calibration={"eis": 1, "delta": 0.025, "alpha": 0.11, "rho_e": 0.966, "sd_e": 0.5, "L": 1.0,
                 "nE": 7, "nA": 500, "amin": 0, "amax": 200},
    steady_state={"unknowns": {"beta": 0.98, "Z": 0.85, "K": 3.0},
                  "targets": {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}},
    equilibrium={"unknowns": ["K"], "targets": ["asset_mkt"]},
    shocks={"Z": AR1(rho=0.8, sd=0.01)},           # parameters rho_Z and sd_Z, for estimation
    horizon=300,                                    # T: the length of the sequences
)
```

- `households` is the plugin's library of standard het blocks: sequence-jacobian's own (`hh_sim`, `hh_labor`,
  `hh_twoasset`) with their grids and income processes made from named parameters. A user can write their own
  with `@sj.het`.
- The model checks each block as it is added: output names readable, every input either calibrated, an unknown,
  a shock or another block's output, the DAG acyclic. A problem names the block and the input.
- A script or notebook is the place for this. In the app, the model is a `.py` file attached to the project
  (`session.attachments["pymetrix-ha"]`), so it saves, replays and can be read. Running it is running typed code,
  under the project-trust rules that already apply to Code-window steps.

### Phase 4: model text

The aggregate equations are what most users change, and pymetrix's model text already reads equations with lags
and leads. A model file would add two sections to pymetrix's syntax:

```
households
    hh = one_asset(income = w * e, grid = asset_grid(0, 200, 500), income_process = rouwenhorst(0.966, 0.5, 7))
equations
    firm_r: r = alpha*Z*(K[-1]/L)^(alpha - 1) - delta
    firm_w: w = (1 - alpha)*Z*(K[-1]/L)^alpha
    asset_mkt: 0 = hh.A - K
steady state
    beta = 0.98 targets r = 0.01
equilibrium
    unknowns K  targets asset_mkt
```

The equations become simple blocks: the plugin writes their Python into the project's attachments, which also
solves the source-text problem above. The household line chooses a standard block. A model file cannot write a
new household problem; that stays Python.

## 5. What the plugin provides

The four verbs, applied to an `HAModel`:

| Call | What it does | sequence-jacobian | Returns |
| --- | --- | --- | --- |
| `ks.steady_state()` | the calibration solved: unknowns found, targets met, the distribution and policies | `solve_steady_state` | `HASteadyState` |
| `ks.linearize()` | block Jacobians and the general-equilibrium Jacobians for horizon T | `jacobian`, `solve_jacobian` | `HALinear`, holding `G` |
| `lin.irf(periods=40)` | responses to each shock in `shocks`, by their AR(1) or a path given | `G @ dZ` | pymetrix `IRF` (`values[h, response, shock]`) |
| `lin.moments()`, `lin.simulate(n, seed=)`, `lin.fevd()` | from the MA representation | `estimation.all_covariances` | `Moments`, Dataset, `FEVD` |
| `lin.decompose("C", shock="Z")` | an output's response split by the inputs of the block that makes it: consumption into the effects of `r`, `w`, taxes, dividends | the household Jacobians times the input paths | a decomposition, as `models/decompose.py` gives |
| `ks.solve({"Z": path}, nonlinear=True)` | the nonlinear perfect-foresight transition after an unexpected shock | `solve_impulse_nonlinear` | `Solution` (a Dataset of paths with provenance), with the linear path for comparison |
| `ks.estimate(data, measurement, estimate=[...], priors=...)` | the likelihood of the MA representation, maximised; then `fit.mcmc()` | `estimation.log_likelihood` | `FittedHAModel` |

Shocks are AR(1) processes by default (`AR1(rho, sd)`), or a path given. `measurement` is pymetrix's
`MeasurementMap`, linearised as it already is for DSGE models: an observable's MA coefficients are the map applied
to the responses.

### Estimation

The likelihood is the paper's: the observables' autocovariances from the impulse responses, then the exact
Gaussian likelihood of the stacked data by a Cholesky factor (`sequence_jacobian.estimation.log_likelihood`).
`FittedHAModel` keeps the `log_posterior` (log likelihood plus the priors, which are pymetrix's own `priors`), the
`params` at the mode with the inverse Hessian, and the `priors`. That is all `bayes.mcmc` reads, so
`fit.mcmc(draws=...)` and the `Posterior` views, HPD bands and marginal likelihood work unchanged. What each
evaluation recomputes depends on which parameters are estimated:

| Parameters estimated | Recomputed at each evaluation | Cost (Krusell–Smith, T = 300) |
| --- | --- | --- |
| shock persistence and size, measurement errors | the responses `G dZ` and the likelihood | milliseconds |
| aggregate blocks' parameters (a Taylor rule, price stickiness) | the simple blocks' Jacobians and `G`; the household Jacobians kept from the steady state | tenths of a second |
| household parameters (`beta`, `eis`, the income process) | the steady state and the household Jacobians | seconds: maximum likelihood is possible; MCMC is not practical (**gap**) |

Missing observations are allowed: the rows and columns of the stacked covariance for the missing values are left
out before the Cholesky factor. The paper's code needs complete data; this is the plugin's own few lines, checked
against the Kalman filter (section 7).

Gradient-based samplers (NUTS) need the likelihood's gradient, which the library does not give. Random-walk
Metropolis-Hastings and SMC work; NUTS uses a numerical gradient, and is slow (as `docs/known-gaps.md` already says
for large DSGE models).

## 6. What results look like

- **The steady state** (`HASteadyState`):
  - a table of the aggregates; the calibration's targets with their residuals; the unknowns found;
  - the distribution: the wealth histogram by income state, the Lorenz curve and Gini coefficient, wealth shares
    (bottom 50%, top 10%, top 1%), the share at the borrowing limit;
  - policy functions, consumption and savings against assets, one line per income state;
  - marginal propensities to consume: the intertemporal MPCs (the first column of the household Jacobian of
    consumption on income), and the MPC by wealth;
  - checks: the largest Euler-equation error at interior grid points, the distribution's mass, and the untargeted
    market (Walras' law) residual.
- **Dynamics:**
  - impulse responses, as pymetrix `IRF` objects, so the IRF window, `plot`, Selected horizons and reports work as
    for a DSGE model;
  - the general-equilibrium Jacobians as heat maps (`heatmap_figure`): which dates of a shock move which dates of
    an output;
  - decompositions as stacked bars, as shock decompositions are drawn: consumption's response by channel (direct
    effect of the interest rate, wages, taxes);
  - distributional responses: the consumption response by wealth decile, from the household block's internals
    along the path *(verify which internals `solve_impulse_nonlinear` returns)*;
  - linear and nonlinear transitions on one figure, for a check on size effects.
- **Estimates:** a `FittedHAModel` with its parameter table, log likelihood and posterior mode, and `mcmc()`'s
  `Posterior` with its usual views.
- **In a scenario set:** a transition is a `Solution`, so it can join a `ScenarioSet` beside a DSGE model's
  scenarios, and a `policy_brief` report shows both.

## 7. In the app

The plugin follows the extension API 1.10 and the existing first-party plugins (`pymetrix-eviews`,
`pymetrix-finance`).

- **Model → Heterogeneous agents…**, a dialog with a stable ID in `command_ids.json`:
  - *Model:* an example (Krusell–Smith, one-asset HANK, two-asset HANK) or a `.py` file of blocks;
  - *Calibration:* an editable table of the calibrated values;
  - *Steady state:* its unknowns with starting values, and its targets;
  - *Equilibrium:* unknowns and targets, the horizon T;
  - *Shocks:* each with its persistence and size.

  Buttons run the steady state, impulse responses, a nonlinear transition and estimation. Each is a recorded step,
  so the History and the script hold it.
- **Long work in the background.** A steady state takes about a second, a two-asset model's Jacobians several, an
  estimation minutes. Each runs with the app's progress reporting (`pymetrix.progress`) and can be cancelled.
- **Results** open in the Results area with the views in section 6. The DAG is drawn as a figure (blocks as
  boxes, inputs and outputs on arrows), with matplotlib, since sequence-jacobian's own drawing needs Graphviz.
- **Help:** a topic for the plugin, linked from the dialog, with the method, the blocks, how to write a household
  block, the checks and the references. **Help → Plugins** lists it and its library's version.

Package layout, as the other plugins:

```
packages/pymetrix-ha/
  pyproject.toml          dependencies: pymetrix, sequence-jacobian>=1.0,<2, numba, numpy, scipy
                          entry points: pymetrix.plugins, pymetrix.commands, pymetrix.models
  src/pymetrix_ha/
    __init__.py           PLUGIN = PluginInfo("pymetrix-ha", ...); register()
    model.py              HAModel, AR1; the checks on blocks
    households.py         the standard het blocks, with their grids
    steady.py             HASteadyState and its views
    linear.py             HALinear: irf, moments, simulate, fevd, decompose
    nonlinear.py          transitions as Solutions
    estimation.py         the likelihood, FittedHAModel
    examples/             krusell_smith.py, hank_one_asset.py, hank_two_asset.py
    app.py                the dialog
  tests/                  fixtures/ (numbers recorded from the notebooks), test_*.py
```

## 8. Provenance

Every result records, in `provenance`:

- **backend:** `{"name": "sequence-jacobian", "version": "1.0.0", "wraps": "sequence_jacobian"}`, and numba's and
  NumPy's versions;
- **the specification:** the calibration; each block's name, inputs and outputs, and a fingerprint of its source
  text (blocks are code, so the code is part of the model); the steady-state and equilibrium unknowns and targets;
  the grids; the horizon T; the shocks' processes;
- **`resolved`:** every tolerance and iteration limit passed to the library (pymetrix passes them all, so none is
  the library's silent default); the iterations each solve took; the residuals at the solution; the largest
  Euler-equation error; the distribution's convergence;
- **warnings:** the library's warnings, captured, as for every wrapped library.

A project keeps the blocks' source text and the calibration, and rebuilds the model when it opens. Replaying it
needs the plugin, as a PSETS project needs `pymetrix-psets`.

## 9. Validation

Every number has an independent check, in CONTRIBUTING.md's order of preference.

1. **The library called directly.** Each wrapper's result equals sequence-jacobian's own call on the same model,
   in the test: the steady state, `G`, the responses, the transition and the likelihood (1e-12).
2. **The published notebooks' numbers.** The package's notebooks (Krusell–Smith, one-asset HANK, two-asset HANK, at
   the repository's commit `eb4ac4c`, 9 September 2024) print some results; the paper's figures are drawn from the
   same code. Recorded and checked:
   - Krusell–Smith: the 5×5 corner of the household Jacobian `J['C']['r']` (8 printed digits; reproduced, section
     3); the steady state;
   - one-asset HANK: the market-clearing residuals and the nonlinear solver's error at each iteration
     (`1.46E-02`, ...);
   - two-asset HANK: liquid assets `1.04` and the market-clearing residuals, and the nonlinear solver's errors.

   The plugin's first deliverable is Krusell–Smith matched to these numbers.
3. **The representative-agent limit, against pymetrix and Dynare.** A model with no het block is an ordinary DSGE
   model, so two independent methods must agree:
   - the RBC model of sequence-jacobian's first notebook, written once as simple blocks and once in pymetrix's
     model text: sequence-space responses against pymetrix's QZ solution (1e-8 at horizons well inside T);
   - the same model's nonlinear transition against `PerfectForesight` (1e-8);
   - the MA likelihood of simulated data against pymetrix's Kalman-filter likelihood (1e-6 relative: the MA
     representation is truncated at T), which also checks the missing-data rows;
   - pymetrix's RBC is already checked against Dynare 6.0's recorded output (the `dynare-reference` workflow), so
     the chain reaches Dynare.
4. **Hand checks.**
   - Krusell–Smith's calibration fixes the capital stock and wage by hand: `K = alpha Y / (r + delta)` =
     0.11 / 0.035 = 3.142857 and `w = (1 - alpha) Y / L` = 0.89, which the library's steady state gives.
   - The distribution has mass 1 and no negative entries; the aggregates are the distribution's sums of the
     policies; the untargeted market clears by Walras' law (1e-8).
   - The stationary distribution is the eigenvector, for eigenvalue 1, of Young's transition matrix, found by
     SciPy's sparse eigensolver: an independent route to the forward iteration (1e-8).
   - The fake-news Jacobian against the direct method (one transition per input date) on a short horizon (1e-6).
   - A small shock's nonlinear transition converges to the linear response as the shock shrinks.
   - Away from its edges, `G` is close to Toeplitz: a news shock moved one date later moves the response one date
     later.
5. **Recovery.** Shock persistences and sizes estimated from data simulated by the model, over 50 samples:
   the true values inside the posterior bands at their nominal rate.

Each check gets a row in `docs/validation.md`, with sequence-jacobian's version.

## 10. Performance

Measured in this note's checks on one shared core of a 4-core machine (rough, not a benchmark):

| Model | Steady state | `G`, T = 300 | Nonlinear transition |
| --- | --- | --- | --- |
| Krusell–Smith, 7 income states, 500 asset points | 0.5 s (1 s with numba's compilation) | 2 s | 3 s (4 Newton iterations) |
| One-asset HANK (the package's test calibration) | 1.7 s with compilation | 0.9 s | not timed |
| Two-asset HANK (the package's test calibration) | 1.9 s with compilation | 3.7 s | not timed |

- **Compilation.** numba compiles the household routines on first use in each process, about a second. The
  plugin warms it on import in the app's background worker, not in the interface's thread.
- **Memory.** Each Jacobian is T × T: 0.7 MB at T = 300. A two-asset model holds a few dozen.
- **Estimation.** With `G` reused, a likelihood evaluation of 100 quarters of 4 observables takes about 2 ms (the
  notebook: 100 evaluations in 195 ms), so MCMC over shock processes takes seconds to minutes. Household
  parameters are another matter (section 5).
- **Threads.** numba's and BLAS's threads add up; the plugin respects `set_blas_threads` and sets numba's to match.
- **Tests.** The examples are slow: CI runs Krusell–Smith on a small grid in every run and the HANK models in a
  weekly job, as the Dynare and JDemetra+ references do.

## 11. Phases and size

Rough figures for one developer who knows the codebase. Tests add about as much again.

| Phase | What | Lines | Days |
| --- | --- | --- | --- |
| 1 | The plugin's skeleton; `HAModel` over the user's blocks; steady state, `G`, IRFs as `IRF`, nonlinear transitions as `Solution`; provenance; the Krusell–Smith example matched to the notebook; hand checks; help topic | 1,200 | 10–14 |
| 2 | One- and two-asset HANK examples; standard household blocks; decompositions; distributional results; the RA-limit checks against pymetrix and Dynare; the dialog and result views | 1,500 | 15–20 |
| 3 | Estimation: the likelihood with missing data, `FittedHAModel`, `mcmc`; shock and aggregate parameters; recovery checks | 800 | 8–12 |
| 4 | Model text (simple blocks from equations; standard household blocks); transitions on dated data, as satellite scenarios over a forecast round's baseline | 1,000 | 10–15 |
| **Total** | | **about 4,500** | **about 45–60** |

Phase 1 alone is a usable deliverable: Krusell–Smith's steady state and responses, in code, with provenance and
checks.

## 12. Left out

- **Aggregate risk beyond first order.** The method is linear in aggregate shocks (and exact in the
  distribution's response). Krusell and Smith's algorithm with aggregate uncertainty, higher-order or global
  solutions: **gap**.
- **Occasionally binding aggregate constraints** (the zero lower bound) except in nonlinear perfect-foresight
  transitions. An OccBin for HANK: **gap**; econpizza is the candidate if it is wanted.
- **Estimating household parameters by MCMC** (section 5): **gap**. Combining micro data or distributional
  moments with the time-series likelihood: **gap**.
- **Continuous-time models** (the Achdou et al. finite-difference methods), life-cycle and overlapping-generations
  households, discrete choices beyond what the library's own blocks provide.
- **Dynare, IRIS or EViews files** for heterogeneous-agent models. None of the importers reads them.
- **JAX, GPUs and automatic differentiation** of the likelihood.
- **Writing new household problems in model text.** They stay Python.

## 13. Risks

- **The library is dormant.** One release, no change since 2024. Mitigation: pinned versions, the library's own
  tests run in pymetrix's weekly job, and a fork under its MIT licence as the fallback.
- **numba lags new Python releases.** pymetrix supports Python 3.11 to 3.13. If numba is late for a new Python, the
  plugin is late too. Being a plugin keeps this out of core.
- **Blocks are code.** A project with blocks runs code when it opens. The project-trust rules already cover typed
  code; the dialog says so.
- **Silent misreads of blocks** (outputs read from source text). The plugin's checks on each block catch them
  before solving.
- **Fragile steady states.** Calibrations fail to converge from poor starting values. The dialog keeps the last
  solved steady state as the next start, and errors name the unknown and the target.
- **Scope creep.** The literature moves fast. The phases list what is in; the rest is listed above.

## 14. Open questions for the owner

1. **Wrap sequence-jacobian (option A), or write it natively (B)?** *Recommendation:* A. It works on today's
   stack, and the native route stays open with the same tests if it stops working.
2. **Its own plugin, `pymetrix-ha`, or the `pymetrix-macro[full]` extra that plugin-split.md sketched?**
   *Recommendation:* its own plugin now. `pymetrix-macro` does not exist, and the split waits for the macro
   interfaces to settle. It can move into `pymetrix-macro` later.
3. **The release or `master`?** PyPI's 1.0.0 lacks `master`'s February 2022 fixes to stage blocks and solver
   options, which phases 1 to 3 do not use. *Recommendation:* depend on the release, and ask the authors for a
   1.0.1 that includes `master` and declares its dependencies; vendor only if that fails and something breaks.
4. **Python blocks first, model text later (phase 4)?** *Recommendation:* yes. Blocks are how the method's users
   write models now; model text helps most for the aggregate equations, and needs the converter.
5. **In `pymetrix[all]`?** numba is large and supports new Python versions late. *Recommendation:* no; offer
   `pymetrix[ha]`, as `pymetrix[eviews]` is offered, and leave it out of `[all]` until numba has wheels for every
   Python that CI tests.
6. **Estimation scope.** *Recommendation:* shock processes and aggregate-block parameters in phase 3; household
   parameters by maximum likelihood only, listed as a gap for MCMC.
7. **Priority.** plugin-split.md ranks HANK below closing gaps with Dynare and the forecasting extras.
   *Recommendation:* keep that order. Phase 1 is small enough to do as a demonstration for research departments,
   and the rest waits for reviewers' views.

## References

- Auclert, A., Bardóczy, B., Rognlie, M. and Straub, L. (2021). Using the sequence-space Jacobian to solve and
  estimate heterogeneous-agent models. Econometrica 89(5), 2375-2408.
- Boppart, T., Krusell, P. and Mitman, K. (2018). Exploiting MIT shocks in heterogeneous-agent economies: the
  impulse response as a numerical derivative. Journal of Economic Dynamics and Control 89, 68-92.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems.
  Economics Letters 91(3), 312-320.
- Kaplan, G., Moll, B. and Violante, G. L. (2018). Monetary policy according to HANK. American Economic Review
  108(3), 697-743.
- Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational expectations model.
  Journal of Economic Dynamics and Control 24(10), 1405-1423. doi:10.1016/s0165-1889(99)00045-7
- Krusell, P. and Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. Journal of Political
  Economy 106(5), 867-896.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith
  algorithm and non-stochastic simulations. Journal of Economic Dynamics and Control 34(1), 36-41.

The DOIs of all but Klein (2000), whose string is reused from the help, are left out until
`scripts/check_references.py` confirms them.
