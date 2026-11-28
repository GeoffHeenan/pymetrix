# Known gaps

pymetrix is at version 0.1.0a21, a development build. This page lists what is missing or rough, so nobody has to
discover it the hard way. The [design document](design.md) has a "Not yet" list for each area;
this page collects the ones that matter most to a first-time user. If something here is
blocking you, say so: that is how priorities get set.

## Project and release

- **A development build, pre-release on PyPI.** pymetrix 0.1.0a21 is on PyPI (`pip install --pre pymetrix`), and so
  are the plugins pymetrix-eviews, pymetrix-seasonal, pymetrix-finance, pymetrix-guide, pymetrix-labbook and
  pymetrix-assistant (0.1.0a1, `pip install --pre pymetrix-seasonal`, say); pymetrix-psets is installed from the
  repository. Alpha versions: APIs may still change. No installer yet; see [installing pymetrix](install.md). A one-step installer for
  locked-down Windows machines is not started; `pymetrix-gui --shortcut` makes a desktop shortcut once installed.
- **One developer so far.** Most code was written quickly, with AI assistance, and checked
  against other implementations (see [validation](validation.md)). It has not yet been reviewed
  by anyone else, and some internals will change.
- **Packaging.** CI checks on every change that the wheel installs and passes the whole test
  suite in a fresh environment, and that every dependency works at the oldest version
  `pyproject.toml` allows (Python 3.11). The course labs are in the repository, not in the installed package.
- **The API may change** before 0.1. Saved projects are versioned, and older ones are
  migrated when opened, so projects should keep opening as it changes.
- **Documentation** is the README, the [user guide](guide.md), the in-app help, the course packs and a
  [command and function reference](reference/reference.md) generated from the code; there is no hosted site yet.

## Methods

- **Applied microeconometrics (double ML, shift-share IV, interactive fixed effects, conditional logit):** double ML
  takes one treatment at a time, the partialling-out score only for the partially linear models (not the IV-type
  score), one-way clusters (not DoubleML's multiway), regression (not classification) learners for a 0/1 treatment in
  the partially linear models, no tuning of the learners' settings, no joint confidence bands or multiplier
  bootstrap, and no sensitivity analysis; IRM's propensities are trimmed, not normalised (Hajek). Shift-share IV has
  one endogenous regressor and one shift-share instrument (no overidentified or multiple-shock designs), no
  region-level clustering beside the shock-level errors, no shock-level controls in the BHJ regression (it has no
  constant, which is what makes it equal the regional IV; control for the sum of shares at the regional level), no
  AKM degrees-of-freedom correction, and Rotemberg weights without each share's first-stage F; AKM needs no more
  industries than regions. Interactive fixed effects need a balanced panel without gaps; the bias correction covers
  heteroskedasticity across units and over time only (not serial or cross-sectional correlation, Bai's D term), and
  the number of factors is chosen from the residuals of the fit with the most factors; there is no dynamic-panel
  (Moon-Weidner) version or CCE comparison in one result. The conditional logits give no nested or mixed logit (IIA
  is assumed), no ranked or multiple choices per case (a case with other than one choice is dropped), and the panel
  logit's recursion is in pure Python, slow for long groups. Their marginal effects are derivatives and discrete changes
  (0/1 variables from 0 to 1, or any variable by a chosen amount from its value) averaged over the cases or at the
  means, with no effects at other user-chosen values (Stata's `at()`), no discrete change between two arbitrary
  values (x0 to x1) and no second differences (interactions); the panel logit has odds ratios, with effects and
  discrete changes only at a fixed effect the user chooses (the probabilities depend on the unidentified fixed
  effects).
- **Robust regression and restricted least squares:** `RobustLS` is M-estimation only (no S- or MM-estimators, so no
  protection against outlying regressors), with Huber's and Tukey's bisquare norms (not Hampel, Andrews or others),
  the MAD scale (not Huber's proposal 2) and the H1 covariance (not H2 or H3, nor robust or clustered sandwiches);
  linear-form equations without fixed effects or weights. Restrictions on OLS and IV are linear and on linear-form
  equations only (substitute nonlinear ones, or restrictions on explicit-form parameters, into the equation), and are
  not offered on GMM, ML or the wrapped estimators; no test of the restrictions is reported with the fit (use
  `fit.test("wald", ...)` on the unrestricted fit). A coefficient a restriction fixes has a zero standard error, so
  its t statistic is undefined.
- **GMM on moment functions** (`MomentGMM`) is code only, with numerical derivatives, and has no dialog (a Python
  function cannot be a dialog field).
- **Regression without a formula** (`pymetrix.ols`) is least squares only (no IV, ML or fixed effects to absorb:
  put dummies in `X`, or use a formula), with the standard errors `OLS` has (White is HC1; no HC2 or HC3, as in
  `Equation.fit`). It is for code, with no dialog. Column names must be plain series names (letters, digits and
  `_`); other names are refused, not rewritten. Data without a `PeriodIndex` (or dates with a frequency) are a
  cross-section, so HAC and Durbin-Watson need one. A row whose values are all present but infinite is dropped with
  the reason "lag", as a formula fit records it.
- **Breusch-Godfrey** offers pymetrix's convention and statsmodels' (`auxiliary_constant=`); the F form, and
  variants that drop the first `lags` observations rather than set their lagged residuals to zero, are not offered.
- **Stochastic simulation** draws equation errors (and optionally coefficients) for backward-looking models, and
  shocks for first-order solutions; not yet: nonlinear models with expectations (perfect foresight is one path),
  cross-equation coefficient covariance for systems estimated jointly, or targets with first-order draws.
- **ARMA errors with GARCH variance** take normal or t errors (not skewed-t or GED), by conditional likelihood (the
  first p periods conditioned on). Dynamic forecasts with regressors that depend on the dependent variable other
  than as lags of the left-hand side (`y ~ d(y)[-1]`) have no standard errors; stochastic simulation of the equation
  as a model gives them.
- **Distribution and moving-window functions:** no other families (Weibull, negative binomial, noncentral t,
  chi-squared or F). No derivatives in the degrees of freedom of t, chi-squared and F, the shape parameters of gamma
  and beta, or a binomial's n: linearising a model in one of those (or estimating it by a method that needs that
  derivative) is refused. Moving minima and maxima, and the integer quantiles `poissinv` and `binoinv`, are not
  differentiable either. Only `normcdf`, `normpdf` and `erf` can be written to Dynare or IRIS. Moving windows
  (trailing or centred) are of a fixed length up to 1000 periods, with no `min_periods`; expanding windows (`cumavg`
  ...) work on data only (not in models or estimation by arrays), have no derivative, and skip missing values as
  pandas' `expanding()` does.
- **Co-skewness and co-kurtosis** use the periods every series has (no pairwise version) and the T denominator only
  (no bias-corrected co-moments, so their diagonal is g1 and b2 rather than `describe()`'s G1 and G2); no
  shrinkage or structured (single-factor, constant-correlation) estimators of M3 and M4, and no portfolio
  skewness or kurtosis from them. The full M4 has n³ columns: thirty series give 27,000.
- **The random generator** (`set_random_generator`) chooses among NumPy's bit generators (PCG64, MT19937, Philox,
  SFC64); none reproduces MATLAB's, Dynare's or Stata's draws, even MT19937 with the same seed (seeding and the
  normal transform differ). Paths through a library's `np.random.RandomState` (`arch` bootstraps, statsmodels'
  Markov-switching starts, scikit-learn's `random_state`) stay Mersenne Twister whatever is chosen; the provenance
  of a scikit-learn learner still names the chosen generator, which covers pymetrix's own draws (the bagging
  bootstrap) only. Steps of a project made before the setting existed replay with PCG64, as they were made.
- **Matrices with names** are dense, two-dimensional and real or complex; there are no sparse or three-dimensional
  ones. `qz` gives the real form without reordering (no `ordqz`), and the Riccati solver is discrete only. In the
  app, a matrix window's Matrix menu offers the common steps (transpose, inverse, Cholesky, SVD, eigenvalues, QR, LU,
  Schur, power, Lyapunov); QZ, the Sylvester and Riccati equations, `expm`/`logm`/`sqrtm`, null spaces and the
  matrices of matrix calculus are written as expressions (Model → Matrix → Expression…). An expression is one Python
  expression of names, calls, operators, indexing and literals: no comprehensions, conditionals or `lambda`.
- **Calculation:** only sweeps and scenario lists run in parallel; other steps run one at a time, and
  typed code that writes reports cannot be checked for out-of-date inputs.
- **Trending models:** steady states with growth use additive trends; variables in levels growing by a factor
  are handled by writing them in logs (`model.in_logs`). Second-order solutions of growing models are not
  available.
- **Fixes and holds in some periods** (`Scenario(partial=...)`): perfect foresight takes partial fixes but not
  partial holds, and solves them with the interpreted stacked system (slower for a large model); optimal control
  takes neither. `Solution.convergence` is the deterministic solution's: a stochastic simulation's draws report only
  how many failed. A period that does not converge still stops the solve (the error carries the table).
- **Solver options** (`convergence=`, `line_search=`, `refresh=`, `Fallback`, `ordering="blocks"`) are for the
  period-by-period solvers. Perfect foresight keeps its own residual test and line search, and the draws of a
  stochastic simulation are solved by Newton with the residual test, whatever the solver (its `tol` and `max_iter`
  only). `ordering="blocks"` is refused with links between entities and with regimes. A recursive equation is solved
  once only when it is linear in its own variable; one that is not (`log(q) = ...`) is iterated on as a block of one.
  With blocks the convergence table's residual is the largest left in the blocks iterated on, not recomputed for
  the equations solved once. The Solve dialog's Fallback is one fixed sequence (damped Gauss-Seidel, then Newton with
  a line search); other sequences are written in code.
  In wall time `ordering="blocks"` is not faster on every model: each equation solved once is a call of its own, so
  on a model of a couple of hundred compiled equations a joint Newton solve takes about as long, though it evaluates
  several times as many equations (`benchmarks/solve_options.py`: on its generated model, 0.19 s against 0.15 s at
  200 equations, 0.69 s against 1.03 s at 600, on one busy machine).
- **Solve timing** (`Solution.convergence.attrs["seconds"]`) is recorded by the period-by-period solvers only, not by
  perfect foresight or a first-order solution, nor for each draw of a stochastic simulation. It is metadata of the
  table: a `.pymxr` file keeps the table without it, and table operations that drop pandas attrs drop it.
- **Model solving** handles backward- and forward-looking models (first-order perturbation, Blanchard-Kahn), Dynare
  `.mod` import and IRIS `.model` import (`&x` and `!dtrends` parameters are held at the values read, not
  estimated). This is the youngest part of the package compared with IRIS or Dynare.
  - *Occasionally binding constraints* (OccBin): simulation, the OccBin filter and its smoother iterated on the
    smoothed path, estimation with it, forecasts and posterior forecasts that respect the constraints, targets held
    by anticipated free shocks, and shock decompositions with the regimes held at the smoothed ones, all with
    correlated shocks and known-date regimes too. Moments still use the linear solution. OccBin forecasts hold
    targets only with anticipated shocks, without draws, and with only regimes known in advance (a surprise regime
    would need the targets solved again when it starts).
  - *Regimes with known dates* (`Regime`) need a model without growth trends. A model estimated with regimes has
    moments and variance decompositions one regime at a time (each held for good at its own steady state); its
    identification check uses the moments of every period of the sample through the regimes. Dynare and IRIS model
    files have no place for regimes (pymetrix's model text has).
  - *Second-order solutions* are estimated with the particle filter (bootstrap or guided; maximum likelihood or
    posterior mode with common random numbers, then particle marginal Metropolis-Hastings). The fit's `smoothed`
    comes from the particle smoother (backward simulation in the shocks, `smooth_paths` paths), whose cost grows with
    the square of the sample's length (about `smooth_paths` x `candidates` x T^2/2 particle steps: seconds for 100
    periods, minutes for several hundred) and whose error, at the default 100 paths and 200 candidates a period, is
    about a tenth of a smoothed standard deviation (more where the smoothed distribution has several modes). Not
    built: a forward-backward marginal smoother, smoothed quantiles weighted beyond the paths drawn, an auxiliary
    particle filter, SMC squared (`fit.smc` refuses a particle-filter likelihood, as does NUTS, which needs exact
    gradients). The standard errors come from a Hessian of a noisy likelihood (approximate); the filter needs a
    measurement error for every observed series, starts from the stationary distribution (no diffuse start, no
    growth trends), and runs in one process; forecasts, moments, decompositions and identification of such a fit use
    its first-order solution. Second-order impulse responses shock one shock at a time, not orthogonalised.
  - *Third-order solutions* (`solve(order=3)`) are solved, simulated (pruned or not), and give responses and moments
    by simulation, but are not estimated: `estimate(order=3)` is refused (the particle filter follows second-order
    rules only). Not built: the closed-form variance and autocovariances of the pruned process (Andreasen et al.;
    Dynare has them), orders above three, skewed shocks (the shocks are taken to be symmetric, so `ghsss` is zero),
    growth models (write them in ratios). The solver keeps full, unfolded tensors in the states, shocks and the
    scale of uncertainty, so time and memory grow with the cube of their number or faster: a fraction of a second
    for the growth models in the tests, about 15 seconds for 22 states and 7 shocks (Dynare's k-order solver, with
    folded tensors, is much faster there), too slow for hundreds of states.
- **The diffuse log likelihood's convention** (`diffuse_terms`): `"exclude"` covers the state-space models, unobserved
  components and the Kalman likelihood of linearised models, not the OccBin filter, known-date regimes or the
  particle filter (refused). With correlated measurement errors the native engine (statsmodels absent) uses the
  approximately diffuse filter, which already leaves out its first periods: there the two conventions coincide;
  likewise with `presample` on the native engine. The SARIMA estimator is statsmodels' SARIMAX and keeps its
  likelihood (approximately diffuse, the first `d` periods left out, `"exclude"`'s convention) with no option.
- **Estimation of linearised models**: maximum likelihood, posterior mode, random-walk Metropolis-Hastings,
  sequential Monte Carlo and the No-U-Turn sampler (`mcmc(sampler="nuts")`, chains in parallel processes with
  `workers`), with Iskrev identification checks.
  - *The gradient* of the Kalman-filter likelihood is analytic (`pymetrix.models.kalman_score`: the steady state, the
    solution and the filter differentiated exactly) for the mode search, the Hessian and NUTS, but not with known-date
    regimes, the OccBin filter, a DSGE-VAR's likelihood, diffuse starts (unit roots), balanced-growth models or
    equations that cannot be compiled: there NUTS and the optimiser use central differences (2k + 1 likelihood
    evaluations a gradient, so NUTS is slow on medium and large models). The analytic gradient costs about two log
    posteriors on a small model (the New Keynesian template, 7 parameters) and three to six on Smets and Wouters
    (2007) (36 parameters). Its filter's loops over periods run in Python unless numba is installed (`pip install
    pymetrix[fast]`, which compiles them and the plain likelihood's filter, about ten seconds once on first use):
    about twice as fast on the template, a fifth faster for the gradient and twice for the log posterior on Smets and
    Wouters, where the batched products over 230 variance states, the solution's derivative and building the model
    remain. Not done: the Chandrasekhar recursions (no gain measured here). Not built: second derivatives (the
    Hessian is a difference of gradients).
  - *NUTS* with the dense mass matrix (the models' default) beats random-walk Metropolis-Hastings per effective draw
    a second on both models measured: on the template 10.5 against 4.1 (smallest effective sample per second, one
    chain; 13.1 against 7.1 with four chains in parallel), on Smets and Wouters 0.023 against 0.005 (one chain of
    1,000 draws after 500 warmup against Metropolis-Hastings for the same time, on a loaded machine; mean tree depth
    5, 33 gradients a draw), and there NUTS's posterior means matched Dynare's long run within 0.14 posterior s.d.
    where Metropolis-Hastings's were off by up to 0.64. Four parallel chains were measured on the template only
    (Smets and Wouters would have held the shared machine's four cores for an hour). Smets and Wouters still mixes
    slowly in a few parameters (smallest effective sample 78 of 1,000 draws), and its dense metric, estimated from the
    200 draws of the last warmup window for 36 parameters, is noisy (a longer warmup should help; not measured).
    `metric="mode"` (the inverse Hessian at the mode kept, no metric adaptation) is fastest on the near-Gaussian
    template (16 a second, 150 warmup draws) but poor on Smets and Wouters (smallest effective sample 5 of 1,000),
    where the curvature at the mode misstates some parameters' spread. No Riemannian (position-dependent) metric, so
    funnels and posteriors whose scale changes across the support still cost divergences or deep trees. On the
    template about one draw in ten ends a trajectory at the edge of determinacy (counted as a divergence; the
    posterior there is zero, and the means agree with Metropolis-Hastings's). Not built: slice sampling.
  - Forecasts from a posterior draw parameters and shocks, without measurement errors.
  - *DSGE-VAR(lambda)* (`dsge_var`): stationary observables only (no unit roots, no cointegration or trends in the
    prior), a constant as the only deterministic term, no exogenous variables, complete data (missing values only at
    the ends of the sample); lambda-hat's marginal data density is the Laplace approximation at each lambda's mode (not
    a Metropolis-Hastings run per lambda, as DSSW 2007 report); the model's rotation needs as many shocks as
    observables; variance decompositions only at the posterior mean given the mode (no bands); no sequential Monte
    Carlo, no conditional forecasts, no historical decompositions for it; not compared with Dynare's `dsge_var` output
    (none is recorded in the tests). Dynare's `dsge_var` option is used by `DynareModel.dsge_var()`, not by
    `estimate()`.
- **Model files and other syntaxes.** Dynare files: no generalised prior bounds, no correlations of measurement
  errors; Dynare's macro processor is read for the directives most files use, not `@#line`, set operations on arrays
  (`|`, `&`) or Dynare 6's type tests. IRIS's correlations of measurement shocks are not read. Writing models in other
  syntaxes (`export_model`): Dynare's estimation blocks and commands, comments and IRIS's loops do not carry across,
  estimated equations must have named coefficients, IRIS models cannot have deterministic exogenous variables, and
  measurement transforms are written only in pymetrix's syntax (reported series in pymetrix's and IRIS's). The model
  editor reads pymetrix's syntax, Dynare's, IRIS's and EViews's; not TROLL's.
- **EViews workfiles (.wf1) and EViews databases are not read**: save the series as Excel or CSV and load those
  (the Coming from EViews help page says so). The `pymetrix-eviews` plugin reads a *bundle* that its export program
  writes from EViews, but that program has never been run in EViews (the project has no licence; every point to
  check is marked `(verify)` in it), so it is experimental and CSV or Excel stays the supported way in. Its reader
  does not read: semi-annual, multi-year, intraday, bimonthly, fortnightly and ten-day pages with a time axis
  (pandas has no such periods: they read as undated, `undated=True`); value maps as value labels (pymetrix has
  none: the map is kept in `Variable.extra["value_map"]` and the codes stay as data); groups whose members are
  expressions, and samples with conditions or several pairs of ends (kept as text in `meta.extra["eviews"]`); and a
  project made with File → Import EViews workfile… reopens only with the plugin installed (Save the data as Parquet frees the
  script, not the project). It refits only least-squares equations without options against EViews's estimates.
  Running EViews through COM is not built yet. Nor are IRIS databanks saved as .mat files; one saved as CSV loads
  once the rows of classes and units under the names are left out (`skiprows=[1, 2]`), which Load data cannot do.
- **EViews programs are translated for data work, estimation, tests, models and control flow (phases 1 to 3)** by
  `pymetrix-eviews` (`translate_program`). Left out, each line with its reason in the report: in control flow,
  subroutine parameters that are objects (`series s`, `group g`: string and scalar ones are translated),
  a `local` subroutine's data and estimation lines, names with `$` in them (`_$_x`, as the FRB/US MCE library uses),
  a group's i-th series in an expression (`g(i)`), names made from an expression in braces (`{!i+1}`), pages and
  models made or changed inside a loop, an `if` block or a subroutine, `for` over several string variables at once,
  `@log`/`@exp` of a value known only as the program runs, `@wlookup` of other object types than series, `@equaloption`
  and the other program functions not listed in the guide; a subroutine's body is translated once, at its first
  `call`, on the page selected then; after a name made as the program runs, which series a page has is no longer
  checked; a number in a made name is written as Python writes it (`2.0` for a division's result, where EViews
  writes `2`); a `while` loop in steps mode stops after `while_cap` passes; dates known only when the program runs
  are passed to `Sample` as written (quarterly and annual EViews dates read, monthly ones do not); objects the
  program uses but does not make (a model an add-in reads, a database's series) are given with `objects=` and the
  bundle (the app's dialog has no field for `objects=`: translate in code for that); the translation panel's Run as
  steps runs every step of the program, from the first, into the open session (no choosing a range of lines); graphs,
  tables, spools,
  display and settings, EViews databases (`fetch`, `store`, `dbopen`), add-ins and user commands, external
  programs. Within data work: `read` and `import` (use `Dataset.read`), `copy` between pages (`to_freq`), page
  operations (`pagestruct`, `pagedelete`...), alpha series, weekly, daily, undated and semi-annual new pages,
  samples of several pairs of dates, conditions with lags or functions other than log, exp, abs and sqrt, groups of
  expressions, `rename` and `delete` with patterns, element assignment in matrices and tables, `label(r)` remarks,
  `@trend`, `@seas` and `@elem` in data work (`eviews_series` makes them only for a model), sample statistics inside
  a larger expression (give a scalar the statistic first), objects' data members inside a series expression
  (`eq.@coefs(2)` is translated as a scalar's value) and string, date and matrix functions (`@str`, `@val`,
  `@word`, `@datestr`, `@rows`...); `@datestr` without a format. Within estimation: `stepls`, `breakls` (`bai_perron`), `threshold`
  (`ThresholdAR`), `switchreg` (`MarkovSwitching`), `glm` and `nls` options; **`testadd`** (the omitted-variables
  test has no pymetrix equivalent); **the extreme-value link** of `binary`, `ordered` and `censored` (`d=x`: no
  complementary log-log family), and `censored`'s logistic errors and limits given as series; `count`'s other
  distributions; seasonal ARMA terms (`sar`, `sma`), ARMA ranges, `pdl` and `@expand`; weights (`w=`) and every
  other option not mapped (the line is left out naming it); VAR lags other than one range from 1, VEC cases e and
  exogenous variables; systems estimated by `ls`; `c(i)` inside a series expression; a VAR's forecast (EViews
  makes a model of it); `hettest` with its own regressors, `chow` with several dates. Within models: `merge` of an
  equation not estimated by least squares, or with ARMA terms; add factors on some equations only (`@all` is
  translated), `addinit` other than `v=n`; `solveopt`'s stochastic, static and fitted solutions (`s=`, `d=`), and
  its other options listed in a note, not mapped (`g=`, `z=`...); the Actuals scenario; `makegroup`. Translated
  with a note, to check against EViews: White is HC1; the HAC bandwidth; the ARMA estimator (exact ML here, by
  version in EViews); GARCH's presample variance; GMM's weighting iterations; quantile regression's standard
  errors; `heckit` by ML; ARDL's lag search; information criteria per observation; the default solver (Newton
  here); add factors used in every scenario; the alias series outside the solved span holding the data. A declared
  scalar starts as NA and `x.label` without an option sets the description; both are marked to verify. Not in the
  app yet: no Translate EViews program dialog or translation panel; translate in code.
- **EViews expression functions pymetrix's expressions lack** (the audit of phase 1): `@date` (each observation's
  date number: use `@year`, `@quarter`, `@month`, `@day`), `@trend`, `@seas` and `@elem` in data work
  (`eviews_series` makes them for a model), `@obssmpl` inside a series' formula (it is translated as a scalar's
  value), `@strdate` (an alpha series), and the by-group statistics grouped by an expression (`@sumsby(x, g(-1))`:
  make the group series first). The translator leaves a line that uses one out, naming it. Translated with a
  difference: `@cumsum` and the by-group statistics are computed over all the page's rows (EViews: the current
  sample; the report says so under a `smpl`); `@obsnum` counts within each cross-section (EViews's counts a panel's
  rows straight through); random draws are pymetrix's own, seeded from `rndseed`, never EViews's numbers, and inside a
  loop every pass draws the same ones. In pymetrix's own formulas, the calendar functions and random draws cannot be
  lagged or differenced (`d(year())`): make them a series first.
- **EViews's syntax** is a subset, confirmed against Brillet's *Structural Econometric Modelling* (EViews) and the
  FRB/US packages, without EViews itself (the book's programs are copyright and not in the repository, so those tests
  run only with `PYMETRIX_BRILLET` set to a folder with one's own copies; CI does not run them): EViews's programming
  language (loops, `if`, `!`/`%` variables, `{%s}` substitution, subroutines), moving variances and standard
  deviations, `and`/`or`, `@elem` at a date without its frequency, `@pchy`-like functions when the frequency is not
  known, `-a^b` and `a^b^c`, estimation methods other than least squares (and ARMA terms in specifications), `@add(v)`
  where the variable is not alone on the left, and add factors or innovations on equations fitted from data are
  refused at their place. Linked equations are read only with their specification in the same text. A state-space
  object maps to `StateSpace` only without regressions in its signals (data with estimated coefficients), covariances
  of errors, or variances other than a coefficient squared or a number; FRB/US's supply-side `estimation_code.prg`
  reads only in part for those reasons. EViews's own estimates and solves are not run or compared against: the checks
  are against pymetrix's own equations and the FRB/US package's Python versions and baseline.
- **The FRB/US worked example** runs the Fed's example1.prg to example4.prg, the federal-purchases ping and
  policy-rule switches. The zero bound and thresholds (example3.prg) are solved under VAR expectations only, as the
  program does; with model-consistent expectations the `max` would be a kink in the stacked system, which is not
  tried. The thresholds' logistic switches are smooth (slope 25), so the switch leaks slightly before a threshold is
  crossed (0.003 the quarter before, in example3.prg). Stochastic simulations (stochsim.prg) and optimal control
  (ocpolicy.prg) run under VAR expectations only: `optimal_control` takes backward-looking models (its derivatives
  come from the period-by-period Jacobians, not the stacked perfect-foresight system), imposes no bounds (the
  program's zero-lower-bound penalty is left out; add a penalty variable to the loss if needed), and has no
  time-consistent (`opttc`) variant; stochastic simulation of a model with leads needs its first-order solution. It
  does not cover the Fed's MCE algorithms themselves (quasi-Newton, the `terminal` option for permanent shocks).
  Model-consistent simulations take their terminal values from LONGBASE (or a terminal rule): permanent shocks need
  terminal values chosen by the user. The FRB/US packages are not in the repository, so those checks run only where
  `PYMETRIX_FRBUS` and `PYMETRIX_FRBUS_DATA` point at them.
- **Simulation plans:** least-norm plans and soft holds are for first-order solutions only; the nonlinear solvers
  hold targets exactly, one free variable per hold.
- **Bayesian single equations** are linear (Gaussian, logit, probit, Poisson), without ARMA or GARCH errors or
  explicit forms; samplers are Gibbs, random-walk Metropolis-Hastings and NUTS (diagonal or dense mass matrix; no
  Riemannian metric for funnels). Monte Carlo standard errors are reported for the posterior means only (not for
  quantiles or HPD bounds), and only for single equations: the DSGE posterior and Bayesian VARs report ESS without
  them.
- **Bayesian VARs** take the Minnesota prior in its conjugate form (`Minnesota`), its hierarchical form (`GLP`:
  Giannone, Lenza and Primiceri's hyperpriors, at the mode or by MCMC over the hyperparameters) or its independent
  normal-Wishart form (`NormalWishart`, by Gibbs, with Chib's marginal likelihood and a search over it). Not yet: GLP's
  hyperprior on the lag decay (it is fixed) or GLP's MCMC for the independent form. The search for the independent
  form is slow (a Gibbs run for each trial, minutes on large VARs) and only as precise as Chib's estimate from its
  shorter runs. GLP's published replication results were not reproduced here (their data and code were not
  available offline); the marginal likelihood is checked against the conjugate one and the MCMC against a grid.
- **The eigenvalue fluctuation test** (`vecm.fluctuation_test`) takes its p-values from a bootstrap, not Hansen and
  Johansen's asymptotic distribution, which could not be made to hold its size here; it costs a few seconds, and its
  power is moderate (54% at 5% for a large fall in adjustment, T = 200).
- **Models solved per entity** (`Model(..., over="country")`) solve each entity on its own, or jointly through links
  (foreign variables, world aggregates) with weights fixed over time; draws of linked entities are for
  backward-looking models, without coefficient draws or GARCH errors; not yet: time-varying weights, and checks,
  steady states and first-order solutions of the linked model as a whole (they are for one time series).
- **Historical decompositions** are for VARs, Bayesian VARs, VECMs, panel VARs, proxy SVARs, BVAR-SVs and MS-VARs (not
  quantile VARs or DSGE-VARs, below and above); bands come from a Bayesian VAR's or BVAR-SV's posterior draws, a
  sign-identified set, a panel VAR's coefficient draws (with the residual covariance held at its estimate, as its
  impulse-response bands do), a proxy SVAR's moving-block bootstrap or an MS-VAR's draws from its estimates'
  asymptotic distribution, not from a frequentist bootstrap of a point-identified VAR or VECM. A panel VAR's are
  entity by entity (the result window shows one entity, chosen in the dialog or by `hd.select(entity)`, not with a
  picker in the window); its fixed effects are the entities' means of the data less the lags' effect, not estimated by
  GMM themselves. A proxy SVAR's other shocks are shown together, not one by one.
- **BVAR-SV, MS-VAR and quantile VAR decompositions** are recursive (Cholesky in the VAR's order) only. A BVAR-SV's
  variance decomposition holds the volatilities at one date's values over the horizon (not their expected future
  path); its point values are posterior means. An MS-VAR's variance decomposition holds the regime fixed (no
  switching over the horizon); its historical decomposition conditions on the most likely regime path (smoothed
  probabilities). An MS-VAR's bands (responses, variance and historical decompositions) come from draws of the
  parameters from the ML estimates' asymptotic normal distribution, not a bootstrap: they rest on the asymptotics
  (responses' 90% bands covered the truth 87 to 88% of the time in a 300-period Monte Carlo), keep the estimates'
  regime labels, hold a transition row with a probability at 0 or 1 fixed, and are refused when the likelihood is
  flat (regimes that barely differ); the historical decomposition's bands leave out the uncertainty about the regime
  path. An MS-VAR's fit is a local maximum still: its starts (regime paths read off the data, then random splits,
  each screened by a short EM run) find clearly separated regimes, but regimes that differ only in their lag
  coefficients, or only a little, can still be missed, and nothing searches for a higher maximum once the best few
  starts have converged; refit with more `starts` or another `seed` and compare log likelihoods. A regime whose own
  covariance rests on a handful of periods can make the likelihood unbounded (as in any switching-variance model);
  such starts are dropped as collapsed (under half a period of probability) but nothing bounds the covariances. A
  quantile VAR's variance decomposition orthogonalises the quantile residuals' covariance by Cholesky (Ando,
  Greenwood-Nimmo and Shin use the generalised decomposition), without bands; a quantile VAR has no historical
  decomposition: its residuals are not centred away from the median (at 0.1, nine in ten are positive), so a
  "baseline" quantile path plus shock contributions mixes location with shocks, and the literature has no
  established version.
- **Zero and sign restrictions** compute ARW's importance weights by numerical differentiation (a few milliseconds a
  draw for a few variables, growing with n^4); zeros are on impact and in the long run, not at other horizons or on
  A0's coefficients; a shock's columns are not searched over (each shock keeps its column, as in ARW). Draws are
  resampled by weight rather than kept with their weights. Not compared with the authors' MATLAB code.
- **Cointegrating regressions** take the rows a span and a row condition (`where=`) keep, and rows with missing values
  dropped, as consecutive periods: leaving rows out of the middle of the sample joins the periods either side.
  Their automatic Bartlett bandwidth is Newey and West's (1994), as arch 9.0 and R's sandwich choose it, and like any
  plug-in rule it is noisy: one more autocovariance in the pilot can move it by several lags, and it can be 0 (no
  serial-correlation correction at all) when the summed series' autocovariances cancel, as for FMOLS and CCR with a
  trend on the 400-row simulated system in the tests, whose errors are AR(1) with 0.5. Give a bandwidth and check that
  the conclusion survives a few. pymetrix 0.1.0a18's rule (`bandwidth_rule="legacy"`, arch's before 9.0) was
  removed after 0.1.0a20: results from 0.1.0a18 and earlier can be repeated only with pymetrix 0.1.0a20 or earlier
  (or arch 7 or 8, with a whole-number bandwidth given, where the two rules use the same lags).
- **Phillips-Ouliaris** is computed by arch, and with the bandwidth left empty arch chooses it: arch 9.0 chooses it
  differently from arch 7 and 8 (Newey and West's rule as above), which can move the statistic and p-value a long way
  (US consumption on income, statsmodels' macrodata: Zt -3.03, p = 0.105 with arch 8; -3.33, p = 0.052 with arch 9).
  pymetrix supports arch 7 and later, so the details give arch's version and the bandwidth used; give a bandwidth to
  repeat a result under any version.
- **Simulated critical values (ADF, Engle-Granger):** the null is independent Gaussian random walks with no drift and
  i.i.d. normal increments; serially correlated or heavy-tailed increments, a drift under the null, or regressors with
  different variances are not simulated (the lag search absorbs some serial correlation, as it does in the data).
  `exog` is for Engle-Granger only (ADF's regression has no fixed regressors), and needs the simulated values. The
  p-value is the share of draws at or below the statistic, so it is zero when no draw is that small; it has a Monte
  Carlo standard error, and more draws narrow it. KPSS, Phillips-Perron, DF-GLS and Phillips-Ouliaris keep their
  tabulated or response-surface values.
- **Seasonal dummies (`seasonal()`, `C(quarter)`, `C(month)`):** quarterly and monthly seasons only (no weekly or
  daily seasons, and `seasonal()` on daily data is refused); fixed 0/1 dummies, not centred or sum-to-zero seasonal
  contrasts; like `C()` they cannot be used in explicit-form equations, models, GMM or forecasts from the equation
  (prediction with `C()` terms is not supported yet).
- **Default largest lag of ADF and Engle-Granger in short samples:** capped so the largest search regression keeps 4
  residual degrees of freedom (counting Engle-Granger's levels regression); below about 26 observations this differs
  from statsmodels' and arch's defaults (arch caps at `(n - 1)//2 - 1 - terms`), so results there match theirs only
  given the same largest lag. Four degrees of freedom is a floor against a degenerate regression, not a guarantee
  that the information criteria choose well: in samples this short they still lean to long lags, and a small fixed
  `max_lags` is usually wiser. Pedroni's rule, `int(12 (T/100)^(1/4))`, is capped the same way; the
  Ng-Perron and ERS tests keep theirs (the modified AIC's penalty grows without bound as the fit becomes exact, so
  it does not pick a degenerate lag).
- **Panel unit roots and cointegration:** Breitung, Pedroni, Kao and Westerlund are checked by hand, against published
  moments and by simulation, not against another package's output; Westerlund's bootstrap for cross-section dependence
  and CIPS with several factors are not built.
- **Seasonal adjustment:** STL, classical, X-13 (needs the X-13 binary) and TRAMO-SEATS (needs Java 21 and JDemetra+'s
  libraries); no JDemetra+ X-13, revision analysis or diagnostics tables beyond what each method records. TRAMO-SEATS
  takes interventions and user regressors (holidays as series of the dataset), but not JDemetra+'s national holiday
  calendars or fixed ARIMA coefficients.
- **Black-Litterman (finance plugin):** views are given as variances (omega) or left to He and Litterman's default;
  no Idzorek percentage confidences, no views on factors or relative views with uncertain weights. The optimal
  weights are the unconstrained mean-variance ones; for constraints give the posterior mean to `optimize_portfolio`.
- **Copulas (finance plugin):** BB1, BB6, BB7, BB8, Tawn's and the rotated copulas are for two series (in a vine, for
  each pair); a vine's pair copulas are the plugin's bivariate families (no nonparametric pair copulas), its standard
  errors are each pair's with its inputs taken as known, and its goodness-of-fit bootstrap refits the parameters but
  does not repeat the selection of structure and families. Time-varying copulas: Patton's recursion for two series,
  the DCC Gaussian and t copula for several (no time-varying vines, no asymmetric DCC copula, no DCC for the
  Archimedean families); Patton's tail recursions are often weakly identified. IFM standard errors ignore the margins'
  estimation (no Godambe sandwich; joint maximum likelihood gives the full ones, without GARCH), and the DCC copula's
  treat Qbar as known; the t copula's correlations come from Kendall's tau and have no standard errors except under
  joint estimation. The goodness-of-fit tests are the Cramer-von Mises one on the empirical copula (parametric or
  multiplier bootstrap; the multiplier not for the t copula, nor with Kendall's tau for more than two series, and
  conservative in small samples: 1.5% rejections at 5% for Gumbel with 200 observations) and, for vines, the
  Rosenblatt-transform ECP2 one (not the Kendall-process, White or Breymann tests); in more than two dimensions the
  Gaussian and t CDFs are by quasi-Monte Carlo (about 1e-5). No skewed-t GARCH errors (skewed-t margins are fitted to
  the residuals of a normal or t GARCH).
- **Simulated critical values.** The Im-Pesaran-Shin, Pedroni and Westerlund moments and the CIPS critical values are
  simulated for each panel rather than read from published tables, so results differ from other
  packages by simulation error (reported with the result; typically under 0.02).
- **Temporal disaggregation and reconciliation** are checked against formulas written in the tests, not yet against
  R's tempdisagg or another package's numbers. Soft constraints take one standard deviation per low-frequency value
  (no correlated errors) in the data's units; for Denton it is a penalty against the Denton criterion, whose scale for
  proportional Denton is that of ratios, so the same `soft` is tighter there than for the regression methods. With
  soft constraints the regression methods' coefficient covariance is the maximum-likelihood one, without the
  degrees-of-freedom factor of the exact fit. Seasonal differences are for quarterly (lag 4) and monthly (lag 12) data
  only, and proportional Denton keeps the ratio's seasonal changes smooth rather than the pattern exactly; without the
  level penalty (`level_penalty=0`) the within-year pattern is the minimum-norm one. No Santos Silva-Cardoso dynamic method, and no multivariate (several low-frequency
  series at once) Chow-Lin. Reconciliation is in first differences or least squares (variances in proportion to the
  values, or one constant variance per component: no variances that differ by period, and no covariances between
  components or periods), with exact constraints only
  (no reliability weights on the total or on the low-frequency values), no ratio (RAS) or non-negativity constraints,
  and dense matrices: a few hundred periods and a dozen components are quick, much larger systems are slow. Denton
  with no low-frequency totals to meet has no unique answer (a constant can move between components), and the one
  returned depends on the linear-algebra library. An
  aggregation matrix entered in code is not part of a saved project unless a step made it.

## Results

- **Provenance** (with a data fingerprint, the rows actually used and why others were not, and the backend that did
  the numbers) is recorded for every result estimated from data, and for tests on data; backtests, combinations,
  rolling and recursive fits and quantile processes gather their fits' (a report's appendix gives a backtest's origins
  and the estimation samples of its first and last re-estimates). A rolling fit's data fingerprint is left blank (no
  one window holds every row); a recursive fit's is its last window's. Results computed
  from a model alone (impulse responses, decompositions, OccBin paths, sweeps, optimal policy) have none: they depend
  on the model, not on data.
- **Results that look wrong but are not.** Two outputs of the modelling demos surprise people:
  - *Data contributions* with no measurement errors: a variable observed as it is (in the New Keynesian template,
    inflation through "infl" and the policy rate through "rate") is known exactly, so its smoothed value comes from
    its own series alone, and output, observed through its growth, leans almost entirely on "growth". The series
    share the work only when the measurement equations have errors (checked against the smoothed-state weights
    calculated directly).
  - *Parameter sweeps* where the lines overlap: in the New Keynesian template (serially uncorrelated shocks, a
    smoothed policy rule, a flat Phillips curve) output's response to a demand shock changes by under 1% between
    phi_pi = 1.2 and 3, while its response to a cost-push shock doubles (the panel the demo now opens on).
    `table()` shows the differences; `sweep(..., response=, shock=)` or `plot(response, shock)` draws another pair.
- **Convergence** is recorded for the iterative estimators (ML, ARMA, GARCH, Markov switching, MIDAS,
  state space, model estimation); quantile regression reports non-convergence only as a warning.

## Workflow

- **Row conditions** (`where=`, a sample's condition) are pandas' `DataFrame.eval` syntax, checked first: pandas'
  `.str` and `.dt` accessors, `@` references to Python variables and functions other than pandas' maths functions
  are refused. A time axis compares with text in order (`time >= '2000Q1'`) but not with `==`, `!=` or `in`, as
  before. The dialogs' check as you type reads a condition with pymetrix's formula parser, so it can flag a valid
  condition (`country in ['DEU']`) that runs.
- **Assignment within a sample and empty datasets** (`with_series(..., within=...)`, `Dataset.empty`). Formulas have
  no missing-value literal (EViews' `na` inside an expression, as in `@recode(x > 0, x, na)`): a series of missing
  values is `with_series(y=None)`, and missing values in some rows `with_series(y=None, within=...)`. A formula that
  uses the series it assigns is computed one period at a time, so on a long daily series with a wide sample it is
  slower than one pass. The active sample (Set sample) still does not limit Generate series: give the sample under
  Within. `Dataset.empty` makes a dated time series only (no empty panel or undated page; `from_frame` builds those
  from their entities), with no dialog field to declare series: add them with Generate series or the spreadsheet.
- **Forecast rounds** carry their own record (data end, judgement, notes, conditions) but not the model's
  provenance: a report built only from a round has no provenance appendix. Add the fitted model
  to the report as well (Lab 3 does).
- **Forecasting** (chapter by chapter in the [forecasting course pack](../examples/forecasting/README.md)) follows
  Elliott and Timmermann's chapters on evaluation, combination, model averaging, asymmetric losses, binary forecasts,
  exponential smoothing and conditional forecasts, with these limits:
  - no backtests of probability forecasts: score Logit and Probit predictions with `binary_scores`;
  - dynamic forecasts from plain regressions with a lagged target have no standard errors yet, so no density scores
    (equations with ARMA or GARCH errors have them, except as noted under Methods);
  - Bayesian model averaging takes up to 15 candidates, with the g-prior only;
  - conditional forecasts with ranges ("between 3 and 4%") draw from the truncated distribution by Gibbs sampling
    with a fixed number of sweeps per chain (100), with no convergence diagnostics; very tight, strongly correlated
    ranges over many periods mix slowly. Ranges are hard bounds: no density conditions (a condition given as a whole
    distribution, as in Andersson, Palmqvist and Waggoner 2010). VARs, Bayesian VARs and first-order model plans only
    (model plans keep variables within ranges deterministically, with the smallest shocks, not by drawing);
  - structural scenarios (`shocks=`) from a Bayesian VAR identified by sign restrictions are not available, and a
    classical VAR's use the median-target impact matrix only (no draws across the identified set); when the shocks
    named cannot meet the conditions the forecast is refused rather than met in a least-squares sense;
  - `update_posterior=True` (Waggoner and Zha's Gibbs sampler, the posterior updated by the conditions) keeps the
    prior's hyperparameters fixed at the fitted values, and needs every shock free to move;
  - threshold and smooth-transition autoregressions have two regimes, split by a lag of the series itself; no neural
    networks;
  - shrinkage and machine learning need scikit-learn (`[ml]`); learners give no standard errors beyond one step
    ahead for dynamic forecasts that feed on themselves, no classification learners, and no hyperparameter search
    beyond the shrinkage penalty;
  - realized-volatility (HAR) and multivariate GARCH models (chapter 13) are in the finance plugin
    (`pymetrix-finance`), not the core package.
- **Data connectors** cache searches and metadata but not downloaded data (each `Dataset.read` asks the provider
  again), and have no FRED API-key route for per-observation vintages: ALFRED vintages are read by date, but there is
  no browser of a series' vintages or release dates. File → Browse series… searches FRED only with an API key (reading
  by ID needs none). World Bank search filters a source's indicator list locally (the World Development Indicators by
  default; "all" downloads every indicator once); SDMX search matches dataflow names and descriptions, not series
  titles, and the browser does not check that a key has data before Load; SDMX providers' structure queries follow
  SDMX 2.1 REST and were tested only against hand-written structure messages (the providers could not be reached from
  the build machine), and Eurostat's full dataflow list is large on first use. Searches are not recorded as steps.
  How a dimension's codes are found (a code list of another agency, a concept's representation, a code list asked for
  on its own, content constraints) follows the SDMX 2.1 standard and is tested on a hand-written World Economic
  Outlook structure, not yet on the IMF's own answer. The dataflow folders are the provider's own categories (SDMX 2.1
  category schemes and categorisations), else name families, ID prefixes or a flat list; the category trees are tested
  on hand-written ECB-, Eurostat- and IMF-like answers only, not yet on any provider's live answer, and the request
  used (`categoryscheme/{agency}/all/latest?references=parentsandsiblings`) has not been checked against each
  provider's documentation. A category tree is read whole, once a week; Eurostat's may be large on first use.
- **Reports** write Word, Excel, HTML and Markdown, but not PowerPoint or PDF. Charts are
  images. There are five built-in themes and three templates; custom templates are a list of view
  titles written in code, and there is no template editor or institutional branding (logos, fonts
  shipped with the report).
- **Generate series for each** (Data → Generate series… with For each; `Dataset.with_each`) matches names with
  `*` and `?` only (no `[...]` sets or regular expressions) and does not take a named selection or metadata in the
  dialog: type the names, pick them in the series list, or pass `data.series(units=...)` to `with_each` in code. The
  preview checks names, parsing and that the series a formula uses exist; a formula that fails only when computed (a
  function given the wrong arguments) fails at OK, with the step not recorded.
- **Saving a single result** (`save_result`, File → Save result…) keeps its views, summary and provenance, not the
  live object: a saved result cannot be forecast from or estimated again (its project can). Its call is kept only for a
  result a session step made (a dialog, or typed code recorded as calls): a result made in a plain script, or by
  typed code kept as code, has none unless one is given (`save_result(..., call=...)`). The call names the datasets
  it read; it does not carry them, so making the result again needs the same data.
- **Programs that read and write projects** use a public interface that is provisional before 1.0 (`Call`,
  `make_call`, the codec's `encode` and `decode`, `Session.attachments`): no stability policy has been decided, so
  its names and details may change before 1.0 (saved projects keep opening, through their migrations).
- **Deferred steps** (`Session.add(cmd, deferred=True)`) keep the objects they read, as at their place, until
  computed, so a replaced dataset or model stays in memory while a step that uses it is pending. Computing one puts
  its result in the workspace under its name even when a later step has made that name since. A project saved by
  0.1.0a16 or earlier whose deferred step uses a step marked dead runs that step again on opening: the result's name
  is looked for as a word in the deferred step's saved fields (its code, for typed code), so a dead step whose name
  merely appears there is run again too.
- **The commit in provenance** (`provenance.build`) is read from git when pymetrix runs from a checkout of its source;
  an installed package records its version only (wheels do not carry their commit), and plugins record their
  versions only. `dirty` covers tracked files under the package folder, not files git does not track.
- **Trust:** a project's typed code runs only once you trust it. Trust can be withdrawn in code
  (`pymetrix.trust.forget`), but the app has no list of trusted projects to review, and projects are not signed.

## Desktop app

- Tested automatically without a screen on Windows, macOS and Linux, but with little hands-on use
  beyond the developer's own, mostly on Windows.
- **On a Mac:** dialogs' fields have no Alt (Option) access keys, as in every Mac app (Qt turns them off there by
  design): Tab moves between the fields, and Space opens or folds the Advanced settings header once Tab reaches it.
  With macOS's Keyboard navigation off (System Settings → Keyboard), Tab stops only at text boxes and lists, not at
  pop-up menus, check boxes or that header; turn it on to reach every control from the keyboard. Message boxes have no
  title on a Mac, so where the title says what failed it is the first line of the text instead. The help topics,
  written for every platform, say Ctrl for keys that are ⌘ on a Mac (the app's tooltips and Help → Keyboard shortcuts
  give the Mac's keys).
- **BLAS threads on a Mac.** NumPy and SciPy use Apple's Accelerate there, which takes no thread count: pymetrix's
  limit (`set_blas_threads`) makes it single-threaded for a limit of one, and leaves it to choose for more. That
  switch arrived in macOS 15; before it, the limit does nothing on Accelerate.
- Dialogs on the main demo path (load data, estimate, test, model, forecast, report, project)
  are the most used. Others work but have had less polish.
- Pivot tables are commands, not an interactive drag-and-drop grid.
- **Exercise packs** (provisional): a check compares one value (a number within a tolerance, a string, true or
  false); there are no checks of a whole table, a range, a sign alone, or "any of", and none that depend on another
  check. The Exercise panel does not tick steps as the history changes (press Check), remembers no progress
  between sessions, and has no Run button for a step's code (Copy it into the Immediate window). The app runs nothing
  in a pack, so `solution.py` is only exercised by `pymetrix-exercises check`. A plugin's pack must be a folder on
  disk (not inside a zipped install). A pack's CSV files need a `[data.<name>]` table naming their time or entity
  column. A step's code that reads a pack's file (`PACK_DIR`) runs as written only from the CLI: the Immediate window
  has no `PACK_DIR`. Models are imported by the user (Import…), never on start; EViews programs and FRB/US XML are
  not among a pack's model files. Packs are not translated, and the exercise folder's number orders exercises only within a pack.
- **How do I… task pages:** the fifteen planned are written (`docs/how-do-i/README.md` lists them). Results windows'
  Next ▸ menus and the dialogs' help pages do not link to them yet. The connector page's code is tested against
  recorded responses only, not the providers themselves; the EViews translation page cannot show its bundle route
  running (the export program is untested in EViews), and its code runs only where `pymetrix-eviews` is installed.
- The front window's accent title bar is checked automatically with the Fusion and Windows styles only; on the
  windowsvista and windows11 styles (Windows' own) it is painted the same way but has been looked at by hand less.
  Status colours and muted text are picked when a widget is made, so switching the system between light and dark
  while the app runs leaves them as they were until the window is reopened (the title bars follow at once).
- Charts under a dark palette turn neutral dark greys light, including a dark grey a user chose for a line or note;
  the saved file keeps the colour chosen. A chart already open keeps its colours when the system switches between
  light and dark until it is drawn again.
