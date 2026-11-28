# Changelog

## Unreleased

- **Heterogeneous-agent models: a new plugin, `pymetrix-ha`** (`packages/pymetrix-ha`, 0.1.0a1; `pymetrix[ha]`, not in
  `[all]`: it brings numba). It wraps sequence-jacobian 1.0 (Auclert, Bardóczy, Rognlie and Straub 2021), phase 1 of
  docs/design/heterogeneous-agents.md, in code only:
  - `HAModel` over sequence-jacobian's blocks, with the calibration, steady-state and equilibrium problems and AR(1)
    shocks; the blocks are checked when the model is made.
  - Its results are pymetrix objects. The steady state (`HASteadyState`) is a table with its checks; block and
    general-equilibrium Jacobians are arrays by output and input. Impulse responses are an `IRF`; a nonlinear
    perfect-foresight transition is a `Solution`, with the linear one beside it.
  - Every tolerance is passed explicitly (`HAOptions`, the library's own defaults), and the library's printing is
    kept in provenance. Provenance also records the grids and each block's source fingerprint.
  - Blocks run by `exec`, which the library cannot read, work through `blocks_from_source`.
  - The Krusell-Smith example reproduces the numbers the package's notebook prints, and equals the library called
    directly. A help topic, *Heterogeneous-agent models*.
- **Extension API 1.11** (additive): **`pymetrix.translation`**, the translation framework, for a plugin that
  translates programs written for other software (docs/design/r-translator.md, phase 0). The EViews plugin's
  translator machinery that is not about EViews moved into core: the intermediate representation a statement
  becomes, the report with a row per line (`ReportRow.source`; its column named by the language), `Translation`
  (subclassed per language: its steps' source and project key), the readable script, steps mode, the outcome and its
  statuses, the name allocator, the translation panel (`PanelSpec`, `translation_panel`) and checks for a front
  end's tests (`pymetrix.translation.testing`). `Def` gains `result`, the one value a function returns.
  `pymetrix.translation` is a stable module (`pymetrix.extension.STABLE`); `Translation`, `TranslationReport`,
  `TranslationOutcome`, `TranslationPanelSpec` and `translation_panel` are also in `pymetrix.extension`. The help
  topic "Writing a translator for another language" and `examples/translation_toy.py`, a toy language of two
  statements, show a front end end to end. `pymetrix-eviews` now builds on it, with no change in what it does (its
  modules keep their names; `ReportRow.eviews` stays). `plugins.API` is now `"1.11"`.
- **HC0, HC2 and HC3 standard errors, and a time axis from a first period** (R's `vcovHC` and `ts`). `OLS(cov="hc3")`
  (and `"hc0"`, `"hc2"`; Standard errors in Estimate → Equation…) gives R's `sandwich::vcovHC` default; `"white"` is
  still HC1, and no default changed. HC2 and HC3 are for linear OLS, weighted or restricted; HC0 also for IV.
  `Dataset.from_frame(df, start="1990Q1", freq="Q")` dates a frame without dates one period per row, as
  `ts(x, start = c(1990, 1), frequency = 4)`; `Dataset.read` takes the same `start=` for CSV and Excel files, as
  does File → Load data… (Dates start at). A statsmodels result imported with HC0, HC2 or HC3 now says so in its
  summary, not "White (HC1)".
- **Prior stability mapping** (Model → Analysis → Stability mapping…, `stability_mapping`): global sensitivity
  analysis of a linearised model, as Dynare's `dynare_sensitivity` (Ratto 2008). Parameters drawn from their priors
  are classified as determinate, indeterminate, explosive, without a steady state or without a solution, and each
  parameter's Smirnov test (SciPy's `ks_2samp`) compares its determinate draws with the others; the results show the
  shares, the tests and each parameter's cumulative distributions. A steady state whose residuals are not numbers
  (the log of a negative value) is now refused, where it was accepted before.
- **Dynare files: correlated measurement errors, generalised priors, Dynare 6's macro operations.** `corr y, c = r`
  (or a covariance) on two observed variables in a shocks block is the correlation of their measurement errors, used
  by the Kalman filter, estimation and its analytic gradient; `corr y, c, ...` in `estimated_params` estimates it
  (`corr_y_c`; for any model, `MeasurementMap(error_corr=...)`, estimated when `corr_<series>_<series>` is named).
  Priors with a third and fourth parameter are Dynare's generalised distributions (`GeneralisedBeta`,
  `ShiftedGamma`, `ShiftedInvGamma1`, `ShiftedInvGamma2`; a normal's are bounds), and `prior_trunc` bounds the
  parameters by their priors' quantiles. The macro processor reads set operations on arrays (`|`, `&`, `-`, `*`,
  `^`), type tests, casts, `sum`, `gamma`, `erf`, `normcdf`/`normpdf` and `@#line`. Two changes to what macros write:
  `@{...}` writes numbers to 15 significant digits and strings inside arrays without quotes, and `==` compares type
  and value (`1 == true` is false), as Dynare 6.0's preprocessor does. Checked against Dynare 6.0's likelihood,
  prior densities, prior bounds and preprocessor.
- **Third-order perturbation.** `solve(order=3)` on a template, a Dynare or an IRIS model (or `third_order(model,
  shocks, shock_sd)`) gives Dynare's `stoch_simul(order=3)` decision rules: the cubic terms and the corrections for
  risk to the responses to states and shocks (ghxxx, ghxxu, ghxuu, ghuuu, ghxss, ghuss). Simulations are pruned
  (Andreasen, Fernández-Villaverde and Rubio-Ramírez 2018) unless `pruning=False`; responses start from the
  stochastic steady state or a given state, and are generalised with `replications`; `moments()` simulates. Every
  term, and simulated paths, agree with Dynare 6.0's to about 1e-13. In the app: Model → Solution → Second-order
  solution…, Order 3. Not estimated at third order.
- **Reading part of a large Parquet file.** `Dataset.read("big.parquet", columns=[...], where="...", span=(...))`
  reads only the columns asked for and, through `pyarrow.dataset`, skips the partition folders (a folder partitioned
  as `year=2010/country=DEU/` reads too) and row groups whose statistics rule the condition out; the condition is
  pymetrix's row-condition syntax, pushed down where it translates (comparisons, `in [...]`, and/or) and applied
  exactly afterwards, so the rows are those reading everything and selecting gives. The load's provenance records the
  filter pushed down, what was applied after reading, and the rows and row groups read against those in the file.
  `pymetrix.io.scan(path)` gives the rows, row groups and each column's type, min, max and missing values from the
  metadata only. File → Load data… on a Parquet file over 100 MB shows Part of a large file (columns, Rows where,
  From, To). Projects save a selection as read; untick Keep a copy in the project (`embed=False`) to keep only the
  file's path and fingerprint, and opening refuses a changed file (`load_project(path, changed_data="read")` reads
  it as it is).
- **LightGBM and XGBoost learners, SHAP values and conformal intervals.** `LightGBM()` and `XGBoost()` (extras
  `[lightgbm]`, `[xgboost]`) fit equations as the scikit-learn learners do, on one thread in the libraries'
  reproducible modes, and are double ML's learners `"lightgbm"` and `"xgboost"`. `shap_values(fit)` or `fit.shap()`
  (extra `[shap]`) gives each regressor's contribution to each prediction: TreeSHAP for trees, exact or seeded
  permutation explainers otherwise. `conformal(fit, data, span)` gives split-conformal intervals for any estimated
  equation, conformalised quantile regression for boosted trees, and adaptive conformal inference for series that
  are not exchangeable. Estimate → Machine learning → SHAP values… and Conformal intervals…; a learner's Next ▸
  menu offers them, with Forecast… and Backtest….
- **Nested and mixed logits, Kaplan-Meier and Cox models.** `nested_logit(data, choice, regressors, nests=...)` fits a
  nested logit by full-information maximum likelihood in the form consistent with utility maximisation (RU2), with a
  lambda per nest or one shared, and the likelihood-ratio test of IIA (every lambda 1 is the conditional logit,
  exactly). `mixed_logit(..., random={"price": "lognormal", "time": "normal"}, panel="person", correlated=True)` fits
  a random-parameters logit by simulated maximum likelihood with Halton or seeded pseudo-random draws (the number of
  draws in `provenance.resolved`), with the coefficients' moments and each decision maker's conditional means. Both
  take the conditional logit's long data and give its marginal effects and elasticities. `kaplan_meier` (curves by
  group, intervals, medians, log-rank and weighted tests) and `cox_ph` (Breslow or Efron ties, strata, left
  truncation, robust and clustered errors, baseline survival, Grambsch-Therneau tests) wrap `statsmodels.duration`.
  In the app: **Estimate → Microeconometrics → Nested logit…**, **Mixed logit…**, **Kaplan-Meier survival…** and
  **Cox proportional hazards…**. Checked against Biogeme's and xlogit's recorded estimates, lifelines', statsmodels and
  the published Freireich leukaemia results.
- **The task pages from a result and from a dialog's help.** A result's Next ▸ menu lists, after its next dialogs,
  the How do I… pages that carry on from that kind of result (an equation: estimate and test it, check its forecast
  record, make a report; a VAR: estimate a VAR, run a forecast round, check its record...), each opening in the
  help. A dialog's help lists the pages whose steps open it, found from the menu paths the pages write.
- **`pymetrix[eviews]`** installs the EViews plugin with pymetrix, as `pymetrix[finance]` and the others do.
- **CI on Node 24.** The workflows use `actions/checkout@v5`, `setup-python@v6`, `setup-java@v5`, `cache@v5`,
  `upload-artifact@v6` and `download-artifact@v7`, the first versions that run on Node 24, before GitHub stops
  running Node 20.
- **The FRB/US tests no longer fix numbers of one release of LONGBASE** (they run only with `PYMETRIX_FRBUS` set):
  the optimal-control loss, the direction of the optimal funds rate in its first year and the size of the
  threshold rule's leak at liftoff changed with the Fed's current data; the first-order conditions, the threshold
  logic and the equations along each path are still checked.

## 0.1.0a21: the EViews plugin, more expression functions and task pages (27 November 2026)

- **Projects keep data from a data service as it was read; File → Refresh data… reads it again.** A dataset from
  File → Load from connector… (or Browse series…) was downloaded again whenever its project opened, so its numbers
  could change with the provider's revisions. It is now saved in the project, as a loaded file's data is, and opening
  uses that copy. File → Refresh data… (`Session.refresh`) reads chosen datasets again from their files or services,
  runs again the later steps that changed them in place (series generated or added), and recomputes what was computed
  from them (Automatic mode) or marks it out of date (Manual); it refuses a dataset that typed code changed, naming the
  step. Projects saved before keep downloading on opening until they are saved again.
- **Rounding, missing values, the calendar and random draws in formulas.** Expressions gain `round(x, n)` (halves
  away from zero, on the number as written: `round(2.5)` is 3, `round(1.005, 2)` is 1.01), `floor`, `ceil` and
  `log10`; the missing value `na`, `isna(x)` and `fillna(x, v)`; the conditional `iif(c, a, b)`, missing where c is;
  and, on data only, `year()`, `quarter()`, `month()`, `day()` (of each period's first day), `obsnum()` (within
  each entity) and the seeded draws `normrnd(mu, sigma, seed=s)` and `unifrnd(a, b, seed=s)`, from the generator
  `set_random_generator` chose. They work in Generate series, derived series (draws excepted: they are values, not a
  recipe), `data.eval` and regressors written in an equation; rounding, `iif` and the missing-value functions also
  in models, compiled and differentiated (rounding is flat, as `step` is); models refuse the calendar and draws.
  Dynare and IRIS exports write `log10` and refuse the rest. Other packages' names (`ceiling`, `ifelse`, `isnan`,
  `rnorm` ...) give a "did you mean" hint. The calendar and draws cannot be lagged or differenced in a formula
  (`d(year())`): make them a series first.
- **EViews: the new expression functions, random numbers and dates** (`pymetrix-eviews`, the core EViews reader).
  `@log10`, `@floor`, `@ceiling`, `@round`, `@isna`, `@nan`, `@iif` and `na` translate in series formulas and model
  text, and `@recode(c, a, b)` other than max/min is now `iif`. In data work only (models refuse them, saying why):
  `@year`, `@quarter`, `@month`, `@day`, `@obsnum`/`@obsid`, `@cumsum`, `@sumsby`/`@meansby`/`@minsby`/`@maxsby`,
  and `nrnd`, `rnd`, `@rnorm`, `@runif` as seeded `normrnd`/`unifrnd` (`rndseed n` gives n·1000 + 1, + 2 ...;
  translated with a note "to check"). EViews's date functions (`@dateval`, `@datestr`, `@dateadd`, `@datediff`,
  `@dtoo`, `@otod`) are worked out as the program's values by `pymetrix_eviews.dates`, so FRB/US's `dateshift`
  translates as written. `convert_expression` takes `data=` and `seed=`, and `render_eviews_expression` takes
  `data=` and `seeds=`; the EViews writer writes the new functions back.
- **How do I… task pages: the rest of the plan, and one for EViews programs.** Eleven more pages, each one task from
  start to finish in the app and in code, with the numbers to expect and the pitfalls (`src/pymetrix/howto/`): load
  data with Browse series; estimate a VAR, then impulse responses and a forecast; run a forecast round (baseline,
  judgement, a scenario, the store, the report); save, reopen and share a project and replay its script; seasonally
  adjust a series; bring an EViews model across, then check and solve it; run a script in the app; make a report;
  check a forecast's record (backtest); reconcile national accounts components to a total; and translate an EViews
  program, a page in the core help for a task of the `pymetrix-eviews` plugin, which it says how to install (its
  front matter names the plugin, and the index says the task needs it). Solving large backward-looking models moves
  from `docs/solving-large-models.md` into the same template as the task page "Solve a large backward-looking model";
  its separate summary topic (`large-model-solution`) is gone, as the help shows the page itself. The pages are
  numbered in the plan's order. A test runs every page's code (the connector page's against recorded responses, never
  the network; the EViews page's where the plugin is installed), runs its steps in the app as the dialogs' commands
  with the fields as the page names them, and checks the numbers it quotes against an independent calculation.
  Help text that named a dialog's fields by old labels now uses the dialogs' own: Hold only where given (was
  Partial), Error draws: The data's residuals (was Errors: sample) and Parallel processes (was Workers), and Model >
  Solution > Add factors; STL's trend smoother length is labelled Trend length (it read Deterministic terms).
- **Issue templates and a defaults page.** `.github/ISSUE_TEMPLATE` has forms for a bug report, a numerical
  discrepancy (the pymetrix version, the other tool and its version, the inputs, both outputs and the tolerance
  expected), a feature request and a validation case, beside the review-feedback form; `config.yml` sends security
  reports to the Security tab, as SECURITY.md asks. `docs/defaults.md` is the in-app help page on defaults that differ
  from EViews, Stata, statsmodels and linearmodels, written from it by `scripts/make_defaults_page.py`, and a test
  keeps the two the same; the README and the guide link it.
- **CI now that the repository is public.** Every push to `main` and every pull request run the tests; macOS (Python
  3.13) is back in the everyday set, and the `release` scope adds the installed wheel on macOS
  (`.github/workflows/tests.yml`, `docs/releasing.md`).
- **The first-party plugins go to PyPI at 0.1.0a1:** pymetrix-assistant, pymetrix-finance, pymetrix-guide,
  pymetrix-labbook and pymetrix-seasonal (`pip install --pre pymetrix-seasonal`, say). The finance plugin's README
  links to the repository with full addresses, so they work on its PyPI page.

- **A new first-party plugin, `pymetrix-eviews`: EViews workfiles through a bundle** (`packages/pymetrix-eviews`, data
  route 1 of `docs/design/eviews-plugin.md`). An export program, `pymetrix_export.prg`, run once in the user's own
  EViews, writes the open workfile as a bundle: each page's series as CSV and a JSON manifest (format version 1, with
  a JSON Schema) with the labels, units and sources, groups, samples, scalars, strings, matrices, value maps,
  equations with EViews's estimates, and models. `Dataset.read("macro.evbundle", page="quarterly")` reads a page,
  recorded and replayed like any read: dated (annual, quarterly, monthly, weekly, five- and seven-day), panel
  (balanced or not) and undated pages; NA and empty fields as missing; alpha series as text; value maps in
  `Variable.extra`; the rest in `meta.extra["eviews"]`. `eviews_objects(data)` gives the groups as selections, the
  samples, scalars, strings, matrices and models (read by `read_eviews`, their equations fitted to the page), and
  `compare_estimates(data)` refits each least-squares equation and sets pymetrix's estimates beside EViews's. What
  does not read is refused at its place: a folder without its manifest, a newer format version, a missing column,
  semi-annual and other frequencies pandas has no period for (`undated=True` reads those pages undated). **File →
  Import EViews workfile…** reads a bundle's pages (one `Dataset.read` line each; *Save the data as Parquet* reads them from
  Parquet copies, so the script replays without the plugin) or saves a copy of the export program. **The export
  program is experimental: it has never been run in EViews** (the project has no licence), and every point to check
  is marked `(verify)` in it; saving the workfile from EViews as CSV or Excel stays the supported way in. Running
  EViews through COM is later work.
- **EViews programs translated, phase 1: data work** (`pymetrix-eviews`, section 2 of `docs/design/eviews-plugin.md`).
  `translate_program("macro.prg", bundle="macro.evbundle")` reads a `.prg` and returns a `ProgramTranslation`: `.script`,
  readable Python using pymetrix's public API with each EViews line above what it became; `.report`, a row per line
  with text (translated, translated with a note, left out, error; the EViews text, the Python and the reason, with
  pymetrix's equivalent), `report.summary()` a `Table`; and `.run(session)`, the same work as recorded steps
  (`make_call`), with the script and the report kept in the project (`session.attachments["pymetrix-eviews"]`). Both
  come from one intermediate representation, and the tests require the script and the steps to give equal objects.
  Translated: `wfopen` (of the bundle, or of a CSV or Excel file saved from EViews, with `read_options`), `wfcreate`
  and `pagecreate` (`Dataset.empty`), `pageselect`; `smpl` (dates, `@first`/`@last`, offsets, `if` conditions, named
  samples) and `sample` (`Sample`, `with_sample`); `series`, `genr` and plain assignments through the core expression
  converter, within the current sample (`with_series(within=)`), a formula that uses its own series worked out period
  by period as EViews does; `series y` and `y = na`; `frml` (`derive`); `group` and `g.add`/`g.drop` (`data.series`);
  `x.label`, `x.displayname`, `x.setattr` (`with_info`); `rename`, `delete`, `x.fill(o=...)` (`edit`); scalars,
  `!` numbers and `%` strings as Python variables, worked out where they are known before the program runs, so
  `{%s}` and `{!i}` in names are put in; `@mean`, `@sum`, `@min`, `@max`, `@stdev`, `@median` and `@obs` as a scalar's
  value over the sample; `vector`, `matrix`, `sym` and `fill` (`Matrix`). Names follow the bundle reader's rules, and
  a Python word or pymetrix name gets `_v`. Estimation, tests and models (phase 2) and loops, `if` blocks,
  subroutines and `include` (phase 3) are left out with that reason, and so is what has no equivalent (graphs,
  tables, spools, display, databases, add-ins), each with the pymetrix way where there is one: nothing is dropped
  silently. The expression functions pymetrix lacks are listed in known gaps (`@year`, `@quarter`, `@month`,
  `@date`, `@obsnum`, `@round`, `@floor`, `@ceiling`, `@nan`, `@isna`, `@iif`, `nrnd`, `rnd`, ...); none was added
  to core. Checked on hand-written programs with every value worked out by hand, and on the Brillet and FRB/US
  programs with the count of each status fixed. Not in the app yet: translate in code.
- **EViews programs translated, phase 2: estimation, tests and models** (`pymetrix-eviews`, the phase 2 table of
  `docs/design/eviews-plugin.md`). `translate_program` now turns `ls`, `tsls`, `gmm`, `liml`, `arch`, `binary`,
  `ordered`, `censored`, `count`, `qreg`, `heckit` and `ardl` (as `equation eq.ls ...`, `eq.ls ...` or bare) into a
  fitted `Equation` (or `ARDL`) with the method the options map to: `cov=white` (HC1) and `cov=hac`, the older `h`
  and `n`, `ar(k)`/`ma(k)` terms (`errors=ARMA(...)`, `method=ML()`), instruments after `@` (`IV`, `GMM`, `LIML`),
  `arch(p,q)` with `tdist`, `ged`, `egarch` and `thrsh=` (`GARCH`, `EGARCH`), `d=l`/`d=n` (`Logit`, `Probit`,
  `OrderedLogit`, `OrderedProbit`), `censored(l=, r=)` (`Tobit`), `count(d=p|n)` (`Poisson`, `NegativeBinomial`),
  `qreg(quant=)` (`Quantile`), `heckit ... @ selection` (`Heckman`), `ardl(deplags=, reglags=, fixed, trend=, ic=)`.
  A list specification `y c x y(-1)` becomes `y ~ x + y[-1]` (`- 1` without `c`), an expression specification
  `y = c(1) + c(2)*x` an explicit form with parameters `c_1`, `c_2`; each estimate keeps its coefficient map
  (EViews's `c(i)` against pymetrix's coefficient), which later lines use: `eq.@coefs(i)`, `@stderrs`, `@tstats`,
  `@r2`, `@rbar2`, `@se`, `@ssr`, `@dw`, `@regobs`, `@logl`, `@ncoef`, `@aic`/`@schwarz`/`@hq` (per observation, as
  EViews), and `c(i)` after an estimation (the last one that fills `c`), as scalars' values. `var v.ls 1 4 y1 y2 @ c x`
  (`VAR`), `v.ec(case, rank)` (`VECM`); `system` with `append` and `@inst`, then `sur`, `3sls`, `fiml`, `tsls`
  (`System` with `SUR`, `ThreeSLS`, `FIML`, `TwoSLS`). Tests: `wald` (through the coefficient map, `eq.test("wald",
  ...)`), `testdrop`, `auto`, `hettest`, `archtest`, `reset`, `chow`, `hist`, `rls`; `x.uroot(adf|pp|kpss|dfgls|ers|np,
  ...)` (`data.test`) and `g.coint(a|c, lags)` (`data.johansen`). `eq.forecast`, `eq.fit` and `eq.makeresids` add
  the forecast, its standard error and the residuals to the page (Add to dataset). Models: `model`, `append`, `merge`
  of a least-squares equation, `@identity`, `@add` and `@innov` are collected into one text, read by
  `read_eviews(text, data=..., params=...)` before the model is first used (or after its last line), the core
  reader's refusals passed into the report at their lines; `solve` and `solveopt` (`o=g|n|b`, `m=`, `c=`),
  `addassign @all` and `addinit(v=n) @all` (`Model.add_factors`, then `Scenario(adjust=...)`), `scenario(n, a=_1)`,
  `override` (the alias series) and `exclude` (`Scenario(fix=...)`), the solution's endogenous series added to the
  page under the scenario's alias (`y_0`, `y_1`). Known default differences (the HAC bandwidth, the ARMA and GARCH
  estimators, information criteria per observation, the default solver) and every *(verify)* item of the design are
  translated with a note saying what to check; an option that is not mapped leaves the line out, naming it.
  `stepls`, `breakls`, `threshold`, `switchreg`, `glm` and `nls` are left out naming pymetrix's equivalents;
  `testadd` and the extreme-value link (`d=x`) are gaps. Checked against NumPy least squares, two-stage least
  squares, a Newton logit, the Wald and White formulas, an ADF regression, a model's response worked out by hand
  and the bundle fixture's EViews estimates; the script and the steps give equal objects.
- **EViews programs translated, phase 3: control flow** (`pymetrix-eviews`, the phase 3 table of
  `docs/design/eviews-plugin.md`). Loops stay loops in the script: `for !i = 1 to n step k` (`for i in range(1, n +
  1, k):`, a negative step too), `for %s a b {%names}` (`for s in [...]`), `while ... wend`; `if ... else ... endif`;
  `exitloop` (`break`), `return`, `stop` (`raise SystemExit`). Subroutines are functions defined first, with
  `string` and `scalar` parameters; EViews passes arguments by reference, so a parameter the subroutine assigns is
  returned to the caller's variable (`simend = dateshift(simstart, simend, 23)`); one that is not `local` shares the
  program's names (`global`), a `local` one shares none (its data and estimation lines are left out: EViews deletes
  what it makes). `include` translates the included program in line, its rows in the report under its name
  (`report.line(3, file="lib.prg")`, `report.files()`); a subroutine's body is translated at its first `call`. Names
  made from the program's variables (`series y_{%s} = log(x_{%s})`, `equation eq_{%c}.ls ...`) are f-strings in the
  script (`**{f"y_{s}": f"log(x_{s})"}`, `globals()[f"eq_{c}"]`) and plain names in the steps. The program's
  values: conditions and bounds with `and`, `or`, `not`, string comparison and `+`, `@word`, `@wcount`, `@wfind`,
  `@wlookup` (the page's series), `@isobject`, `@lower`, `@upper`, `@replace`, `@trim`, `@left`, `@right`, `@mid`,
  `@instr`, `@len`, `@str`, `@val`, the sample's statistics and `@obssmpl` (worked out from the data when the line
  runs), a group's `@count`, `@seriesname(i)` and `@members`, an equation's `@r2` and the other phase 2 members, so
  `if eq.@r2 > 0.5` is decided by the equation just estimated. `makegroup` of a model's `@endog`/`@exog`. The
  program's arguments (`%0`, `%1`) are the script's parameters (`arg0 = "us"` at its top; `translate_program(...,
  args="us 2020q1")`, `run(args=...)`). `objects=` gives what the program uses but does not make (a model an add-in
  reads, a page a database fills). Steps mode runs the control flow in Python and records only the steps, flat, each
  undoable; a `while` loop stops after `while_cap` passes (10,000; `run(while_cap=...)`); `run(run=...)` runs each
  step through another runner (the app's). Inside a block the sample is the one the page has when the line runs
  (`within=data.sample`). Left out with the reason: EViews's date numbers (`@dateval`, `@dateadd`, `@datestr`),
  object parameters of subroutines (`series s`), names with `$`, a group's i-th series `g(i)`, `{!i+1}`, pages and
  models made inside a block. Checked on programs written for the tests with every value worked out by hand
  (loops with steps, while, exitloop, if on strings and on an estimate's R-squared against NumPy, subroutines by
  reference, local and returning early, include, arguments, stop), the script and the steps giving equal objects;
  and on the Fed's example1.prg with its library, the model and LONGBASE given as `tests/test_frbus_example.py` does:
  the funds-rate shock's responses equal to the hand-written simulation to the last bit (1e-8 required), both ways.
  The FRB/US and Brillet counts are updated, errors still 0.
- **File → Translate EViews program… and the EViews translation panel** (`pymetrix-eviews`, section 3 of
  `docs/design/eviews-plugin.md`). The dialog asks for the program, the workfile data its `wfopen` opens (the bundle,
  or a CSV or Excel file with its date column and frequency) and the program's arguments (`%0`, `%1`); its step
  (`eviews:translate_program`, fixed in the plugin's `command_ids.json`) translates, keeps the program, its script
  and its report in the project, and changes no data; its code is `translate_program(...)`. The docked panel (View →
  EViews translation, at the bottom) lists the session's translations and shows the report: a row per line (an
  included program's under its name), the status in words and in colour, the EViews text, the Python and the reason;
  the program and the script side by side, both scrolled to the selected row's line; a filter for each status with
  its count; **Copy** (the selected rows, or the whole report) and **Copy script**; and **Run as steps**, which runs
  the translation's steps through `AppInterface.run`, so each is recorded and undoable, stopping at the first that
  fails and marking its row. The panel uses only `AppInterface`. Undoing the translate step removes it from the
  panel; a reopened project's translations come back from its History (from the program the project keeps, when the
  file has moved). Tested offscreen and under the Windows font emulation.
- **Assigning a series within a sample** (EViews' `smpl 2000q1 2010q4` then `y = ...`). `data.with_series(y="...",
  within=Sample(...))` assigns only the rows the sample selects and keeps the others' values; a new series is missing
  outside it. `within` takes a `Sample` (periods, a panel's entities, `where=` conditions, windows per entity), a
  `(start, end)` span or `"start:end"`. Lags reach outside the sample, and a formula that uses the series it assigns
  is computed one period at a time, as EViews does, so `gdp="gdp[-1]*1.02"` extends `gdp` by 2% a period.
  `with_series(y=None)` makes a series of missing values (EViews' `series y`). **Data → Generate series…** has
  **Within** under Advanced, a named sample or a first and last period (step schema 3; older steps open unchanged);
  `with_each(..., within=...)` too. Without `within`, `with_series` is unchanged.
- **An empty dataset: `Dataset.empty("Q", "1990Q1", "2030Q4")`** (EViews' `wfcreate q 1990q1 2030q4`), every period of
  the frequency from the first to the last and no series, recorded and replayed; **File → New dataset…**.
- **Extension API 1.10** (additive): the core EViews reader's text helpers under public names in
  `pymetrix.extension`, for a plugin that translates EViews programs: `eviews_blank_comments`, `eviews_statements`
  (statements with their offsets), `parse_eviews_expression` and the tree's node classes (`EViewsNumber`,
  `EViewsName`, `EViewsCoefficient`, `EViewsCall`, `EViewsNegative`, `EViewsBinary`, `EViewsBrackets`),
  `walk_eviews_expression`, `render_eviews_expression` (as `convert_expression` writes), `EViewsExpressionError`,
  `Diagnostic` and `line_and_column`. They are the reader's own helpers; the reader is unchanged. The command and
  function reference gains section 6, the extension API's names (`docs/reference/extension.csv`). `plugins.API` is
  now `"1.10"`: compare versions as numbers.
- **Impulse responses by name: `irf.paths`.** Every impulse-response result (VARs, VECMs, local projections' and
  solved models' `irf(...)`) has a `paths` attribute: a DataFrame with a column per response and shock labelled
  `"y <- shock"` (the response, then the shock, the labels `to_frame()` and `irf["y", "shock"]` already used) and a
  row per horizon, so `irf.paths["y <- ei"][2]` reads what `irf.values[2, 0, 2]` reads by position. The bands stay in
  `to_frame()`; `values` is unchanged. Exercise checks can use it by name, `value = "paths.y <- ei.2"`, as well as
  `values.2.0.2` (docs/exercise-packs.md lists both, checked against the New Keynesian template).

### Removed (breaking)

- **`bandwidth_rule="legacy"` of cointegrating regressions is gone**, as 0.1.0a19 announced: it reproduced
  pymetrix 0.1.0a18's automatic Bartlett bandwidth and lags (arch's before 9.0), which leave out a lag of a
  fractional bandwidth and can give a negative long-run variance. FMOLS, CCR and DOLS use Newey and West's (1994)
  rule, as arch 9 and R's sandwich, the default since 0.1.0a19; results that left the rule out do not change. The
  argument stays and accepts only `"newey-west"`, so scripts that spell out the default still run; asking for
  `"legacy"` (`cointegrating_regression`, `long_run_covariance` and its helpers), including in a replayed script or
  recorded call, raises a `SpecificationError` that says the rule was removed and that pymetrix 0.1.0a20 still has
  it. The dialog loses its **Bandwidth rule** field under Advanced (step schema 2): a step saved by 0.1.0a19 or
  0.1.0a20 with the default rule opens as before, and one saved with legacy stops with an error naming the rule
  rather than giving other numbers. With arch 7 or 8 installed, the tests compare with arch by handing it a
  whole-number bandwidth, under which the two use the same lags.

## 0.1.0a20: first public release (24 November 2026)

The first public release: material that pymetrix may not redistribute leaves the repository, and what
stays is attributed.

- **The Fama-French example is downloaded on first use, not shipped.** Kenneth R. French's data library marks its
  files as Eugene F. Fama's and Kenneth R. French's copyright and gives no licence to redistribute them, so
  `pymetrix-finance` no longer bundles `french.parquet`. The first `Dataset.example("french")` downloads five of the
  library's CSV files, builds the same 35 columns over January 1949 to March 2017 and keeps the dataset in
  `~/.pymetrix/cache/datasets/` (`$PYMETRIX_HOME/cache/...`); later loads need no network. Offline, the first load
  raises a `DataError` that names the library and the cache folder. The library revises its history, so the values
  differ slightly from the copy shipped before (correlations above 0.97 column by column); the tutorial's claims
  still hold on the September 2026 release. The tests need no network: they serve a simulated stand-in in the
  library's file layout, and check the tutorial's claims about the real data only with `PYMETRIX_FRENCH` set to a
  downloaded copy. A plugin's `ExampleDataset.path` may be any path-like object, resolved only when the dataset is
  loaded.
- **Brillet's book excerpts are out of the repository.** The two short excerpts of J.-L. Brillet's *Structural
  Econometric Modelling* (EViews) used in the tests are its author's copyright, shipped without permission, and are
  removed. The three Brillet tests now read the reader's own copies of the book's programs from the folder named by
  `PYMETRIX_BRILLET` (`brillet_basic.prg`, `brillet_small_model.txt`, `structmodel.txt`; or the path of
  `structmodel.txt` itself, as before) and skip without it. `Model.from_file`'s EViews test reads a small model of
  pymetrix's own instead (`tests/fixtures/eviews/small_model.prg`), and checks its equations.
- **Third-party notices.** The FRB/US excerpts in the tests are kept, with their source and terms: the Federal
  Reserve Board's website terms put its information in the public domain, to be cited to the Board, and its FRB/US
  disclaimer asks that the model not be presented as official government material; the fixtures' headers say so.
  `THIRD-PARTY-NOTICES.md` (and the finance plugin's) list the Fama-French data under "downloaded at run time" and
  cite the Board's terms for the H.15 yields. They now also list the recorded data-connector responses the tests
  use (FRED, the World Bank, the IMF, the ECB, Eurostat and the BIS), each with its source and reuse terms.
- **Neutral plugin examples.** The PSETS plugin's tests and the lab book's example brief use a made-up country and
  statistics office (Norland, the Norland Statistics Office) instead of a real one.
- **Release workflow.** `pypa/gh-action-pypi-publish` is pinned to the commit of its v1.14.3 release rather than to
  its `release/v1` branch.
- **Public wording.** The README, GOVERNANCE.md, the objectives page and the release notes for maintainers no longer
  speak of the project as private or invitation-only.

### Fixes

- **The model editor's impulse-response panels (Solution tab, after Check solution) have their titles and tick labels
  in the app's text size again.** The loop over panels reused the name of the text scale, so each panel's title and
  tick labels were sized by its position (0, 8, 16... points; the first drawn at matplotlib's 1-point minimum) and the
  figure's title by the last panel's; they are now 8 and 7 points, and the figure's title 9, times the text scale.

## 0.1.0a19: development build (15 November 2026)

The long-run variance of DOLS and of any fractional bandwidth is fixed (it left out a Bartlett lag and could be
negative, a correctness fix), and FMOLS, CCR and DOLS now choose their bandwidth by Newey and West's (1994) rule as arch
9 and R's sandwich do, so their standard errors change (`bandwidth_rule="legacy"` keeps 0.1.0a18's); exercise packs can
name their models and reach integer labels in checks, Check model shows the code it records, and Model → Analysis →
Impulse responses… gives a solved model's responses.

### Changed

- **FMOLS, CCR and DOLS choose their automatic Bartlett bandwidth by Newey and West's (1994) rule as arch 9 and R's
  sandwich compute it**: the pilot lag is the integer part of 4 (T/100)^(2/9) rather than rounded up, Newey and West's
  S is the bandwidth of the kernel k(j/S) (the weights 1 - j/(b+1) with b = S - 1, rather than b = S), and FMOLS and
  CCR use ceil(S) - 1 rather than ceil(S). Results equal arch 9's `FullyModifiedOLS`, `CanonicalCointegratingReg`
  and `DynamicOLS` and sandwich's `kernHAC` and `NeweyWest` to 1e-10. Standard errors move: with a constant, by up to
  about 8-9% on the examples checked (DOLS -8% to +3%, FMOLS -6% to +9%, CCR -4% to +11%); with a trend, where the
  bandwidth moves more, by -12% to +13%, and by -36% where FMOLS's and CCR's bandwidth fell from 4 to 0 (a simulated
  400-row system). FMOLS and CCR coefficients move by up to about 0.1 standard errors with a constant and 0.6 with a
  trend; DOLS coefficients do not change. `cointegrating_regression(..., bandwidth_rule="legacy")`, or Advanced →
  **Bandwidth rule** = legacy in Estimate → Cointegrating regression…, restores 0.1.0a18's numbers exactly; it will
  be removed in a later release. The reported bandwidth is the b of the weights, as arch 9's `bandwidth`, now usually
  about one less. Scripts and saved steps that leave the rule out use the new rule.
- **Phillips-Ouliaris** (computed by arch) records arch's version in its details: arch 9 chooses the automatic
  bandwidth differently from arch 7 and 8, so the same data can give a different statistic and p-value (US
  consumption on income: p = 0.105 with arch 8, 0.052 with arch 9). Give a bandwidth to repeat a result.

### Teaching

- **Exercise packs name their models.** A `[models.<file stem>]` table in `pack.toml` with `name = "..."` gives the
  session name a model of `models/` is imported under (`[models.nk_2019]` with `name = "nk"` imports
  `models/nk_2019.model` as `nk`), so a walkthrough no longer has to ask users to rename it. The Exercise panel's
  Import…, `model_command()` and `load_model()` use it, and default to the file's stem as before; `load_model`,
  `model_command` and an exercise's `models:` take either name, and `Pack.model_name()` gives it. A table naming
  no file, a name that is not a Python name, or two models with the same name are listed as problems.
- **Check paths reach integer labels.** A part of a check's `value` path that is a whole number is now tried as the
  string key, then as an int key (a label of a Series or a column of a DataFrame with an integer index, such as
  `mroz.wage.2` for woman 2, or a dict's int key), then, for a list, tuple or array, as a position. Before, `"1"` was
  never an int label, so checks fell back to `values.<n>`, a position. Series are still never read by position, and
  calls and `__` names are still refused.

### Models & DSGE

- **Model → Analysis → Impulse responses…** for a solved linear (rational-expectations) model: the solution, the
  shock (listed from the solution), its size, the horizon and the result's name. It records
  `irf = sol.irf(periods=12, shock_sd={'ei': 0.25}, shocks=['ei'])`, so exercises and scripts no longer need the
  Immediate window for a model's responses; the result is the usual impulse-response result (graph and tables), and
  a solution's next steps offer the dialog. `LinearSolution.irf` is now a recorded call and takes `shocks=`, the
  shocks whose responses to keep (each as among all of them, correlated shocks orthogonalised over all), and refuses
  a negative or fractional horizon and unknown shocks with a `SpecificationError`.

### Fixes

- **The long-run variance of DOLS (and of any fractional bandwidth given) can no longer be negative.** With a
  fractional bandwidth b, the Bartlett long-run covariance left out lag floor(b) + 1, whose weight 1 - j/(b+1) is
  positive, so the weights were not a kernel and the estimate was not positive semi-definite: an AR(1) with
  phi = -0.9 at b = 1.5 gave -0.54 (standard errors NaN) where the Bartlett kernel gives +1.40. Every lag j < b + 1
  is now used, as the documented formula, arch 9 and R's sandwich have it. DOLS standard errors move by up to about
  1% from this alone; FMOLS and CCR (whole-number bandwidths) are unchanged by it.
- **Model → Check model…: the code shown is the code recorded.** Show the code built the step as if a dataset called
  `data` existed: with no dataset in the session it showed `check(data)` (and, with a period, `check(, span=...)`,
  which is not Python), and it left out an imported model's own shocks and guesses, which the recorded step has.
  Dialogs whose code depends on the session now show it built from the session's own objects, as the step records
  it; a notebook form shows the recorded code too, and `CheckModel(data="").to_code()` leaves the data out.

## 0.1.0a18: development build (14 November 2026)

Exercise packs bring guided estimation exercises to the app and the `pymetrix-exercises` command; ADF and
Engle-Granger no longer report a statistic of 0.0 in short samples (a correctness fix), the Wald test takes
coefficient names such as `dlog(y)`, and formulas take seasonal dummies by `seasonal()`, `C(quarter)` and `C(month)`.

### Estimation & tests

- **Seasonal dummies in formulas: `seasonal()`, `C(quarter)` and `C(month)`.** `y ~ x + seasonal()` adds one 0/1
  dummy per period of the year (quarters of quarterly data, months of monthly data), named `seasonal[Q2]` ...
  `seasonal[Q4]` or `seasonal[M2]` ... `seasonal[M12]`: the first season is left out when the equation has an
  intercept, and every season has one when it has none (`- 1`), as `C()` does. `C(quarter)` and `C(month)` now work
  too: with no series of that name, the quarter or month comes from the time axis (fiscal quarters for a `Q-JUN`
  axis), in quarterly or finer data for `quarter` and monthly or finer for `month`, ordered by number. The dummies
  follow each row's period, so a sample that starts mid-year or has gaps gets the right ones, and their names work
  in the Wald test and restricted least squares. Estimate → Equation's formula check accepts them; on annual data
  they are refused with the reason. Previously `C(quarter)` failed with "'quarter' is neither a series nor an axis".

### Teaching

- **Exercise packs (provisional).** Guided estimation exercises that ship on their own, as a folder or in a plugin
  (entry-point group `pymetrix.exercises`). Help → Exercises… lists the packs, with Open folder… to add one; Start
  loads the exercise's data into the session once you agree (as File → Load data… steps, undoable) and docks the
  Exercise panel: each step's instructions, Show hint, Show the code, and Check, which runs the step's checks
  against the session and shows each green or red with its message. Checks are data (`checks.toml`): a value is read
  by attribute and key lookup only, never calling anything, so a pack from any folder runs nothing in the app. The
  `pymetrix-exercises` command checks a pack against its reference solutions in a fresh session (`check`, exit code
  1 on a failed check, for a pack's own CI) and scaffolds a new one (`new`). In code, `pymetrix.exercises`:
  `load_pack`, `installed_packs`, `check`, `load_data`, `load_model`. A pack can carry models (`models/`: Dynare
  `.mod`, IRIS `.model`, pymetrix `.pymodel`), which the panel's Import… opens in their import dialogs filled in; a
  reference solution runs with the pack's folder as its working directory and `PACK_DIR` defined. Packs show their
  audience and an optional, informational classification; exercises have a level (beginner, intermediate,
  advanced). Value paths read a result's standard errors, t statistics and p-values by parameter name (`se.x1`,
  `tvalues.x1`, `pvalues.x1`) and `nobs`. The format is in `docs/exercise-packs.md`, with every documented path
  checked on real results; the guide has "Writing an exercise pack".

### Fixes

- **Engle-Granger and ADF no longer return a statistic of 0.0 in short samples.** With Maximum lags left empty on 16
  annual observations, Engle-Granger searched up to 7 lags (statsmodels' default, `min(ceil(12 (n/100)^(1/4)),
  n//2 - terms - 1)`), which leaves the residuals' ADF regression 8 rows and 8 regressors: no residual degrees of
  freedom, a zero sum of squares that the information criterion picks, and a t-ratio of 0.0, reported without an
  error; with simulated critical values every draw was degenerate the same way, so the critical values were 0 too.
  The same happened with 20 observations, with ADF without deterministic terms on an even number of observations,
  and in the Fisher-type and Pedroni panel tests on short series. The default largest lag is now Schwert's rule
  capped so that the largest search regression keeps 4 residual degrees of freedom, counting, for Engle-Granger and
  Pedroni, the cointegrating regression's columns as used: `min(ceil(12 (n/100)^(1/4)), n//2 - terms - 1,
  (n - 2 - terms - used - 4) // 2)` (4 lags at 16 observations). It is statsmodels' rule from about 26 observations
  on, so longer samples give the same results as before. A lag or largest lag given that leaves no residual degree of
  freedom is refused with a `SpecificationError` that gives the largest allowed, and a statistic of exactly zero or
  undefined from the data is refused rather than reported. The fixed results equal statsmodels' `coint` and
  `adfuller` given the capped largest lag, and arch's `engle_granger` and `ADF` at the lag chosen.
- **The Wald test takes coefficient names with parentheses.** A coefficient named after a formula term such as
  `dlog(y)` or `log(P)` gave "unknown coefficient(s) ['y']": the Wald test parsed the hypothesis as a formula, so
  `dlog(y)` was read as the function dlog of a series y. Restricted least squares accepted the same names. The Wald
  test (`fit.test("wald", ...)`, `fit.wald(...)`, Test → Run test…) now reads hypotheses with restricted least
  squares' parser: any coefficient name is matched whole (`dlog(y)`, `x[-1]`, `log(P)`, `const`, a name with a comma
  such as `L(x, 1)`, whose spaces are optional), and a name may be put in backticks (`` `dlog(y)` ``), in the Wald
  test, restricted least squares and system restrictions alike. A system's Wald test reads `c.dlog(y) = i.dlog(y)`
  too, which also failed. Results equal those of the same hypothesis written with plain names.

## 0.1.0a17: development build (13 November 2026)

Projects with deferred steps reopen, Check model takes imported models, and heatmaps colour each column on its own
scale; row conditions and model-file expressions are checked before they run (a project's row condition could run
code on opening), dependency floors are raised past known advisories, the repository is public, and the first How
do I… task pages are in the help.

### Fixes

- **A project with a step added deferred reopens after what it uses is redefined.** With `Session.add(cmd,
  deferred=True)` followed by a step that makes one of its inputs again (a model defined, an estimate added deferred,
  the model defined again), saving marked the first definition as a dead end, since a step that has not run had no
  recorded inputs, and opening the project failed with "no object named 'm'". A deferred step's inputs are now told
  from its command when it is added (the objects its fields name and its call's references): the steps that make them
  are kept, and `Session.compute()` runs it against them as they were at its place in the history, as if it had run
  there (a later redefinition then makes its result out of date, as it would have). On opening, a deferred step whose
  inputs are missing stays deferred with a note (History's tooltip, the Problems list), in either calculation mode;
  a step that uses a result not computed yet waits for it, even when a later step changes its other inputs; opening
  a project never fails because of a pending step. Projects saved by 0.1.0a16 with such a step marked dead open, the
  step run again. `Session.add(..., deferred=True)` now refuses a step whose dialog names an object that does not
  exist, as running it would.
- **Model → Check model… takes an imported model.** Its model picker offered only equations, so a model read by
  Model → Import IRIS file… or Import Dynare file… (`read_iris`, `read_mod`) was not listed; typed in, it was checked
  without its shocks or steady-state guesses (no shocks row, and a steady state searched from zero). The picker now
  lists every whole model in the session, however it was made (imported from IRIS, Dynare or EViews, or built), and an
  imported model is checked with its own shocks and guesses unless Shocks are given: the result equals the
  template's own `check()`, and the script (`Model([iris]).check(shocks=(...))`) replays.
- **The simulation estimators take a pandas Series.** `indirect_inference`, `simulated_moments` and
  `efficient_method_of_moments` accept arrays, but a Series (a dataset's column) or a DataFrame failed: the recorded
  call turned plain pandas `data` into a `Dataset`, which the estimators could not read as numbers ("setting an array
  element with a sequence"), and a Series without periods was refused outright. A Series, a one-column DataFrame or a
  one-series Dataset is now read as a vector (a wider one as periods by series), and gives the same estimates as the
  same numbers in an ndarray.
- **Solve model: the solver's own settings with the other solver settings.** Newton's Tolerance, Convergence, Line
  search and Refresh were at the foot of the Advanced settings, under a second "Solver settings" label, away from
  Ordering and Fallback under the first. They now follow those, under the name of the solver chosen ("Newton"), and
  change with it (Gauss-Seidel's own, none for First order). The step the dialog makes is unchanged.
- **Browse series: one "Provider".** With SDMX chosen, File → Browse series… labelled both the service and the
  SDMX provider (ECB, IMF...) "Provider". The second is now "Source", as for the World Bank's sources and as
  `connectors.search(..., source=...)` calls it.
- **Heatmaps of tables whose columns are in different units.** The Heatmap button coloured a whole table on one
  scale, so a posterior summary's effective draws (in the hundreds) took every colour and its means, HPD bounds and
  R-hat (0 to 1) were all white. `heatmap_figure` and the `Heatmap` layer take `scale`: `"table"` (one scale, the
  default, as before), `"column"` (each column on its own: its smallest value at one end of the colour map and its
  largest at the other; the cells still show their values, and a note, "each column coloured on its own scale",
  replaces the colour bar) or `"auto"`, which picks `"column"` when one column's typical size (the median of its
  absolute values) is more than ten times the largest absolute value in another, and `"table"` otherwise and always
  for a square matrix with the same labels on rows and columns (correlations, covariances), columns that are
  horizons or dates, columns that all take both signs (responses, residuals) and shares (rows or columns adding up
  to the same multiple of 1 or 100: variance decompositions). The button uses `"auto"`, and leaves out a posterior
  table's R-hat, effective-draws, ESS and MCSE columns (`diagnostics=False`), which describe the sampler rather than
  the posterior; recorded as `heatmap_figure(posterior, view='Summary', scale='auto', diagnostics=False)`. The hover
  readout and a click give the cell's value, not its shade, and `fig.matplotlib_code()` draws the same.
- **Readable numbers in heatmap cells.** The values were pale grey on pale cells. Each is now dark grey or white,
  whichever has the higher contrast with its cell (WCAG 2), and black where dark grey falls short of 4.5:1, so every
  number reads at 4.5:1 or more on any colour map.
- **The Heatmap button on a table with repeated column names** (a VAR's Summary: `(se)` after each variable) no
  longer fails with "The truth value of a Series is ambiguous".

### Security and licensing

- **The repository is public.** The README, PyPI page, known gaps, roadmap, pitch, install guide, contributing
  guide, how-to-help page, review issue form and release workflow now say so, and no longer describe a closed review.
  `pymetrix` and each plugin now give PyPI their project URLs (Homepage, Source, Issues, Changelog) on
  github.com/GeoffHeenan/pymetrix.
- **Each plugin ships its licence.** The first-party plugins in `packages/` (finance, lab book, guide, assistant,
  problem sets, seasonal) carry a copy of the BSD-3-Clause `LICENSE` and name it in `license-files`, so their sdists
  and wheels include it.
- **Plugins ship their notices; new plugins get a licence.** `pymetrix-finance` ships a `THIRD-PARTY-NOTICES.md` in
  its sdist and wheel for the datasets it bundles from elsewhere (the Ken French data, with linearmodels' NCSA notice;
  the H.15 Treasury yields through FRED), and `pymetrix-seasonal` one for the JDemetra+ libraries it downloads when
  asked and the X-13 program it runs. `pymetrix-new-plugin` now writes a BSD-3-Clause `LICENSE` in the author's name
  and names it in the new plugin's `license-files`.
- **Third-party notices.** `THIRD-PARTY-NOTICES.md` lists the bundled datasets and test fixtures that come from
  elsewhere, with their sources and licences (the Wooldridge sets through linearmodels, NCSA; statsmodels' public
  domain sets; the Ken French data; Federal Reserve H.15 yields through FRED; the FRB/US and Brillet excerpts used in
  tests), and the JDemetra+ libraries that `pymetrix-seasonal` downloads when asked (EUPL 1.2). It is included in the
  `pymetrix` sdist and wheel. The simulated `sample_macro` and `sample_monthly` datasets now state their licence
  (BSD-3-Clause) in their metadata, like the other examples; their values are unchanged.
- **Higher dependency floors for known advisories.** `pyarrow>=17` (PYSEC-2024-161, fixed in 17.0.0) and
  `scikit-learn>=1.5` (PYSEC-2024-110, fixed in 1.5.0) in the package's requirements and extras; `pytest>=9.0.3`
  (PYSEC-2026-1845) for development and in the plugins' test extras and the plugin template. The oldest-versions CI
  job installs pyarrow 17.0.0 and scikit-learn 1.5.0.
- **Expressions in model files are checked before they run.** Dynare parameter, `initval` and shock values, IRIS
  `!if` conditions and `!parameters` values, the model editor's Dynare check and simple MATLAB data files were
  evaluated with Python's `eval` and no built-ins, which an expression such as
  `().__class__.__bases__[0].__subclasses__()` can get around. Each is now parsed and checked first
  (`pymetrix.models.safe_eval`): numbers, the file's own names, arithmetic, comparisons, calls to the functions
  Dynare allows (`exp`, `log`, `normcdf`, ...), `and`/`or`/`not` in IRIS conditions and `x(:,k)` columns in MATLAB
  files. Attribute access, other indexing, `lambda`, comprehensions, keyword arguments, strings and names starting
  with `__` are refused with a `ModelError` saying what was found. Whole numbers are evaluated as floats, so a value
  such as `9^9^9^9` overflows at once instead of running for ever. Every expression the Dynare, IRIS, EViews and
  model-editor tests evaluate gives the same result as before.
- **Row conditions are checked before they run.** A sample's or a step's row condition (`Sample(where=...)`,
  `data.select(where=...)`, `where=` in estimation, the dialogs' **Rows where**) was handed to pandas'
  `DataFrame.eval`, which can reach Python variables with `@` and call methods on them: a condition such as
  `@self.write('file')` in a project from someone else wrote a file when the project was opened, even with its typed
  code not trusted, as did one saved in a dataset's active sample. Each condition is now parsed and checked first
  (`pymetrix.data.conditions`): column and axis names, numbers, text and dates in quotes, comparisons, `in` and
  `not in` with a list, `and`/`or`/`not` (and `&`, `|`, `~`), arithmetic and pandas' functions (`abs`, `sqrt`, `log`,
  `exp`, ...). `@` references, attribute access, indexing, other calls and names starting with `__` are refused with
  a `ConditionError` (a `SpecificationError`) before anything runs; a project step that has such a condition is kept,
  not run, and the project still opens. A condition that passes selects the same rows as before: every condition in
  the documentation, the help and the tests was compared with the old evaluation.
  0.1.0a16 and earlier let a project's row condition run code when the project was opened, even with
  `load_project(trust=False)`; 0.1.0a17 fixes it.

### Documentation

- **How do I… task pages.** Pages that each take one task from start to finish: the steps in the app (menu paths
  and fields as the dialogs show them), the same in code, how to check the result, and the pitfalls. The first
  three: estimate an equation and test its residuals; bring an IRIS or Dynare model across, then check and solve
  it; test for cointegration in a short annual sample, with simulated critical values. Help → How do I… opens their
  index, the help contents list them first, and Ctrl+K finds them. They are Markdown files in the package
  (`src/pymetrix/howto/`), shown by the help as they are; `docs/how-do-i/README.md` has the plan for the rest. A test
  runs every page's code, and runs its steps in the app as dialog commands to check they give the same results.
- The Critical values choice of ADF and Engle-Granger reads "MacKinnon (asymptotic)" and "Simulated for this
  sample" (was "Mackinnon" and "Simulated"); help text that named Model > State space > now names Model > Analysis >,
  where those dialogs are.

## 0.1.0a16: development build (10 November 2026)

### Fixes

- **A Dynare `long_name` with parentheses or a % sign.** `var y (long_name='Output gap (% of potential)') pi;` ended
  the description at its first ")", so `read_mod` lost the names after it ("2 equations for 3 variables"), and the
  model editor's check took the % for a comment ("pi is not declared"). Both now read the quoted text whole. The QPM
  toolkit's IRIS model, written as Dynare from the model editor (Write as → Dynare), had 13 errors and was "not
  checked against the original"; it now reads back and gives the same decision rules.
- **A wide result table opens in seconds.** Sizing each column of a result table refitted every row, once a column,
  so a table of hundreds of columns took quadratic time: the solution of a 200-equation model took about 27 seconds
  to open, and as long again at each change of View → Text size. The rows are now fitted once, after the columns
  (about 2.5 seconds for that solution); a column widened by hand still makes the table's frame follow.
- **Model editor: Check solution is not called stale at once.** Check solution within a moment of the text changing
  (a file just opened, typing just stopped) was followed by the check still due from the typing, which said "The
  text has changed since the last check" though nothing had. A check now takes the place of one due, and the
  Solution tab is called stale only when the text or its syntax differs from what was solved.

## 0.1.0a15: development build (9 November 2026)

### Models & DSGE

- **A convergence test that does not depend on the equations' units.** `Newton(convergence=...)` and
  `GaussSeidel(convergence=...)`: `"residual"` (the default, as before: the largest residual below `tol`), `"step"`
  (the largest scaled step `|x_new - x_old| / (1 + |x_old|)` below `tol`) or `"both"`. An equation multiplied by a
  million takes the same number of iterations under `"step"`. With a test other than the residual's,
  `Solution.convergence` records which test stopped each period (`test`), the solver (`solver`) and the Jacobians
  evaluated (`jacobians`). Model → Solution → Solve model: Solver settings, Convergence.
- **A failed period retried by other solvers.** `Fallback(GaussSeidel(), GaussSeidel(damping=0.5, max_iter=400),
  Newton())` tries its solvers in order on a period the first cannot solve, from that period's starting values;
  periods the first solves are as before. `Solution.convergence`'s `solver` column names the solver that solved each
  period; when every one fails, the error lists each attempt with its worst equation. Solve model: **Fallback**
  (Advanced) retries with damped Gauss-Seidel, then Newton with a line search.
- **Newton with a line search.** `Newton(line_search=True)` backtracks (Armijo) when the full step does not reduce
  the residuals enough, so a start far from the solution, where plain Newton overshoots, still converges. Off by
  default.
- **Newton keeping its Jacobian.** `Newton(refresh="period")` evaluates and factorises the Jacobian once a period and
  keeps it while each step cuts the largest residual to a quarter or less (a modified Newton), evaluating it again
  when convergence slows; `"iteration"` (the default) is as before. The Jacobians evaluated are counted in
  `Solution.convergence` (`jacobians`).
- **Block-recursive solution.** `Model.solve(..., ordering="blocks")` splits each period's equations into blocks
  (the strongly connected components, with holds and free variables paired by matching), solves each recursive
  equation once, in order, and iterates only within the simultaneous blocks, with the solver given (Newton,
  Gauss-Seidel or a Fallback, block by block). The same solution with far fewer equations evaluated when most
  equations are recursive; `Solution.convergence` gives the blocks and recursive equations by period and the blocks
  themselves in `.attrs["blocks"]`. `ordering="joint"` (the default) is as before. Solve model: **Ordering**
  (Advanced).
- **The seconds each period's solve took.** A period-by-period solve (Newton, Gauss-Seidel, a Fallback, with or
  without blocks) records them in `Solution.convergence.attrs["seconds"]`, one a period, timed by
  `time.perf_counter` around the solver's call; the table's columns and every result are as before. Kept with a
  solution kept in a project; a `.pymxr` file keeps the Convergence table without them. Perfect foresight, which
  solves all periods at once, does not record them.
- **How-to: solving large backward-looking models** (docs/solving-large-models.md, and the help topic of the same
  name): when to use `ordering="blocks"`, `Fallback(...)`, `convergence="step"`, `Newton(line_search=True)` and
  `Newton(refresh="period")`, how they combine, a recommended starting configuration for a model of mostly
  identities with a few simultaneous blocks, and how to read `Solution.convergence`. A test runs the page's code.
- **Benchmark of the solver options** (`python benchmarks/solve_options.py`): a generated model of 200 equations
  (a recursive chain of identities and two small nonlinear simultaneous blocks) over 40 quarters, solved by default,
  with `ordering="blocks"`, with `Newton(refresh="period")` and with the recommended configuration: median wall time,
  time in the periods' solves, equations and Jacobians evaluated, and the largest difference from the default's
  solution. `--model`, `--data` and `--span` time a model of your own.

### Estimation & tests

- **Robust regression (M-estimation): `method=RobustLS(norm="huber" | "bisquare", tuning=None, scale=None)`.**
  Huber's norm down-weights large residuals; Tukey's bisquare gives residuals beyond its tuning constant no weight
  (defaults 1.345 and 4.685, 95% efficient with normal errors). Estimated by iteratively reweighted least squares from
  the least-squares start, with the scale re-estimated each round by the MAD of the residuals (or held at a given
  `scale`); standard errors are Huber's H1, as statsmodels' RLM computes them. `fit.extra["robust_weights"]` gives each
  observation's final weight; the summary notes the norm, the scale and the iterations. Estimate → Equation offers it
  as **Robust regression (M)**, with the norm in Method settings and the tuning constant, a fixed scale, the iterations
  and the tolerance under Advanced. Checked against statsmodels' `RLM` with `HuberT()` and `TukeyBiweight()`
  (default and other tuning constants): coefficients, standard errors and scale to 1e-8; with a fixed scale, the
  estimating equations and the H1 formula.
  Help topic "Robust regression".
- **Restricted least squares: `OLS(restrictions=[...])` and `IV(restrictions=[...])`.** Linear restrictions on a
  linear-form equation's coefficients, such as `"y + c[-1] = 1"` or `"const = 0"` (several in a list, or separated by
  commas), are imposed in estimation: `b_r = b - A R' (R A R')^-1 (R b - r)` with `A = (X'X)^-1` (2SLS:
  `(Xhat'Xhat)^-1`), and the covariance of the chosen kind (classic, White, HAC, clustered) from the restricted
  residuals, mapped by `M = I - A R' (R A R')^-1 R`, with one more residual degree of freedom per restriction: the
  substitution estimate and its covariance. Coefficients are named as in the summary; a restriction that cannot be
  read, names an unknown coefficient, is not linear, involves no coefficient, or repeats or contradicts another raises
  an error that quotes it. The parser is the one system estimation uses (`pymetrix.estimation.restrictions`), shared
  now. Estimate → Equation: **Restrictions** in OLS's and IV's Method settings; the step records
  `OLS(restrictions=[...])`. Without restrictions (the default) results are unchanged. Checked against the
  substitution estimate and its delta-method covariance for OLS (classic, White, HAC, clustered) and IV (classic,
  White), to 1e-12 (coefficients) and 1e-10 (covariance, relatively).
- **Estimate → Cointegrating regression: Start, End and a row condition.** The dialog gains **Start** and **End**
  (recorded as `span=`) and, under Advanced, **Rows where** (`where=`); `cointegrating_regression` and
  `gregory_hansen` take `where=` (a condition on the rows, as elsewhere), the rows kept taken as consecutive periods,
  as rows with missing values already were. Empty boxes record nothing and give today's results.
- **Small-sample critical values for ADF and Engle-Granger: `critical_values="simulated"`, with `draws=2000` and
  `seed=0`.** MacKinnon's response surfaces are asymptotic, and with 13 to 30 observations they are far off (the lag
  search moves them further). With the option, the null is simulated for the sample at hand: independent Gaussian
  random walks of the sample's length (one for ADF, one per variable for Engle-Granger), each tested exactly as the
  data are, with the same trend terms, the same ADF regression on the residuals and the same lag rule (`autolag`,
  `max_lags`), searched and refitted. The result reports the 1, 5 and 10% critical values, the simulated p-value (the
  share of draws at or below the statistic) with its Monte Carlo standard error, the draws and the seed, and a note
  that the values are simulated for this sample; the provenance records the seed, the draws and the random
  generator. Engle-Granger also takes `exog=`: fixed regressors that are not I(1) (impulse or break dummies) in the
  levels regression of the data and of every draw. Each draw's statistic comes from `pymetrix.stats.adf_arrays`, an
  ADF on NumPy arrays that reproduces statsmodels' `adfuller` and `coint` lag for lag (statistics within 1e-9) and computes 2,000
  draws at n = 20 in a few hundredths of a second. In code:
  `data.test("engle_granger", "lc, ly", critical_values="simulated", exog="d2008")`; in the app, **Run test…**'s Test
  settings. The default, `"mackinnon"`, gives exactly today's results.
- **Monte Carlo standard errors of posterior means in Bayesian equations.** A Bayes fit's posterior table and summary
  gain an MCSE column beside the ESS: the standard deviation of the draws over the square root of their effective
  sample size, how far the mean of the draws may be from the posterior mean itself (as Stan and ArviZ report it).
  `fit.stats` gains `mcse_<name>` and `ess_<name>` for each parameter, and `pymetrix.models.bayes.mcse(chains)`
  computes it for any draws. The estimates and the other columns are unchanged.

### Sessions, projects and saved results

- **A step can be recorded without running it: `Session.add(cmd, deferred=True)`.** The step is added as Manual
  mode defers a slow step when a project is opened: listed as not computed (`session.pending()`, History, the
  Objects panel), marked in the script, saved deferred in the project, and run in its place by `session.compute()`
  (Calculate → Compute). What it will make is read from the step (a call's `output`, a dialog's `name`, the names
  typed code assigns) or given as `outputs=`; it is checked as far as it can be without running, and Undo takes it
  back. `Session.add(cmd)` without `deferred` runs the step, exactly as `Session.run(cmd)`. History now shows a
  deferred step as "not computed" (grey) rather than as failed.
- **A saved result keeps the call that made it.** `save_result` (File → Save result…) writes the call a session step
  made the result with into its provenance: `provenance["call"]` (the function's ID, its arguments as recorded,
  with the span and row condition as given, and the name it was kept under) and `provenance["call_uses"]` (the
  datasets and results it read, by name), so `Call(**provenance["call"]).invoke({"data": data})` makes the same
  numbers again. A result made outside a session can be given its call (`save_result(fit, path, call=make_call(...))`);
  otherwise the file is as before. `pymetrix.calls.call_of(result)` gives the call a session step made a result with.
- **The pymetrix commit beside the version.** Run from a checkout of its source, pymetrix records the git commit and
  whether files under the package differed from it (`dirty`) in each result's provenance (`provenance.build`, shown
  on the Provenance tab), at each save in a project's `save_log`, and in a saved result's manifest (shown on its
  Saved tab). An installed package records `"commit": None` and a note that the version identifies the code. It is
  read once per process (`pymetrix.build.build_info()`), only when the package's folder is tracked by git, with a
  two-second limit and without taking git's locks; it never raises. Results saved before keep no commit rather than
  taking today's.
- **Reading and writing projects from programs: a documented public interface, provisional before 1.0.** `Call`
  (`invoke`, `to_code`), `make_call`, `call_of` and `call_record` (`pymetrix.calls`), the codec's `encode`, `decode`
  and `Ref`, `Session` with `attachments`, `save_project`/`load_project` and `save_result`/`load_result` are listed
  in their modules' `__all__` (`pymetrix.commands.session` now has one; `pymetrix.saved` adds `result_bytes` and
  `result_from_bytes`) and documented together, with examples that are run by the tests, in the user guide's new
  section "Reading and writing projects from programs" and the help topic of the same name. They are public, and may
  still change before 1.0.

### Data & dialogs

- **Weighted least-squares reconciliation.** `reconcile(..., method="weighted")` minimises
  sum_c sum_t e_ct^2 / w_c: each component's adjustments have a constant variance, its weight from `weights=`
  (default 1 each), whatever the component's level and sign, so a component entering with a negative sign (imports)
  is adjusted as one with a positive sign, and with equal weights the adjustment is the minimum-norm one. The
  constraints are those of the other methods (the components add up to the total, keep their low-frequency totals,
  fixed ones held exactly). Weights must be positive; a component that should not move goes in `fixed`. The default
  (`denton`) and the other methods give the same results as before. **Data → Transform → Reconcile components…**
  offers it as a choice of Method (no change to the step's fields).
- **File → Browse series… groups an SDMX provider's dataflows into folders.** The IMF lists hundreds of dataflows,
  unsorted; they are now sorted (ignoring case) and grouped by family, the leading name with its abbreviation
  ("Monetary and Financial Statistics (MFS)"), each dataflow labelled by the rest of its name ("Interest Rates",
  "2026 January"). In a family the dataflow with the family's name alone comes first, then the others by label, each
  label's dated releases newest first (the dates are read, not sorted as text). A family of one stays a single row,
  and a provider whose names have no families (the ECB's) gets a flat, sorted list. Every row's tooltip gives the full
  name and the dataflow's ID. The words in the search box filter the list as they are typed, opening the folders with
  matches. What is loaded and recorded is unchanged (`Dataset.read(address)`). The grouping is a function of the
  names alone (`pymetrix.ext.dataflow_groups.group_dataflows`), so it applies to any SDMX provider.
- **Browse series shows an SDMX provider's own folders.** The provider's category schemes and categorisations
  (`categoryscheme/{agency}/all/latest?references=parentsandsiblings`, kept a week like other metadata) put its
  dataflows in its own nested folders, such as the ECB's "Monetary and financial statistics › Balance sheet items" or
  Eurostat's "Economy and finance › Prices › HICP"; dataflows in no category go in "Other", and a dataflow in two
  categories is in both. Where a provider publishes no categories, the request fails, or its categories are not
  useful, the dataflows are grouped by name family as before, else by the first part of their ID (Eurostat's
  `nama_`, `prc_`, `une_`, `ei_`, named by theme), else listed flat. The rule is explicit: a grouping is used when it
  has at least two folders, its folders hold at least half the dataflows and no folder holds more than four fifths,
  so a coarse IMF scheme gives way to the name families. The tree's header says which ("Browse · by category"; its
  tooltip says why another was not used). The search box's words filter the tree at any depth, opening the folders
  down to each match, and what is loaded is still `Dataset.read(address)`. In code:
  `pymetrix.ext.connector_search.dataflow_tree("ECB")`. Tested on hand-written ECB-, Eurostat- and IMF-like answers,
  not yet on the providers' own.
- **A long World Bank topic or source has a folder per indicator-code prefix** (`NY.`, `SL.`, `SP.`, `NE.`…, named
  from a small map, such as "NY. National accounts: income and GDP", else the prefix itself), when it lists more than
  40 indicators; the topic itself still lists them all. In code: `connectors.browse("worldbank", "topic/1/AG")`.
- **Browse series' result columns sort by clicking their headings**: ascending, then descending, then back to the
  provider's order (for FRED, its relevance order, `search_rank`, which is now asked for explicitly and kept; FRED
  gives no score to sort ties by). Ties keep the provider's order, empty cells go last, and the selection follows its
  rows. The tree has a little more of the window's width.
- **SDMX dimensions' codes are found however the provider gives them.** Choosing the IMF's World Economic Outlook
  showed its dimensions with empty lists of codes. A dimension's codes are now found as SDMX 2.1 allows: the
  dimension's own code list (by agency, id and version), its concept's representation, a code list of the same id
  kept by another agency, or the code list asked for on its own (`codelist/{agency}/{id}/{version}`); references
  may be URNs; and the dataflow's content constraints narrow them to the codes it has. A dimension whose codes still
  cannot be loaded says so, and why, in its box, and takes typed codes (`Dimension.note` gives the reason in code;
  `Dimension.descriptions` the codes' descriptions). Tested on hand-written structures, not yet on the IMF's answer.
- **Search inside the chosen dataflow.** With a dataflow chosen, the search box searches the names and descriptions
  of its codes ("current account" finds BCA and BCA_NGDPD under WEO's INDICATOR), listed with their dimension;
  double-click a code, or press Enter, to choose it. A tick box beside Search ("In WEO's codes") shows the mode;
  untick it to search the provider's dataflows again. Choosing another provider forgets the dataflow chosen.
- **Browse series' Series key box says what goes in it** before a dataflow is chosen.

### App & figures

- **Icons by object kind.** The Objects panel's rows, the windows' title bars (and tabs) and the objects in Find
  anything (Ctrl+K) show a small icon of the object's kind: dataset, series, equation, system or VAR, model (with its
  solutions and scenarios), forecast, figure, report, test result, table or matrix, or a generic one. They are drawn
  in code (`pymetrix.ui.icons`, no picture files or new dependencies) at each size a screen needs, in the theme's
  colours on light and dark palettes, with shapes that differ so they tell apart without colour. The kind comes from
  one table keyed by class (`icons.CLASS_KINDS`); plugins' objects get the generic icon unless they subclass a known
  kind. The rows' text is unchanged. The icons grow and shrink with View → Text size (below).
- **One text size for the whole app: View → Text size** (Smaller, Normal, Larger, Largest: 85%, 100%, 115% and 130%
  of the system's font). Dialogs, panels, tables, result windows, help, the code editors and the Immediate console all
  follow, and so do charts on screen and the objects' icons; exported and copied figures keep their fixed sizes. The choice is kept for the
  next start and applies at once, without a restart; windows reflow and still fit (a dialog scrolls up and down on a
  small screen, never sideways). The per-window zoom (Ctrl+=, Ctrl+-, Ctrl+0) is unchanged and multiplies the text
  size. No shortcut of its own: the zoom keys are taken.
- **Consistent spacing.** Margins and spacing in the app come from a few named sizes on a 4-pixel grid that scale with
  the text size (`pymetrix.ui.metrics`). Every dialog has the same margins; a section heading (a form's "Sample"),
  the Advanced settings and Show the code toggles, the Start window's headings and a group box all have the same room
  above them; the deliberate exceptions say why where they are set.
- **The model editor reads the QPM toolkit's IRIS dialect, as `read_iris` does.** A file that `read_iris` built
  (`!modelname`, hyphenated or aliased sections such as `!Transition-Variables` and `!dynamic-equations`, `!shocks:transition`,
  `//` and `#` comments, `name:` equation labels, descriptions after names, `!parameter-values`) failed in
  `ModelSource(text, "iris")` with "unknown section !modelname": the editor's check had its own section list and
  splitting. It now uses `read_iris`'s own (one section table, comment stripping, declaration and equation-name
  reading, with places kept), so `check()` is clean and `build()` gives `read_iris`'s model; errors in the dialect are
  still placed at their line and column, and a `%` inside a quoted description is no longer taken for a comment.

### Mac

- **Message boxes say what failed on a Mac.** macOS draws message boxes without a title (Qt drops it there), so a
  failed dialog step's "Estimate an equation: could not run", and "Could not save the project", "Could not open the
  project" and the app's other failure titles, were lost on a Mac, leaving only the bare error. There the title is now
  the first line of the box's text; other systems are unchanged.
- **One BLAS thread on a Mac too.** NumPy on a Mac uses Apple's Accelerate, which threadpoolctl cannot limit, so
  estimation, sampling, identification and sweeps ran with Accelerate's own threading there. On macOS 15 or later
  pymetrix now sets Accelerate's own switch (single-threaded for a limit of one, restored afterwards); before macOS
  15 the limit still does nothing there (docs/known-gaps.md).
- **Help → Keyboard shortcuts gives a Mac's keys everywhere.** The Code windows' Find was listed as Ctrl+F; it now
  reads ⌘F on a Mac, as the menus' keys already did.
- **Tooltips name a Mac's keys.** The workspace buttons (⌘1 to ⌘9, ⌘` for Code), the Code window's Save, Find,
  Replace and Comment, the spreadsheet's Apply (⌘↵), a result's list of views (⌘⇟, ⌘⇞) and the Start window's tour
  (⌘K finds any command) said Ctrl+… on a Mac; they now say the keys as the platform does, through one helper
  (`pymetrix.ui.keys.key_text`, Qt's native text, which the Keyboard shortcuts list also uses). Linux and Windows
  read as before (the Apply tooltip says Ctrl+Return, its key; Replace shows the platform's own Replace key, or none
  where the platform has no Replace key). A test keeps "Ctrl+" out of the app's text outside a short list of reasons.

### Assistant (plugin)

- **Full context to a model on this computer.** Under the local model's address, **Full context (it stays on this
  computer)** also sends the steps' code and lets the model read an object's tables when it asks, without asking
  each time, since nothing leaves the computer. The record keeps what was sent. It is the project's choice, off
  until ticked.
- **Continue where you left off.** A reopened project offers its last conversation with the assistant (**Continue
  where I left off**). It is put back on screen from the project's record and given to the model with the next
  question. It is only offered when questions go to the same place, with the same mode and full-context choice, as
  when it was held, so answers made from full context never go to a destination told less. **New conversation**
  starts afresh; the record numbers each conversation.
- **Proposal cards show their whole reason.** They sit below the conversation in a split you can drag, and scroll;
  a long reason (as local models give) was cut off to one line.

### Project

- **CI on a push to `main` includes one macOS job** (full, macOS, Python 3.13) beside Windows, so a change that breaks
  only on a Mac shows on the merge rather than at the weekly run. Pull requests still run the Linux subset;
  docs/releasing.md now lists which jobs run on each event.
- **Plugins can be released to PyPI.** The `release` workflow asks which package to release: pymetrix or one of the
  plugins in `packages/`. A plugin is tagged `<plugin>-v<version>`; the workflow checks the tag against the plugin's
  version, builds it, publishes it to TestPyPI, installs it from there and loads its entry points before PyPI.
  docs/releasing.md has the steps.
- **The R comparison for TRAMO-SEATS runs on a pull request only when it touches seasonal adjustment** (the plugin,
  `seasonal.py`, `jdemetra.py`, `extension.py` or the workflow). Every push to `main` still runs it, as do the weekly
  and on-demand runs.

### Fixes

- **No crash on ticking a series in a list.** Ticking an entry in a dialog's list (Regressors in Estimate →
  Cointegrating regression, for one) refilled the list while Qt was still telling the dialog's other handlers about
  the tick, so a handler after the refill was handed an entry already deleted: a crash with PySide6 6.6 and on macOS
  once the dialog had a row condition. A list now keeps its entries when its choices have not changed, and one
  handler per list runs the others in turn.
- **`show()` goes to a window still open.** With two app windows, closing the older one first left `show()`, once
  the other closed too, sending results to the closed window: nothing was shown or printed. A result now opens in the
  newest window still open, whichever closes first, and `show()` prints again once every window has closed.
- **Every field in view in a dialog has its access key.** When a dialog has more fields than free letters, the
  letters went to fields in the order they were first given out, so a field in view could be left without one while
  a hidden setting had a letter it could have taken (Estimate → Equation with GMM: Steps). The fields in view now
  keep their keys, and hidden settings take the letters left.
- **Opening or closing a dialog's Advanced settings refits the window with PySide6 6.6.** The section scheduled the
  refit (and a hidden field's access key its second focus) with `QTimer.singleShot(0, dialog, callable)`, a form
  PySide6 6.6.0, the oldest supported version, does not have: it raised TypeError in the slot, so the dialog kept its
  old size. A new helper, `pymetrix.ui.timers.single_shot`, uses that form where PySide6 has it and otherwise a
  single-shot timer owned by the dialog, so the callable still never runs once the dialog is gone.
- **Closing a window no longer leaves two threads behind.** The Immediate window's in-process Jupyter kernel keeps a
  ZeroMQ context of its own, with an I/O thread and a reaper; stopping the kernel never ended it, so every window
  closed kept both threads for as long as the app ran (a Windows test worker had over 200). The console now terminates
  the context when it shuts down, and the test suite shuts down consoles its tests leave open.
- **No crash after TRAMO-SEATS when a window with a timer is freed.** Once Java was running, JPype ran Python's garbage
  collector on a Java thread whenever Java's had run; a window or dialog freed there was destroyed on that thread with
  its timers still registered, and the next timer event crashed the app (an access violation on Windows). While the app
  runs, Python's automatic collection is now off and the GUI thread collects instead, a few times a second
  (`pymetrix.ui.collector`); the test suite does the same once it has made a QApplication.
- **CI on Windows names the C/C++ code behind a crashed test worker.** The Windows jobs turn on Windows Error
  Reporting's crash dumps for python.exe before the tests (clearing the runner's SEM_NOGPFAULTERRORBOX, which kept
  crashes from it); when a worker dies of a native fault (an access violation, 0xC0000005), the job prints the
  faulting thread's native stack with cdb and keeps the dump as a `crash-dumps-…` artifact for five days. A passing
  run pays only the few seconds of setting this up.
- **CI's native stacks put the faulting thread first.** The Windows jobs' crash-dump step prints each dump's exception
  and faulting thread in full, then every thread summarised (counts by module, each distinct stack once), then
  `!analyze -v`, from separate debugger runs: a worker's hundreds of threads had pushed the faulting one out of the log.
- **The assistant's server answers a wrong address every time.** It sent its "404 Not Found" before reading the
  request's body and then closed the connection, which reset it when the body had just arrived; on Windows the reset
  could lose the answer, and the client saw "connection aborted" (WinError 10053) in a few requests in a hundred.
  The body is now read first, and the answer is the same 404 as before.
- **The JDemetra+ reference job installs libcurl's headers before pak**, which it has to build from source when
  CRAN has a newer pak than Posit's binaries for the runner's Ubuntu (pak 0.11.2 broke it on 8 November 2026); the job
  also names its runner image (ubuntu-26.04) and R (4.6.2).

## 0.1.0a14: development build (6 November 2026)

### Estimation & tests

- **Regression without a formula: `pymetrix.ols(y, X, cov=..., constant=True, weights=None, clusters=None,
  hac_lags=None, hac_correction=True, missing="drop", names=None)`.** For code that runs many small regressions
  (loops, simulations, model building): `y` a Series or 1-d array, `X` a DataFrame, Series, mapping of named
  columns or 2-d array (named by `names=`, else `x1`, `x2`...). It returns the `FittedEquation` that
  `Equation("y ~ x1 + x2").fit(data, method=OLS(cov=...))` gives on the same data: the same least-squares code, so the
  same coefficients, standard errors (classic, White, HAC with its lags and correction, one- and two-way clusters by
  columns of `clusters=` or index levels), statistics, residuals, views, tests, diagnostics and forecasts, and the same
  provenance (fingerprint of the data used, rows used and why others were not, method, backend), except
  `vintage_request = {"policy": "direct"}`: the data were passed directly. A `PeriodIndex` (or dates with a frequency)
  makes the data a time series, as in a dataset. In a session the call is recorded as Python code; the arrays are not
  copied into the project. 10,000 regressions of 40 observations on 3 to 6 regressors: 6.1 ms each by formula and
  `Dataset`, 1.0 ms by `ols` on pandas (6x), 0.8 ms on numpy arrays (8x); `benchmarks/ols_arrays.py`.
- **Breusch-Godfrey without a constant: `auxiliary_constant=`.** pymetrix's auxiliary regression has exactly the
  equation's regressors (Breusch 1978; Godfrey 1978; Greene), with zeros for the missing initial lags and R-squared
  about the residuals' mean; statsmodels' `acorr_breusch_godfrey` always adds a constant, so for an equation without
  one the two differed. `BreuschGodfrey(lags, auxiliary_constant=True)` (Run test → Test settings, **Add a constant
  to the auxiliary regression**) adds it, matching statsmodels; the default, and every result with a constant in
  the equation, is unchanged. The test's help says which convention it follows.
- **The diffuse log likelihood's constant terms, documented, and statsmodels' convention as an option.** With diffuse
  (non-stationary) states, pymetrix's log likelihood is the exact diffuse likelihood of Durbin and Koopman (2012,
  section 7.2.2): the first `d` (diffuse) periods contribute `-1/2 (log 2 pi + log F_inf)` per observation, and every
  observation counts `log 2 pi`. statsmodels' `UnobservedComponents` and `SARIMAX` report by default the likelihood of
  the later observations given the first `d` (Harvey's; their approximately diffuse start drops the first `d`
  periods), so their log likelihood is higher by `d/2 log 2 pi + 1/2 sum log F_inf`, the same at every parameter value
  where the diffuse part of the system has no parameters: 7.367 for a local linear trend with a quarterly seasonal
  (`d = 5`, `5/2 log 2 pi + log 16`), 1.838 for a local linear trend, 0.919 for a local level. Estimates and standard
  errors are the same either way; the log likelihood, AIC, BIC and HQ move by the constant (twice it). New
  `diffuse_terms="include" | "exclude"` (default `"include"`, today's results) on `StateSpace.fit`, `.loglike`,
  `.filter`, `UnobservedComponents.fit`, `StateSpaceSystem.loglike` / `loglike_observed`, `statespace.loglike` and
  `diffuse_loglike`, and `Model.estimate` (Kalman filter only); `FittedStateSpace.criteria(diffuse_terms=)` gives the
  log likelihood, AIC, BIC and Hannan-Quinn in either convention from any fit; a non-default choice is recorded in
  `provenance.method["diffuse_terms"]`. Dialogs: **Diffuse periods in the likelihood** under Advanced in Unobserved
  components, State space model and Estimate model. Checked against statsmodels (exact diffuse filter with its
  burn-in) to 1e-8 on a local level, a local linear trend, a seasonal UC model and an ARIMA(1,1,1), and the offset
  against `d/2 log 2 pi + 1/2 log det(X X')` (rows of `X`: `Z T^(t-1)`) for `d` = 1, 2, 3 and 5. Help topic
  "State space and the Kalman filter"; `docs/guide.md` (State space) and `docs/validation.md`.

### Data & dialogs

- **Temporal disaggregation, extended.** `TemporalDisaggregation(...).fit(low, high, aggregation=C)` takes any
  aggregation matrix in place of the conversion, so calendar- and fiscal-year totals (or quarters) can be mixed and
  can overlap; C·y = Y holds exactly. `aggregation_matrix(high, periods)` builds one from periods of any frequency
  (`pd.Period("2020", "Y-JUN")`); a Matrix, a DataFrame, a high-frequency dataset of weights (one series per value),
  a CSV, Excel or Parquet file or an array also do. Rows with no low-frequency value are dropped; the rest are
  checked (independent, within the indicators' sample). **Soft constraints** (`soft=` a standard deviation, or
  `{period: sd}` with the rest exact): the regression methods treat it as measurement error on the aggregates
  (GLS, its scale by maximum likelihood), Denton as a penalty; as it goes to zero the result is the exact one, as it
  grows the indicator path. **Litterman** (`litterman`, `litterman-minrss`: a random walk with AR(1) increments;
  rho = 0 is Fernández). **Denton in seasonal differences** (`differences="seasonal"`, lag 4 or 12, with
  `level_penalty=`, by default 0.001, on first differences) keeps the indicator's seasonal pattern. Chow-Lin's and
  Litterman's rho is searched in [0, 0.999] by default, as in R's tempdisagg; `rho_bounds=(-0.999, 0.999)` allows
  negative values, and an estimate at an end of the range is now noted in the summary and the provenance. Defaults
  are unchanged, and so are the results they give. The Temporal disaggregation dialog has the new methods and
  options (aggregation matrix from a workspace matrix or dataset, or a file; soft constraints; seasonal
  differences).
- **Two-way reconciliation.** `reconcile(data, components, total, low=None, fixed=(), method="denton")` moves
  components (disaggregated series, say) as little as possible so that they add up to their high-frequency total
  every period while each keeps its own low-frequency totals; fixed components are not moved. Methods: Di Fonzo and
  Marini's multivariate Denton, proportional (`denton`) or additive, and least-squares balancing in proportion to
  the values; `weights=` sets how much each component moves. The constraints hold to rounding error, contradictory
  ones are reported, and `Reconciliation.diagnostics()` gives each component's adjustments and gaps before and
  after. **Data → Transform → Reconcile components…**.

### Models & DSGE

- **A loss to rank a scenario set's scenarios.** `ScenarioSet.with_loss(loss, desired=None, discount=1.0)` sets a
  quadratic loss as optimal control's: the sum over the periods solved and the variables weighted of discount^t ×
  weight × (value − desired)². `losses()` gives each scenario's loss, split by variable, and less the baseline's
  (negative: better than the baseline); the scenarios' summary lists it. The window has a **Loss** tab and the
  `policy_brief` report template includes it. In the app: **Loss**, **Desired values** and **Discount** in Compare
  scenarios, recorded as `scenarios(...).with_loss(...)` (or `compare_solutions(...).with_loss(...)`).
- **Sequential Monte Carlo: `target` is now `ess_fraction`.** `FittedModel.smc(ess_fraction=0.5)` and
  `pymetrix.models.bayes.smc(..., ess_fraction=)`: the share of the particles that the weights' conditional
  effective sample size is kept at in each tempering step (higher means more, smaller steps). Same default and
  meaning; no argument is named `target`. The Model → Sequential Monte Carlo… dialog's advanced field is now
  **Effective sample fraction** (`RunSMC.ess_fraction`, schema 2): projects and scripts saved with `target` open
  and replay unchanged.
- **Deprecated:** `smc(target=...)` still works for one release, with a DeprecationWarning saying to use
  `ess_fraction`; giving both raises a SpecificationError.

### App & figures

- **The window in front is obvious.** Its title bar takes the theme's accent colour (the palette's Highlight, the
  title in HighlightedText) with a thin frame of the same colour; the other windows' titles are muted. It follows
  whichever window comes to the front, on the Fusion, Windows, windowsvista and windows11 styles and on light and
  dark themes alike: the colours are the palette's (the bar darkened a little where the palette's own pair would read
  at less than 4.5:1). The panel with the keyboard focus gets a thin accent line along its top.
- **One set of colours.** Errors, warnings, successes, links and secondary text take named colours from one place
  (`pymetrix.ui.theme`), each with a light-theme and a dark-theme value, rather than a dozen slightly different reds,
  greens and blues written where they were used. Each reads at 4.5:1 or more on the window and list backgrounds (the
  old error red, link blue and warning amber did not on a light grey window), and secondary text is a grey derived to
  read at 4.5:1 too: the palette's placeholder text it used, half-opaque black on Qt's default palette, was 3.9:1.
- **Dialogs show the essentials; expert options fold into Advanced settings.** Each command dialog shows the data
  or object it acts on, the formula, variables or series, the method, the sample and the result's name; settings most
  steps leave alone sit in the collapsed **▸ Advanced settings** section: VAR's lag search limit and Bayesian prior
  variants, draw counts and burn-in (impulse responses, historical decomposition, TVP-VAR, BVAR-SV, forecasts,
  bootstrap tests), seeds, double machine learning's folds, repeats and treatment learner, the breaks' trimming,
  long-memory bandwidths, Plot series' shading, events, reference lines and value labels, axis splits and annotation
  geometry, report themes, and plugins' tuning (vine copula selection, Black-Litterman's tau and delta). No field,
  default or argument changed, and a field without a default is never folded away (a test checks every command).
  The section opens by itself when a reopened step changed one of its settings or one is not valid, and its header
  counts the changes (**Advanced settings (2 changed)**); it has its own access key, a hidden field's key opens it
  and goes to the field, and whether it was left open is remembered for each dialog (closed at first).
- **Show the code, in every dialog.** Under each dialog's fields, **Show the code** gives the Python the step will
  record, following the fields as they change, with Copy; folded by default, and kept open in every dialog once an
  expert opens it. The figure's Format… form shows its change as matplotlib's keywords, which can be typed over: what
  is typed is what OK runs. In the figure's right-click menu, **All its properties…** is now **Edit as code (every
  property)…**.
  Open, the code takes its height from the form's share of the screen, so a dialog still opens within a 1024 by
  600 screen with the Advanced section open too; folded, the form has its share back.
- **Summary statistics grouped, not listed.** Under a result's table, the "Label: value" lines (Method, Sample,
  R-squared, S.E. of regression, Log likelihood, the information criteria, Durbin-Watson, a test's distribution...)
  are grouped under small headings: Estimation, Fit, Information criteria, Residual checks, and Other for a label
  not recognised. Labels are muted on the left, values on the right, numbers right-aligned so their decimals line
  up. A wide window puts the groups side by side, two or three across; a narrow one stacks them. Notes that are
  sentences (warnings, explanations, a test's null hypothesis) follow as text; none is left out. The coefficient
  table fits its columns, with more room in each cell, a bold header and every other row shaded. The same layout
  is in a notebook (`_repr_html_` of a fit, a test and every other result shown through a table) and in an HTML
  report. The plain text (`print(fit)`, Copy summary), CSV and Markdown are unchanged. The grouping of labels is
  one table in `pymetrix.wording` (`STAT_GROUP_PREFIXES`, `stat_group`).
- **One statistic a line in more summaries.** Results that wrote their statistics in one line ("Log likelihood
  -280.58; AIC 565.17; BIC 570.38") now write one "Label: value" line each, with a fitted equation's names (Log
  likelihood, Akaike info criterion, Schwarz criterion, S.E. of regression...), so they are grouped too: ARIMA,
  exponential smoothing, ARDL, MIDAS, Markov switching, Markov-switching VAR, systems of equations, VAR (and a
  FAVAR's VAR: its criteria one a line, its largest root), VARMA, VECM, structural VAR estimates, DSGE estimation,
  the OccBin filter, count series models, spatial models, conditional logit, dynamic factor models, pooled mean
  group, and in Finance the copulas (constant, time-varying, DCC), vines, multivariate GARCH and dynamic
  Nelson-Siegel. Their printed summaries and Copy summary change accordingly; the values are the same.
- **A test's values right-aligned.** The value column of a test's table (statistic, df, p-value) is right-aligned
  in its window and in HTML, as a coefficient table's numbers are, also where the df is written as text.
- **Coming from EViews, MATLAB/IRIS or Dynare.** Two help pages, **Help → Coming from EViews** and **Help → Coming
  from MATLAB/IRIS** (also linked at the top of the help contents), map the familiar objects to pymetrix's
  (workfile and page to a project and a Dataset, equation object to FittedEquation, model object to Model, spool to
  Report; IRIS's `model`, `sstate`, `solve`, `simulate`, `filter`, `estimate` and databanks; Dynare's blocks and
  commands), with a table of commands and menu items: where each is in the app and the Python call. They link the
  defaults that differ, say what is different by design (recorded steps, scripts, Undo, projects) and how to bring
  models across (`read_eviews`, `read_iris`, `read_mod`, the model editor's Write as). **Find anything** (Ctrl+K)
  knows the familiar names from the same tables (`pymetrix.coming_from`): typing `genr`, `smpl`, `ls`, `wfopen`,
  `freeze`, `spool`, `sstate`, `dbload`, `stoch_simul` or `estimated_params` offers the command, as "Generate
  series… (EViews: genr)", after whatever pymetrix's own names match. The X-13 and TRAMO-SEATS rows name the
  seasonal plugin (`pymetrix_seasonal.X13()`, `pymetrix_seasonal.TramoSeats()`) and point to the core's Seasonal
  adjustment dialog; EViews' `seas` is classical decomposition, which needs no plugin.
- **One house look for every chart.** Every pymetrix figure is drawn with one set of matplotlib settings
  (`pymetrix.figure_style`), in the app, scripts, notebooks, reports and exported files alike: the default theme's
  Okabe-Ito colours in a fixed order (safe with the common colour-vision deficiencies), a light grey grid of major
  lines behind the data (on bar charts across the values only), no top or right spines, short ticks pointing
  outwards, and the same sizes for titles, labels, ticks and legends. Type is DejaVu Sans, which matplotlib always
  ships, so a chart looks the same on every machine; in the app it is the app's own font when matplotlib has it
  (the family Qt really draws in, else the one the app asks for), and DejaVu Sans when it has neither: an alias
  ("Sans Serif"), or nothing at all where Qt finds no fonts, is never handed to matplotlib to log "font not found".
  The settings apply to pymetrix's figures only: your own matplotlib plots and `rcParams` are untouched. A theme's
  style sheet still has the last word, and `fig.matplotlib_code()` writes the same settings into its script.
- **Charts follow a dark theme, and are sharp on high-resolution screens.** Under a dark palette a chart is drawn on
  the window's own colour with light text and a subdued grid; its dark greys (an "actual" line, reference lines,
  notes) turn light so they still read, and colours stay as they are. Copied and saved images stay light. The
  canvas draws at the screen's device pixel ratio from the start (and zoom keeps it), with the figure's size in
  inches unchanged, so saved projects and layouts do not shift.
- **Fan chart legends shade as the chart does.** A fan's bands overlap, so the inner ones are darker on the chart;
  their legend keys now show that shading (90%, 60%, 30% from light to dark) rather than three equal swatches.
- **Format… and a right-click menu on figures.** A right-click on a figure (a two-finger click or Ctrl+click on a
  Mac) opens a menu for what is under the mouse. On an arrow, note, text box, line, band or change marker: **Format…**,
  **Edit the words…**, **Bring to front**, **Send to back**, **Edit as code (every property)…** and **Delete**. On a series:
  **Format series…**, Figure style filled in with the series. Anywhere in a panel: **Format axes…**, **Draw an
  arrow**, **Add a note here…**. And always the figure's titles and shading, Copy image and Save. **Format…** is a
  form in words and lists, not keywords: an arrow's colour (with a colour picker), line width, head, head size and
  curve; a text box's box shape, fill and edge; a line's style; a band's opacity; a change marker's shape and label.
  A double-click on an item opens it too. Each change is a Change a style step, so Undo, replay, reports and the
  figure's matplotlib code follow (`pymetrix.ui.chart_format`).
- **Right-click menus in Problems, the Report panel, Code windows, the Script panel, the Lab book and on windows.**
  Problems: for a step, Show the step in History, Open its dialog filled in, Copy its code; for an out-of-date
  result, Open and Recalculate; for a plugin, Plugins…; and Copy the problem (a double-click does the first). The
  Report panel: edit, open the result, move (up, down, to the top or the end), add a heading or paragraph after,
  remove. A Code window: Run this line or the selection, Run the whole script, Comment, Find and **Help on "name"**,
  the documentation of the name under the mouse. The Script panel: Copy the whole script, Open in a Code window. The
  Lab book's entries: a note on that entry (recording its number), or the entry copied. A window's title bar, or its
  tab: Copy the window as an image, Add to the report, Close the other windows, Tile, Cascade.
- **Right-click menus in the spreadsheet and on result tables.** In the data window, a right-click on the cells
  offers Copy, Paste, **Fill down** and **Clear (make missing)**, each one change for Undo and pending until Apply,
  then the selected series' **Plot…**, **Describe…**, **New series from it…**, **Keep as a new dataset…**,
  **Rename…**, **Remove**, **Freeze** and **Series info**; on a series' heading, the same and **Copy the column**.
  A right-click on a result's table adds **Sort by** the column under the mouse (the rows shown only; **Back to the
  table's order** undoes it), **Show as a heatmap** and **Add to report** to its copy options.

### Seasonal adjustment (plugin)

- **X-13ARIMA-SEATS and TRAMO-SEATS move to a plugin, `pymetrix-seasonal`** (`packages/pymetrix-seasonal`; the first
  split in `docs/design/plugin-split.md`). They need programs outside Python (x13as; Java and JDemetra+), so the core
  no longer carries that install risk. STL and classical decomposition stay in the core.
  - **Installed:** nothing changes for users. The methods appear in the Seasonal adjustment dialog as before, and
    `from pymetrix import X13`, `from pymetrix.seasonal import X13` and `pymetrix.jdemetra.install()` still work.
    Recorded scripts import them from the plugin.
  - **Not installed:** a project that used them opens with that step kept, not run, and the plugin named.
  - **Install:** from the repository, `pip install -e "packages/pymetrix-seasonal[jdemetra]"`. `pymetrix[jdemetra]`
    now means the plugin's extra, and JPype is no longer in `[all]`.
  - **For other plugins:** a seasonal method can be given the series' periods (`needs_periods`; Extension API,
    below).

### Assistant (plugin)

- **What an assistant may be told, in one tested place** (`pymetrix.disclosure`; stage 2 of the assistant design,
  its core part). `describe_session(session)` describes a project in `metadata_only` terms: each object's name, kind,
  series and variable names, counts, sizes, period labels and settings that are identifiers, whole numbers or
  true/false, and the latest steps by dialog and what they made. It is an allow-list: data values, numbers with
  decimals, formulas, file paths, labels and other free text, report text and the code of steps never appear, and a
  name that is not an identifier is counted, not quoted. `MODES` lists the confidentiality modes (`off`, the
  default; `metadata_only`; `full_context`; `local_model`). A test fills a project with marked values, a formula,
  a path, labels, report text and typed code and checks none reaches the description.
- **Ask Claude in the app** (`pymetrix-assistant`, the Assistant panel's new **Ask** tab; stage 2, step 2). A
  question goes to Anthropic's API with the project in metadata only. The answer comes with proposals: **Open…**
  opens the step's dialog filled in, for you to run with OK, and **Skip** drops it. The model runs nothing and sees no
  result, and it never proposes Python.
  - What is sent is the project's choice, off until chosen: the panel asks the first time, and **What is sent…**
    shows the exact text.
  - The API key is kept in the system keychain (**API key…**).
  - The Allow connections controls of stage 1 are on the **Connect** tab.
- **What the assistant proposed stays visible** (stage 2, step 3).
  - A step an assistant proposed, and you ran from its dialog, says "proposed" in History. A comment above it in
    the script names who proposed it, and the lab book records it. The mark is saved with the project.
  - Extension API 1.9 carries the mark (`Session.proposing`, below).
  - The Ask tab keeps a record in the project: each question, the exact text sent, the answer, and what became of
    each proposal (run as step N and which settings you changed first, closed, or skipped). **Record…** shows it.
- **Ask a model on your own computer** (the `local_model` mode): Ollama, LM Studio or another OpenAI-compatible
  server, for those without an Anthropic account. Nothing leaves the machine: only a localhost address is accepted,
  and requests bypass proxies and refuse redirects. It gets the same metadata, tools, proposals, record and History
  mark as Claude. **Find models** lists the server's models.
- **Full context, to Anthropic** (the `full_context` mode). With each question the assistant is also sent the code
  of the project's steps, and it can ask to see one of an object's tables (estimates, tests, data). Each send is
  shown to you first and goes only if you press **Send**; the record keeps what was sent.
  `pymetrix.disclosure.describe_session` takes the mode.
- **Course mode for the assistant.** A course note can say `assistant: explain`. While that course is loaded, the
  assistant explains methods and results but proposes no steps, and an outside assistant cannot connect.
  `assistant: off` turns the assistant off. The rule is `pymetrix.help.Course.assistant`, and the panel names the
  course that set it.

### Extension API

- **Extension API 1.9** (additive): `Session.proposing(CommandClass, by)` and `CommandRecord.proposed_by`, which
  mark a step an assistant proposed. The Assistant plugin uses it, and now requires pymetrix 0.1.0a14.
- **Seasonal methods given the series' periods.** A seasonal method class can set `needs_periods = True` to be
  given the series' periods and a dict for its provenance notes. X-13 and TRAMO-SEATS in `pymetrix-seasonal` use it,
  so that plugin requires pymetrix 0.1.0a14.

### Fixes

- **Narrow windows with a large font (Windows): words break, choices wrap.** A dialog's radio choices (Method,
  Standard errors, What switches, Objective, ...) go onto more lines where the form is narrower than them side by
  side, so a large font no longer makes System of equations, Cointegrating regression, Markov-switching VAR, VECM
  restrictions or Optimize portfolio scroll sideways on a laptop screen; where there is room they stay on one line.
  A field too wide to sit beside the labels goes under its own label rather than holding the form wider than the
  window (Reconcile's method, VECM's deterministic terms with the Advanced section open).
  In the Start window, a word wider than the room (in a demo's description, the title, the gallery line) breaks
  where it must, so the window keeps to a narrow workspace without cutting text off.
- **Fixed, on Python 3.11 the desktop app could end with "Fatal Python error: none_dealloc: deallocating None"** after
  a while in use. PySide6 6.12.0 drops a reference to `None` on every Qt call that returns nothing (a setter,
  `update()`), harmless where `None` is immortal (Python 3.12 and later) but fatal on 3.11 once enough calls have run.
  The `qt` and `all` extras now leave PySide6 6.12.0 out on Python 3.11 (6.11.3 is the newest release without the
  fault there); later Pythons keep it. A test checks that the installed binding keeps `None`'s reference count.
- **Fixed, a crash with no message after TRAMO-SEATS had run (Linux).** Java, started from the main thread, took
  that thread's stack to be its default thread stack size, 1 MB rather than the usual 8, and put its guard pages
  there, so later work in the same session (the app's windows, a script) could overflow the stack and end the
  process without a word. Java now starts with a thread stack size as large as the main thread's (`ulimit -s`).
- **Tests: a test process that crashes says why.** Each pytest-xdist worker keeps a crash report (its last steps,
  every thread's stack on a fatal signal, and what it wrote to stderr last, Python's own fatal-error message among it),
  and CI prints the reports, with the worker's exit status, Java's error file and the kernel's crash lines, when a job
  fails.

## 0.1.0a13: development build (5 November 2026)

### Models & DSGE

- **A scenario set: several named scenarios against one baseline, compared side by side and reported together.**
  `ScenarioSet` has a `baseline`: the one given (`scenarios(..., baseline="tight")`, the Baseline field of
  **Model → Compare scenarios…**), else one named "baseline", else the first; `with_baseline(name)` compares the same
  scenarios with another. `deviations(variable, percent=False)` gives each scenario less the baseline;
  `impact(variables, at="average" | "end" | "peak" | a period)` a scenario per row and a variable per column, how far
  each departs from the baseline; `plot(variables, deviations=True)` the departures, the baseline drawn first in the
  paths. Its window adds the **Impact** table, a **Less the baseline** graph and every scenario less the baseline (it
  showed the second less the first only). A new report template, `policy_brief`, puts the scenarios, the Impact
  table and both graphs in a report. `difference(scenario)` now defaults to the baseline.

### App & figures

- **Split axes.** `ybreak=(low, high)` (or `"6:12"`) splits a figure's value axis, leaving out the values
  between: each panel they fall inside is drawn in two parts, one above the other, each as tall as its range, with
  break marks, so a spike no longer squashes the rest. `xbreak=(first, last)` (or `"1986Q1:2004Q4"`) splits the time
  axis, leaving out those periods: two spans side by side on one value scale. Panels the break does not reach are
  drawn whole. Recorded styles reach both parts, the title and legend go on the first; saved results keep the split
  and `matplotlib_code()` writes it. In the app: **Split values** and **Split dates** in Plot series' Advanced
  settings and in Edit… → Titles, shading and marks.
- **Arrows drawn on a figure as in a spreadsheet's chart.** An **Arrow** button on the figure's toolbar: press
  where the arrow starts and drag to where it points. The arrow is drawn as you drag, added at once when you let go
  (an Add a note step, no dialog; two clicks still work) and left selected, its ends ready to drag. A double-click on
  an arrow, line or band opens Change a style on it.
- **Formulas are checked as you type.** Formula fields (Generate series, Derived series, Estimate → Equation and the
  other estimators that take one, Define equation, the templates of Equation for each selected series, and Rows where)
  were checked only when OK was pressed. A moment (300 ms) after typing stops, the formula is parsed as the step will
  read it (nothing is computed): a mistake shows in red under the field with the parser's reason and where ("Expected
  ')' at character 16, after “log(consumption”"), and OK waits until it is fixed; a name the dataset does not have is a
  grey hint with the nearest name ("Not in the data: incme (did you mean income?)"), which does not hold OK
  (`pymetrix.ui.formula_check`).
- **Open demo project lists the demos.** File → Open demo project… was a dropdown of cut-off lines. It is now a small
  dialog listing each demo with what it shows and which windows it opens, in the Start window's list: Up and Down
  move, Enter or a click opens, Esc cancels (`pymetrix.ui.demo_chooser`). `pymetrix-gui --demo NAME` and Reload demo
  are unchanged.
- **No dialog scrolls sideways at 1366 by 768.** System of equations and Equation for each selected series (and, with
  a larger font, seven more) opened with a sideways scroll bar one scroll bar wide: their forms open shorter than their
  content, and the vertical scroll bar took its width from the fields. A form's least width now leaves room for its
  vertical scroll bar whenever that shows (checked for every dialog with this machine's font, a wider one and
  Windows' style).
- **Access keys on dialog fields.** Each field label of a dialog has an access key (Alt and the underlined letter goes
  to the field), from the label's own words, unique within the dialog and none of OK's, Cancel's or Help's
  (`pymetrix.forms.access_keys`); each label's buddy is the field's input itself (the text box, a file's path, the
  chosen radio button).
- **The Model menu follows the modelling workflow.** Its 23 top-level lines become 12: write the model (Model editor,
  Define equation, Template model, Import, Export model), Check model, then one submenu per stage: **Solution** (Solve
  model, Add factors; Rational expectations, Second-order solution; Occasionally binding constraints, OccBin filter),
  **Estimation** (Calibrate model; Estimate model, Metropolis-Hastings, Sequential Monte Carlo; Estimate DSGE-VAR),
  **Analysis** (Identification; State-space form, Kalman filter, Data contributions, Shock decomposition; Forecast from
  estimated model; the former State space submenu), then Policy, Scenarios and Matrix. The submenus are named by nouns, so
  "Model → Estimation → Estimate model…" never reads like the Estimate menu. Estimate → VARs and local projections
  now lists the estimators first (TVP-VAR next to the other VARs), then VECM and cointegrating regression, local
  projections, then identification and what it shows (Structural VAR, Proxy SVAR, Impulse responses, Variance and
  Historical decomposition), and the lag-order searches last. Command ids are unchanged; the docs' paths follow.
- **Every menu line has an access key.** Each line of every menu (core and plugins', and the lines a menu adds as it
  opens: recent projects, open windows, Undo's step) has an underlined letter unique within its menu (Alt+M, S, R:
  Model → Solution → Rational expectations…). Keys are given from the label: the first free first letter of a
  significant word, then any free letter, rearranging earlier lines when that frees one; a label that marks its own
  (E&xit) keeps it (`pymetrix.commands.menus.with_access_keys`). History's and the dialogs' menu paths now name the
  submenu too ("Model → Solution → Rational expectations…").
- **A demo opens with its results in view.** Opening a demo project (File → Open demo project…, the Start window, a
  Guide case study) left the workspace empty. It now opens the demo's headline results, each on the view that shows
  the work (forecast_round: the round's fan chart, the VAR behind it; consumption: the dynamic consumption function;
  nk_estimation: the posterior; policy_and_nonlinear: the zero-lower-bound episode), selects it in Objects, and says in
  the status bar what is open and where to look next (`pymetrix.demos.headline(name)`).
- **The Start window puts the demos first.** At 125% scaling on a 1366×768 laptop the demo list started below the
  fold. The Start window now opens with one line of introduction, the four ways to start two by two, and the demo
  projects; the tour of the workspaces and scripts is folded under **About this window** (the chart gallery link stays under the buttons).
- **The Guide's buttons stay in view.** Open…, Show me, Back, Next and Start over sit under the part of the Guide that
  scrolls (the step's text and the list of steps), so a short dock no longer scrolls them away; a long case-study
  title shortens rather than pushing the reload button off the edge.
- **OK waits for the fields a step needs.** 165 of 166 dialogs that open with a required field empty let OK run the
  step anyway: it failed, and a broken line of Python (`forecast = .forecast(data, span=('', ''))`) went into History
  and Problems. OK is now disabled until every field marked * has a value, and a line above the buttons (and OK's
  tooltip) names what is missing ("Choose Fit; fill in Start and End to run this step."). The * marks exactly those
  fields (`pymetrix.forms.needs_value`; a field whose empty value means something, as Solve's Start and End, is not
  marked). A step whose code is not Python is refused before it runs and never recorded. A start or end period's
  default (the sample's first or last period) reads in the text's colour, as "2000Q1 (first period)", not in the grey
  that looked disabled.
- **A picker with nothing to pick says so.** An object field with no object of its kinds showed internal class names,
  "(no FittedEquation,FittedVAR,… available)", which opened Forecast 1,278 px wide. It now says what is missing in
  words, under the field ("No estimated models yet: make one first (Estimate → Equation… or Estimate → VAR…)"), with a
  button that opens the dialog that makes one, and the dialog keeps a normal width.
- **Template models solve from Model → Rational expectations….** After Model → Template model…, Solve and Rational
  expectations showed an empty Equations list. Rational expectations now takes a template or Dynare model (recorded as
  `re = nk.solve()`) and picks the newest one; Solve says that a template is solved with Rational expectations, with a
  button to it.
- **Failed steps in plain words.** Every error box was titled "Error" and showed the library's text ("[Errno 21] Is a
  directory: '.'", "zero-size array to reduction operation maximum which has no identity"). A failed step's box is now
  titled after its dialog ("Forecast: could not run"), says what went wrong in plain words with a next step (a folder
  for a file, no observations in the sample, series that do not cover the sample, nothing left after missing values,
  a division by zero), and keeps the technical text under **Show Details…**; pymetrix's own messages are kept as they
  are (`pymetrix.ui.messages.explain`). The box opens over the dialog, which stays as it was, to correct.
- **Fan charts start at the last actual, with the legend clear of the data.** The forecast round's graph (Lab 3's
  headline figure) had its legend over the data and the dates, and a gap between the history and the fan. Fan charts
  (`ForecastRound.plot`, `fan_figure`, a stochastic simulation's fan) and forecast intervals (`forecast_figure`: a
  forecast's or a BVAR's predictive Graph) now start at the last actual value, the bands opening from it, and show one
  legend in a row above the first panel. A panel's new `legend_loc` places its legend: `""` the best place inside
  (as before), `"above"`, or any matplotlib `loc`; it is saved with the figure and in its matplotlib code.
- **Result windows list their views down the side.** A fitted equation's 13 tabs and a VAR's 9 ran behind scroll
  arrows, even maximised at 1366×768; at 125% scaling only three or four showed. A result window now lists its views
  down its left side (`pymetrix.ui.view_switcher.ViewSwitcher`), every one in view and named in full, grouped under
  Summary, Tables, Graphs, Diagnostics and Provenance when there are six or more; Up and Down (and Ctrl+PgDown,
  Ctrl+PgUp) move between them; the view picked last is where the next window of that kind of result opens, and a
  window keeps its view when its result is recalculated. The switcher takes `QTabWidget`'s calls (`count`, `tabText`,
  `widget`, `setCurrentIndex`...), plus `titles()`, `select(title)` and `current_title()`.
- **Guided case studies, as a plugin** (`pymetrix-guide`, in `packages/pymetrix-guide`; first version, experimental;
  `pip install pymetrix[guide]`). A case study is a Markdown note in a course folder with fenced `pymetrix-step`
  blocks: a dialog, its settings and a declarative check (an object exists or has a type, a series is in a dataset,
  a dialog's step ran), with optional `title`, `summary`, `demo` and `level` front matter. `##` sections before the
  first step make an introduction, shown as pages before step 1; each step can name its `stage` of the workflow and
  say `why` it is done and what to `look` for in its result. The **Guide** panel (View → Guide) shows the
  introduction, then one step at a time with a strip of the workflow's stages ("Stage: Judgement — step 8 of 19");
  **Open…** fills in the step's dialog, **Show me** runs it, and each step ticks itself when its check passes,
  following the history (Undo takes the tick back, Redo gives it again). The steps are ordinary recorded steps, so
  the script, the project and the lab book keep them. Lab 3, the forecast round, ships as the first case study: an
  overview of the forecast workflow, the scenario it illustrates, and 19 steps in seven stages. `check_case_study`
  reports bad blocks with their line, and warns about a case study with no introduction or a step with no text.
  [Writing a case study](docs/case-studies.md).
- **The Start window's chart gallery line is a link.** It named a repository path
  (`examples/gallery/chart_gallery.ipynb`) that meant nothing in an installed app and could not be clicked. Run
  from a checkout, it is now a **Chart gallery** link that opens the notebook in the system's default application;
  an installed copy, which does not ship the repository's examples, leaves the line out. It sits under the
  buttons, so it shows at any window size (at the foot of the window it was cut off when the window was short).
- **Dialogs speak in words.** 1,284 of the 1,493 field labels in the menu dialogs were Python argument names ("Var",
  "Exog", "Dist", "Reps", "Rf", "Mce", "Low", "High"). One display-name table (`pymetrix.wording`) now labels every
  field of every registered dialog, the core's, the finance plugin's and the settings sub-forms of every test,
  method, filter and variance model: 614 argument names, with 352 per-command overrides where one name means
  different things ("Low-frequency data" in MIDAS, "Horizon" for an impulse response's `periods`). Labels are in
  sentence case, with no colons and units where they help (Expiry (years), Width (inches)). Each field's tooltip,
  and its label's, ends with the Python argument it sets (*Python: exog=*). A plugin's field the table does not know
  reads as its name's words.
- **Choices show names, scripts keep codes.** Method `fmols / ccr / dols` reads "FM-OLS (fully modified)", "CCR
  (canonical)", "DOLS (dynamic)"; Missing `drop / raise` reads "Drop rows with missing
  values" / "Stop with an error"; copula families, GARCH errors, MGARCH models, learners and dozens more read as
  names, in sentence case, each with its code in its tooltip. The step, the history and the script record the code
  as before (`method='dols'`). A series' adjustment shows as "Not seasonally adjusted" (not `nsa`) in the
  spreadsheet's description panel and the dataset's data dictionary.
- **One Advanced settings section per dialog.** Dialogs had an Advanced tab and, on Main, a sub-form's own
  "▸ Advanced settings" expander (Equation's method settings, Run test's test settings). Every dialog now has one
  collapsed **▸ Advanced settings** section at the bottom of its form, shown only while something in it applies; it
  holds the command's advanced fields and the chosen method's or test's, under the sub-form's label. Reopening a step
  that changed an advanced setting opens it. The Jupyter forms follow suit. The fields, and what steps record, are
  unchanged.
- **Dialog titles say what the dialog does.** Titles come from the menu item, less its "…": Model → Import → FRB/US
  XML file… was titled "US XML file" (the title was cut at the last "/" of the menu path; only the first "/" now
  separates a menu from its label, in titles, history, workspace buttons and help). The Model menu's **Check
  model…**, **Solve model…** and **Calibrate model…** (were Check…, Solve…, Calibrate…) and Data → **Derived series
  for each selected series…**; one-word titles that said too little now say more (Estimate an equation, Spectral
  analysis, Identification check, Compute returns, Forecast backtest).
- **Terms used in pymetrix.** The app's words follow one glossary (series for a column of data, variable for a
  model's, sample, Start and End, estimate, estimated model, horizon, draws and replications, British spelling):
  the in-app help topic *Terms used in pymetrix* and [the guide's section](docs/guide.md#terms-used-in-pymetrix).
  Calculate's mode items read **Recalculate automatically** and **Recalculate on request (F9)**, and the Start
  window's **Load example data…** has the ellipsis of a button that opens a dialog.
- **Menus where they are looked for.** **Forecast…** (a forecast from an estimated model) moved from Estimate to the
  top of the **Forecast** menu, which now lists forecasts, then rounds, then their evaluation; **Gregory-Hansen
  cointegration test…** moved from Estimate → VARs and local projections to **Test**, beside Johansen; **Random
  generator** moved from View to **Calculate**. The finance plugin's 28 items, one list that ran off a laptop's
  screen, are grouped under **Risk**, **Copulas**, **Portfolios**, **Options**, **Fixed income** and **Credit**
  (Returns… and Performance… stay at the top). Command ids, steps and scripts are unchanged.
- **File → Exit (Ctrl+Q)**, Quit on macOS, last in the File menu; it asks to save unsaved work as closing the window
  does. The menu bar has access keys (Alt+F for File, Alt+S for Estimate, Alt+O for Forecast...; a plugin's menu takes
  a free letter, Alt+I for Finance).
- **Keyboard shortcuts lists Redo's two keys.** Redo answered, and the list showed, the platform's every binding
  ("Ctrl+Y, Alt+Shift+Backspace, Ctrl+Shift+Z, Redo"), which pushed the Command column out of view; it is Ctrl+Y and
  Ctrl+Shift+Z, and Command takes the width. The core workspaces keep **Ctrl+1 to Ctrl+6** whatever plugins are
  installed (Finance moved Report to Ctrl+7); a plugin's workspace takes Ctrl+7 on.
- **Panels that say what they are empty of, in one pattern.** Objects, History, Problems, Report, Script and the
  Model editor's Problems tab now say "No … yet: how to get some" inside their own view, at its top (Objects put its
  line under an empty list with a sideways scroll bar; the Model editor's Problems tab was blank). The Objects list
  shows its Status column only once a result is out of date, not computed or frozen, and never scrolls sideways.
  History hides # and Time at first, so Code has room in a 270-pixel panel (right-click the header to show them);
  From's tooltip gives the menu path, the step's number and its time.
- **A status bar that fits and follows.** While the status bar says something (a step's outcome, a demo's guide),
  the last step's line gives it the room (messages were cut off); with no dataset, no empty cell is left; the window
  brought to the front is selected in Objects and named in the status bar, which said "No object selected" while a
  result was in front. A time series counts periods, not rows, in the status bar, a dataset's overview and status
  line, and the spreadsheet.
- **Result windows open at the workspace's size** (most of it, up to 1100 by 760 pixels), not a fixed 680 by 460
  that left a 3-by-3 grid of impulse responses 70-pixel panels. The app's icon replaces Qt's wherever the window is
  built (the launcher set it already).
- **Result windows in plain words.** Copy producer command reads **Copy the step's code** (and in Objects' menu,
  **Show the step in History**); Copy matplotlib code reads **Copy plotting code**; the figure toolbar's Values toggle
  reads **Show values**; the Fields view is **Python access**; the corner's Add to report, which adds the whole
  result, reads **Add result to report** beside each view's own Add to report. Table columns are as wide as their
  headers ("imaginary", "modulus", "converged" were cut), each header in full in its tooltip. A VAR's summary names
  its criteria (AIC (total), not AIC_TOTAL) and shows them to six significant figures (FPE 8.84e-14, not 0.0000);
  impulse responses are titled "Impulse responses: Cholesky (VAR order)", not with parentheses inside parentheses.
- **Counts and plugins in words.** The Model editor says "1 equation for 1 endogenous variable"; Help → Plugins says
  what a plugin provides in words ("1 panel, 2 report notes, 1 step observer", not registry names), its Provides
  column wrapping instead of scrolling. Help's contents follow the page shown (its entry selected, its section open),
  and a dialog's Settings table shows defaults as words (data, not 'data') and descriptions in sentence case.
- **A dialog opens with the keyboard on its first field.** It went to the form's scroll area, an extra Tab stop
  before the first field (and the tab bar another); the scroll area takes no focus now.
- **A demo is built in the background.** Opening a demo (File → Open demo project…, Reload demo project, the Start
  window) ran its steps with the window frozen; nk_estimation, which estimates a model, froze it for over ten
  seconds. Its steps now run as a long step does: the window stays live, the status bar shows progress and Cancel,
  and a cancelled demo leaves the work open as it was.
- **The Lab book panel explains its lists.** Its objectives list says what it is for while there are none ("No
  objectives yet: a brief's objectives and hypotheses are listed here, to tick…"), Checks and Latest lab-book
  entries wrap (in full in their tooltips) instead of scrolling sideways, the first entry reads "lab book started"
  rather than "0 start · not recorded", and Curve… reads **Specification curve…**.
- **Find anything lists commands before help pages.** On a like match a menu item now comes before a help page, and
  a dialog's own help page (reached by F1 in the dialog) no longer sits beside its menu item: "var" gave
  "VAR · Help (Commands)" and "VAR · Help (Models)" among the commands.
- **The Code toggle gives All its panels back.** Turning Code on and off in All hid the Immediate panel a first run
  shows; Code off now puts back the code panels All showed before it was turned on.
- **The Immediate window follows a dark palette** (qtconsole's dark colours); it stayed white. Figures keep their
  white background, as they print and export.
- **Smaller things in dialogs and data windows.** A text field's grey hint is short ("e.g. recession == 0", not "a
  condition on the rows, e.g. recession == 0 or year >= 19…"), the full help in its tooltip and F1; an End before
  Start shows in End's text colour (with its reason above the buttons and for screen readers), not a style-sheet
  border that replaced the spin box's arrows. An unknown function in a formula says "did you mean log()?" without
  the 150 function names (the error box points to the help instead; the names stay under Details). The series
  browser gives FRED's results the window's width (a third was left empty), its note in sentence case and the
  "Enter a FRED API key" line in full; a dataset's Variables view heads its columns in words (Source key, not
  source_key) and says to select series before its buttons act. The demo picker (File → Open demo project…)
  lists short lines: it was 1,081 pixels wide.
- **Window titles in one pattern, "name — kind".** A VAR's window read "var — VAR" beside "bvar — Fitted BVAR": it is
  "var — Fitted VAR" (and "Fitted VECM"); the model editor's "Model editor: model" is "model — Model editor"; a
  solution's overview is headed "Solution g_solution", not "Dataset g_solution".
- **A dialog that opens a little too tall has room for its scroll bar.** A form a few pixels taller than its share
  of the screen (Solve model, State space model, Compare scenarios at 1366 by 768) opened without room for the
  vertical scroll bar, which then covered the right edge of its fields.
- **Generate series for each.** **Data → Generate series…** has a **For each** field: names (`gdp cons inv`),
  wildcards (`*_sa`, `ip_*`) or both, or the series picked in a dataset's series list (**Generate for each…**). Series
  and Formula are then templates: `{x}` is each series (`{x}_g`, `100*d(log({x}))`), `{1}`, `{2}`... the parts a
  wildcard matched (`ip_*` with `{1}_growth` makes `us_growth` from `ip_us`), `{{` and `}}` literal braces. A preview
  table lists every new series and its formula before it is made, flagging invalid names, formulas that do not parse
  or use missing series, names made twice and series that exist (Replace decides), and says so when nothing matches;
  OK waits until it is clean. The step records the expansion as one ordinary call listing every new series
  (`data = data.with_series(gdp_g='100*d(log(gdp))', ...)`, or `derive` with Keep updated), so scripts replay exactly
  whatever series are added later; the step keeps the templates, so reopening it (History, projects) shows them.
  Empty For each is the single series, as before; steps saved earlier open unchanged. In code,
  `data.with_each(series, name, formula, derived=False, replace=False)` does the same and is recorded as the same
  expanded call; `pymetrix.data.each.expand_each` and `preview_each` give the expansion and its checks. Panels:
  computed within each entity. A dialog whose step is built from the workspace now records the call as it was
  before the step ran (a wildcard's matches, not the series the step added). Help topic "Generate series for each".
- **An AI assistant in the open app** (`pymetrix-assistant`, a plugin; stage 1 of `docs/design/assistant-plugin.md`).
  View → Assistant, **Allow connections**: an assistant on this computer (Claude Code, VS Code, other MCP clients)
  works in the window's own session through pymetrix's MCP tools, at an address with a secret in it, which the panel
  shows and copies (with the Claude Code command). Its steps run as yours do: one at a time, in the background, in
  History, undoable, and recorded by a lab book as an agent's; the panel lists them. No Python, no file writes, no
  network data. Unticking it, or closing the app, stops it (extension API 1.8, below).

### Extension API

- **Extension API 1.7** (additive): `AppInterface.open_demo(name)`, so a plugin's panel opens a demo project as
  **File → Open demo project…** does, asking to save unsaved work first; an unknown name raises `NotFoundError` with
  the demos there are, and an interface made without the app raises `NotSupportedError`. The Guide panel uses it for
  a case study's demo and **Start over**, and still points to the File menu with an earlier extension API (1.5 or 1.6).
- **Extension API 1.8** (additive): `build_mcp_server(session_of=, run=, call=)` and `MCPPolicy`, pymetrix's MCP
  tools over a session that lives elsewhere. The Assistant plugin uses it.

### Fixes

- **Fixed, an arrow with no words could not be selected, moved or resized** on the figure: matplotlib's hit test
  misses a thin arrow line. An arrow is now picked within a few pixels of its line.
- **Fixed, a second arrow drawn on a result's graph replaced the first:** it now joins the figure already made from
  it (`<result>_graph`).
- **Fixed, `AppPanel` with an unknown dock area** raised an `ImportError` instead of its `SpecificationError`.
- **Fixed, a backtest's sample in a report's appendix** (and its Provenance tab) was its first two origins, "2015Q1 to
  2015Q2" in Lab 3's report for origins running to 2022Q4: `provenance.sample` took the second origin, not the last.
  It now reads "origins 2015Q1 to 2022Q4 (32 re-estimates), estimated on 2000Q3 to 2015Q1 at the first and 2000Q3 to
  2022Q4 at the last" (the re-estimates are recorded in `provenance.spec["re-estimates"]`); a backtest saved before
  shows its origins from its specification. Combinations say they combine backtests over those origins. Rolling and
  recursive fits and quantile processes, which had no row, now gather their fits' provenance (`RollingFit.provenance`:
  the windows' ends and the first and last estimation samples; `QuantileProcess.provenance`) and show it in the
  appendix and a Provenance tab.
- **Fixed, the Start window was hidden under the panels.** It opened at a fixed 560 by 560 pixels at the
  workspace's corner, so where the workspace was smaller (the Report workspace on a laptop: Objects and History on
  the left, Report on the right) Qt moved it partly under the Objects panel, cutting off its text and buttons, and
  the demo list scrolled sideways. It now opens sized to the workspace's visible area and centred in it: at its
  natural size where there is room, no larger than the area, never under a panel, and it scrolls when the area is
  smaller than its least size. It follows the area as panels are shown, docked or resized, and the window is
  resized, until it is moved, resized or maximized. The demo list wraps its entries (the full text is also in the
  tooltip) with no sideways scroll bar. Checked at 1366 by 768, 1280 by 720 and 1920 by 1080 in every workspace,
  and beside the Guide panel.
- **Fixed, the Assistant panel's server could not start in an app opened from the desktop shortcut** ("Unable to
  configure formatter 'default'"): uvicorn's logging set-up needs a console, which an app started without one has
  not. The server now leaves logging to the app.
- **The Start window's demo list on Windows.** At 1024 by 600 with every panel open, Windows' list items kept a
  width wider than the list (its style's item padding on top of text wrapped to the list's width), so every demo was
  cut off at the right and the window scrolled sideways. The demo and recent-project lists are now a column of
  wrapped entries (`pymetrix.ui.fit.EntryList`) laid out like any wrapped label, the same in every style: a click,
  Enter or Space opens an entry, Tab and Up/Down move between them, and the tooltip holds the full text.
- **The Start window never scrolls sideways.** Where the workspace is narrower than the page's longest word (Windows
  CI at 1024 by 600 with every panel open), the page was held at that word's width and a sideways scroll bar
  appeared. Its scroll area now never scrolls sideways (`scroll_area(..., sideways=False)`): the page is always as
  wide as the window and its text and buttons wrap to it, and a demo's text breaks after hyphens as well as
  underscores (`2025Q1-2026Q4`), with the full text in each entry's tooltip.
- **Running code after another window had closed could crash the app.** Every window's console shares IPython's one
  shell, and a closed window's console left its hooks on it, so code run in a later window also refreshed the closed
  window and its plugins' panels (the Guide's, the Lab book's), whose widgets Qt might have deleted: a segmentation
  fault. A window's console now takes its hooks off the shell when the window closes (or the console is deleted),
  a closed window's panels stop following the history, and the extension API drops a panel's `refresh` once Qt
  deletes the panel; the Guide and Lab book panels also check that their widgets are still there.
- **Tests: the window-fit check left layout items behind.** It walked layouts with `itemAt`, whose items PySide6 6.6
  keeps referenced; once Qt deleted one, a widget made later at its address came back as a `QWidgetItem` ("no
  attribute 'installEventFilter'", the oldest-versions job). It now finds a layout's widgets without `itemAt`.
- **Switching workspaces could stop with an error.** The workspace's tools bar took the action that
  `QToolBar.addAction(text)` returned, which PySide6 could hand back as the wrapper of a toolbar layout item deleted
  at the same address ("'QWidgetItem' object has no attribute 'setToolTip'"). Each tool's action is now made, then
  added.
- **Undo and Redo in every workspace.** Ctrl+Z and Ctrl+Y did nothing outside All: the Edit menu was hidden there,
  and an action in a hidden menu does not answer its keys. Edit is now shown in every workspace, and every menu
  action with a shortcut (Undo, Redo, Save, Open, Find anything, Recalculate...) is also added to the window, so its
  keys work whichever menus the workspace shows.
- **The progress bar of the window that is open.** Closing a window cleared the app's progress bar and menus even when
  a newer window had registered its own since, so a step then ran with no progress bar or Cancel. A window now
  clears only its own. This was also why a background-step test failed now and then in CI.
- **A folder chosen as a file, on Windows.** Opening a folder as a file is refused there (PermissionError) rather
  than reported as a folder, so the message said permission was refused. A path that is a folder now says "Choose a
  file, not a folder" whatever the error.
- **The Start window's demo list at its narrowest.** A demo's name with no space in it (`policy_and_nonlinear`) did
  not wrap, so with a wider font in a narrow workspace its entry was wider than the list; names now break after an
  underscore. The list also lays its entries out again at once when its width changes (not 100 ms later), and the
  ways to start go in two columns only where the page's own margins and spacing leave room for them (they were
  counted as a fixed 24 pixels, and a style that spaces by the kind of widget gave no spacing to count, so the
  buttons could go in two columns a few pixels short and the window scroll sideways).
- **The Start window in a very narrow workspace.** At 1024 by 600 with every panel open the workspace is
  about 200 pixels wide, and with a larger font (DejaVu Sans on Linux, Segoe UI on Windows) the ways to start and the
  "About this window" toggle, each on one line, were wider than that, so the window scrolled sideways. Their text now
  wraps onto a second line where one is too narrow (`pymetrix.ui.fit.WrapButton`), the lists' headings wrap too, and
  the ways to start go in two columns only where each has its text on one line.
- **The Guide's buttons in a narrow dock.** Where Back, Next and Start over are too wide to sit side by side (Windows'
  wider buttons in a dock 300 pixels wide), Start over goes on a row of its own instead of the buttons scrolling
  sideways.

## 0.1.0a12: development build (3 November 2026)

### Models & DSGE

- **Fixing or holding a variable in some periods only.** `Scenario(..., partial=True)` (or `partial=["x"]` for some
  variables) applies `fix=` and `hold=` only in the periods their paths give a value: there the variable is exogenous
  at it and its equation is dropped (or its free variable solved for); elsewhere (no value, or NaN) its equation
  determines it as usual, and a held variable's free partner keeps its data. Newton and Gauss-Seidel, compiled or
  interpreted, solve each period's own equations, so the result equals solving the fixed periods and then the rest;
  stochastic draws follow the same equations. `PerfectForesight` takes partial fixes (the variable's equation is
  `x = value` in the fixed periods) and refuses partial holds; optimal control refuses both. Without `partial` a
  fix or hold with gaps is the same error as before. Model → Solve: Partial (Advanced) for its holds.
- **Missing inputs before solving.** `Model.missing_inputs(data, span, scenario=None)` lists each period, equation
  and input (`x`, `c[-1]`) an equation lacks: exogenous series and starting values before the span, after the
  scenario's paths, fixes (partial ones too), holds and free variables; values the solve determines are not inputs.
  `Model.check(data, span)` keeps it as `ModelCheck.missing` (`scenario=` optional; the check's rows are unchanged),
  shown as Missing inputs in its result window.
- **Convergence by period.** `Solution.convergence`: a DataFrame by solved period of the solver's iterations, the
  largest absolute residual at the end and its equation, and whether the period converged (Newton, Gauss-Seidel and
  the compiled period solvers; perfect foresight with each period's residuals and the stacked iterations in every
  row; `None` for a first-order solution). `converged` and `iterations` are unchanged. A period that does not converge
  raises as before, the `ModelError` carrying `convergence` with that period marked `False`. With draws, the table
  is the deterministic solution's. Shown as Convergence in the solution's window and kept with a saved solution.
- **`Model.from_file` reads model files.** It used to say reading `.model` files was planned; it now reads by the
  file's extension: `.mod` (Dynare, `read_mod`), `.model` (IRIS, `read_iris`), `.prg` (an EViews program,
  `read_eviews`) and `.xml` (an FRB/US-style model file, `read_eviews_xml`), returning the `Model` the reader's result
  holds. Keyword options go to the reader (`params=`, `mce=`, ...); another extension is refused with the ones it
  reads, and a missing file raises `FileNotFoundError`. The readers themselves still give the template, with its
  shocks, their sizes and the measurement.

### App & figures

- **The default layout fits a laptop screen.** At 1366 by 768 (or 1280 by 720) the Script, Immediate and Problems
  panels took about half the height, leaving the result windows some 250 to 380 pixels. In a default arrangement (a
  first run, a workspace not yet arranged, **View → Reset layout**) the bottom panels now take a fifth of the window's
  height (120 to 240 pixels) and the left panels a fifth of its width (260 to 360), kept so as the window is resized
  until a panel's edge is dragged or a panel moved; the result windows get about two thirds of the height in every
  workspace. A saved arrangement is never changed.
- **The Objects panel keeps each object's type in view.** In a narrow panel a long name is shortened (in full in its
  tooltip) rather than pushing the Type column out of sight.
- **A failed step run again successfully is no longer a problem.** When a dialog fails and the same dialog then runs
  successfully on the same target (the object it makes, or else the one it works on), the failure is resolved: it
  stays in History, greyed and marked "failed, resolved" (its tooltip names the step that resolved it), and no longer
  counts in Problems. Undoing that step makes the failure count again; typed code is never resolved this way.
  Projects and replay are unchanged (failed steps are neither saved nor replayed).

### Speed

- **A quicker `import pymetrix`.** `scipy.stats` and `scipy.signal` (which imports it) are loaded on first use, not
  when pymetrix is imported (`pymetrix._lazy`): about 1.6 s to 1.1 s warm, more on a cold start.
- **Multivariate GARCH, panel VARs, long expressions and lags are faster, with the same results.** The BEKK, VECH
  and DCC likelihoods run their recursions over all periods at once: a full VECH fit takes about 30 s instead of
  260 s, and diagonal VECH, BEKK and asymmetric DCC fits about a tenth of the time. A panel VAR splits its sample by
  entity once (the wage panel's historical decomposition with 20 draws: about 3 s instead of 22 s; panel VAR fits
  several times faster). Differentiating a long expression no longer walks each subtree again (the Jacobian of a
  240-term sum: 0.5 s instead of 24 s), and `Dataset.lagged` shifts consecutive periods by position, which speeds
  up estimation, forecasting and backtests that take many lags.

### Lab book (plugin)

- **The project brief in the lab book** (`pymetrix-labbook`, research-audit stage 3, first part). `brief(...)` records
  the project's question, objectives, hypotheses (with what would support or contradict each), outcomes, intended
  data and methods before the work; calling it again amends it, with a reason, and both stay in the chain.
  `skip_brief` records that there is none. `note(...)` records the reasoning behind a choice. Steps run inside
  `objective("O1", "H1")` are linked to what they address. `verify` reports a brief written after the work began,
  objectives with no analysis, analyses linked to none (exploratory) and hypotheses added after results on them.
  In the app, the Lab book panel (View → Lab book) writes the brief, its amendments and notes, links the steps run
  to the objectives ticked, and shows the brief's checks and the latest entries as the work goes on.
- **The pre-registered analysis plan** (`pymetrix-labbook`). `lock_plan(...)` locks the exact specifications, the
  datasets, the sample and exclusion rules, the hypotheses and the primary outcomes before the data are read, with a
  hash of the plan; changing it needs a reason. Each later step is labelled planned or exploratory, and `verify`
  reports the deviations: planned data read before the lock, planned specifications not run, exploratory steps, and
  each change with the work done before it. **Plan…** in the Lab book panel does the same in the app.
- **A private brief, proved by its date** (`pymetrix-labbook`). `commit_brief(...)` keeps the brief in a private file,
  with a random salt, and puts only its hash in the lab book, for time-stamping outside pymetrix; `reveal_brief(...)`
  discloses it at publication, checked against the commitment and counted as written when committed. In the app:
  "Keep it private" in the brief form, and Reveal… in the Lab book panel.
- **Counting what was tried** (`pymetrix-labbook`). Each fit's outcome, sample and estimates go into the lab book;
  `tried(session, "fit")` says where a result stands among the specifications fitted for its outcome ("1 of 23"),
  every fit counting, kept or not; `specification_curve(...)` draws them. `verify`, the panel's checks and its
  Curve… window show the same.
- **An AI auditor's tools** (`pymetrix-labbook audit study.pymx`, research-audit stage 5). A read-only MCP server
  over a project's lab book: summary, entries (by kind, outcome or range), the brief, deviations from the brief and
  plan, each fit of an outcome, and the verify report. The assistant is told to cite entries, flag patterns for a
  person to judge and give no verdict. It never writes the project, and it is separate from the researcher's own
  assistant. Needs `pymetrix-labbook[audit]`.

### Extension API

- **Report notes** (extension API 1.6). A plugin registers a `ReportNote` (`report_notes` group): a line about each
  result a report exports, under its own heading, before the appendix. Export report's new **Plugins' notes** option
  (Advanced; `plugin_notes=True`) adds them; it is off by default, because the notes come from the session's record
  and a report's recorded script cannot write them again. The lab book's note says how many specifications of the
  result's outcome were tried.

### Fixes

- **docs/features.md** has a row for second-order estimation and the particle filter (bootstrap or guided, particle
  marginal Metropolis-Hastings, the particle smoother; experimental), taken out of the OccBin row, and rows of their
  own for the No-U-Turn sampler and the analytic Kalman-filter gradient (beta), taken out of the Bayesian
  estimation row.
- **A step run in the background** (the progress bar and Cancel) no longer lets Python's garbage collector run on its
  worker thread, where it could free Qt objects left from earlier windows off the GUI thread: a crash, or the window's
  timers stopping. Collection waits for the GUI thread while the step runs.
- **A step run in the background** also keeps a collection started on another thread from freeing those Qt objects:
  JPype runs one on a Java thread whenever Java's collector has run (after TRAMO-SEATS has started Java), and a
  library can turn automatic collection back on during the step. The objects that exist when a step starts are
  frozen until it ends, and the collector is turned off again if something turned it on.
- **The default panel layout** fits again a workspace left before its panels were sized to the window (on a slow
  machine), and sizes the Objects panel and the bottom panels from the font, scroll bar and toolbars as they are, so
  the Type column stays in view and the result windows keep most of the height with a larger font (as on Windows).

## 0.1.0a11: development build (3 November 2026)

### Data & dialogs

- **Load from connector, and a series browser to find the codes.**
  - **File → Load from connector…** reads from FRED, the World Bank, SDMX services or a plugin's connector: the
    provider (the installed connectors, `pymetrix.io.connectors()`), series codes, first and last periods and, where
    the connector keeps them (FRED, through ALFRED), a vintage date; it writes the address `Dataset.read` takes
    (`fred://GDPC1?start=1990Q1`), so the step replays in code. FRED addresses take `start=` and `end=` (the periods
    kept after the download) and World Bank addresses `start=` and `end=` (for `date=`). Readers describe their
    addresses with `register_reader(scheme, reader, Connector(...))`.
  - **File → Browse series…** searches and browses FRED (series search and categories; FRED's web service needs an
    API key, from `FRED_API_KEY` or typed in the browser for the session only, while reading series by ID still needs
    none), the World Bank (the indicators of a source, the World Development Indicators by default; sources, topics
    and countries) and SDMX providers (a provider's dataflows, then a picker per dimension of the chosen dataflow's
    data structure, with the series key previewed as it is built). Results show title, code, frequency, units, first
    and last period and last update, with the provider's notes beside them; several can be selected. **Load** reads
    them as a recorded `Dataset.read(address)` step (replay needs neither the browser nor the cache); **Add to Load
    from connector…** opens that dialog with the codes filled in.
  - In code: `pymetrix.connectors.search("worldbank", "gdp per capita")`, `browse(...)`, `structure("sdmx",
    "ECB/EXR").key(...)` and `to_frame(...)`, all answering with one `SeriesInfo` type. Searches are kept on disk for
    a day and metadata for a week in `~/.pymetrix/cache/connectors` (`refresh=True`, or "Ask again" in the browser,
    skips them; `configure_cache`, `cache_entries`, `clear_cache`); requests are kept without their credentials and no
    key is ever written. `Connector` descriptions say whether a connector can be searched (`search=`,
    `search_note=`); plugins add searchers with `pymetrix.connectors.register_searcher`.
    `register_sdmx_provider(..., structure_url=, agency=)` makes an institution's own SDMX service searchable. Help
    topic "Data connectors".
- **Dialogs for frequency conversion and aggregation.** **Data → Transform → Convert frequency…** (`data.to_freq`:
  to quarterly, annual, a July-June fiscal year, monthly or weekly; sum, average, first or last for every series, or
  each series by its measure with rules for some under Advanced; to a higher frequency it points to Temporal
  disaggregation) and **Data → Panel → Aggregate over entities…** (`data.aggregate`: totals or averages by period, or
  by a category and period). `to_freq` and `aggregate` are now recorded calls, so typed code and the dialogs record
  the same step. Help topic "Changing frequency and aggregating".

### Estimation & tests

- **New dialogs: tests that need the data.**
  - **Test → Weak-instrument tests…**: Anderson-Rubin (a test of one coefficient, or its confidence set over a grid),
    Stock-Wright S and the C (difference-in-Sargan) test, on an equation estimated by IV or GMM (its equation,
    instruments and sample are used), with the fields each test needs.
  - **Test → Non-nested tests (J, Cox)…**: one fitted equation against another.
  - **Estimate → VARs and local projections → Gregory-Hansen cointegration test…**, beside Cointegrating regression.
  - **Estimate → Quantile process…**: a fitted equation's coefficients across quantiles, with bands and least squares.
  - `anderson_rubin`, `anderson_rubin_set`, `stock_wright`, `c_test`, `j_test`, `cox_test` and `quantile_process` are
    now recorded calls, so typing them records the same step as the dialogs.
- **Dialogs for principal components and factor analysis**, in **Estimate → Factors**: **Principal components…**
  (`PCA`: components kept, standardized or not, sample) and **Factor analysis…** (`FactorModel`: number of factors,
  no, varimax or promax rotation, sample); the result windows show the loadings, the number-of-factors criteria or
  test and the scores. **Data → Add results to dataset…** (Next ▸ on the results) adds the components, or regression
  or Bartlett factor scores, as series. Scripts using the core `FactorModel` no longer import the finance plugin's
  result class of the same name over it. Help topic "Principal components and factor analysis".
- **New dialog: heterogeneous panels.** **Estimate → VARs and local projections → Panel mean group and CCE…**: mean
  group, CCE mean group, CCE pooled or the pooled mean group (whose window has its Hausman test).
- **Spectra and long memory dialogs.** **Test → Spectrum…**: the periodogram, a smoothed spectrum or the
  cross-spectrum (coherence, phase and gain) of two series, with the bandwidth. **Estimate → Time series → Long
  memory…**: d by GPH or the local Whittle (each with its power), or an ARFIMA(p, d, q).

### VARs & identification

- **Dialogs for more VAR identification and VAR types** (Estimate → VARs and local projections):
  - **Impulse responses…**, **Variance decomposition…** and **Historical decomposition…** add the schemes MaxShare
    (the shock explaining most of a variable's forecast-error variance up to a horizon: Variable, Horizon and,
    advanced, the first horizon) and News (the same with no impact effect, Barsky and Sims); responses and variance
    decompositions add Heteroskedasticity (Rigobon: Breaks, the first period of each new variance regime), and
    responses Generalised (`var.girf`, Pesaran and Shin). `HeteroskedasticitySVAR.fevd(periods)` (new) gives the
    variance decomposition in the first variance regime.
  - **BVAR with stochastic volatility…** (`BVARSV`), **Markov-switching VAR…** (`MSVAR`: what switches, the number of
    regimes, starts) and **Quantile VAR…** (`QuantileVAR`). **Impulse responses…** takes their results, with Period
    (the BVAR-SV's volatilities) and Regime (the MS-VAR's) under Advanced, and their windows offer it under Next ▸.
  - **VARMA…** (`varma`), and under Estimate → Nonlinear **Threshold VAR…** (`threshold_var`) and **Bilinear
    model…** (`bilinear`); **Test → Threshold VAR linearity (Tsay)…** (`tvar.test`, also Next ▸ on the result).
- **VECM tests dialog.** **Test → VECM restrictions and stability…** (also Next ▸ on a VECM): Johansen's tests on the
  cointegrating vectors (`test_beta`, H written `1 0; -1 0; 0 1`) and of weak exogeneity (`test_weak_exogeneity`), and
  the eigenvalue fluctuation test (`fluctuation_test`, with Start, Draws and Seed).
- **Zero and sign restrictions together** (Arias, Rubio-Ramírez and Waggoner 2018). `SignRestrictions` takes
  `zeros={"monetary": ["gy", "pi"]}` (variables a shock does not move on impact) and `long_run_zeros={...}` (nor in the
  long run, the cumulated response), with or without signs; a shock may have zeros alone. Rotations are drawn
  column by column in the null space of the zeros (ARW's algorithm), signed by each shock's first sign and kept when
  every sign holds, with ARW's importance weights (the volume-element ratio, computed numerically) so that the kept
  draws come from the posterior conditional on the zeros; they are resampled by weight. It works wherever sign
  restrictions do: median and median-target responses, bands across the set, from bootstrap draws or from a
  Bayesian VAR's posterior draws (ARW's one rotation per draw, in passes, weighted), variance and historical
  decompositions, narrative restrictions. New module `pymetrix.zerosign`; new **Zeros** field (and **Long-run
  zeros**, advanced) in the Impulse responses, Variance decomposition and Historical decomposition dialogs, written
  `monetary: gy, pi`. Sign restrictions without zeros are unchanged.
- **Historical decompositions: bands, and more kinds of VAR.**
  - Bands: `historical_decomposition(draws=..., alpha=0.05, seed=0)` gives pointwise quantiles of each shock's
    contribution over a Bayesian VAR's posterior draws (each draw with its own coefficients, residuals, baseline and
    impact matrix; under sign or zero-sign restrictions a rotation accepted for that draw), or over the identified set
    at the estimates under sign restrictions. The result's `lower`, `median` and `upper`, `bands_table(variable)` and
    `plot_bands()` (shaded bands, median, point); the result window adds the bands' graph and tables. The default
    (`draws=0`) is the point decomposition as before (a BVAR's at the posterior mean).
  - Panel VARs: `pvar.historical_decomposition(draws=0, alpha=0.05, seed=0, entity=None)` decomposes each entity's
    data (GMM on forward orthogonal deviations or within least squares) into the identified shocks' contributions
    (Cholesky by default, or `pvar.identify(...)`) and the baseline from the entity's first lags and its fixed effect;
    a `PanelDecomposition` holds one `Decomposition` per entity (`hd[entity]`, `hd.select(entity)` for the views of
    another, `hd.summary()` with each entity's sample and fixed effects, `hd.to_frame()` long). Bands are quantiles
    over the coefficient draws the panel VAR's impulse-response bands use. `FittedPanelVAR` keeps its rows in levels
    (`levels`) and `identify` is a recorded call.
  - Proxy SVARs: `psvar.historical_decomposition(draws=0, alpha=0.05, seed=0)` splits each variable into the
    proxy-identified shock's contribution, the other shocks' together and the baseline, with bands from Jentsch and
    Lunsford's moving-block bootstrap draws (the same draws as the responses' bands for the same seed). `ProxySVAR`
    keeps its proxy (`z`) and block length for the bootstrap.
  - BVAR-SV: `FittedBVARSV.historical_decomposition(coverage=, draws=)`, draw by draw, each period's shocks through
    that period's impact matrix A^-1 D_t, the mean over draws with bands. MS-VAR:
    `FittedMSVAR.historical_decomposition()` along the most likely regime path (`regime_path()`: the regime with the
    highest smoothed probability each period supplies the intercepts, dynamics and impact; its intercepts are in the
    baseline). No historical decomposition for quantile VARs (see known gaps).
  - Every entity's and every draw's parts add up to the data. **Historical decomposition…** takes all of these, with
    **Draws** (default 0) and, advanced, **Coverage** and **Seed**, and a new **Entity** field for panel VARs.
- **Variance decompositions of BVAR-SV, MS-VAR and quantile VAR results.** `FittedBVARSV.fevd(periods, period=,
  coverage=)`: the shares with the volatilities of a date (as `irf`), posterior means with bands over the draws.
  `FittedMSVAR.fevd(regime, periods)` within a regime held fixed. `FittedQuantileVAR.fevd(periods)`: quantile
  connectedness (Ando, Greenwood-Nimmo and Shin), orthogonalised by Cholesky; `residuals()`. `FEVD` gains optional
  bands (`lower`, `upper`, `coverage`, `band_source`, `bands_table(variable)`, shown in its window). **Variance
  decomposition…** takes these results (Period, Coverage, Regime, Draws and Seed under Advanced), and their windows
  offer it and Historical decomposition… under Next ▸. The fitted MS-VAR and quantile VAR keep the data fitted, with
  the first lags' observations, as `sample_rows`.
- **Markov-switching VARs find clearly separated regimes, in seconds.** `MSVAR.fit` could miss regimes that stand far
  apart: its random soft splits of the sample start the regimes alike and EM keeps them alike, so on a simulated VAR
  whose intercepts switch by 4 and 3 standard deviations it stopped at a log likelihood of -892.8 (69% of periods
  classified right) where the maximum is -791.5. It now also starts from regime paths read off the data (each
  variable split at its quantiles, raw and as a rolling mean; rolling variances of the linear VAR's residuals;
  k-means on the data and its lags and on the residuals; a one-variable Markov-switching fit of each variable),
  screens every start by a short EM run, restarts any that collapsed (identical regimes, or one never in effect)
  from a perturbation of the best, and runs the best few to convergence. The final polish is BFGS on the exact
  likelihood over all the parameters (was Nelder-Mead on the transition matrix, alternating with EM steps), and the
  Hamilton filter and Kim smoother are vectorised over time (prefix products of the transition-density matrices):
  the tests' fits that took 1 to 35 seconds take 0.3 to 2. On every fit of the earlier tests the log likelihood is
  the same or higher (by up to 3e-8). Arguments are unchanged: `starts` now counts the random splits among the starts
  (`starts - 1` of them, as before), so results with the same seed can differ slightly from earlier versions'.
- **Bands for Markov-switching VAR responses and decompositions.** `FittedMSVAR.irf`, `fevd` and
  `historical_decomposition` take `draws=`, `coverage=` (default 0.68) and `seed=`: pointwise quantiles over draws of
  the parameters from the ML estimates' asymptotic normal distribution, on a scale where every draw has valid
  covariance and transition matrices (Cholesky factors with log diagonals, transition logits). The covariance is the
  inverse of minus the numerical Hessian of the exact log likelihood (`parameter_covariance()`, equal to statsmodels'
  for one equation); `parameter_draws(draws, seed)` gives the draws. The historical decomposition's bands hold the
  most likely regime path fixed, so they leave out the uncertainty about the regimes; a transition row with a
  probability at 0 or 1 is held at its estimates. Point estimates are unchanged. **Impulse responses…**,
  **Variance decomposition…** and **Historical decomposition…** pass Draws, Coverage and Seed to an MS-VAR (Impulse
  responses' default of 500 draws now gives an MS-VAR bands).

### Forecasting

- **Richer conditional forecasts: ranges, structural scenarios, the posterior updated by the conditions, conditions
  in forecast rounds.**
  - VAR and Bayesian VAR conditional forecasts take ranges as well as exact values: `Bounds(3, 4)` ("between 3 and
    4"), `Bounds(lower=3)`, `Bounds(upper=4)`, `(3, 4)` or `{"lower": 3, "upper": 4}`, for every period or some.
    Given the exact conditions, the bounded values are drawn from their normal distribution truncated to the bounds
    by Gibbs sampling (Geweke 1991), and the rest of the path from its distribution given them; every draw meets
    every bound. A VAR's forecast with a range is simulated (`draws=`, default 5,000; `seed=`): mean, sd, median and
    quantile interval. Exact conditions alone give the same results as before.
  - Structural scenarios (Antolín-Díaz, Petrella and Rubio-Ramírez 2021): `shocks=[...]` names the identified shocks
    that may move to meet the conditions, the others keeping their usual distribution (minimum-norm when there are
    many ways to meet them, refused when there are none); `shock_conditions=` conditions the structural shocks
    themselves (`{"y": 0.0}`: no demand shocks; `Bounds(lower=0)`: positive ones only).
  - Bayesian VARs: `update_posterior=True` runs Waggoner and Zha's (1999) Gibbs sampler, which alternates the path
    given the parameters and the conditions with the parameters given the data and the path, so the conditions also
    inform the posterior (conjugate Minnesota, GLP and independent normal-Wishart priors, under the hyperparameters
    the fit reports: the searched tightness and cross-tightness of `NormalWishart(optimize=True)`, GLP's estimated
    psi). Without it, as before, each posterior draw is conditioned in turn.
  - First-order model plans keep variables within ranges: `Scenario(hold_lower=..., hold_upper=..., free=[...])`,
    the least-norm free shocks with the bounds as constraints (a range that does not bind changes nothing); the
    Solve and Forecast from estimated model dialogs gain Hold lower and Hold upper.
  - Forecasts record their conditions (`meta.extra["conditions"]`), and forecast rounds keep them in their record:
    `ForecastRound.conditions`, `conditions_table()`, the round's view, `from_forecast(conditions=...)`, saved in the
    store (rounds saved before load with none) and replayed with the project. A model plan's holds and ranges are
    recorded too.
  - The Conditional forecast dialog takes ranges (`pi=3..4`, `pi>=3`, `pi[2026Q1:2026Q4]<=4`) and, under Structural
    scenario, the shocks that may move and conditions on the shocks; Update posterior for Bayesian VARs.
- **New dialogs: forecast evaluation.**
  - **Forecast → Superior predictive ability…** (Hansen's SPA test against a benchmark backtest) and **Forecast →
    Binary forecast scores…**, after Compare many forecasts. New `Backtest.spa(models, variable, horizon, ...)`;
    `binary_scores` takes `data=` (its arguments then name series) and is importable from `pymetrix`.
  - **Forecast → Backtest…** has a **Vintages** field for real-time backtests: `origin=dataset` for every origin.
  - **Estimate → Forecast…** offers smooth-transition autoregressions.
- **Dialogs for the nowcasting data tools**, in **Estimate → Mixed frequency** after the bridge and dynamic factor
  models: **Ragged edge…** (`data.ragged_edge`), **Data as known on a day…** (`data.as_of`), **Fill the ragged
  edge…** (new `data.fill_edge`, as `nowcast.fill_edge`), **Realign the ragged edge…** (new `data.realign`) and
  **Combine nowcasts…** (`nowcast.combine` of bridge equations' and dynamic factor models' nowcasts, equal or given
  weights). Publication lags can be overridden under Advanced. `as_of`, `ragged_edge`, `FittedBridge.nowcast`,
  `FittedDynamicFactor.nowcast` and `nowcast.combine` are recorded calls, and `from pymetrix import *` now gives the
  `nowcast` module, as scripts name it (`nowcast.combine(...)`). Next ▸ on bridge and dynamic factor results opens
  Combine nowcasts. Help topic "Nowcasting and the ragged edge".

### Models & DSGE

- **Model dialogs: shock decompositions, exporting and importing models, add factors, optimal control and residual
  samples in Solve.**
  - **Model → State space → Shock decomposition…** splits an estimated model's smoothed history into each shock's
    contribution and the starting point (`FittedModel.decompose`, now a recorded call), with **Start** (first period
    or presample) and, advanced, **OccBin** (the OccBin-smoothed path with the regimes held at the smoothed ones;
    by default when the model was estimated with the OccBin filter). A model estimated with known-date regimes is
    decomposed through their rules. Next ▸ on an estimated model offers it.
  - **Model → Export model…** writes a template, Dynare model or Model in pymetrix's, Dynare's, IRIS's or EViews's
    syntax (`export_model`; EViews as a program or the model's text, and the origin, advanced): the result has the
    translation notes, the check against the original and the text. (The model editor's **Write as** button already
    does this from the text being edited; the editor has a toolbar, not a menu bar.)
  - **Model → Import → FRB/US XML file…** (`read_eviews_xml`, with the model-consistent-expectations groups to use)
    and **EViews state-space object…** (`read_eviews_sspace`, with starting values as `c(4)=1.1` or `c_4=1.1`), next
    to the Dynare and IRIS imports. The XML model's window lists each variable's kind, sector, MCE group, stochastic
    type and definition, and its equations; Next ▸ offers Add factors, Solve and Optimal control. A state-space
    object read shows its equations and starting values.
  - **Model → Add factors…** (`Model.add_factors`, now a recorded call): each equation's residual at the data over a
    span, so the model solved with them tracks the data. **Model → Solve…** takes them under Scenario, advanced,
    **Add factors** (`Scenario(adjust=...)`).
  - **Model → Policy → Optimal control…** (`optimal_control`): instruments, loss weights, desired values, span,
    discount; advanced, the control span, add factors for the baseline, Newton or Gauss-Seidel, tolerance and
    iterations. The window shows the loss with and without the moves, the moves, the loss's variables on both paths
    and the gradient.
  - **Model → Solve…**, Stochastic simulation: **Errors** gains **sample**, a sample of residuals resampled a period at
    a time: a dataset's series named by equation (**Residual series**, empty = all) over chosen periods, or add
    factors over a historical span; **Error scale** (`2026Q2=0.5`) and **Workers**. `Model.solve(errors=...)` now
    also takes a Dataset (its series over its active sample) or add factors by equation (the periods they all have)
    besides a DataFrame. The **Equations** box (Solve, Add factors, Optimal control) takes a model read (an FRB/US
    XML model) too: `Model([...])` takes Models, and templates or XML models holding one, bringing their equations,
    parameter values and roles. `Dataset.keep` is a recorded call.
- **The particle filter and smoother, and estimation of second-order solutions.**
  - `SecondOrderSolution.particle_filter(data, measurement)` (and `LinearSolution.particle_filter`,
    `pymetrix.particle_filter`) runs a bootstrap filter (systematic resampling when the effective sample size falls
    below half the particles; `proposal="guided"` draws the shocks from the conditional normal of the measurement
    equations linearised at the steady state) on the pruned second-order state (Kim-Kim-Schaumburg-Sims;
    Andreasen-Fernández-Villaverde-Rubio-Ramírez), with the measurement equations evaluated exactly on each particle
    (transforms, deterministic trends, lags and reported series) and Gaussian measurement errors. A `ParticleFilter`:
    the log likelihood estimate and its path, filtered means and weighted quantiles, the effective sample size and
    resampling by period, provenance with the seed and generator.
  - The particle smoother: `particle_filter(..., smooth_paths=N)` (and on first- and second-order solutions) draws N
    paths by backward simulation in the shocks (Godsill, Doucet and West 2004; for degenerate transitions as Lindsten
    and Schön 2013 set out): `ParticleFilter.smoothed`, `smoothed_quantiles`, `smoothed_shocks`.
  - `Model.estimate(..., order=2, filter="particle" | "guided", particles=N)` (templates and Dynare models too)
    estimates on it with common random numbers from `seed`: Nelder-Mead on transformed parameters, restarted, and
    standard errors from a Hessian over about one standard error; `FittedModel.nonlinear` and `.particle` keep the
    solution and the filter at the estimates, and `smoothed` holds smoothed values (`estimate(...,
    smooth_paths=100)`; 0 keeps the filtered means). `fit.mcmc` on such an estimate is particle marginal
    Metropolis-Hastings (a fresh likelihood estimate at each proposal, the current draw's kept); `fit.smc` and NUTS
    refuse it. New options under Advanced in **Model → Estimate model…** (order, filter, particles, smooth paths).
  - Checked against the Kalman filter and smoother on a linear model (the likelihood converges as 1/sqrt(N), the
    filtered means agree), grid filters and smoothers for a nonlinear measurement and for pruned second-order
    dynamics, the second-order simulation path for path, and parameter recovery on simulated data. Existing results
    are unchanged.
- **DSGE-VAR(lambda) (Del Negro and Schorfheide 2004; Del Negro, Schorfheide, Smets and Wouters 2007).**
  `model.dsge_var(data, measurement, lags=4, lambda_=...)` (also `template.dsge_var(data, ...)` and, for a Dynare file,
  `dyn.dsge_var()`): a VAR(p) with a constant in the observables whose normal-inverse-Wishart prior comes from the
  linearised model's population moments (`G_yy`, `G_xy`, `G_xx`, uncentred, from the first-order solution and the
  measurement equations), weighted as lambda T observations; the model's parameters estimated by the closed-form
  `p(Y | theta, lambda)` (posterior mode, then `.mcmc(...)`, with the parameters' priors, bounds and transforms of
  `Model.estimate`; numerical gradients, as the analytic one is the Kalman filter's). `lambda_` is fixed (at least
  `(k + n) / T`), chosen on a grid by the marginal data density (`"grid"`, lambda-hat; `lambdas()` and
  `lambda_plot()` show the hump), or estimated with a prior (`Gamma(1, 0.5)`). The result, `FittedDSGEVAR`, gives
  the VAR's prior centre and posterior given theta (`var_prior`, `var_posterior`, `moments`), draws of the VAR given
  theta's draws (`var_draws`), impulse responses identified by the model's rotation (QR of the model's impact matrix
  mapped onto the VAR's Cholesky factor) with posterior bands beside the model's own (`irf`, `dsge_irf`), variance
  decompositions (`fevd`) and predictive forecasts with bands (`forecast`). New dialog **Model → Estimate
  DSGE-VAR…**; **Model → Metropolis-Hastings…** takes the result. Dynare's `estimation(dsge_var=..., dsge_varlag=...)`
  and `dsge_prior_weight` in `estimated_params` are read by `DynareModel.dsge_var()`. Checked against a long
  simulation of the model (moments), the normal-inverse-Wishart identity (marginal likelihood), the limits in lambda
  (least squares and the model's VAR) and the model's own responses at a very large lambda (see validation.md).
  Existing results are unchanged.

### Bayesian

- **The analytic gradient of a linearised model's Kalman-filter likelihood.** The steady state (implicit-function
  theorem), the linearised equations (symbolic second derivatives), the first-order solution (Iskrev's
  implicit-function derivatives of `P` and `Q`, a Sylvester-type equation solved on the Schur form, by triangular
  systems), the measurement equations, shock sizes and correlations, measurement errors, the stationary initial
  variance (a Lyapunov equation) and the Kalman recursion are differentiated exactly (`pymetrix.models.kalman_score`).
  The filter's derivatives are taken in reverse mode (the filter forwards, its adjoint backwards, each state's periods
  summed directly, then one contraction with every parameter's derivatives of the system matrices), so a period's
  work does not grow with the number of parameters; the filter calls LAPACK directly and does everything that does
  not need the loop for all periods at once. Compiled model expressions (`pymetrix.expr.compile`) work out repeated
  subexpressions once and run on Python floats when given plain numbers (steady states, linearisations), and
  `InvGamma1` finds its parameters once. Estimation uses the gradient in BFGS through the parameter transforms and for
  the Hessian (differences of the exact gradient); NUTS (`fit.mcmc(sampler="nuts")`) uses it at every leapfrog step.
  `fit.gradient(at=..., method="analytic" | "numerical")` shows it; `gradient="auto" | "analytic" | "numerical"` in
  `estimate` and `mcmc` (Gradient, Advanced, in **Model → Estimate model…** and **Metropolis-Hastings…**). Where it
  does not apply (known-date regimes, OccBin, a DSGE-VAR's likelihood, diffuse starts, growth models) central
  differences are used as before. Checked against five-point differences to 1e-6 relative on the New Keynesian
  template (measurement errors, correlated shocks), Smets and Wouters (2007), a Dynare cash-in-advance model, the
  growth model and a model with lags, statics, missing data, presample and each start.
- **The Kalman filter's loops compiled, optionally.** With numba installed (`pip install pymetrix[fast]`, never
  required) the filter's period-by-period loops run compiled (`pymetrix.models._kalman_compiled`): the analytic
  gradient's filter and adjoint, and the plain log likelihood where no state starts diffuse (instead of statsmodels'
  filter, whose set-up dominated it). Equal to the NumPy path to 1e-10 (tested; the tests skip without numba).
  Timings, with and without numba, measured together on a loaded machine: a gradient on the New Keynesian template
  (7 parameters) 4.4 ms (8.0 without) and a log posterior 2.3 ms (4.3); on Smets and Wouters (2007) (36 parameters)
  40 and 7.2 ms (49 and 13.4). Measured alone without numba: 6.7 and 3.3 ms on the template, 41 and 11.6 ms on Smets
  and Wouters. Without numba, the Sylvester step on the Schur form by triangular systems and the adjoint's sums over
  each state's periods gain a few percent.
- **The No-U-Turn sampler** (`pymetrix.nuts`): Hoffman and Gelman's NUTS as Stan runs it (multinomial sampling, the
  generalised no-U-turn criterion with Stan's extra checks, dual-averaging step size, the mass matrix adapted in
  Stan's warmup windows, divergences reported), in NumPy, several chains with their own random streams.
  - Bayesian single equations: `Bayes(..., sampler="nuts")`, with analytic gradients for Gaussian, logit, probit and
    Poisson equations (log sigma sampled; bounded priors through the logistic or log, with the Jacobian).
  - Linearised models: `fit.mcmc(..., sampler="nuts", target_accept=0.8, max_depth=10)`, with the analytic gradient
    above, or central differences on the transformed parameters where it does not apply (2k + 1 evaluations a
    leapfrog step); a particle-filter estimate is refused. `fit.mcmc(sampler="nuts", workers=...)` runs chains at once
    in other processes, as Metropolis-Hastings chains run (by default when the run is long enough), with the same
    draws. **Model → Metropolis-Hastings…** has a sampler choice.
  - The mass matrix: `metric=` in `fit.mcmc(sampler="nuts")` and `Bayes(sampler="nuts")`. `"dense"` adapts the
    posterior's whole covariance in Stan's warmup windows (each window's sample covariance regularised as Stan's,
    `n/(n+5) S + 1e-3 5/(n+5) I`), starting from the inverse Hessian at the mode; `"diag"` its variances; `"mode"`
    keeps the mode's inverse Hessian and adapts the step size alone (a short warmup then suffices). Linearised models
    default to `"dense"`; single equations and `pymetrix.nuts.sample(metric="diag" | "dense")` to the diagonal.
    **Metric** (Advanced) in **Model → Metropolis-Hastings…** and the Bayes method's settings.
  - Smallest effective sample a second (docs/validation.md, and the guide on which to use): on the New Keynesian
    template, one chain, dense 10.5, diagonal 4.2, Metropolis-Hastings 4.1; four chains in parallel, dense 13.1,
    diagonal 4.9, Metropolis-Hastings 7.1. On Smets and Wouters (2007), dense 0.023, diagonal 0.010,
    Metropolis-Hastings 0.005.
  - Checked on a correlated, ill-scaled Gaussian, the conjugate regression's closed-form posterior, least squares and
    the t variance, a logit against random-walk Metropolis-Hastings, an AR(1) model's posterior on a grid, and the New
    Keynesian template against Metropolis-Hastings. Existing samplers are unchanged.
- **Bayesian VARs: the hierarchical prior of Giannone, Lenza and Primiceri, and a marginal likelihood for the
  independent normal-Wishart prior.**
  - `Bayes(prior=GLP())`: the conjugate Minnesota prior with the sum-of-coefficients and single-unit-root dummy
    observations whose hyperparameters have hyperpriors, as in Giannone, Lenza and Primiceri (2015): gamma on the
    tightness (mode 0.2, s.d. 0.4), the sum-of-coefficients and the single-unit-root hyperparameters (mode 1, s.d. 1),
    inverse gamma (0.02^2, 0.02^2) on the prior scales psi, which are estimated too; the authors' bounds. By default
    the hyperparameters are at their posterior mode (marginal likelihood times hyperpriors); `GLP(mcmc=True)` draws
    them by random-walk Metropolis-Hastings, and the coefficients and covariance given each draw, so forecasts,
    responses and their bands integrate over them (`fit.hyperparameter_draws`). The marginal likelihood is GLP's
    eigenvalue formula (`bvar_hierarchical.glp_log_ml`), equal to the conjugate one at fixed hyperparameters.
  - `NormalWishart` fits now report a log marginal likelihood: Chib's (1995) estimate from the Gibbs draws (two
    blocks, Rao-Blackwellised; the dummy observations' own normalisation by a second run). `NormalWishart(optimize=True)`
    chooses the tightness, cross-tightness and any dummy hyperparameters by it. The VAR dialog offers both (the
    hierarchical prior, with MCMC over its hyperparameters; the search for the independent prior, which it refused).
  - Checked against the conjugate marginal likelihood (exact), quadrature and importance sampling (Chib's estimate),
    and the hyperparameters' posterior on a grid (the MCMC). Existing results are unchanged, except that
    `NormalWishart` fits have a finite `log_ml` where it was NaN.
- `priors.Beta` and `priors.Gamma` have docstrings (shown in notebooks' reference cells).

### Applied micro

- **Applied microeconometrics.** `double_ml(data, outcome, treatment, controls)`: double/debiased machine learning
  (Chernozhukov et al. 2018) for the partially linear model (`model="plr"`), partially linear IV (`"pliv"`,
  `instrument=`) and the interactive model's ATE or ATT by the doubly robust score (`"irm"`, `score=`), with K-fold
  cross-fitting (`folds=`), repeated splits aggregated by the median (`repeats=`), clustered errors and folds
  (`cluster=`), and learners `"ols"` (least squares and logit, always available), `"lasso"`, `"forest"`,
  `"boosting"` or pymetrix's `Lasso()`, `RandomForest()`, `GradientBoosting()` (scikit-learn, `[ml]`).
  `shift_share(data, outcome, treatment, shares, shocks)`: Bartik IV by 2SLS with robust, Adao-Kolesar-Morales (AKM,
  AKM0) and Borusyak-Hull-Jaravel shock-level standard errors (optionally clustered by groups of industries) and
  Goldsmith-Pinkham-Sorkin-Swift Rotemberg weights; its `summary()` and `shock_level()` tables name the interval
  computed (`[0.05` and `0.95]` for `alpha=0.1`). `interactive_fe(panel, outcome, regressors, factors=)`: Bai's
  (2009) interactive fixed effects with additive effects, Bai and Ng's criteria (`factors="auto"`), Bai's standard
  errors and bias correction. `conditional_logit(panel, outcome, regressors)` (Chamberlain's fixed-effects logit) and
  `choice_logit(data, choice, regressors, case=, alternative=)` (McFadden's conditional logit, with alternative
  constants and case variables), on one likelihood. Dialogs under Estimate → Microeconometrics; help topics
  `double-ml`, `shift-share`, `interactive-fixed-effects`, `conditional-logit`.
- **Marginal effects and discrete changes after the conditional logits.** `clogit_effects(fit)` (also
  `fit.marginal_effects()`): after `choice_logit`, own and cross derivatives of each alternative's probability in each
  attribute (beta P_j (delta_jk - P_k)) and in each characteristic of the case (P_j (gamma_j - sum_k P_k gamma_k)),
  with elasticities, averaged over the cases (`at="average"`) or at the means (`"mean"`), delta-method standard errors
  and tables alternatives by alternatives (`.table("cost")`, `.table("cost", "elasticity")`). With `dummy=True` and
  `changes={"cost": 1}`, beside the derivatives, the change in each alternative's probability when a 0/1 variable goes
  from 0 to 1 (`dummy=True` finds them, as statsmodels' `get_margeff(dummy=True)`) or a variable from its value to
  the value plus a chosen amount (`changes`); for an attribute, one alternative's value at a time (own and cross
  changes, alternatives by alternatives: `.table("wifi", "discrete change")`), for a characteristic of the case, the
  case's; computed from the probabilities themselves, with delta-method standard errors. They replace the
  derivatives in the summary and get a view per variable. After `conditional_logit` the probabilities depend on the
  fixed effects, so it gives odds ratios with standard errors and, with `fixed_effect=` (e.g. 0), effects and
  discrete changes for a unit with that fixed effect, labelled as such. Dialog **Estimate → Microeconometrics →
  Conditional logit marginal effects…** (with **0/1 variables** and **Changes**), and **Next ▸ Marginal effects…** on
  a conditional logit's window; the choice logit's window has a Marginal effects tab.

### Finance plugin

- **New finance dialogs.** **Finance → Black-Litterman…** (views written as `a - b = 0.002`, their variances, tau),
  **Bond measures…** (price, yield, duration, convexity from zero rates), **Heston option price…**, **Merton at a
  point…** and **Copula portfolio VaR…** (weights for a fitted copula; also a copula window's Next step).
  `black_litterman` takes a pymetrix table as the covariance (`data.cov([...])`), `bond_measures` takes lists and is
  exported, and these, `heston_price`, `merton_point` and `FittedCopula.portfolio_var` are recorded calls. Results
  that are a dict of numbers or a tuple of arrays show as tables.
- **Black-Litterman returns a result.** `black_litterman(...)` gives a `BlackLittermanPosterior`: the implied
  equilibrium returns (`prior`), `posterior_mean` and `posterior_cov` labelled by asset (from the covariance's names,
  e.g. `data.cov([...])`), the views labelled as written (`a - b`) with `q` and `omega` (`confidences`), the
  unconstrained optimal weights `(delta Sigma)^-1 mu` (`weights`), `summary()`, `views_table()`, a prior-versus-posterior
  bar chart and provenance. Its window has Summary, Views, Graph, Posterior covariance, View weights and Provenance in
  place of "Item 0" and "Item 1". It still unpacks as `mu, sigma = black_litterman(...)` (and `bl[0]`, `bl[1]`), with
  the same numbers.

### App & figures

- **Edit → Redo.** Next to Undo: Ctrl+Y or Ctrl+Shift+Z (and the platform's own Redo keys) run the step undone again,
  as recorded, with the recalculation after it; the menu names the step. Several undone steps redo in order; a new
  step clears what can be redone, and Redo refuses, saying why, once objects have changed since the Undo.
  `session.redo()`, `redo_label()` and `redo_blocked()` in code. Figure edits are steps, so Undo and Redo cover them
  too, including those made on the figure.
- **Arrows, text boxes and change markers on figures.** Edit… → **Add an arrow…** (style ->, -|>, <->, fancy or
  wedge; straight or curved; colour, width, head size; optional words) and **Add a text box…** (round, square,
  sawtooth or no box; fill and edge colours, padding, font size, alignment), recorded as `annotate` and `text` with
  `arrowprops` and `bbox`; **Draw an arrow on the figure** takes its ends from two clicks. **Mark a change…** (or two
  clicks on a series) draws an arrow or bracket between two points labelled with the change, the percent change or
  both, in percentage points for a rate, optionally with the rate a year: `fig.change(series, start, end, show=,
  units=, annualise=)`, recomputed from the data whenever the figure is drawn and written out by
  `matplotlib_code()`.
- **Edit notes, arrows, lines, bands and change markers on the figure.** Click one to select it (handles show),
  drag it or a handle (an arrow's head or tail, a band's edge, a change marker's end) to move it, with Shift to keep
  it level or upright; dates snap to periods and change-marker ends to the series' data points (the label is
  recomputed). Delete removes it, a double-click edits its words, Esc puts a drag back. Each edit is recorded as the
  step's new arguments (Change a style, Recorded styles), so it replays, exports and appears in reports; the
  handles never do.
- **Click on a chart to see what is there.** In a figure window, a click on a series (line, step, area, band, bar
  or scatter points, within a few pixels) shows a small box with its name; a click on or near one of its data
  points shows the period as the axis writes it (2020Q1) and the value, with the point circled. Bars give their
  period or category and value, heatmap cells their row, column and value, in every panel; 3-D panels are left out
  (a drag turns them). Empty space or Esc takes the box away. It is drawn on the window only: never recorded as a
  style step, and not in saved or copied images or reports. The running x, y readout as the mouse moves (the
  toolbar's and the line under the figure) is now off; **Values** on the figure's toolbar turns the value under the
  mouse back on.
- **Small windows scroll instead of overlapping.** A result window made small no longer draws its buttons over the
  figure or table: the buttons under a figure or table, and those beside the tabs (Copy summary, Next ▸, Add to
  report, the dataset's status line), fold into a **»** menu when there is no room, and come back when there is.
  Below a figure's readable size (320 by 220) the window scrolls, sideways when narrow and up and down when short,
  each on its own; above it the figure still follows the window. Table tabs, the dataset window's series browser,
  File → Browse series… (its buttons fold into » too), the model editor's Solution tab and the Start window scroll
  the same way, and the spreadsheet's edit line no longer runs over its ✓ and ✗ buttons. A long command dialog keeps
  its size on a normal screen and scrolls on a small one (1024 by 600); Help, Plugins, Keyboard shortcuts, the
  command palette, course folders and the app's window open no bigger than the screen.
- **Window fit holds with other fonts and styles.** A form whose fields need more than its set minimum width (a
  command dialog's 420 pixels, with a wider font or another style) is no longer squashed to that width in its scroll
  area with no scroll bar: the content keeps its layout's minimum, re-checked whenever the layout changes (a field
  shown, a longer choice, a tab's first showing), and the dialog's least size grows with it. The button row under a
  figure or table folds again when a button's size hint changes (its text, font, zoom or style), not only on resize.

### Extension API

- Session observers hear of a redone step (Edit → Redo, `session.redo()`) as a `step` event, as of any step run;
  the extension API is unchanged.
- **Course reference cells list classes made through renamed arguments.** `write_api_reference` traced a call such as
  `Scenario()`, whose `__init__` carries `@renamed_arguments`, as `Scenario.__init__` and then dropped it as private, so
  the class never reached the lab's reference cell (the crash on such calls was fixed earlier). `stats.available_tests`
  and `stats.run_test` gained the docstrings `check_api_reference` needs. Reported by the monetary-model-laboratory
  course pack.

### Fixes

- **Fixed, a plugin that is installed but cannot be imported now says why and what to do.** "could not be imported:
  No module named 'pymetrix_finance'" told nothing of the usual cause: the plugin's metadata is installed but its code
  is not, typically an editable install (`pip install -e packages/...`) whose folder was moved, deleted or checked out
  at a branch without it. The Problems line now says so in one line; **Help → Plugins** (the plugin's details) and
  **Check plugin** give the package and version, whether it is an editable install and from which folder (and whether
  that folder exists), the Python in use, and the command to reinstall or remove it with that Python (**Copy command**
  copies it). A missing library the plugin imports is told apart: "the finance plugin needs *module*, which isn't
  installed", with the pip command (the package name from the plugin's requirements when they name it). A plugin that
  failed this way is listed once, by its package name, and is now shown in Help → Plugins. New:
  `pymetrix.plugins.import_problem` and `problem_detail`.
- **Fixed, the Conditional forecast dialog's conditions:** a condition on a variable for every period and another on
  the same variable for some periods (`pi[2026Q1]=3, pi<=4`) kept only the one written last, silently. Mixing the two
  for one variable is now refused with a message; write them all for the periods they cover.
- **Fixed, an unknown function in an expression now suggests pymetrix's name for it.** `diff(log(y))` in Data →
  Generate series gave "unknown function" and a list of 120 names. Other packages' names (`diff`, `ln`, `lag`,
  `lead`, `pct_change`, `std`, `mavg`, ...) now say what to write instead ("did you mean d(x) (the first difference;
  ...)?"), and a close misspelling names the function (`movavgg` → `movavg()`); the list follows.
- **Fixed, Test → Run test… with a required setting missing names it.** The ADF test with no variable under Test
  settings failed with Python's "ADF.__init__() missing 1 required positional argument: 'variable'". `run_test` now
  checks the test's required settings first and says "the adf test needs variable: give variable=... (in the Run test
  dialog, under Test settings)", in the dialog and in code.

## 0.1.0a10: private review build (November 2026)

- **Copulas: BB6, BB8 and Tawn pairs, a goodness-of-fit test for vines, DCC copulas, the multiplier bootstrap**
  (finance plugin).
  - New bivariate families: `bb6` (Joe-Gumbel), `bb8` (Joe-Frank), Tawn's asymmetric extreme-value copulas `tawn1`
    and `tawn2`, and the rotations of BB1, BB6, BB7, BB8 and Tawn (`bb1_180`, `bb7_90`, `tawn1_270`, ...): density,
    CDF, h-functions and inverses, Kendall's tau, tail dependence, simulation and maximum likelihood, in `copula`,
    `compare_copulas` (when listed) and as vine pair copulas (`families=[...]` or `families="all"`; VineCopula's
    codes 8, 10, 17-40, 104-134, 204-234 read and written). The default family sets are unchanged.
  - `vine_copula(..., gof=True)` and `FittedVine.goodness_of_fit()`: the vine's Rosenblatt transform
    (`FittedVine.rosenblatt()`) tested against independence by VineCopula's ECP2 Cramer-von Mises statistic, with a
    parametric bootstrap p-value that refits the vine's parameters on each sample.
  - `dcc_copula(data, series, family="gaussian" | "t")`: a Gaussian or t copula of two or more series whose
    correlation matrix follows Engle's DCC recursion on the copula's scores (mgarch's recursion, now also as
    `mgarch.dcc_path_linear`); maximum likelihood with standard errors, the constant copula's likelihood beside,
    filtered correlations, forecasts, forward simulation and portfolio VaR and ES. New dialog **Finance → DCC
    copula…**.
  - `copula(..., gof=True, bootstrap="multiplier")` and `compare_copulas(..., bootstrap="multiplier")`: Kojadinovic
    and Yan's (2011) multiplier bootstrap for the Cramer-von Mises test, as R's `gofCopula(..., simulation = "mult")`
    (not for the t copula). The copula dialogs take the option; the vine dialog takes the goodness-of-fit test.
  - `FittedVine.matrix()` and `to_vinecopula()["matrix"]` have zeros above the diagonal (they had ones), as R's
    `RVineMatrix` requires.
  - Checked against R's VineCopula 2.6.1 (BiCopPDF, BiCopCDF, BiCopHfunc, BiCopPar2Tau, BiCopPar2TailDep,
    RVineLogLik, RVinePIT, RVineGofTest ECP2) and copula 1.1-7 (gofCopula's multiplier statistic and p-value), the
    values written into the tests; by integration, finite differences, simulation, parameter recovery, size and power
    studies, and loops written in the tests. Existing results are unchanged.

- **FRB/US worked example: stochastic simulations (stochsim.prg) and optimal-control policy (ocpolicy.prg).** The
  tutorial ([docs/tutorials/frbus.md](docs/tutorials/frbus.md), sections 11 and 12) runs both programs of the
  Federal Reserve Board's public FRB/US package under VAR expectations.
  - stochsim.prg: 1000 draws over 24 quarters, each quarter's shocks one historical quarter's de-meaned errors of
    the 51 stochastic equations (1970Q1-2014Q4), the first quarter's halved, with the funds rate floored at 0.25.
    About 105 s. The floor binds at least once in six years in 16 percent of the draws.
  - ocpolicy.prg: the funds-rate path (`rfftay_aerr`, 60 quarters) that minimises the discounted squared
    unemployment gap, inflation gap and rate changes over 80 quarters. 3 Gauss-Newton iterations, 12 s; the loss
    falls from 11.76 under the Taylor rule to 10.81.
  - `Model.solve(errors=<DataFrame>)`: a sample of residuals given by equation (as `Model.add_factors` returns them
    for a historical span), resampled a whole period at a time; the equations shocked are its columns, fitted or
    not. `error_scale={"2026Q2": 0.5}` multiplies the errors of some periods. `workers=n` solves the draws in `n`
    processes, the errors drawn first so the draws are the same.
  - `Simulations.errors`, `Simulations.error_names` and `Simulations.error(equation)`: the errors each draw used.
  - Stochastic simulation of backward-looking models evaluates the compiled equations for all the draws at once and
    solves each period first by the chord method with the draws' mean Jacobian, then by Newton's method for the
    draws that need it. FRB/US: 500 draws over 40 quarters in 96 s.
  - `optimal_control(model, data, span, instruments, loss, desired=, control=, discount=)` and `OptimalControl`: the
    paths of exogenous instruments that minimise a discounted quadratic loss, the model (backward-looking) solved as
    the constraint, by Gauss-Newton with the derivatives from the model linearised along its solution (the method
    of the Fed's MCE Solve Users' Guide, section 5).
  - `EViewsXmlModel.stochastic`: each equation's `<stochastic_type>` in an FRB/US XML model file.
  - Checked (`tests/test_frbus_example.py`):
    - Stochastic simulation: with zero errors every draw is the baseline (1e-8 relative). The draws are
      reproducible by seed and generator. Each quarter's errors are exactly one historical quarter's. The add
      factors a draw's solution implies are the tracking ones plus its errors (1e-6). The median stays within 0.1 of
      the baseline for the funds rate, unemployment and inflation. On y = 0.5 y[-1] + e, the simulated variance is
      within 5 percent of the analytic one (2.3 percent found).
    - Optimal control: on a small linear-quadratic model, the path equals `optimal_policy`'s commitment solution
      (1e-6; 5e-15 found). On FRB/US, the gradient at the optimum is 2e-7 of its size at the baseline, and the
      gradient equals central differences of the loss (1e-4; 1e-6 found).

## 0.1.0a10: development build (1 November 2026)

- **Regimes: surprises with perfect foresight and OccBin, constraints that change, identification.**
  - A surprise regime (`Regime(..., expected=False)`, or `surprise` after its start in text: `"2026Q1 surprise:
    phi_pi=2"`, which the Solve and Forecast dialogs take) now works with perfect foresight and OccBin, which refused
    it before. The path believed before the surprise is kept until it starts, and the model is solved again from the
    state reached, the regime believed to hold for good. Several surprises are solved one after another, and they
    combine with unanticipated shocks. On a model whose equations are linear, perfect foresight gives the first-order
    solution with the same surprises. OccBin with shocks known in advance solves again when each surprise starts; the
    OccBin filter, estimation and forecasts take surprise regimes too. OccBin forecasts that hold targets refuse them.
  - OccBin regimes may now switch a model's max or min off for a while, rewrite it, or bring one in (refused before).
    The guess-and-check runs over the constraints of all the regimes, matched by the equation each is in. A constraint
    a period's regime lacks is slack there.
  - Identification of a model estimated with regimes (refused before) differentiates the observables' means, variances
    and autocovariances of every period of the sample, through the regimes' time-varying rules (`bayes.regime_moments`).
    With regimes equal to the base it is the check without them, exactly. `Model.identification` and
    `Template.identification` take `regimes=` and `span=` to run the same check before any data exist; a parameter of
    a regime that starts after the sample is reported as not identified.

- **PSETS databases as `psets://` addresses (the `pymetrix-psets` plugin).** `Dataset.read("psets://forecasts/CPI.M")`
  reads series from a PSETS database through PSETS's read-only plug-in API (`psets` 1.x).
  - **What it reads:** each series with its metadata, the publisher's release as its vintage, and a reference back
    (`psets://forecasts/CPI.M@2024-08-15`). The figures as released on any earlier date come from
    `vintage="2024-07-31"`, and PSETS's own downward conversion from `?freq=Q&agg=avg`.
  - **What it warns about:** open publisher issues PSETS knows about.
  - **Backtests:** `psets_vintages` builds real-time data for them.
  - **Where it lives:** the earlier sketch (`examples/plugins/pymetrix-psets`) is now a package of its own,
    `packages/pymetrix-psets` (`pip install "pymetrix[psets]"`; PSETS installs from its repository).
  - **Testing:** against a stand-in for `psets` in CI, and against PSETS itself, with a real database built by its
    migrations, wherever PSETS is installed.

- **Who acted, in the lab book.** Session events now say who took each step (`SessionEvent.actor`, extension API
  1.5): the person at the machine, through the app or Python, or an AI agent through the MCP server, named by its
  client. `pymetrix-labbook` records it in every entry, hashed with the rest, counts each actor's steps in `verify`
  and lists them in `show`. The lab book stays a separate add-in (`pip install pymetrix[labbook]`).

- **A new icon, and a shortcut that runs the latest build.** The icon is the first step from data to information: a
  field of data points and a magnifying glass that finds f(x), ink on paper with the x in amber
  (`scripts/make_icon.py` draws it, with simpler 16 to 24 pixel versions); the opening screen takes its colours.
  `pymetrix-gui --latest` updates a git checkout (`git pull --ff-only`) before pymetrix loads, saying on the opening
  screen what it did; offline, with local commits in the way, or installed from a wheel, the app starts as it is.
  `pymetrix-gui --shortcut --latest` makes a desktop shortcut that does this each time.

- **FRB/US worked example: the zero lower bound and liftoff thresholds (example3.prg), and a new policy rule
  (example4.prg).** The tutorial ([docs/tutorials/frbus.md](docs/tutorials/frbus.md), sections 9 and 10) runs both
  programs of the Federal Reserve Board's public FRB/US package.
  - example3.prg: demand and risk-premium shocks from 2003Q3 under VAR expectations and the Taylor rule, with the
    funds rate floored at 0.125 (`rffmin`) and liftoff held back until unemployment falls below 7 or expected
    inflation rises above 3 (`dmptrsh`, `lurtrsh`, `pitrsh`). Newton solves the `max` kinks, 24 quarters in about
    1 s. The floor binds from 2003Q3 to 2005Q2 and from 2008Q4; without it the rate goes to -5.2. The thresholds
    delay liftoff by one quarter, to 2005Q4.
  - example4.prg: a first-difference rule (`rffgen`) with two new model-consistent expectations (`zgap3`, `zpic43`),
    selected by `dmpgen` and added by building a `Model` from the edited equations. In the model of October 2026,
    `rffrule` has lost its `dmpgen*rffgen` term, so the term is added back. Each 240-quarter solution takes about
    7 s.
  - `Newton` (period by period): a derivative that does not evaluate is replaced by a forward difference. This
    covers the overflowing `exp` of a saturated logistic switch, such as FRB/US's thresholds parked at -9999 and
    9999, whose analytic derivative is inf/inf. Before, Newton failed with a non-finite residual.
  - `PerfectForesight`: the line search and the chord steps judge a step by each equation's residual relative to the
    size of its terms (the largest |derivative x value| in its row of the Jacobian, at least 1), not by the raw
    norm. Convergence is still judged on the residuals as written. Before, a 1 percent consumption shock in FRB/US
    stalled: the Newton step left second-order residuals of about 0.7 in identities with levels in the thousands,
    and the steps were cut until the iterations ran out. It now converges in 2 iterations, and the funds-rate shock
    of example2.prg takes 6 s and 2 iterations instead of 11 s and 3.
  - Checked (`tests/test_frbus_example.py`, with the packages):
    - The floor: never breached (1e-8), and binding where the rule is below it. Every equation holds in every
      quarter (1e-8). Gauss-Seidel (damped by 0.5) agrees with Newton (1e-8).
    - The switches, quarter by quarter: `dmptlur` above 1/2 exactly when unemployment is below 7, `dmptpi` exactly
      when `zpic58` is above 3, and `dmptr` the max of `dmptmax` and its lag. The funds rate stays at the floor until
      the quarter after the switch.
    - The appended rule holds along the solution. With the inertial Taylor rule's coefficients it gives the
      `dmpintay` solution (1e-8 relative to the levels; 1.3e-12 in the funds rate, yield, unemployment, inflation
      and output gap). The new rule raises the funds rate to 0.50 after three quarters, against 0.20.
    - Without the packages: a saturated logistic switch solved by Newton, and a stacked level identity converging in
      2 iterations.

- **FRB/US worked example: the Fed's simulations in pymetrix, VAR and model-consistent expectations.** A tutorial
  ([docs/tutorials/frbus.md](docs/tutorials/frbus.md)) runs the example programs of the Federal Reserve Board's
  public FRB/US package with pymetrix. It covers the baseline tracked with add factors from LONGBASE, a funds-rate
  shock (example1.prg), a federal-purchases shock of 1 percent of GDP (pings.prg), a switch of policy rule
  (`dmpex`, `dmptay`), and the model-consistent-expectations version (asset pricing and price-wage setting,
  example2.prg) solved by `PerfectForesight` over 240 quarters against VAR expectations. Figures are made with
  `pymetrix.figures`. The model, equations and data are the Federal Reserve's public packages and are not shipped with
  pymetrix: the tutorial and the tests read them from `PYMETRIX_FRBUS` and `PYMETRIX_FRBUS_DATA`.
  - `Model.add_factors(data, span=None)`: tracking add factors, by equation name, each equation's residual at the
    data (as EViews's `addinit`). With `Scenario(adjust=...)` every solver reproduces the data.
  - Period-by-period solvers (Newton, Gauss-Seidel) evaluate the compiled equations (`pymetrix.models.fast`), the
    residuals and the Jacobian each in one call per period, with the interpreter kept as the reference. They also run
    with one BLAS thread, as estimation does. FRB/US's 284 equations over 40 quarters by Newton now take 1.7 s; they
    took 4.1 s with one BLAS thread and 104 s with the library's default threads on the test machine.
  - Perfect foresight at FRB/US's scale: the stacked Jacobian is factorised with the periods kept in time order
    (natural ordering, its fill-in confined to the band of lags and leads), not reordered by COLAMD. Each Newton
    iteration's factorisation is reused for further steps while each cuts the residuals tenfold. Derivatives are no
    longer also taken symbolically when the compiled ones are used. The 284-equation model-consistent model over 240
    quarters (68,160 unknowns) takes 16 s and 4 iterations, against 169 s and 6 iterations before, with a peak memory
    of 0.83 GB against 0.94 GB; over 80 quarters it takes 5.1 s against 22.7 s. Results are unchanged: the existing
    perfect-foresight tests pass as they were.
  - Checked (`tests/test_frbus_example.py`, with the packages):
    - VAR tracking: the baseline is reproduced from blanked values to 2.1e-8 relative.
    - Solvers: Newton and Gauss-Seidel agree on a funds-rate shock (6e-10 relative).
    - Funds-rate shock: the output gap's trough is -0.50 in the 8th quarter.
    - Federal purchases: 0.80 points of output gap per point of GDP on impact, turning negative in the third year,
      against about 1.0 and the third year in the Fed's own figure (Brayton, *A New FRB/US Price-Wage Sector*,
      Figure 3, p. 16). A doubled shock is within 0.7 percent of twice the response for the output gap, and 3 percent
      at worst, for inflation.
    - Policy rule: the funds rate held raises output more (4.5 against 3.5 point-quarters over two years).
    - Model-consistent expectations: the baseline is reproduced from blanked starting values to 8e-9 relative, and
      every equation holds in every period to 1e-8. The funds-rate shock moves the 10-year yield by 0.15 against
      0.33 under VAR expectations.
    - Without the packages, the excerpt in the repository is tracked by Newton, Gauss-Seidel and perfect foresight
      (1e-9), and the compiled period solvers equal the interpreted ones (1e-11, same iterations).

- **EViews syntax in the model editor; FRB/US's XML model files and state-space objects.** `ModelSource(text,
  "eviews")` and `read_eviews(text, params=, data=, span=)` read an EViews model's text (what its Text view shows and
  `m.append` takes: equations with lags `x(-1)` and leads `x(+1)`, `@identity`, coefficients `c(1)` and named vectors
  `beta(1)` as parameters `c_1` and `beta_1`, `@add` add factors, `@innov` shocks, `:eq` linked equations) and a
  program subset (`model m`, `m.append`, `m.merge`, `equation eq.ls` specifications fitted to `data=` by least squares
  on their `smpl`, `scalar`, `coef`, `param`, `c(1) = ...`, constant series as values and guesses) into a `Template`.
  Functions: `log`, `exp`, `abs`, `sqr`, `@inv`, `d(x, n, s)`, `dlog(x, n, s)`, `@pch`, `@movav`, `@movsum`,
  `@movmax`, `@movmin`, the normal, t, chi-squared and F distribution functions, and `@recode(a > b, a, b)` as
  `max`/`min`; everything else (`@pchy`, `@elem`, `@trend`, other `@recode` forms, EViews's programming language,
  other estimation methods) refused at its line and column, workfile and display commands ignored with a note. The
  editor checks the text as typed (problems placed, unknown names with suggestions, symbols with roles, completions
  and hover help), highlights it, and **Write as → EViews** writes any template, Dynare or IRIS model as a program
  (or the model's text alone: `export_model(..., "eviews", form="text")`), shocks that end an equation as its
  `@innov`, with notes on what does not carry across (measurement, priors, regimes, reported series, descriptions,
  other shocks' sizes; names renamed for EViews's reserved words). `read_eviews_xml` reads the XML model format of
  the Federal Reserve's public FRB/US package (`mods/model.xml`: equations, coefficients, definitions, sectors, the
  model-consistent-expectations versions with `mce=`), and `read_eviews_sspace` an EViews state-space object
  (`@signal`, `@state`, `@ename`, `@evar`, `[var = ...]`) as a `StateSpace`, where it maps. Written without EViews or
  its documentation to hand (the network was closed): the subset is kept conservative, and the documentation says so.
  - Checked: a small macro model in EViews's text against the same Model written by hand (equations, parameters and
    Newton path, 1e-12); its program form's specifications fitted against the pymetrix Equations fitted directly
    (1e-10); every function against the pymetrix expression or arithmetic it stands for (1e-12); round trips EViews
    -> pymetrix -> EViews and the templates -> EViews -> pymetrix (steady states and responses); planted problems
    each once at their line and column. FRB/US: all 309 equations of model.xml (284 standard, 25 model-consistent)
    read, each with the same residual as the Fed's own Python version of it (1e-13 at worst, on random data); the
    284-equation standard model compiled, and with add factors computed from LONGBASE it reproduces the baseline
    when solved by Newton over the 40 quarters after the history ends (2.1e-8 relative at worst, `fynin`; behind
    `PYMETRIX_FRBUS` and `PYMETRIX_FRBUS_DATA`, as the packages are not in the repository); a state-space object
    against the `StateSpace` written by hand (log likelihood, 1e-10), and FRB/US's supply-side `estimation_code.prg`
    read as far as it maps (25 states and 4 of 8 signals; the regressions in signals and a variance in a program
    scalar refused where they are).
  - Confirmed and widened against J.-L. Brillet's *Structural Econometric Modelling: Methodology and Tools with
    Applications under EViews*: comparisons in brackets (`(t <= 2004)`), `@recode` in general, `@pc`, and with the
    data's frequency (`frequency=`, `wfcreate`, a dated `smpl`, the data) `@pchy`, `@pcy`, `@pcha`, `@pca`; `@trend`,
    `@seas` and `@elem` as series made from the data's dates (`eviews_series`), `@elem(@trend(...), date)` as its
    number; `scalar f` without a value. The book's very basic model and estimated small model read without refusals
    and solve; its 83-equation model reads and compiles. Still refused: moving variances and standard deviations,
    `and`/`or`, `@add(v)` inside a function, the programming language.

- **Vine copulas, time-varying copulas and goodness of fit in any dimension (finance plugin).** `vine_copula(data,
  series)` fits an R-vine of pair copulas (C- and D-vines with `structure="cvine"`/`"dvine"`, a path or roots by
  `order=`, any R-vine by a VineCopula `matrix=`): Dissmann et al.'s (2013) maximum spanning trees of |Kendall's tau|,
  each edge's family by AIC or BIC among the plugin's bivariate families, their rotations and independence (with an
  optional independence test), estimated tree by tree from the h-functions' pseudo-observations and optionally all
  together (`method="joint"`); `trunc_level` makes the higher trees independence. A fitted vine has its log
  likelihood, AIC and BIC, density (`logpdf`), simulation by the inverse Rosenblatt transform, a summary of each
  edge's family, parameters, Kendall's tau and tail dependence, the R-vine matrix in VineCopula's codes, and portfolio
  VaR and ES through the margins (ranks, fitted or GARCH, as for `copula`). `tv_copula(data, [a, b])` is Patton's
  (2006) time-varying copula: the Gaussian or t (constant degrees of freedom) correlation by his modified-logistic
  ARMA-type recursion, or the Clayton, Gumbel or symmetrised Joe-Clayton tail dependence by his logistic one; by
  maximum likelihood with standard errors, the filtered path as series, next period's forecast, forward simulation
  and portfolio VaR. The Cramer-von Mises goodness-of-fit test now takes the Gaussian and t copulas of more than two
  series (their CDFs by Genz's separation of variables over a fixed scrambled Sobol sample; the parametric
  bootstrap as before). In the app: **Finance → Vine copula…** and **Finance → Time-varying copula…**. Existing
  copula results are unchanged.
  - Checked: against R's VineCopula 2.6.1 (numbers written into the tests): the h-functions of every pair family
    (1e-9), RVineLogLik at given parameters (1e-8), RVineStructureSelect's trees and families exactly with the
    parameters within 1e-3 (R-vine, C-vine, BIC with truncation), RVineSim's pairwise Kendall's taus within three
    standard errors; Gaussian D-, C- and R-vines with partial correlations equal to the Gaussian copula (1e-10);
    simulated taus and recovered sequential parameters within three standard errors. Patton's path against a loop
    written in the test (1e-12), the constant copula when alpha = beta = 0 (1e-12), simulated parameters recovered
    within three standard errors. The d-dimensional CDF against R's mvtnorm (2e-5; 2e-6 with more points) and
    scipy, S_n against R (relative 2e-3), the test's size on null data and power against a wrong family.
    Not done: goodness of fit for vines, time-varying vines or copulas of more than two series (DCC is in
    `mgarch`), the multiplier bootstrap ([docs/known-gaps.md](docs/known-gaps.md)).

- **A choice of random generator.** `set_random_generator("mt19937")` (also `"philox"`, `"sfc64"`; `"pcg64"`, NumPy's
  default, is the default), `get_random_generator()` and the `random_generator(...)` context manager choose the bit
  generator under every random draw pymetrix makes: stochastic simulation, bootstraps, Metropolis-Hastings chains,
  SMC, Monte Carlo replications, sign-restriction rotations, copula draws in the finance plugin. Every place that
  made a NumPy Generator or SeedSequence now goes through one helper, and the setting reaches chains, particle
  blocks and replications run in other processes. Results record it beside the seed (`provenance.random_generator`,
  shown in the Provenance view), and a project saves it with each step run with another generator and replays the
  step with it whatever is chosen when it is opened (its script sets it; a generator the opening version lacks is
  warned about). In the app: View → Random generator, remembered between sessions. The Mersenne Twister does not
  reproduce MATLAB's or Dynare's draws (NumPy seeds it and makes normal draws differently); the `arch` bootstraps,
  statsmodels' Markov-switching starts and scikit-learn's `random_state` use `RandomState`, already MT19937, and do
  not change. With the default every result is as before, bit for bit.
  - Checked: raw draws of each generator against NumPy's Generator over that bit generator called directly (integer
    seeds, seed lists, spawned streams; exact), the default against `np.random.default_rng`; a scan of the source
    for any other constructor; with MT19937 chosen, seeded simulation, a VAR bootstrap, MCMC, SMC and a copula
    simulation build MT19937s and never call `default_rng`, repeat for the same seed and differ from PCG64; chains in
    two processes equal one; under the default, the same features bit-identical to before the change; provenance
    through a save and a load (old saves read as PCG64); a project round trip (check_project) opened under another
    setting; unknown names refused.

- **Co-skewness and co-kurtosis.** `Dataset.coskewness(names, standardized=True, form="matrix")` and
  `Dataset.cokurtosis(...)` give the standardized co-moments used in portfolio choice, over the periods every series
  has in the active sample: S_ijk = mean(e_i e_j e_k) and K_ijkl = mean(e_i e_j e_k e_l), e the deviations from the
  means divided by the standard deviations (moments over T), as the matrices M3 = E[e (e ⊗ e)'] (n x n², columns
  `"j,k"`) and M4 (n x n³), or with `form="pairs"` the n x n tables mean(e_i² e_j) and mean(e_i³ e_j);
  `standardized=False` gives the co-moments in the series' units. The diagonal is each series' skewness and (not
  excess) kurtosis with the T denominator; `describe()` keeps its bias-corrected skewness and excess kurtosis, and
  the notes give the conversion. `Dataset.systematic_comoments(names, market)` gives each series' beta, co-skewness
  and co-kurtosis with a market series and its co-skewness and co-kurtosis betas. Data → Statistics → Correlations…
  offers all three, recording the call.
  - Checked: every element against means of products of deviations computed by hand (1e-12), both forms,
    standardized or not; M3 and M4 unchanged by every permutation of their indices; the diagonal against scipy's
    skew and kurtosis (bias=True) and, converted, against `describe()` (1e-12); the standardized versions unchanged by
    location and scale; the systematic co-moments against their definitions and beta against a least-squares slope;
    the dialog's recorded code reproducing its tables, and a project round trip.

- **Regimes with OccBin, perfect foresight, posteriors and decompositions; correlated shocks with OccBin.** Known-date
  regimes now reach the nonlinear solutions: `Model.solve(..., solver=PerfectForesight(), regimes=)` switches the
  stacked system's equations at the regimes' dates (the nonlinear model's own, the terminal steady state the last
  regime's), and `occbin(regimes=)` (periods numbered from 1: `Regime(5, {...})`), `occbin_filter(regimes=)`,
  estimation with `occbin=True` and regimes, and OccBin forecasts combine them with the occasionally binding
  constraints: each period's matrices come from its known regime and the constraint's guessed regime, and each
  period's max or min is checked with its regime's parameters. Both take regimes known in advance; with OccBin a
  regime keeps the model's constraints. A posterior's forecasts (`Posterior.forecast(regimes=)`) solve, filter and
  forecast each draw with its own values of the regimes' parameters; `FittedModel.decompose` and the new
  `RegimeFilter.decompose` and `LinearSolution.decompose(regimes=)` go through the regimes' time-varying rules, their
  constants a part of their own (`"regimes"`), the parts adding up to the path; `FittedModel.moments(regime=)` and
  `fevd(regime=)` give one regime's own stationary moments (the identification check still refuses regimes).
  Correlated shocks now work with OccBin: the OccBin filter and smoother with the full covariance (the smoothed shocks
  the likeliest given the states), estimation with the OccBin filter and `corr_` parameters, OccBin forecast draws
  through the Cholesky factor, and decompositions of the shocks orthogonalised by Cholesky as impulse responses are
  (`decompose(orthogonal=False)` for the shocks' own). The Solve dialog takes regimes with perfect foresight and the
  Forecast dialog with a posterior. Without regimes or correlations every result is as before, bit for bit.
  - Checked: perfect foresight with five regimes against the time-varying first-order solution on a linear model
    (1e-8), and the nonlinear growth model against stacked Newton with its parameters as series in the data (1e-8);
    OccBin with regimes against perfect foresight of the model with its max and the same regime dates, anticipated
    shocks and repeated surprises (1e-8); the OccBin filter with correlations against the linear Kalman filter when
    nothing binds (1e-8), and with regimes and correlations on data where the floor binds against statsmodels given
    time-varying matrices built by hand (1e-8); correlated OccBin draws' sample covariance; each posterior draw
    against the regime forecast worked out again at its parameters (1e-10); regime decompositions adding up (1e-10)
    and equal to the linear one when the regimes equal the base (exact).

- **More copulas in the finance plugin, with fitted margins and model checking.** `copula` gains the Frank and Joe
  families, and Clayton, Gumbel, Frank and Joe now take any number of series (exchangeable: one parameter for every
  pair); BB1 (Clayton-Gumbel), BB7 (Joe-Clayton), the survival copulas `clayton180`, `gumbel180`, `joe180` and the
  rotations `clayton90`, `clayton270`, `gumbel90`, ... (negative dependence; Frank takes it directly) for two. Each
  family has its density, CDF, h-functions and their inverses, Kendall's tau both ways and its tail dependence;
  draws are Marshall-Olkin (gamma, positive stable, logarithmic-series and Sibuya frailties) or by conditional
  inversion. Margins are the ranks (as before) or fitted: `margins="normal"`, `"t"` or `"skewt"` (Hansen's skewed
  t), and with `garch=True` the GARCH's own normal or t errors; `method="ml"` (IFM with fitted margins), `"itau"`
  (Kendall's tau) or `"joint"` (margins and copula together, from IFM). Every fit has standard errors from the
  observed information (IFM's ignore the margins' estimation; joint's do not), log likelihood, AIC and BIC; `gof=True`
  or `fit.goodness_of_fit()` adds the Cramer-von Mises test against the empirical copula with a seeded parametric
  bootstrap p-value (Genest, Remillard and Beaudoin 2009); `compare_copulas(data, series, families=...)` fits several
  families to the same data side by side. `portfolio_var` maps draws back through the fitted margins. In the app the
  Copula dialog has the families, margins, method and goodness of fit, and **Finance → Compare copula families…** is
  new. Results of the Gaussian, t, Clayton and Gumbel copulas with rank margins are unchanged (to the last digit).
  - Checked: every family's density, CDF, h-function and inverse against R's copula 1.1-7 and VineCopula 2.6.1
    (1e-8 to 1e-9; values written into the tests); densities integrating to one and equal to the CDF's mixed
    derivative; Kendall's tau against closed forms, Frank's Debye series, Joe's series, BB7's Beta form and 200,000
    draws; tail dependence against the CDF's limit and simulated exceedances; Marshall-Olkin against conditional
    inversion; parameters recovered within three standard errors; R's fitCopula and gofCopula statistic (1e-6);
    the test's size on null data and power against a wrong family; Hansen's skewed t by quadrature; IFM, joint and
    GARCH-t margins recovered from simulated data; portfolio VaR against a direct simulation written in the test.
    Not done: vines and time-varying copulas, the Godambe sandwich ([docs/known-gaps.md](docs/known-gaps.md)).

- **Impulse responses in the model editor; matrix expressions checked as typed; `poissinv` and `binoinv`; centred
  and expanding windows.** The model editor's Solution tab draws the impulse responses when Check solution finds a
  unique solution: a shock chooser, the variables to draw (the first six endogenous ones ticked), 20 periods,
  one-standard-deviation shocks with correlated ones orthogonalised as `LinearSolution.irf` does; a model with regimes
  shows its base regime's, and says so (`SolutionCheck.irf`, `SolutionCheck.solution`, `.shock_sd`, `.shock_corr`,
  `.regimes`). Model > Matrix > Expression… checks the expression a moment after typing stops: unknown names with
  a suggestion, constructs not allowed, parse errors with their column and evaluation errors under the field, with
  OK disabled until fixed; when it is cheap to compute, the result's shape and row and column names, or a
  decomposition's factors (`pymetrix.matrix_expr.preview`, `ExpressionPreview`, `commands.matrices.live_check`).
  `poissinv(p, lambda)` and `binoinv(p, n, p)` join the distribution functions (MATLAB's: the smallest integer k with
  cdf(k) >= p), with no derivative: linearising through one is refused with a message. Every moving function gets a
  centred form, `cmovavg(x, 5)` (x[+2] ... x[-2]; an even n one more lag than leads, as pandas' `center=True`), which
  a model reads as leads, with the trailing form's derivatives; and an expanding form, `cumavg(x)`, `cumsum`, `cumsd`,
  `cumvar`, `cummin`, `cummax`, `cumcov(x, y)`, `cumcorr(x, y)`, `cumbeta(y, x)`, within each entity as pandas'
  `expanding()`, for data only (refused in models, in differentiation and on arrays). Help, completion and hover
  in the model editor list the new functions (the expanding ones left out of model completion).
  - Checked: the preview's responses against `LinearSolution.irf` of the template's solution for every shock, with
    correlated shocks (1e-12), and the base regime's with regimes (offscreen Qt); twelve expressions typed into the
    matrix dialog, their messages and OK's state; `poissinv` and `binoinv` against scipy's `ppf` at 405
    probabilities including 0, 1 and 1e-300 for seven parameter sets (exact), the definition and the edges; centred
    and expanding windows against pandas' `rolling(center=True)` and `expanding()` per entity in a three-country
    panel with a missing value (1e-9), centred derivatives against finite differences, and the refusals.

- **Correlogram bands are ±1.96/√T throughout (behaviour change).** The approximate 95% band in the notes of
  `Dataset.correlogram`, an equation's `correlogram` (and its Correlogram tab and chart) and the band of VAR and VECM
  `residual_correlogram` (its table's count of correlations outside the band, and the dashed lines of its figure)
  were ±2/√T; they are now ±1.96/√T, as `crosscorrelogram` already was. A correlation between the two is now counted
  as outside the band. Help text and the guide say 1.96/sqrt(T).

- **A matrix window's "Copy as CSV" and "Copy as tab-separated (for Excel)", and a figure window's "Copy data", end
  lines with `\n` on every platform.** pandas' `to_csv` ends them with the platform's own line ending, so on Windows the
  copied text had `\r\n` and differed from Linux and macOS (and from tables' copies, which already used `\n`); Excel
  pastes either. The matrix copies were the one test failing main's Windows job.
  - Checked: each copy with pandas made to use `\r\n` (Windows' ending, simulated on Linux) holds no `\r` and equals
    the frame written with `\n`.

- **Regimes and shock correlations in the model text; completion and hover help in the model editor.** pymetrix's
  model syntax gets a `regimes` section, one regime a line: `2009Q1 post: phi_pi = 1.2  "post-crisis rule"`,
  `2025Q1..2026Q4: rho = 1, phi_pi = 0` (temporary), `2030Q1 alt: policy: i = 0.9*i[-1] + 0.2*pi + ei` (an equation
  replaced by name), `2031Q1 hawk surprise: ...` (`expected=False`), parameter values and equations together separated
  by `;`. And correlations of shocks in `shocks`: `corr ey, ei = 0.3  prior uniform(-0.9, 0.9)` (estimation settings
  on `corr_ey_ei`, as a shock's on its size). Both are checked as typed, each problem at its line and column
  (unknown parameters, equations and shocks with the nearest known name, periods that are not periods, regimes out of
  order, overlapping a temporary one, named twice or of another frequency, a replacement that determines another
  variable, values outside (-1, 1), correlations that are not positive definite together). They become the
  template's `shock_corr` and a new `Template.regimes`: `Template.solve()` gives a solution whose `solve(data, span)`
  follows them (`LinearSolution.regimes`), `Template.estimate` estimates with those that start inside the sample and
  keeps those after it for the fit's forecasts (`FittedModel.ahead`), and `regime_filter(template, data)` filters
  through them, now with the template's (or fit's) correlations. `Regime` gets a `description`. The pymetrix writer
  writes both back (the text round-trips); IRIS's `corr_<a>__<b>` parameters are read into `shock_corr`, checked and
  written; the Dynare writer already wrote `corr`; Dynare and IRIS have no place for regimes (a note says so). The
  model editor completes the names the text declares, equation names, functions and keywords (after two characters,
  or Ctrl+Space; not in comments or descriptions), shows a function's signature at its bracket, and on hover shows a
  name's role, value, description, estimation settings and the lines that use it, or a function's signature and what
  it does (`ModelSource.completions`, `ModelSource.hover`, `function_help`); an underlined problem still shows its
  message. A parameter a regime's equation brings in is kept by the regimes after it (`regime_model(carried=)`).
  - Checked: a text with regimes and correlations against the same model built in code with `Regime(...)` and
    `shock_corr`: the path through four regimes (1e-12), the regime filter's log likelihood on data simulated with
    correlated shocks (1e-9), an estimate and its forecast (1e-8); the text built and written back is the same text;
    seventeen planted problems each reported once at their line and column; IRIS correlations read, written and read
    again, and the Dynare text written from them read back by `read_mod` with the same correlations and prior;
    completions and hover help in code and in the editor (offscreen Qt).

- **Matrices in the app** (Model > Matrix, and a Matrix menu on every matrix's window). A named matrix is made as a
  step in three ways: from an **expression** in the package's matrix syntax using the session's objects by name
  (`(X.T @ X).inv() @ X.T @ y`, `Matrix.cov(fit)`, `V.chol(upper=True)`, `A.svd()`); **from data or a result** (a
  fit's coefficients or covariance matrix, a VAR's lag matrix or error covariance, a dataset's series over a span, a
  formula's regressors and dependent variable); or by **entering values** in a grid, typed or pasted from Excel
  (tabs, semicolons, commas or spaces; decimal commas; a first row or column of names), its rows and columns named
  by double-clicking their headers. A matrix window's Matrix menu makes a new named matrix by Transpose, Inverse,
  Pseudo-inverse, Cholesky (lower or upper), Decompose (SVD, eigenvalues, QR, LU, Schur: a table for each factor),
  Power and Solve Lyapunov (Q chosen from the session's matrices), and copies the matrix as LaTeX (a plain `bmatrix`,
  or a `tabular` with the names), CSV or tab-separated text, or exports it to Excel; a decomposition's window makes
  each factor a matrix. A fitted equation, system, VAR or VECM offers its matrices from the workspace's right-click
  menu (As matrix) and its window's Next menu. Each step records `name = <expression>` (`V = Matrix.cov(fit)`,
  `M = Matrix([[1, 0.5], [0.5, 1]], rows=['a', 'b'], cols=['a', 'b'])`), runs exactly that expression, is saved with
  the project and recomputed when the objects it uses change. Expressions are checked before they run
  (`pymetrix.matrix_expr`): names, attribute access, calls, operators, indexing and literals only, so imports,
  `lambda`, attributes starting with `_`, `exec`/`eval`/`open` and file input or output are refused with a message,
  and a project with matrix steps opens without the trust typed code needs. `Matrix.to_latex(names=False)` is new.
  Commands `DefineMatrix`, `MatrixFrom` and `EnterMatrix` (`pymetrix:define_matrix`, `pymetrix:matrix_from`,
  `pymetrix:enter_matrix`); help topic "Matrices with names".
  - Checked: each dialog and action gives the same matrix (names and values) as its recorded script run in a fresh
    namespace; each command round-trips through a saved project; forbidden constructs, unknown names and syntax
    errors are refused; paste parsing, grid sizes and names; the window's menu in the app.

- **OccBin: the path without constraints, with anticipated shocks** (`occbin(..., anticipated=True).linear`, dashed in
  the graph) is now the linear model's perfect-foresight path with the same shocks known from the first period. It
  was simulated with the shocks as surprises, so it was not the counterfactual of the constrained path: in the
  `policy_and_nonlinear` demo it fell only to -0.43 while the constrained rate was held at the floor of -0.75 (the
  linear path with the shocks known goes to -1.0). The constrained path was right. Paths with surprises are
  unchanged.
  - Checked: against Dynare 6.0 on the New Keynesian template with a zero lower bound (pymetrix's own model files,
    `tests/fixtures/dynare/nk_zlb.mod` and `nk_zlb_pf.mod`; the dynare-reference workflow, case `occbin_zlb`, makes
    the reference): `occbin_solver` with surprises, both paths, and `perfect_foresight_solver` with the shocks known,
    with the floor as `max` and without it (1e-9); and against pymetrix's own perfect-foresight solver (1e-10).
- **Data contributions and parameter sweeps: checked, not changed.** Two results of the modelling demos that looked
  wrong are right, and the help now says why. Contributions are checked against the smoothed-state weights
  calculated directly (`Cov(a, y) Var(y)^-1`, no Kalman recursion; 1e-8), with and without measurement errors:
  without them, a variable observed as it is comes from its own series alone. A sweep over `phi_pi` is checked
  against impulse responses by hand (undetermined coefficients; 1e-10): output's response to a demand shock moves by
  under 1% across the sweep, so the four lines overlap.
- **Sweeps choose the impulse response their graph shows**: `sweep(..., response="y", shock="epi")`, and Response
  and Shock in the Parameter sweep dialog; `plot()` and the app's Graph tab draw that pair, and its table comes first
  (unknown names are refused). The `policy_and_nonlinear` demo's sweep over `phi_pi` now opens on output's response
  to the cost-push shock, which doubles across the sweep, rather than to the demand shock, which barely moves.

- **Correlated shocks** (`corr` in Dynare files; `shock_corr={("ey", "ei"): 0.3}` for any first-order model). Dynare's
  shocks block is read with `corr e1, e2 = 0.3;` and `var e1, e2 = covariance;` (`DynareModel.shock_corr`), and
  `estimated_params` with `corr e1, e2, ...` in all three forms: an estimated `corr_e1_e2` (the shocks in `varexo`
  order), within (-1, 1), with its prior and bounds. The covariance reaches everything that uses shock sizes: the
  state-space form (off `Q`'s diagonal: the filter, likelihood and smoother), `estimate` (correlations given are
  estimated by default, as shock sizes; a correlation matrix that is not positive definite is penalised), draws
  (normal with the full covariance), theoretical moments, the second-order risk terms, and posterior IRF bands.
  Impulse responses and variance decompositions with correlated shocks are orthogonalised by Cholesky in the
  shocks' order, as Dynare's `stoch_simul`; without correlations, nothing changes. `Model.estimate`, `Model.solve`
  (first-order draws), `Template` (a `shock_corr` field) and `LinearSolution.system`, `irf`, `simulate`,
  `moments`, `fevd` and `solve` take `shock_corr`; `Param("corr_ey_ei", prior=...)` specs carry their metadata. The
  Dynare writer writes `corr` in the shocks block and in `estimated_params`. See `pymetrix.models.covariance`.
  - Checked: the log likelihood against statsmodels' Kalman filter on the same system (1e-9), the covariance `D C D`
    by hand, draws' sample covariance within four standard errors, responses against `Q chol(Σ)` and moments
    against the Lyapunov equation (1e-10); a correlation of 0.5 estimated from 600 simulated periods within three
    standard errors; without correlations, estimates identical to the last bit.

- **Dynare's macro processor** (`@#define`, `@{...}`, `@#if`/`@#ifdef`/`@#ifndef`/`@#elseif`/`@#else`, `@#for x in
  ...` and `@#for (a, b) in ...` with `when`, `@#include` and `@#includepath`, `@#echo`, `@#error`, macro functions;
  numbers, strings, booleans, arrays, tuples, ranges `1:5`, comprehensions, `+ - * / ^`, comparisons, `&&`, `||`,
  `!`, `in`, indexing): `read_mod` carries the directives out before reading the file, as Dynare's preprocessor does,
  with includes found next to the model file. Unknown directives, unclosed blocks, includes not found (or text read
  without a file) and `@#error` are refused with their line. The model editor's Dynare check expands first and puts
  each problem at the line it comes from (a macro error at its directive). See `pymetrix.models.dynare_macros`.
  - Checked: each directive against text expanded by hand; a three-country model written with `@#define`, `@#for`
    loops over the countries, `@#if` on a define and an `@#include`d calibration solves as its hand-written twin
    (decision rules to 1e-12).

- **Regimes with known dates: parameter changes and alternative policies** (`Regime("2009Q1", {"phi_pi": 1.2})`,
  `Regime("2025Q1", {"rho": 1, "phi_pi": 0}, end="2026Q4")`, `Regime(..., equations={"policy": "i = ..."})`; as text
  `"2009Q1: phi_pi=1.2; 2025Q1..2026Q4: rho=1, phi_pi=0"`), after DSGE.jl's regime-switching parameters and temporary
  alternative policies. Regimes go in order of their starts; each changes the parameters and equations it names and
  holds until the next (a temporary one until its `end`, when the regime before it resumes).
  - **Solutions and forecasts:** `LinearSolution.solve(data, span, scenario, regimes=)`, `Model.solve(...,
    solver=FirstOrder(), regimes=)` and `FittedModel.forecast(span, regimes=)` follow time-varying decision rules found
    backwards from the last regime, which holds for good, with agents knowing the regime path (a credible
    announcement); `expected=False` makes a regime a surprise, believed permanent when it arrives. Every regime is
    linearised at the steady state of the one in force before the first period, with the constant its equations leave
    there, so a new inflation target moves the path to its own steady state; a regime in between needs no stable
    solution (a peg for a while). Scenarios (shock paths, holds with free shocks) and draws work through the regimes;
    `Solution.provenance.method["regimes"]` says which holds when.
  - **Filtering and estimation:** `regime_filter(fit_or_model, data, regimes)` runs the Kalman filter and smoother
    through the time-varying rules and reports the regime of each period; `estimate(..., regimes=)` maximises its
    likelihood, with each regime's values estimated as `<parameter>@<regime>` (`phi_pi@2009Q1`, or the regime's name),
    with the parameter's own bounds and prior unless `specs`, `priors` or `bounds` name them. The fitted model keeps its
    regimes at the estimates, and its forecasts continue from the last of them.
  - **In the app:** Regimes (under Advanced) in **Model → Estimate model…**, **Model → Forecast from estimated model…**
    and **Model → Solve…** (first-order solver), as text, recorded as `regimes="..."`.
  - Checked: regimes equal to the base model, or one regime throughout, give exactly the results without them (path,
    likelihood, estimates, forecast); a path through a new policy response, a temporary peg, a new inflation target and
    an alternative rule is the nonlinear perfect-foresight solution of the model with the regimes' parameters as
    exogenous series (to 1e-8; 1e-15 found); the filter's likelihood is statsmodels' Kalman filter with the
    time-varying matrices built by hand (1e-8); estimation recovers a regime's policy response from simulated data
    (within three standard errors), and the dialogs give the same estimates and forecasts as code.

- **OccBin: the iterated smoother, posterior forecasts, held targets and shock decompositions.** The remaining parts
  of a model with occasionally binding constraints now respect them.
  - **Iterated smoother** (`occbin_filter(model, data, smoother="iterated")`; Model → OccBin filter, Advanced), as
    Dynare 6's OccBin smoother: each period's regimes are worked out again from the smoothed state and shock (with
    the regimes expected after it), the Kalman filter and smoother run again with them, until they settle (a cycle
    keeps the regimes the smoothed path contradicts least). Shocks a binding regime mutes (a policy shock at the
    floor) take the values nearest zero consistent with the regime. `result.regimes` gives the settled regimes
    beside the filter's `binding`; the default smoother is unchanged.
  - **Posterior forecasts** (`Posterior.forecast(..., occbin=)`): at each posterior draw, the OccBin-smoothed state
    and OccBin's forecast at that draw's parameters, shocks drawn as surprises; the default follows the estimation.
  - **Targets held by free shocks** in OccBin forecasts (`Scenario(hold=, free=)`, anticipated): Newton's method on
    the free shocks through the piecewise solution; a target the shock cannot reach while a constraint binds is
    refused with a message.
  - **Shock decompositions** (`OccBinFilter.decompose()`, and `FittedModel.decompose()` for a model estimated with
    the OccBin filter): the shocks' contributions through the smoother's time-varying rules, with the regimes held at
    the smoothed ones, and the binding regimes' constants as a part of their own, so the parts add up to the path.
    Moments still use the linear solution.
  - Checked: when nothing binds, the iterated smoother is the linear smoother and the decomposition the linear one
    (1e-10); on simulated data the iterated smoother's regimes agree with the truth at least as often as the
    filter's, and its smoothed path satisfies the model's max period by period (1e-8); each posterior draw equals
    OccBin's own simulation from the OccBin-smoothed state at that draw's parameters (1e-10), and the floor holds in
    every draw; held targets equal the nonlinear perfect-foresight solution with the same holds (1e-8); the
    decomposition adds up (1e-10).

- **Fixed, normcdf in models:** the model editor's checker and the Dynare reader accepted `normcdf`, `normpdf` and
  `erf`, but nothing evaluated them, so a model using them checked and then failed when solved. They now evaluate and
  differentiate (Dynare's meaning: `normcdf(x)` standard, `normcdf(x, mu, sigma)` too), in models, Dynare files
  (parameter values included), data and equations. Checked: a probit-style rule `p = normcdf(a + b*y)` solved and
  linearised against its decision rules by hand (1e-8), and the same model from a Dynare file (1e-10).

- **Distribution functions in expressions** (MATLAB's names and parameterisations): `normpdf/normcdf/norminv(x, mu,
  sigma)`, `tpdf/tcdf/tinv(x, df)`, `chi2pdf/chi2cdf/chi2inv(x, df)`, `fpdf/fcdf/finv(x, d1, d2)`,
  `gampdf/gamcdf/gaminv(x, a, b)` (shape a, scale b), `betapdf/betacdf/betainv(x, a, b)`,
  `lognpdf/logncdf/logninv(x, mu, sigma)`, `unifpdf/unifcdf/unifinv(x, a, b)`, `exppdf/expcdf/expinv(x, mu)` (mu the
  mean), `poisspdf/poisscdf(k, lambda)`, `binopdf/binocdf(k, n, p)`, `erf`, `erfc`. They work in derived series,
  equations, measurement maps and model text, with exact derivatives in x and in the parameters where they are simple;
  otherwise linearisation is refused with a message. Algebraic notation shows Φ, φ and Φ⁻¹; Dynare and IRIS export
  writes `normcdf`, `normpdf` and `erf` and refuses the rest. Checked: against scipy.stats at many points (1e-12),
  inv(cdf(x)) = x, derivatives against finite differences.

- **Moving-window statistics in expressions:** `movavg(x, n)`, `movsum`, `movsd`, `movvar`, `movmin`, `movmax`,
  `movcov(x, y, n)`, `movcorr(x, y, n)` and `movbeta(y, x, n)` (the rolling regression slope): trailing windows over
  the time axis, within each entity of a panel, missing until a full window (and across gaps in the dates). Checked
  against pandas' rolling functions, per entity.

- **Moments of series:** `Dataset.describe(names=None, percentiles=(0.25, 0.5, 0.75))` adds `skewness` and `kurtosis`
  (bias-corrected sample statistics as pandas and Excel compute them; kurtosis is excess kurtosis), keeping its
  columns; `corr(names, method="pearson" | "spearman" | "kendall")`; `cov(names)`; `partial_corr(names, given=)`;
  `crosscorrelogram(x, y, lags=12)` (corr(x(t), y(t+k)) for k = -lags ... lags, band ±1.96/√T); `autocov(name,
  lags=20)`. In the app: Data → Statistics → Descriptive statistics… and Correlations…, Test → Cross-correlogram…,
  each recording the call it stands for. Checked against scipy, pandas, statsmodels and OLS residuals by hand.

- **Deprecated, `**` on a square matrix as the matrix power:** `Matrix`'s `**` moves to NumPy's convention, element
  by element. For now `A ** n` on a square matrix (larger than 1 x 1) with a whole n still gives the matrix power,
  with a DeprecationWarning; a later build makes it element by element. Use `A.power(n)` (or `A.matrix_power(n)`)
  for the matrix power, negative n through the inverse, and `A * A` or `np.square(A)` element by element. Other
  powers were already element by element and do not warn.

- **`Matrix.chol(upper=True)`:** the upper triangular factor R with `R.T @ R = A` (MATLAB's `chol`), named as the
  matrix; checked against `scipy.linalg.cholesky(lower=False)` (1e-12).

- **Matrices with names, extended** (`Matrix`): every result keeps names that meet, and products that do not meet
  are refused as before.
  - Factors: `svd()` (U, s, V named s1, s2, ...), `qr()`, `lu()` (P, L, U; L's rows are the rows in pivot order),
    `schur()`, `qz(B)`, and `eig()` for any square matrix. A general matrix's complex eigenvalues are now returned
    (complex matrices), where they used to be refused; a symmetric matrix's results are unchanged, and
    `eigh()` or `eig(symmetric=True/False)` choose the method. Factors come as named tuples (`A.svd().U`) whose
    window shows a table for each.
  - Functions of a matrix: `expm`, `logm`, `sqrtm`, `norm(kind)`, `cond()`, `null()`, `orth()` and `perp()`, the
    orthogonal complement as for a VECM's alpha-perp (orthonormal columns, as MATLAB's `null(A')`).
  - `vech()`, `Matrix.unvech`, `Matrix.commutation`, `Matrix.duplication`, `Matrix.elimination`; the solvers
    `Matrix.lyapunov` (discrete, as a VAR's unconditional variance, or continuous), `Matrix.sylvester` and
    `Matrix.riccati` (discrete algebraic).
  - Assembly: `Matrix.hcat`, `Matrix.vcat` (refusing clashing names), `set` (an element or a block by name, as a new
    matrix), `sum`/`mean` by row or column, `outer`, `inner`, `Matrix.from_dataset` and `to_dataset`.
  - NumPy's element-wise functions keep the names (`np.log(m)`, `np.exp`, `np.sqrt`, `abs`; also as methods). An
    array on the left of `+`, `-`, `*`, `/` or `@` with a Matrix now gives a named Matrix rather than a plain array.
  - Checked against NumPy and SciPy and by reconstruction (1e-12), the Lyapunov solution against a VAR(2)'s
    unconditional covariance by the vec formula (1e-10), the Riccati solution against the Kalman filter's covariance
    recursion (1e-10), and the commutation, duplication and elimination matrices by their defining identities.

## 0.1.0a9: development build (30 October 2026)

- **In the dialogs:** Model → Solve and Model → Forecast from estimated model take soft holds (`hold_sd`, e.g.
  `y[2025Q4]=0.25`) and least-norm weights (`free_sd`, e.g. `ei=0.2, ey=0.5`) under Advanced; Estimate equation has
  *small-sample correction* beside HAC lags (untick for statsmodels' HAC); Model → Identification has *elasticities*.
  Each records the call it stands for, and the script reproduces it.

- **Fixed, the app's version:** the opening screen and About box read the version from the installed package's
  metadata first, so an editable install showed the version it was installed at (0.1.0a8) after its source moved on.
  They now read `_version.py` beside the app (the code that runs, from a clone or a wheel), and the metadata only
  when that file is missing.

- **Identification without data** (`Template.identification(params, lags=4, at=None)`,
  `Model.identification(measurement, params, shocks=, shock_sd=, lags=4, at=None)`): Iskrev's test at the calibration
  (or `at`), before any data exist, so a calibrated model need not be fitted on something first. `params` are named
  as in `estimate` (default: the calibrated parameters and every `sd_<shock>`). Checked: the same Identification as
  fitting and then testing at the same point (1e-12), and a product a*b of two parameters found one short of full rank.

- **Pairwise collinearity, elasticities and public moments** (`Identification.pairwise`,
  `identification(..., elasticity=True)` on fitted models, models and templates, `pymetrix.models.bayes.moments`):
  - The multiple correlation (`collinearity`) is near one for most parameters of a large model, so it does not say
    which parameters the data confuse. `pairwise` gives each parameter's |cosine| with the one it is most like
    (`closest`), as Dynare's pairwise measure.
  - `elasticity=True` measures each moment's change relative to its size, leaving out moments near zero, so that
    the largest variances do not make every pair look alike.
  - The moments vector (means, covariances, autocovariances) is public as `moments(system, lags)`; `_moments`
    still works.
  - Existing fields and defaults are unchanged; the summary gains a "pairwise" column.
  - Checked: pairwise and closest by hand in both modes, and the moments against an AR(1)'s.

- **The QPM toolkit's dialect in `read_iris`:** the IRIS-style files of `qpm_model_toolkit` read
  directly: section names in any case, hyphenated, aliased or as `!variables:transition`; `//` and `#`
  comments (comment markers inside descriptions kept); descriptions after names; equation names (`is_curve: y = ...`);
  `!modelname`; `!parameter-values`. Checks: the New Keynesian test model in that dialect gives the same state-space
  form as in IRIS's; a country QPM written in that dialect solves to its own QZ solution (P and Q to 5e-15).

- **Least-norm conditional forecasts and soft holds** (`Scenario(hold=..., free=[...], free_sd=..., hold_sd=...)`, on
  first-order solutions: `LinearSolution.solve`, `Model.solve(..., solver=FirstOrder())`, `FittedModel.forecast`):
  with more free shocks than holds, or with `free_sd` (weights) or `hold_sd` (soft holds), every free shock is free in
  each period up to the last held one, and the plan takes the smallest shocks, in units of `free_sd`, that meet the
  holds, as Waggoner and Zha's conditional forecasts. A hold with a standard deviation is soft, met more closely the
  smaller it is; 0 or none is exact. One free shock per hold, without `free_sd` or `hold_sd`, is solved exactly as
  before (a different number of free shocks than holds was refused, and is now a least-norm plan). The nonlinear
  solvers refuse `hold_sd` and `free_sd`. Checked: the weighted minimum-norm and the soft solutions equal numpy's on
  the responses taken from separate simulations (1e-10); square plans reproduce the paired solution.

- **The version is written once**, in `src/pymetrix/_version.py`: `pyproject.toml` reads it (hatch's dynamic
  version, as the plugin packages already do), and the release workflow and the review-repository script check the
  tag against it. The install guide says when to run `pip install -e` again in a clone (after a pull that changes the
  version or the dependencies: pip's record is made at install time), and the release workflow lists the release
  steps. No version change.

- **Import cost, measured and guarded:** `import pymetrix` takes about 1.7 s here, 1.4 s of it numpy, pandas and
  scipy, which it needs; statsmodels, matplotlib, linearmodels, PySide6, jpype and the other optional libraries are
  already imported only where used, and a test now keeps it so. The top-level names stay eager: making them lazy
  would save a few tenths of a second and change how the registries fill.

- **Measurement errors in Dynare files** (`read_mod`): `stderr` (or a variance) on a `varobs` variable in the
  `shocks` block is that series' measurement error (`DynareModel.measurement_sd`), and `stderr <variable>` in
  `estimated_params` estimates it as `sd_<variable>`, as Dynare reads them; `estimate` and `log_posterior` use them.
  Before, the first was taken as a shock size and ignored, and the second refused. A measurement error on a variable
  not in `varobs` is refused. Checked: the same log posterior and estimates as the model written by hand with a
  `MeasurementMap` carrying the errors.

- **Blanchard-Kahn counts on a solution** (`LinearSolution.n_stable`, `LinearSolution.n_predetermined`): the stable
  roots, unit roots included as the solver counts them, and the variables that appear lagged; the roots table states
  both, and labels roots within 1e-6 of one "unit root" (they were labelled "unstable", though the solver kept them
  as stable). A model that fails the conditions is still refused by `solve`, with both counts in the message.

- **Model estimation without the Hessian** (`Model.estimate(..., cov="none")`, and through `Template.estimate` and
  `DynareModel.estimate`): the same mode with no standard errors, skipping the inverse Hessian, for speed when only the
  mode is wanted; the summary says so, and `mcmc` starts from its diagonal proposal. `cov="classic"`, the default, is
  the inverse Hessian, as before; any other value is refused (it was silently treated as "none").

- **HAC without the small-sample factor** (`OLS(cov="hac", hac_correction=False)`, `IV(..., hac_correction=False)`):
  the Newey-West covariance without the n / (n - k) factor, statsmodels' default. `hac_correction=True`, the default,
  keeps the factor, as before (and as EViews and Stata's newey); the summary's covariance note says when it is left
  out. Other covariances ignore the option. Checked: both settings equal statsmodels' HAC with `use_correction` set
  to match (1e-10).

- **Decompositions from a pre-sample state** (`LinearSolution.decompose(levels, start="presample")`, and
  `FittedModel.decompose(start=...)`): the first period of the paths is the pre-sample state x(-1), and the
  decomposition starts in the next period, so that period's shocks are counted as the shocks' (the initial condition
  is P x(-1), carried forward). `start="first"`, the default, counts the first period's state, shocks included, as the
  initial condition, as before. Checked: the parts add up to the path in both modes, and the two converge as P^t.

- **Soft filter tunes** (`StateSpaceSystem.filter(data, tunes=..., tune_sd=...)`, and `contributions`): a tune can carry
  a standard deviation, `{"y_gap": {"2024Q4": (-1.0, 0.25)}}` (a list `[-1.0, 0.25]` as well, as a saved call has it)
  or `tune_sd={"y_gap": {"2024Q4": 0.25}}` (or one number for all of a variable's tunes). A soft tune is an extra
  measurement row seen only in its own period, with variance sd**2; None or 0 is exact, and a variable whose tunes
  are all exact is filtered exactly as before. In `contributions`, a variable's soft tunes add up to one part.
- **Smoothed shocks** (`StateSpaceSystem.filter(data, ..., shocks=True)`): the smoothed shocks e(t) of
  x(t) = P x(t-1) + Q e(t) as extra columns named after the shocks, as a solution's paths carry them, without carrying
  the shocks as states; `shocks=False`, the default, gives the same columns as before. The Kalman run gains
  `eta_smooth`, the smoothed state disturbances: statsmodels' `smoothed_state_disturbance` on that engine, a
  disturbance smoother on the native one. e(0) comes from the start: E[e(0) | y] through the states that start from
  their stationary distribution. Checked: x(t) - P x(t-1) = Q e(t) to rounding error on both engines, with and
  without tunes, and the shocks equal the smoothed states of the same model with the shocks as states (e(0) included).

- **IRIS files fill the new fields; draws of linked entities.**
  - **IRIS `!reporting_equations`** (`name = expression;`, with descriptions, `x{-1}` timing and log variables as in
    measurement equations) are read into the measurement map's reported series (`reported`), so they appear with the
    smoothed history, forecasts and draws; a later one may use an earlier one. A file with reporting equations but no
    measurement equations gets a map that only reports: it gives no state-space form and cannot be estimated, and says
    so. The model editor's IRIS check takes the block (each name on the left a reported series in Symbols), and
    writing a model to IRIS writes its reported series as `!reporting_equations` instead of a note.
  - **IRIS parameter descriptions** become the model's parameter metadata (`Param(name, label=...)`, `model.spec`),
    and those of `std_<shock>` the labels of the shock sizes (`sd_<shock>`) and measurement errors (`sd_<series>`);
    `labels` keeps them too. Writing IRIS puts the shock sizes' labels back on `std_<shock>`.
  - **Stochastic simulation of linked entities** (`model.solve(panel, span, draws=..., seed=...)` with `links`): each
    entity's errors are drawn as when solved alone (seed + its position), and each draw is the joint solution of all
    the entities with that draw's errors as add factors, links included in the draws. Backward-looking models;
    coefficient draws, GARCH errors and first-order solutions are refused with links.
  - Checked: reported series against the same quantities worked out by hand from given paths (a lag, a log variable
    in levels, an earlier reported series put in); an IRIS file written and read again keeps its reported series,
    labels and descriptions; each draw of a three-country linked model against the stacked model solved with the same
    errors as add factors (1e-8), and the seed reproducing the draws.

- **Parameter metadata on models** (`Model(..., specs=[Param("rho", prior=Beta(0.7, 0.1), bounds=(0, 1), label=...,
  symbol="ρ", start=0.75, fixed=False)])`, after DSGE.jl's parameter objects): every parameter of a model, and its
  shock sizes and measurement errors (`sd_<shock>`, `sd_<series>`), can carry its prior, bounds, starting value, a
  description, its symbol and whether it is fixed. `estimate()` takes them all from there, so a model carries its
  estimation set-up: explicit `priors=` and `bounds=` still override, `fixed` parameters are left out, and
  `transform="log"` or `"logit"` bound a parameter that has no bounds. The parameter table and the fitted summary show
  priors and descriptions, and algebraic notation uses the symbols. `Model.spec(name)` gives a parameter's merged
  metadata (the model's over its equation's).
  - **Read and written:** Dynare's `$\rho$` names (turned into Unicode), `long_name` descriptions and
    `estimated_params` become specs, and the variables' descriptions are kept (`DynareModel.labels`); writing Dynare
    puts them back, with an `estimated_params` block. In the model text, a parameter or shock takes `bounds 0 1`,
    `prior beta(0.7, 0.1)`, `start 0.8`, `fixed` and `symbol ρ` after its value (a shock's apply to its size), shown
    in the editor's Symbols, with problems placed where they are (an unknown prior shape, a misspelt keyword). IRIS
    files get the descriptions; priors and symbols are noted as not written.
  - Checked: estimating with specs equals estimating with the same priors, bounds, starts and parameter list given to
    `estimate` (to 1e-10); a Dynare file's specs written and read again are the same; the model text written back is
    the text.

- **Measurement from raw data and back, and reported series** (`MeasurementMap(..., transforms={"growth":
  "400*dlog(gdp)"}, reported=["gap = 100*y"])`; in the model text, `growth = ... from 400*dlog(gdp)` and a `reported`
  section; Model → State-space form, Advanced). After DSGE.jl's observable mappings and pseudo-observables.
  - **Transforms:** an observed series made from raw data. The filter, the likelihood and estimation take the raw
    data (GDP and CPI levels, not growth rates). Forecasts and every draw of a stochastic forecast report the raw
    series too, rebuilt beyond the data by solving the transform each period, so no inverse is written: four-quarter
    changes and per-capita transforms work as well.
  - **Reported series:** quantities in the model's variables reported but never fitted (the output gap, real rates),
    with the smoothed history (Kalman and OccBin filters), forecasts, draws and posterior forecasts.
  - Exported to Dynare, neither carries across: the notes say so. IRIS takes the reported series (see above).
  - **Fixed:** the deterministic trends of observed series (IRIS `!dtrends`) were left out of the observed series in
    each draw of a stochastic forecast.
  - Checked: estimating from levels equals estimating from growth rates worked out beforehand (likelihood to 1e-6);
    rebuilt levels equal levels compounded by hand, in forecasts and each draw (1e-12); four-quarter and per-capita
    transforms rebuilt by hand; reported series equal their expressions of the smoothed and forecast paths.

- **Links between the entities of a model solved per entity** (`Model([...], over="country", links=[Link("ys", "y",
  trade)])`; Model → Solve, Advanced → Panel → Links: `ys = trade @ y`): foreign variables and world aggregates, as in
  a global VAR or a multi-country model. Each country's `ys` is a weighted sum of the other countries' `y`, with a
  weight matrix (a `Matrix`, a DataFrame or nested dicts, a row per country) or one set of weights for all (an
  aggregate). Before the span the links come from the data, so their lags are there; over the span the countries
  are solved jointly, each with its links as given paths, until the links settle (fixed-point rounds with Anderson
  acceleration). A link may also be made from data (an oil price weighted by shares). Works with Newton,
  Gauss-Seidel and perfect foresight, each country's own parameters and scenarios; `PanelSolution.links` and
  `rounds` say what was linked and how many rounds it took. Draws with links: see above. Checked: the same
  three-country model written out by hand as one stacked model and solved by Newton, backward-looking with lags of
  two links and forward-looking by perfect foresight, agree to 1e-8 or better; a pair of countries with no joint
  solution is refused.

- **The model editor shows the equations and checks the solution** (`src.algebraic()`, `src.latex()`,
  `src.check_solution()`; `pymetrix.models.algebraic_equations`, `write_latex`, `check_solution` for any template,
  Dynare model or Model): an Equations tab beside the text shows each equation in algebraic notation (yₜ = yₜ₊₁ −
  (1 / σ) (iₜ − πₜ₊₁) + eyₜ; a click goes to its line), and Copy LaTeX puts them on the clipboard as `align` blocks
  tagged with their names. Check solution finds the steady state and the roots of the model linearised there and says
  whether there is a unique stable solution, none, or many (Blanchard-Kahn), with the roots shown even when the
  conditions fail. `LinearModel.roots()` gives the roots without solving. In Unicode notation a division multiplied
  by a bracket keeps its own brackets, (1 / σ) (iₜ − πₜ₊₁), and LaTeX writes `\max` and `\min`. Checked: the roots
  against the eigenvalues of the model written out by hand (determinate and indeterminate Taylor rules), an explosive
  model with no stable solution, and the New Keynesian model's roots and Dynare's steady state against the solvers'.

- **The model editor in IRIS's syntax, and models written in other syntaxes** (`ModelSource(text, "iris")`;
  `src.export("dynare")`, `export_model(model, syntax)`; Model editor → Write as): IRIS's `.model`
  text is checked as it is typed, as pymetrix's and Dynare's are (unknown sections with the nearest known one,
  declarations with their descriptions, each equation parsed alone, then built by `read_iris`). Any template, Dynare
  model or `Model` with named parameters can be written in pymetrix's syntax, Dynare's or IRIS's: a `Translation` with
  the text, notes on what does not carry across (comments, `!for` loops written out, Dynare's estimation settings and
  commands, measurement equations as observed variables in Dynare, `max` constraints) and the check that the written
  text builds and solves to the same model (impulse responses and steady state). The editor opens the written text in
  a second editor with the notes beside it. Checked: every pair of the three syntaxes for the New Keynesian model
  agrees to 2e-15, and IRIS's RBC model with log variables to 1e-13.

- **Models written as text, and the model editor** (`ModelSource(text, syntax="pymetrix"|"dynare")`, `.check()`,
  `.build()`; Model → Model editor): a model typed in pymetrix's syntax (sections for parameters, shocks, exogenous
  variables, variables, equations and measurement, with descriptions in quotes and `#` comments) or Dynare's, checked as
  it is typed. Each problem is shown at its line and column and underlined there: an unknown name with the nearest
  declared one, an unclosed bracket, an equation for a variable another already determines, a left-hand side that is
  not a variable, a value that uses a parameter not yet given, unused parameters and shocks. One mistake gives one
  error, not a cascade through the lines after it. Beside the text, the Problems (click to go there) and the Symbols
  (role, value, description, the lines that use each). Apply builds a `Template` (pymetrix's syntax) or a
  `DynareModel` as a step that keeps the text as written, comments and all: a project saves the text and builds the
  model from it when it opens. Checked: the New Keynesian model in pymetrix's syntax solves to the template's
  solution, and Dynare's text to `read_mod`'s.

- **Sequential Monte Carlo** (`fit.smc(particles=2000, steps=2, target=0.5)`, a `Posterior`; Model → Sequential Monte
  Carlo): Herbst and Schorfheide's sampler, as Dynare 6's. Particles from the prior move to the posterior through the
  likelihood raised to rising powers, each step chosen to keep the weights' conditional effective sample size at
  `target`; systematic resampling below half; random-walk moves with the particles' covariance, scaled towards an
  acceptance of 0.25. The log marginal data density comes from the same run (`marginal_likelihood()`; the modified
  harmonic mean with `method="mhm"`). It needs no starting point and finds several modes where a Metropolis-Hastings
  chain stays in one. Particles move in fixed blocks with their own random streams, in other processes when a stage is
  long: the same draws either way. Checked: an AR(1) posterior and marginal density against the exact ones on a grid
  (0.08); both modes of a coefficient that enters squared; the New Keynesian posterior and density against
  Metropolis-Hastings and Laplace.

- **Forecasts that respect occasionally binding constraints** (`fit.forecast(span, scenario, occbin=True)`; the default
  for a model estimated with the OccBin filter; OccBin under Advanced in Model → Forecast from estimated model): OccBin
  from the OccBin-smoothed state at the end of the data (newer `data` filtered again with the OccBin filter), with the
  scenario's shock paths, anticipated unless all are named unanticipated; with `draws`, shocks drawn in each period as
  surprises, each draw solved by OccBin. `provenance.method["binding"]` says when each constraint binds. Holds and free
  shocks stay with linear forecasts. Checked: with an anticipated shock path the forecast is the nonlinear
  perfect-foresight solution of the model with its max (1e-8); a constraint that never binds gives the linear forecast;
  the floor holds in the forecast and every draw, where the linear forecast breaks it.

- **The OccBin filter and estimation with occasionally binding constraints** (`occbin_filter(model, data)`, an
  `OccBinFilter`; `model.estimate(..., occbin=True)`, also from templates and Dynare files; Model → OccBin filter, and
  OccBin under Advanced in Model → Estimate model): the Kalman filter and smoother of Dynare 6's piecewise approach
  (Giovannini, Pfeiffer and Ratto 2021). Each period uses the decision rule of the regimes expected from the filtered
  state, re-solved with the estimated shock until the regimes settle; where the guesses cycle (an observation near
  the bound), the regime under which the period is likelier is taken. The result has the smoothed and filtered
  variables, the smoothed shocks, the regimes used, the constraint checked on the smoothed path and the likelihood,
  which posterior-mode estimation and Metropolis-Hastings use. A series fixed at the bound and observed without
  error is allowed (the density of the others). Checked: a constraint that never binds gives the linear Kalman
  filter (1e-9); given its regimes, the likelihood and filtered states are statsmodels' time-varying Kalman filter
  (1e-9); on data simulated by OccBin and observed without error, the true states, binding periods and shocks.
  `occbin` keeps its results; its solver is now shared with the filter.

- **Fixed, `pymetrix.testing.check_project`:** on a busy machine a step can take long enough to count as expensive, and
  opening the project in Manual mode then defers it unless its result was saved, so the check found the result
  missing (seen as an intermittent failure of the nowcasting dialog test). It now computes deferred steps after
  reopening, as Compute does, before comparing.

- **TRAMO-SEATS with interventions and user regressors** (`TramoSeats(interventions="ls 2020Q2, ao 2008Q4",
  regressors="holidays, strike=irregular")`): outliers known in advance (additive, level shift, transitory change,
  seasonal), kept without testing; and series of the dataset as regressors (holidays, strikes, policy changes), each
  effect assigned to a component: calendar (the default, removed from the adjusted series with the seasonal),
  irregular, trend or seasonal. The adjusted series records the effects (`tramo_regression`), and a derived adjusted
  series recomputes when a regressor changes. Under Advanced, Regression, in the Seasonal adjustment dialog. Checked:
  the regression effects equal generalised least squares by hand given TRAMO's ARIMA parameters (1e-6); an effect in
  the irregular changes the adjusted series by exactly the effect; planted effects recovered.

- **Restricted VARs** (`VAR([...], lags=2).fit(data, method=Restricted(zeros={"pi": ["r"]}, block_exogenous=["oil"]))`,
  which was a placeholder for methods other than least squares and Bayes): coefficients restricted to zero, a whole
  variable's lags, one lag or a deterministic term, or every other variable's lags for block-exogenous ones; feasible
  GLS (SUR), iterated to maximum likelihood with `iterate=True`; standard errors, impulse responses and their
  bootstrap bands (each draw re-estimated under the restrictions) follow. The VAR dialog's Restrictions group.
  Checked against linearmodels' SUR (estimates and standard errors to 1e-10, one-step and iterated); with no
  restrictions the coefficients are least squares'.

- **Maximum likelihood for explicit-form equations** (`Equation("y ~ a + b*x^c", params=["a", "b", "c"],
  family=Logit()).fit(data, method=ML(start={...}))`, which was a placeholder): the right-hand side as the Logit,
  Probit or Poisson index; quasi-Newton with the analytic score, within any bounds; classic standard errors from the
  exact Hessian (with the index's second derivatives), white, OPG and clustered ones from the scores; `ML(start=)`.
  Checked: a linear index written out equals the linear-form fit (1e-8), and nonlinear indexes (a power in a logit, an
  identity-link Poisson) match statsmodels' GenericLikelihoodModel (estimates 1e-6, standard errors 1e-5).

- **Models solved for each entity of a panel** (`Model([...], over="country").solve(panel, span)`, which was a
  placeholder): each entity on its own time series, with the same parameters or its own (`entity_params`), one
  scenario or one per entity, draws with a seed per entity; a `PanelSolution` holds the panel of solutions and each
  entity's `Solution` (`solution("UK")`). The Solve dialog takes the entity axis under Advanced. Checked against each
  country's recursion by hand, with its own parameters and scenario.

- **A fluctuation test of the cointegrating eigenvalues** (`vecm.fluctuation_test(draws=199)`,
  `fluctuation_plot()`): Hansen and Johansen's statistic, the largest sqrt(T) (t/T) |xi(t) - xi(T)| over the
  recursions of xi = ln(lambda / (1 - lambda)) with the short run at the full-sample estimates, for each of the first
  rank eigenvalues and their sum, with p-values from a bootstrap of the fitted VECM (constant parameters). The
  asymptotic critical values (an absolute Brownian bridge's supremum) proved unreliable here: with the variance of the
  eigenvalue's influence function the test hardly ever rejected, and even with the true variance it rejected 12% of
  the time at 5%. The bootstrap version rejects 3% at 5% and 7% at 10% under constant parameters, and 54% and 69%
  after a mid-sample fall in adjustment (200 samples of 200 periods).

- **A command and function reference** (`docs/reference/reference.md`, with CSV tables), built from the code by
  `scripts/make_api_reference.py`: every dialog (menu, settings, the API call it records), every API call (signature,
  return type, the dialog that records it), every method, test, filter and solver, and the other public names. Each says
  whether it wraps a library function (which, and which library, from the adapter specs) or is PyMetrix's own code
  (with the library functions it uses), and a last table lists each statsmodels, SciPy, linearmodels, arch and
  scikit-learn function PyMetrix calls, with the PyMetrix code calling it.

- **Lag selection for panel VARs** (`VAR([...], lags="bic", max_lags=4, by="firm").select_order(panel)` or `.fit(panel)`,
  which was a placeholder): Andrews and Lu's MBIC, MAIC and MQIC from Hansen's J, with the J test and the overall
  coefficient of determination, every lag length estimated with the same instruments on the same rows, as Abrigo
  and Love's pvarsoc; Lutkepohl's criteria for `transform="within"`. The Panel VAR dialog chooses the lags under
  Advanced, and VAR lag selection takes a panel's entity axis. Checked: the criteria from each fit's J by their
  formulas (1e-10), CD by hand, and the true lags found in simulated VAR(1) and VAR(2) panels.

- **FAVARs, filled out:** the number of factors by Bai and Ng's ICp1, ICp2 or ICp3 (`favar(..., factors="icp2")`,
  `factor_criteria`), the VAR's lags by a criterion (`lags="bic"`); bands that bootstrap the whole two-step procedure
  (`fv.irf(series, 48, draws=500)`, as Bernanke, Boivin and Eliasz; `bootstrap="var"` holds the factors fixed);
  variance decompositions of panel series' common components (`fv.fevd`); a Factor-augmented VAR dialog; tabs for
  the factors, each series' R-squared and the factor criteria. `VAR(..., factors=)` now points to `favar`. Principal
  components come from the smaller cross-product's eigenvectors (equal to the SVD's to 1e-13, seven times faster
  here). Checked: Bai-Ng's criteria by hand, the lags as `select_order` on the factor block, the bootstrap's VAR as
  `VAR.fit`, the decomposition by hand.
- **Fixed, FAVARs:** bands with `draws` were the VAR's band ends mapped through the loadings, which is not a
  quantile and gave lower above upper for negatively loaded series; they are now each draw mapped, then quantiles.
  The purge failed with fewer slow series than factors; it now uses as many of their components as they have.

- **Residual correlograms for VARs and VECMs** (`var.residual_correlogram(12)`, also Bayesian VARs): each residual's
  correlation with each residual h periods earlier, with the ±2/sqrt(T) band and a count of those outside it, as a
  table (a column per pair) and a panel per pair (`plot()`); Residual correlogram tabs in their result windows.
  Checked against statsmodels' resid_acorr (1e-12).

- **Lag search for single equations** (`Equation("dly ~ dlx").select_lags(data, max_lags=8, of=None)`): lags 1 to p
  of the left-hand side as written (or of the variables in `of`, for distributed lags) added for p = 0 ... max_lags,
  every candidate fitted on the sample the largest allows by the equation's own method (least squares, or maximum
  likelihood with ARMA errors or GARCH), compared by AIC, BIC and Hannan-Quinn; `best`, `equation("bic")`. Lags the
  equation already has are not added again. Estimate → Time series → Equation lag search…. Checked against
  statsmodels' ar_select_order (criteria differences to 1e-8, the same lags chosen).

- **Recursive Johansen eigenvalues** (`vecm.recursive_eigenvalues(start=None, base="Z")`, Hansen and Johansen
  1999): the eigenvalues on samples ending each period from a third of the way in (or `start`), everything
  re-estimated (`"Z"`) or the short run held at the full-sample estimates (`"R"`); seasons, exogenous and restricted
  terms as in the fit. `recursive_eigenvalues_plot()` draws the first rank, also as the VECM's Recursive eigenvalues
  tab. Checked against R's urca on truncated samples (1e-9).

- **Roots plots and cointegrating relations:** VARs and VECMs have `eigenvalues` (the companion matrix's, complex)
  and `roots_plot()`, the roots against the unit circle (a VECM's levels VAR has K - rank on it), with Roots and Roots
  graph tabs; the roots table now gives the real and imaginary parts. VECMs have `relations()`, the cointegrating
  relations beta' y(t - 1) with any constant, trend or exogenous terms restricted to them, and `relations_plot()`, in
  Relations tabs. Checked against R's urca (1e-8) and statsmodels' VAR roots (1e-10).

- **A Correlogram tab on equations fitted to a time series:** the residuals' autocorrelations, partial
  autocorrelations and Q-statistics up to 20 lags (a quarter of the observations if fewer; standardised residuals and
  ARMA-adjusted degrees of freedom after ARMA errors), with a chart; with a GARCH variance, the squared residuals' too.
- **The Hannan-Quinn criterion** (`fit.stats["hq"]`, -2 log L + 2k ln ln n) among equations' statistics and in their
  summaries, beside AIC and BIC: least squares, maximum likelihood (logit, probit, counts, Tobit, Heckman), ARMA
  errors and GARCH. Checked against statsmodels ARIMA's hqic.

- **Historical decompositions of VECMs** (`vecm.identify(Cholesky([...])).historical_decomposition()`): the levels
  split into the identified shocks' contributions, through the levels VAR (permanent shocks keep contributing), and
  the baseline the VECM gives from its first observations and deterministic terms; the parts add up to the data.
  Estimate → Historical decomposition… and Next ▸ take VECMs too. Checked against each shock's contribution rebuilt in
  error-correction form from statsmodels' alpha, beta and gamma, in the five deterministic cases (1e-8).

- **Bootstrap bands for VECM impulse responses** (`vecm.irf(20, draws=500, alpha=0.05, seed=0)`, which was a
  placeholder): a residual bootstrap as in JMulTi and R's vars, each series rebuilt from the estimated VECM (its
  deterministic and exogenous terms in every case) and re-estimated with the same rank, lags and case; percentile
  bands. The Impulse responses dialog now draws them for VECMs too (its Draws setting, 500 by default); `irf()` in code
  still draws none unless asked. Checked against statsmodels' asymptotic standard errors of the levels-VAR
  coefficients (within 10%).

- **PyMetrix, as the app calls itself:** window titles, the start page, the file dialogs, the help window and its
  pages, the Immediate window, the plugin manager, the desktop shortcut and the codespace say PyMetrix. The package,
  `import pymetrix` and the commands keep their names.
- **An opening screen, at once:** `pymetrix-gui` now starts in a small launcher (`pymetrix_app`) that needs only Qt, so
  a screen with the name, version and copyright (the holder left blank until settled) appears within about a second,
  while pandas, SciPy and statsmodels load behind it; it says what is loading and gives way to the main window.
  **Help → About PyMetrix** shows the same. Shortcuts start the launcher (`python -m pymetrix_app`).
- **An icon, and its own taskbar entry:** the window, the taskbar and the desktop shortcuts show PyMetrix's icon (a
  rising line on the opening screen's blue), not Qt's or Python's. On Windows the app has its own taskbar entry
  instead of being grouped with Python.
- **Menus end every dialog entry with "…"**: commands and plugins still write `"Data/Set sample..."`, and the menus,
  Find anything and the help's command lists show `Set sample…`, as the app's own entries already did.
- **Secondary text follows the light or dark theme:** notes, counts and hints that were a fixed grey (hard to read on
  a dark theme) use the theme's own muted colour.

- **Historical decompositions of VARs** (`var.identify(Cholesky([...])).historical_decomposition()`, which was a
  placeholder): each variable over the estimation sample split into the contributions of the identified shocks and
  the baseline (deterministic terms and initial conditions), adding up to the data exactly; under any identification
  with as many shocks as variables (sign restrictions at the median-target member), and for Bayesian VARs at the
  posterior mean. A `Decomposition`, as models' shock decompositions: `contributions`, `baseline`, `shocks`,
  `table`, and `plot()` as stacked bars. In the app: Estimate → Historical decomposition…, and Next ▸ on a VAR.
  Checked against a companion-form recursion from statsmodels' estimates (1e-10).

- **Dynamic forecasts of equations with lags of the dependent variable and ARMA errors or GARCH variance**
  (`Equation("y ~ y[-1] + x", errors=ARMA(1, 1)).fit(data).forecast(data, span, dynamic=True)`, which was a
  placeholder): each forecast feeds the next period's lags, and the standard errors combine the equation's own
  lags with the ARMA weights and the GARCH variance path. Lags of a transformed left-hand side work the same way
  (`dlog(y) ~ dlog(y)[-1]`); other functions of the dependent variable (`y ~ d(y)[-1]`) are solved as a model, without
  standard errors. Checked against statsmodels' ARIMA(2, 0, 1) and arch's AR-GARCH at the same parameters, and against
  the equation solved as a model.
- **Fixed:** static forecasts of equations with ARMA errors or GARCH variance and a lagged dependent variable were all
  missing (the first estimation period's lags were not looked up). Stochastic simulation no longer drops draws
  whose Newton steps reach where an equation is undefined, such as the log of a negative level: the step is
  halved (a log equation over twelve periods lost a fifth of its draws).

- **Fixed:** seasonal ARIMA could stop short of the maximum likelihood. statsmodels' default optimiser (L-BFGS)
  declared convergence early on flat likelihoods, by an amount that depended on the CPU: with statsmodels 0.14 the
  airline model's MA(1) came out as -0.4135 on some machines, where R gives -0.4018. `SARIMA` now fits with BFGS,
  which reaches R's estimates on every CPU and version tried; `backend_options={"method": "lbfgs"}` restores the old
  optimiser. Estimates that were already at the maximum change only in the fourth decimal or beyond.

- **Renamed:** `Equation.require_target()` is now `require_dependent()`, the last name with `target` in it;
  the old name still works, with a DeprecationWarning.

- **Equations with ARMA errors or GARCH variance in models:** a fitted `Equation(..., errors=ARMA(p, q),
  variance=GARCH(...))` can be used in a `Model`. AR terms enter the equation's residual; MA terms through the
  equation's innovations, a variable of the model (`e_<equation>`, zero ahead), whose history comes from the fit and
  is continued from the data over any gap before the span. Solved, the model gives the equation's dynamic forecast.
  Stochastic simulation follows each draw's GARCH variance path; errors are correlated (or bootstrapped) in
  standardised units, and t errors are drawn through a Gaussian copula. An unfitted equation with such errors is
  refused. Checked: the model's solution equals the equation's dynamic forecast (1e-10), and the simulated
  variances its forecast variances (5%).

- **Renamed: the last `target` arguments and fields.** Each old keyword still works, with a DeprecationWarning, and
  saved projects (settings, calls and dialog steps) open and replay unchanged:
  - `backtest(..., variable=)`: the column name for a single-equation forecast.
  - `BMA(outcome, candidates, ...)`, and the Bayesian model averaging dialog's field.
  - `MaxShare(variable, ...)`.
  - `Dataset.write(path)` and `pymetrix.io.write(dataset, path)`.
  - `run_test(obj, test)`, `available_tests(obj)`, `Test.run(obj)` and `help_page(obj)`; `show_help(topic)` in notebooks.
  - The Correlogram and Run test dialogs' object field is now `source`, as in other dialogs.
  - Smaller helpers: `proxy_impact(..., normalise_on, ...)`, `pymetrix_name(name, variable)` and
    `pymetrix.testing.check_help(obj)`.
  - Results: the dependent variable is `dependent` on equations and their fits (`Equation`, `FittedEquation`,
    the Bayesian fit) and on ARDL, SARIMA, theta, STL, exponential smoothing, threshold and smooth-transition
    autoregressions, model averaging, scikit-learn learners and growth at risk; `ProxySVAR.normalise_on`. Reading
    `.target` still works, with a DeprecationWarning, and results cached in saved projects load.
  - Records: a recorded call's object is `Call.receiver`, a figure style call's name `StyleCall.kind`, and the
    variable a model equation determines `dependent`. Projects, result files and house styles saved with the old
    names open unchanged.

- **Bayesian equations in the Equation dialog:** method Bayes, with the family (Gaussian, logit, probit, Poisson) and
  the priors typed under Method settings (`x=Normal(0.5, 0.1); sigma=InvGamma1(1, 1)`), the draws and chains (seed,
  sampler and burn-in under Advanced). The recorded code is the full `Bayes(prior={...})`; `Bayes(prior="...")`
  takes the same text in code. A help topic on Bayesian estimation of an equation.

- **GMM on moment conditions written in Python** (`MomentGMM(moments, start={...}).fit(data)`): the function returns
  each row's moment contributions; one-step, two-step, iterated or continuously updated, robust or Newey-West
  weighting (optionally centred), bounds, Hansen's J, and the moments at the estimates (`FittedGMM`). Checked
  against statsmodels' generic GMM on a consumption Euler equation (estimates, standard errors and J to 1e-6), and
  least squares written as moments against OLS with White errors.

- **TRAMO-SEATS, set up where it runs:** the codespace now has Java 21 and fetches JDemetra+'s libraries when it is
  built, so TRAMO-SEATS works there (Debian's own Java, 17, is too old). The full test job installs Java 21, JPype and
  the libraries, so the TRAMO-SEATS tests run in CI instead of being skipped. `pymetrix.jdemetra.install()` waits and
  tries again when Maven Central turns requests away (429, or 5xx), and says which jar failed.
- **The app stops Java as it quits** (`pymetrix.jdemetra.shutdown()`), before Qt's windows are destroyed: on Windows
  CI, a process that had run TRAMO-SEATS and loaded Qt crashed as Qt was torn down, after every test had passed.
- **TRAMO-SEATS checked against Eurostat's R interface** (rjd3tramoseats, the same JDemetra+ code, so a check of
  pymetrix's bridge): a new workflow, `jdemetra-reference.yml`, runs `tests/fixtures/reference/make_rjd3tramoseats.R` on
  five of R's datasets with each predefined specification, then pymetrix on the same 35 cases, on every pull request and push to main: the same
  adjusted series (to 1e-8, relative), model, logs and outliers.

- **ARMA errors with GARCH variance** (`Equation("r ~ 1", errors=ARMA(1, 1), variance=GARCH(1, 1))`, which was a
  placeholder): conditional maximum likelihood, as fGarch and rugarch, with normal or t errors, GJR terms and subset
  lags; static and dynamic forecasts (multi-step variances from the GARCH path and the ARMA weights). Checked against
  the arch package with AR terms (log likelihood to 1e-6, forecasts to 1e-4) and R's fGarch with MA terms (within 0.2
  of a standard error), and the forecast variances against simulation.

- **Renamed:** the proxy SVAR's `target` argument is now `normalise_on`
  (`var.proxy_svar(data, "ff4", normalise_on="ff")`): the variable whose shock the instrument identifies, and which a
  `scale="unit"` shock moves by one on impact. The Proxy SVAR dialog's field is renamed too. `target=` still works,
  with a DeprecationWarning; scripts that use it record the new name, and saved projects with the old name open and
  replay unchanged.
  `Scenario`'s `target` is now `hold` too (`Scenario(hold={"i": 0.5}, free=["ei"])`: the variables held to a path,
  each by a free variable), and so is the Solve and Forecast from estimated model dialogs' field. `Scenario(target=...)`
  still works, with a DeprecationWarning, and saved projects open with the old names.
- **Renamed, in the finance plugin:** `optimize_portfolio`'s `target` is now `target_return` (the mean return per
  period for `objective="target_return"`), and so is the Portfolio dialog's field. `target=` still works, with a
  DeprecationWarning, and saved projects replay.
- **Extension API 1.4** (additive): `migration`, `call_migration` and `renamed_arguments` in `pymetrix.extension`,
  so a plugin can rename a dialog's fields or a call's arguments and keep saved projects and scripts working.

- **Plugins' report templates and added themes in the dialogs:** the report dialog's template and theme, and the
  figure options' theme, list the templates and themes pymetrix knows now (`report_templates()`, `themes()`): those
  plugins register and those added with `Theme.from_style` or `set_theme`, after the built-in ones. The empty choice
  still means the summary and graph, or the session's default; the recorded code is unchanged.

- **Forecasts with parameter uncertainty** (`post.forecast(span, draws=500)` on a Metropolis-Hastings posterior; in
  Model → Forecast from estimated model, choose the posterior): at posterior draws spread over the chains, the model
  is solved and its variables filtered from the data again, and one path of its shocks drawn (`shocks=False`:
  parameter uncertainty alone). The result is the forecast at the posterior mode with the draws, for fan charts and
  quantiles. Checked on an AR(1): each draw's forecast is rho^h y_T at that draw, exactly, and the one-step
  predictive mean and variance are E[rho] y_T and E[sd^2] + Var(rho y_T).

- **Stochastic simulation of models** (`model.solve(data, span, draws=1000, seed=1)`, which was a placeholder):
  - **Backward-looking models:** each draw adds errors to the fitted equations as add factors and solves the model
    again, nonlinear equations included, all draws at once period by period. Errors are normal (each equation's
    standard error, the residuals' correlations) or bootstrapped (whole periods of residuals). `shock_sd` sets an
    equation's size, or gives one to a calibrated equation; `coefficients=True` draws the coefficients too. Fixed
    variables and targets hold in every draw; draws that cannot be solved are dropped and counted.
  - **First-order solutions** (`solver=FirstOrder()`, `LinearSolution.solve(..., draws=, shock_sd=)`): shocks drawn
    in each period around the deterministic path, exactly, by the decision rules.
  - **Forecasts from an estimated model** (`fit.forecast(span, draws=1000)`, and a new dialog, Model → Forecast from
    estimated model): the estimated shocks drawn, with each draw's observed series through the measurement
    equations.
  - **Results:** the `Solution` keeps the deterministic path and gains the draws, with `quantiles()`, `mean()`,
    `sd()`, `bands()` and `fan()`, and Fan chart and Quantiles tabs. Model → Solve has the settings under
    Stochastic simulation.
  - **Checked** against the exact distributions (an AR(1)'s forecasts, a log random walk's lognormal quantiles, the
    one-step variance with coefficient uncertainty), statsmodels' VAR forecast MSE, and the theoretical moments of a
    first-order solution.

- **TRAMO-SEATS** (`TramoSeats()`, and in the Seasonal adjustment dialog): the Bank of Spain's method as Eurostat's
  JDemetra+ 3.9 computes it, run in Java through JPype (`pip install pymetrix[jdemetra]`; Java 21; the JDemetra+
  libraries, about 7 MB, fetched by `pymetrix.jdemetra.install()` and checked against recorded SHA-1s, or found in
  `JDPLUS_PATH`). JDemetra+'s predefined specifications RSA0 to RSAfull, changed by settings for logs, the ARIMA model,
  outliers (types, critical value), trading days (working days, six contrasts, or chosen by an F-test), Easter, and
  SEATS's approximation, algorithm and root boundaries. Adjusted series record the model, outliers and calendar effects
  TRAMO found. Checked against statsmodels (TRAMO's ARIMA, trading-day and leap-year estimates, to 2e-4 and 1e-5) and
  against x13as's SEATS given the same model (adjusted series to 2e-4, monthly and quarterly, logs and levels).

- **A research lab book, as a plugin** (`pymetrix-labbook`, in `packages/pymetrix-labbook`; first version):
  - **What it records:** installed, it keeps an append-only, hash-chained record of every session in the project:
    each step run, failed, cancelled, undone or removed, and each open and save.
  - **`pymetrix-labbook verify`** checks the chain, the save log's coverage, the steps against those recorded at
    the last save, and a replay against the results' recorded fingerprints. It lists what it could not check.
  - **`pymetrix-labbook show`** lists the entries.
  - **No data values:** entries hold code, names, versions and fingerprints, so a lab book can go to a reviewer
    without the data.
- **Extension API 1.3, for plugins that keep a record of the work** (the planned lab book,
  [docs/design/research-audit.md](docs/design/research-audit.md)):
  - **Session observers:** a plugin registers a `SessionObserver` in the new `observers` group. Every session, in
    scripts, notebooks, the app and the MCP server, then tells it each step run, failed, cancelled, undone or
    removed, and each project opened or about to be saved, with whether it happened during work, a recalculation or
    opening.
  - **Project attachments:** `session.attachments` holds a plugin's own files, saved in the project and reopened
    with it, and kept unchanged when the plugin is missing.
  - **A save log:** each project records every save, with the pymetrix version and the recording observers
    active, so work done without them shows.
- **Ready for PyPI, not published:** the package has PyPI metadata and its own page (`PYPI.md`), the source
  distribution holds the package and its licence only, and a release workflow (run by hand) publishes through
  TestPyPI to PyPI. Publishing waits for the decision on when pymetrix goes public.
- **The procedures' other settings in their dialogs**, on the Advanced tab: a condition on the rows for equations
  (`where`); bandwidth, leads and lags (and a quadratic trend) for cointegrating regressions; the TVP-VAR's prior scales
  and thinning; the proxy SVAR's block length, band level and seed; multi-start ML for structural VARs; for sign
  restrictions the rotations, tries, their own seed and narrative restrictions; forecast draws and seeds for Bayesian
  VARs and threshold models; the model confidence set's bootstrap; the event study's reference period, staggered DiD's
  window and seed, propensity weighting's seed, synthetic DiD's standard error, placebos and seed; other statsmodels
  arguments for seasonal ARIMA, theta and STL forecasting; local projections' band level and the smooth projection's
  penalty, order and folds; Chow-Lin's rho (fixed, or its range); the Kalman filter's start, presample, iterations,
  bounds, workers, seed and balanced growth for model estimation; Metropolis-Hastings' proposal scale, tuning and mode
  jumps; identification at other values; contributions on newer data; template shock sizes; equation labels, starting
  values, bounds and priors; per-entity sample windows; the panel axis for variation; state-space covariance and
  iterations; the density grid; occbin's horizon and iterations; growth-at-risk quantiles; the data check's outlier
  threshold and strict mode; IRIS's trend base; a report's date and figure width (and any result in a report); a
  figure's dates, theme, columns, shared scale, bar labels and house style. Each is passed only when changed.
- **Fixed:** the impulse-response and variance-decomposition dialogs' seed set only the bootstrap; the sign-restriction
  rotations always used seed 0. They now have their own seed. Model estimation with several starts was always seeded 1;
  the seed is now a setting (default 1).

- **The learners' settings in their dialogs:** random forests gain depth, split size, criterion, bootstrap, sample share and
  pruning; boosted trees their loss (absolute error, Huber, a quantile), leaf and split sizes, the share of regressors per
  split, early stopping and pruning; neural networks their activation, solver, learning rate, batch size, iterations and
  early stopping; nearest neighbours the distance's power. Each is passed to scikit-learn only when changed, so fits
  that leave them alone are as before. Checked against scikit-learn given the same arguments directly.

- **Every seasonal-adjustment setting in the dialog:** X-13 runs the `x13as` program directly on a specification pymetrix
  writes, so the dialog (and `X13(...)`) offers X-11's mode (multiplicative, additive, pseudo-additive, log-additive),
  seasonal and Henderson trend filters and sigma limits, SEATS instead of X-11, an ARIMA model or the automatic search's
  limits, outlier types and critical value, Easter effects, a forecast extension and any other specification text; the
  rarer ones on the Advanced tab. With the defaults the result is as before (statsmodels' specification). STL gains its
  low-pass filter, LOESS degrees, jumps and iterations; the classical decomposition one-sided averages and trend
  extrapolation. Checked against the program run on specifications written by hand, and statsmodels called directly.
- **Fixed:** an optional number left empty in a dialog or Jupyter form (STL's trend length, "empty = automatic") was sent
  as the spin box's minimum (3); it is now left empty.

- **A desktop shortcut:** `pymetrix-gui --shortcut` makes one to the app, on the desktop and in the Start menu on
  Windows (run by pythonw.exe, so no console window; the desktop Windows reports, a OneDrive desktop included), as a
  `.desktop` launcher on Linux and a `.command` file on macOS. It starts the Python it was made with.

- **Bayesian single equations** (`Equation(...).fit(data, method=Bayes(prior={"x": Normal(0.5, 0.1), "sigma":
  InvGamma1(1, 1)}))`, which was a placeholder): Gaussian, logit, probit and Poisson equations; Gibbs sampling where it is
  exact, random-walk Metropolis-Hastings otherwise; posterior summaries with HPD intervals, R-hat and effective sample
  sizes. `Bayes` gains `chains` and `burn`, and its `sampler` defaults to "auto" (NUTS was never built). Checked against
  exact posteriors (least squares and the t distribution; quadrature; two-parameter grids).
- **Fixed:** an `InvGamma1` prior with a mean of 1 or more was a near point mass (its degrees of freedom were solved as
  about 2e17); it now has the mean and standard deviation asked for. Priors with smaller means were right.

- **Monte Carlo experiments** (`MonteCarlo(dgp, estimator, reps=1000, truth={...}).run()`, which was a placeholder):
  one random stream per replication, so results do not depend on `workers` (replications run in other processes when
  the functions can be sent there); moments, bias, RMSE and rejection rates; failed replications counted and kept.

- **Maximum-likelihood factor analysis** (`FactorModel(variables, factors, rotation="varimax")`, which was a
  placeholder): loadings, uniquenesses (Heywood cases flagged), varimax and promax rotations with the factors'
  correlations, regression and Bartlett scores, and the test that the number of factors is enough. Matches R's
  factanal (to 1e-8 on mroz with two factors).

- **The independent normal-Wishart prior for Bayesian VARs** (`Bayes(prior=NormalWishart(tightness=0.2,
  cross_tightness=0.5))`; Independent prior in the VAR dialog): other variables' lags shrunk harder than a variable's
  own, sampled by Gibbs. Responses, forecasts and conditional forecasts work as for the conjugate prior. Checked
  against an AR(1)'s exact posterior (by quadrature) and a two-variable VAR's closed-form posterior given Sigma.

- **Panel unit roots and cointegration:** Breitung's test (`data.test("breitung", "y")`; `robust=True` for Breitung and
  Das's version under a common factor); CIPS on unbalanced panels (as R's plm computes it, with the null simulated for the
  same missing periods); and panel cointegration tests: Pedroni's seven statistics (`"pedroni"`, variables as
  `"y, x1, x2"`), Kao's (`"kao"`) and Westerlund's error-correction tests Gt, Ga, Pt and Pa (`"westerlund"`). Checks:
  CIPS against plm's `cipstest` (balanced and unbalanced, to 1e-9); Breitung written out by hand and its null
  distribution simulated; Pedroni's simulated moments against his Table 2; Kao's and Westerlund's regressions against
  statsmodels; every test's size and power in repeated samples.

- **3-D charts, contours, violins and hexagonal bins.** New figure layers, each one matplotlib call: `Surface`
  (`plot_surface` with a colour bar, or `plot_wireframe`), `Line3d` and `Scatter3d` on 3-D panels (a panel's
  `projection="3d"`, set by its layers), with dates on any axis labelled as periods; `Contour` (`contourf` or
  `contour`, `levels`, `clabel`, a colour bar); `Violin` (`violinplot`) and `Hexbin` (`hexbin`). A table over a
  grid is drawn by `surface_figure(values, kind="surface" | "wireframe" | "contourf" | "contour")`; `data.plot`
  takes `kind="violin"`, `"hexbin"` (`bins`: hexagons across) and `"surface"` (series side by side by date, e.g. a
  yield curve over time), as does **Plot series**. `fig.view_init(elev=, azim=)` turns a 3-D panel, recorded as a
  style call; `set(zlim=...)`, `ticks(axis="z")` and `line(...)` on 3-D lines work too. All save in projects and
  saved results (periods as values too), and `matplotlib_code()` writes each. In results: a volatility surface's
  **Surface** tab (`FittedVolSurface.plot_surface`, `grid`; by strike or log moneyness) and a TVP-VAR's
  **Responses over time** (`irf_surface`, `irf_over_time`, `irf_paths`: median responses at every date). In the
  app, clicking a 3-D panel to place a note says notes go on flat panels; the hover readout reads 3-D panes.
  Checks: surface faces against the mean of each cell's corners, wireframe rows against the table, contour lines on
  their circles, filled levels against MaxNLocator, violin outlines against SciPy's Scott-bandwidth KDE, hexagon
  counts against nearest-centre assignment, the standalone scripts drawing the same artists; TVP-VAR paths against
  `irf` at each date, the vol surface against Hagan's SABR formula.
- **Fixed:** a scatter plot or 2-D histogram whose series had whole-number labels got whole-number x ticks.

- **Quantile VARs, spatial models and the full VECH:** recursive quantile VARs with quantile impulse responses
  (`QuantileVAR`, Chavleishvili and Manganelli; quantile regressions as linear programmes); spatial weights
  (`knn_weights`), Moran's I and Anselin's LM tests (`spatial_diagnostics`) and the spatial lag and error models by
  maximum likelihood (`spatial_regression`); and the full VECH in the finance plugin (`mgarch(..., model="vech")`).
  Checks: each quantile equation against statsmodels' QuantReg, the median system near the OLS VAR, a location-scale
  model's quantile responses; spatial estimates, standard errors, log likelihoods, Moran's I and all four LM tests
  against spreg (to 1e-6 or better), the weights against libpysal; the full VECH's likelihood by a loop and no lower
  than the diagonal VECH it nests.

- **Spectra and long memory:** the periodogram and Daniell-smoothed spectrum (`periodogram`, `spectrum`), the
  cross-spectrum with coherence, phase and gain (`cross_spectrum`), the GPH and local Whittle estimators of d (`gph`,
  `local_whittle`) and ARFIMA by conditional sum of squares (`arfima`). Checks: the periodogram against scipy, the
  variance recovered, an AR(1)'s spectrum, coherence and gain for a noisy multiple, GPH by hand, the local Whittle
  objective minimised, d recovered from a simulated fractional process by all three, ARFIMA on an AR(1) finding no
  long memory.

- **Larger and switching VARs:** a Bayesian VAR with stochastic volatility (`BVARSV`, Gibbs sampling with a Minnesota
  prior), factor-augmented VARs (`favar`, Bernanke, Boivin and Eliasz's two-step method, with their slow-variable
  purge; responses of any panel series), and Markov-switching VARs (`MSVAR`, switching intercepts, variances or
  everything, by EM with an exact-likelihood polish). Checks: the BVAR-SV coefficient step equal to SUR least squares
  and its covariance to the SUR formula, a volatility break recovered; the FAVAR's factors spanning the principal
  components, loadings and responses by hand, the purge leaving nothing to the observed variable; a one-equation
  MS-VAR equal to statsmodels' MarkovRegression (likelihood to 1e-6), a simulated VAR's regimes recovered.

- **More SVAR identification:** the max-share shock (`var.identify(MaxShare("tfp", horizon=40))`; with
  `zero_impact=True` a Barsky-Sims news shock), narrative sign restrictions (`SignRestrictions(...,
  narrative_signs=..., narrative_major=...)`, Antolín-Díaz and Rubio-Ramírez, with their importance weights), and
  identification through heteroskedasticity (`var.heteroskedasticity_svar(["1984Q1"])`, Rigobon). Checks: max share at
  horizon 0 equal to Cholesky with the target first, no rotation explaining more, its share equal to the FEVD's; the
  news shock's zero impact; every narrative rotation satisfying the narrative by hand and drawn from the sign set; both
  regimes' covariances reproduced exactly and a simulated impact matrix recovered.

- **Cointegrating regressions and heterogeneous panels:** FMOLS, CCR and DOLS (`cointegrating_regression`, Estimate →
  VARs and local projections → Cointegrating regression…), the Gregory-Hansen test with a break (`gregory_hansen`),
  and for panels the mean group (`mean_group`), common correlated effects (`cce`, mean group or pooled) and pooled
  mean group (`pooled_mean_group`, with a Hausman test against MG). Checks: the long-run covariance and all three
  estimators (with DOLS's lead and lag choice) against arch to rounding; Gregory-Hansen by statsmodels' ADF at its
  break and its simulated critical values against the published table; MG by per-unit OLS, CCE by hand and near the
  truth where MG is biased, standard errors against their spread; PMG's likelihood by hand, equal to one unit's own
  ARDL long run, and its Hausman test's size.

- **Autoregressive conditional hazard** (`ach`, Hamilton and Jorda 2002): the probability of an event each period as
  the inverse of an ACD of the durations between events. Checks: its likelihood by a loop over periods, no duration
  dependence found in independent events, simulated parameters recovered.

- **Synthetic difference-in-differences** (`synthetic_did`, Estimate → Treatment effects → Synthetic DiD…): unit and
  time weights (Arkhangelsky, Athey, Hirshberg, Imbens and Wager 2021), with DiD and the synthetic control beside it,
  jackknife or placebo standard errors. **Diffusions by exact maximum likelihood** (`diffusion`: Vasicek, CIR, GBM).
  Checks: the estimate equal to weighted two-way fixed effects (statsmodels WLS), DiD by hand, the unit weights
  against an independent Frank-Wolfe solution, DiD biased and SDID right in a factor model; Vasicek equal to the mapped
  AR(1) least squares, CIR's likelihood by scipy's noncentral chi-squared, GBM's closed forms and their standard
  errors against the spread over simulated samples.

- **The rest of the Martin et al. list:** threshold VARs (`threshold_var`, with Tsay's bootstrapped sup-LR test),
  bilinear models (`bilinear`), VARMA by conditional maximum likelihood (`varma`), BHHH (`ML(optimizer="bhhh")`) and
  outer-product-of-gradients standard errors (`ML(cov="opg")`), and in the finance plugin dynamic equicorrelation and
  the diagonal VECH (`mgarch(..., model="deco" | "diagonal_vech")`). Checks: the threshold VAR's regimes and its
  brute-force threshold by hand, the bootstrap test rejecting a threshold model but not a linear VAR; the bilinear
  residual recursion by a loop; VARMA(p, 0) equal to the VAR and VARMA(1, 1) near statsmodels' VARMAX; BHHH reaching
  Newton's estimates and OPG errors by hand; DECO with two series equal to DCC and its closed-form likelihood against
  numpy; the diagonal VECH's likelihood by a loop and no lower than the scalar BEKK it nests.

- **Unit roots and regression details:** Ng and Perron's M tests (`data.test("ng_perron", ...)`: MZa, MZt, MSB, MPT),
  Elliott, Rothenberg and Stock's point-optimal test (`"ers_point_optimal"`), both on GLS-detrended data with the
  modified AIC, and Perron's test with a break at a known date (`"perron_break"`, models A, B, C, simulated p-values);
  AR(1) errors by **Cochrane-Orcutt** and **Prais-Winsten**, **multiplicative heteroskedasticity** (Harvey), the
  **stochastic frontier** (normal-half-normal, production or cost, with JLMS inefficiencies), as methods for
  `Equation.fit`; and the non-nested **J** and **Cox** tests (`j_test`, `cox_test`). Checks: the GLS detrending through
  arch's DF-GLS, the M statistics by formula, their size and power, Perron's tables, Cochrane-Orcutt against
  statsmodels' GLSAR, Prais-Winsten against GLS, the likelihoods maximised independently, J and Cox against
  statsmodels.
- **Nonparametric estimation:** kernel densities (`kde`; Silverman, Scott or cross-validated bandwidths), kernel
  regression (`kernel_regression`; Nadaraya-Watson or local linear, several regressors, cross-validated bandwidths),
  Robinson's partially linear model (`partially_linear`), nonparametric autoregression, and the semiparametric GARCH
  of Engle and Gonzalez-Rivera (`semiparametric_garch`, bootstrap standard errors); Estimate → Nonparametric in the app.
  Checks against statsmodels' KDEUnivariate, KDEMultivariate and KernelReg (to rounding), by hand, and in repeated
  samples.
- **Counts and durations:** INAR(1) (`inar`), the Poisson autoregression with feedback (`poisson_autoregression`), the
  ACD model with exponential or Weibull errors (`acd`); the dynamic probit is a probit with a lagged outcome. Checks:
  transition probabilities by explicit summation, likelihoods by loops, the exponential ACD as twice the Gaussian
  GARCH quasi-likelihood on square-root durations (near arch), simulated parameters recovered.
- **More nonlinear and VAR tools:** generalised impulse responses (`var.girf()`, Pesaran-Shin), GARCH-in-mean
  (`garch_in_mean`), and Lee, White and Granger's neural-network test (`fit.test("neural_network")`). Checks: each
  generalised response equal to the Cholesky response with that variable first, GARCH-in-mean without its in-mean term
  equal to arch's GARCH, the neural-network F by hand with its size and power.

- **Structural identification:** SVARs by restrictions on A and B (`var.identify(AB(A=..., B=...))`, Estimate →
  Structural VAR (A-B)…), estimated by maximum likelihood with standard errors and the over-identification LR test
  (`var.structural()`), and with zeros of the long-run effects combined with zeros on impact (`AB(B=..., long_run=...)`);
  **FIML** for simultaneous systems (`System(...).fit(data, method=FIML())`, identities included); and tests on a
  VECM's cointegrating vectors (`vecm.test_beta(H)`) and of **weak exogeneity** (`vecm.test_weak_exogeneity([...])`).
  Checks: recursive A-B and B models equal to Cholesky, long-run restrictions equal to Blanchard-Quah, an
  over-identified model against statsmodels' SVAR, B's standard errors against b / sqrt(2T), the LR test's size;
  FIML equal to 2SLS when exactly identified, to LIML (linearmodels) beside the reduced form and to iterated SUR without
  endogenous regressors; the VECM's reduced-rank regression equal to statsmodels' in all five cases, the beta test
  against least squares with the restricted beta, weak exogeneity against a conditional regression by hand, and both
  tests' size.
- **Fixed:** the Proxy SVAR dialog failed: it named a recorded call that does not exist, and a target variable clashed
  with the call builder's own `target`. A contract test now checks every call a dialog names.

- **GMM flavours:** the continuously updated estimator (`GMM(..., steps="cue")`, Hansen, Heaton and Yaron); the
  **C test** (`c_test`, difference-in-Sargan) that some instruments are valid; the weak-instrument-robust
  **Anderson-Rubin** test and confidence set (`anderson_rubin`, `anderson_rubin_set`) and **Stock-Wright S test**
  (`stock_wright`); and **classical minimum distance** (`minimum_distance`). Checks: CUE with homoskedastic S equal to
  LIML (linearmodels), the C test by hand and near linearmodels' `c_stat`, an invalid instrument found, the
  Anderson-Rubin F by hand with its 5% size kept under weak instruments where the t test over-rejects, Stock-Wright as
  the CUE objective by hand, and minimum distance equal to GLS in the linear case.

- **Estimation by simulation** (`indirect_inference`, `simulated_moments`, `efficient_method_of_moments`): for models
  that can be simulated but whose likelihood cannot be written down, the parameters whose simulated data match the
  real data's auxiliary estimates, moments or auxiliary score, with common random numbers, sandwich standard errors
  inflated by (1 + 1/S) and the J test; continuous-time models through their discretisation. Checks: an MA(1) by
  indirect inference and EMM near exact maximum likelihood (statsmodels), geometric Brownian motion equal to its
  closed-form estimates, a normal's mean and standard deviation by simulated moments, and the standard errors against
  the spread of estimates over 40 simulated samples.

- **TVP-VARs with stochastic volatility** (`TVPVAR`, Estimate → TVP-VAR with stochastic volatility…): Primiceri's
  model by Gibbs sampling in Del Negro and Primiceri's corrected order, forward-filtering backward-sampling paths and
  Kim, Shephard and Chib's mixture for the volatilities, priors from a training sample; volatility paths,
  coefficient paths and responses at any date. Checks: the mixture against the exact density, the sampler against
  the Kalman smoother, a simulated volatility break recovered.

- **Smooth local projections** (`LocalProjection(...).fit_smooth(data)`; Smooth in the Local projection dialog):
  Barnichon and Brownlees's penalised B-spline response, all horizons together, the penalty chosen by blocked
  cross-validation, Newey-West errors. Checks: with no penalty exactly the ordinary local projection, a large penalty
  giving a straight line (or a constant), errors by hand, and responses nearer the truth on average.
- **Covariates in staggered DiD** (`staggered_did(..., covariates=["x"], estimator="dr")`): conditional parallel
  trends by outcome regression, inverse probability weighting or the doubly robust estimator (Sant'Anna and Zhao
  2020). Checks: an uninformative covariate reproduces the unconditional estimate exactly, outcome regression by hand,
  and the true effect recovered where trends depend on the covariate and the unconditional estimate is biased.

- **Staggered difference-in-differences** (`staggered_did`, Estimate → Staggered DiD…): Callaway and Sant'Anna's
  group-time effects against never-treated or not-yet-treated units, aggregated by event time, cohort and overall
  with a bootstrap over units, or Sun and Abraham's interaction-weighted event study; and the **Goodman-Bacon
  decomposition** (`bacon_decomposition`, Estimate → Bacon decomposition…) of the two-way fixed effects estimate.
  Checks: ATT(g, t) by hand, Sun-Abraham equal to Callaway-Sant'Anna, the decomposition summing exactly to the
  two-way estimate (itself checked against linearmodels), and growing effects recovered where two-way fixed effects
  understates them.

- **Proxy SVARs** (`var.proxy_svar(data, "ff4", target="ff")`, Estimate → Proxy SVAR…): one shock of a fitted VAR
  identified by an external instrument (Stock and Watson; Mertens and Ravn), as a one-standard-deviation or unit
  shock, with moving-block bootstrap bands (Jentsch and Lunsford) and the first-stage F. Checks: the impact by hand,
  a simulated non-recursive SVAR's column recovered, and the identity with LP-IV at horizon 0.

- **Smooth-transition autoregressions** (`SmoothTransitionAR`, Estimate → Smooth-transition autoregression…):
  logistic (LSTAR) and exponential (ESTAR) transitions in a lag of the series, by least squares concentrated on the
  transition's speed and centre, with Gauss-Newton standard errors, simulated forecasts and Teräsvirta's test.
  Checks: known parameters recovered, the very-fast-transition limit equal to the threshold AR, the covariance against
  the Hessian.
- **LP-IV:** `LocalProjection(..., instruments=["surprise"])` estimates each horizon by two-stage least squares, the
  shock instrumented externally (Stock and Watson 2018), with the same HAC or clustered errors and each horizon's
  first-stage F; the dialog has the field. Checks: the shock as its own instrument equals least squares, each
  horizon against linearmodels' IV2SLS, the true responses recovered where least squares is biased.
- **Growth at risk** (`growth_at_risk`, Estimate → Growth at risk…): quantile local projections of future growth on
  today's conditions, rearranged so quantiles never cross, and Azzalini-Capitanio skewed t distributions matched to
  each period's quantiles, giving growth at risk, expected shortfall and the predictive density (Adrian,
  Boyarchenko and Giannone 2019). Checks: the skewed t against Student's t, the skew normal and numerical
  integration, parameters recovered from quantiles, the regressions against statsmodels' QuantReg.

- **Panel VARs** (`VAR(..., by="firm").fit(panel)`, Estimate → Panel VAR…): GMM on forward orthogonal deviations
  with the lags' levels as instruments (Love and Zicchino; Abrigo and Love's pvar), two-step with Hansen's J when
  over-identified, or least squares on demeaned data; standard errors clustered by entity; Granger tests, impulse
  responses with bands from coefficient draws, and variance decompositions. Checks: the within estimator against
  linearmodels' PanelOLS, the GMM estimate against linearmodels' IV2SLS on hand-transformed data, the transform by its
  formula, and known coefficients recovered.

- **More outside checks for the finance plugin:** Black-Litterman against He and Litterman's (1999) published
  seven-country example, the CDS bootstrap against QuantLib's hazard-rate bootstrap, and the copula densities against
  statsmodels' copulas (`test_finance_outside.py`); [docs/validation.md](docs/validation.md) updated.

- **The plugins' public surface:** `pymetrix.extension` now offers, under stable public names, everything a plugin
  needs from pymetrix's inside (registration, recorded calls, dialogs, views, progress, help, report templates,
  example datasets, numerical helpers), and `pymetrix.extension.STABLE` names the modules a plugin may import from
  besides. The finance plugin now uses only these, and a test keeps it so.
- **A plugin template:** `pymetrix-new-plugin pymetrix-myname` (or `pymetrix.plugin_template.new_plugin`) writes a
  working plugin laid out as the finance plugin is, with its entry points, a call, a result with views, a dialog,
  help, tests, a tutorial run as a test and a validation table; a test makes one and runs its tests in a fresh process.
- **Fixed:** the finance plugin reported pymetrix's version as its own.
- **Fixed: Cancel in the status bar could be lost** when pressed before the progress bar had appeared (it
  appears after 0.4 s): the step then ran on. It now stops the running step either way. This also fixes a test
  that hung intermittently.
- **The finance plugin is its own package, `pymetrix-finance`** (in `packages/pymetrix-finance`, with its own
  version, README, tests, tutorial and dataset script), installed with `pip install pymetrix[finance]` or from its
  folder. It loads through entry points like any plugin. `from pymetrix import value_at_risk` (and the other 39
  finance names) still works when it is installed; the canonical import is `from pymetrix_finance import ...`.
  Recorded projects replay unchanged (their steps are by call ID); without the plugin, their finance steps are kept
  as missing steps, as for any plugin.
- **Extension API 1.2** also adds **example datasets** from plugins (`register("datasets", ExampleDataset(...))`):
  `Dataset.example("french")` finds the finance plugin's six datasets, which moved with it.

- **Validation of the finance plugin** documented: [docs/validation.md](docs/validation.md) lists what each finance
  method is checked against (arch, QuantLib, statsmodels, Hull's examples, closed forms, simulation) and how closely,
  and which finance pieces have no outside reference yet.
- **Volatility surfaces** in the finance plugin (`vol_surface`, Finance → Volatility surface…): SVI (with butterfly
  and calendar arbitrage checks) and SABR smiles for each expiry, and the Heston model for the whole surface, priced
  by Lewis's integral of the characteristic function; `heston_price`. Checks: Heston prices against QuantLib's
  analytic engine and the Black-Scholes limit, SABR against QuantLib's `sabrVolatility`, SVI's butterfly function
  against the density from call prices, each model recovering the parameters of its own quotes, and the arbitrage
  checks catching calendar arbitrage. QuantLib is a test oracle only, not a dependency.

- **Extension API 1.2** (additive): plugins add report templates (`register("report_templates", ReportTemplate(...))`),
  which `report_templates()` and `Report.from_template` find beside the built-in ones.
- **The finance plugin, finished off:** a `finance_summary` report template; the Finance workspace's toolbar has
  Returns, Performance, Value at risk, Portfolio, Strategy backtest and Yield curve; a help overview, "Finance: which
  tool for which question"; and the tutorial as a notebook (`packages/pymetrix-finance/docs/tutorial.ipynb`, made from the page by
  `scripts/make_tutorial_notebook.py` and checked against it).
- **Fixed:** the value-at-risk graph failed to draw (saving a report, exporting the figure): a scatter of dated points
  now draws on the dated axis. A test now draws every finance result's figures.

- **Credit risk** in the finance plugin: `merton` (Finance → Merton distance to default…) backs out asset values and
  volatility from equity by the KMV iteration and gives distances to default and default probabilities by period;
  `merton_point` does one firm at one date; `cds_curve` (Finance → CDS curve…) bootstraps piecewise-constant default
  intensities from CDS spreads with quarterly premiums and accrual. Checks: Merton's equations round-tripped through
  Black-Scholes, a simulated firm's assets and volatility recovered, every CDS repriced at zero, and the credit
  triangle exact on a flat curve.

- **Copulas** in the finance plugin (`copula`, Finance → Copula…): Gaussian and t for any number of series, Clayton
  and Gumbel for two, on the ranks of the returns or of GARCH(1,1) standardised residuals; tail dependence, draws, and
  a portfolio's VaR and ES by simulation through empirical or normal margins. Checks: densities integrating to one
  and matching scipy's, parameters recovered from simulated copulas, the samplers' Kendall's tau, the t's tail
  dependence by simulation, and the Gaussian copula giving the closed-form normal VaR.

- **Fat tails and asymmetry in the finance plugin's GARCH:** `value_at_risk(method="garch" or "fhs", dist="t",
  asymmetric=True)` uses Student-t errors and the GJR form (the Value at risk dialog has both); `mgarch(model="adcc")`
  is Cappiello, Engle and Sheppard's asymmetric DCC, and `asymmetric=True` gives the correlation models GJR
  volatilities. Checks: the GJR-t fit and its VaR against the arch package, the variance update by hand, the
  standardised t's VaR and ES by simulation, asymmetric DCC nesting DCC, its stationarity bound, and its asymmetry
  recovered from a simulated process. The tutorial now shows that on monthly market returns t-GJR GARCH cuts the
  violations but still fails at 99%, while Cornish-Fisher passes.

- **Portfolio strategy backtests** in the finance plugin (`backtest_strategy`, Finance → Strategy backtest…): equal
  weight, inverse volatility, minimum variance, maximum Sharpe and risk parity, rebalanced every k periods from the
  window before, with drifting weights, turnover and proportional costs; `risk_parity_weights` (Spinu's convex
  problem) and `black_litterman`; `optimize_portfolio(mu=...)` takes given means. Checks: buy-and-hold and costs by
  hand, no look-ahead against the public optimiser, equal risk shares, and Black-Litterman's two forms agreeing.
- **Bonds and the dynamic Nelson-Siegel model** in the finance plugin: a fitted yield curve's `forwards`,
  `discount`, `bond` (price, yield, duration, convexity, Fisher-Weil duration) and `par_coupon`; and
  `dynamic_nelson_siegel` (Finance → Dynamic Nelson-Siegel…), estimated in one step by maximum likelihood with the
  Kalman filter. Checks: forwards averaging to yields, par bonds at 100, durations against finite differences, the
  likelihood against statsmodels' Kalman filter with missing yields, simulated parameters recovered.
- **Fixed:** maximum-Sharpe portfolios no longer fail when the optimiser stops at a solution for numerical precision;
  the Sharpe ratio's gradient is now exact.

- **Realised volatility and HAR** in the finance plugin (Finance → Realised volatility…, HAR model…):
  `realized_measures` turns intraday returns into daily realised variance, bipower variation, tripower quarticity,
  the Barndorff-Nielsen-Shephard jump test and annualised volatility; `har` fits Corsi's HAR (levels or logs) with
  Newey-West errors, forecasts and a one-day VaR. Checks: each measure against its formula, the measures against a
  simulated diffusion's known variance, the jump test's size and power, the HAR against least squares.
- **Finance example datasets:** `french` (Fama-French factors and portfolios, 1949-2017, real), `treasury_yields`
  (US Treasury constant-maturity yields from FRED, 1982-2026, real), and simulated `yield_curves`, `option_chain`,
  `intraday_returns` and `event_returns` with their true values stated.
- **A finance tutorial** ([docs/finance-tutorial.md](packages/pymetrix-finance/docs/tutorial.md)) from returns to options on those
  datasets; a test runs it as written.
- Portfolio weights the optimiser leaves below 1e-10 are now exactly zero.

- **Risk contributions** in the finance plugin (`risk_contributions`, Finance → Risk contributions…): each asset's
  marginal and component VaR and ES, normal (sample, given or multivariate GARCH covariance) or historical, with
  components that add up to the portfolio's VaR and ES. Checks: Euler's additivity, the marginal values against
  finite differences, and the GARCH case against its own portfolio VaR.
- **Event studies of returns** in the finance plugin (`abnormal_returns`, Finance → Event study of returns…): abnormal returns
  around events from the market model, a factor model or the mean return; AAR and CAAR with a band; the
  cross-sectional, Patell, Boehmer-Musumeci-Poulsen and Corrado rank tests. Checks: the market model against
  numpy, Patell's standardisation against its textbook formula, a known effect recovered, and each test's size near
  5% by simulation.

- **Options** in the finance plugin (Finance → Option price…, Implied volatility…):
  - `option_price`: Black-Scholes-Merton prices and Greeks for European options, and a Cox-Ross-Rubinstein tree
    for American ones (delta, gamma and theta from the tree's nodes, vega and rho by re-pricing);
  - `implied_volatility` of a dataset of quotes (calls, puts or both), with the smile by strike and expiry and the
    volatility surface;
  - checks: Hull's worked examples (including the five-step American put, 4.49), put-call parity, the tree
    converging to Black-Scholes, the Greeks against finite differences, and implied volatility recovering the
    volatility that made each price.
- **Yield curves** in the finance plugin (Finance → Yield curve…): `yield_curve` fits Nelson-Siegel or Svensson
  level, slope and curvature period by period (Diebold-Li), with the decay given or estimated over the sample, and
  forecasts the curve from an AR(1) for each factor. Checks: known factors and decays recovered exactly from
  noise-free curves, the loadings' limits, Svensson nesting Nelson-Siegel, and the AR(1)s against statsmodels.

- **Returns and performance** in the finance plugin (Finance → Returns…, Performance…):
  - `returns(data, prices, kind, periods)` adds log or simple returns from prices;
  - `performance(data, series, rf, benchmark)` gives annual return and volatility, the Sharpe ratio with Lo's
    standard error, the Sortino and Calmar ratios, maximum drawdown, hit rate, skewness and kurtosis, and, against a
    benchmark, alpha, beta, tracking error and the information ratio, with the growth of 1 and drawdowns drawn;
  - checks: every measure from its definition by hand, the drawdown by a plain loop.
- **Factor models and portfolios** in the finance plugin (Finance → Factor model…, Portfolio…):
  - `factor_model(data, assets, factors, rf)`: CAPM or Fama-French regressions with Newey-West errors, the
    Gibbons-Ross-Shanken test that every alpha is zero, and rolling betas;
  - `optimize_portfolio(data, assets, objective)`: minimum variance, maximum Sharpe ratio or a target return, long
    only or within bounds, from the sample covariance or a multivariate GARCH's forecast, with the efficient
    frontier;
  - checks: GRS against alpha's squared t statistic (one asset) and the maximum-Sharpe-ratio identity (several);
    weights against their closed forms, and long-only weights against a grid search.

- **Multivariate GARCH** in the finance plugin (`mgarch`, Finance → Multivariate GARCH…):
  - models: CCC, DCC (Engle 2002), and scalar and diagonal BEKK with variance targeting;
  - results: conditional covariances, correlations and volatilities, forecasts h periods ahead, and a portfolio's
    next-period VaR and ES from the forecast covariance;
  - checks: simulated DCC and BEKK parameters recovered, the DCC likelihood against a plain loop, DCC with no
    dynamics equal to CCC, and forecasts tending to their long-run values.

- **Unit-root and cointegration tests go through the statsmodels adapter** (ADF, KPSS, Engle-Granger, Fisher ADF,
  Johansen), and so does VECM lag selection: their provenance records the arguments statsmodels was given and its
  warnings (KPSS's p-value at the edge of its table). The single-series and Fisher tests take `backend_options` for
  other statsmodels arguments. Johansen results and VECM lag selections now carry provenance, with the rows the
  lagged differences take as presample.
- **Seasonally adjusted series record the statsmodels call that made them** (STL, classical decomposition or
  X-13): in their variables' `extra`, `backend` (library, version and function), `backend_options` (its arguments)
  and any `backend_warnings`, beside X-13's own notes.
- **Dynamic factor nowcasts go through the statsmodels adapter,** and take `backend_options`. **Fixed:** a model
  `update`d with newer figures kept the original fit's provenance; it now records the newer data's sample and
  fingerprint, and that its parameters come from the earlier fit.
- **arch's tests go through the arch adapter:** Phillips-Perron, Zivot-Andrews, DF-GLS, variance ratio and
  Phillips-Ouliaris record arch's arguments and warnings in their provenance and take `backend_options`. `spa`
  results now carry provenance (the losses, settings, seed and arch's arguments), and the model confidence set and
  StepM tables say which arch computed them.
- **Backtests record how their forecasts were made:** the model and method re-estimated, the library and its
  options, the re-estimates' warnings and messages (with the origins where they arose), whether every one converged,
  the forecasts not yet scored (no outcome), and which data were asked for, at each origin too. A forecast
  combination names each member's backend. `nobs` is now the forecasts scored, and `dropped` those still to come.
- **Fixed: real-time backtests of an equation with a lag of its target** failed whenever each vintage ended at its
  origin (the usual real-time case): the empty static forecast past the data's end was compared with the dynamic one.
- **An example plugin for PSETS:** `examples/plugins/pymetrix-psets`, a sketch of a `psets://` reader over PSETS's
  plug-in API, with metadata, vintages, provenance and known publisher issues; `docs/design/plugins.md` explains it.
- **The finance plugin** (`pymetrix-finance`, bundled, built on the extension API alone, with a Finance menu and
  workspace), first piece: **value at risk and expected shortfall** (`value_at_risk`, Finance → Value at risk…).
  - **Methods:** one period ahead, from historical simulation, normal, Student-t, Cornish-Fisher, GARCH(1,1) or
    filtered historical simulation, each from the returns before it only.
  - **Backtests:** Kupiec, Christoffersen (independence and conditional coverage), Engle-Manganelli's dynamic
    quantile test, McNeil-Frey for shortfall, and the Basel zone.
  - **Checks:** the closed forms against numerical integration, the tests against hand counts and statsmodels, and
    the GARCH forecast against the arch package.

- **Extension API 1.1** (additive; plugins written for 1.0 are unchanged): plugins can
  - add a **workspace** to the stage bar (`workspaces`);
  - lay out their own top **menu** (`menus`, `MenuLayout`);
  - dock a **panel** (`panels`, `AppPanel`), which works the app through `AppInterface`: the session, the
    selection, `open_dialog`, `run` (through the app's runner, with Cancel and Undo), `on_history_changed` and
    `status`;
  - give their results a **stage** and **next steps** as class attributes.
- **Plugins in the app:**
  - plugins that could not load are listed in Problems and noted in the status bar at start-up;
  - a plugin's dialogs and History steps say which plugin they come from;
  - Help → Plugins has **Check plugin**, which runs the conformance kit;
  - steps kept as not run show in Problems as such, not as failures.

- **Autosave and recovery:**
  - every 2 minutes, a copy of unsaved work goes to `~/.pymetrix/recovery`, an ordinary project file, leaving the
    session's own path and unsaved state alone;
  - after a crash, the next start offers it back, with its original path, marked unsaved;
  - the copy is deleted when the work is saved, closed or replaced;
  - on or off with File → Autosave for recovery (`pymetrix.recovery`).
- **Load data previews the file:** the first rows, the time column and its frequency, and the column identifying
  the units, guessed and filled into the fields still empty (`pymetrix.io.registry.sniff`).
- **The Estimate dialog suggests a family from the outcome:**
  - a 0/1 variable: logit or probit;
  - a count: Poisson or negative binomial;
  - a few ordered values: ordered models;
  - a pile-up at the lowest value: Tobit.
  It is a hint with a button; nothing changes unless the button is pressed.

- **Projects open when a plugin is missing or disabled:**
  - a step from a plugin that isn't here, a recorded call it provides, or a step using a method it provides is kept
    in the History, marked as not run, with what to install;
  - the steps that use what it made are kept the same way;
  - saving writes them back unchanged;
  - the app says which plugins the project needs.
  (Before, the whole project failed to open.)
- **Plugin manager** (Help → Plugins…):
  - each plugin's status and what it provides, including a data plugin's address schemes;
  - enable, disable, and copy an install command;
  - the app and scripts share the choices.
- **New plugins ask first:** the app asks before a new plugin first runs, from its package's metadata alone; scripts
  load it with a warning.
- **Example cross-section and panel datasets** (`Dataset.example(...)`, File → Load example data):
  - real: `mroz`, `card`, `wage_panel` and `jobtraining` (from linearmodels, NCSA licence), and `nile`, `randhie`,
    `anes96` and `grunfeld` (statsmodels, public domain);
  - simulated, with known effects: `policy_panel`, `rd_sample` and `synth_panel`.
  Each dataset's `license` says where it comes from; `scripts/make_examples.py` builds them.
- **GMM with your own moments** (`GMM(instruments=...)`): linear or explicit-form equations, such as Euler
  equations; one-step, two-step or iterated weighting; robust, homoskedastic or HAC weighting; Hansen's J.
  Matches pymetrix's 2SLS and linearmodels' IV-GMM.
- **Treatment effects** (Estimate → Treatment effects):
  - `did` (two-way fixed effects, clustered errors) and `event_study` (leads and lags around adoption);
  - `rdd` (local linear, triangular kernel, bandwidth by cross-validation, sensitivity to the bandwidth);
  - `treatment_effect` (inverse probability or doubly robust weighting, bootstrap errors);
  - `synthetic_control` (donor weights, gaps and in-space placebo p-values).
- Bai-Perron finds the Nile's break where R's strucchange does (1898), now tested on the bundled data.

- **Limited dependent variables**, by maximum likelihood (`ML()`), written in pymetrix and checked against
  statsmodels and Wooldridge:
  - counts: `NegativeBinomial` (NB2) and `ZeroInflatedPoisson(inflate=...)`;
  - choices: `OrderedLogit`, `OrderedProbit` and `Multinomial` (multinomial logit);
  - censoring: `Tobit(left=, right=)`;
  - sample selection: `Heckman(selection=..., estimator="twostep" | "ml")`, the two-step version with Heckman's
    corrected covariance.
  In the Estimate dialog's family choice and methods.
- **Marginal effects** (`fit.marginal_effects(at="average" | "mean")`) for these and for logit, probit and Poisson:
  0/1 regressors as discrete changes, delta-method standard errors; a Marginal effects tab in the result window.
- **Structural breaks at unknown dates:**
  - the Quandt-Andrews sup-F test (`fit.test("quandt_andrews")`), its p-value from Andrews' limiting
    distribution, simulated;
  - Bai-Perron multiple breaks (`fit.breaks()`, `bai_perron`, **Test → Structural breaks (Bai-Perron)…**), the
    number chosen by BIC, with each regime's coefficients.

- **Seasonal ARIMA** (`SARIMA`, statsmodels' SARIMAX): ARIMA(p, d, q)(P, D, Q)s with optional regressors, exact
  maximum likelihood, standard errors from the numerical Hessian (as R's `arima`), forecasts with intervals.
  Checked against R on the airline model. **The theta method** (`Theta`) and **STL forecasting** (`STLForecast`,
  STL then ARIMA or ETS of the adjusted series) as benchmarks. All three in **Estimate → Time series**, in Forecast
  and in backtests.
- **`Matrix`:** matrices with named rows and columns. Products check that the names meet; `inv`, `pinv`,
  `solve`, `det`, `trace`, `rank`, `chol`, `eig`, `kron`, `vec`, `diag` and `T` keep them. Made from results
  (`Matrix.coef`, `Matrix.cov`, `Matrix.var_lag`, `Matrix.var_sigma`), formulas (`Matrix.design`) and pandas
  tables. Shown as a table and a heatmap in its window.
- **Sign-restricted VARs** (`SignRestrictions`): shocks named by the signs of their responses on impact and for a
  number of periods after, by random rotations (Rubio-Ramírez, Waggoner and Zha 2010). Responses are the median of
  the accepted rotations, with bands across the identified set or across bootstrap or posterior draws; variance
  decompositions use the median-target rotation (Fry and Pagan 2011). Bayesian VARs take them too. In the app:
  SignRestrictions in the Impulse responses and Variance decomposition dialogs.
- **Fixed:** a saved result's table column that held only missing numbers came back as `None` instead of NaN.

- **Edit → Undo** (Ctrl+Z) takes back the last dialog step: the objects as they were before it, and its record (with
  the recalculation it caused) out of the history; up to 20 steps. It refuses, saying why, when objects changed since
  (typed code, or by hand), since that work would be lost; typed code and opening a project start afresh. Ctrl+Z
  stays with text fields, the spreadsheet and Code windows while they have the focus and something to undo. Undo
  also refuses after Clean up history or a freeze, and asks before closing a spreadsheet with edits not applied. In
  code: `session.undo()`.
- **Cancel reaches more long steps:** model estimation, state-space fits and optimal simple rules stop within one
  likelihood (or loss) evaluation, seasonal adjustment between series, and the Kalman filter every 50 periods.
- **matplotlib 3.9 or later is required** (was 3.8): IPython 9 removed `backend2gui`, which matplotlib 3.8's pyplot
  imports when it runs under IPython (a notebook, the Immediate window), so plotting there failed with IPython 9.
  matplotlib 3.9 imports it only from older IPython.

- A step that cannot stop (it has no check for Cancel) and finishes after Cancel was pressed now says it was recorded,
  instead of closing without a word; the status bar shows "Cancelling…" when Cancel came before the progress bar.

- **Help → Keyboard shortcuts:** every shortcut, read from the menus as they are, with the keys of dialogs and
  panels, in a table to filter. The spreadsheet edit line's ✓ and ✗ buttons, its value field and the progress bar
  have names for screen readers.
- **Fixed** (from a review of the background steps, the start page and the stage bar): a dialog's OK pressed again
  while its step ran ran it twice, and Cancel, Esc or closing hid the dialog while the step went on and was recorded;
  OK now waits and Cancel stops the step. Cancel during the automatic recalculation after a step said nothing was
  recorded: the step is kept, and the dialog says the results after it are out of date. Export report's recorded
  code used the template with the outline on. Results not computed yet counted under Estimate on the stage bar,
  whatever their stage. A value typed in the edit line was lost on moving to another cell; it is accepted first. New
  session stacked Start windows and opening a project left one open. Code turned off in All did not hide the code
  panels. The Report panel had no View-menu toggle. Dropping several files opened one without a word, and a dropped
  file's name could clash; the status bar says what was left, and the name is one not taken.

- **Long steps in the background:** a dialog's step runs on a worker thread while the window stays live; after a
  moment the status bar shows its progress, what it is doing and **Cancel**. MCMC chains, bootstrap draws, backtest
  origins, sweeps and rolling windows report progress and stop when cancelled (`pymetrix.progress`); a cancelled
  step is not recorded. While a step runs, starting another is refused and the Immediate window waits.
- **The start page** lists recent projects and the demo projects, introduces the workspaces and Ctrl+K, and points to
  the chart gallery. **Drop a file on the window:** a project opens; a data file opens Load data filled in.
- **Badges on the stage bar:** each stage shows how many of its results are out of date or not computed ("Forecast ·
  3"), with their names in the tooltip; choosing a stage with a badge opens the Problems panel.

- **Next steps:** each result window, and a dataset's, has **Next ▸**: the steps that usually follow (testing,
  plotting, forecasting, backtesting, impulse responses, estimating a model...), each opening its dialog with the
  result filled in and, from a stage's workspace, moving to the step's own.

- **A report outline:** **Add to report** under every figure and table and in every result window collects items from
  any workspace; the **Report** panel (the Report workspace) orders them, adds headings and text, and exports them
  with **Export report**, whose recorded code writes every item out. Saved with the project.
- Dialogs keep the fields that have no editor (set in code) of the command they were opened on, instead of dropping
  them.

- **Workspaces:** a stage bar under the menus, **All · Explore · Estimate · Model · Forecast · Report** (Ctrl+1 to
  Ctrl+6, View → Workspace), shows each stage's menus, panels and most used commands, and hides the rest; **Code**
  (Ctrl+\`) adds the Script, Immediate and Problems panels to any of them; each remembers its arrangement. Find
  anything still reaches every menu.
- **Fixed:** submenus could be deleted under older PySide6 (6.6); they are owned by their menus now. The notebook help
  test skips without IPython.

- **Find anything (Ctrl+K):** **View → Find anything…** searches every menu item (with its place in the menus), the
  workspace's objects and the help pages by a few letters of any words; Enter opens the best match.
- **A series browser:** a dataset window's **Variables** tab is a table to filter (any column) and sort (numbers as
  numbers), with **Plot…**, **Edit in spreadsheet** (only those series), **New selection…** and **Use as working set**
  for the series picked.

- **Working sets for large datasets:** right-click a dataset → **Working set…** (names, patterns or a selection)
  limits the series shown in every dialog's lists, the spreadsheet and the dataset's window, with the count and
  **Show all**; right-click a selection → **Use as working set…**. A view, saved with the project; commands still
  name their series (`Session.set_working_set`, `working_set`).
- **Searchable series lists:** a filter, **Ticked only** and a count above every list of series; single-series
  dropdowns narrow as you type.

- **An edit line in the spreadsheet editor:** above the table, the selected cell's name and value in full; ✓ or
  Enter accepts a new value (Enter moves down a row), ✗ or Esc puts the old one back; a pending edit as typing in a
  cell makes. Apply takes a value still in the line.
- **Help contents, one page per line, in sections that fold:** the contents page's headings open and close their
  sections (Open all, Close all; the open sections are in the link, so Back works); notebooks show the sections as
  `<details>`; the MCP server's contents have every section open.

- **The Data, Estimate, Model and Test menus, grouped by task.**
  - **Data:** Plot series and the spreadsheet editor first; making series; checking data and the **Sample**
    submenu; **Transform** (filters, seasonal adjustment, calendar regressors, temporal disaggregation) and
    **Panel** submenus. 11 lines instead of 21.
  - **Estimate:** single equations first, then **Time series**, **Nonlinear**, **Mixed frequency** and **VARs and
    local projections** submenus, then Forecast and Coefficient plot.
  - **Model:** defining (with an **Import** submenu), solving, estimating, then **State space**, **Policy** and
    **Scenarios** submenus.
  - **Test:** Correlogram moves here from Data.
  - The layout is one table (`pymetrix.commands.menus`); a plugin's commands are added at the end of their menu. The
    notebook's list of commands follows the same order and shows each one's place.

- **Mixed frequencies in a panel:** the date axis takes the finest frequency and every value is drawn at the
  middle of its period when frequencies mix; a span given in another frequency covers whole periods (a monthly
  series clipped to 2023Q4 now keeps November and December).
- **A chart gallery:** `examples/gallery/chart_gallery.ipynb`, every kind of chart with its code; a test runs it and
  each figure's standalone matplotlib script.
- **Fixed, from a review of the charting code:**
  - the standalone matplotlib script for: calls on every panel (`ax="all"`); restyled legends (a repeated
    `frameon`, right-hand entries, series styled before the legend); horizontal panels (the log scale, limits and
    reference lines on the x-axis; one flip for several bar series); shading on panels without a line; a band and
    a line sharing a name; empty histograms and distributions; constant Q-Q data; scatter labels;
  - notes on annual figures (a year typed as 2007 was day 2007 of 1970) and on number axes; a band "through
    2013Q4" on a monthly axis now ends with December;
  - event lines and a fan's start line outside the dates shown no longer widen the axis;
  - Figure style with only a width keeps the figure's height (`layout(figwidth=...)`, `figheight`);
  - Recorded styles refuses steps out of range instead of wrapping round;
  - Figure options keeps every digit of values it fills in;
  - copying a style reaches series on the right-hand axis; a span limits histograms, distributions and x-y
    layers; saved heatmaps keep their column labels' types;
  - empty histograms and distributions draw nothing instead of failing.
- `VAR.fit` has a docstring.

- **The new charts in results:** equations have a **Q-Q plot** tab (a `QQ` layer; `fit.qq_plot()`); impulse
  responses and variance decompositions a **Heatmap** tab (`irf.heatmap()`, `fevd.heatmap()`); any table of numbers
  opens as a heatmap (the **Heatmap** button; `heatmap_figure(result, view=...)`, step `TableHeatmap`).
- **A house style:** `set_house_style(figure)` starts every figure from one figure's style calls, where they apply;
  **Edit… → Use as house style…** (step `UseHouseStyle`); opening a project sets its own.
- **Hover readout** in the app: the nearest point, bar or cell under the mouse, with its period.
- **Reports:** figures drawn at the size they appear (`Report(figure_size=...)`), with alt text from
  `fig.describe(technical=False)` in Word, HTML and Markdown. A result whose first view is its graph adds no
  second figure by default.
- Heatmap values that round to zero are written without a sign ("0.00", not "-0.00").

- **Distributions:** `kind="ecdf"` (an `Ecdf` layer, matplotlib's `ecdf`: empirical distributions in one panel)
  and `kind="hist2d"` (a `Hist2d` layer, `hist2d` with a colour bar); histograms take `bins`. All in **Plot series**.
- **Fixed:** scatter plots of dated series had a date axis (an x tick reading "1970Q1"); their axes are numbers.

- **Coefficient plots and heatmaps.**
  - **Coefficient plots:** `coefficient_figure(results, terms=None, level=0.95)` draws estimates with their
    intervals (an `Estimates` layer, matplotlib's `errorbar`), several results side by side. In the app:
    **Estimate → Coefficient plot…**.
  - **Heatmaps:** `heatmap_figure(table)` (a `Heatmap` layer, `imshow` with a colour bar and the values in their
    cells), and `data.plot(names, kind="correlation")`.
  - Both save with results, style, and write as standalone matplotlib code.

- **Macro charts.**
  - **Contributions:** `data.plot(parts, kind="contributions", total="dy")` and `contributions_figure` draw the
    parts as stacked bars with the total as a line; any part of the total the parts leave unexplained is drawn as
    other. Shock and data decompositions use it.
  - **Fans:** a `Fan` layer (`Fan.normal`, `Fan.from_frame`) draws graded bands at each coverage and a dashed line
    where the forecast starts; `kind="fan"` reads the bands from `_lo90`/`_hi90` or `_se` columns. Forecast rounds'
    fan charts are drawn with it.
  - **Vintages:** `kind="vintages"` and `vintages_figure` draw each release lighter to darker, the latest thickest.
  - In the app: three more kinds in **Plot series**, with a total and coverages.
- **Saving for publication:** `fig.save(path, size=..., dpi=..., transparent=..., editable_text=...)` with sizes
  for journals and slides (`SIZES`) or in mm, cm or inches; PDF and EPS embed TrueType fonts; files record what
  made them (`fig.describe()`) and are the same for the same figure. In the app, **Save…** opens **Save figure**,
  recorded as a step.

- **Managing a figure's recorded styles.**
  - **Listed:** `fig.style_steps()` writes each recorded style call as the call that records it.
  - **Changed:** `without_style(*steps)`, `reorder_style(order)` and `restyle(step, **props)` remove, reorder and
    change them. In the app: **Edit… → Recorded styles…** (Up, Down, Remove, Change…).
  - **Copied:** `fig.with_style_of(other)` adds another figure's style calls where they apply (its series, panels
    and right-hand axes; notes only with `notes=True`), and `style_skipped` lists those left out. In the app:
    **Edit… → Apply style from…**.

- **The rest of matplotlib on figures** (phase D of [docs/notes/figure-styling.md](docs/notes/figure-styling.md)).
  - **Any Axes method:** `fig.call("ticklabel_format", axis="y", style="sci")` records any `Axes` method that styles
    or annotates a panel, checked against its signature. In the app: **Edit… → matplotlib method…**.
  - **Style sheets:** `Theme.from_mplstyle(path)` turns a `.mplstyle` file into a theme; settings the theme has no
    field for are applied while drawing.
  - **matplotlib alone:** `fig.matplotlib_code()` writes a standalone script (NumPy, pandas and matplotlib only)
    that draws the same chart; **Copy matplotlib code** under a figure copies it. `fig.to_matplotlib()` returns the
    drawn figure.

- **Notes, lines and bands on figures** (phase C of [docs/notes/figure-styling.md](docs/notes/figure-styling.md)).
  - **The calls:** `fig.annotate`, `text`, `axhline`, `axvline`, `axhspan` and `axvspan` take matplotlib's own
    arguments, with dates as the data's periods (`xy=("2008Q3", 3.2)`). A band covers its last period in full.
  - **Clicking:** in the app, **Add a note** under a figure waits for a click, then opens **Add a note** with the
    panel, date and value there filled in. **Edit… → Add a note…** opens it without clicking.
  - **Recorded:** each note is recorded as the matplotlib call, in data coordinates, so it replays at any size and
    survives saved results.

- **Every style property matplotlib has, in the app** (phase B of
  [docs/notes/figure-styling.md](docs/notes/figure-styling.md)).
  - **Figure properties:** **Edit… → All properties…** lists every property matplotlib can set on a series' line,
    bars or band, or on a panel's axes. The list is generated from matplotlib's own `ArtistInspector`, with its
    note on each property's valid values. Properties holding the plotted data or positions are left out.
  - **Recorded:** each change is one recorded call, such as `figure_of(fit).line('actual', drawstyle='steps-post')`,
    which replays.
  - **In code:** `pymetrix.figure_properties.properties(part)` and `property_table(part)` give the same catalogue.
  - **Tabs:** **Figure style** now puts its Series, Axes, Legend and Layout sections on tabs (a dialog can ask for
    this with `group_tabs = True`).

- **Styling figures in matplotlib's terms** (phase A of [docs/notes/figure-styling.md](docs/notes/figure-styling.md)).
  - **The calls:** `fig.line(label, ...)`, `bars`, `band`, `set(ax=..., ...)`, `ticks`, `grid`, `spines`, `legend` and
    `layout` take matplotlib's own keyword arguments (`color`, `linestyle`, `yscale`, `loc`, `ncols`...), and
    resolve its aliases (`lw`, `ls`).
  - **Recorded:** each call is kept on the figure as data and applied when it is drawn, so styles replay in scripts
    and projects and survive saved results.
  - **Checked:** matplotlib checks every value when it is set, and an unknown series or panel is named.
  - **In the app:** **Edit… → Style** opens **Figure style** (series, axes, legend, size). It records the chain of
    calls, such as `fit_graph = figure_of(fit).line('actual', color='C3').set(ax=0, yscale='log')`.
- **Fixed:** Figure options (Edit… → Titles, shading and marks) failed to open in the app, because it has no menu
  item to take its title from. A dialog without one is now titled from its command.
- **Fixed:** saved results (`.pymxr`) kept a figure's panels but lost its shading, events, reference lines, scales,
  labels, right-axis label and box-plot layers. They now keep every option, and the styles.

- **Recession dates built in:** the NBER's US business-cycle dates (monthly and quarterly, from 1948) and the CEPR's
  euro-area dates (quarterly, from 1974).
  - **Shading:** `shade="nber"` or `shade="cepr"` shades the recessions on any figure, in code and in both
    dialogs, dated at the chart's own frequency. Only those within the data's dates are drawn.
  - **Spans and dummies:** `recessions("nber", "Q")` gives the spans, and `recession_series(data, "nber")` gives a
    0/1 series for a dataset.
  - **Dating:** a recession runs from the period after the peak through the trough, as in FRED's recession
    indicators.
- **Lines named at their ends:** `line_labels=True` (or the tick box in Plot series and Figure options) writes each
  line's name at its last value, in its colour, instead of in a legend. Names that would overlap are moved apart,
  and the date axis stops where the data do.
- **Shading never widens a chart's date axis,** and a labelled shading is explained in a legend even on a
  one-series chart.

- **Zoom and pan in the app's chart windows:** a toolbar above every figure, with Home, Back, Forward, Pan and Zoom,
  showing the values under the mouse.
- **Conditions in formulas:**
  - **Operators:** `<`, `<=`, `>`, `>=`, `==` and `!=`, joined with `and`, `or` and `not`. Each gives 1 where it
    holds and 0 where it doesn't.
  - **Missing values:** the result is missing where an input is missing.
  - **Examples:**
    - a recession dummy, `recession = gy < 0`;
    - a regressor in the equation, `c ~ y + (gy < 0)`;
    - a switch in a model.
  - **Everywhere formulas go:** they work in data, equations and models (compiled, and with zero derivatives, as
    `step()` has), and print as ≥ and ≠ in algebraic notation.

- **Edit… on every figure in the app**, beside Copy image, Save… and Copy data.
  - **Plot series figures:** it reopens the Plot series dialog as the figure was made. That dialog now also has a
    subtitle and a footnote (Advanced tab).
  - **Any other figure,** such as an equation's Graph tab, a forecast's fan chart or impulse responses: it opens
    the new **Figure options** dialog, filled in with how the figure is drawn now. It covers the title, subtitle,
    footnote, shading, events, reference lines, log scale and limits.
  - **A result's figure is kept as a new object** (`fit_graph`), so the result is untouched. A workspace figure is
    changed in place.
  - **Recorded as a step:** for example `fit_graph = figure_of(fit).with_options(...)`, so it replays.
- **`figure_of(result, tab="Graph")`** returns the figure on a result's tab, to change with `with_options`.
- **Shading from a series in code:** `with_options(shade=data["recession"])` shades a 0/1 series' runs.

- **More kinds of series plot** (`data.plot`, **Data → Plot series…**):
  - **New kinds:** stacked bars (negative values below zero, as contributions to growth are drawn), horizontal bars
    (plain or stacked), stacked areas, step lines and box plots.
  - **A right-hand axis:** `right=` names series to draw against a second scale.
  - **Transforms:** `transform=` plots series as an index (first period shown = 100), their change, their growth
    on the previous period or on a year earlier, or their log. Changes use the periods before the first one
    shown.
  - **Zero line:** added where bars, areas or changes cross zero.
  - **Cross-sections:** their bars are labelled with the entities' names.
  - **Shading:** marks periods such as recessions. Give them as `shade="2008Q3:2009Q2=GFC, 2020Q1:2020Q2"` (labels
    go in the legend), or as the name of a 0/1 series, whose runs of ones are shaded. It works on any figure
    (`fit.plot().with_options(shade=...)`), in every dated panel, and in the dialog.
  - **Events and reference lines:** `events="2016Q2=Brexit vote, 2020Q1=COVID"` draws a labelled dashed line at
    each date, and `reference="2=inflation target"` a labelled dotted line at each value.
  - **Scales:** `log=True` gives a log scale labelled in plain numbers, `ylim=(0, None)` fixes either end of the
    values shown, and `sharey=True` gives one scale for all panels.
  - **Values on bars:** `bar_labels=True` writes each bar's value on it, with one number format per chart, inside
    stacked segments.
  - These are figure options, so they work on any figure and in **Data → Plot series…**.
  - **Fixed:** bars for several series stood at the same place, so each hid the ones behind; they now stand side
    by side.

- **An MCP server for AI assistants** (`pip install pymetrix[mcp]`, the `pymetrix-mcp` command; stage 1 of
  [docs/notes/mcp-server.md](docs/notes/mcp-server.md)).
  - An assistant that speaks the Model Context Protocol (Claude Desktop, Claude Code, VS Code and others) can
    discover pymetrix's commands and run them by name in one session, with the settings their dialogs take.
  - It reads the results as pymetrix's own tables and charts, and searches the help.
  - Every step is recorded, so the session exports as a Python script and saves as a project that replays.
  - Closed by default: no Python, no file writes and no network data without `--allow-code`, `--allow-writes` or
    `--allow-network`, and files only inside the `--root` folder.
  - Tested through the MCP SDK's own client, in-process and over standard input and output.

- **The forecasting course pack is easier to use in Jupyter:**
  - `00_start_here.ipynb`, the README as a notebook, opens formatted; Jupyter opens Markdown files as plain text. It
    is built from the README, and a test keeps the two in step.
  - The labs keep their help front matter in the notebook metadata, so each opens on its title.
  - Each lab says how to get help (`backtest?`, Shift+Tab). Its last cell lists every pymetrix function, class and
    method it uses, with their arguments and what they do.
  - Tests fail if a lab's list falls out of date or a lab is saved with its outputs.
- **`pymetrix.testing.write_api_reference` and `check_api_reference`**, for instructors' own labs:
  - `write_api_reference` runs a notebook and writes that reference cell, from the docstrings;
  - `check_api_reference` says when the cell needs rewriting.
- **Docstrings for `Equation`, `Equation.fit`, `FittedEquation.test`, `PCA` and more**, which had none. The reference
  cells showed the gaps.

- **A guide to wrapping another library** ([docs/wrappers.md](docs/wrappers.md)).
  - It covers the six parts every wrapper has: settings, lazy import, data in, the call, results out, and
    registration.
  - It covers the three shapes a wrapper takes: an estimation method, a model with its own fit, or a test.
  - It ends with a checklist and a table of the bundled wrappers.
  - The worked example, `examples/custom_wrapper.py`, wraps statsmodels' robust regression as an equation method
    and its Theta forecasting method as a model with a menu command. A test checks both against statsmodels
    directly, and shows that they pass the plugin checks and work in scripts, projects and backtests.
- **Fixed:** when scikit-learn is missing, the error now says `pip install pymetrix[ml]`. It used to say
  `pip install sklearn`, which is the wrong package. A test checks that every bundled plugin's library maps to
  an install extra.

- **A forecasting course pack** (`examples/forecasting/`), six lab notebooks following the topics of Elliott and
  Timmermann's *Economic Forecasting* (2016):
  - F1, loss functions and efficiency;
  - F2, comparing forecasts, with a Monte Carlo of the Diebold-Mariano and Clark-West tests;
  - F3, combination and Bayesian model averaging;
  - F4, density and probability forecasts;
  - F5, many predictors: factors, shrinkage and bagging;
  - F6, breaks and nonlinearity.

  Each lab simulates its data from a stated model, so the right answers are known, and checks pymetrix by hand,
  by numerical integration or with statsmodels. The README maps every chapter of the book to pymetrix, the labs,
  and what is not covered yet. The labs run in the test suite and load into the app's help (**Help → Course
  material…**).

- **Shrinkage and machine learning** (Elliott and Timmermann, chapters 5, 6 and 11), a new scikit-learn plugin
  (`pip install pymetrix[ml]`), as estimation methods for equations:
  - `Ridge`, `Lasso` and `ElasticNet`, with the penalty chosen by time-series cross-validation (expanding windows,
    each validated on the periods after it), and `Bagging` (Inoue-Kilian bagged pretest, no extra dependency). They
    return ordinary fitted equations, so forecasts, backtests, combinations and reports work as for least squares.
  - `RandomForest`, `GradientBoosting` (boosted trees), `NeuralNetwork` and `NearestNeighbours` return a
    `FittedLearner`: cross-validated error (the forecast standard error), permutation importance, and static or
    dynamic forecasts (a transformed left-hand side such as dlog(c) is solved for the series). Backtests re-estimate
    them like equations.
  - In the app: the Method list of **Estimate → Equation…** offers them when scikit-learn is installed.
  - Checked against ridge's closed form, the lasso's and elastic net's optimality conditions, cross-validation and
    bagging redone by hand, scikit-learn called directly, and the recursions by hand.
  - The custom-estimator example and Lab 4 now call their teaching estimator `MyRidge` ("My ridge"), since pymetrix
    has a `Ridge` of its own.

- **Asymmetric losses** (Elliott and Timmermann, chapter 2): `linlin_loss` (the check loss, whose optimal forecast is a
  quantile) and `linex_loss`; backtests compare forecasts under them (`compare(..., loss="linex", a=0.5)`), in code
  and in the Compare backtests and Compare many forecasts dialogs.
- **Binary and directional forecasts** (chapter 12): `binary_scores` (Brier score against the base rate's, log score,
  ROC area, hit and false-alarm rates, Kuipers score), `roc_curve`, and the Pesaran-Timmermann test of directional
  accuracy.
- **Romano-Wolf StepM** (section 17.7): which models beat a benchmark, with the familywise error rate controlled
  (through arch).
- **Exponential smoothing** (section 7.5): `ExponentialSmoothing` (simple, Holt, damped trend, Holt-Winters, additive
  or multiplicative errors) through statsmodels' ETS models, with forecast standard errors. **Estimate → Exponential
  smoothing…**; it forecasts from the Forecast dialog and is re-estimated in backtests like equations and VARs, so its
  densities are scored and it can be combined and compared.
- The Combine forecasts and Compare many forecasts dialogs pick backtests from a list.
- **Conditional forecasts** from VARs and Bayesian VARs (section 9.5): `var.conditional_forecast(data, span,
  {"r": {"2025Q1": 2.0, ...}})` forecasts the other variables given paths for some, such as an assumed policy
  rate (Waggoner and Zha 1999: the most likely shocks that meet the conditions, so identification does not matter).
  Bayesian VARs impose the conditions draw by draw, so the intervals include parameter uncertainty.
  **Forecast → Conditional forecast…**, with the scenario syntax `r[2025Q1:2025Q4]=2.0`.
- **Bayesian model averaging** (section 14.6): `BMA(target, candidates)` weighs every regression on a subset of up to
  15 candidate regressors by its posterior probability (Zellner's g-prior, the Fernandez-Ley-Steel benchmark g).
  It reports inclusion probabilities, averaged coefficients and the most probable models, and averages the
  models' forecasts, which backtests evaluate like any other. **Estimate → Bayesian model averaging…**
- Backtests compare forecasts of an expression, such as `dlog(c)`, with the expression evaluated on the latest data.
- **Number of factors** (section 10.3): `pca.factor_criteria()` gives Bai and Ng's IC_p1, IC_p2 and IC_p3 and the number
  of factors each picks; principal-component windows have a Number of factors tab.
- **Threshold autoregressions** (chapter 8): `ThresholdAR(series, lags)` estimates a two-regime SETAR by least squares
  over the threshold and the delay, with Hansen's sup-F test (bootstrap p-value) and Teräsvirta's test against smooth
  transition (`terasvirta`). Forecasts are simulated, so multi-step forecasts follow the regimes, and backtests score
  the simulated draws. **Estimate → Threshold autoregression…**
- **Forecast evaluation** (Elliott and Timmermann, *Economic Forecasting*, chapters 14-18):
  - Backtests score each forecast's density against the outcome: log score, CRPS and PIT, as a normal from the
    forecast's standard error or from a Bayesian VAR's simulated draws. `density_accuracy()` reports them with the
    coverage of the 50% and 90% intervals; `plot_pit()` draws PIT histograms.
  - Tests: Mincer-Zarnowitz efficiency, Berkowitz (PITs), Christoffersen (interval coverage), Harvey-Leybourne-
    Newbold encompassing, Giacomini-White and Clark-West (nested models), and Diebold-Mariano on any loss, including
    CRPS and the log score. `evaluation()` gives the p-values at a glance. For any forecasts they are functions in
    `pymetrix.stats`.
  - Many forecasts at once: Hansen's SPA test (with White's Reality Check) and the model confidence set, through arch.
  - `combine_forecasts`: equal, inverse-MSE or regression weights, estimated in real time; the combination is a
    backtest of its own, and its density is the linear pool of its members'.
  - Backtests can use a rolling estimation window (`window=`).
  - New dialogs under **Forecast**: Combine forecasts, Compare backtests, Compare many forecasts; Backtest has a
    rolling-window option. Backtest windows gain Density scores, PIT, Tests and (for combinations) Weights tabs.
  - Every number is checked against an independent calculation: numerical integration, brute force, statsmodels'
    regressions, hand counts, and simulated size.
- **Save a single result** to a file, to send or keep outside a project: **File → Save result…** or
  `save_result(fit, "consumption.pymxr")`. **File → Open result…** or `load_result(...)` reads it back with the same
  tabs, summary and provenance, plus a Saved tab saying what it was and when. Figures redraw from their series. The
  file is JSON and text only, so opening one runs nothing. It is a record, not the live object: it cannot be
  forecast from or estimated again (the project that made it can). Projects keep a copy of each result they open.
  Every kind of result comes back with the same numbers.
- When X-13 cannot adjust a series, the error names the series and gives the program's own message on one line
  (a `DataError`); it was statsmodels' `X13Error`, which also escaped the recomputation of derived series.

## 0.1.0a8: private review build (October 2026)

Recent projects, a Copy summary button, faster steady states, and X-13's model in the provenance of adjusted series.
Install as before ([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

- **File → Open recent**: the last eight projects opened or saved, newest first; files that no longer exist drop out,
  and Clear recent empties the list.
- **Copy summary** on result windows: the result as text, as `print(result.summary())` shows it, for an email or notes.
- Steady states start with one exact Newton step: a linear model, as most Dynare imports are, is solved without a
  root search. On Smets and Wouters (2007) a log-posterior evaluation is 16% faster (16.7 ms against 19.9 ms), with
  the same values.
- Tests keep the app's settings in a folder of their own, not the user's.
- Series adjusted by X-13 record what the program did: its version (`x13_version`, e.g. "1.1 Build 62"), the ARIMA
  model it estimated (`x13_model`), and the outliers it found (`x13_outliers`), beside its folder (`x13_program`).
  In a panel, a note that differs between entities names each. The notes follow the series when it is recomputed,
  are kept in Parquet files and projects, and show in the dataset's Variables tab. Checked against x13as 1.1 Build 62
  run directly on a spec of its own: the same model and outliers, and the adjusted series equal to its D11 table.

## 0.1.0a7: private review build (October 2026)

Every kind of result now opens in the app and reaches its views from Python, with a crash fixed on the way.
Install as before ([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

**The desktop app, third readability pass**
- The Script pane shows Python in a fixed-width font, as the code editor does.
- The spreadsheet shows numbers as the result tables do: four decimals, with thousands separators above a million.
  Copy in the spreadsheet gives the values at full precision (it copied the text shown).
- Spreadsheets and code editors open in cascade inside the workspace, like result windows.
- History and Problems show their empty states across the whole row. The Problems pane said "No problems_view:" when
  empty; it now says "No problems:". In History, the Code column takes the spare width.

**Every result from Python**
- Every result gives any of its table views as a DataFrame: `to_frame(view)`, by the names of its window's tabs, in
  any case. It was `FittedEquation` only. `IRF`, `Posterior` and `Results` keep their own default and also take a
  view name. Results without a notebook display show their first table in Jupyter, and datasets show their data
  with a one-line description.
- tests/test_result_views.py checks every table view of every kind of result, without Qt.

**Every kind of result opens in a window**
- Fixed: a temporal disaggregation result could not be opened. Its Fields tab stopped on a pandas index, whose
  `.names` are level names, often missing.
- A table with no rows says so instead of showing a blank grid. Short tables no longer leave an empty strip below
  their last row.
- tests/test_result_windows.py opens one object of each of the 54 kinds that have a window, every tab, from
  tests/_gallery.py and the demo projects. It fails if a new kind gains a window without an example.

## 0.1.0a6: private review build (October 2026)

A simpler install, a clearer desktop app, and Smets and Wouters checked against Dynare away from the modes. Install
as before ([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

**statsmodels and matplotlib are core dependencies.** A plain `pip install pymetrix` now includes them: unit-root
tests, diagnostics, seasonal adjustment, the compiled Kalman filter and every figure (Graph tabs, `.plot()`, report
figures) work without an extra. `[stats]` and `[plot]` are kept, empty, so older install commands still work. `[all]`
is unchanged in what it installs.

Found rehearsing the reviewer demo on 0.1.0a5:
- The Hausman test refuses to compare a fit with itself, or two fits with nothing to test between them. The dialog's
  defaults used to give a statistic of 0 on 0 degrees of freedom.
- The posterior summary shows effective draws as whole numbers.
- Opening or reloading a demo shows a busy cursor and a message while it builds (`nk_estimation` takes about 20
  seconds) instead of freezing silently.
- New result windows open in cascade inside the workspace; on a small screen one could open partly off its edge or
  exactly over the last.
- Dialogs with two fields picking the same kind of object (Hausman's two fits) start on different objects.
- The review brief points to **Test → Run test…** for a test on your own equation.
- The demo path is a test: tests/test_demo_rehearsal.py drives the desktop app through the 15-minute demo script.
- The install page states system requirements, with disk and memory measured on 0.1.0a5.

**Readability in the desktop app**
- Result tables: numbers right-aligned with their headers, compact rows (about twice as many on screen), and
  p-values below 0.0001 shown as `<0.0001`. That is on screen only: every copy, including Copy selected cells,
  now gives the numbers themselves at full precision.
- Short tables take only the height they need, so their notes (R-squared, sample, acceptance rates...) follow them
  instead of sitting at the foot of the window. Very wide text wraps.
- Provenance reads as text: "2000Q2 to 2024Q4", "OLS; cov: classic", a time to the minute, yes and no. Row numbers
  are hidden where the first column already names each row.
- Parameter bounds read `[0, ∞)` instead of `[0, 1e+09]`.
- **Fixed: Data → Seasonal adjustment… crashed in every project.** Its log setting (yes, no or automatic) had no
  choices. Such settings now offer automatic / yes / no, and a test opens every menu dialog, with data and without.
- Dialogs are wide enough for their hints. A 0 that means something says so ("automatic", "as many as kept"), and a
  blank choice says what it means ("(none)", "(summary and graph)").
- Graphs: horizons and periods have whole-number ticks (0, 5, 10), not 2.5 and 7.5. A response to a shock reads
  "y, shock to ey" in every view (sweeps said "y <- ey" and "y after a shock to ey"). Model impulse responses are
  titled "(rational-expectations solution)", without nested brackets. The chains' axis is labelled "draw".
- Help contents name each test, then give the code scripts use: "Augmented Dickey-Fuller test (adf)".
- The Immediate window opens with one line, not Python's and IPython's banners. The status bar names the frequency
  ("quarterly") and shortens a long last step with "…". The Objects pane shows types in full. The dataset overview
  reads "4 (4 original, 0 derived)".

**Validation**
- Smets and Wouters' log posterior agrees with Dynare 6.0's own at sixteen points to 2e-8. Ten of the points are in the
  crhopinf tail where pymetrix's chains put more weight than Dynare's run. So the two compute the same posterior, and
  the difference is in the sampling: Dynare's two random-walk chains visit that tail less. The Dynare reference
  workflow gains a case for it (sw2007_points), and a fast test checks the points.

## 0.1.0a5: private review build (October 2026)

Faster model estimation, and Metropolis-Hastings that handles posteriors with several modes. Install as before
([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

**Speed**
- Estimation and Metropolis-Hastings no longer rebuild the model at each trial value. `Model.with_parameters(values)`
  gives the same model at other values and shares the compiled equations and the linearisation's structure. On Smets and
  Wouters (2007), one log-posterior evaluation takes 12 ms instead of 24.5 ms, with the same value (to 3e-11).
- Normal, beta, gamma and inverse-gamma prior densities are computed in closed form instead of through SciPy (equal to
  SciPy to 1e-11).

**Posteriors with several modes**
- Metropolis-Hastings jumps between modes. When the estimate's starts found more than one mode (`estimate(...,
  starts=n)`), or `mcmc(modes=[...])` names others, 5% of steps (`jump=`) propose moving the draw by the difference
  between two modes. The move is symmetric, so the draws keep each mode's share exactly, and chains start around the
  modes in turn. The summary and provenance report the modes and the jumps accepted; `modes=False` turns jumps off.
- Smets and Wouters (2007) has two modes of almost equal height (log posterior -1484.28 and -1484.82; risk-premium
  persistence 0.14 and 0.82). Dynare 6.0 finds the same two (its 90% interval for that persistence: 0.08 to 0.82).
  Four chains of 40,000 draws now cross about 40 times each and reached R-hat 1.03 in one run, against 1.43 without jumps.
  Eight starts find both modes. The modes jumped between by default are those within 5 of the best log posterior,
  highest first, which leaves out a minor third mode 6 below.
- Still open on Smets and Wouters: a thin tail in the price-markup shock's persistence, which chains enter rarely and
  leave slowly, holds R-hat between 1.03 and 1.22 at these lengths. pymetrix gives that tail more weight than Dynare's
  run does. (Since settled: see the next build.)

**Validation**
- The Smets and Wouters reference now holds Dynare 6.0's Metropolis-Hastings with 2 x 200,000 draws (was 2 x 20,000).
- A slow test (`PYMETRIX_SLOW=1`, about 45 minutes) estimates Smets and Wouters with eight starts. It requires both
  modes, then compares the posterior with Dynare's: means, the risk-premium persistence's interval and the marginal
  data density.
- New checks of the sampler against exact answers: a two-component normal mixture with known weights, and a posterior
  with two modes computed on a grid.

## 0.1.0a4: private review build (October 2026)

Provenance that records what was actually used, and a bridge to NumPy, pandas and statsmodels. Install as before
([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

**Provenance**
- Results record choices made automatically next to what was asked for (`provenance.resolved`): the HAC lag used,
  a VAR's lags chosen by AIC, the lag a unit-root test chose.
- Results say how their rows were chosen (`sample_source`, `sample_definition`): an explicit span, the dataset's
  active sample in full (conditions included, not only its first and last period), or all observations.
- MIDAS now applies the dataset's active sample (it used only an explicitly given span) and records its indicators'
  vintages; each data load is listed with the dataset; scripts name the Kalman filter engine when one was chosen;
  seasonally adjusted series name the X-13 program used.
- Defaults are guarded: a test fails if a default of a command, method or test changes without a migration or a
  changelog entry, since scripts leave defaults out.

**NumPy, pandas and statsmodels**
- `Dataset.to_frame`, `to_numpy` and `design(y, x)`: data on pymetrix's sample and missing-value rules, and a
  labelled `y` and `X` built by pymetrix's own formula language, for statsmodels, SciPy or scikit-learn.
- `add_series`, `apply_np`, `apply_pd`, `transform_each` and `Dataset.from_frame(source=)`: data back in, refusing
  an existing name unless asked and a series of another frequency, recording where each series came from.
- `FittedEquation.from_statsmodels`: a statsmodels regression as an ordinary pymetrix result, with its views and
  provenance; statsmodels OLS on `design`'s output reproduces pymetrix's own OLS.

**For reviewers:** the review brief, a 15-minute demo script, an invitation template and a feedback form
(**Issues → New issue → Review feedback**).

## 0.1.0a3: private review build (October 2026)

Metropolis-Hastings on models with long, curved posterior ridges, found checking Smets-Wouters against Dynare.
Install as before ([docs/install.md](docs/install.md)); projects saved with earlier builds open unchanged.

- Halfway through burn-in, each chain's proposal takes the shape of its own draws so far (`adapt=False` turns this
  off; nothing adapts while draws are kept). On Smets-Wouters, one chain had started in the model's second mode
  and taken about 8,000 kept draws to reach the main one; now both chains move along the ridge between the modes
  (R-hat for the risk-premium shock's persistence 1.74 → 1.07, and its 90% range, 0.06-0.70, matches Dynare's
  0.05-0.70). Smets-Wouters still needs longer chains than 2 × 20,000 draws for every parameter to converge.
- The posterior summary names any parameter whose chains disagree (R-hat above 1.1) and suggests longer chains or
  several estimation starts.
- A demo project opened or reloaded is no longer marked as having unsaved changes.

## 0.1.0a2: private review build (October 2026)

Fixes and polish from a reviewer walkthrough of 0.1.0a1, and from an audit of settings, provenance and data
handling. Install as before ([docs/install.md](docs/install.md)); projects saved with 0.1.0a1 open unchanged.

**Estimation in the app**
- Model → Estimate model now accepts a Dynare file imported with Model → Import Dynare file (in 0.1.0a1 it offered
  only template models): leave the data empty to use the file's own datafile, priors and options. Starts (under
  Advanced) optimises from several points and reports other modes, as `estimate(..., starts=n)` does in code.
- Model → Metropolis-Hastings gains burn-in and parallel-workers settings.
- Smets-Wouters measured against Dynare 6.0: the mode from the file's starting values within 0.002 standard
  errors, standard errors within 0.5%, the Laplace density 0.013 apart (a slow test checks these).

**Data safety**
- Setting a new sample marks results estimated on the old one as out of date.
- Generate series, Derived series, Add to dataset and Derive for each series refuse an existing name unless
  Replace is ticked; a replaced series no longer keeps the old values' source and vintage.
- A series of another frequency is refused instead of becoming a column of missing values.
- Each series records where it came from: source, ID there, release date where the provider gives one, and when
  it was fetched or imported; `external_ref` holds a reference for plugins with their own source system.

**The desktop app**
- Zoom acts on the active window (tables, figures, the spreadsheet, Code windows).
- A Problems panel; a status bar (dataset, sample, selection, last step); Show producer step and Copy producer
  command; the Fields tab lists entries such as `fit.stats["aic"]`; right-click copies table cells.
- A one-line dataset header, clearer window titles, empty-pane text, and File → Reload demo project.

## 0.1.0a1: private review build (October 2026)

The first build shared outside the project, with a few econometricians and policy modellers, before
a public 0.1. The API may still change; saved projects are versioned and migrated when opened.

**What is in it.** Data handling with metadata and vintages; single equations (OLS, IV, ML, NLS,
quantile, ARMA and GARCH errors), systems, panels and dynamic panel GMM; unit-root and
cointegration tests; VARs, Bayesian VARs and VECMs; temporal disaggregation, MIDAS and
nowcasting; seasonal adjustment and filters; model solving (Newton, Gauss-Seidel, scenarios) and
forward-looking models (perfect foresight, rational expectations, simulation plans); state space
and unobserved components; forecast rounds with judgement, backtests and reports; data from FRED,
the World Bank and SDMX services; a desktop app whose every dialog records the Python it runs, and
projects that rebuild their results. [features.md](docs/features.md) has the full list.

**Model builders** (new since the first demos):

- Dynare `.mod` and IRIS `.model` files read directly, estimation options included.
- Bayesian estimation of linearised models: posterior mode by BFGS on transformed parameters, from
  several starting points if asked; Metropolis-Hastings with parallel chains; Laplace and
  harmonic-mean data densities; Iskrev identification; shock decompositions and data contributions.
- OccBin for occasionally binding constraints, second-order perturbation, optimal policy under
  commitment and discretion, optimal simple rules, parameter sweeps and scenario lists.
- First-order solutions on Dynare's reduced pencil, warm-started from the previous draw in
  estimation; the likelihood filters only the states the data need; one BLAS thread during
  estimation (threadpoolctl). A 300-equation model's estimation step takes about 0.3-0.6 s.

**Checked against Dynare 6.0** (reference runs in CI, numbers only kept):

- fs2000: posterior mode, log posterior, standard errors and Laplace density; Metropolis-Hastings
  posterior means.
- Smets and Wouters (2007): decision rules to 1e-12, log posterior to 2e-10 at Dynare's mode, and
  Dynare's mode reached from the file's starting values.
- Second-order solutions to 1e-14; OccBin against nonlinear perfect foresight; optimal policy
  against closed forms. [validation.md](docs/validation.md) lists every check.

**Known gaps** are in [known-gaps.md](docs/known-gaps.md): among them `corr` priors and
generalised prior bounds in Dynare files, third-order perturbation, a reference manual, and an
installer for locked-down machines.

Installing: [docs/install.md](docs/install.md).
