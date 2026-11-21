# Object model and backend wrappers: plan

Status: **approved 24 October 2026**, with the recommendation taken on every decision (see [Decisions](#decisions-needed)) · 24 October 2026 · branch `claude/pymetrix-object-model-plan-uspus0`

This plan covers the twelve items in the brief "Strengthen the object model and backend wrappers". Each item has
five parts: the current state (with file references), the proposed change, the affected modules, migration and
compatibility, and risks. Section [Decisions needed](#decisions-needed) lists the choices that would change the
public API or are ambiguous. Each one has a recommendation, and no item that depends on one starts until it is
answered.

File references are to `src/pymetrix/` unless they say otherwise.

## Progress (24 October 2026)

Every item is built, each with a design note for the Priority 1 items:

| Item | Design note | What remains |
| --- | --- | --- |
| 1 | [commands.md](commands.md) | Converted so far: LoadData, LoadExample, FitEquation, FitVAR, GenerateSeries, SetSample, and the single-model estimation dialogs (ARDL, ETS, local projections, Markov switching, threshold AR, BMA, state space, unobserved components, VECM) SelectSeries, FitEach, TestEach, FitSystem, the dataset, panel and lag-order dialogs, the mixed-frequency and rolling fits, the Kalman filter, calendar regressors and Hausman, the panel and data summaries, model imports and templates, optimal policy, OSR, OccBin, sweeps, backtests and their combination, DefineEquation and NewSample (settings defined in code are recorded as steps), and Forecast, ImpulseResponse, VarianceDecomposition, Correlogram and RunTest (each recording the method of the object's class), and the model dialogs (solve, rational expectations, calibrate, check, second order, estimate, state-space form, MCMC, identification, contributions, conditional forecasts, ARMA order), and making, adjusting and comparing forecasts and scenarios, Derive for each, the figure dialogs, Edit data, and Save and Compare rounds. The other 6 dialogs are listed in `tests/fixtures/dialogs_without_calls.json` (the list may only shrink). Deprecating hand-written `to_code` in plugins waits for the wrapper guide to document `to_call`. |
| 2 | [serialisation.md](serialisation.md) | No format bump was needed. |
| 3 | [provenance.md](provenance.md) | Results computed from a model alone (known gaps). Backtests are done (D7). |
| 4 | [definitions.md](definitions.md) | |
| 5 | [adapters.md](adapters.md) | Done: every statsmodels and arch call goes through its adapter, except the Kalman engine and `from_statsmodels`, which stay outside by design (`adapters.statsmodels.OUTSIDE`). |
| 6 | [time-index.md](time-index.md) | |
| 7 | [results-record.md](results-record.md) | Results whose verbs need the library (VECM, ETS, dynamic factors, learners) keep it, and projects recompute them. |
| 8–12 | (usability; see `docs/guide.md`) | Selection loops' `Results.coefficients()` keeps its own side-by-side layout. It is not yet built on each fit's tidy `coefficients()`. |
| Reference suite | `docs/validation.md`, `tests/reference/` | The NIST StRD sets other than Longley, once `www.itl.nist.gov` is reachable. |

## Summary

| # | Item | Where it stands today | Size | Public API change |
| --- | --- | --- | --- | --- |
| 1 | API and commands unified | Separate: dialog commands have their own string fields and hand-written `to_code`; typed code is an opaque `RunCode` step | **Large** | Additive, with deprecations later (decision D1) |
| 2 | Versioned, JSON-serialisable commands | Mostly there: stable IDs, schema versions, migrations, format-2 projects. No published JSON schema; `to_jsonable` is lossy and one-way | Medium | Additive |
| 3 | Effective state in every result | Good for `Equation.fit`; uneven elsewhere (wrong `nobs` and `dropped`, some warnings suppressed, no backend version on some results) | Medium | Additive fields |
| 4 | Canonical definitions | Totals AIC/BIC for equations, per-observation Lütkepohl for VARs, some values taken unchanged from backends. Not documented in one place | Medium | Possible renames (decision D3) |
| 5 | One adapter per backend | Ad hoc: `adapters.require`/`backend` helpers; mappings are inline in each wrapper; no pass-through | Medium–large | Additive (`backend_options`) |
| 6 | One time-series abstraction | `PeriodIndex` throughout pymetrix's own results; leaks through raw backend objects and a few wrappers | Small–medium | Some attributes deprecated |
| 7 | Results own their record | Mostly true for `save_result`; raw backend objects are public fields on 6 classes and pickled into project caches | Medium | Attribute deprecations (decision D4) |
| 8 | Explicit, chainable samples | `Dataset.with_sample(...)` exists and is recorded. `ds.sample` is already a property; `estimate` is not a pymetrix verb | Small–medium | Naming (decision D2) |
| 9 | Simple pandas path | `Dataset.from_frame` exists; plain DataFrames are rejected with an `AttributeError`; no `to_pandas` | Small | Additive |
| 10 | Lists wherever a Selection is expected | Mostly true already (Selection is a `Sequence[str]`); some APIs require a Selection, some take comma strings | Small | Additive |
| 11 | Default vintage explicit | Per-series `Vintage` or `None`; "latest" is never written down | Small–medium | Additive |
| 12 | Generic verbs and tidy tables | `FittedEquation` has most verbs; other results vary widely; tidy tables only via `to_frame(view)` | Medium | Additive |

The codebase is already well along on several items. A command contract, provenance with warnings, data
fingerprints and sample rules, a conformance kit and over 150 cross-checks all exist. Most of the work below
closes gaps and makes contracts uniform; it does not start from scratch. Item 1 is the exception. It is a
genuine architectural change and needs your decision before anything is built.

## Cross-cutting findings from the inspection

These came up under several items and are worth seeing together.

1. **Two ways to say the same thing.** `FitEquation` (`commands/builtin.py:249`) has 21 string fields
   (`formula`, `ar="1, 4"`, `cov`, `start`, `end`, `method_settings="..."`), and its `to_code` hand-builds a
   Python line. The API it calls is `Equation(...).fit(data, span=..., method=OLS(...))`. Script code that calls
   the API directly is recorded only as `RunCode` text (`commands/code.py:29`). The session cannot see a fit
   inside it, so AI tools and the History window see an opaque blob. There are about 89 `@command` classes.
2. **Provenance is built in about 15 places by hand.** Examples: `estimation/design.py:179`,
   `_capture.provenance_for`, `vecm.py:182`, `var.py:125`, `systems.py:521` and `statespace.py:903`. They
   disagree:
   - `dropped=0` is hard-coded in `provenance_for` and VECM.
   - `method` is the full settings dataclass in some places and a hand-written `{"type": "ARDL", "backend": ...}`
     in others (so no defaults are recorded).
   - The backend is a `"lib X.Y.Z"` string in `method["backend"]` in some places, has no version
     (`nowcast.py:308`), or is missing (quantile, the statsmodels-backed tests).
3. **The effective sample is sometimes wrong.** Several backends consume presample rows or drop rows
   themselves, and provenance does not know:
   - VECM (`vecm.py:186`, `nobs=len(index)` includes `lags+1` presample rows);
   - ARDL, Markov and ETS (silent `dropna`, `dropped=0`);
   - linearmodels FirstDifference (one row per entity lost);
   - unit-root tests (`.dropna()` silently joins interior gaps).
4. **Warnings are not always captured.**
   - `ext/arch.py:173` wraps the fit in `simplefilter("ignore")`, so arch convergence warnings never reach
     provenance.
   - VAR, VECM, System and StateSpace do not use `capture()` at all.
5. **Raw backend objects are public.** Six result classes keep them as public fields:
   - `FittedVECM.results`, `FittedARDL.model/.result`, `FittedMarkovSwitching.model/.result`,
     `FittedExponentialSmoothing.result`, `FittedDynamicFactor.result` and `FittedLearner.model`;
   - in `extra["arch"]` and `extra["linearmodels"]` on `FittedEquation`.

   Several of them carry a `DatetimeIndex`. Project files pickle the objects of slow steps
   (`commands/project.py:196`) including these raw objects. This is gated by trust and the package version, so
   it is safe, but it contradicts the design document's "pickle is never used for persistence".
6. **Serialisation is one-way.** `_serial.to_jsonable` writes `{"type": ClassName, ...}` with an unqualified
   class name and falls back to `repr(obj)` for anything it does not know. Nothing reads it back. Commands
   round-trip only because their fields are plain strings.

## Item 1: unify the API and the command layer

**Current state.**
- Commands are dialog-shaped dataclasses (`commands/registry.py`) whose fields are mostly strings, parsed in
  `run` (e.g. `FitEquation._method()` parses `method_settings`), with a hand-written `to_code`.
- The public API (`Equation.fit`, `VAR.fit`, `Dataset.with_series`, ...) knows nothing about commands.
- So there are two parallel APIs. Script code is recorded only as `RunCode` text. The dialog's generated code
  and the equivalent hand-written script are similar but not guaranteed identical, and the dialog can express
  only what its string fields can.
- `tests/test_generated_scripts.py` checks that generated scripts reproduce numbers, not that they are the same
  calls.

**Proposed change** (recommended option A of decision D1): **the public function is the command.**

1. **A call record.** A new module `pymetrix/calls.py` defines `Call(id, version, target, args, kwargs)`:
   - a frozen, JSON-serialisable record of one public API call;
   - `target` and object arguments are *references* to named workspace objects (`{"$ref": "data"}`);
   - settings objects are encoded by the new typed codec (item 2);
   - its `to_code()` is the single, canonical renderer: `fit = Equation('c ~ y').fit(data, method=OLS(cov='hac'))`.
2. **A decorator** `@api(id="pymetrix:equation.fit", version=1)` on public functions and methods:
   - Outside a recording session it costs one context-variable lookup and does nothing else.
   - Inside `Session.recording()` it builds a `Call` from the bound arguments (`inspect.signature`, defaults
     applied and recorded), runs the function, and records a `CommandRecord` whose command is the `Call`.
   - Nested decorated calls (e.g. `Selection.fit` calling `Equation.fit`) record only the outermost call.
3. **Script code becomes typed commands.** In a session, code typed in the Immediate or Code window runs with
   recording on. A cell that calls `eq.fit(data)` therefore produces a typed `pymetrix:equation.fit` step, not
   only a `RunCode` blob. Code that is not a decorated call is still kept as `RunCode`, so nothing is lost.
4. **Dialogs become form builders.**
   - A dialog command keeps its form fields and gains `to_call(session) -> Call`.
   - `run` becomes `session.run(self.to_call(session))`, and `to_code` becomes `self.to_call(...).to_code()`.
   - The UI and scripts then produce *identical* `Call` records and identical code by construction.
   - Dialog commands stay registered (IDs frozen in `tests/fixtures/command_ids.json`), so old projects keep
     opening.
5. **Typed arguments, not strings, at the call level.** `ar="1, 4"` becomes `ARMA(ar=[1, 4])` when the form
   builds the call. Parsing lives in one place, the form builder.

**Order of work.** Start with the verbs in Priority 1 scope:
- `Equation.fit`, `VAR.fit`, `VECM.fit`, `System.fit`, `Dataset.read`, `Dataset.example`,
  `Dataset.with_series`, `Dataset.derive` and `Dataset.with_sample`;
- the six dialogs behind them: `LoadData`, `LoadExample`, `FitEquation`, `FitVAR`, `GenerateSeries` and the
  sample commands.

The remaining about 80 commands move over in later, mechanical commits. Each conversion is checked by a test:
for every converted dialog, the generated code, re-run through `RunCode` with recording on, yields an equal
`Call`.

**Affected modules.**
- New: `calls.py`.
- Changed: `commands/registry.py`, `commands/session.py`, `commands/code.py`, `commands/builtin.py` (converted
  commands), `commands/project.py` (save and replay `Call` steps), `estimation/equation.py`, `var.py`,
  `vecm.py`, `systems.py`, `data/dataset.py` (decorators), `mcp_server.py` (exposes calls) and `ui/*` (no change
  expected: forms still come from dialog fields).

**Migration and compatibility.**
- Purely additive at first:
  - `Command`, `@command`, `COMMAND_IDS`, `upgrade` and project format 2 are unchanged;
  - project format 3 adds steps of kind `call`;
  - format-2 dialog steps still load and replay through their command class.
- A dialog command's `to_code` output may change slightly: it becomes the canonical form, for example showing
  defaults the same way scripts do. The generated-script tests are updated to the canonical form.
- Hand-written `to_code` overrides on converted commands are deprecated for plugin authors through a
  `DeprecationWarning` when a plugin command defines `to_code` without `to_call`. They are not removed.

**Risks.**
- **Recording object references needs names.** In a session every workspace object has a name, so the session
  keeps an `id(obj) -> name` map. An argument that is not a named object, such as a DataFrame built inline,
  falls back to a `RunCode` step for that cell. It is never silently mis-recorded.
- **Magic must stay out of plain scripts.** Outside a session the decorator must have no effect. It cannot
  change signatures, docstrings (`functools.wraps`) or performance (there will be a micro-benchmark test).
- **Scale.** About 89 commands. A half-converted state is acceptable because both paths coexist, but it must
  be tracked: a test lists unconverted commands, and the list may only shrink.

## Item 2: a versioned, JSON-serialisable command schema

**Current state.**
- `Command.schema_version`, `@migration`, `upgrade()` (`commands/registry.py:122-163`) and format-2 projects
  (`{id, version, command, fields}`) already exist and are tested against a real format-1 fixture.
- Missing:
  - a machine-readable schema that UI and AI tools can read. `mcp_server.py:176-206` derives one ad hoc from
    `form_spec`;
  - a typed codec for settings objects. Today they survive only as strings inside command fields;
  - `to_jsonable` cannot be read back, writes unqualified class names and falls back to `repr`.

**Proposed change.**
1. **A typed codec**, `pymetrix/_serial.py` extended into `encode(obj)` and `decode(value)`:
   - dataclasses as `{"$type": "pymetrix:OLS", "$v": 1, ...fields}`, using the registry's qualified names
     (`plugin:Name`), so plugins round-trip;
   - `Cluster`, `Sample`, `Selection`, periods, dates, tuples, enums and `inf`;
   - object references as `{"$ref": name}`;
   - unknown types raise `SerialisationError` and never fall back to `repr`.

   `to_jsonable` stays as the display and provenance form.
2. **A JSON Schema for every command and API call**, generated from the dataclass fields and `Widget` metadata
   and from decorated signatures:
   - `pymetrix.calls.schema(id)` returns one; `pymetrix.calls.catalogue()` returns all of them, with id,
     version, plugin, help text and argument types;
   - the MCP server uses this instead of its own derivation;
   - a test pins the catalogue's ids and versions (like `command_ids.json`).
3. **Envelope version.** Each saved step carries `{"kind": "call"|"command", "id", "version", "pymetrix": "0.1.0a8",
   "plugins": {...}}`. Replay after an upgrade runs `upgrade()` on calls exactly as it does on commands.
4. **Round-trip guarantees, tested.**
   - `decode(encode(x)) == x` for every registered settings class (parametrised over the registries).
   - `Call` → JSON → `Call` → `to_code()` → exec → `Call` is the identity.
   - Replay of a saved session after a simulated schema bump (a test-only migration) gives equal results.

**Affected modules.** `_serial.py`, `calls.py`, `commands/registry.py`, `commands/project.py`, `mcp_server.py`,
`testing.py` (`check_roundtrip` extended to the codec).

**Migration and compatibility.**
- Additive. Format-2 projects are unchanged and keep opening.
- `to_jsonable` output in existing provenance is unchanged, so saved `.pymxr` files still open.

**Risks.**
- Settings with callables or arrays (e.g. user `start` functions, prior objects) need explicit encodings or a
  clear refusal. I will inventory them first.
- JSON Schema generation for `Annotated`/`Literal`/union types needs care, but uses no new dependency: I will
  write it on the existing `forms.form_spec` machinery.

## Item 3: effective state in every result

**Current state.**
- `Provenance` (`metadata.py:176`) already has: `sample`, `nobs`, `dropped`, `seed`, `package_version`,
  `warnings`, `converged`, `resolved`, `sample_source`, `sample_definition` and `data_fingerprint`.
- `Equation.fit` fills these well: it records the effective first and last period, all method fields including
  defaults, and the HAC lag chosen.
- The gaps are in findings 2–4 above:
  - wrong `nobs` and `dropped` on VECM, ARDL, Markov, ETS, DynamicFactor and FirstDifference;
  - no warnings captured for VAR, VECM, System and StateSpace, and suppressed for arch;
  - no method defaults where `method` is a hand-written dict;
  - no backend on Quantile or the statsmodels tests;
  - `TestResult`, IRF/FEVD, rolling fits and backtests have no provenance (known gaps).
- Also, `sample` is only `(first, last)`. A sample with interior gaps (`where=` or missing values) is not
  identifiable from provenance, apart from the fingerprint.

**Proposed change.**
1. **One builder.** Every result's provenance is built by `provenance.build(...)`, which replaces the
   hand-built ones and `provenance_for`. It takes:
   - the spec and the settings object, encoded in full by the item 2 codec, so defaults are always included;
   - the data, the requested sample rule, and the **effective row index as reported by the backend**;
   - the presample rows the backend consumed;
   - the backend adapter (item 5), which supplies the backend name and version.
2. **New Provenance fields.** All have defaults, so saved results still load:
   - `rows`: a compact run-length encoding of the periods (or entities and periods) actually used, so gaps are
     visible;
   - `presample`: periods consumed for lags or initialisation by the backend;
   - `dropped_rows`: the dropped periods, with the reason (`"missing"`, `"lag"`, `"presample"`), capped with a
     count beyond 50;
   - `backend`: `{"name": "statsmodels", "version": "0.15.0", "adapter": "pymetrix.adapters.statsmodels"}`.
     `method["backend"]` stays as the string for compatibility;
   - `options`: the effective options passed to the backend, after mapping (item 5), including
     `backend_options`;
   - `messages`: backend convergence and optimiser messages (e.g. `mle_retvals["warnflag"]`, arch's
     `convergence_flag`, SciPy's `message`), separate from Python warnings.
3. **Warnings everywhere.** The builder runs inside the adapter's call wrapper (item 5), so every backend call
   is in `capture()`.
   - arch's blanket `simplefilter("ignore")` is replaced by capture plus targeted filtering of arch's known
     noise, which is recorded in `messages`.
   - Library deprecations stay excluded from `warnings`, as now, but are counted in `messages`, so nothing is
     silently lost.
4. **Fix the wrong numbers.** VECM, ARDL, Markov, ETS, DynamicFactor, linearmodels FirstDifference and Between,
   and the unit-root tests report the backend's effective sample. Each fix has a test that compares
   `provenance.nobs` with the backend's own `nobs`.
5. **Provenance for the remaining data-based results:** `TestResult` from data, rolling fits and backtests.
   Results computed from a model alone stay without provenance, as documented.

**Affected modules.**
- `metadata.py`, `_capture.py` (becomes the builder, or a new `provenance.py`) and `estimation/design.py`.
- Every module that builds provenance today: `var.py`, `vecm.py`, `systems.py`, `statespace.py`, `ardl.py`,
  `markov.py`, `ets.py`, `midas.py`, `nowcast.py`, `bvar.py`, `interop.py`, `forecasting.py`,
  `models/estimation.py`, `estimation/quantile.py`, `stats/*`, `ext/*` and `toolkit.py`.
- `views.py`: the Provenance view shows the new fields.

**Migration and compatibility.**
- Additive fields with defaults. `Provenance` stays a frozen dataclass, and old `.pymxr` files load.
- `nobs` and `sample` values **change** (they become correct) for VECM, ARDL, Markov, ETS and FirstDifference.
  This is a bug fix, but it changes fingerprints of results in saved projects. A changelog line will say so.
- `provenance_for` and `toolkit.result(..., backend=)` stay as thin wrappers over the builder, so plugin authors
  are not broken. They are deprecated with a warning once the builder is documented.

**Risks.**
- Run-length rows for large panels must stay small. The encoding is per entity, and a cap with a hash keeps
  pathological cases bounded.
- Capturing warnings around every backend call could surface warnings users never saw before. This is
  intended, but summaries will list them.

## Item 4: canonical definitions

**Current state.** The conventions are mixed and not written down in one place:
- **Equations** (OLS, IV, NLS, ML, ARMA, GARCH, MIDAS, StateSpace, FittedModel): `aic = -2·loglike + 2k` and
  `bic = -2·loglike + k·ln n`, as totals (`estimation/design.py:226`, `estimation/ml.py`,
  `estimation/timeseries.py:135`).
- **VARs and BVARs:** Lütkepohl per-observation criteria (`var.py:237-245`): `aic = ln|Σ_ml| + 2·K·k/T`.
- **Values taken unchanged from backends:** ARDL `aic`/`bic`/`llf` (statsmodels), arch `aic`/`bic`,
  `interop.from_statsmodels`, the ETS and Markov summaries.
- **R² variants:** centred, uncentred when there is no constant, within for fixed effects, and pseudo-R²
  (McFadden) for ML. Quantile has its own pseudo-R².
- **k includes:** variance parameters in GARCH, but not σ² in OLS (as statsmodels). This is undocumented.
- **Degrees of freedom:** `df_resid = n - k`; cluster covariance applies a small-sample correction
  (`estimation/covariance.py:63`). Both are undocumented.
- EViews reports AIC and BIC per observation. Stata, R and statsmodels report totals. Users comparing packages
  hit this often.

**Proposed change.**
1. **`docs/design/definitions.md`**, the canonical definitions with formulas and the reference each matches.
   It covers:
   - log-likelihood (Gaussian concentrated, exact ML);
   - k (which parameters count);
   - AIC, BIC, HQ (totals) and their per-observation forms;
   - R², adjusted R², uncentred, within and pseudo-R²;
   - `df_resid`, `se_regression`, Durbin-Watson;
   - each covariance estimator: classic, HC0–HC3 (which is "white"), HAC with kernel, lag rule and small-sample
     factor, one- and two-way cluster with its correction;
   - VAR information criteria;
   - the F and Wald conventions (F against χ²).
2. **One function per definition** in a new `pymetrix/estimation/criteria.py`. Every result computes its
   statistics through these functions; no backend value is passed through unchanged. Where a backend value
   is kept for comparison, it is stored under a labelled key (`aic_backend`) together with a note on how its
   definition differs.
3. **Labels in `stats`.**
   - Each result exposes `stats.definitions`, a mapping from key to definition id (e.g.
     `"aic": "aic.total"`), so a reader, a UI or an AI tool can tell which formula a number follows.
   - The Statistics view shows a footnote that links to the definitions page.
4. **VAR criteria** (decision D3): either keep the Lütkepohl values under `aic` and add `aic_total`, or adopt
   totals under `aic` and move Lütkepohl to `aic_lutkepohl` with a deprecation period.
5. **Tests.**
   - A table-driven test checks each definition against an independent formula, and against statsmodels where
     statsmodels agrees.
   - Explicit tests document each place where pymetrix and a backend differ (e.g. ARDL's statsmodels `aic`
     against ours).

**Affected modules.** `estimation/design.py`, `estimation/ml.py`, `estimation/timeseries.py`,
`estimation/covariance.py`, `var.py`, `bvar.py`, `ardl.py`, `ets.py`, `markov.py`, `ext/arch.py`,
`ext/linearmodels.py`, `interop.py`, `estimation/quantile.py`, `views.py`, `docs/validation.md`.

**Migration and compatibility.**
- For equations the canonical form is today's form (totals), so nothing changes there.
- ARDL, arch and ETS values change only if their backend's definition differs. I will check each one and list
  the differences before changing anything.
- VAR depends on decision D3. If it changes, the old values stay under the old key for one minor release, and
  reading them warns once.

**Risks.**
- Some backend definitions differ subtly (e.g. arch counts parameters differently for some distributions, and
  statsmodels ARDL's `llf` includes or excludes presample terms). Recomputing them in-house must be
  cross-checked, or numbers that agree today will drift from the backend.

## Item 5: one adapter per backend

**Current state.**
- `adapters.py` has `require()` (lazy import with install hints) and `backend()` (a "lib version" string).
- `backend.py` is only the NumPy/JAX expression switch.
- The mapping from pymetrix settings to library arguments is inline in each wrapper, for example
  `ext/linearmodels._cov` and `ext/arch._model`/`_distribution`.
- There is no pass-through for backend-specific settings. Some options are hard-coded:
  - `QuantReg max_iter=5000`;
  - arch `options={"maxiter": max(max_iter, 100)}`;
  - sklearn `tol=1e-10`;
  - ETS `disp=False`.
- FamaMacBeth silently ignores cluster settings (`ext/linearmodels.py:241`).
- Private backend attributes are read in three places:
  - `ZivotAndrews._all_stats`;
  - `DynamicFactorMQ.model._index`;
  - `model.data.orig_endog`.

**Proposed change.**
1. **One adapter module per backend:** `pymetrix/adapters/statsmodels.py`, `linearmodels.py`, `arch.py`,
   `sklearn.py` and `native.py`, the native one for pymetrix's own estimators so they are described the same
   way. `adapters/__init__.py` keeps `require`, `backend` and `EXTRAS`, so existing imports work.
2. **An explicit mapping spec per wrapped estimator**, as data, `AdapterSpec`:
   - `wraps`: the backend callable;
   - `inputs`: how y, X, instruments, entity and time map;
   - `options`: pymetrix field → backend argument, with transforms;
   - `fixed`: options pymetrix sets itself, with reasons;
   - `outputs`: backend attribute → pymetrix field, plus any normalisation (item 4);
   - `unsupported`: pymetrix settings this backend cannot honour, which raise a clear `SpecificationError`
     instead of being ignored, as FamaMacBeth's cluster settings are today.

   The spec is rendered on the help page and in `docs/design/adapters.md`, and a test checks that every
   method's fields are mapped, fixed or declared unsupported.
3. **The call wrapper.** `adapter.call(spec, settings, design) -> (raw, outputs)`:
   - applies the mapping;
   - checks unsupported options;
   - runs the backend under warning capture;
   - extracts the outputs into pymetrix types;
   - returns the effective options and messages for provenance (item 3).
4. **`backend_options`**, a new `Mapping[str, Any]` field (default empty) on every backend-wrapped settings
   class:
   - It is merged last into the backend call.
   - Keys that collide with mapped options raise an error naming the pymetrix field to use instead.
   - Values must be JSON-encodable, so they serialise in commands and provenance.
   - Hard-coded options (`max_iter=5000` etc.) become defaults that `backend_options` can override.
   - It is shown under an Advanced "Backend options" text field in forms.
5. **No private attributes.** Each of the three private reads is replaced with public API, or isolated in the
   adapter behind a version check, with a test.

**Affected modules.**
- New: `adapters/` (a package replacing `adapters.py`).
- Changed: `ext/arch.py`, `ext/linearmodels.py`, `ext/sklearn.py`, `vecm.py`, `ardl.py`, `markov.py`, `ets.py`,
  `nowcast.py`, `seasonal.py`, `estimation/quantile.py`, `stats/data_tests.py`, `stats/panel_tests.py`,
  `statespace.py` (engine), `interop.py`, `toolkit.py`, `docs/wrappers.md` and `examples/custom_wrapper.py`.

**Migration and compatibility.**
- `pymetrix.adapters.require`, `backend` and `EXTRAS` keep working.
- Plugins that build results with `toolkit.result` keep working. The adapter spec is recommended, then required
  by the contrib conformance standard in a later minor release.
- Adding a `backend_options` field with a default needs no command migration. Command fields that render
  method settings gain one key.

**Risks.**
- This is the largest refactor of wrapper code. Doing it one backend per commit, with the existing cross-checks
  green after each, contains the risk.
- `backend_options` lets users bypass pymetrix's checks. Provenance records them, and summaries flag "backend
  options set" so the results are not mistaken for defaults.

## Item 6: one time-series abstraction

**Current state.**
- `Dataset` uses a pandas `PeriodIndex`, or a `MultiIndex` with a period level. `TimeAxis` is metadata.
- pymetrix's own fields are mostly converted back correctly: ARDL resid, Markov probabilities, ETS, arch,
  sklearn, quantile and linearmodels.
- Leaks:
  - the raw backend objects in finding 5 (`DatetimeIndex` via `to_timestamp` in `ardl.py:97` and
    `markov.py:80`);
  - `FittedDynamicFactor.factors` returns statsmodels' index unchanged (`nowcast.py:322`);
  - `interop.from_statsmodels` keeps `RangeIndex` or `DatetimeIndex` (`interop.py:239`), which gives
    `sample=None` and an all-NaN fingerprint;
  - the Between estimator indexes by entity only;
  - ARDL and Markov parameter names are statsmodels' (`y.L1`, `x.L0`);
  - the Johansen and VECM select-order tables have integer indexes.

**Proposed change.**
1. **The rule.** Every user-facing `pd.Series`/`DataFrame` on a result is indexed by the dataset's axes (a
   `PeriodIndex`, or a `MultiIndex` of entity and period). Rows that are not on the time axis (lags, horizons)
   use named integer indexes (`lag`, `horizon`). Raw backend objects are not user-facing (item 7).
2. **One converter**, `adapters.to_axes(values, design)`, used by every adapter:
   - it maps backend output back to the design's rows by **label** where the backend keeps labels, and by
     position only where the adapter has checked the length;
   - it raises an error, rather than silently misaligning, when lengths differ.
3. **Parameter names in pymetrix's language:** `y[-1]` and `x` in place of `y.L1` and `x.L0`, matching the
   expression syntax. This applies to ARDL and Markov.
4. **A conformance check** in `testing.check_result`: walk the public attributes of a result and fail on any
   `DatetimeIndex`, unnamed `RangeIndex` on time data, or backend type. It runs over every result type in the
   suite.

**Affected modules.** `ardl.py`, `markov.py`, `nowcast.py`, `interop.py`, `ext/linearmodels.py`, `vecm.py`,
`ets.py`, `testing.py`, `adapters/*`.

**Migration and compatibility.**
- **ARDL parameter names change.** This breaks code that indexes `fit.params["y.L1"]`. Mitigation: a
  `ParamIndex` alias table accepts the old names with a `DeprecationWarning` for one minor release.
- `from_statsmodels` on NumPy input gains a required `data=` or `index=` to place results on axes. Without it,
  the call warns and keeps today's behaviour for now.

**Risks.**
- Parameter renames ripple into Wald restriction strings, saved projects (Wald steps referencing `y.L1`) and
  course notebooks. A project migration rewrites restriction text in saved steps, and the notebooks are
  updated.

## Item 7: results store their own record; raw backend objects optional

**Current state.**
- `save_result` (`saved.py:66`) writes views, summary text and provenance only. It is already a pymetrix
  record.
- Live results expose raw backend objects as public fields (finding 5) and in `extra`.
- Project caches pickle whole results, raw objects included.
- Some pymetrix fields are derived lazily from the raw object, for example ARDL's `params`, `resid` and
  `long_run_coefficients`, and Markov's probabilities. The raw object therefore *is* the record for those
  classes.

**Proposed change.**
1. **Own fields first.** Every result stores its numbers in pymetrix fields at construction: params (a
   `PointEstimate`), covariance, resid, fitted, stats and method-specific fields. Nothing is derived lazily
   from a raw object.
2. **The raw object is optional.**
   - It moves to `result.backend_result`, a field with `compare=False` and `repr=False`.
   - It is set only when `keep_backend=True`, which is the default (D4).
   - It is excluded from every serialisation: `__getstate__` drops it, so pickled caches never include it, and
     the codec never encodes it.
   - Accessing it on a result that was loaded or cached returns `None`, with a note.
3. **A record form.** `result.to_record() -> dict` (JSON) plus Parquet blobs for series, and
   `Result.from_record(...)`:
   - It is defined once on a shared base protocol for the fitted classes.
   - The project cache moves to records in place of pickle. Pickle remains only for objects without a record
     form yet, still trust-gated as today, and a test lists them; the list may only shrink.
   - This also lets a saved result (`.pymxr`) carry the live numbers, not only the views, which narrows the
     known gap "a saved result cannot be forecast from".
4. **Equality and fingerprints.** `testing.fingerprint` and equality ignore `backend_result`.

**Affected modules.** `ardl.py`, `markov.py`, `ets.py`, `vecm.py`, `nowcast.py`, `ext/sklearn.py`, `ext/arch.py`,
`ext/linearmodels.py`, `estimation/fitted.py`, `saved.py`, `commands/project.py`, `testing.py`.

**Migration and compatibility.**
- The old attribute names (`.result`, `.model`, `.results`, `extra["arch"]`, `extra["linearmodels"]`) keep
  working as deprecated aliases of `backend_result`. They warn on access for one minor release.
- `.pymxr` gets a format bump with records added. Old files still open as views only.
- Project caches from older versions are already discarded on a version change (`_usable_cache`), so no
  migration is needed.

**Risks.**
- `FittedLearner.model` is needed to predict, so it is not just a raw object. A fitted sklearn estimator cannot
  be stored as pymetrix fields in general. Proposal: keep it as `backend_result` and mark `predict` and
  `forecast` as "needs the live backend object". A loaded learner then reports that clearly, and refitting from
  the record's spec is offered.

## Item 8: explicit but chainable samples

**Current state.**
- `Sample` (`data/sample.py`) is a value; `Dataset.with_sample(...)` (`data/dataset.py:499`) returns a dataset
  with an active sample.
- Every fit records how its rows were chosen: `sample_source` is "explicit", "active sample of the dataset" or
  "all observations", and `sample_definition` holds the rule.
- So `eq.fit(data.with_sample(("1990Q1", "2019Q4")))` already works and is recorded.
- Gaps:
  - `Dataset.sample` is already a **property** returning the active sample, so `ds.sample("1990Q1:2019Q4")`
    cannot also be a method without a breaking change;
  - pymetrix's verb is `fit`, not `estimate`, and principle 1 in `docs/concepts.md` forbids a second name;
  - the `"1990Q1:2019Q4"` string form is not accepted;
  - the dialogs write `span=(...)` rather than the chained form.

**Proposed change** (recommended option of decision D2):
1. Accept `"1990Q1:2019Q4"`, `"1990Q1:"` and `":2019Q4"` wherever a span or `Sample` is accepted (`Sample(...)`,
   `span=`, `with_sample`).
2. Chain through a new method, **`Dataset.over(...)`**, an alias of `with_sample` that reads well in chains:
   `ds.over("1990Q1:2019Q4").fit(eq, method=OLS())`. Leave `Dataset.sample` as the property, since it is
   already in use.
3. Add **`Dataset.fit(spec, **kw)`**, which calls `spec.fit(self, **kw)`. It is the same verb with the subject
   reversed, not a second name.
4. **Sample visible in generated code and provenance.**
   - Generated code writes the chained form whenever the call carries a sample, e.g.
     `fit = data.over('1990Q1:2019Q4').fit(Equation('c ~ y'))`.
   - Provenance already records the rule. Item 3 adds the exact rows.

If you prefer the exact spelling in the brief (`ds.sample(...)` and `.estimate(...)`), see D2 for its
deprecation path.

**Affected modules.** `data/sample.py`, `data/dataset.py`, `calls.py` (code rendering),
`commands/samples.py`, `docs/guide.md`.

**Migration and compatibility.** Additive under the recommended option.

**Risks.** Two spellings of a sample (`span=` and the chain) could confuse users. Generated code always uses one
canonical form, and the guide says which one to prefer.

## Item 9: a simple pandas path

**Current state.**
- `Dataset.from_frame(df, time=..., entity=..., freq=...)` exists, and a frame with a `PeriodIndex` needs no
  arguments.
- `Equation("y ~ x").fit(df)` with a plain DataFrame fails with `AttributeError: 'DataFrame' object has no
  attribute 'names'` (checked).
- There is `Dataset.to_frame(names, sample, missing)` but no `to_pandas()`.
- Variables, vintages and flags already default to empty, so they are opt-in in practice. Only the constructor
  signature makes them look mandatory.

**Proposed change.**
1. **`as_dataset(x)`**, a single coercion applied at the top of every public entry point that takes data
   (through the item 1 decorator, so it cannot be missed):
   - `Dataset` → unchanged;
   - a DataFrame with a `PeriodIndex` (or a `MultiIndex` with one period level) → `Dataset.from_frame`;
   - a DataFrame with a `DatetimeIndex` and an inferable frequency → converted to periods, with the frequency
     recorded;
   - anything else → a `DataError` that says how to build one (`Dataset.from_frame(df, time="date", freq="Q")`).
2. **`Dataset.to_pandas(names=None, sample="all")`**: a plain wide `DataFrame` with the dataset's index and no
   metadata. It is the documented exit to pandas; `to_frame` stays.
3. **`Dataset(df)` constructor convenience.** `Dataset.__init__` cannot change (it is a frozen dataclass with
   positional `frame, axes`). Instead `Dataset.from_frame(df)` is the documented path, and `pm.dataset(df)` a
   short alias.
4. **Provenance for coerced input** records `source="user:pandas"` (already supported by `from_frame`), so a
   result shows its data came from a plain frame.

**Affected modules.** `data/dataset.py`, `interop.py`, `calls.py` (coercion hook), every `fit(data, ...)`
entry point (through the decorator), `docs/guide.md`.

**Migration and compatibility.** Additive. Code that failed before now works.

**Risks.** Silent frequency inference from a `DatetimeIndex` could guess wrongly, for example month-end
against business days. The proposal accepts only an exact `freq` or `inferred_freq` and refuses otherwise.

## Item 10: plain lists wherever a Selection is expected

**Current state.**
- `Selection` is a frozen `Sequence[str]`, so most loops accept lists already.
- Inconsistencies:
  - some functions accept only a `Selection` (its methods `fit`, `test` and `formulas` exist only on the
    class);
  - some accept a single string;
  - commands take comma-separated text (`_names(text)` in `commands/builtin.py:21`);
  - `Dataset.series(...)` returns a Selection, but `VAR(variables=...)` takes a tuple.

**Proposed change.**
1. **`as_selection(x)`**: it takes a `Selection`, a list or tuple of str, a single name, or a glob pattern when
   a dataset is given. It is applied by the decorator wherever a parameter is annotated `SelectionLike`
   (`Selection | Sequence[str] | str`).
2. **Module-level functions** mirror the Selection methods: `pm.fit_each(names, template, data)` and
   `pm.test_each(names, data, test)`. Both accept lists. The Selection methods stay and delegate to them.
3. **Annotate and test every parameter that takes series names.** A test introspects the public API for
   `SelectionLike` parameters and calls each with a list, a tuple and a Selection.

**Affected modules.** `selection.py`, `var.py`, `vecm.py`, `systems.py`, `data/dataset.py`, `stats/*`,
`calls.py`.

**Migration and compatibility.** Additive.

**Risks.** A single string is ambiguous: is `"c,i"` one name or two? Rule: a string is one name, or a pattern
if it contains `*`. Comma splitting happens only in forms.

## Item 11: the default vintage, explicit

**Current state.**
- Each `Variable` may carry a `Vintage(released, retrieved, revision)`, or `None` when it is not known.
- Connectors set the vintage from the provider's release date.
- `Dataset.read("fred://GDPC1")` with no vintage means "latest as of retrieval", but nothing records that the
  latest was requested.
- `Provenance.data_vintages` maps series to `Vintage | None`, so "latest" and "unknown" look the same.
- Generated code omits the vintage when it is the default.

**Proposed change.**
1. **A vintage request, separate from the vintage released.** `Dataset.read(..., vintage="latest" | date)`:
   - `"latest"` is the explicit default;
   - the request is recorded in `DatasetMeta.vintage_request` and on each series as `Variable.vintage_request`.
2. **Results record both.** Provenance gains `vintage_request`, e.g. `{"policy": "latest", "as_of":
   "2026-10-24T09:12"}` or `{"policy": "as_of", "date": "2020-01-30"}`, beside `data_vintages`. A series with
   no known release says `"unknown"`, not `None`.
3. **Generated code always writes it:** `Dataset.read('fred://GDPC1', vintage='latest')`. Replay then shows
   that the data may differ when re-run, and a replay warns when a `latest` read returns different data
   (the fingerprint changes).
4. Local files (CSV, Parquet) record `{"policy": "file", "modified": ...}`, so all data has an explicit
   vintage policy.

**Affected modules.** `metadata.py`, `data/dataset.py`, `io/*`, `ext/connectors.py`, `commands/builtin.py`
(`LoadData`), `calls.py`, `views.py` (Provenance view).

**Migration and compatibility.** Additive. Saved data without a request load as `{"policy": "unknown"}`.

**Risks.** Little.

## Item 12: generic verbs and tidy tables on every result

**Current state.**
- `FittedEquation` has `summary`, `params`, `resid`, `fitted`, `predict`, `forecast` and `plot`.
- The others vary:
  - `FittedVAR` has no `fitted` and no `plot`;
  - `FittedMarkovSwitching` has no `stats`;
  - `FittedVECM` has no `params`;
  - `FittedLocalProjection` has neither `summary` nor `params`.
- `to_frame(view)` is attached to every result with views (`views.py:408`) and gives tidy-ish tables by view
  name ("Coefficients": coef and std err).
- No result has `coefficients()`, apart from `Results` (selection loops).

**Proposed change.**
1. **A `Result` protocol** (`pymetrix/results.py`, a `typing.Protocol` plus a mixin with defaults; there is no
   deep inheritance, as the design rules require):

   | Verb | Returns | Default when not meaningful |
   | --- | --- | --- |
   | `summary()` | `Table` | always available |
   | `params` | `PointEstimate` or `Draws` | `NotSupportedError` naming what to use instead |
   | `resid`, `fitted` | `pd.Series` (single equation) or `Dataset` (systems) on the data's axes | `NotSupportedError` |
   | `predict(data=None, ...)` | `pd.Series` or `Dataset` | `NotSupportedError` |
   | `plot(kind=None)` | `Figure` | the first figure view |
   | `coefficients()` | tidy `DataFrame`: `term, estimate, std_error, statistic, p_value, ci_low, ci_high` (+ `equation` for systems) | empty frame, for results without parameters |
   | `statistics()` | tidy `DataFrame`: `statistic, value, definition` (`definition` from item 4) | from `stats` |
   | `observations()` | tidy `DataFrame` on the data's axes: `actual, fitted, resid` (+ `equation`) | `NotSupportedError` |

   `NotSupportedError` is a `PyMetrixError` and an `AttributeError`, so `hasattr` and duck typing still behave.
2. **Names.** These are my proposal; D5 offers broom-style `tidy`/`glance`/`augment` as an alternative.
3. **Conformance.** `testing.check_result` checks the protocol, column names and index types for every
   result type in the suite.
4. **Results tables.** Selection-loop `Results.coefficients()` is rebuilt on each result's `coefficients()`.

**Affected modules.** New: `results.py`. Changed: every result class (mostly a mixin and small adapters),
`views.py` (tables built from the tidy frames where possible, so views and frames cannot disagree),
`selection.py` and `testing.py`.

**Migration and compatibility.**
- Additive.
- Where a result already has an attribute of the same name with different semantics, the existing name wins,
  and the clash is listed in the PR. I found none, but I will check each one.

**Risks.** Wide surface. I will do it per result family with tests, starting with equation, VAR, VECM, system and
state space.

## Conformance suite against reference values

The existing `tests/test_conformance.py` is the *extension* conformance kit. The new suite is separate:
`tests/reference/`, with data and certified values in `tests/fixtures/reference/`.

| Estimator | Reference | Tolerance target |
| --- | --- | --- |
| OLS | **NIST StRD linear regression**: Norris, Pontius, NoInt1, NoInt2, Filip, Longley, Wampler1–5 (certified coefficients, s.e., residual SD, R², ANOVA) | LRE ≥ 9 for lower-difficulty sets; Filip and Wampler4–5 ≥ 6 (or a documented failure, see risks) |
| OLS (HAC, White, cluster) | statsmodels, and published Stata output where available | 1e-8 |
| ARIMA (ARMA errors) | statsmodels SARIMAX on fixed data; Box-Jenkins Series A/G published estimates | 1e-4 coefficients, 2e-3 s.e. (as now) |
| VAR | Lütkepohl (2005) West German data, Table 3.1 estimates (e1, e2, e3, 1960Q1–1978Q4); statsmodels VAR | published to the printed digits; statsmodels 1e-8 |

Each case is one JSON fixture: data, spec, settings, certified values, source and digits. One parametrised test
runs them through the **public API and through a replayed command** (item 1), so the suite also checks that the
two paths agree.

**Reference data available offline.** The installed statsmodels (BSD-3) ships:
- Longley's data (`statsmodels/datasets/longley`) with NIST's certified Longley values
  (`regression/tests/results/results_regression.py`);
- Lütkepohl's West German data (`tsa/vector_ar/tests/results/e1.dat`, `tsa/tests/results/lutkepohl2.dta`).

These can be copied into `tests/fixtures/reference/` with their source noted, because wheels do not always
include statsmodels' tests. The other NIST StRD sets cannot be fetched: the sandbox's network policy refuses
`www.itl.nist.gov`. Options are under D6.

## Commit plan

The work goes in small commits, grouped by item, on this branch, in dependency order:
1. Item 2 codec and catalogue (needed by 1, 3 and 7).
2. Item 1 `Call`, the decorator and session recording, and the first six converted dialogs.
3. Item 5 adapter package, one backend per commit: native, statsmodels, linearmodels, arch, sklearn.
4. Item 3 provenance builder and fixes, one family per commit.
5. Items 6 and 7 per result family.
6. Item 4 definitions and criteria.
7. Items 9, 10, 8, 11 and 12, in that order (the small ones first).
8. The reference conformance suite, with a commit for each estimator.
9. Design notes: `docs/design/commands.md`, `serialisation.md`, `provenance.md`, `definitions.md`,
   `adapters.md`, `time-index.md` and `results-record.md`, one per Priority 1 item, written with that item's
   code.

Every commit keeps the full suite green, and each item adds:
- tests;
- serialisation round-trips;
- replay tests where commands are touched.

No new dependencies are needed. JSON Schema, run-length rows and the codec are written on the standard library.

## Decisions needed

**Decided (24 October 2026).** The recommendation was approved for each decision:
- D1: option A.
- D2: `Dataset.over` and `Dataset.fit`, with span strings.
- D3: keep Lütkepohl's criteria under `aic`/`bic`/`hq` and add `aic_total` etc.
- D4: `keep_backend=True`, but the backend object is never serialised.
- D5: `coefficients()`/`statistics()`/`observations()`.
- D6: Longley now; the other NIST StRD sets once `www.itl.nist.gov` is reachable. It was still refused when
  building started.
- D7 had no recommendation. The choice made: provenance for `TestResult` computed from data in this round, and
  backtests in a follow-up.

**D1: how far to unify commands and the API (item 1).**
- **A (recommended).** Public functions carry `@api`, and calls are recorded as `Call` records inside a session.
  Dialogs become form builders that emit the same `Call`, and dialog commands stay for old projects.
- **B.** Keep dialog commands as the record, but generate their `to_code` from the API signature. Script code
  stays as `RunCode`. This is smaller, but it does not remove the parallel API.
- **C.** A, plus dialog command classes deprecated altogether in favour of forms generated directly from API
  signatures. This is the most uniform option, but every UI form is rebuilt and plugin command authors are
  affected.

**D2: sample chaining spelling (item 8).**
- **Recommended:** `ds.over("1990Q1:2019Q4").fit(spec)`, plus the span string form. `Dataset.sample` stays a
  property.
- **Alternative:** your spelling, `ds.sample("...")` and `.estimate(...)`:
  - it turns `Dataset.sample` into a callable that still behaves as the property when not called (a
    descriptor), which works but is surprising;
  - it adds `estimate`, against the "one name per concept" rule, unless `fit` is deprecated.

**D3: VAR information criteria (item 4).**
- **Recommended:** keep Lütkepohl under `aic`/`bic`/`hq`. It is what statsmodels' VAR and the VAR literature
  report and what lag selection uses. Add the totals as `aic_total` etc., with both documented.
- **Alternative:** totals under `aic` everywhere, for one convention across the package, with Lütkepohl moved
  to `aic_lutkepohl` after a deprecation period.

**D4: raw backend objects (item 7).**
- **Recommended:** keep them by default (`keep_backend=True`) in `backend_result`, but never serialised.
- **Alternative:** off by default, to save memory and keep results lean.

**D5: tidy-table names (item 12).**
- **Recommended:** `coefficients()`, `statistics()`, `observations()`.
- **Alternative:** broom's `tidy()`, `glance()`, `augment()`, familiar to R users.

**D6: NIST StRD data.**
- **Recommended:** start with Longley (data and certified values available offline, above), and add
  `www.itl.nist.gov` to the environment's allowed hosts so that Norris, Pontius, NoInt1/2, Filip and Wampler1–5
  are taken verbatim from NIST, with a checksum recorded.
- **Alternative:** you provide the `.dat` files.
- I will not transcribe certified values from memory.

**D7: depth of item 3 in this round.** Provenance for `TestResult` and backtests (today's known gaps) adds
breadth. Include it now, or leave it to a follow-up?

## Baseline

- Python 3.13, pandas 3.0.5, statsmodels 0.15.0, linearmodels 7.1, arch 8.0.0.
- Full suite excluding `test_course_notebooks.py` (it needs a Jupyter kernel): **894 passed, 32 skipped**, in
  7 minutes. This is the bar every commit must keep.
