# Backend adapters: one per library, with an explicit mapping

Plan item 5 · built 24 October 2026 · package `pymetrix.adapters`

## The problem

- Each wrapper mapped pymetrix's settings to its library's arguments inline, in its own way.
- Some library arguments were fixed in the code (`QuantReg max_iter=5000`, Lasso's `tol=1e-10`, arch's
  `maxiter`), with no way to change them.
- Users had no way to reach a library argument pymetrix did not expose.
- One estimator silently ignored a setting: Fama-MacBeth dropped clusters.
- arch's warnings were switched off entirely.
- Three wrappers read private library attributes in the middle of their code.

## The decisions

1. **One module per library.** The modules are `adapters.statsmodels`, `adapters.linearmodels`, `adapters.arch`,
   `adapters.sklearn` and `adapters.native`, the last for pymetrix's own estimators.
   - Each defines the library's `Adapter`, its name and the distribution its version is read from.
   - Each also holds the `AdapterSpec` of every estimator it serves.
   - `pymetrix.adapters` keeps `require`, `backend` and `EXTRAS`, so existing imports work.
2. **The mapping is data.** An `AdapterSpec` says:
   - how pymetrix's arrays reach the library (`inputs`);
   - for each setting, the library argument and the stage (model constructor or `fit`) it goes to, or that
     pymetrix uses it itself (`options`);
   - what pymetrix fixes and why (`fixed`);
   - which outputs fill which pymetrix fields (`outputs`);
   - what it refuses (`unsupported`).

   `spec.describe()` renders it, and every wrapped method's help page shows it under *How the settings reach the
   library*. A test checks that every field of every wrapped settings class is in its spec, apart from those
   shared with the estimation method, which the notes describe.
3. **`backend_options`.** Every wrapped settings class has `backend_options`, a mapping or text such as
   `"maxiter=500, tol=1e-8"`, normalised to a read-only, hashable `Options`.
   - Keys go to the spec's default stage; `model.x` and `fit.x` choose one.
   - They are merged last, after the fixed arguments, so they can override what pymetrix fixes.
   - A key that a pymetrix setting already controls is an error naming that setting (`q (set it with q=...)`).
   - Values are JSON-like, so the options serialise in commands, calls and projects (item 2).
4. **One way to call.** `with adapter.run(spec, settings) as run: run.call(stage, fn, ...)`:
   - first refuses unsupported settings;
   - merges the arguments of each stage, records them (data by type and shape), and runs the library inside
     warning capture;
   - lets the wrapper add messages (`run.message(...)`, e.g. arch's convergence flag and optimiser message).
5. **Provenance.** `with_backend(provenance, run.record)` sets:
   - `provenance.backend`: `{"name", "version", "wraps"}`;
   - `provenance.options`: the arguments of each stage, plus the user's `backend_options`;
   - `provenance.messages`;
   - `method["backend"]`: the familiar `"statsmodels 0.15.0"` label.

   It also adds the library's warnings. Native estimators record `{"name": "pymetrix", ...}`, so every
   estimated result says which code did the numbers.
6. **No silent behaviour.**
   - Fama-MacBeth with clustered covariance is refused.
   - arch's warnings (for example about data scale) are captured and still shown, no longer silenced; the
     forecast path still silences them, as before.
   - The fixed arguments are listed with their reasons, and can be overridden.
7. **Private library attributes, read in one place.** These are `ZivotAndrews._all_stats`,
   `DynamicFactorMQ.model._index` and `model.data.orig_endog`. Each is read by a small function in its adapter that
   raises a clear `PyMetrixError` if a new library version moves it.

## Coverage

| Library | Through the adapter, with `backend_options` | Not yet |
| --- | --- | --- |
| linearmodels | LIML, IV-GMM, random effects, between, first difference, Fama-MacBeth | — |
| arch | EGARCH, TARCH, FIGARCH, APARCH (fit and forecast); the Phillips-Perron, Zivot-Andrews, DF-GLS, variance-ratio and Phillips-Ouliaris tests; SPA, the model confidence set and StepM (no `backend_options`: they are functions, not settings) | GARCH with `backend="arch"` uses the adapter, but has no `backend_options` (it is a native-first setting) |
| statsmodels | quantile regression, exponential smoothing, Markov switching, ARDL, VECM, the ADF, KPSS, Engle-Granger, Fisher ADF and Johansen tests, VECM lag selection, STL, classical and X-13 seasonal adjustment, DynamicFactorMQ nowcasts | — (two calls stay outside it by design: see below) |
| scikit-learn | ridge, lasso, elastic net, random forest, boosted trees, neural network, nearest neighbours | — |

Two statsmodels calls stay outside the adapter by design (`adapters.statsmodels.OUTSIDE`):

- **The Kalman filter engine** (`MLEModel`, for `StateSpace` and the models' likelihoods). It is a numerical engine
  that pymetrix calls inside its own optimiser, often thousands of times in one estimate, and the native engine can
  stand in for it (`statespace.ENGINE`). There are no library arguments for a user to pass. The result's provenance
  already names the engine that ran (`backend`, and `method["backend"]`).
- **`interop.from_statsmodels`**. It wraps a result someone made with statsmodels outside pymetrix, so pymetrix
  made no call to record. The wrapper says where the result came from.

## Compatibility

- **New field, with a default.** Settings classes gain `backend_options` (default empty), so no command
  migration is needed. Method settings forms show it under Advanced.
- **Wider provenance.** Provenance gains `backend`, `options` and `messages` (with defaults), so saved results
  load as before.
- **Labels.** The `method["backend"]` label is unchanged, except that scikit-learn now carries its version
  (`"sklearn 1.7.2"`, where it was `"sklearn"`).
- **Course notebooks.** The two forecasting labs' reference cells were regenerated, because three signatures
  gained `backend_options`.

## Tests

`tests/test_backend_adapters.py` covers:
- each spec against its class's fields;
- `Options`;
- round trips;
- options reaching the library and recorded (overriding a fixed argument);
- clashes and unknown stages refused;
- model-stage options;
- an unsupported setting refused;
- both linearmodels stages recorded;
- arch warnings kept;
- arch options pymetrix sets refused;
- messages and warnings in a run;
- scikit-learn options on every cross-validation fit;
- the private-attribute readers;
- every spec rendering.
