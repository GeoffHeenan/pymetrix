# Commands and the API: one step, wherever it came from

Plan item 1 · built 24 October 2026 · modules `pymetrix._api`, `pymetrix.calls`, `pymetrix.commands`

## The problem

There were two APIs:
- **The public API**: `Equation(...).fit(data, method=OLS(...))`.
- **The dialog commands**: dataclasses with string fields (`formula`, `ar="1, 4"`, `method_settings="..."`)
  and a hand-written `to_code`.

Code typed in the Immediate or Code window was recorded only as a block of text (`RunCode`). The history, the
project file and an AI tool could not see that a cell had fitted an equation, and the code a dialog wrote was
similar to hand-written code but not guaranteed to be the same.

## The decisions

1. **The public function is the command.**
   - Public functions and methods carry `@api("pymetrix:equation.fit")`.
   - Outside a session the decorator does nothing but check one thread-local value. The overhead is well under
     a microsecond and is tested. `functools.wraps` keeps the name, docstring and signature.
   - Inside a session that is recording, each **outermost** call is captured as a `Call`. Calls a function makes
     internally are part of it, not steps of their own.
2. **A `Call` records the effective arguments.**
   - It holds the call's ID and version, its target, and every argument with defaults applied, all encoded by
     `pymetrix.codec`.
   - Named workspace objects (datasets, results, a named equation or sample) are recorded by name, as the code
     that used them names them. Other settings are recorded by value.
   - Numbers, text and tuples are always values, so `lags=1` never turns into a reference to a variable that
     happens to hold `1`.
3. **A `Call` is a command** (`pymetrix:call`). The session runs, records, saves and replays it like any other
   step, and `Call.to_code()` is the single canonical renderer:
   - the first argument is given by position when it is required, or when no argument is (`with_sample(...)`,
     `Dataset.example('sample_macro')`);
   - other arguments are shown only where they differ from the default;
   - settings are shown by keyword, except specifications marked `code_positional` (`Equation`, `VAR`,
     `VECM`, `System`), which give their first field by position;
   - tuples are written as tuples, so running the generated code gives the identical call.
4. **Typed code becomes calls only when that is exact.** A cell is recorded as calls only if:
   - every statement is `name = <call>`, or, as the last statement, a bare `<call>`;
   - each statement made exactly one recorded call, and each name holds that call's result;
   - every object a call uses existed before it, and each name is written once.

   Anything else (imports, loops, helper variables, a DataFrame passed inline) stays `RunCode`, run and saved as
   before. Nothing is ever recorded inexactly.
5. **Dialogs build calls.** A converted dialog has `to_call()`. Its `run` runs that call and its `to_code` is the
   call's code. So a dialog, and the code it writes typed by hand, give records that are `same_as` each other.
   Converted so far:
   - `LoadData` and `LoadExample`;
   - `FitEquation` and `FitVAR`;
   - `GenerateSeries` (and its *keep updated* form, `derive`);
   - `SetSample`;
   - the single-model estimation dialogs: `FitARDL`, `FitExponentialSmoothing`, `FitLocalProjection`,
     `FitMarkovSwitching`, `FitThresholdAR`, `FitBMA`, `FitStateSpace`, `FitUnobservedComponents` and `FitVECM`;
   - the selection dialogs `SelectSeries`, `FitEach` and `TestEach`, on `Dataset.series`, `Selection.fit` and
     `Selection.test`. Ticked series give a chain, `data.series('c', 'i').fit(...)`; a named selection is used by name;
   - `FitSystem`, which names its equations: `System([ce, ie]).fit(data, ...)` reads `ce` and `ie`;
   - the dataset dialogs `RenameSeries`, `DropSeries`, `FreezeSeries`, `DeriveSeries`, `Filter`, `SeasonalAdjust`
     and `Johansen`, the panel dialogs `ToLong`, `ToWide`, `BalancePanel` and `Slice`, and the lag-order dialogs
     `VARLagOrder` and `VECMLagOrder`, on the `Dataset`, `VAR` and `VECM` methods of the same names;
   - the mixed-frequency fits `FitBridge`, `FitMIDAS` and `FitDynamicFactor`, `FitRolling` (rolling or recursive),
     `KalmanFilter`, `CalendarRegressors` and `Hausman` (`compare(iv, ols, test='hausman')`, naming both fits);
   - `PivotTable`, `PanelStatistics`, `SeriesInfo` and `CheckData`; `ImportDynare`, `ImportIris` and `LoadTemplate`
     (`templates.new_keynesian(...)`: a function pymetrix exports through a module is written with it);
     `SolveOptimalPolicy`, `OptimalSimpleRule`, `OccasionallyBinding` and `ParameterSweep`; `RunBacktest` and
     `CombineForecasts`.

   - `DefineEquation` and `NewSample`, which keep a setting under a name (below);
   - `Forecast`, `ImpulseResponse`, `VarianceDecomposition`, `Correlogram` and `RunTest`, which act on whatever
     object they are given (below);
   - the model dialogs: `SolveModel`, `SolveRationalExpectations` (`Model([...]).linearize(...).solve()`),
     `Calibrate` and `CheckModel` name their equations in `Model([cons, inv, identity])`; `SolveSecondOrder`,
     `EstimateModel`, `StateSpaceForm`, `RunMCMC`, `CheckIdentification`, `DataContributions` and
     `ConditionalForecast` record the method of the model's or fit's class; `SelectOrder`;
   - the forecasting dialogs `MakeRound`, `AdjustRound`, `CompareBacktests` and `CompareScenarios`, and
     `DeriveEach`, which writes its recipes out (`data.derive(dc='dlog(c)', di='dlog(i)')`);
   - the figure dialogs `PlotSeries`, `FigureOptions`, `StyleFigure`, `SetProperties`, `AnnotateFigure`,
     `CallAxesMethod`, `ArrangeStyles`, `ChangeStyle`, `CopyStyle`, `SaveFigure`, `CoefficientPlot` and
     `TableHeatmap`: `figure_of(fit).line('actual', color='C3').grid(visible=False)` is one step, each `Figure`
     method nested on the one before. Shading from a 0/1 series of the data is not a saved setting, so that one
     case of Figure options stays typed code;
   - `EditData` (`data.edit({'c': {'2020Q1': 1.5}})`), and `SaveRound` and `CompareRounds` on
     `ForecastStore('rounds')`, which the codec saves as its folder (`register_codec(..., code=...)` says how
     generated code writes a class registered that way).

   **Dialogs acting on any object.** `fit.forecast(...)` is a different recorded call for each class
   (`pymetrix:fitted_var.forecast`, `pymetrix:fitted_equation.forecast`...), so these dialogs take the session in
   `to_call(session)`, look the object up and record its class's method (`calls.method_call`). The session takes
   their code from that call; their own `to_code()` is the preview where there is no session. An identification
   is a chained call: `var.identify(Cholesky(['y', 'c'])).irf(8)`.

   **Settings defined in code.** `name = Setting(...)`, where `Setting` is a settings class the codec saves (an
   `Equation`, `OLS`, `ARMA`, `Sample`...), is recorded as a step of the internal call `pymetrix:value`, whose code
   is the setting itself. It replays without trust, and named settings inside it stay references
   (`eq = Equation('c ~ y', errors=ar)` reads `ar`). The Define equation and New sample dialogs build these steps.

   Settings classes that pymetrix exports only through a module are written with it (`priors.Gamma(...)`), as
   functions are. With nothing required, a first argument that is a number is named (`rbc.solve(order=2)`).

   Generated code leaves out every argument at its default, except those a call marks with `show` (the example's
   name; the sample of `with_sample`, so `with_sample(None)` says it clears the sample).

   The remaining 6 dialogs are listed in `tests/fixtures/dialogs_without_calls.json`. A test lets that list only
   shrink, and any new dialog must build a call. Each needs a decision first:
   - `AddToDataset` adds a series taken from a result (`result_series`, internal), and a series is data, not a
     setting. It needs a public verb such as `data.with_result(fit, 'fitted')`.
   - `Disaggregate` writes two objects (the fit, and the high-frequency data when *add to data* is ticked); a call
     has one output.
   - `CompareManyForecasts` computes each backtest's losses, then the model confidence set: two steps, or a new
     function taking the backtests.
   - `ExportReport` builds a `Report` with several calls and writes a file.
   - `OpenResult` reads a file and keeps its bytes in the session for the project.
   - `UseHouseStyle` changes a global setting.
6. **History records keep the dialog.** For dialogs, the record's `command` is still the dialog, so *Open
   dialog…* and old projects work as before. `record.call` holds the API call. A project saves it beside the
   dialog's fields under `"call"`, for readers and tools; replay uses the fields.
7. **Calls do not need trust.** A `Call` runs only an installed `@api` function on decoded settings: no `exec`.
   Projects whose typed code was recognised as calls therefore replay without being trusted.
8. **Versions and migrations.** Each call has a version (`@api(..., version=2)`). `@call_migration(id, 1)`
   turns old arguments into new ones. A call saved by a newer version, or naming arguments the installed
   function lacks, is refused with a message saying which.
9. **Schemas.** `schema.catalogue()` lists every call with the JSON Schema of its arguments, generated from its
   signature. Each annotation is resolved on its own, so names imported only for type checkers do not hide the
   others.

## Calls so far

`pymetrix:dataset.read`, `dataset.example`, `dataset.with_series`, `dataset.derive`, `dataset.with_sample`,
`equation.fit`, `var.fit`, `vecm.fit`, `system.fit`.

## Compatibility

- **Projects.** Project format 2 is unchanged; a `Call` step is simply a step with ID `pymetrix:call`, so there
  was no format bump. An older pymetrix opening a project with call steps refuses it, naming the missing command.
- **Generated code** of the converted dialogs is now the canonical form. It differs from before in small ways:
  - defaults are left out (`method=OLS()` is shown, being an explicit choice; `missing='drop'` is not);
  - `params=[]` is no longer written.

  The scripts still reproduce the same numbers (`tests/test_generated_scripts.py`).
- **`Equation.params`** is now always a tuple, so `Equation(..., params=[...])` and `params=(...)` are the same
  equation.
- **Plugin dialogs** with a hand-written `to_code` keep working. Plugin functions can carry `@api` too.
  Deprecating hand-written `to_code` in plugins waits until the wrapper guide documents `to_call`; that is not
  done yet.

## Tests

`tests/test_calls.py` covers:
- the decorator outside a session, and its cost;
- typed calls recorded as calls with their effective arguments and their reads and writes;
- code that stays typed;
- references against values;
- outermost-only recording;
- each converted dialog against its code typed by hand (the same call, code and numbers);
- JSON round trips;
- project replay without trust;
- calls saved with dialog steps;
- version migration and refusal;
- the Immediate-window path;
- switching recording off;
- the catalogue;
- the shrinking list of unconverted dialogs.
