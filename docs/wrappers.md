# Wrapping another library

pymetrix wraps a mature library wherever one does the job well: statsmodels, linearmodels, arch and scikit-learn
today. This page gives the general form of a wrapper, whether it ships with pymetrix or in a plugin of your own.

[adding-features.md](adding-features.md) says *when* to wrap rather than write native code. The rules that every
extension must meet (forms, help, round trips) are in the Extension API section of [design.md](design.md).

**The rule:** pymetrix owns the specification, the data handling and the result types; the library does the
numbers. A user never sees the library's objects, argument names or result classes. They see pymetrix settings,
pymetrix samples and pymetrix results, which work with everything else: forecasts, backtests, tests, reports,
scripts, projects and the app.

The worked example is [examples/custom_wrapper.py](../examples/custom_wrapper.py). It wraps two statsmodels
features, one of each common shape:
- `Huber`, robust regression, as an estimation method;
- `Theta`, the Theta forecasting method, as a model with its own fit.

[tests/test_wrapper_example.py](../tests/test_wrapper_example.py) checks both against statsmodels called directly.

## The general form

Every wrapper has the same six parts, whatever it wraps.

| Part | What it does | In the example |
|---|---|---|
| 1. Settings | A frozen dataclass: the user's choices, in pymetrix words | `Huber(t, scale, max_iter)` |
| 2. Lazy import | The library is imported when first used, with an install hint if it is missing | `require("statsmodels.api")` |
| 3. Data in | pymetrix turns formulas, lags, samples and missing values into arrays; the wrapper checks what the library cannot | `toolkit.design(eq, data, span=..., where=..., missing=...)` |
| 4. The call | Settings mapped to the library's names, and the library run | `sm.RLM(d.y, d.X, M=HuberT(t=self.t)).fit(...)` |
| 5. Results out | The library's output converted to a pymetrix result, with provenance naming the library and its version | `toolkit.result(..., backend=backend("statsmodels"))` |
| 6. Registration | The class registered under a plugin, so the app, help and conformance checks find it | `register("methods", Huber, plugin=PLUGIN)` |

### 1. Settings

```python
@dataclass(frozen=True)
class Huber:
    """Robust regression with Huber's weights (an M-estimator), which downweights outliers."""

    label: ClassVar[str] = "Huber robust regression"           # its name in menus and lists
    kind: ClassVar[str] = "ls"                                  # methods: ls, iv, ml or panel
    wraps: ClassVar[str] = "statsmodels.robust.robust_linear_model.RLM"  # the help shows its documentation
    docs: ClassVar[str] = "https://www.statsmodels.org/stable/rlm.html"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)      # help topics it belongs to
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)      # importable names it needs

    t: Annotated[float, Widget("number", min=0.1, max=10, step=0.1,
                               help="where downweighting starts, in robust standard deviations")] = 1.345
    scale: Annotated[Literal["mad", "huber"], Widget("choice", advanced=True, help="...")] = "mad"
```

- **Frozen:** the settings are immutable and compare equal by value. Results record them, scripts print them,
  and projects save and rebuild them.
- **Class variables** describe the wrapper; **fields** are the user's choices. Forms are built from the fields,
  so a wrapper writes no interface code.
- **Field help:** every field has one line of help in economists' terms.
- **Vocabulary:** use pymetrix names for common settings: `cov`, `lags`, `trend`, `span`, `alpha`, `seed`.
- **Fixed choices** are `Literal` codes.
- **Defaults** are the library's own, unless documented otherwise.
- **Advanced settings:** rarely changed settings get `advanced=True`.
- **Field types:** only plain types (numbers, text, true/false, lists and nested settings dataclasses), so that
  every setting saves to a project.

### 2. Lazy import

```python
sm = require("statsmodels.api")
```

`pymetrix.adapters.require` imports the module, or raises an `ImportError` that says how to install it. For an
optional library, that's `pip install pymetrix[ml]`.

- **Bundled wrappers:** the library's extra must be listed in `adapters.EXTRAS`. A test checks this for every
  bundled plugin.
- **Your own plugin:** `PluginInfo.install` gives the command instead.
- **Why lazily:** importing on first use keeps `import pymetrix` fast. It also means a missing optional library
  only affects the features that need it.

### 3. Data in

The library never sees a `Dataset`.

- **Equations:** `toolkit.design` gives `y`, `X`, the coefficient names and the rows used. It has already
  applied the formula, lags (within each entity, for panels), the span, the `where` condition and the
  missing-value rule.
- **Single series:** use `data.sample_mask(span)` and the series itself.

Before calling the library, check what it would get wrong or reject with an obscure message. Raise
`SpecificationError` with a sentence a user can act on: "Huber regression does not take weights", "the Theta
method needs at least 8 observations".

### 4. The call

Map pymetrix's settings to the library's arguments, here and nowhere else:

- a pymetrix code to a library object (`"huber"` to `sm.robust.scale.HuberScale()`);
- `cov="hac"` to the library's kernel name;
- `"none"` to `None`.

If the library warns about something the user should know (convergence, an identification problem), run it
inside `pymetrix._capture.capture()`. The warnings then go into the result's provenance as well as to the screen.

**Recommended for new wrappers: an adapter spec.** The bundled wrappers declare the mapping as data, an
`AdapterSpec` in `pymetrix.adapters.<library>`, and call the library through `Adapter.run(spec, settings)`. That
gives the user a `backend_options` field for any other library argument, refuses settings the library cannot
honour, captures warnings, and records the arguments of each stage in the result's provenance. The mapping also
appears on the method's help page. See [design/adapters.md](design/adapters.md).

### 5. Results out

Return a core pymetrix result wherever one fits:

| You wrapped | Return | Made with |
|---|---|---|
| An estimator for equations | `FittedEquation` | `toolkit.result(eq, data, d, self, params, cov, resid, stats=..., extra=..., backend=...)` |
| A test | `TestResult` | `TestResult(name, null, statistic, df, p_value, distribution, critical_values=..., details=...)` |
| A table of numbers | `Table` | `Table(columns, rows, index=..., title=..., notes=...)` |
| Series | a `Dataset` | `Dataset.from_frame(frame)` |

- **A new result class** is only for a model whose results don't fit these. Exponential smoothing, the threshold
  AR and the Theta example each have one.
- **Library objects:** keep the library's result object in `backend_result` (`FittedEquation(..., backend_result=res)`,
  or a field of that name on your own result class) for power users. Never make it the main interface: read every
  number you report into your result's own fields when it is made. Pickling, and so projects, drop
  `backend_result`; a result whose verbs need the live object sets `needs_backend = True` (see
  [design/results-record.md](design/results-record.md)).
- **Provenance** records the data vintages, the sample, the settings, and `backend("statsmodels")`, which gives
  `"statsmodels 0.15.0"`. A result can then say exactly what produced it, and the report appendix can print it.

### 6. Registration

```python
PLUGIN = PluginInfo(name="pymetrix-example-wrappers", version="0.1", requires={"statsmodels": ">=0.14"},
                    license="BSD-3-Clause", maintainer="...", description="...", install="pip install statsmodels")

register("methods", Huber, plugin=PLUGIN)   # the Estimate dialog's method list
register("models", Theta, plugin=PLUGIN)    # help pages and conformance checks
```

- **Registry group:** the group decides where the wrapper appears. The groups are `methods`, `models`, `tests`,
  `filters`, `seasonal`, `variances`, `solvers`, `io`, `views`, `commands` and `courses`.
- **The plugin** says who provides the wrapper, what it needs, and under which licence. The plugin table and the
  help show that.
- **Plugins in their own package** register through entry points, and pymetrix finds them when it starts:
  ```toml
  [project.entry-points."pymetrix.methods"]
  huber = "my_package:Huber"
  ```

## Shape 1: an estimation method

A method has one job: `estimate(eq, data, *, span, where, weights, missing) -> FittedEquation`. Everything else
comes from pymetrix:

- **Formulas and samples:** formulas with lags and functions, samples, and loops over groups of series.
- **The Estimate dialog:** the method appears in its method list, with a settings form.
- **Every equation tool:** diagnostic tests, forecasts, backtests (re-estimated at each origin), combinations,
  side-by-side tables and reports.
- **Scripts and projects:** `FitEquation(..., method="Huber robust regression")` exports as
  `method=Huber(t=2.0)`, and projects replay it.

The linearmodels plugin (`LIML`, `IVGMM`, panel estimators), the arch plugin (GARCH family) and the scikit-learn
plugin's `Ridge`, `Lasso` and `ElasticNet` all have this shape.

## Shape 2: a model with its own fit

A model that isn't a single equation has a specification and a fitted result. A forecasting model needs this much
for forecasts, backtests and combinations to work:

| Member | Used by |
|---|---|
| `spec.fit(data, span=None) -> fitted` | Estimation. Backtests call it at each origin with `span=(start, origin)`. |
| `fitted.spec` | Backtests, to re-estimate |
| `fitted.dependent` | Backtests: which series the forecasts are scored against (`fitted.target`, its old name, is still read) |
| `fitted.forecast(data, span)` returning a `Dataset` with a `forecast` column, and `se` if it has one | Forecasts and backtests. With `se`, the forecasts get normal densities, so log scores, CRPS and PITs. |
| `fitted.predictive_draws(data, span)`, optional | Backtests score simulated densities instead (Bayesian VARs and the threshold AR use it) |
| `fitted.summary()` and `fitted.provenance` | Result windows and reports |

To appear in the app's menus, a model also needs a command and, if it has its own result class, a view:

```python
with plugin_context(PLUGIN):                      # the command's project ID: example_wrappers:fit_theta

    @command(menu="Estimate/Theta method...")
    @dataclass(frozen=True)
    class FitTheta(Command):
        """The Theta method: forecast a series by simple exponential smoothing plus half its trend."""

        series: Annotated[str, Widget("series", depends_on="data", help="the series to forecast")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset holding it")] = "data"
        theta: Annotated[float, Widget("number", min=0, max=10, step=0.5, help="...")] = 2.0
        name: str = "theta"

        def run(self, session):
            result = Theta(self.series, self.theta).fit(session[self.data])
            session.objects[self.name] = result
            return result

        def to_code(self):                         # what Export script and project replay use
            return f"{self.name} = Theta({self.series!r}, theta={self.theta!r}).fit({self.data})"
```

- **A view:** `pymetrix.views.register_view(FittedTheta)` gives the result window its tabs. It is a function
  returning `[(title, Table or Figure), ...]`.
- **Existing examples of this shape:**
  - `ExponentialSmoothing` (statsmodels ETS) in `pymetrix/ets.py`, with `FitExponentialSmoothing`;
  - `DynamicFactor` (statsmodels `DynamicFactorMQ`) in `pymetrix/nowcast.py`;
  - the seasonal adjustment methods (STL, X-13) in `pymetrix/seasonal.py`;
  - the scikit-learn learners (`RandomForest` and others), which return their own `FittedLearner`.

## Shape 3: a test or a function

A test is a `Test` subclass. It declares:
- `name`, its identifier in scripts;
- `null`, the null hypothesis in words;
- `applies_to`, the result or data types it tests.

It implements `run(target) -> TestResult`, and joins **Test → Run test** and `fit.test(name)` when it is
registered with `@register`. `ADF`, `KPSS` and `EngleGranger` in `pymetrix/stats/data_tests.py` wrap statsmodels
this way, and `PhillipsPerron`, `ZivotAndrews` and `DFGLS` in `pymetrix/ext/arch.py` wrap arch.

A procedure with no settings worth a form can be a plain function that returns a `TestResult` or a `Table`, such
as `spa`, `model_confidence_set` and `stepm` in `pymetrix/stats/evaluation.py`, which wrap arch.

## Before it ships

- [ ] **Cross-check** against the library called directly, on data where the answer matters: coefficients,
  covariance and forecasts to rounding error. This checks the wrapper (alignment, sample, degrees of freedom,
  names), not the library. Record it in [validation.md](validation.md) as a wrapper check.
- [ ] **Independent check where possible:** if the method can be computed another way (a closed form, an
  optimality condition, a brute-force search), test that too. That checks the library, and pymetrix's use of it.
- [ ] **Conformance:** `pymetrix.testing.check_plugin(PLUGIN)` passes. It checks the forms, the help (including
  that `wraps` resolves), and that settings survive a round trip, for everything the plugin registers.
- [ ] **Library missing:** pymetrix still imports, the feature's error names the install command, and tests
  that need the library skip (`pytest.importorskip`).
- [ ] **Oldest version:** for a bundled wrapper, the library's oldest supported version is in
  `.github/oldest-supported.txt` and CI's oldest-versions job passes.
- [ ] **Docs:** the help comes from the declarations. Add a line to the [user guide](guide.md), and to the
  [CHANGELOG](../CHANGELOG.md).

## The bundled wrappers

| Library | Where | What |
|---|---|---|
| statsmodels (core) | `pymetrix/ets.py`, `nowcast.py`, `seasonal.py`, `stats/data_tests.py`, `estimation/quantile.py` | Exponential smoothing, dynamic factor models, STL, classical decomposition and X-13, ADF, KPSS and Engle-Granger tests, quantile regression |
| linearmodels | `pymetrix/ext/linearmodels.py` | LIML, IV-GMM, random effects, between, first difference, Fama-MacBeth |
| arch | `pymetrix/ext/arch.py`, `stats/evaluation.py`, `stats/data_tests.py` | GARCH family; Phillips-Perron, Zivot-Andrews, DF-GLS, variance-ratio and Phillips-Ouliaris tests; SPA, MCS, StepM |
| scikit-learn | `pymetrix/ext/sklearn.py` | Ridge, lasso, elastic net; random forests, boosting, neural networks, nearest neighbours |
| LightGBM, XGBoost | `pymetrix/ext/boosting.py` | Gradient-boosted trees (`LightGBM`, `XGBoost`), returning the scikit-learn plugin's `FittedLearner` |
| shap | `pymetrix/ml_explain.py` | SHAP values of learners: TreeSHAP, exact and permutation explainers |

The `ext` modules are written against the Extension API only, so each could move to its own package unchanged.
