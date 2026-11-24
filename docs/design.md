# pymetrix — Design

Started 9 October 2026 · Geoffrey Heenan · exported from the design document (claude.ai artifact, rev 108)

> **How to read this document (updated 11 October 2026).** The package is called **pymetrix** (repository github.com/GeoffHeenan/pymetrix, public under BSD-3-Clause). The sections from *Purpose and principles* to *Command registry, UI and dispatch* record the design agreed on 9 October, before building; some of what they describe is still planned. The subsections marked *built* record what now exists and the decisions made while building it. Where the two differ, the built subsections are right. A copy of this document is kept in the repository as `docs/design.md`.

## Purpose and principles

A pure-Python package for estimating, simulating and solving economic models: the estimation range of EViews and the model-solving range of MATLAB with IRIS (and Dynare), with a menu-and-forms UI on top. It deliberately does not copy EViews' object model; the aim is the smallest set of clear, intuitive classes.

**Audience (clarified 11 October 2026):** an open-source replacement for EViews and MATLAB/IRIS for **professional** use (central banks, finance ministries, statistics offices, consultancies), **training** (courses for practitioners) and **academic** teaching and research. pymetrix is **standalone**: it does not depend on any one institution's data systems or models. Institutional data stores and model collections connect through plugins; none is built in. The repository is public, under the BSD-3-Clause licence.

**In scope:** time-series, cross-section, panel and multi-dimensional data; single-equation, system and VAR estimation; state-space models; backward-looking macro models with scenarios; linear and nonlinear rational-expectations models; Bayesian estimation; a forms UI that writes reproducible scripts.

**Design principles**

- Few nouns, four verbs: `fit`, `solve`, `predict`/`forecast`, `linearize`. A user who learns the verbs can use the package.
- Specifications describe; results record. Specifications are frozen dataclasses; every verb returns a new, immutable result object.
- Configuration is passed as small objects (`OLS(cov="hac")`, `Newton(tol=1e-10)`), never as string flags or keyword sprawl.
- No hidden state: samples, scenarios and methods are explicit arguments.
- Shallow types: `Protocol` interfaces, no inheritance deeper than one level.
- The core has no UI dependencies and is fully usable from scripts and notebooks.
- Every result records its provenance, so any number can be traced to data vintage, specification and code version.

## Architecture

Dependencies run one way, downward. The core never imports UI code, and frontends never call estimators directly.


In the code, from the top down:

1. **Front ends** (`pymetrix.ui`): the PySide6 desktop app (main window, spreadsheet, Code, Immediate, History and Help windows) and Jupyter forms. They only build and run commands.
2. **Commands and session** (`pymetrix.commands`): one dataclass per dialog; the `Session` runs commands, records each as a Python step, writes scripts and saves projects.
3. **Core**: data (`data`), expressions (`expr`), estimation (`estimation`, `systems`, `var`, `vecm`, `statespace`, `projections`), models and solvers (`models`), tests (`stats`), filters, output and figures, help.
4. **Back ends and extensions**: NumPy, pandas, SciPy, statsmodels and matplotlib underneath (the last two core dependencies since 0.1.0a6); the plugin registries (`plugins`); optional adapters for linearmodels and arch (`ext`).

Swapping the frontend leaves the command layer and core untouched; swapping a back end leaves everything above it untouched.

## Core vocabulary

Four verbs and about a dozen nouns cover the whole package. One `Equation` class serves both estimation and simulation: `~` marks a stochastic equation with parameters, `=` an identity.

| Verb | Takes | Returns |
| --- | --- | --- |
| `fit` | specification + data | `FittedEquation`, `FittedVAR`, `FittedSystem`, `FittedStateSpace` |
| `solve` | `Model` + data (+ scenario) | `Solution` (a `Dataset`) |
| `predict` / `forecast` | fitted object + data | series or `Dataset` |
| `linearize`, `steady_state` | `Model` | `LinearModel`, `SteadyState` |

| Kind | Classes |
| --- | --- |
| Data | `Dataset` (with axes), `Variable`, `TimeAxis`, `EntityAxis` |
| Specifications | `Equation`, `System`, `VAR`, `Model`, `LinearModel`, `StateSpace` |
| Configuration | methods (`OLS`, `IV`, `GMM`, `ML`, `Bayes`, ...), solvers (`Newton`, `GaussSeidel`, `PerfectForesight`), families, priors, `Scenario` |
| Results | `Fitted*`, `Solution`, carrying `Params` and `Provenance` |
| Output | `Table`, `Figure` |

**Typical session**

```python
from pymetrix import Dataset, Equation, Model, Scenario, OLS, IV

data = Dataset.read("warehouse://national_accounts", vintage="2026-09-30")   # an institution's data store, through a reader plugin

cons = Equation("dlog(c) ~ b0 + b1*dlog(yd) + b2*(log(c) - log(yd))[-1]", params=["b0", "b1", "b2"])
inv  = Equation("dlog(i) ~ a0 + a1*dlog(y)[-1] + a2*r[-1]", params=["a0", "a1", "a2"])
gdp  = Equation("y = c + i + g + x - m")

cons_fit = cons.fit(data, span=("2000Q1", "2019Q4"),
                    method=IV(endog=["yd"], instruments=["dlog(g)", "dlog(yd)[-1]"]))
inv_fit  = inv.fit(data, method=OLS(cov="hac"))
cons_fit.summary(); cons_fit.wald("b1 + b2 = 1")

model = Model([cons_fit, inv_fit, gdp])
base  = model.solve(data, span=("2025Q1", "2030Q4"))
shock = model.solve(data, span=("2025Q1", "2030Q4"), scenario=Scenario(shift={"g": 0.01}))
(shock - base).plot(["y", "c"])
```

The rational-expectations path uses the same verbs: `Model.from_file("nk.model").linearize(ss).solve()` returns a `StateSpace` with `irf`, `simulate`, `filter` and `fit`.

## Data model

`Dataset` is the only data class. It wraps a long-format pandas DataFrame whose index levels are named axes, so one class covers time series, cross-sections, panels and multi-dimensional data.

```python
@dataclass(frozen=True)
class TimeAxis:
    name: str = "time"
    freq: str = "Q"                 # "Y-JUN" for a July-June fiscal year
    calendar: str | None = None      # holidays, for business-day data

@dataclass(frozen=True)
class EntityAxis:
    name: str                        # "country", "household", "vintage", "draw"
    codes: CodeList | None = None

@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame              # long format; index levels = axes
    axes: tuple[TimeAxis | EntityAxis, ...]
    variables: Mapping[str, Variable]
    flags: pd.DataFrame | None = None   # observation status
    meta: DatasetMeta = DatasetMeta()

    @classmethod
    def read(cls, source: str, **kw) -> "Dataset": ...
    def write(self, path: str, **kw) -> None: ...
    def __getitem__(self, name) -> pd.Series: ...
    def eval(self, expr: str) -> pd.Series: ...
    def with_series(self, **exprs) -> "Dataset": ...
    def select(self, where=None, span=None, **levels) -> "Dataset": ...
    def to_freq(self, freq, how=None) -> "Dataset": ...
    def kind(self) -> Literal["cross_section", "time_series", "panel", "multi"]: ...
    def to_xarray(self): ...
    def plot(self, names=None) -> Figure: ...
```

| Axes | Kind |
| --- | --- |
| `(EntityAxis("household"),)` | cross-section |
| `(TimeAxis("Q"),)` | time series |
| `(EntityAxis("country"), TimeAxis("A"))` | panel, balanced or not |
| `(EntityAxis("country"), EntityAxis("sector"), TimeAxis("A"))` | multi-dimensional (input-output, SAM) |
| `(EntityAxis("vintage"), TimeAxis("Q"))` | real-time revision triangle |
| `(EntityAxis("draw"), TimeAxis("Q"))` | stochastic simulation or forecast density |

**Why pandas.** Pandas has a real `Period` type, including fiscal-year periods, and period-aware frequency conversion; Polars has neither. Pandas is hidden behind `Dataset`: no public method requires a DataFrame, and estimation and solving extract NumPy or JAX arrays once and run compiled code. Metadata never lives in `DataFrame.attrs`, which pandas propagates unreliably.

**Expressions.** One expression language, parsed to one syntax tree, serves data transformations, regression formulas, model equations, analytic derivatives and UI validation.

- `x[-1]` lags along the time axis, always within each entity; a lag on data with no time axis is an error. Leads are `x[+1]`.
- Functions: `log`, `exp`, `d`, `dlog`, `pch`, `mean(x, by=)`, `demean(x, by=)`, `share(x, by=)`, `rank`.
- The tree supports `variables()`, `lags()`, `evaluate()`, symbolic `diff()` and `compile()` to NumPy or JAX code.
- Samples are explicit: `span=(start, end)` for time ranges and `where="urban == 1"` for subsets, passed to `select` or any `fit`. There is no global sample state.

## Metadata

Metadata is typed where it drives behaviour and open-ended (`extra`) everywhere else. It sits at five levels: dataset, variable, axis, observation and result.

```python
class Measure(Enum):    STOCK, FLOW, RATE, INDEX, RATIO, COUNT
class Adjustment(Enum): NSA, SA, SWDA, TREND
class Prices(Enum):     CURRENT, CONSTANT, CHAINED, NONE
class Status(Enum):     NORMAL, ESTIMATED, PROVISIONAL, FORECAST, BREAK, MISSING

@dataclass(frozen=True)
class Vintage:
    released: date                   # release date, never the load date
    retrieved: datetime | None = None
    revision: int | None = None

@dataclass(frozen=True)
class Variable:
    name: str
    label: str = ""
    description: str = ""
    units: str = ""                  # "WST", "persons", "%"
    scale: int = 0                   # 6 = millions
    measure: Measure | None = None
    prices: Prices = Prices.NONE
    base_period: str | None = None
    adjustment: Adjustment = Adjustment.NSA
    source: str = ""
    source_key: str = ""
    vintage: Vintage | None = None
    derived_from: str | None = None  # expression that produced it
    symbol: str | None = None        # LaTeX symbol for equations and tables
    extra: Mapping[str, Any] = {}

@dataclass(frozen=True)
class CodeList:
    codes: Mapping[str, str]                    # "WSM" -> "Samoa"
    groups: Mapping[str, frozenset[str]] = {}   # "Pacific" -> {...}

@dataclass(frozen=True)
class DatasetMeta:
    title: str = ""; description: str = ""; license: str = ""; contact: str = ""
    extra: Mapping[str, Any] = {}

@dataclass(frozen=True)
class Param:
    name: str; label: str = ""; symbol: str | None = None
    bounds: tuple[float, float] = (-inf, inf)
    transform: Literal["none", "log", "logit"] = "none"
    prior: Distribution | None = None

class Role(Enum): ENDOGENOUS, EXOGENOUS, INSTRUMENT, SHOCK, ADD_FACTOR

@dataclass(frozen=True)
class Provenance:
    data_vintages: Mapping[str, Vintage]
    spec: dict; method: dict
    sample_used: tuple | Sequence[Hashable]
    dropped: int; seed: int | None
    package_version: str; created: datetime
```

**Rules that make metadata do work**

- `to_freq` chooses aggregation from `measure`: sum for flows, last value for stocks, mean for rates and indices. An explicit `how=` overrides it.
- Derived series record `derived_from`; units are inferred for simple transforms (`dlog`, `pch`) and otherwise set to `"derived"`. There is no full units algebra.
- Merging checks that `units`, `scale`, `prices` and `base_period` agree, and raises on conflict.
- `select` keeps metadata; arithmetic between datasets keeps fields that agree and drops those that conflict.
- Observation status lives in the parallel `flags` frame, keeping provisional and forecast values distinguishable from the numbers.
- SDMX maps directly onto this model (dimensions to axes, codelists to `CodeList`, attributes to `Variable`, observation status to `flags`); the mapping lives in an adapter, not the core.

**Adding metadata later.** `symbol` (a LaTeX symbol, used when equations and tables are typeset) is included now; other fields can be added later without breaking existing code or files. Every field has a default, readers ignore keys they do not recognise, `project.toml` carries a format version for migrations, and `extra` holds ad hoc fields in the meantime.

## Equations and estimation

An `Equation` says what to estimate, a method object says how, and `fit` returns an immutable `FittedEquation` that can go straight into a `Model`.

```python
@dataclass(frozen=True)
class Equation:
    text: str
    params: Sequence[str | Param] = ()   # required for the explicit form
    family: Family = Gaussian()      # Logit(), Probit(), Poisson(), Tobit(left=0), ...
    priors: Mapping[str, Distribution] = {}
    name: str | None = None          # defaults to the left-hand variable
    # parsed from text: lhs, rhs, is_identity, variables

    def fit(self, data, span=None, where=None, method: Method = OLS(),
            weights: str | None = None, missing="drop") -> "FittedEquation": ...
    def with_params(self, **values) -> "FittedEquation": ...   # calibrate

@dataclass(frozen=True)
class FittedEquation:
    equation: Equation
    params: Params                   # PointEstimate or Draws
    resid: pd.Series
    method: Method
    stats: Mapping[str, float]
    provenance: Provenance

    def summary(self) -> Table: ...
    def predict(self, data, span=None, dynamic=False) -> pd.Series: ...
    def test(self, name: str, **opts) -> TestResult: ...
    def wald(self, restriction: str) -> TestResult: ...
```

**Two formula styles, one rule**

- Linear form, `"wage ~ educ + exper + exper^2 + C(region)"`: coefficients are created automatically and named after their terms; the intercept is included unless `- 1` is written.
- Explicit-parameter form, `"dlog(c) ~ b0 + b1*dlog(yd)"`: parameters must be declared in params, never inferred. `fit` raises an error if a declared parameter is unused, if an undeclared name is not in the data, or if a parameter name collides with a series or named constant.
- `~` is stochastic and estimable; `=` is an identity with no parameters.
- Fixed effects belong to the specification: `+ fe(country) + fe(year)`.
- Nonlinear-in-parameters equations under `OLS` become nonlinear least squares automatically, with analytic gradients from the expression tree.

**Methods**

| Need | Expressed as |
| --- | --- |
| Least squares, NLS | `OLS(cov="classic" \| "white" \| "hac")` |
| Clustered errors | `OLS(cov=Cluster("country", "year"))` |
| Instrumental variables, GMM | `IV(endog=[...], instruments=[...])`, `GMM(instruments=[...], weights="optimal")` |
| Maximum likelihood (discrete choice, counts, censored, GARCH) | `family=` on the equation + `ML(errors=...)` |
| Panel estimators | `RandomEffects()`, `Between()`, `FirstDifference()` |
| Dynamic panels | `ArellanoBond(instruments=GMMLags(2, 4))`, `SystemGMM(...)` |
| Quantile, selection | `Quantile(q=0.5)`, `Heckman(selection=Equation(...))` |
| Bayesian | `Bayes(draws=5000, sampler="nuts")`, priors on the equation |
| Survey weights | `fit(..., weights="w")` with any method |

Each method is a frozen dataclass implementing `Method.estimate(eq, data, sample) -> FittedEquation`. Where statsmodels, linearmodels or arch already implement an estimator well, the method wraps it.

**Parameters** are an interface with two implementations: `PointEstimate` (values and covariance) and `Draws` (posterior or bootstrap samples). Both expose `values`, `cov`, `interval()` and `sample(n)`, so downstream code works with either.

## Multi-equation estimation

`System`, `VAR` and `StateSpace` follow the same specification-then-`fit` pattern as `Equation`.

```python
@dataclass(frozen=True)
class System:
    equations: Sequence[Equation]
    def fit(self, data, span=None, method=SUR()) -> "FittedSystem": ...   # SUR, ThreeSLS, FIML

@dataclass(frozen=True)
class VAR:
    variables: Sequence[str]
    lags: int
    exog: Sequence[str] = ()
    by: str | None = None            # panel VAR over an entity axis
    def fit(self, data, span=None, method=OLS()) -> "FittedVAR": ...       # or Bayes(prior=Minnesota())

@dataclass(frozen=True)
class FittedVAR:
    params: Params; resid: pd.DataFrame; provenance: Provenance
    def identify(self, scheme: Identification) -> "FittedVAR": ...  # Cholesky, LongRun, SignRestrictions
    def irf(self, periods=20) -> IRF: ...
    def fevd(self, periods=20) -> Table: ...
    def historical_decomposition(self) -> Dataset: ...
    def forecast(self, data, span, draws=None) -> Dataset: ...

@dataclass(frozen=True)
class StateSpace:                    # x[t] = T x[t-1] + R e[t];  y[t] = Z x[t] + H u[t]
    T, R, Z, H, Q: np.ndarray | ParamMatrix
    states: Sequence[str]; observables: Sequence[str]
    def irf(self, shock, periods=20) -> IRF: ...
    def simulate(self, periods, seed=None) -> Dataset: ...
    def filter(self, data) -> Dataset: ...
    def smooth(self, data) -> Dataset: ...
    def loglike(self, data) -> float: ...
    def fit(self, data, method=ML()) -> "FittedStateSpace": ...
```

`StateSpace` serves both as a directly specified filtering model (unobserved components, time-varying parameters) and as the solved form of a rational-expectations model, so users learn one class for both.

## Statistics and tests

Tests are objects, registered by what they apply to, and every one returns the same `TestResult`. Each test lives with the thing it tests: data tests on `Dataset`, diagnostics on fitted objects, comparisons as a free function over several fits.

| Level | Applies to | Examples | Entry point |
| --- | --- | --- | --- |
| Descriptive statistics | `Dataset` | mean, sd, quantiles, correlations, autocorrelation and partial autocorrelation, cross-tabs, by-group summaries | `ds.describe(by=)`, `ds.corr()`, `ds.acf("y")`, `ds.crosstab(...)` |
| Data tests | `Dataset` | ADF, KPSS, Phillips-Perron, Zivot-Andrews; panel unit roots (IPS, LLC, Breitung, CIPS); Johansen and Engle-Granger cointegration; panel cointegration (Pedroni, Kao, Westerlund); normality; Bai-Perron breaks; seasonality | `ds.test("adf", "y", lags="aic")` |
| Coefficient tests | fitted objects | Wald, likelihood ratio, Lagrange multiplier, omitted and redundant variables | `fit.wald("b1 + b2 = 1")`, `fit.test("omitted", add=["z"])` |
| Residual diagnostics | `FittedEquation` | Breusch-Godfrey, Ljung-Box, ARCH-LM, White, Breusch-Pagan, Jarque-Bera | `fit.test("breusch_godfrey", lags=4)` |
| Stability and specification | `FittedEquation` | Chow, CUSUM, recursive coefficients, RESET, first-stage F, overidentification (Sargan/Hansen) | `fit.test("chow", at="2009Q1")` |
| System and VAR checks | `FittedVAR`, `FittedSystem`, `FittedStateSpace` | lag-order criteria, stability (roots), residual serial correlation, Granger causality | `var_fit.test("granger", cause="r", effect="y")` |
| Comparisons | two or more fits | Hausman, nested likelihood ratio, Vuong, information criteria table, Diebold-Mariano | `compare(fe_fit, re_fit, test="hausman")`, `compare(*fits)` |

```python
class Test(Protocol):
    applies_to: ClassVar[tuple[type, ...]]   # Dataset, FittedEquation, ...
    null: ClassVar[str]                      # "no serial correlation up to lag p"
    def run(self, obj) -> TestResult: ...

@dataclass(frozen=True)
class BreuschGodfrey(Test):
    lags: int = 4
    applies_to = (FittedEquation,)

@dataclass(frozen=True)
class TestResult:
    name: str; null: str
    statistic: float; df: float | tuple | None
    p_value: float | None
    critical_values: Mapping[str, float] = {}   # unit-root tests report these
    distribution: str = ""                     # "chi2", "F", "MacKinnon"
    details: Table | None = None               # auxiliary regression etc.
    def reject(self, alpha=0.05) -> bool: ...
    def to_table(self) -> Table: ...
```

- `fit.test("breusch_godfrey", lags=4)` is shorthand for `fit.test(BreuschGodfrey(lags=4))`; the string is looked up in the registry.
- `fit.tests()` lists the tests valid for that object, and the UI uses it to fill its diagnostics menu.
- `fit.diagnostics()` runs a default set (serial correlation, heteroskedasticity, normality, RESET) and returns one `Table`.
- `summary()` on any fitted object is a `Table` built from `params`, `stats` and selected `TestResult`s, so it renders and exports like everything else.
- Tests are plugins: a new test is a class plus a registry entry, and it appears in `tests()` and the UI automatically.
- Where statsmodels or arch already implement a test well, the test class wraps it; the value added is the uniform `TestResult`.

## Factors, quantiles, simulation and tabulation

Each of these fits an existing pattern: principal components and factor models are specifications with `fit`, quantile regression is a method, Monte Carlo reuses the `draw` axis, and pivot tables are `Dataset` operations.

**Principal components and factor models**

```python
@dataclass(frozen=True)
class PCA:
    variables: Sequence[str]
    n: int | None = None             # None: keep all, choose later
    standardize: bool = True
    def fit(self, data, span=None) -> "FittedPCA": ...
# FittedPCA: loadings, scores (Dataset), explained (variance shares), scree() -> Figure

@dataclass(frozen=True)
class FactorModel:
    variables: Sequence[str]
    factors: int
    dynamics: int = 0                # 0 = static; p = factors follow a VAR(p)
    rotation: Literal["none", "varimax", "promax"] = "none"
    def fit(self, data, span=None, method=ML()) -> "FittedFactorModel": ...   # or PrincipalComponents(), Bayes()
# FittedFactorModel: loadings, factors (Dataset), communalities,
#                    to_statespace(), nowcast(data) for ragged-edge data
```

- A dynamic factor model is estimated through `StateSpace`, so missing observations and series ending at different dates are handled by the Kalman filter. That is the basis for nowcasting.
- Factor-augmented VARs are fitted by `favar(data, panel, observed, factors, lags, slow)`, which estimates the factors from the panel itself; `VAR(..., factors=)` points there.
- Factor-number criteria (Bai-Ng, parallel analysis) are tests on the fitted object: `fitted_pca.test("bai_ng")`.

**Quantile regression**

- `Quantile(q=0.5)` estimates one quantile; `Quantile(q=[0.05, 0.25, 0.5, 0.75, 0.95])` returns a `FittedQuantiles` with a coefficient-by-quantile plot and a predictive distribution.
- Covariance options: kernel, bootstrap, or clustered, as for other methods.
- Growth-at-risk is a short recipe on top: quantile regressions of future growth on financial conditions, then `fq.distribution(fit="skew_t")` for the full predictive density.
- Quantile local projections use the same method inside `LocalProjection(horizons=12, method=Quantile(q=0.1))`.

**Monte Carlo and bootstrap**

```python
@dataclass(frozen=True)
class MonteCarlo:
    dgp: Model | StateSpace | VAR | FittedVAR | Callable[[np.random.Generator], Dataset]
    estimator: Callable[[Dataset], Any]          # e.g. lambda d: Equation(...).fit(d)
    reps: int = 1000
    seed: int = 0
    workers: int = 1
    def run(self) -> "MonteCarloResult": ...
# MonteCarloResult: draws (Dataset with a 'rep' axis), bias(), rmse(),
#                   size(test, alpha), power(test, alpha), summary() -> Table
```

- Seeds come from NumPy's `SeedSequence`, so every replication is reproducible and recorded in provenance, including with parallel workers.
- Bootstrap is the same machinery applied to one dataset: `fit.bootstrap(Bootstrap(kind="residual" | "wild" | "block", reps=999))`, or as a covariance option `cov=Bootstrap(...)`.
- Stochastic simulation of models (`solve(..., draws=)`) already uses this infrastructure.
- Equations with ARMA errors enter a model with their AR terms in the residual and their MA terms through an innovation variable (`e_<equation>`, an identity, zero ahead), so the solver needs nothing new; equations with GARCH variance scale each draw's standardised error by that draw's own variance path, which `stochastic._GarchPath` carries from the fit's last variance over any gap.
- For teaching, size and power experiments become a few lines: simulate a near-unit-root process and record how often ADF rejects.

**Pivot tables and aggregation**

```python
ds.pivot(rows="region", columns="year", values="gdp", agg="sum", margins=True) -> Table
ds.aggregate(by="region", how=None) -> Dataset    # how=None: chosen from each Variable.measure
ds.aggregate(to="Pacific")                         # uses CodeList groups on the entity axis
ds.reshape("wide", by="country") / ds.reshape("long")
```

- `pivot` returns a `Table` for display and export, and `aggregate` returns a `Dataset` for further computation; both take the same arguments.
- Default aggregation comes from metadata (flows summed, stocks and rates averaged or weighted), and mixing incompatible units raises an error.
- A `PivotTable` command gives the forms UI a drag-and-drop pivot view in PySide6 and an ipywidgets version in Jupyter; both write the equivalent `ds.pivot(...)` line to the session script.

## Models, scenarios and solvers

A `Model` is a set of equations, fitted, calibrated or identities. `solve` simulates it under a scenario; `steady_state` and `linearize` lead to the rational-expectations path.

```python
@dataclass(frozen=True)
class Model:
    equations: Sequence[Equation | FittedEquation]
    roles: Mapping[str, Role] | None = None     # inferred if omitted
    over: str | None = None                     # apply per entity (multi-country)

    @classmethod
    def from_file(cls, path) -> "Model": ...
    @property
    def endogenous(self) -> frozenset[str]: ...
    def blocks(self) -> list[Block]: ...        # recursive vs simultaneous structure
    def solve(self, data, span, scenario=None, solver: Solver = Newton(),
              draws: int | None = None) -> "Solution": ...
    def steady_state(self, params=None, guess=None) -> "SteadyState": ...
    def linearize(self, at: "SteadyState") -> "LinearModel": ...

@dataclass(frozen=True)
class Scenario:
    set:    Mapping[str, pd.Series | float] = {}   # replace exogenous paths
    shift:  Mapping[str, pd.Series | float] = {}   # add to exogenous paths
    fix:    Mapping[str, pd.Series] = {}           # hold an endogenous variable
    adjust: Mapping[str, pd.Series | float] = {}   # add to an equation's residual
    hold: Mapping[str, pd.Series] = {}             # endogenous path to hit...
    free:   Sequence[str] = ()                     # ...by freeing these exogenous variables
    name: str = "scenario"

@dataclass(frozen=True)
class Solution(Dataset):
    model: Model; scenario: Scenario | None
    converged: bool; iterations: int; provenance: Provenance
    def quantiles(self, qs) -> Dataset: ...     # when solved with draws
```

- `hold` + `free` swaps exogenous and endogenous variables ("what policy rate delivers this inflation path?"). The solver therefore takes the endogenous set as an input to each solve, not a fixed property of the model.
- `Solution` is a `Dataset`, so differences, plots and exports need no extra code. With `draws=`, it keeps the deterministic solution and gains its draws (`Solution.simulation`: an array of draws by periods by variables), summarised by `quantiles()`, `mean()`, `sd()`, `bands()` and `fan()`; a `draw` axis was not used, so a solution keeps its shape and the draws (thousands of paths) are not handled as data.
- `over="country"` applies the same equations per entity, with pooled parameters or each entity's (`entity_params`), and a scenario for all or per entity (done: `Model.solve` returns a `PanelSolution`). Cross-entity links (30 October 2026: `Link`, `pymetrix.models.links`) are variables each entity takes from the others, by a weight matrix; the entities are solved jointly by fixed-point rounds over whole paths (each entity's own solver unchanged, links as exogenous paths), accelerated by Anderson mixing, rather than by building one stacked model: the per-entity machinery, parameters and scenarios carry over, and the stacked model is the test's independent check. Draws of linked entities (backward-looking models) draw each entity's errors as for an entity alone and solve each draw jointly by the same rounds, with the draw's errors as scenario add factors: one joint solve per draw, slower than the vectorised draws of one entity but the same machinery, so a draw is exactly a joint solution (checked against the stacked model). Coefficient draws and GARCH errors would need per-draw parameters and per-draw variance paths threaded through the rounds, and first-order draws unanticipated shocks period by period; they are refused with links.

**Solvers**

| Solver | Use |
| --- | --- |
| `GaussSeidel(tol, max_iter, damping, convergence)` | Large, mostly recursive backward-looking models |
| `Newton(tol, max_iter, convergence, line_search, refresh)` | Default; sparse analytic Jacobian from the expression tree |
| `Fallback(GaussSeidel(), Newton(), ...)` | Period-by-period solvers tried in order on a period the first cannot solve |
| `PerfectForesight(terminal="steady_state")` | Stacked-time Newton for models with leads |
| `Klein()`, `Gensys()` | `LinearModel.solve`: first-order rational-expectations solution |

**Solver options (8 November 2026).** The convergence test, the line search, the Jacobian's reuse and the fallback
are settings of the period solvers, so they compose: `Fallback` is itself a solver (its `solve_period` tries the
others from the period's starting values), and `ordering="blocks"` is an argument of `Model.solve` that hands each
block to whatever solver is given, a `Fallback` included, through the same `solve_period(system)` protocol with a
period system built for the block's equations. Recursive equations linear in their own variable are solved by one
evaluation of the equation and its derivative (`x -= f/f'`, exact for them) on the period's whole system, so no
system is built for them. Solvers report how they solved a period (`test`, `solver`, `jacobians`) on the system's
`report`, which a plugin solver may ignore; the convergence table shows those columns only when an option asks for
them, so the defaults' table and numbers are unchanged. The blocks are found per layout of the equations (partial
fixes and holds change them): equations are paired with unknowns by bipartite matching (each with its own variable
first, then augmenting paths for held variables and their free partners), then Tarjan's components, iteratively, so
long chains do not reach Python's recursion limit. `Model.blocks()` is left as it was, the diagnostic on variables.

**Solve timing (9 November 2026).** Each period's seconds are kept in `Solution.convergence.attrs["seconds"]`, a
list of floats, not in a column: the table's columns are part of what the defaults must leave unchanged, and the
seconds differ from run to run while every column is reproducible. A list of plain floats, not an array, so pandas
can compare and copy the attrs. The time is `time.perf_counter` around the solver's call alone, so the compilation
and setting up (shared by all periods, and done mostly in the first) do not make the first period look slow.
Pickling keeps attrs, so a result a project keeps has them; a `.pymxr` file is a record of the views and does not.

```python
@dataclass(frozen=True)
class LinearModel:                   # A E_t x[t+1] = B x[t] + C e[t]
    A: np.ndarray; B: np.ndarray; C: np.ndarray
    variables: Sequence[str]; shocks: Sequence[str]
    def solve(self, method=Klein()) -> StateSpace: ...
```

The rational-expectations pipeline is `Model` → `steady_state` → `linearize` → `LinearModel.solve` → `StateSpace` → `fit`, each stage an inspectable object.

## Uncertainty, provenance and back ends

These are built into the core from the start because each changes what the core objects are; adding them later would mean rewriting every estimator and solver.

- **Parameters as distributions.** `Params` is either `PointEstimate` or `Draws`, with bounds and transforms on each `Param`. Bayesian estimation is then just `method=Bayes(...)` plus priors on the specification.
- **Uncertainty through results.** Forecasts and solutions accept `draws=` (normal or bootstrapped errors, and coefficient draws) and carry the draws with the result; `quantiles()` and `bands()` give fan-chart bands (built 27 October 2026: `pymetrix.models.stochastic`). A backward-looking model's draws are solved together, period by period, from the compiled equations: the chord method with the draws' mean Jacobian first, Newton's method with each draw's own for the rest; the errors are drawn before solving, so `workers` processes can share the draws and give the same ones (1 November 2026). Optimal control of a nonlinear backward-looking model's path (`pymetrix.models.control`) is Gauss-Newton with the sensitivities from one forward pass through the period Jacobians along the solution, the method of the Fed's MCE Solve `opt` simulations.
- **Exogenous/endogenous swap** in `Scenario` (`hold` + `free`), as described above.
- **Replaceable numerical back end.** The expression compiler targets an interface: NumPy by default, JAX optionally. JAX gives exact gradients, Hessians for standard errors, Jacobians for large models, NUTS sampling and GPU-batched simulation from the same expressions.
- **Provenance on every result.** Data vintages, specification, method, sample actually used, observations dropped, seed, package version and timestamp.
- **Explicit missing-data policy.** `missing="drop" | "raise"`; each fit reports exactly which observations it used. Ragged data edges are a normal case.
- **Plugins through Python entry points.** Methods, solvers, families, data adapters, formula functions and frontends register the same way, built-in or third-party.

## Command registry, UI and dispatch

The UI is generated from typed command classes rather than hand-built. Each command produces a form, a menu entry and a line of replayable Python.

```python
@command(menu="Estimate/Single equation...")
@dataclass
class FitEquation(Command):
    formula: Annotated[str, Widget("formula")]
    method: Annotated[Method, Widget("choice", source="methods_for:Equation")] = OLS()
    span: Annotated[tuple[str, str] | None, Widget("span")] = None
    name: str | None = None

    def validate(self, session: Session) -> list[str]: ...
    def run(self, session: Session) -> FittedEquation: ...
    def to_code(self) -> str: ...    # "cons_fit = Equation(...).fit(data, method=OLS(cov='hac'))"

class Session:                       # UI-owned state; not part of the core
    objects: dict[str, Any]          # named objects in the workspace
    data: Dataset | None
    history: list[CommandRecord]
    def run(self, cmd: Command) -> Any: ...
    def script(self) -> str: ...
```

- **Forms** come from field types: `Literal` gives a dropdown, `Widget("formula")` a validated formula box, a method field the methods registered for that specification.
- **Script capture:** every UI action appends `to_code()` to the session script, which reproduces the session without the UI.
- **Session state** (names, history, current selection) lives in `Session`, owned by the UI. In the core, objects are plain Python values.
- **Frontend:** PySide6 (Qt) desktop is the first frontend, for the multi-window modelling workflow. Jupyter is the teaching frontend: the same command registry generates ipywidgets forms (`pymetrix.ui.jupyter.form(FitEquation, session)`), each form shows the Python it runs so students see the code, and `Table`, `Figure` and fitted objects display through `_repr_html_`. Students can also call the core API directly in cells. JupyterLite could give a no-install browser version, but only for back ends that run in the browser (NumPy and statsmodels do; JAX does not).

**Where multiple dispatch is used** (plum-dispatch; `functools.singledispatch` for single-argument cases)

| Area | Mechanism |
| --- | --- |
| Public API | Ordinary methods (`eq.fit`, `model.solve`) |
| Data arithmetic and coercion (`align`, `to_dataset`) | plum multiple dispatch |
| Rendering object × frontend (`render(Table, QtFrontend)`) | plum multiple dispatch |
| Expression-tree operations (`diff`, `evaluate`, `compile`) | `singledispatch` |
| Choosing estimators and solvers | Explicit registry the UI can list |
| Numerical inner loops | None: compiled code |

`summary()`, `irf()`, `plot()` and similar return plain `Table` and `Figure` objects. Frontends render them through dispatch, and exporters write HTML, LaTeX, Excel and Word.

**The words a dialog shows (4 November 2026).** Labels, choice names and titles come from one module, `pymetrix.wording`, not from each dialog: a field is labelled by its argument name through a shared table (with per-command overrides where one name means different things, and a field's own `Widget(label=...)` between the two), so a new dialog that uses pymetrix's standard names needs no labels of its own, and a plugin's field the table does not know falls back to its name's words. Choices are shown by name and stored by code: the script is the API call, so it keeps `method='dols'`, and the form says "DOLS (dynamic)", with the code in the tooltip; every field's tooltip ends with its argument (*Python: exog=*) so scripters can go from a dialog to code. The words follow a glossary (help topic *Terms used in pymetrix*). Advanced settings are **one collapsed section at the bottom of the form**, not a tab: a tab hid them behind a second page while a chosen method's or test's own advanced settings sat in an expander on the first, two mechanisms for one idea; a single section holds both, shows only while something in it applies, and opens when a reopened step changed one. Titles are the menu item's words, where only the first "/" of a menu path separates the menu (labels may contain "/": FRB/US).

### History, Immediate and Code windows (built 11 October 2026)

The desktop app mixes dialogs and code in one session, and records both.

| Window | What it is | Notes |
| --- | --- | --- |
| Immediate | A Jupyter console (qtconsole) running in-process | Shares the session's namespace: dialog results are there to use, and objects made there appear in the Objects panel |
| Code | A Python editor built on Qt's text editor | Syntax colouring, line numbers, auto-indent, block indent and comment, bracket matching, find and replace; Jedi completion and signatures from the live objects; pyflakes checks; F5 runs the file and F9 the selection, through the Immediate window, and a failing line is highlighted |
| History | Every step, from dialogs and code alike | Multi-select; copy code with imports; send to the Immediate or Code window; drag; re-run; reopen a step's dialog filled in as it was |

**Recording.** Everything typed is recorded as a `RunCode` step, so the script and projects replay sessions that mix dialogs and code. Cells in IPython-only syntax are recorded but appear in the script as comments. User functions stay in the namespace but are not workspace objects. Monaco (VS Code's editor) remains an option if teaching shows the Qt editor falls short.

**Out-of-date results.** Each step records the versions of the objects it read and wrote. A result is out of date when something it was computed from has changed since, or depends on another out-of-date result. Datasets are compared series by series: dialog steps record which series they use (from their formula and series fields), so only changes to those count; adding a series never does. Typed code is checked against every series of the dataset it read. Out-of-date results are flagged with the reason and recomputed only on request (decided: flag, never recompute silently). `Session.stale()` and `Session.recompute()` expose the same logic to code.

### Derived series (built 11 October 2026)

A series can keep the recipe that made it: `Formula("dlog(y)")` or `Filtered("y", HP(), "trend")`, stored in `Variable.recipe` and saved in Parquet metadata and projects. `Dataset.derive(name=recipe)` adds one; `Dataset.filter` makes its trend and cycle derived by default (`derived=False` for plain values). When `with_series` replaces a series, the derived series that depend on it are recomputed in dependency order (circular definitions are refused). Replacing a derived series directly makes it plain; `freeze` drops recipes but keeps values; subsets and frequency conversions drop recipes, because recomputing on a different sample would change the values. A derived series' vintage is that of its most recently released input. This is the dataset-level half of keeping results current; the session-level half is the out-of-date flagging above, which picks up the recomputed values series by series. In the app: Data → Derived series, and Keep updated in Data → Filter (whose result now goes into the dataset itself by default).

### Spreadsheet editor (built 11 October 2026)

A table view over a dataset (periods by series; for panels, one entity or all rows), reading values on demand. Typing and pasting (Excel blocks; thousands separators and NA understood) create pending edits with Undo and Redo; derived series preview their effect; Apply records one `EditData` step, `data = data.edit({...})`. Renaming (`rename`, which rewrites recipes that use the series), removing (`drop`, refused while a derived series uses it), freezing and describing series (`with_info`) are steps too, applied after any pending edits so steps stay in order.

- **Edits are flagged, not new releases.** `Dataset.flags` marks edited cells (`Status.EDITED`), saved in Parquet and projects; the series' vintage is unchanged.
- **Derived series are read-only** in the editor; freezing one makes it editable. An edit that a derived series cannot follow (a gap inside a filtered series) is refused with the reason.
- **Concurrent change.** If the dataset changes elsewhere while edits are pending, the editor shows a notice and Apply puts the edits on the new version.
- Results built on edited series are flagged out of date by the session, series by series.

### Programs, groups and loops (built 11 October 2026)

**Programs are Python.** A pymetrix program is an ordinary `.py` file starting `from pymetrix import *`, decided over an EViews-style command language: the skill transfers, editors and other tools work unchanged, and there is no second language to maintain. EViews program features map to Python: `%s` substitution to f-strings or `{s}` templates, `!i` scalars to variables, `smpl` to explicit `span=`, `include` to `import`. A partial `.prg` translator for migrating users is possible later.

- **One step per statement.** Running a file or selection from the Code window records each top-level statement as a step (a loop or function definition is one), so History and Recompute work at that level; the first failure stops the run and its line is highlighted.
- **`show(obj)`** opens the object's window in the app, displays it in a notebook and prints it in a terminal; it is quiet while projects are replayed and results recomputed.

**Selections** (renamed from groups on 11 October 2026, away from EViews' term, towards the column selectors of R's tidyselect, polars and Julia's DataFrames). `Selection` is an ordered set of series names (`data.series("c", "i")`, by pattern `"*_cycle"`, by metadata `units="USD bn"`, `derived=True`, or `where=`), a sequence you can loop over, combine (`+`, `-`) and template (`format`, `formulas`). `fit`, `test` and `map` loop for you and return `Results`: an ordered mapping by series with side-by-side tables (`coefficients()`, with rows lined up across series using the template, `stats()`, `tests()`, `failures()` when `errors="collect"`). Dialogs: Select series, Derived series for each selected, Equation for each selected series, Test each selected series; each records one step whose code is the selection call. `data.group()` remains as an older name, and projects saved with the old step names still open.

**Generate series for each (4 November 2026).** Batch generation is a **For each** field on Data → Generate series rather than a sibling dialog: one place to make series, the single case unchanged (empty For each), and the same Keep updated and Replace options. Templates use `{x}` for the series (`{s}` is accepted too, as in selection templates) and `{1}`, `{2}`... for the parts a wildcard's `*` or `?` matched; `{{`/`}}` are literal braces. The step stores the templates (so reopening it shows them) but records the **expansion** as one ordinary `with_series`/`derive` call listing every new series, so the script says exactly what was made and replays the same whatever series are added later; to make that true the session now builds a session-dependent dialog call before the step runs (a wildcard's matches, not the series the step adds). `Dataset.with_each` is a plain method, not a recorded call of its own: it calls `with_series` or `derive`, and the recorder keeps that outermost recorded call, so typed `with_each` and the dialog record the same explicit line. `for_each` was added with a default, which per the commands contract needs no schema bump: steps saved before load unchanged.

**The diffuse log likelihood's constants (5 November 2026).** pymetrix keeps the exact diffuse likelihood of Durbin and Koopman (2012, section 7.2.2) as the default: every observation counts `log 2 pi`, the diffuse periods count `log F_inf`, nothing is concentrated out, and it equals statsmodels' exact diffuse filter. statsmodels' structural models and SARIMAX report by default the likelihood given the first `d` periods (their approximately diffuse start's burn-in), higher by `d/2 log 2 pi + 1/2 sum log F_inf`. Rather than switch, the other convention is an option, `diffuse_terms="exclude"`, named for what it does rather than for a package. It is applied to the objective as well as the report, so a fit's log likelihood is the maximum of what it maximised; where the diffuse block has parameters (a loading on a diffuse state, a DSGE with `start="diffuse"`) the conventions are different likelihoods and the estimates can differ, which the docstrings say. `FittedStateSpace.criteria` reports either from any fit, so the comparison does not need a refit. `stats` keeps its keys; a non-default choice is in `provenance.method`. pymetrix's SARIMA is statsmodels' own likelihood and keeps statsmodels' convention (known gaps).

**Fixed on the way:** `d()`, `dlog()` and `pch()` regressors in linear-form equations were split at the minus sign of their expansion (and parentheses did not group a term, despite the error message). Expanded and parenthesised expressions are now single terms, labelled as written (`dlog(c)[-1]`).

### State space (built 11 October 2026)

`StateSpace` replaces the matrix stub with a specification written as equations with declared parameters: transition equations (states from lagged states and shocks), measurement equations (observed series or expressions of them from current or lagged states and measurement errors), shock standard deviations (parameters or numbers) and an optional list of stationary states. Coefficients come from symbolic differentiation, which also checks linearity; lags beyond one become lagged copies of the state automatically. The matrices follow Durbin and Koopman.

- **Start.** Declared-stationary states start from their stationary distribution; the others are diffuse, treated exactly (Koopman and Durbin) so the likelihood is smooth in the parameters. With none declared, all states are stationary if the model is, and diffuse otherwise.
- **Engines.** statsmodels' compiled filter and smoother when installed (exact diffuse, about 0.2 ms per likelihood); a native filter and smoother otherwise (approximate diffuse start scaled to the data). Tests cross-check native against statsmodels, the exact diffuse likelihood against statsmodels, and the HP filter against a smooth-trend model.
- **Estimation.** Maximum likelihood, or posterior mode when parameters have priors (priors now have densities); bounded optimisation (L-BFGS-B, with a derivative-free polish on flat ridges); standard deviations kept non-negative; parameters at a bound get no standard error; degenerate models (no shock variance) have no likelihood. `fixed=` calibrates parameters.
- **Results.** `FittedStateSpace`: smoothed (two-sided) and filtered (real-time) states with standard errors, standardised innovations, forecasts with intervals, impulse responses, simulation; views; smoothed and filtered states can be added to datasets; the Forecast dialog accepts it.
- **Unobserved components.** `UnobservedComponents(series, trend, cycle, irregular)` builds the common trend-cycle models (local level, random walk with drift, local linear trend, smooth trend; AR cycle of any order, stationary). Estimates match statsmodels' own UC model.
- **Dialogs.** Estimate → Unobserved components, and Estimate → State space model (with multi-line fields for the equations).

This is the base for the next step, forward-looking models: their linear rational-expectations solution is a state-space model, so filtering and estimation carry over.

### Toolkit for your own estimators and solvers (built 11 October 2026)

`pymetrix.toolkit` is the stable, documented set of building blocks pymetrix's own estimators use, part of the Extension API: `design` (formula and sample to `y`, `X`, names, rows used, weights, instruments, absorbed fixed effects, cluster groups), `arrays` (series or expressions as a matrix, evaluated on all the data before the span is cut), `least_squares`, `covariance` (classic, white, HAC, cluster), `sandwich` (for M- and ML estimators), `hessian`, `jacobian`, `gradient`, `fit_stats`, `result` (a `FittedEquation` with statistics and provenance), the Kalman filter and smoother, and the solver interface (`PeriodSystem`). An estimator is a frozen dataclass with `estimate(eq, data, ...)`; a solver has `solve_period(system)`; `toolkit.register` puts either in the dialogs. Worked examples: `examples/custom_estimator.py` (ridge regression) and `examples/custom_solver.py` (damped Newton), both passing the conformance kit.

NumPy, pandas and SciPy are in the session namespace as `np`, `pd` and `sp`; scripts get their import lines only when they use them. Arrays, DataFrames and Series made in code open as tables (matrix, vector, table, series, with their shape).

### Local projections (built 11 October 2026)

`LocalProjection(response, shock, horizons, lags, controls, contemporaneous, long_difference, cov, state, scale)` runs one OLS regression per horizon (`y[t+h]` on the shock at t, lags of the response, shock, controls and contemporaneous controls) and returns `FittedLocalProjection`: `irf` (response, standard error, 68% and 90% bands, observations by horizon) and `fits` (a `Results` collection, so the regressions sit side by side). Standard errors: Newey-West with h+1 lags, or lag-augmented robust (Montiel Olea and Plagborg-Møller); panels use entity fixed effects and clustering by entity. State-dependent responses (Ramey and Zubairy) interact every regressor with the state at t-1 and test the difference at each horizon. Checked against a simulated recursive VAR: exact at impact, close afterwards. Dialog: Estimate → Local projections.

### Figures (built 11 October 2026)

`Figure` (Python's term, not EViews' "graph") is data: panels of layers (`Line`, `Band`, `Bars`, `Scatter`, `Histogram`, `Rule`), drawn with matplotlib without pyplot, so the same object works in the app, notebooks (`_repr_png_`) and scripts (`save` to PNG, SVG or PDF; `data()` gives the series drawn). Dates are labelled in the data's periods (2023Q1); the palette is Okabe-Ito.

- **Figures of results:** `Dataset.plot` (line, bar, area, scatter, histogram; one panel or each; spans; a line per entity for panels), `FittedEquation.plot` (actual, fitted, residuals with ±2 s.e.; scatter for cross-sections), forecasts with intervals (and history for VARs), impulse-response grids with bootstrap bands, local projections with 68% and 90% bands, state-space states with 95% bands, filters (series, trend, cycle), correlograms with significance bands, rolling coefficient paths.
- **In the app:** a Graph tab in result windows, with Copy image, Save and Copy data; Data → Plot series makes a figure that is a workspace object and a recorded step (`figure = data.plot([...], kind=...)`).

### Forward-looking models (built 11 October 2026)

- **Calibration:** `Model(equations, params={...})` gives values for parameters of equations not estimated; an uncalibrated equation without them is an error naming the missing parameters.
- **Perfect foresight:** `PerfectForesight(terminal="constant"|"data"|"steady_state")` is a solver for `Model.solve`. All periods are stacked and solved jointly by Newton with a sparse Jacobian (symbolic derivatives) and backtracking; values before the span come from the data, leads beyond it from the terminal condition. It equals Newton on backward-looking models.
- **Perfect foresight at scale (31 October 2026, the FRB/US example):** the stacked Jacobian is block-banded, since
  each period's equations reach a few periods back and forward. It is factorised by SuperLU with the natural column
  ordering (periods in time order), which keeps the fill-in inside that band; COLAMD's general ordering spread it
  across the matrix. For FRB/US's 284 equations over 240 quarters that took 169 s and is now 16 s, and memory fell from
  0.94 to 0.83 GB. The ordering grows linearly in the horizon. Each Newton iteration's factorisation is reused for
  further (chord) steps while each cuts the residuals' norm tenfold, at most ten. The iteration count is still that
  of Jacobians, so small models report what they did. With compiled equations the symbolic derivatives are no longer
  also taken for the interpreter. A block-elimination solver (Laffargue-Boucekkine-Juillard, keeping only the forward
  variables' dependence) would cut memory further; it was not needed at this scale.
- **Period-by-period solvers, compiled (31 October 2026):** Newton's residuals and Jacobian for one period are each
  one call of the compiled equations (`pymetrix.models.fast`). Lags, leads and exogenous values are gathered once
  per period, and only the derivatives with respect to the period's unknowns are compiled for the Jacobian.
  Gauss-Seidel's update of each equation's own variable is compiled per equation. The interpreter stays the
  reference (`fast.ENABLED = False`), and a test compares the paths and iteration counts. The loop runs with
  `blas_threads()`, since a 284 x 284 dense solve took 0.78 s with the library's threads on a busy machine, against
  1.5 ms on one. `Model.add_factors` gives the tracking add factors (`addinit`) as `Scenario.adjust` paths, by
  equation name, which every solver already applies.
- **Steady state and linearisation:** `model.steady_state(exogenous, guess)` solves the static model (every lag and lead collapsed). `model.linearize(at, shocks=[...])` gives `LinearModel(A0, A1, A2, B)` for `A0 E[x(t+1)] + A1 x(t) + A2 x(t-1) + B e(t) = 0` in deviations; leads and lags beyond one become auxiliary variables (`x_lag1`, `x_lead1`), shown in output as `x[-2]`.
- **Rational-expectations solution:** `LinearModel.solve()` uses QZ (`scipy.linalg.ordqz`) on the companion pencil, checks Blanchard-Kahn (no stable solution / indeterminacy, with the Taylor-principle hint) and returns `LinearSolution(P, Q)`: `rules()`, `roots()`, `irf()` (a VAR-style `IRF`, so the existing figures apply), `plot()`, `simulate()`, and `system()` / `filter()` for the Kalman smoother on observed data.
- **Dialogs:** Model → Rational expectations… (equations, shocks, params; steady-state values and guesses under Advanced); Model → Solve… gains Params and the Perfect foresight solver. Help topic `forward-looking-models`.
- **Checks:** perfect foresight matches the analytic forward solution and the linear impulse responses to 1e-10; the auxiliary-variable path satisfies the model to 1e-12.
- **Not yet:** second-order perturbation, occasionally binding constraints (zero lower bound), estimation of linearised models by likelihood, a model-file format (since 30 October 2026: `ModelSource`, below).
- **The particle filter (1 November 2026).** Second-order solutions are estimated on the bootstrap particle filter's
  likelihood (`pymetrix.models.particle`). The filter's state is the pruned pair (first-order part, second-order
  part) of the states, so the filter follows the same rules as `SecondOrderSolution.simulate` and a test checks them
  path for path; observations are the measurement map evaluated exactly on each particle, so transforms, trends,
  lags and reported series need nothing new. Everything is vectorised over particles in NumPy; workers were not
  needed at the sizes tried (a growth model, 500 particles, 100 periods: about 65 ms an evaluation, a third of it
  the second-order solve). Common random numbers: each period draws its normals and one resampling uniform whatever
  happens, so a resampling decision that flips does not shift later draws. With them the likelihood is piecewise
  smooth with small jumps, which defeats finite-difference gradients (BFGS stopped at the start), so the mode search
  is Nelder-Mead on the same transformed parameters, restarted until a restart gains less than 0.01, and the
  Hessian is taken over about one standard error (found from the diagonal), not a tiny step, which had given standard
  errors a hundred times too small. Pseudo-marginal Metropolis-Hastings needed no new sampler: the existing chain
  already keeps the current draw's log posterior, so giving each chain a log posterior that draws a fresh seed per
  call (from the chain's own stream, reset with the chain) makes it PMMH, the same draws in any process. The
  guided proposal is the conditional normal of the shocks given the data with the measurement linearised at the
  steady state (one matrix per period for all particles), corrected exactly by importance weights.
- **The analytic gradient of the Kalman likelihood (1 November 2026).** `pymetrix.models.kalman_score` differentiates
  the whole map from parameters to log likelihood in forward mode, all parameters at once: the steady state by the
  implicit-function theorem on the static residuals (so it holds however the steady state was found), the
  linearisation from symbolic second derivatives of the compiled residuals (compiled once, shared by
  `with_parameters` copies like the first derivatives), the solution by Iskrev's implicit-function route on
  `A0 P^2 + A1 P + A2 = 0`, solved column by column on the complex Schur form of P's lagged block (an n x n solve per
  lagged variable, not the n^2 x n^2 Kronecker system), and the filter's recursion with the derivative of every
  quantity carried alongside. Two choices keep the gradient exactly that of the value computed: the filter's
  converged-variance shortcut (riccati_tol) freezes the gain's and variance's derivatives with them, and the state
  is restricted as `needed()` restricts it, with the derivatives' sparsity added to the pattern. The value is the
  native filter's to rounding (statsmodels' differs by its own convergence tolerance, 1e-9 relative). Priors are
  differentiated by one-dimensional central differences of their log densities (closed forms, negligible error and
  cost). Rejected: automatic differentiation (JAX or similar would be a new heavy dependency and would not go
  through QZ), and finite differences of the cheap parts only (the solution's derivative is where the cost is).
  Where it does not apply the optimiser and NUTS keep central differences, and a point where it fails (a singular
  static Jacobian, say) falls back to them for that point. The optimiser passes the gradient through its transforms
  (BFGS on unbounded transforms, then L-BFGS-B); the Hessian at the mode is the central difference of the gradient
  (2k gradients, against 2k^2 log posteriors, and far more accurate). Measured on the New Keynesian template (seven
  parameters, three observables, 100 quarters): a log posterior 4.5 ms, its analytic gradient 23.6 ms, central
  differences 67 ms; the gradient's cost is mostly the filter's Python loop over the periods before the variance
  converges. (Those numbers were the forward-mode filter's; the reverse-mode filter below replaced them.)
- **The gradient's filter in reverse mode; NUTS in parallel (1 November 2026).** Profiled on the New Keynesian
  template (7 parameters, 4 states, 100 quarters) the forward-mode gradient spent 80% of its 21 ms in the filter's
  loop: about 45 small NumPy calls a period, a quarter of them `numpy.linalg`'s Cholesky and inverse (7-8 us each of
  checking for a 3 x 3 matrix), and the variance never converged inside the sample (it would after 112 periods), so
  the converged-variance shortcut never applied. On Smets and Wouters (2007) (36 parameters, 27 states, 230 periods)
  the gradient took 282 ms (16 log posteriors): the filter's K-batched products, and 50 ms evaluating the compiled
  second derivatives (70,000 expression nodes, 810 distinct, each a NumPy call on a scalar). And there the forward
  recursion was wrong: carrying dP through the non-Joseph update amplified rounding errors until it overflowed
  (1e30 by period 230; correct for the first fifty). Changes: (1) the filter's derivatives in reverse mode, the filter
  forwards and its adjoint backwards over the same periods, giving the derivatives in every entry of `T`, `Z`, `d`,
  `H`, `R Q R'` and `P0`, contracted once with each parameter's (`einsum` over K): the work per period no longer grows
  with K, and the variance's adjoint collects into `Pbar(s) = L' Pbar(s+1) L + C(s)` with the closed-loop transition
  `L`, three products a period, stable because `L` is; the stationary start's adjoint is one Lyapunov equation, which
  for up to 12 states shares one LU of `I - T (x) T` with the forward one. (2) LAPACK's `dpotrf` and `dtrtri` called
  directly, `ndarray.dot` for products, and everything that does not need the loop done for all periods at once.
  (3) Common subexpressions in compiled expressions, and Python float arithmetic for plain numbers, with NumPy at the
  edges (`pymetrix.expr.compile`): this also speeds every steady state and linearisation, so the plain log posterior
  of Smets and Wouters fell from 17.3 to 11.6 ms. (4) Small caches: `InvGamma1`'s parameters (a root search on every
  log density), the measurement equations' symbolic derivatives, the second derivatives keyed on the compiled
  functions rather than hashed expression trees. Result: a gradient 6.7 ms on the template (2.0 log posteriors), 41 ms
  on Smets and Wouters (3.6). Considered and not taken: Chandrasekhar recursions (Morf, Sidhu and Kailath; Herbst
  2015) save flops only when the filter is flop-bound, which the template's is not; statsmodels' own version refuses
  missing observations (the template's growth rate has one), gave 2% on Smets and Wouters from the stationary start,
  and a different likelihood from the scaled start. Replacing statsmodels' compiled filter in the plain likelihood by
  the NumPy one: no faster. Numba: not needed for these gains, and a dependency. Caching solution derivatives across
  leapfrog steps: every step moves every parameter, so nothing repeats. NUTS chains now run in other processes
  (`workers`, `pymetrix._processes`), each with its own stream and a target that forgets the last solution at each
  chain's start, so the draws are identical either way (tested).
- **A dense mass matrix for NUTS; the filter's loops compiled (1 November 2026).** NUTS with a diagonal metric scales
  each parameter but cannot undo correlations, and DSGE posteriors correlate their parameters (persistence with shock
  size, a rule's coefficients with each other): on a Gaussian with correlation 0.99 its effective draws per gradient
  are 50 times fewer than a dense metric's. `nuts.py` now takes `metric="diag" | "dense"`: the dense kernel keeps
  `minv` (k x k) and `L^-T` from its Cholesky factor, draws momenta as `L^-T z`, and adapts in Stan's windows from the
  window's sample covariance regularised exactly as Stan's (`n/(n+5) S + 1e-3 5/(n+5) I`); `adapt_metric=False` keeps
  a starting metric. Models offer three: `"dense"` started from the mode's inverse Hessian (carried to the
  unconstrained parameters, `D^-1 cov D^-1`, its eigenvalues floored at 1e-6 of the largest), `"diag"`, and `"mode"`
  (the inverse Hessian kept, step size alone: Laplace's approximation as the metric). Measured (validation notes,
  numba on): the template, one chain of 1,000 draws, the smallest effective sample a second 4.2 diagonal, 10.5 dense,
  16.0 mode (150 warmup), Metropolis-Hastings 4.1; Smets and Wouters, 0.010 diagonal, 0.023 dense, 0.006 mode,
  Metropolis-Hastings 0.005, and NUTS's posterior means matched Dynare's 400,000-draw run within 0.14 posterior s.d.
  where the equal-time Metropolis-Hastings chain's were off by up to 0.64. So dense is the models' default (2.3 times
  the diagonal metric's rate on Smets and Wouters, three times per gradient); `"mode"` wins on the near-Gaussian
  template but on Smets and Wouters the Hessian is a poor guide for some parameters (smallest effective sample 5 of
  1,000, 22 divergences), so it is an option, not the default. Single equations keep the diagonal default (Stan's, and their
  existing results), with `metric=` to choose. Considered: shrinking each window towards the previous metric rather
  than towards the identity (fewer noisy early windows for 36 parameters); not taken, to stay Stan's algorithm.
  The filter: after the reverse-mode rewrite the gradient's time on the template was mostly the Python loop over
  periods (about 10 small NumPy calls a period, at 1-5 us of call overhead each), so a compiled path pays there:
  numba kernels for the variance recursion (Cholesky and triangular inverse by their textbook loops, products by
  BLAS), the means' recursion and the adjoint's two backward recursions, and one kernel for the plain likelihood
  (variance and means in one pass, the converged-variance shortcut, missing observations) that replaces statsmodels'
  filter where nothing starts diffuse: statsmodels spends most of its call setting up a model for one filter.
  Kernels are module-level (`_kalman_kernels.py`) so numba's disk cache serves later processes (closures were
  recompiled in every chain's process: ten seconds each). Numba stays optional (`pymetrix[fast]`, not in `all`, so the
  oldest-versions CI is unchanged), switched off by `_kalman_compiled.COMPILED = "off"` or `PYMETRIX_NUMBA=0`, and
  the NumPy loops remain the reference (equal to 1e-10, tested where numba is installed). On Smets and Wouters the
  gradient gains less (49 to 40 ms): its remaining time is the batched products over 230 distinct variance states
  (the variance does not converge in sample), the solution's derivative and building the model. Tried and dropped:
  replacing NumPy's stacked products by one large product through reshapes (slower: NumPy already calls BLAS per
  matrix, and the reshapes copy).
- **The particle smoother (1 November 2026).** The pruned second-order state moves deterministically given the shocks
  (and lagged copies for the measurement equations, and fewer shocks than states, make even first-order transitions
  degenerate), so FFBSi with a transition density, and the O(N^2) marginal FFBSm, have no density to weigh by: their
  backward weights collapse onto each particle's own ancestor, which is the genealogy smoother and its path
  degeneracy. Written in the shocks, the unknowns (start and shocks) are independent a priori, and backward
  simulation for such non-Markovian models (Lindsten and Schön 2013) weighs each candidate particle by its filter
  weight times the likelihood of the later data along the path re-simulated from it with the path's shocks. That is
  exact for any measurement equations, lags and proposal, at a cost of order paths x candidates x T^2/2 particle
  steps; to bound it each period's candidates are drawn from the filter's weighted particles (200 by default, each
  path its own), and the re-simulation stops, exactly, once every candidate has reached the same state. Paths are
  re-simulated from their start and shocks at the end, so they are model paths and give smoothed shocks too. The
  smoother draws its random numbers after the filter's, so smoothing never changes the likelihood. Measured: the New
  Keynesian template, 60 quarters, 2,000 particles, 100 paths of 100 candidates: about 2 seconds.

### Systems of equations (built 11 October 2026)

- **Native, not a wrapper:** `pymetrix.systems` (replacing the stage-3 stub in `multivariate.py`). `System(equations, instruments=, endogenous=, restrictions=)`; `.fit(data, method=, span=, where=, start=)` returns `FittedSystem`. linearmodels is used only to cross-check in tests (SUR, 3SLS and White errors agree to 1e-10).
- **Methods** (new extension group `systems`): `SystemOLS`, `TwoSLS`, `SUR`, `ThreeSLS`; options cov (classic, white, hac), iterate, df\_correction, hac\_lags, tol, max\_iter. Iterated SUR is ML; its standard errors use the converged residual covariance (linearmodels' iterated SEs differ slightly).
- **One estimator:** Gauss-Newton on sum over t of u\_t' S u\_t, with S = I or inverse Sigma and derivatives projected on the instruments for IV methods. Linear equations converge in one step; nonlinear explicit-form equations start from their own NLS fits. Sandwich covariance B^-1 M B^-1 covers all methods and covariance choices.
- **Naming and restrictions:** linear-form coefficients are `equation.coefficient` (`c.y`); explicit-form parameters keep their names, so a shared parameter is a cross-equation restriction. `restrictions="c.y = i.y"` (linear or not) is imposed exactly, through a KKT step and the null space of R in the covariance. As with single-equation Wald, only coefficients of plain-name terms can be referenced.
- **Instruments:** identities are not estimated but make their left-hand sides endogenous (Stata reg3 convention). The instruments are the constant, every regressor (or explicit-form variable at its lag) without a current endogenous variable, and `instruments=`. Order condition checked per equation.
- **Results:** `FittedSystem` has the joint params and cov, `sigma`, `correlation`, `resid`, each equation as a `FittedEquation` (`fit["c"]`, with forecast and tests), `summary()`, `equation_stats()`, `wald()`, `test_diagonal()` (Breusch-Pagan LM), a Hansen-Sargan J for 3SLS, `plot()`, and `model()`. `Model([fitted_system])` brings in its equations and identities.
- **UI:** Estimate → System of equations… (equations, method plus settings, instruments, restrictions, sample; further endogenous variables under Advanced). The view has Summary, Graph, Equations, Residual covariance, Residual correlation and Residuals tabs. Help topic `systems`.
- **Not yet:** FIML, system GMM, panel systems, equation-specific instruments, and running system tests from the Run test dialog.

### Panel data tools (built 11 October 2026)

- **Storage unchanged:** a panel is a long pandas frame with a MultiIndex (entity axes, then time); `data.frame` exposes it for any pandas operation.
- **Group-wise functions in the expression language:** `mean`, `sum`, `sd`, `min`, `max`, `median`, `count`, `demean`, `share`, `rank`, taking `(x)` or `(x, group)`. The group is an axis or a category series, held in a new `Level` node (not a `Name`, so it is not a variable of the expression). They work wherever expressions see pandas Series: `eval`, derived series (a grouping series counts as an input, so changes recompute) and linear-form regressors. Model solvers raise a clear error. Statistics use the whole dataset, not the estimation sample.
- **Dataset methods:** `pivot` (now wraps the whole of `pivot_table`: several rows, columns, values and aggregations, totals, fill, span/where; default aggregation changed from sum to mean, as in pandas); `wide` / `long` (stubs inferred from a shared prefix, or `name=` for a column-per-entity sheet; unmatched columns copied to every entity); `xs`, `entity`, `cross_section`; `coverage` (first, last, observations, gaps; balanced note); `balance(keep="periods"|"entities")`; `complete`; `describe_panel` (overall, between and within, Stata xtsum-style).
- **Fix:** `Dataset.from_frame` no longer adds a stray `index` column from a filtered frame's row numbers.
- **Dialogs:** Data → Pivot table…, Panel to wide…, Wide to panel…, One entity or period…, Balance panel…, Panel statistics…. Two new field kinds: `level` (axes, then category series, then numeric series) and `entity` (the values of the entity axis). Help topic `panel-data-tools`.
- **Not yet:** pivot tables as an interactive drag-and-drop grid, more than one entity axis in wide/long (by= chooses one), and group-wise lags of other entities (spatial or network weights).

### Bayesian VARs (built 11 October 2026)

- **API as planned:** `VAR(...).fit(data, method=Bayes(prior=Minnesota(...)))`. `Bayes` gained a `prior` field and seed 0 by default; for single equations it is still planned. Module `pymetrix.bvar`; result `FittedBVAR`, a subclass of `FittedVAR`, so identification, FEVD, tests, views and the IRF dialog work unchanged.
- **Prior:** Minnesota in conjugate normal-inverse-Wishart form (Kadiyala-Karlsson; Giannone-Lenza-Primiceri). Prior mean `own_mean` (1 random walk, 0 growth rates, or per variable). Coefficient variance `(tightness / lag**decay)**2 * psi_i / psi_j`, with psi the residual variances of univariate AR(`scale_lags`) regressions and decay 2 by default. Deterministic terms get a loose variance (1e6 times Sigma). Sigma \~ IW(diag(psi), K + 2), so E\[Sigma\] = diag(psi). Optional sum-of-coefficients (mu) and single-unit-root (delta) dummy observations.
- **Exact posterior and marginal likelihood:** the GLP closed form; with dummies, p(Y) = p(Y, Yd) / p(Yd). `optimize=True` maximises it over log hyperparameters (L-BFGS-B), without GLP's hyperpriors.
- **Results:** the posterior mean is the point estimate (`params` cov = E\[Sigma\] (x) Omega\~, the exact posterior covariance); `draws_B`, `draws_sigma` exact draws (seeded, so projects replay identically); `irf` gives the posterior median with credible bands (`draws=0`: at the posterior mean); `forecast` simulates the predictive distribution (mean, sd, median, interval, including parameter uncertainty); `explosive_share()`; `log_ml`; `hyperparameters`.
- **Checks:** the marginal likelihood equals SciPy's multivariate-t density exactly; a loose prior gives least squares to 1e-6; a tight prior gives the prior mean; the chosen hyperparameters are a maximum.
- **UI:** Estimate → VAR… gains a Bayesian group (tightness, own mean, the two dummy priors, choose hyperparameters; draws and seed under Advanced). Help topic `bayesian-var`.
- **Not yet:** GLP hyperpriors and MCMC over hyperparameters, stochastic volatility, conditional forecasts and scenarios, large-BVAR speed-ups. (Sign restrictions on posterior draws: done, one accepted rotation per draw; zero and sign restrictions with ARW's weights: done.)

### Project trust (built 11 October 2026; review finding R1)

- **The risk:** opening a `.pymx` replays its steps, and steps typed in the Code or Immediate windows (`RunCode`) are run with `exec`. Dialog steps only run pymetrix commands, and their expressions go through our own parser. So only typed code can do arbitrary things. Installing a plugin package is already an act of trust and is not covered here.
- **Decision (owner, 11 October):** typed code runs on opening only when the project is trusted. Trust is remembered by a SHA-256 fingerprint of the project's runnable typed code, not of the file, so re-saving keeps trust and any change to the code asks again. The store is `trusted_projects.json` in `~/.pymetrix` (or `PYMETRIX_HOME`; the test suite points it at a temporary folder). Projects you save yourself, with all their code run, are trusted automatically.
- **API:** `load_project(path, trust=None | True | False)`. None runs only trusted code; True runs it and remembers; False never runs it. `project_code(path)` returns `ProjectCode(steps, fingerprint, trusted)` so the code can be read first. Untrusted opens warn with `ProjectTrustWarning`. Module `pymetrix.trust`: `code_fingerprint`, `is_trusted`, `remember`, `forget`, `trust_file`.
- **Not run, not lost:** untrusted code steps, and later steps that then fail (they may need the code's results), are kept as `CommandRecord(skipped=True)`. The History window shows them as *not run* in grey, the script has them as comments, and saving keeps them, but without trusting them.
- **Desktop:** File → Open project shows the code in a read-only box with *Run the code and trust this project*, *Open without running the code* (the default) or *Cancel*.
- **Not yet:** signed projects, a Help → Trusted projects list to revoke trust, and a notebook-friendly prompt.

### Review quick fixes (11 October 2026; findings R3 to R6)

- **R3, plugin versions:** `PluginInfo.requires` specifiers are now enforced with `packaging` (a new core dependency). `installed_version` maps import names to distributions; `version_problem`; `PluginInfo.unsupported` and `.problem`; `Entry.problem`. An installed but unsupported library makes a plugin unavailable, and the message names the version needed (`pip install "numpy>=999"`). An unreadable specifier is not held against the plugin.
- **R4, CI:** `.github/workflows/tests.yml` runs the full suite on Windows, macOS and Linux with Python 3.11, 3.12 and 3.13, with every optional library, plus a core-only job on Linux (3.11, 3.13). Tests that need statsmodels or matplotlib skip or adapt without them; the core-only run was reproduced locally first (206 passed, 29 skipped). A tokenizer scan found no f-strings that need Python 3.12. `jedi` is now listed in the `qt` extra.
- **R5, teaching path:** the example data ship with the package (`pymetrix.datasets`, `Dataset.example("sample_macro")`), so notebooks run from any folder, including courses unpacked from projects; the sample data's units are now neutral. Lecture 1's wrong instruction is fixed. New **Lab 2** (note and notebook): estimate and test, build and solve a model, a government-spending scenario (the multiplier rises from 2.8 to 6.4), save and reopen a project and check the results. Every course notebook runs in the tests, from a different folder.
- **R6, public API:** planned placeholders (`GMM`, `Quantile`, `PCA`, `FactorModel`, `MonteCarlo`, `Bootstrap`, `SignRestrictions`) are no longer exported from `pymetrix`; they remain in their modules.
- **Still open:** R2 (stable command IDs, schema versions, migrations) and R7 (a page of defaults compared with EViews, Stata and IRIS).

### Command contract and project format 2 (built 11 October 2026; review finding R2)

- **Stable IDs:** every command has `command_id` (`plugin:snake_name`, such as `pymetrix:fit_equation`), set by `@command(menu, id=None)` from the plugin being registered, unless the class or decorator gives one. It is checked for form and uniqueness. `COMMAND_IDS` maps IDs to `CommandInfo(cls, menu, id, plugin, plugin_version)`; `COMMANDS` by class name remains. The 43 built-in IDs are frozen in `tests/fixtures/command_ids.json`: a test fails if one disappears or a new one is not added. A renamed class must keep its old ID.
- **Schema versions and migrations:** `Command.schema_version` (default 1). `@migration(command_id, from_version)` registers a function from old fields to new. `upgrade(id, version, fields)` applies the chain and then rejects fields the installed command does not have, with a clear `MigrationError`. Adding a field with a default needs no migration. The conformance kit checks that every schema step has a migration.
- **Project format 2:** steps are saved as `{id, version, command (class name, for readers), fields}`, and the manifest lists the plugins used, with versions. Opening refuses a newer project format, a step from a newer schema ("upgrade X"), or a step whose plugin is missing, naming the plugin and its saved version. Format-1 projects (steps by class name) still open, with the pre-ID renames kept as a legacy table. Re-saving writes format 2.
- **Compatibility fixture:** `tests/fixtures/projects/format1.pymx`, a real format-1 project made with the code as it was (selections, derived series, fits, an identity, a model solution, a Bayesian VAR, a pivot table and typed code, with the old `DefineGroup` and `group=` names). Tests open it, check the results, re-save it in format 2 and reopen it.
- **CI fixes found by the first run:** the in-process IPython console no longer keeps IPython's history database (pymetrix records history itself). Its background thread crashed Python at exit with Qt loaded, failing runs whose tests had all passed. Three tests assumed one platform's numerics; each now checks what the method guarantees: the trend forecast rises by the drift, the filter agrees to 1e-5, and the hyperparameter optimiser runs to tight tolerances.

### Seasonal adjustment and calendar effects (built 11 October 2026)

- **Library rule:** `pymetrix.seasonal` wraps statsmodels for `STL` (LOESS; seasonal smoother, robust, optional trend length) and `Classical` (`seasonal_decompose`), and statsmodels' X-13 interface for `X13` (log, outliers, trading days). X13 needs the Census `x13as` program, found from `program=`, `X13PATH` or the PATH, with a clear message when it is missing. Methods register in a new `seasonal` extension group, so plugins can add TRAMO-SEATS or JDemetra+.
- **Dataset:** `seasonal_adjust(names, method=STL(), derived=True)` adds `{n}_sa` and `{n}_seasonal`. Log is None by default, which means multiplicative when every value is positive: the seasonal is then a factor and sa = y / factor; otherwise additive. Panels are adjusted within entity; monthly and quarterly data only; at least three years; the series must be unbroken. The results are derived series (new recipe `SeasonallyAdjusted`, saved in Parquet and projects), so revisions recompute them. The sa variable is marked `Adjustment.SA`, with label and units from the source.
- **Calendar regressors (native):** `calendar_effects(index, trading_days="six"|"one"|"none", easter=8, leap_year=True)` and `Dataset.calendar(...)`, using X-13's definitions: weekday-less-Sunday contrasts, or weekdays less 5/2 of weekend days; Easter\[w\] = the share of the w days before Easter Sunday in the period; leap year 0.75 or -0.25 in February (Q1 for quarters).
- **Checks:** STL and classical results equal statsmodels' to machine precision; calendar values match the calendar (March 2021, Easter 2021 split 0.625/0.375, Q1 2021 weekday contrast -1, leap Februaries, Easter shares summing to one each year). X-13 is tested where the program is installed; elsewhere its message is tested.
- **UI:** Data → Seasonal adjustment… (method and settings, Keep updated) and Data → Calendar regressors…. Help topic `seasonal-adjustment`.
- **Not yet:** X-13's diagnostics tables, seasonal adjustment of weekly or daily data.
- **27 October 2026, every setting:** X13 writes its own specification and runs x13as itself (statsmodels' wrapper wrote a
  fixed specification and read only D11, so it could not do SEATS). Fields left at their defaults leave their options
  out, so the default specification is statsmodels' and results are unchanged; `spec` appends any other text. The
  seasonal factor is y / SA unless the decomposition is additive (mode add, or no logs). STL and Classical take every
  statsmodels argument, statsmodels' defaults kept out of the call and its record.
- **27 October 2026, TRAMO-SEATS:** `TramoSeats` runs JDemetra+ 3.9 (Eurostat's reference implementation) in Java
  through JPype rather than a reimplementation, as X13 runs the Census program: results are the ones statistical offices
  publish. `pymetrix.jdemetra` holds the jar list with SHA-1s, `install()` (Maven Central, checked), `jar_folder()`
  (`libraries=`, `JDPLUS_PATH`, then `~/.pymetrix/jdemetra/<version>`) and `classes()` (starts the JVM once, with
  the jars; explains a missing JPype, Java, jars or an old Java). The jars are EUPL 1.2 and are not shipped. The spec is
  JDemetra+'s predefined one changed through its builders (`TramoSeats.specification`); settings at their defaults
  leave the specification as JDemetra+ defines it (a test checks it equals RSAfull). The seasonal factor is y / SA,
  or y - SA when the decomposition is additive. Not done: national holiday calendars, fixed coefficients,
  JDemetra+'s diagnostics beyond the model and outliers, revision histories.
- **29 October 2026, TRAMO-SEATS interventions and user regressors:** `interventions` become pre-specified outliers in
  the TRAMO regression spec (`AdditiveOutlier`, `LevelShift` zero-ended as TRAMO's own, `TransitoryChange` with 0.7,
  `PeriodicOutlier`), labelled as JDemetra+ labels the outliers it finds, so they can be told apart in the notes.
  `regressors` are user variables (`TsContextVariable("r.<name>")`) in a `ModellingContext` of the series' own, filled
  from the dataset over the series' sample (`StaticTsDataSupplier`), each given its component with
  `SaVariable.setComponentType`; the default is the calendar effect, as holiday regressors are in JDemetra+'s own
  workflow. The regressors are inputs of the derived adjusted series. Holidays as JDemetra+ national calendars were
  left out: a series of holiday counts covers them, and calendars would need a calendar model in pymetrix.

### Forecast workflow (built 11 October 2026)

- **`pymetrix.forecasting`**, written natively: no Python library covers forecast rounds; Diebold-Mariano is not in statsmodels or arch.
- **ForecastRound:** paths (variables and optional `{v}_se`), made, data\_end, notes, source, the logged `adjustments` and `base` (the paths before judgement). `from_forecast` accepts an equation's forecast (`as_name=`), a VAR or BVAR forecast, or a model Solution (`variables=`). `adjust(variable, {period: value} | value, reason, how="add"|"set")` needs a reason; `judgement()`, `table()`, `plot()` (fan chart).
- **ForecastStore(folder):** one Parquet file (paths plus `__base__` columns) and one JSON record per round. `save` refuses a different round with the same name but accepts an identical one, so project replay is harmless. `names`, `load`, `rounds`, `compare` (revisions), `history`, `plot_history` (a hedgehog chart), and `accuracy` (the real-time errors of past rounds by horizon from their `data_end`).
- **Fan charts:** `fan_figure` draws bands from `{v}_se` (normal) or from explicit `{v}_lo{pct}`/`{v}_hi{pct}` columns; 30/60/90 per cent by default.
- **Backtests:** `backtest(fitted|spec|function, data, origins, horizon, vintages=None)` re-estimates a fitted equation or VAR (with its method, so a BVAR keeps its prior) on data to each origin and forecasts dynamically. `vintages` maps each origin to the dataset as then released (real-time); errors are always against the latest data. `Backtest.accuracy()`, `.compare(other, variable, horizon)`, `.plot()`.
- **`diebold_mariano(e1, e2, horizon, loss="squared"|"absolute", small_sample=True)`:** MA(h-1) long-run variance, the HLN correction and t(n-1) p-values. At h = 1 it equals SciPy's one-sample t-test of the loss differential, which is tested.
- **UI:** a new Forecast menu with Make round, Adjust round, Save round, Compare rounds and Backtest; views for rounds (Paths, Graph, Judgement) and backtests (Accuracy, Graph, Forecasts). Help topic `forecast-workflow`.
- **Fixed on the way:** empty Start boxes in dialogs were passed on as a period called "" (a VAR with only an End date failed); an empty box now means no limit.
- **Not yet:** forecast combination, density forecast scores (log score, CRPS, PIT), conditional forecasts in rounds, and a store database for many users.

### Data connectors (built 11 October 2026)

- **`pymetrix.ext.connectors`**, a bundled reference plugin (`PluginInfo` pymetrix-connectors), written against the Extension API only: one `io` reader per URI scheme. It uses the standard library (urllib, csv, json, ElementTree) and the providers' public web services. `sdmx1` would be the library to wrap, but it is not installed and the services' CSV is simple.
- **`fred://ID1,ID2`:** FRED's CSV download, one request per series (several frequencies at once come back as a ZIP, so mixing frequencies is refused). `?vintage=YYYY-MM-DD` reads the ALFRED vintage. Frequency comes from the date spacing. No release date is available, so there is no vintage; the source records the ALFRED date.
- **`worldbank://IND1,IND2?country=A;B&date=2000:2023`:** the JSON API, following pages; a panel by ISO3 country. The World Bank's `lastupdated` date is the vintage.
- **`sdmx://PROVIDER/FLOW/KEY?startPeriod=...`:** SDMX-CSV (ECB, Eurostat, OECD, BIS) and SDMX-ML structure-specific XML (the IMF returns XML even when CSV is asked for). The dimensions are the key's positions (later columns or attributes are attributes; a live test found attributes such as BIS compilation notes being used as names). Columns are named by the dimensions that differ; `name=` names a single series; `entity=DIMENSION` makes a panel. Eurostat's LAST UPDATE and the IMF's UPDATE\_DATE become the vintage. SDMX error messages are passed on. `register_sdmx_provider(name, url_template)` adds any SDMX service, such as an institution's own warehouse.
- **Tests use recorded responses** (`tests/fixtures/connectors`, trimmed real responses; the IMF one is synthetic because of its licence) by replacing `connectors._get`, so they never use the network. Every provider was also checked live once.
- **Qt fixes found on the way:**
  - The Code window's editor no longer owns its document. A document deleted inside the editor's destructor emitted signals into Python slots on a half-destroyed widget, which could crash when a Code window was destroyed or at exit.
  - The test suite deletes leftover Qt windows at the end of the session (leaving popups to their owners), so Python's final clean-up no longer crashes after all tests pass. This was reproduced locally and was the cause of the CI failures.
- **Not yet:** a FRED API-key route for per-observation vintages, SDMX structure (codelist) look-ups, caching, and a Data → Get data dialog with a provider browser. (Look-ups, the cache of searches and metadata, and the browser were built on 1 November 2026: below.)

### Series browser and connector cache (1 November 2026)

- **`pymetrix.connectors`** (core, public): `search(provider, text, limit=, source=, refresh=)`, `browse(provider, path)`, `structure("sdmx", "PROVIDER/FLOW")`, `to_frame`, and the cache functions. Every provider answers with the same frozen `SeriesInfo` (provider, code, title, frequency, units, start, end, last\_updated, notes, seasonal\_adjustment, source, kind; `address` is what `Dataset.read` takes), `Category`/`Listing` for browsing and `Structure`/`Dimension` for SDMX keys (`key(...)` checks codes against the codelists; an empty dimension is the SDMX wildcard). These are plain records, not results: a search is a lookup, not a step, so it is not recorded and has no chart.
- **Searchers are a registry** (`register_searcher(scheme, obj)`), so plugins add their own; `Connector(search=True, search_note=...)` declares it to the dialogs. The bundled ones (`pymetrix.ext.connector_search`): FRED's web service (`series/search`, `category/children`, `category/series`; it needs a key, so a missing key is a clear error before any request; the CSV route for data is unchanged), the World Bank (indicator lists of a source or topic, fetched whole and cached, then filtered locally, since the v2 API has no indicator search; title and code matches before notes-only matches), and SDMX 2.1 REST (`/dataflow/{agency}` and `/dataflow/{agency}/{id}/latest?references=all`, Eurostat `descendants`), parsed by local tag names so prefixes do not matter. A flow of another agency is written `AGENCY,ID` (as the IMF's are), the form the reader already takes.
- **The cache is at the HTTP level**: each answer is a JSON file in `$PYMETRIX_HOME/cache/connectors` (default `~/.pymetrix`), named by a hash of the request URL with credential parameters (`api_key`, `token`, ...) removed, holding that redacted URL, the time stored and the body; time to live by kind (search a day, metadata a week); written atomically; a damaged entry is asked again; `refresh` skips reading but rewrites. Keys never reach disk: they are removed from the stored URL, any echo of them in a body is masked, and transport errors that carry the URL are rewritten with `api_key=***`. API keys come from `<PROVIDER>_API_KEY` or `set_api_key`, held in memory only. Downloaded data are not cached: `Dataset.read` keeps its provenance (retrieval time) honest.
- **File → Browse series…** (`pymetrix.ui.series_browser_qt`) is an app action in the File menu (`@series-browser`, like the spreadsheet), not a command: it records nothing itself. Load runs `LoadFromConnector` (so the step is `Dataset.read(address, vintage='latest')`); Add to Load from connector… opens that dialog with the command as its initial value. Mixed FRED frequencies are refused before loading.
- **Tests** answer every URL from fixtures; the search fixtures are hand-written from documented formats because the build machine could not reach the providers (no FRED key either).

### Reports (built 11 October 2026)

- **`pymetrix.report.Report(title, author, date, provenance=True)`:** an ordered list of blocks. `heading(text, level)`; `text()` (blank lines separate paragraphs); `add(obj, caption, sections)`. A Table, Figure or DataFrame goes in as it is. Any result goes through the same view sections as its window: by default its summary and graph, `sections=[...]` by tab title, or `"all"`.
- **Excel (openpyxl):** a Contents sheet with title, author and date, the headings and text, and internal hyperlinks to one sheet per table or figure. Numbers are stored as numbers with formats; figures are PNG images. **Word (python-docx, new `report` extra; also in `dev`):** title, headings, numbered table and figure captions, tables with notes, pictures 6 inches wide, document properties. **HTML:** self-contained, with figures embedded.
- **Provenance appendix:** for every result added, the method, sample, observations, data vintages (release dates) and pymetrix version. Provenance is made visible in deliverables.
- **UI:** File → Export report… (objects in order, path, title, author; all sections and the appendix under Advanced). Help topic `reports`.
- **Testing note:** python-docx is not installed globally. The Word tests ran in a repo-local `.venv` (git-ignored, with system site packages plus python-docx), and CI installs it through `dev`.
- **Not yet:** PowerPoint (python-pptx), report templates or house styles, and editable Excel charts (native chart objects rather than images).

### Temporal disaggregation (built 11 October 2026)

- **`pymetrix.disaggregation`**, native: there is no mature Python library; R's tempdisagg is the reference, and its conventions are followed.
- **`TemporalDisaggregation(formula, method, conversion, rho=None, rho_bounds=(0, 0.999)).fit(low, high)`:** the formula is the low-frequency target on high-frequency indicator expressions (`gdp ~ ip + emp`, or `gdp ~ 1`). Methods:
  - `chow-lin`: AR(1) residual, rho by maximum likelihood (a 41-point grid, then a bounded search);
  - `chow-lin-minrss`;
  - `fernandez`: a random-walk residual, with covariance (D'D)^-1;
  - `denton` (proportional) and `denton-additive`: Cholette's first-difference form, solved through the KKT system.
- **Conversions:** sum, average, first, last, through an explicit conversion matrix that checks every low-frequency period is fully covered.
- **Extrapolation:** the indicators' unbroken span sets the high-frequency sample, so regression methods extrapolate past the last low-frequency period.
- **FittedDisaggregation:** `series`, `params`, `cov`, `rho`, `check` (the largest gap from the low-frequency data, below 1e-8 in tests), `summary()`, `plot()` (with the low-frequency data spread evenly).
- **Checks:**
  - Chow-Lin's GLS step equals statsmodels' GLS (coefficients and standard errors).
  - Every method meets the totals.
  - Proportional Denton recovers a series proportional to its indicator exactly.
  - Rho is estimated near the true 0.8.
- **UI:** Data → Temporal disaggregation… (adds the series to the high-frequency data). Help topic `temporal-disaggregation`.
- **Not yet:** MIDAS and bridge-equation helpers (Litterman, aggregation matrices, soft constraints, seasonal Denton and two-way reconciliation were added on 1 November 2026, below).

### Temporal disaggregation, extended, and reconciliation (built 1 November 2026)

- **Aggregation matrix:** `fit(low, high, aggregation=C)`, an argument of `fit` rather than of the specification because it is data (it is named in the recorded call, `aggregation=C`, like the datasets), replaces `conversion`. Rows are matched to the low-frequency values by name (a cross-section dataset's row names, or periods), columns to the high-frequency periods by name; unnamed matrices and arrays by position. A dataset at the high frequency is read as the transpose (one series of weights per value), which is how such weights sit in a spreadsheet. Rows without a value are dropped; the rest must have weights, lie within the indicators' sample and be independent (rank). `aggregation_matrix(high, periods)` builds one from periods of any frequency, named `2020 Y-JUN` where the period alone would be ambiguous.
- **Soft constraints** (`soft=`): a standard deviation in the data's units, so that one number means the same thing whatever the method's scale. Regression methods: Y = C X b + C u + e with e ~ (0, diag(sd²)); the scale of u can no longer be concentrated out, so it is chosen by maximum likelihood (a grid in its log around the exact fit's, then a bounded search), for each rho when rho is estimated. Denton: the penalty ((C y − Y)/sd)² in a scaled KKT system (multipliers times sd), which stays well conditioned from sd → 0 (the exact system) to sd → ∞ (the indicator moved by a constant). sd = 0 for a value keeps it exact.
- **Seasonal differences** (`differences="seasonal"`): Denton-Cholette with lag 4 or 12 differences and `level_penalty` (0.001 by default) times the first-difference criterion, which pins the within-year pattern that seasonal differences leave free (with zero, the minimum-norm solution).
- **Litterman** (`litterman`, `litterman-minrss`): covariance (D'H'HD)^-1 with H = I − rho L, as tempdisagg's; the rho search is Chow-Lin's.
- **Rho's range:** unchanged default [0, 0.999]; an estimate within 1e-4 of an end is noted in the summary and in `provenance.messages`.
- **Reconciliation** (`pymetrix.reconciliation.reconcile`): the free components stacked, a block-diagonal criterion (proportional or additive first differences, Di Fonzo and Marini; or least squares with variances proportional to the values), constraints for the total each period and each component's low-frequency values. Solved by a minimum-norm particular solution and the criterion's minimum over the constraints' null space, so the constraints hold to rounding error; a particular solution that misses is reported as contradictory constraints. One result type, `Reconciliation`, with provenance, diagnostics, a graph of the adjustments and a window.
  **Weighted least squares (8 November 2026):** `method="weighted"` is the same machinery with block `I / w_c` for each free component, so the variance of an adjustment is the component's weight alone, not scaled by its level: the least-squares method's `|x|` scaling makes a component near zero almost immovable and has no natural meaning for one that changes sign, and the weighted criterion is what a user means by "this component takes four times the share of the gap". A zero weight is refused rather than read as "fixed", since `fixed=` already holds a component exactly and the other methods refuse it too; the message points there. It is a choice of the existing `method` field, so the Reconcile step keeps schema 1.
- **Defaults:** every new argument's default gives today's numbers bit for bit (checked against the previous module on every method, conversion and rho option while building).
- **UI:** the Temporal disaggregation dialog gains the Litterman methods, seasonal differences and their weight, soft constraints and the aggregation matrix (a workspace matrix or dataset, or a file); Data → Transform → Reconcile components… (`Reconcile`, ID `pymetrix:reconcile`).
- **Checks** (`tests/test_disaggregation_extensions.py`): C·y = Y to 1e-10 with calendar and fiscal years mixed; GLS formulas; the soft limits; the seasonal QP by elimination; Litterman = Fernández at rho = 0 and against the process's covariance; reconciliation against the KKT system and the pro-rata rule.

### Dynamic panel GMM (built 11 October 2026)

- **`pymetrix.estimation.dynpanel.ArellanoBond`**, native (no installed library does it; xtabond2 conventions). It replaces the stage-5 placeholder and the `GMMLags` class. Fields: `gmm` (variables instrumented GMM-style; default the dependent variable), `lags` ("2:", "2:4"), `system`, `two_step`, `collapse`, `time_effects`. Registered twice in `methods`: Arellano-Bond GMM, and Blundell-Bond system GMM (preset `system=True`), so the Estimate dialog offers both.
- **Construction:**
  - The panel is made complete; regressor terms are evaluated with lags within entity.
  - Differenced rows are instrumented by lagged levels (one column per period and lag, or per lag when collapsed). Missing instruments count as zero.
  - Regressors not involving a GMM variable instrument themselves (their differences, or their levels in the levels equation).
  - System GMM adds levels rows instrumented by lagged differences, plus a constant. Period dummies are optional.
  - Never-varying or collinear dummies and all-zero instrument columns are dropped.
- **Estimation:** the one-step weight is (Σ Z\_i' H\_i Z\_i)^-1, with H the MA(1) pattern (2, -1) for differenced rows and the identity for levels rows. One-step standard errors are robust. Two-step uses the one-step residuals, with the Windmeijer (2005) correction computed from the derivative of the weight matrix. Hansen J. Arellano-Bond AR(m) tests follow Roodman's formula, including the estimation-error terms.
- **Checks:**
  - On a simulated dynamic panel (rho 0.5): OLS 0.80, within 0.36, Arellano-Bond 0.53, system 0.51. AR(1) rejects, AR(2) does not.
  - The exactly identified case equals linearmodels' IV2SLS on differenced data to 1e-10.
  - Across 40 simulated panels, the mean Windmeijer standard error is within 30% of the spread of the estimates (0.86 in practice).
  - Unbalanced panels work.
- **FittedEquation.summary** now shows method notes from `extra["notes"]` (here the GMM description and the specification tests).
- **Not yet:** different lag ranges per GMM variable, orthogonal deviations, the difference-in-Hansen test, and predictions from difference GMM (it has no constant).

### Panel unit-root tests (built 11 October 2026)

- **Tests that need no tables, in `pymetrix.stats.panel_tests`.**
  - `fisher_adf`: ADF entity by entity (statsmodels), combined by Maddala-Wu (P = -2 Σ ln p, χ² with 2N degrees of freedom) and Choi (Z = Σ Φ^-1(p) / √N). It allows unbalanced panels.
  - `hadri`: the mean of the entities' KPSS statistics, standardised with the closed-form moments (1/6 and 1/45 with a constant; 1/15 and 11/6300 with a trend). It needs a balanced panel. Options: heteroskedastic or pooled variance, and `lags` (Bartlett long-run variance, or `auto`).
- **Checks:**
  - Fisher statistics equal the combination of statsmodels' ADF p-values.
  - Each entity's Hadri statistic equals statsmodels' KPSS (nlags=0).
  - In simulation, the original Hadri test rejects about 3% of white-noise panels but 100% of stationary AR(0.5) panels (its known over-rejection); with `lags=8` that falls to 10%. Random walks are always rejected. The help says so.
- **Not yet:** panel cointegration tests (Pedroni, Westerlund).

### Levin-Lin-Chu and Im-Pesaran-Shin (built 11 October 2026)

- **`levin_lin_chu`** (`LevinLinChu(variable, lags="aic", trend="c"|"ct"|"n")`), native and needing a balanced panel. It follows the paper's three steps:
  - orthogonalised ADF regressions, entity by entity;
  - the ratio of each entity's long-run standard deviation to its short-run one, with Bartlett truncation at floor(3.21 T\~^(1/3));
  - one pooled regression, whose t-statistic is adjusted by the means and standard deviations of LLC's Table 2. The row is the first whose T\~ is at least T − p̄ − 1, as gretl does.
- **`im_pesaran_shin`** (`ImPesaranShin(variable, lags="aic", trend="c"|"ct", replications=100_000, seed=0)`): W t-bar from the entities' ADF t-statistics, computed by statsmodels.
  - The null mean and variance of each t-statistic are **simulated** for that entity's own T and lag order: 100,000 seeded random walks, vectorised, and cached.
  - IPS's Table 3 is not used, because it covers only a grid of T values and recalling it is not checkable. Simulation handles unbalanced panels and any T.
  - The simulation standard error of W, about sqrt(N / replications), is reported alongside the statistic.
- **Fisher-type test:** now also reports Choi's logit statistic, on t(5N + 4).
- **Checks against gretl 2026b:**
  - `levinlin` z-scores agree to 1e-5 in ten cases: no deterministic terms, constant, and constant and trend; T = 27, 40, 55 and 120 (covering five table rows and the limit); lags 0 to 3.
  - The IPS t-bar agrees exactly. W t-bar agrees within simulation error (−0.711 against −0.705; −0.655 against −0.655); gretl reads the table at T = 40.
  - The simulated statistic equals statsmodels' adfuller to 1e-15, and the large-T moments match the Dickey-Fuller limit (about −1.53 and 0.71).
- **Not yet:** IPS's exact t-bar critical values for lags = 0 (simulation over N would give them), and Breitung's test.

### Pesaran's CIPS (built 11 October 2026)

- **`cips`** (`CIPS(variable, lags=0, trend="c"|"ct"|"n", truncated=False, replications=10_000, seed=0)`), native, for panels whose entities share a common factor. It needs a balanced panel.
  - Each entity's ADF regression is augmented with the cross-section mean's lagged level and its current and lagged differences: Pesaran's CADF regression. CIPS is the average of the CADF t-statistics.
  - `truncated=True` gives CIPS\*, with each statistic clipped to Pesaran's bounds (equation 34).
- **Critical values and p-value are simulated** for the panel's own N, T and lags: independent Gaussian random walks, vectorised over replications, seeded and cached.
  - Pesaran's Table II covers only p = 0 and a grid of N and T, and gives no p-value.
  - The result carries the 1%, 5% and 10% critical values, a simulated p-value, and each entity's CADF statistic.
- **Checks:**
  - The CADF statistic equals statsmodels' OLS t on the explicitly built regression (lags and trend) to 1e-10.
  - Simulated critical values reproduce Pesaran (2007) Tables II(b) and II(c) to within about 0.02 in every cell tried: N = 10 to 50, T = 20 to 100, constant and trend. The table was read from R plm's transcription; its layout puts N down each column vector.
  - With one strong common factor and unit roots, IPS rejected at 5% in 26 of 100 simulated panels, CIPS in 5. CIPS rejects a stationary panel with the same factor.
- **Cost:** about 1 second for N = 20, T = 20 and about 6 seconds for N = 50, T = 100 at 10,000 replications, then cached; `replications` trades speed for precision.
- **Not yet:** CIPS with several factors (Pesaran, Smith and Yamagata's CIPSM).

### Bayesian single equations (27 October 2026)

- **`Bayes.estimate`** (was `planned`) calls `estimation/bayes_equation.py`: linear designs from `linear_design`, families
  Gaussian, Logit, Probit, Poisson. Priors by coefficient name from `estimation.priors`, plus `sigma` (InvGamma1) or
  `sigma2` (InvGamma) for a Gaussian error; flat for coefficients without one and 1/sigma^2 for the error.
- **Samplers:** `sampler="auto"` (the new default; "nuts", the old default, never worked and is refused) uses Gibbs when
  each conditional is exact: normal or flat coefficient priors and an inverse gamma (or nothing) on sigma^2, with
  InvGamma1 on sigma mapped to IG(nu / 2, s / 2) on sigma^2. Otherwise random-walk Metropolis-Hastings on (beta, log
  sigma), with the priors' Jacobians, from the posterior mode, steps shaped by the inverse Hessian there, the scale tuned
  in the burn-in (Robbins-Monro towards 0.25). `chains` (2) and `burn` (1000) are new fields of `Bayes`; each chain has
  its own SeedSequence stream.
- **Result:** `FittedBayesEquation` (a `FittedEquation`): posterior means and covariance as `params`, so predictions and
  Wald tests work; `draws`; `posterior()` with HPD, R-hat and ESS from `models.bayes`; its own summary.
- **Fixed on the way:** `InvGamma1.nu_s` bracketed its root up to log(nu - 2) = 40, where the gap function loses every
  digit and reads exactly 0, which brentq returned as the root: priors with a mean of 1 or more became point masses
  (nu near 2e17). The bracket now ends where the function is accurate. The Smets-Wouters and fs2000 priors (means below
  1) were not affected.

### Monte Carlo experiments (27 October 2026)

- **`MonteCarlo(dgp, estimator, reps=1000, seed=0, workers=1, truth={}).run()`** (`simulation.py`; was `planned`):
  `SeedSequence(seed).spawn(reps)` gives each replication its own stream, so draws do not depend on the workers; with
  workers, chunks go through `_processes.run_in_processes` (no multiprocessing re-import), and functions that cannot be
  pickled run here with a note. The estimator's return is read as a number, sequence, mapping, `TestResult` (statistic,
  p-value) or anything with `params`. `MonteCarloResult`: `draws`, `moments()` (bias and RMSE with `truth`),
  `rejection()` with binomial s.e., `failures` (index: error).
- `Bootstrap` stays a settings record; no estimator takes it yet.

### Maximum-likelihood factor analysis (27 October 2026)

- **`FactorModel(variables, factors, rotation="none"|"varimax"|"promax").fit(data)`** (`factors.py`; the placeholder
  is gone; `dynamics > 0` points to `nowcast.DynamicFactor`): R factanal's algorithm on the correlation matrix of the
  complete rows. The uniquenesses minimise the concentrated discrepancy by L-BFGS-B with factanal's start, bounds
  (0.005 to 1), scaling (psi / 0.01), memory (5) and tolerance; loadings from the eigenvectors of Psi^-1/2 S Psi^-1/2.
  `varimax` and `promax` are R's (Kaiser normalisation, eps 1e-5; power 4). Factors ordered by their sums of squared
  loadings and signed so each column sums positive, the factor correlations reordered and signed with them (R 4.3
  does not adjust them). Regression scores multiply by the factor correlations under promax (Thomson's formula; R 4.3
  omits it); Bartlett scores by weighted least squares. The test: Bartlett's corrected likelihood ratio, as factanal.
- **Not yet:** a dialog; other rotations (oblimin, quartimax); factor analysis of a covariance matrix given directly.

### The independent normal-Wishart prior for VARs (27 October 2026)

- **`NormalWishart`** (`bvar.py`), a prior beside `Minnesota` for `Bayes(prior=...)`: the coefficient on lag l of
  variable j in equation i has prior variance (tightness / l^decay)^2, times cross_tightness^2 psi_i / psi_j off the
  diagonal (so with cross_tightness 1 its variances are the conjugate prior's expected ones); Sigma ~ IW as in the
  conjugate prior; dummy-observation priors as extra rows. `_gibbs` alternates vec(B) | Sigma (precision diag(1/V) +
  Sigma^-1 (x) X'X, drawn through its Cholesky factor) and Sigma | B, from least squares, `burn` draws discarded.
- The result is a `FittedBVAR` with the draws, so responses, forecasts and conditional forecasts work unchanged;
  `params` and the summary use the draws' covariance of vec(B) (not Kronecker here); no marginal likelihood (NaN).
- The VAR dialog's Bayesian group gains Independent prior and Cross tightness.

### Hierarchical BVAR priors, Chib's marginal likelihood and the No-U-Turn sampler (1 November 2026)

- **`GLP`** (`bvar_hierarchical.py`), a third prior for `Bayes(prior=...)` on VARs: the conjugate Minnesota prior with
  both dummy-observation priors, its hyperparameters (tightness, sum-of-coefficients, single-unit-root, and the scales
  psi, which here are estimated rather than fixed at the univariate autoregressions' variances) given the authors'
  hyperpriors and bounds. The marginal likelihood uses GLP's eigenvalue formula (cheaper in the search and the MCMC)
  and equals `_niw`'s; the dummies' own normalisation is removed as for `Minnesota`. The lag decay is a setting
  (default 1: variances fall as l^-2, GLP's alpha = 2 in our parametrisation), not a hyperparameter. `mcmc=True` runs
  random-walk Metropolis-Hastings on the logs of the hyperparameters (Jacobian included; proposal from the inverse
  Hessian at the mode; scale tuned every 100 burn-in draws towards 0.25) and one conjugate draw of (B, Sigma) per kept
  hyperparameter draw, so everything downstream that uses `draws_B`/`draws_sigma` integrates over the hyperparameters
  with no change. Point estimates then come from the draws (`FittedBVAR.from_draws`, as for `NormalWishart`).
- **Chib's estimate for `NormalWishart`**: two Gibbs blocks, so no reduced run (Chib 1995, section 2.1): p(Sigma*|Y)
  Rao-Blackwellised over the B draws, p(B*|Sigma*,Y) exact; the dummies are prior, so their own marginal likelihood
  comes from a Gibbs run on them alone. The search (`optimize=True`) is Nelder-Mead on logs with common random numbers
  and shorter runs (500 draws after 200), which makes the surface smooth enough to optimise; the final fit is a full
  run at the chosen values.
- **NUTS** (`nuts.py`) is one NumPy implementation shared by single equations (analytic gradients) and linearised
  models (central-difference gradients through `estimation.numdiff.jacobian`), following Stan's algorithm and warmup
  rather than the 2014 paper's slice variant, because multinomial sampling is more efficient and Stan's diagnostics
  (divergences, tree depth, acceptance statistic) are what users know. Bounded parameters are sampled on the real
  line (`Transform`), so the sampler never leaves the support. For models, `mcmc(sampler="nuts")` rather than a new
  method keeps the `Posterior` result, dialog and call the same; chains run in this process (the finite-difference
  gradient needs the likelihood object, and the run is already dominated by solutions). No argument is named
  `target`: the step-size goal is `target_accept`.

### Breitung, unbalanced CIPS and panel cointegration (27 October 2026)

- **Unbalanced CIPS, as plm does it:** each period's cross-section means average the entities whose CADF row is complete
  then (so a balanced panel is unchanged), each entity's regression uses its own complete rows, and the null is simulated
  with the panel's own missing periods (the mask is part of the cache key).
- **`breitung`** (`Breitung(variable, lags=0, trend="c"|"ct", robust=False)`), balanced panels. Prewhitening by each
  series' own autoregression of its differences (with a constant when there is a trend, since the differences then have
  a drift); each series over its residual standard deviation; with a constant, levels less the first value; with a
  trend, forward-orthogonalised differences against levels detrended by the first and last values. `robust=True` is
  Breitung and Das's sandwich over the residual cross-section covariance; it needs T > N, and is standard normal only as
  both grow (8-11% size at 5% with ten entities).
- **`pedroni`, `kao`, `westerlund`** (`stats/panel_cointegration.py`), variables as "y, x1, x2".
  - Pedroni: the seven statistics as functions of averages of each entity's parts (Pedroni 1999, Table 1); the panel
    statistics' moments by the delta method on the parts' simulated means and covariance, the group statistics' from the
    simulated terms, for each entity's T, lags and bandwidth (unbalanced panels allowed). The headline statistic is
    group ADF t; all seven are in the details.
  - Kao: the within regression, pooled DF and ADF regressions, adjusted with the conditional variance of the residuals'
    differences given the regressors' differences (contemporaneous and Bartlett long-run); balanced panels.
  - Westerlund: each entity's error-correction regression with p lags and q leads; alpha(1) as the ratio of the long-run
    standard deviations of the errors and of dy net of the regressors' leads and lags (using dy itself over-rejects
    whenever the regressors move y); Pt and Pa from the residualised pair, pooled. Moments simulated, as for Pedroni.
    No bootstrap for cross-section dependence yet.

### Demo readiness for the first collaborators (built 11 October 2026)

- **Purpose.** A demo that is credible but not perfect, for a few econometricians and current and former IMF colleagues to try, several of whom have stakes in related tools (EViews, QuantEcon, IRIS and GPM).
- **Lab 3, a quarterly forecast round** (`examples/course/03_forecast_round`): a VAR forecast, judgement with a reason, revisions against the June round, a backtest of VAR against BVAR with Diebold-Mariano, a report with its provenance appendix, then the generated Python and the project reopened. It runs in CI with assertions, as the demo's smoke test.
- **README** rewritten around the workflow and four points: one workflow, reproducible, numbers you can check, Python first. It has a quick start that a test runs, a link check, and an `[all]` install extra. The old sections moved to `docs/guide.md`, and the status table to `docs/features.md`.
- **`docs/validation.md`:** about 130 independent checks by area, with wrapper checks distinguished from independent ones, and a list of native code not yet checked against an outside reference.
- **`docs/known-gaps.md`:** release status, the one-developer and AI-assisted history (kept public by decision), missing methods, workflow gaps, and limited hands-on use of the app.
- **`GOVERNANCE.md` and `CONTRIBUTING.md`** (drafts):
  - scope as an integration layer that works alongside other tools; you as project lead until three maintainers; lazy consensus;
  - IP rules: no proprietary material, a clean-room rule for related products, staff check their employer's rules; disclosure of conflicts of interest;
  - BSD-3, inbound = outbound, with DCO sign-off and no contributor agreement;
  - every number needs an independent check; library first.
  - Six questions are left open for the collaborators.
- **Fixes the demo work exposed:**
  - Summaries now name the covariance (for example, HAC with its kernel and lags).
  - Four command default names no longer shadow Python built-ins or pymetrix functions in generated scripts (`backtest`, `round`, `slice`, `johansen`); a test guards this.
  - The Phillips-Ouliaris cointegration tests are added, wrapping arch; Zt and Za are checked against the 1990 formulas to 1e-10.
- **IMF material check:** the course is clean; the only IMF references are the public SDMX service, with a synthetic test fixture.
- **Not yet:** a GUI friction pass on the demo dialogs, a clean-install check from a wheel, and a forecast round that carries its model's provenance into reports.

### GUI friction pass on the demo path (11 October 2026)

- **Method.** Lab 2 and Lab 3 were driven through the real dialogs offscreen (`command_dialog`, filled as a user would). An image was saved of each dialog and result window, and the generated script was read after each run.
- **Bug fixed: silent weighted regressions.** The Weights drop-down in Estimate → Equation had no blank entry, so Qt chose the first series: every dialog fit since 9 October was weighted by it.
  - The same applied to any optional single-choice drop-down, in both Qt and Jupyter forms.
  - `Session.choices` now puts a blank first for an optional single choice whose default is empty.
  - Results made through the dialog before the fix should be re-estimated; their scripts show `weights=`.
- **Added:**
  - **File → Load example data** (`LoadExample`, id `pymetrix:load_example`; field kind `dataset`).
  - A Scenario section in **Model → Solve**, with Shift and Set to fields and code `scenario=Scenario(...)`. These are optional new fields, so there is no schema bump.
  - A save-file widget kind (`savefile`) for report paths, and a status-bar message with the full path of any file written.
  - A Start page for an empty workspace.
- **Wording:**
  - Choice codes appear as words in forms (`CHOICE_LABELS`: trend and covariance), while scripts keep the codes.
  - Field labels: Standard errors, Save as, Parameters (`LABELS`, or `Widget(label=...)`).
  - Group headings that only repeat their field are dropped.
  - Summary statistics use their usual names (R-squared, S.E. of regression, Akaike info criterion...).
  - The forecast round's time is shown without seconds.
  - The Immediate window no longer prints IPython's tip.
- **Robustness:** a multi-select set from code keeps every item (signals are blocked while ticking).
- **Tests:** blank optional choices, multi-select set, code labels, example data and scenario commands reproducing Lab 2's multiplier, the savefile widget, and the start page.
- **Not yet:** a check of the app's look with the real Windows fonts, and the same pass on dialogs off the demo path.

### Planning documents and the AI principle (11 October 2026)

- **Added to the repository**, corrected against the code as it stands:
  - `docs/roadmap.md`: stages 0 to 3, from private demo to public 0.1 and hardening. Stage 0 is nearly done; decisions already taken are marked.
  - `docs/adding-features.md`: the path settings dataclass → registry → result → views → generated code → project round trip → validation → documentation.
  - `SECURITY.md`: project trust, plugins, connectors, and reporting.
  - Background notes in `docs/notes/`: positioning among open-source workbenches; AI-assisted development (the TRACE framework, Chen, Zimmermann and Trinkenreich 2026, arXiv:2608.03329); and AI in the UI.
- **Corrections made while placing them:**
  - Paths follow the repository (GOVERNANCE, CONTRIBUTING and SECURITY at the root; `docs/validation.md`; the demo is Lab 3).
  - Registry groups include `seasonal` and `courses`; choices keep pymetrix's codes.
  - Commands do not declare reads and writes: the session records them.
  - The defaults page is a help topic, and the AI assistant is planned, not built. (A first version of this note also said there was no result fingerprint. That was wrong: \`pymetrix.testing.fingerprint\` exists and \`check\_project\` uses it. Corrected later the same day.)
- **AI principle (adopted):** an assistant will sit above the command/session layer. It proposes commands and generated Python; the user approves; the ordinary engine runs them, with history, provenance and replay. It has no private route to change state.
  - Confidentiality modes are off, local\_help, metadata\_only, full\_context and local\_model, with nothing sent silently.
  - Agentic workflows wait until trust, audit, replay, undo and validation are mature.
  - Existing plumbing that serves it: the command registry and `form_spec`, registered views, help topics, and `Session.script()`. Still to build: context bundles, redaction and the proposal format.
- **CONTRIBUTING** adopts the AI-assisted-contributions text; the pull-request template gives a disclosure example; a test checks every relative link in the project documents.

### Three demos, matched to reviewer tracks (11 October 2026)

- **Decision:** three demos on the one simulated economy, each paired with a track in the review brief. Each reviewer starts with their own demo; none depends on AI.
  - **Teaching (track A):** Lecture 1 (data step fixed to Load example data, plus a section on finding course material in the help) and the Lab 2 exercise. The student notebook has hints only; the worked solution, with answers to its questions, moved to `examples/solutions/`, outside the course folder so students' help does not list it.
  - **Python (track B):** new Lab 4, `04_engine.ipynb`:
    - datasets and expressions;
    - one equation fitted by OLS, OLS with HAC and IV;
    - results as objects;
    - a live HAC cross-check against statsmodels;
    - a Ridge plugin registered and passing `check_entry` and `check_result`;
    - a dialog step's script run with `exec`, and project fingerprints compared.
  - **Policy (track C):** Lab 3 gains a structural fiscal scenario.
    - Lab 2's equations are solved past the data, with g held at its last value and at that value plus 5.
    - Each run becomes a round in the store, and the scenario is a comparison of rounds.
    - It goes into the report. Its multipliers match Lab 2's (2.8 on impact, 5.8 after a year).
- **Tests:** every notebook in `examples/course` and `examples/solutions` runs in CI.
  - The student lab must leave the work undone.
  - Lab 3's scenario multiplier exceeds two and builds over time.
  - Lab 4's checks pass.
  - An autouse fixture restores the plugin registries, so a notebook that registers a plugin changes no other test.
- **AI note:** adds "no AI is a complete product" as a guarantee: any assistant is an optional extra. It also adds planned teaching modes (no AI, hint, tutor, instructor), declared by course material in front matter (`ai: hint`), with no AI as the default and the only mode until an assistant exists.
- **Documents:** the README, pitch, guide, review brief and roadmap present the three demos; the roadmap's open question on the main demo is settled.

### Foundations, day 1: concepts, headless views, algebraic notation (11 October 2026)

- **Feature freeze lifted** for a few days of structural work on the model layer. Plan: concepts and boundaries (day 1), one compiled model with checks and parameter status (day 2), measurement maps and a first-class state-space system with template models (day 3), and estimation of linearised models as the test of the whole layer (day 4).
- **`docs/concepts.md`:** the objects and verbs, each with its pymetrix name and status (exists, partial, planned).
  - Principles: one vocabulary; specifications, settings, data and results kept separate; results are values; the code form is canonical; the engine is headless; commands are the only way a session changes.
  - **Decided:** no second names for existing concepts (no `ModelSpec`, no `estimate` beside `fit`).
  - **Decided:** measurement maps are written as equations in the model's own language.
  - **Correction to the plan:** parameters are not plain dictionaries. `Param` already has a symbol, bounds, transform, start and prior. What's missing is a status (estimated, calibrated, fixed) and a `ParameterSet` with values; calibration through `Model(params=...)` is still a dictionary.
- **Views moved to `pymetrix.views`.** The view registry left `pymetrix.ui.render`, which now keeps only the HTML renderer and re-exports the registry, so plugins and old imports keep working.
  - Reports, `pymetrix.testing` and the desktop app import views from the engine.
  - A test runs the engine, views, fingerprints and a report in a fresh interpreter and checks that `pymetrix.ui` is never imported.
- **Algebraic notation (`pymetrix.notation`), as an option.**
  - Methods: `Equation.algebraic()` and `FittedEquation.algebraic(estimates=True)`, in Unicode or LaTeX.
  - The option: `set_notation("algebraic")`, the `notation(...)` context manager, or View → Algebraic notation (saved in the desktop settings). It titles summaries with the equation and labels coefficients β₀, β₁ and so on; in notebooks, equations are typeset.
  - Every fitted equation and every `Equation` gets an Equation view: code, algebraic, estimated and LaTeX forms.
  - The parser accepts Unicode identifiers, so `c ~ α + β*y` works.
  - Rendering: spelled-out Greek names display as letters; parameter digits and underscore suffixes become subscripts, with a `φ_π` fallback where Unicode has no subscript letter; time is a subscript; `d`, `dlog` and `pch` show as Δ, Δlog and %Δ; negative estimates are written with a minus; factors are juxtaposed, with a dot before numbers.
  - `Param.symbol` and variable symbols override the defaults.
  - The notation is display only: scripts and projects keep the code form, and a test checks this.
  - A help topic `algebraic-notation`. A conftest fixture resets the notation for every test, because the desktop setting is read from the real settings store.

### Foundations, day 2: the compiled model, parameter status and checks (11 October 2026)

- **`Model.compile()` returns a `ModelSystem`** (`pymetrix.models.system`):
  - equations as `SystemEquation` residuals, with parameters as symbols and the source `Equation` kept for notation;
  - endogenous, exogenous and shock variables, and a role for every variable;
  - a `ParameterSet`, lead/lag `incidence`, `max_lag` and `max_lead`;
  - `residuals(values)` to substitute other parameter values without rebuilding the model, ready for estimation.
- **Naming:** estimated coefficients are qualified by their equation (`cons.y`), so constants never collide. Calibrated parameters keep their own names, so one parameter (`rho`) can be shared by several equations.
- **Every solution path now uses it.** `Model.__post_init__` compiles, and the internal `model_equations` used by simulation, perfect foresight, steady state, linearisation and the Dynare importer are derived from the compiled residuals. Results are unchanged; a test compares compiled and used residuals numerically.
- **`Parameter` and `ParameterSet`:** a value with a status (estimated, calibrated or fixed), standard error, equation and `Param` metadata (bounds, prior, symbol).
  - `ParameterSet.from_values`, `with_values` and `fix` return new sets; `by_status` and `table` report them.
  - `Model(params=...)` takes a dictionary or a `ParameterSet`; `model.parameters` gives the set.
- **`Model.check(data, span, shocks, steady)` returns a `ModelCheck`,** a table of checks:
  - equations and exogenous variables;
  - targets present in the current period;
  - recursive and simultaneous blocks;
  - parameter values and bounds;
  - timing, with leads noted;
  - shocks exogenous and contemporaneous only;
  - with data, exogenous variables present; with a span, starting values (an error if missing) and exogenous paths (a warning if missing, since a scenario can set them);
  - for models with leads, the steady state and Blanchard-Kahn. A passive Taylor rule is reported as indeterminate.
  - A structural-matching check was considered and dropped: once every equation contains its own target, each block can always be paired with its variables.
- **Desktop:** **Model → Check…** (`CheckModel`, id `pymetrix:check_model`), with views for `ModelSystem` (equations with algebraic notation, variables, parameters, incidence), `ModelCheck` and `ParameterSet`.
- **Tests:** compilation and naming; residual equivalence; parameter sets; shared, bounded calibrated parameters; checks that pass and fail (history, future paths, missing starting values, missing data, a circular equation, Blanchard-Kahn, unknown and lagged shocks); and the dialog through a project round trip with `check_project`.

### Foundations, day 3: measurement maps, the state-space system, template models (11 October 2026)

- **`MeasurementMap`** (`pymetrix.models.measurement`): observed series as equations in model variables and their lags, with the map's own parameters, measurement-error standard deviations, data-series names (`sources`), units and labels.
  - Equations are parsed by the model's own parser and must have the form `observed = expression`. Leads are rejected: an observation depends on the present and the past.
  - `apply(data)` evaluates the equations exactly on model paths, which reports solutions in data units (the reverse mapping).
  - **Parameter metadata** (30 October 2026, after DSGE.jl's parameter objects): `Param` was already the metadata of
    an equation's parameter; models now take it too (`Model.specs`), for any parameter and for shock sizes and
    measurement errors, which are not parameters of the compiled model (`sd_<name>`, kept on the model and read by
    `Model.spec`). A model's spec is merged over its equation's field by field, so an equation's symbol survives a
    model-level prior. Estimation reads priors, bounds (or those a transform implies), starting values and `fixed`
    from the specs; arguments to `estimate` override. Readers (model text, Dynare) fill specs and writers put them
    back, so a model file carries its estimation set-up, as a Dynare file does.
  - **Raw data and reported series** (30 October 2026, after DSGE.jl's observable mappings and pseudo-observables).
    `transforms` say how an observed series is made from raw data; `observed(data)` makes them, and the state-space
    system reads its observations through it, so the filter, likelihood, OccBin filter and estimation all take raw
    data. The reverse is not written by the user: `reverse` solves the transform for the raw series period by period
    (Newton on the expression's derivative, every draw at once), given its past and the other series, which covers
    any transform that moves with the raw series in the same period. `reported` equations are evaluated on model
    paths only: `with_reported` adds them to smoothed histories, `report` to forecasts (with the rebuilt raw series),
    `stochastic.observables` to draws. They live on the measurement map rather than the model because they are
    about reporting in the data's terms, and every model kind (template, Dynare, IRIS, a `Model` with a map) reaches
    data through one. IRIS's `!reporting_equations` are read into it (an earlier one put into a later one that uses
    it, as IRIS evaluates them in order) and written back. A map may observe nothing and only report (an IRIS file
    with reporting equations and no measurement equations): it is accepted, and the state-space form and estimation
    refuse it by name, since they need observed series; the alternative, dropping the reported series when nothing is
    observed, would lose what the file says.
  - `linearize(states, steady)` gives the measurement matrix and constant (Z and d) around the steady state.
  - `direct(names)` gives observation in levels, as before.
  - The table shows each equation in algebraic notation.
- **`StateSpaceSystem`,** a subclass of `SystemMatrices`, so the Kalman filter takes it unchanged.
  - `LinearSolution.system(measurement, shock_sd, measurement_sd)` returns one. A list of variable names still works.
  - The states are the model's variables, plus lagged copies when a measurement equation uses a lag (the transition shifts them).
  - `filter(data, span)` returns a Dataset of the model's variables in levels and `<series>_fitted`; `loglike(data)` gives the likelihood.
  - Views: measurement, measurement matrix, transition, shocks and shock sizes. `LinearSolution.filter` now goes through it.
- **Templates** (`pymetrix.models.templates`): `Template(name, model, shocks, shock_sd, measurement, guess)`, with `check`, `steady_state`, `solve` and `system`.
  - `new_keynesian`: Galí's three-equation model. Measured by annualised inflation, the policy rate and GDP growth with a lagged gap.
  - `rbc`: the stochastic growth model, measured in logs.
  - `gap_model`: a small semi-structural projection model, with the output gap unobserved.
  - Parameters are keyword arguments; shocks have `Role.SHOCK`; `Model.linearize` now defaults its shocks to those roles; `Model.check` accepts a steady-state `guess`.
- **Dialogs:** Model → Template model (`pymetrix:load_template`), Model → State-space form (`pymetrix:state_space_form`), and Model → Kalman filter (`pymetrix:kalman_filter`).
  - `Template.system` and `LinearSolution.system` share the signature (measurement, shock\_sd, measurement\_sd), so the generated code is one line for either source.
  - pymetrix exports `MeasurementMap`, `StateSpaceSystem` and `templates`.
- **Validation:**
  - Z and d agree with a hand calculation.
  - The New Keynesian variables are recovered from data in data units: inflation and the rate exactly, and the output gap up to a constant, because observing GDP growth pins down changes in the gap, not its level.
  - The likelihood equals statsmodels' Kalman filter (rel 1e-9).
  - `apply` equals `log` exactly for RBC.
  - The gap model's unobserved gap correlates above 0.9 with the truth from noisy inflation.
  - Error cases.
  - The dialogs: the generated script runs and reproduces the results, and projects round-trip.
- **Dropped:** a "missing measurement parameter" check could never fire. Only declared names are parameters; others are variables, caught by the "not variables of the model" check.

### Results contract (11 October 2026)

- **Rule adopted** (`docs/concepts.md`, Results): nothing important exists only in display; every number shown is a named field or comes from one. The completeness questions are added to the checklist in `docs/adding-features.md`.
- **Provenance gains three fields:** `data_fingerprint` (a hash of the data used, from `pymetrix._capture.fingerprint`), `warnings` (captured while estimating, then re-issued, excluding library deprecations) and `converged` (None when not iterative or not known). The `seed` is filled from the method's settings.
- **`Equation.fit`** captures warnings for every equation method. The Provenance view shows all the fields, and summaries list warnings.
- **HAC:** the lag count actually used is `stats['hac_lags']`, and the summary note reads it.
- **ARDL:** `stats` (with the centred R-squared), `long_run_coefficients` (a DataFrame with coefficient and std err) and `ecm_params`; the tables are built from them.
- **Provenance on ARDL, Markov switching and MIDAS**, built with `_capture.provenance_for`. Convergence comes from statsmodels' `mle_retvals` for Markov switching and from the least-squares success flag for MIDAS. A test caught ARDL's sample starting before its lags; it now uses the estimation sample.
- **Forecasts:** `ForecastStore.revisions` and `scores`, and `Backtest.scores`, return DataFrames, and the tables are built from them.
- **Views by name:** `pymetrix.views.titles` and `view`, plus `provenance_table`. `Session.producer(name)` already gave the producing step.
- **Not yet with provenance:** disaggregation, quantile process, rolling fits, local projections, IRF/FEVD, linearised solutions, state-space systems, backtests and test results (recorded in known gaps).
- **Internal concordance** (kept outside the repository): brought into the repository, reconciled with the code (actual syntax, honest statuses, decided questions), with a change log, and updated with each piece of work.

### statsmodels as a core dependency (14 October 2026)

- **Decision (the project lead):** statsmodels moves from the `[stats]` extra to the core dependencies.
  - Most of what people come for uses it: unit-root tests, diagnostics, seasonal adjustment, the compiled Kalman filter.
  - Every install route already pointed people at `[all]`.
  - The optional setup cost a CI job and duplicated code paths. The "core only" job had grown to 11 minutes, largely because sampling tests ran through the native filter.
- **What stays:**
  - The native Kalman filter, as the independent cross-check the tests compare statsmodels against. Its `statsmodels`-missing fallback stays too, harmless.
  - The `[stats]` extra, now empty, so that older install commands still work.
  - The `importorskip("statsmodels")` guards in tests, harmless.
- **CI:** the "core only" job becomes "no extras": no Qt, linearmodels or arch, with statsmodels installed.
- **matplotlib too (the same day, the project lead):** a workbench that cannot draw after a plain install looks broken. Graph tabs, `.plot()` and report figures are now always available. The extra cost is roughly 30 to 40 MB (with Pillow, fonttools and a few small compiled packages). `[plot]` is kept, empty. The views' "no Graph tab without matplotlib" check stays, harmless.

### Sequential Monte Carlo (29 October 2026)

- **The algorithm** is Herbst and Schorfheide's (2014), as Dynare 6's: particles from the prior, tempered likelihood
  phi(n), reweighting, systematic resampling when the effective sample size falls below half, and random-walk
  mutation with the particles' weighted covariance, its scale updated by their rule towards an acceptance of 0.25.
- **Tempering** is adaptive: the next phi by bisection so that the conditional effective sample size of the new weights
  (Zhou, Johansen and Aston 2016) is `ess_fraction` (0.5; called `target` until 5 November 2026, still accepted
  with a DeprecationWarning): small steps where the likelihood is sharp, one step where it is not (one parameter of
  a long series may need a single stage).
- **The marginal data density** is the product of the mean incremental weights. Prior draws are kept only inside the
  bounds where the model solves; the log of that share is added, so the estimate integrates over the region the
  harmonic-mean and Laplace estimates integrate over (the same convention, comparable numbers).
- **Processes:** particles move in eight fixed blocks, each with its stream `SeedSequence([seed, stage])`, here or in
  other processes (stages estimated over ten seconds), so the draws do not depend on how many processes ran them.
- **The result** is a `Posterior` with one "chain" of the final particles (`sampler="smc"`), so summaries, densities,
  bands and posterior forecasts work as after Metropolis-Hastings; R-hat is left empty and the effective draws are the
  particles' effective sample size.

### Matrix decompositions and equations (30 October 2026)

- **New dimensions get names, so factors multiply back:** a decomposition names what it creates (s1, ... for
  singular values, q1, ... for QR, l1, ... for LU, u1, ... for Schur vectors, q1/z1 for QZ, λ1, ... for eigenvalues),
  and keeps the matrix's own names where they belong (U's rows, V's rows, null-space rows named by the columns), so
  `U @ s.diag() @ V.T` and `P @ L @ U` meet with no renaming. LU's permutation is named by the rows in pivot order,
  which also name L's rows: P is then a permutation of names, not of numbers.
- **Complex only where the mathematics is:** a matrix may hold complex numbers, but only eigenvalues and vectors of a
  general matrix, a complex Schur form, `logm` and `sqrtm` make them, and those return real matrices whenever every
  imaginary part is negligible (1e-12 relative). `eig` on a symmetric matrix still uses `eigh`, so no earlier result
  changes; what changes is that a general matrix's complex eigenvalues are returned instead of refused.
- **The orthogonal complement is normalised to orthonormal columns** (the left null space from the SVD, MATLAB's
  `null(A')`), not to Johansen's `(0, I)'` forms: any complement is one up to an invertible factor, which cancels in
  the formulas that use it, and orthonormal columns are what the numerics favour.
- **Matrix equations name their solution after the equation's right-hand side** (Q for Lyapunov and Riccati, C for
  Sylvester), and check the names of the matrices that meet it; a VAR's lag matrix has columns named `y[-1]`, so the
  solvers take A's columns to be its rows' variables.
- **NumPy interoperability by `__array_ufunc__`:** element-wise functions keep the names, and an array on the left of
  an operator gives a named Matrix (names checked as for two matrices) instead of silently dropping them. Reductions,
  `out=` and functions with several outputs are left to `m.values`; `sum` and `mean` are methods with an axis, so
  `np.sum(m, axis=0)` returns a named row.

### Matrices in the app (30 October 2026)

- **One step kind, written as its expression.** Every matrix the app makes (Model > Matrix, a matrix window's
  actions, a result's As matrix) records `name = <expression>` and runs exactly that expression through one API call,
  `pymetrix:matrix.expression` (`pymetrix.matrix_expr`), whose code is the expression itself. The dialog, the
  script and the reopened project therefore cannot disagree, and a step reads the objects its expression names, so
  the usual out-of-date and recompute machinery applies. The grid records a `Matrix([[...]], rows=..., cols=...)`
  literal rather than its cells as data, so the script reads as the matrix it makes. The one exception is the design
  matrices, two results from one line (`X, y = Matrix.design(...)`): that step is recorded as its code only.
- **Checked, not trusted.** Typed code in a project needs the user's trust before it runs; matrix expressions do not,
  because they are parsed and restricted before running: one expression of names (the session's objects, `Matrix`,
  `np` and a few harmless built-ins), attribute access, calls, operators, indexing and literals; no attributes
  starting with `_`, no `exec`/`eval`/`open`/`getattr`, no file input or output methods. The checker is a whitelist
  of syntax nodes plus a short list of refused names, not a sandbox: it keeps a project's matrix steps to
  computation, as the dialogs intend.
- **Three entries, not one dialog with tabs.** Model > Matrix has Expression, From data or a result, and Enter
  values: each is an ordinary command, so it reopens from the history, saves with its own fields and has its own
  help; the first two are generated forms, the grid its own window (paste, header renaming).
- **LaTeX:** a plain `bmatrix` (amsmath) for the numbers, or a `tabular` with the names for a matrix whose names
  carry meaning (a covariance matrix of named estimates); `bmatrix` cannot carry row names without a package
  (`blkarray`), so names go in a table.

### Models written as text (30 October 2026)

- **The text is the source of truth.** `ModelSource(text, syntax)` keeps what was typed, in its syntax; `build()` makes
  the model (a `Template` from pymetrix's syntax, a `DynareModel` from Dynare's). The step that builds it,
  `DefineModel`, keeps the text as a field, so a project saves it as written and builds the model from it when it
  opens: no new kind of model object, and comments, layout and descriptions survive. Writing other syntaxes from the
  model (Dynare, IRIS) is for a later phase, as an export with a report of what does not carry across.
- **pymetrix's syntax** is sections of one-line entries, close to how the `Model` and `Template` constructors take
  their pieces; equations are parsed with the expression parser, so they are exactly what `Equation` accepts. An
  equation determines the variable on its left; `~` or `=` is chosen from whether it uses parameters, as `read_mod`
  does.
- **Checking is cheap and positioned.** `check()` reads the text and parses each entry, without solving: the parser's
  positions become line and column. Names are looked up against what is declared and determined, with the nearest
  known name suggested (difflib). One mistake gives one error: an equation whose bracket never closes stops at the
  next entry, and an equation that does not parse still declares its left-hand side and the names it uses, so the
  lines after it are not flagged too. Dynare's text is scanned statement by statement with offsets (comments blanked
  in place), skipping blocks that declare nothing (`shocks`, `initval`, `estimated_params`...); each model equation is
  converted and parsed alone, and the whole file is built by `read_mod` when nothing else is wrong.
- **The editor** (`ui/model_editor_qt.py`) reuses the code editor with a model highlighter, underlines each problem's
  span, and checks 300 ms after typing stops; Apply records `DefineModel` through the main window's `run_step`, and
  reopening a step opens the editor on its text.
- **Regimes and correlations in the text** (30 October 2026). Correlations are lines of `shocks` (`corr a, b = r`),
  read after every shock is declared so the order of lines does not matter, with the same estimation settings as a
  shock's size on `corr_<a>_<b>`; whether they are positive definite together is checked once, and the problem put
  at the first line from which those so far are not. Regimes are a section of one-line entries that map onto
  `Regime`'s fields (start, end, name, `surprise` for `expected=False`, parameter values, equations by name, the
  quoted description). Their order, overlaps and names are checked by `regimes._phases` itself, called on the regimes
  so far, so each problem lands on the regime that causes it and the rules cannot drift from what solving refuses;
  each regime's model is built (`regime_model`) when the rest is clean. They live on the `Template` (`regimes`), not
  on a separate object, because everything that takes regimes starts from it: `solve()` puts them on the
  `LinearSolution` (its `solve(data, span)` default), `estimate` splits them at the sample's end (inside: estimated
  with, as `estimate_model` requires regimes to start in the sample; after: `FittedModel.ahead`, followed by
  `forecast`), and `regime_filter` takes those that start by the sample's end. A parameter only a regime's equation
  uses is given to that regime at its declared value (the base model has no place for an unused parameter), and the
  regimes after it carry it (`regime_model(carried=)`).
- **Completion and hover help** are computed from the text and the last check (`ModelSource.completions`, `hover`),
  not from Qt, so they are tested without a display; the editor only asks them. Comments and descriptions are found
  with one regular expression per syntax over the whole text (Dynare's and IRIS's block comments span lines).
- **Other syntaxes are written from the model, not translated from text** (phase 2, 30 October 2026). `export_model`
  takes what pymetrix built (a `Template`, a `DynareModel`, a `Model` with named parameters), gathers its pieces
  (parameters, shocks and sizes, exogenous values, equations as parsed expressions, measurement rows) and prints them in
  the target syntax: timing as `x(+1)` or `x{+1}`, `d`, `dlog` and `pch` written out with lags, comparisons and
  `step()` refused. So every reader feeds every writer, and a new syntax needs one reader and one writer, not one
  translator per pair. What the text had beyond the model (comments, loops, Dynare's estimation blocks and commands)
  does not carry across, and the `Translation`'s notes say so. Each written text is built again by its own reader and
  compared with the original (impulse responses to every shock over 20 periods, and the steady state): `checked` and
  `difference` record the result, so a writer's mistake shows instead of passing silently.
- **EViews's syntax** (31 October 2026) is read by its own small parser (`models/eviews.py`), not by rewriting text
  into pymetrix's: EViews writes lags and function calls alike with brackets (`x(-1)`, `c(1)`, `log(x)`), so each
  name followed by a bracket is decided by what it is (a coefficient vector, a function, else a series with an
  integer lag), and every node keeps its place, so a refusal points at the function or term itself. The nodes are
  rendered as pymetrix's text (an equation whose left side mixes the variable with others is kept as
  `y = y - ((lhs) - (rhs))`, which has the same residual), so the model built is an ordinary `Template` and every writer
  takes it. Names are lower-cased (EViews ignores case), coefficients become parameters `c_1`, `beta_1` (the FRB/US
  package's own Python names), `@innov` the template's shocks. What is ambiguous between languages (`-a^b`, `a^b^c`) or
  depends on the workfile (`@pchy`, `@trend`) is refused rather than guessed. A text with `model`, `equation` or
  `.append` is a program, in which bare `x = ...` lines are series assignments; otherwise it is model text, in which
  they are equations. The FRB/US XML reader and the state-space reader reuse the same parser. Functions of the
  workfile's dates (`@trend(base)`, `@seas`, `@elem(x, date)`) become exogenous series with names that say what they
  are (`trend_1960_1`, `x_at_2004q4`), made from a dataset's dates by `eviews_series`, since a pymetrix expression has
  no calendar; `@pchy` and the like take the frequency from the text, the data or `frequency=`. The subset was
  confirmed (and widened: comparisons, general `@recode`, the dated functions) against Brillet's book of EViews models
  and the FRB/US packages.
- **IRIS's syntax** is checked block by block (`!transition_variables`, `!parameters`...), with declarations and their
  descriptions, equations split at `;` and parsed alone (steady-state parts after `!!` dropped; `&x`, `$...$` and
  `?` skipped); the whole file is then built by `read_iris`. In `!reporting_equations` the name on the left is new
  (a reported series, its description from the quoted text before it), the right is checked like any equation.
  Parameter descriptions become `Param(name, label=...)` specs on the model (only parameters the compiled model has),
  and `std_<shock>` descriptions label `sd_<shock>`, so the descriptions reach the parameter table and estimation
  summaries as Dynare's `long_name` does.
- **The editor's sidecars beyond the text** (phase 3, 30 October 2026). The Equations tab prints the built model's
  equations through `pymetrix.notation` (the same notation as fitted equations): it is filled only when the tab is
  shown, since building a large Dynare model on every keystroke would be slow; a click goes to the line that
  determines the variable. Check solution is a button, not a live check, for the same reason: it finds the steady
  state, linearises there and reports the roots with the Blanchard-Kahn verdict. The roots come from
  `LinearModel.roots()`, which shares the solver's pencil (`_pencil`, split out of `_solve`) but does not reorder it,
  so they are there to see when no unique stable solution exists, which is when they are most wanted.
- **The impulse-response preview** (30 October 2026). `check_solution` keeps the `LinearSolution` it found when the
  verdict is unique (and the model's shock sizes, correlations and number of regimes), so `SolutionCheck.irf` is
  exactly `LinearSolution.irf(periods, shock_sd, shock_corr)`: the preview cannot drift from what the solution gives.
  The figure is a plain matplotlib canvas in the Solution tab, redrawn when the shock or the ticked variables change,
  cleared when the text has errors or the solution is not unique. With regimes the solution is the base model's (the
  regimes are dated paths, not alternative solutions to preview), and the note says so.
- **Matrix expressions checked as typed** (30 October 2026). The dialog runs `matrix_expr.preview` 300 ms after
  typing stops (`commands.matrices.live_check`, adding the Name's problem): `check_expression`'s whitelist first,
  then evaluation only when it is cheap, judged before running (the session objects used hold at most 40,000
  numbers, no number written above 1,000, no arithmetic on numbers alone such as `2**40`, no tuple of sizes above a
  million elements, `kron`/`tile`/`repeat`/`power` only on small inputs) and bounded after (a worker thread given
  0.5 s; a slower one is reported as not evaluated). The checker's whitelist already keeps evaluation side-effect
  free; the cost guards keep a half-typed `np.ones((10**6, 10**6))` from freezing the window. An evaluation error
  (shapes that do not conform, a number rather than a matrix) disables OK as a check error does, since the step would
  fail the same way. The hook is generic (`Command.live_expression` names the field), so other expression dialogs
  can opt in.

### Mode jumps in Metropolis-Hastings (13 October 2026)

- **Diagnosis.** The long Smets-Wouters run (4 x 160,000 kept) had R-hat 1.13 and as few as 31 effective draws.
  - Its posterior has two modes of almost equal height: -1484.28 (crhob 0.14, chabb 0.81, sd_eb 0.36) and -1484.82 (crhob 0.82, chabb 0.46, sd_eb 0.15). L-BFGS-B from a draw near the second first stopped at -1494.5; a Nelder-Mead polish found its height.
  - The chains crossed between them 3 to 30 times in 160,000 draws.
  - Adaptation was not the cause: without it the chains crossed less (11 against 23 times in 4 x 40,000).
  - Dynare 6.0 (2 x 200,000) sees the same two modes: crhob 90% HPD 0.078-0.816.
- **The move.** With probability `jump` (0.05) a step proposes `x + (m_j - m_i)` for a random ordered pair of modes. The reverse shift has the same probability, so the proposal is symmetric and the usual acceptance gives the exact posterior. Jump steps never enter the random walk's scale tuning or its acceptance rate.
  - On Smets-Wouters about 2% of proposed jumps were accepted, against one crossing in about 10,000 random-walk steps.
- **Which modes.** By default, the ends of the estimate's starts within 5 log points of the best, highest first, skipping any within three proposal standard deviations of one already kept (the estimate's mode first). A mode more than 5 lower holds well under 1% of the posterior unless it is far wider, and jumps to it are mostly refused. `modes=` gives others (by name or as arrays, checked); `modes=False` none. With one mode nothing changes: no extra random numbers are drawn, so the draws are identical to before.
- **Starting points and adapting.** Chain j starts around mode j modulo the number of modes. While adapting, burn-in draws are taken less the jumps made, so the adapted proposal keeps the shape of one mode rather than spanning both.
- **Reporting.** `Posterior.modes`, `jumps` (proposed and accepted per chain, kept draws) and `jump`; a summary note; provenance `modes`, `jump` and `jumps accepted`.
- **Not in the dialog.** Jumps follow from the estimate's starts, so **Model → Metropolis-Hastings…** needs no new field.
- **Checks** (tests/test\_mode\_jumps.py):
  - A three-dimensional two-component mixture (weights 0.7 and 0.3; second component the same width, and 0.4 times as wide): the share in each mode and the means within 4 Monte Carlo standard errors. Every chain jumps more than 50 times; without jumps a chain never leaves the first mode.
  - `x = rho^2 x[-1] + e` with a prior centred at 0.2: modes near plus and minus rho with unequal weight, the share of draws below zero against the exact posterior on a grid.
  - Starts find both modes, and the sampler uses them by default; without jumps each chain keeps to one mode; with one mode the draws are unchanged; options are checked.
- **Smets-Wouters, measured:** 4 x (20,000 + 40,000) with the two modes: about 40 jumps accepted per chain, R-hat 1.03 (1.43 without), minimum effective draws 116. Means within 0.25 of Dynare's posterior standard deviations (median 0.09), modified harmonic mean -1571.14 against Dynare's -1571.42. 14 minutes on the laptop. The share in the second mode still varies between chains (0.15 to 0.37), so long runs remain advisable.
- **Do the starts find the modes? (Smets-Wouters, `sw.estimate(starts=8, seed=1)`, 34 minutes.)**
  - Two starts reached the first mode (-1484.279) and three the second (-1484.82 to -1484.90). Two stopped at -1490.56, and one failed far away (-4716).
  - The two at -1490.56 arrived from different places, and restarting BFGS from them climbs no further: it is a third, minor local mode, not an optimiser stall. So no polishing step was added.
  - With the first rule (within 20, in start order) the sampler would have jumped among three modes, with the second represented by its lowest start. Hence the 5-point gap and the highest-first order: it now picks exactly -1484.279 and -1484.824.
- **Slow test and what it showed** (test\_dynare\_sw2007, PYMETRIX\_SLOW, 42 minutes): eight starts found both modes, and the means and crhob interval matched Dynare. But R-hat was 1.22.
  - The worst parameter was crhopinf, not crhob. Raising `jump` to 0.2 quadrupled the jumps (95 to 177 per chain) but left R-hat at 1.21, so mode crossing is no longer the limit.
  - One chain spent 17% of its kept draws in a tail where crhopinf is about 0.87, ctrend 0.37 and cmap 0.70 (log posterior about 3 below typical draws). The others spent 0.1 to 0.5% there. crhopinf correlates 0.75 with ctrend and 0.56 with cmap. The tail is not the minor mode at -1490.56 (cmap 0.99 there).
  - The test therefore checks R-hat only for gross failure (below 1.3). The tail's weight is open: pymetrix's 5% point for crhopinf is 0.909, while Dynare's HPD starts at 0.955. A 4 x 400,000 run was started to settle it.
- **The long run** (4 x (100,000 + 400,000), jumps, 134 minutes):
  - Max R-hat was 1.06; crhopinf had 67 effective draws, ctrend 112.
  - Every chain spent 9 to 22% of its draws below crhopinf 0.955, where Dynare's HPD implies about 5%. Mean 0.967 against Dynare's 0.976; modified harmonic mean -1570.19 (against -1571.42).
  - The tail is real in pymetrix's posterior, and weighted more than in Dynare's run.
- **Settled (14 October 2026):** a new dynare-reference case (sw2007_points) has Dynare 6.0 evaluate its log posterior (`evaluate_posterior_kernel`) at sixteen points from tests/fixtures/sw2007-points.json. These are the published mode, Dynare's mode, ten draws in the crhopinf tail (to 0.745) and two from each mode's region. pymetrix equals Dynare to 1.7e-8 at every one. Same posterior, so the difference is Monte Carlo: Dynare's two random-walk chains (jscale 0.2) under-visit the tail. tests/test_dynare_sw2007.py checks the sixteen points (1e-6), using the downloaded model files only.
- **Fixture:** tests/fixtures/dynare-sw2007.json now holds Dynare's 2 x 200,000 run (workflow run 38369423293) in place of 2 x 20,000.

### DSGE-VAR (1 November 2026)

- **What.** Del Negro and Schorfheide's (2004) DSGE-VAR(lambda), as Dynare's `dsge_var`: a VAR(p) with a constant whose
  normal-inverse-Wishart prior is the model's population moments weighted as lambda T observations, and the model's
  parameters estimated by the VAR's marginal likelihood with the coefficients integrated out (DS 2004, eq. 21).
- **Where.** A module of its own, `models/dsge_var.py`; `Model.dsge_var`, `Template.dsge_var` and
  `DynareModel.dsge_var` call it. `_DSGEVARLikelihood` subclasses estimation's `_Likelihood` and replaces only
  `loglike`, so building the model at trial values, priors, bounds, transforms, the optimiser (`_optimise`), the
  Hessian and `bayes.mcmc` are shared unchanged with `Model.estimate`: no edits to `estimation.py` or `bayes.py`.
- **Moments.** From the state-space form (`LinearSolution.system`): the state's variance by the Lyapunov equation,
  autocovariances `Z T^h S Z'`, measurement errors at lag 0 only, and the means `d` for the constant's rows (uncentred
  moments, as DS 2004). A unit root is an error: no population moments.
- **Result.** `FittedDSGEVAR` holds a `FittedModel` at the mode (its `log_posterior` the DSGE-VAR's, so the existing
  sampler and Laplace approximation work on it) and, after `mcmc`, a `Posterior`; `mcmc` returns a new
  `FittedDSGEVAR`, so responses, forecasts and VAR draws integrate over theta once it has draws. Forecasts reuse
  `FittedBVAR.predictive_draws` on a `FittedBVAR` built from the draws; responses and variance decompositions reuse
  `IRF` and `FEVD`.
- **Lambda.** Fixed; `"grid"`, every grid value estimated (warm-started from the previous mode) and the best Laplace
  marginal data density kept, the grid's table and the hump figure on the result; or estimated with a prior as the
  parameter `lambda`, bounded below by `(k + n) / T`.
- **Rotation.** QR of the transposed impact `A0' = Q R` with R's diagonal made positive: `A0 = R' Q'`, `Omega = Q'`,
  and the VAR's impact `chol(Sigma) Omega` for each draw.
- **Checks:** tests/test\_dsge\_var.py (see validation.md).

### Estimating linearised models (day 4, 11 October 2026)

- **API:** `Model.estimate(data, measurement, shocks=None, shock_sd=None, estimate=None, priors=None, bounds=None, span=None, guess=None)` and `Template.estimate(data, estimate=None, priors=None, span=None)` (the template's own measurement, shocks, shock sizes and steady-state guess). Dialog: **Model → Estimate model…** (`EstimateModel`, id `pymetrix:estimate_model`); priors are given in code.
- **Built from days 1–3 without new machinery.** Parameter status says what is free: by default, calibrated parameters plus `sd_<shock>`. Measurement-error sizes `sd_<series>` are free only if named. Fixed parameters and coefficients fitted equation by equation are held. At each trial value: the `Model` at the trial values (a `with_parameters` copy sharing its compiled structure; until 13 October 2026, rebuilt), then steady state, linearise, solve (Blanchard-Kahn), `solution.system(measurement, shock_sd)` and the Kalman log likelihood. Failures are penalised, not raised.
- **Estimator:** L-BFGS-B within `Param` bounds (standard deviations positive), with a Nelder-Mead polish if it stops early. `converged` is checked with `_at_optimum`. With any prior (`Param.prior` or `priors=`), the objective adds the log prior and the result is the posterior mode. Standard errors come from the inverse numerical Hessian over interior parameters.
- **Result: `FittedModel`** with `params` (PointEstimate with covariance), `parameters` (ParameterSet with status `estimated` and standard errors), `shock_sd`, `measurement`, `solution`, `system`, `smoothed`, `stats` (loglike, logpost, AIC, BIC, nobs), `priors`, `bounds`, `method` and `provenance` (sample, fingerprint, estimated names, evaluations, warnings, converged). Views: Summary, Parameters, Shock sizes, Smoothed variables, Measurement, Decision rules, Provenance.
- **Independent checks:**
  - An AR(1) written as a model matches statsmodels SARIMAX: estimates to 1e-4, log likelihood to 1e-7.
  - Parameters of a New Keynesian model simulated with the template are recovered within three standard errors.
  - A tight prior moves the estimate to the prior's mean.
- **Learned:** in the New Keynesian template, the Phillips-curve slope and the Taylor-rule inflation response are weakly identified even with 400 observations. Documented in the guide.
- **Not built:** MCMC, identification diagnostics, and estimating steady-state levels jointly with the measurement map's parameters.

### Report and chart options (11 October 2026)

- **Rule kept:** reports and charts are views over structured results. Templates choose views by title, views are built from named fields, and nothing reads printed text. The spec's separate ReportSpec, ChartSpec and TableSpec objects were not added: `Report`, `Figure` and `Table` already are those specs, and their options are recorded by the Export report command for replay.
- **Themes** (`pymetrix.themes`): a frozen `Theme` with palette, line styles, line width, font sizes and family, grid, band shading and table digits. Five built-in themes: default, publication, teaching, policy, monochrome. `set_theme` sets the session default and `get_theme(...).with_options(...)` customises one. Colours that results give their layers (the standard palette's n-th colour) map to the theme's n-th, so fan charts follow the theme without changes to results.
- **Chart options:** `Figure.with_options(theme, span, subtitle, footnote, title, columns, width, height)`. `span` clips what is drawn and `data()`. Unknown options and themes fail when set, not when drawn.
- **Reports:** `Report(theme=, span=)`. Figures inherit them unless they set their own, and tables use the theme's digits in Word, HTML, Markdown and Excel number formats. Markdown export writes figures as PNGs in `<name>_files`.
- **Templates:** `Report.from_template(name, {heading: result}, title=, theme=, span=)`, one of three named `ReportTemplate`s:
  - `estimation_summary`: Summary, Graph, Diagnostics, Long run, Bounds test.
  - `forecast_round`: Graph, Paths, Judgement, Accuracy, Summary.
  - `technical_appendix`: all.

  Each template has a version. A result with none of a template's views is an error that names what it has. `Report.add(sections=[...])` now keeps the order asked for.
- **Dialog:** Export report gains template, theme and dates in figures; `.md` is accepted. Its code is `Report.from_template(...)` when a template is chosen. Tests check that the project round-trips and that running the generated script writes an identical Markdown file.
- **Not built:** PDF, PowerPoint, a template editor, user template files, and branding packs.

### Next steps (11 October 2026)

The foundations week is complete: one vocabulary, the compiled model, parameter status, measurement maps, the state-space system, templates, model estimation, the results contract and report options. The internal concordance holds the same list (Next Steps), with status rows marked as scheduled. In order:

1. **Fold the new work into the demos, before the first collaborator sessions.**
   - Lab 3's report from the `forecast_round` template in the `policy` theme.
   - Estimating the New Keynesian template in Lab 4, showing its weak identification honestly.
   - A forecast round that carries its model's provenance into reports.
   - A pass to check that the three demos, the pitch and the review brief match what is built.
2. **Model layer, from estimates to the story.**
   - Shock decompositions of a `FittedModel`; the smoother already gives the shocks. Also the VAR historical decomposition (done: `FittedVAR.historical_decomposition`).
   - Forecasts from a fitted model, unconditional and conditional on observed paths, through the Kalman filter.
   - Identification checks: Hessian conditioning, and the rank of the Jacobian of model moments.
   - Several starting values.
3. **Posterior sampling.**
   - Random-walk Metropolis-Hastings from the mode, with acceptance and convergence diagnostics.
   - Validated against a conjugate case and against Dynare on a template.
   - Priors in the Estimate model dialog.
4. **Finish the results contract.**
   - Provenance on the remaining types: temporal disaggregation, quantile processes, rolling fits, local projections, impulse responses and decompositions, linearised solutions, state-space systems, backtests and test results.
   - `converged` for the other iterative estimators.
   - A common `PyMetrixError` base and `data.validate()`.
5. **After collaborator feedback**, in the order reviewers ask:
   - IRIS model file import.
   - More report templates (`policy_brief`, `teaching_note`, scenario comparison charts).
   - PDF export.
   - Second-order perturbation.

**CI note.** The oldest-versions job crashed once in the Qt window clean-up after all tests passed. Skipping the clean-up was tried: locally it brings back the crash in Python's final clean-up that it prevents, and a hard exit crashes on Windows. Decision: leave it and re-run the job if it recurs.

### Readability pass on result tables (14 October 2026)

- **Found by screenshots of every result tab at 1600 x 950 with Segoe UI 9 pt** (.tools/ui_audit.py, local). The off-screen renderer's own fallback font made the first set misleading.
- **One grid builder (`_fill_grid` in ui/qt.py), so every result window changes at once:**
  - Columns that are all numbers are right-aligned, headers too. A number among text keeps the text's side.
  - Each number is kept on its cell (UserRole), so Copy selected cells gives the full-precision value, as Copy and Export already did. Before, it copied the four decimals shown.
  - Columns named like p-values (p>|t|, p-value, prob) show values below 0.0001 as "<0.0001"; Markdown, HTML and reports are unchanged.
  - Rows are sized to their contents.
  - The vertical header is hidden when the table has no index and its first column is text.
  - Columns wider than 520 px wrap.
  - Tables are capped at the height of their rows, measured on the Show event (sizes are final only once laid out), with a stretch of factor 0 after the notes. A long table still takes the whole window.
- **Elsewhere:**
  - `output.bounds_text` writes bounds with ∞ (1e8 or more counts as infinite) for `ParameterSet.table` and `FittedModel.summary`.
  - `views._readable` turns provenance values into text: samples, method dictionaries (lists of dictionaries counted, "see Fields"), datetimes to the minute, booleans as yes and no. Unknown values are left out.
  - The Immediate window drops IPython's `banner1`/`banner2`.
  - The status bar names the frequency and elides at 60 characters.
  - The Objects pane sizes its Type column.
  - The dataset overview uses "N (parts)".
- **Checks:** test_ui_demo_polish (alignment, `<0.0001` shown and the number copied, readable provenance, hidden row numbers, height cap, `bounds_text`); the full suite and the demo-path test.

### Cointegrating regressions and heterogeneous panels (26 October 2026)

- **Native, but to arch's conventions:** FMOLS, CCR and DOLS work without arch, and follow its kernel, bandwidth and
  trend conventions so that they can be checked against it to rounding.
- **Gregory-Hansen's critical values simulated** for the sample's size and regressors, as with Perron's test; the
  published asymptotic table is shown beside them.
- **The pooled mean group by concentrated likelihood:** for a given long run each unit is least squares, so only the
  common long run is searched.

### Synthetic DiD and diffusions (26 October 2026)

- **Block designs only:** synthetic DiD here needs every treated unit to adopt at once (staggered adoption has
  `staggered_did`); its weights are quadratic programmes on the simplex, solved by SLSQP and checked against an
  independent Frank-Wolfe solution.
- **Diffusions by their exact densities,** so no discretisation bias at any dt; models without one go through
  estimation by simulation.

### Threshold VARs, bilinear, VARMA, BHHH, DECO and VECH (26 October 2026)

- **The threshold test by bootstrap:** the threshold is unidentified under linearity, so Tsay's sup-LR gets its
  p-value from the linear VAR resampled, the threshold searched again in every draw.
- **VARMA by conditional likelihood,** checked against statsmodels' exact-likelihood VARMAX rather than wrapping it,
  so q = 0 is exactly least squares and the residual recursion can be read.
- **BHHH as an option of ML,** not a separate method: the same likelihood and line search, the outer product of the
  scores in place of the Hessian.
- **DECO and diagonal VECH join the finance plugin's multivariate GARCH,** sharing DCC's first stage and BEKK's
  variance-targeted recursion; DECO's closed-form determinant and inverse make it usable with many series.

### Unit roots, regression details, nonparametrics, counts and durations (26 October 2026)

- **Our own GLS detrending,** verified through arch's DF-GLS statistic rather than trusted, so the M and PT tests that
  arch lacks rest on a checked base; Perron's p-values are simulated for the sample's own T and break date.
- **Regression variants as methods:** Cochrane-Orcutt, Prais-Winsten, Harvey's model and the frontier return
  `FittedEquation`s, so tables, tests and forecasts work on them unchanged.
- **Kernel tools checked to rounding against statsmodels,** with our own cross-validation; the semiparametric GARCH
  reports no Hessian standard errors, because a Monte Carlo showed them about five times too small; a bootstrap is
  offered instead.
- **Counts and durations share one result class** (`SeriesModel`): parameters, covariance, likelihood and the
  conditional mean.

### Quantile VARs, spatial models and the full VECH (27 October 2026)

- **Quantile regression as a linear programme** (scipy's HiGHS), so the quantile VAR needs no statsmodels and
  statsmodels' QuantReg is an independent check.
- **Spatial models with spreg's conventions:** the eigenvalue log-determinant and Anselin's analytic information
  matrix, so estimates and standard errors match spreg; spreg itself is a test-only reference.
- **The full VECH for small systems only:** 2 m^2 parameters, started from the diagonal VECH it nests, so its
  likelihood can only improve on it.

### Spectra and long memory (27 October 2026)

- **One scaling throughout:** the two-sided density in radians, so the periodogram integrates to the variance; scipy's
  one-sided per-cycle density is a factor 4 pi away, which the test checks.
- **d as a test result:** GPH and the local Whittle return a `TestResult` with d as the statistic (testing d = 0), its
  standard error in the details, like the other single-number diagnostics.

### BVAR-SV, FAVAR and MS-VAR (27 October 2026)

- **BVAR-SV shares the TVP-VAR's samplers** (FFBS and the mixture) and draws the coefficients jointly by GLS, which is
  fine for the small and medium VARs the app expects; the step is checked as SUR least squares in its limit.
- **FAVAR is a VAR plus loadings:** the VAR is an ordinary `FittedVAR`, so every identification and tool applies;
  panel series' responses are its responses through the loadings. Bands bootstrap the whole two-step procedure
  (Kilian 1998, as Bernanke, Boivin and Eliasz): simulate the VAR, rebuild the panel from the loadings and resampled
  idiosyncratic rows, estimate again. Principal components come from the eigenvectors of the smaller cross-product,
  which in this container is seven times faster than the SVD and equal to 1e-13.
- **MS-VAR by EM, then the exact likelihood:** EM's closed-form transition update ignores the ergodic start, so a
  final polish maximises the exact likelihood, which is what makes it equal statsmodels' MLE.

### More SVAR identification (27 October 2026)

- **Max share as another scheme:** an eigenproblem in the Cholesky basis, so bootstrap bands, FEVDs and historical
  decompositions work unchanged; the shocks it leaves unidentified are named so.
- **Narratives as a filter on sign restrictions,** with Antolín-Díaz and Rubio-Ramírez's importance weights applied by
  resampling, so every downstream summary (median target, bands) is unchanged; they need the residuals, so the
  bootstrap is not used.
- **Heteroskedasticity as its own result,** since it needs the residuals by regime rather than one covariance.

### Zero and sign restrictions; bands on historical decompositions (1 November 2026)

- **Zeros as fields of `SignRestrictions`,** not a new scheme: the same argument style (`zeros`, `long_run_zeros`
  beside `signs`), so the median, the median target, bands, FEVDs, historical decompositions and narratives work
  unchanged. Without zeros nothing changes (the column search stays); with zeros each shock keeps its column, as in
  ARW, since a zero ties a shock to its column.
- **ARW's weights by numerical differentiation, in few coordinates:** A+ enters the zeros and the draw's coordinates
  only through the lag sum, so its other directions factor out of the volume element analytically, leaving a Jacobian
  in A0 (and the lag sum, with long-run zeros); the null-space bases are made smooth by projecting a fixed basis, so
  finite differences are well defined. Weighted draws are resampled, keeping every downstream summary a plain list.
- **Bands on decompositions as quantiles over draws,** stored beside the point decomposition (which stays the
  default and is unchanged), so existing figures and tables are untouched and the bands are an addition.

### Historical decompositions of panel VARs and proxy SVARs (1 November 2026)

- **The VAR's `Decomposition`, reused:** a proxy SVAR's is one `Decomposition` with two parts (its shock, "other
  shocks"); a panel VAR's is a `PanelDecomposition`, a mapping of entities to `Decomposition`s with the entity its
  views show, so every figure, table and band view is the VAR's. Selecting an entity returns a copy (results stay
  frozen); the window shows the entities' table first, then the selected entity's views.
- **Fixed effects recovered after estimation,** as the entities' means of y less the lags' effect: the within
  estimator's constants, and for GMM on forward orthogonal deviations the effect the transform removed; then each
  entity is decomposed as a VAR with that constant, so the parts add up to its data exactly. The panel keeps its rows
  in levels for this (a few columns, not the dataset).
- **The proxy's shock as a projection,** b' Sigma^-1 u / (b' Sigma^-1 b): the first row of B^-1 for any full B with
  B B' = Sigma containing b, so the other shocks need not be identified; their sum is u - b e.
- **The proxy bootstrap as one generator** (`proxy.bootstrap_draws`) used by the responses' bands and the
  decomposition's, so both use the same draws for a seed; the responses' bands are unchanged by the refactor.

### Applied microeconometrics (1 November 2026)

- **Functions, not `Equation.fit` methods:** double machine learning, shift-share IV, interactive fixed effects and the
  conditional logits each name roles a formula cannot (treatment versus controls, shares and shocks, groups, cases and
  alternatives), as `treatment_effect` and `staggered_did` do, so they are functions returning results of their own
  (`FittedDoubleML`, `FittedShiftShare`, `FittedInteractiveFE`, `FittedConditionalLogit`) with the generic verbs,
  provenance and views. They live in new modules, with dialogs in `commands/micro.py`.
- **Double ML's learners by name:** `"ols"` (least squares, and a Newton logit for a propensity) needs nothing, so the
  method runs on a core install; `"lasso"`, `"forest"`, `"boosting"` or pymetrix's `Lasso()`, `RandomForest()`,
  `GradientBoosting()` (their settings reused) build scikit-learn models. Folds are drawn with `pymetrix._random` and
  kept on the result (`splits`), so another implementation can be run on the same folds (the DoubleML check).
  The estimator is DML2 (one score over all folds); repeated splits use the paper's median rule,
  sqrt(median(se_s^2 + (theta_s - theta)^2)).
- **Shift-share as one regression with every variance:** the coefficient is 2SLS on the built instrument; robust,
  AKM and BHJ variances share the residuals and the share-weighted sums R_k, so they are reported side by side;
  AKM0 is inverted in closed form (a quadratic), and can be unbounded (with very few shocks it must be).
- **Interactive fixed effects estimate the constant jointly:** demeaning by the grand mean would add a rank-one term
  to the factor structure, so with `effects="none"` the constant is a regressor; unit, time and two-way effects are
  removed by the balanced within transform, which keeps the rank. Two starts (least squares; principal components
  of y) guard against a poor local minimum. Eigenvectors come from SciPy, not NumPy, whose threaded `eigh` was slow
  on small matrices on a busy machine.
- **One conditional likelihood:** Howard's recursion gives the fixed-effects logit's denominator with exact first
  and second derivatives (scaled around the k largest indexes so that long groups neither overflow nor underflow);
  one chosen alternative per case is the softmax, so McFadden's model is the same code with cases as groups.

### Marginal effects after the conditional logits (1 November 2026)

- **A result of its own, `ConditionalLogitEffects`,** not the `Table` that `marginal_effects` gives for a logit:
  the effects of an attribute are a matrix (alternatives × alternatives), so the result keeps one long frame
  (variable, measure, probability, change in) and gives `.table(variable, measure)` and a view per variable. The fit
  keeps the rows it used (`design`) so that the effects need no data.
- **Averages over every case, zero where unavailable:** with varying choice sets an alternative that is not offered
  has probability zero, so its derivatives are zero; averaging them over all cases keeps each column of derivatives
  summing to zero. Elasticities divide by the probability, so they are averaged over the cases where both
  alternatives were offered. At the means, every alternative is offered, each attribute at its mean over the cases
  that had that alternative.
- **The panel logit gets odds ratios, not effects:** its probabilities depend on the unidentified fixed effects.
  Effects at a value of the fixed effect the user gives (Stata's `pu0` at 0) are offered because people ask for
  them, labelled with that value in the row and notes, never by default.
- **Delta method by central differences** of the effects in the coefficients (the derivatives themselves are in
  closed form), as `marginal_effects` does; checked against a parametric bootstrap and statsmodels' MNLogit.

### Discrete changes after the conditional logits (1 November 2026)

- **Beside the derivatives, not instead of them:** the frame gains rows with measure "discrete change" and the
  result a `discrete` map (variable: "0 to 1" or "x to x + d"), so tables of derivatives and elasticities stay as
  they were; only the summary shows the discrete change in place of a 0/1 variable's derivative, as statsmodels'
  `dummy=True` does.
- **One alternative's attribute at a time,** as the derivatives are taken, so an attribute's changes are a matrix
  (own on the diagonal, cross off it) whose columns sum to zero; a change of an unavailable alternative's attribute
  is zero, averaged in as zero, the same convention as the derivatives.
- **0/1 detection as statsmodels':** every (available) value 0 or 1 and both present; any other change, including
  a count's, is the user's (`changes={name: amount}`, from x to x + amount, not statsmodels' centred
  (P(x + 1) - P(x - 1)) / 2).
- **Exact recomputation, delta method by central differences** of the changes in the coefficients, as for the
  derivatives.

### Decompositions for BVAR-SV, MS-VAR and quantile VARs (1 November 2026)

- **One time-varying recursion** (`var_decompositions.tv_contributions`) for all of them: intercepts, lag matrices
  and impact in effect each period, vectorised over draws; linear, so the parts add up to the data exactly for any
  path of parameters, and with constant parameters it is the VAR's decomposition.
- **BVAR-SV: posterior means as the point,** with quantile bands: the mean is linear, so mean shares add up to one and
  mean contributions plus the mean baseline add up to the data (medians would not). The FEVD takes one date's
  volatilities, as `irf` does, rather than their expected drift over the horizon.
- **MS-VAR: the most likely regime path** (argmax of the smoothed probabilities) rather than probability-weighted
  parameters: a path is a history the model can produce, and its intercepts go in the baseline; weighted parameters
  also add up but describe no regime. FEVDs hold a regime fixed, like the responses.
- **Quantile VAR: FEVD as quantile connectedness** (Ando, Greenwood-Nimmo and Shin 2022), with the residuals'
  covariance orthogonalised by Cholesky so the shares add up and coincide with `irf`'s shocks when the structural
  quantile residuals are uncorrelated; no historical decomposition (uncentred residuals; no literature version).
- **`FEVD` gains optional bands** beside the point values, as `Decomposition` did, so existing tables and heatmaps
  are unchanged.

### Bands for Markov-switching VARs (1 November 2026)

- **Parametric draws from the asymptotic normal of the ML estimates, not a bootstrap.** A bootstrap would rerun EM
  from several starts each replication (seconds each) and its regime labels could switch between replications; the
  observed information is one batched numerical Hessian (the Hamilton filter vectorised over all perturbed parameter
  vectors at once: 0.03 to 0.2 s for two variables), and the draws are then free. Krinsky-Robb draws rather than the
  delta method, so shares stay in [0, 1] and add up to one in every draw.
- **Draws on an unrestricted scale:** covariance matrices by their Cholesky factor with log diagonal, transition rows
  by logits against their last entry, so every draw is a valid model. A row with a probability below 1e-6 is on the
  boundary, where the likelihood is flat in its logits and asymptotic normality fails: it is held at its estimates
  (and said so in the band source) rather than refusing bands. A flat likelihood elsewhere (identical regimes) is
  refused.
- **The historical decomposition's bands hold the most likely regime path fixed** (parameter uncertainty only), as
  the point decomposition conditions on it; drawing paths from the smoothed probabilities as well would mix in the
  regime uncertainty the point does not show.
- **Point values stay at the estimates** (not the mean over draws), as for the VAR's bootstrap bands.

### Markov-switching VAR starting values (1 November 2026)

- **Starts from the data, not only random splits.** A random soft split gives every regime nearly the same weights
  in every period, so the first M-step makes the regimes nearly identical and EM, a local method, keeps them so; the
  earlier fit's random starts stopped at the linear VAR's likelihood. Hard-ish splits (0.9 / 0.1) by quantiles,
  rolling means and variances, k-means and one-variable fits put the regimes apart from the first step. Each is
  screened by 30 EM iterations (EM's early gains decide most comparisons) and only the best three distinct ones run
  to convergence, so more starts cost little.
- **Collapse is detected, not prevented:** regimes within 1e-4 of each other (scaled), or a regime with under half a
  period of probability, mark a start as collapsed; it is replaced by the best start's probabilities mixed with a
  random split. No covariance floor: it would change the maximum the fit reports.
- **The filter as prefix products.** alpha_t = alpha_0 G_1 ... G_t with G_t = P diag(d_t): a Hillis-Steele scan of
  rescaled positive matrices, log2(T) batched matrix products, no cancellation, equal to the recursion to 1e-13; the
  smoother is the same scan on the reversed, transposed sequence. Four times faster than the loop for one parameter
  set; batched over parameter sets, the likelihood alone is a tree reduction (the polish's gradient and the bands'
  Hessian).
- **BFGS on the exact likelihood replaces the alternation** of Nelder-Mead over P with EM steps for the
  coefficients: one joint optimisation on the bands' unrestricted scale, gradient by central differences in one
  batch. EM now stops at a relative change of 1e-8 rather than 1e-12 (thousands of slow linear-rate iterations), and
  the polish finishes; transition rows at the boundary (below 1e-8) stay there.

### Structural identification (26 October 2026)

- **A-B as one more identification scheme:** `AB` sits beside Cholesky, LongRun and sign restrictions, so responses,
  decompositions and bootstrap bands work unchanged; `structural()` adds what only ML gives (standard errors, the LR
  test). Long-run zeros are linear in B with A the identity, so they are imposed exactly through a null space rather
  than by penalty.
- **FIML through the system's own expressions:** the Jacobian in the current endogenous variables is differentiated
  symbolically, so identities and equations nonlinear in variables need nothing special; 3SLS gives the start.
- **VECM tests by our own reduced-rank regression,** checked against statsmodels' estimates, since statsmodels has no
  restricted Johansen estimation.

### GMM flavours (26 October 2026)

- **CUE as a step, not a method:** the continuously updated estimator is a choice of weighting in `GMM`, so the dialog,
  the result and its J statistic are unchanged; it starts from the two-step estimate.
- **The C test in Hayashi's form:** both J statistics minimised with one S, so C is never negative; linearmodels'
  `c_stat` weights the subset differently and agrees only closely.
- **Weak-instrument tests as functions:** Anderson-Rubin and Stock-Wright test values, not fits, so they are functions
  returning test results, and the confidence set is a grid inversion the user can see.

### Estimation by simulation (26 October 2026)

- **Python callables, not a model language:** the user writes `simulate(theta, shocks)` and the auxiliary statistic or
  moments as functions, so any model that can be simulated fits; the calls record in code like any other.
- **Common random numbers:** the shocks are drawn once and held fixed across parameter values, so the objective is
  smooth and an ordinary optimiser works; the weight is the efficient one from the data unless `weight="identity"`.
- **No dialog:** the model is code by nature.

### TVP-VARs with stochastic volatility (26 October 2026)

- **The corrected order:** the volatilities are drawn given the mixture indicators before the other blocks and the
  indicators after them, as Del Negro and Primiceri showed the original algorithm must be changed.
- **One sampler for every path:** forward filtering and backward sampling serves the coefficients, the contemporaneous
  relations and the volatilities, and is tested against the Kalman smoother on its own.
- **W diagonal:** each shock's volatility drifts independently, a simplification of Primiceri's full W.

### Smooth local projections and conditional parallel trends (26 October 2026)

- **A spline that can be anything:** evenly spaced knots, one a horizon, so no penalty is exactly the ordinary local
  projection and a large one exactly a polynomial; the penalty in between is chosen by cross-validation over blocks
  of periods, as serial correlation demands.
- **Covariates without a new estimator:** outcome regression, IPW and the doubly robust estimator plug into the same
  ATT(g, t), aggregation and bootstrap, and an uninformative covariate must give back the unconditional estimate.

### Staggered difference-in-differences (26 October 2026)

- **The diagnosis beside the cure:** the Bacon decomposition shows what two-way fixed effects averages, and the
  Callaway-Sant'Anna and Sun-Abraham estimators avoid the forbidden comparisons; both are one dialog away.
- **One base period:** g - 1 for every ATT(g, t), before and after adoption, so the pre-adoption estimates read as an
  event study and Sun-Abraham agrees exactly in a balanced panel.
- **Exact identities as tests:** the decomposition must sum to the two-way estimate, and the two event studies must
  agree, to rounding.

### Proxy SVARs (26 October 2026)

- **A method of the fitted VAR:** the proxy identifies one column from the VAR's own residuals and covariance, so it
  is `var.proxy_svar(...)`, and its responses reuse the VAR's moving-average coefficients and `IRF`.
- **The bootstrap that is valid for proxies:** the moving-block bootstrap of the residuals and the proxy together,
  as Jentsch and Lunsford showed the wild bootstrap is not.
- **Checked against LP-IV by identity:** at horizon 0 the two must agree exactly with the same lags, which ties the
  two new external-instrument tools together.

### Smooth-transition autoregressions (26 October 2026)

- **Two parameters searched, the rest solved:** for a given speed and centre the model is linear, so the regime
  coefficients are least squares and only (gamma, c) are searched, first on a grid and then by Nelder-Mead.
- **Units out of the speed:** gamma is scaled by the switching lag's standard deviation, so the same value means the
  same sharpness for any series, and the grid needs no tuning.
- **Honest about gamma:** its Gauss-Newton standard error is unreliable when the transition is fast, as the sum of
  squares flattens; the summary says so, and the test compares the other standard errors with the Hessian.

### LP-IV and growth at risk (26 October 2026)

- **LP-IV through the existing IV method:** each horizon is an `Equation` fitted by `IV`, so its standard errors,
  provenance and diagnostics are the same as any IV regression's, and the instrument-as-itself case is exactly OLS.
- **The skewed t made fast by its structure:** location and scale enter the quantiles linearly, so only shape and
  degrees of freedom are searched; with warm starts across periods a long sample takes seconds, not minutes.
- **Quantiles that never cross:** predicted quantiles are rearranged before the distribution is matched, so every
  period has a valid one.

### Panel VARs (26 October 2026)

- **The standard estimator by default:** forward orthogonal deviations with level instruments, as in Stata's pvar and
  the literature; the within estimator is offered but its Nickell bias is stated, and a test shows it.
- **Reuse of the VAR's machinery:** a panel VAR's coefficients feed the same moving-average, identification, IRF and
  FEVD code, so responses and decompositions mean the same as for a VAR.
- **One system GMM for all equations:** the same instruments in each, so the just-identified case is equation-by-
  equation IV and the over-identified case is efficient two-step across equations.

### The plugins' public surface and template (26 October 2026)

- **One module to import from:** `pymetrix.extension` names what plugins need from pymetrix's inside, so internals can
  be renamed or moved by changing one map; the stable modules list says what else is safe.
- **Enforced, not just written down:** the finance plugin and every generated plugin carry a test that fails on an
  import from outside the surface or of a private name.
- **The template is the finance plugin's shape, at its smallest:** each part a real plugin needed (entry points, a
  call, a result with views, a dialog, help, a tutorial run as a test, a validation table) appears once, and a test
  makes a plugin and runs its own tests as an author would.

### The finance plugin as its own package (26 October 2026)

- **Same repository, own package:** `packages/pymetrix-finance` has its own version, tests and docs, so a change that
  needs core and plugin together is one commit; it can move to its own repository later without code changes.
- **Nothing breaks for users:** the 40 finance names stay importable from `pymetrix` when the plugin is installed
  (a module `__getattr__`), recorded projects replay by call ID, and without the plugin the core imports and runs
  and its finance steps become missing steps.
- **Entry points for every group it extends:** the plugin loads when the app or a script first needs any of them.
- **What the move showed:** besides the extension API, the plugin uses the call decorator, `make_call`, three
  private view helpers, the estimators, the state-space engine, progress reporting and help topics (listed in the
  concordance); the plugin template should make these public, or wrap them.

### Volatility surfaces (26 October 2026)

- **QuantLib as the oracle, not a dependency:** Heston and SABR are checked against QuantLib where it is installed,
  as the finance plan said; users need only scipy.
- **Calibration in price space:** Heston is fitted to price errors divided by vega, close to volatility errors but
  without solving for implied volatility at every step, so a fit takes seconds; parameters that end at their bounds
  are named, since they mean the model does not fit, as with a smile that does not flatten.
- **Arbitrage checked where the data are:** SVI's wings are extrapolation; the checks cover the quotes' range.

### Session observers and project attachments; extension API 1.3 (27 October 2026)

- **For a lab book as a plugin** ([research-audit.md](design/research-audit.md)): a plugin cannot see Undo, Cancel
  or a clean-up from outside, so the session reports them; `on_history_changed` exists only in the app, and says
  only that something changed.
- **Every session, not only the app's:** the observers are a plugin group, loaded when a session first reports, so
  scripts, notebooks and the MCP server are observed as the app is.
- **An observer cannot stop the work:** one that raises is reported once and skipped.
- **Attachments survive a missing plugin**, as steps from a missing plugin do: opening a project without the lab book
  and saving it again keeps the lab book.
- **A gap shows rather than hides:** the save log notes the recording observers active at each save, since a lab
  book cannot record work done while it was not installed.

### Finishing the finance plugin; extension API 1.2 (26 October 2026)

- **Report templates as an extension point:** the finance plugin needed one, and built-ins could not be added to
  from outside; a new registry group is additive (1.2), and built-in names win, so a plugin cannot replace one.
- **Draw, not only build:** the result-window tests built every figure but never drew one, which let a broken VaR
  graph through; the finance tests now render each figure to PNG.
- **One tutorial, two forms:** the notebook is made from the page by a script, and a test fails if they drift.

### Credit risk (26 October 2026)

- **Black-Scholes reused:** equity is priced by the plugin's own Black-Scholes, so Merton's model and the options
  agree by construction, and the round trip is a check.
- **The KMV iteration, not two equations a period:** solving Merton's two equations each period makes asset
  volatility jump about with equity volatility; the iteration fixes one asset volatility consistent with the path.
- **CDS on a monthly grid:** quarterly premiums with accrual on default, protection paid at the month of default;
  simple and close to the standard model, and exact for the credit triangle without discounting.

### Copulas (26 October 2026)

- **Margins apart from dependence:** ranks make the margins uniform without assuming their shape (canonical maximum
  likelihood), and the same margins, empirical or normal, map simulated copula draws back to returns.
- **Archimedean families for two series only:** their many-series forms are exchangeable (one parameter for every
  pair), seldom what portfolios need; the Gaussian and t serve any number. (Revised 30 October: see below.)

### More copula families, margins and model checking (30 October 2026)

- **Exchangeable Archimedean copulas for more than two series after all:** Clayton, Gumbel, Frank and Joe take any
  number, said to be one parameter for every pair; they are cheap and a useful contrast with the t. Vines, which give
  each pair its own family, remain the answer for heterogeneous portfolios and are not done.
- **Generator derivatives in closed form, not by numerical differentiation:** the d-dimensional densities need
  psi^(d); Gumbel's by a recursion whose coefficients are all positive (no cancellation, unlike the alternating
  Stirling-number sums), Frank's by the polylogarithm of negative order, Joe's by Stirling numbers of the second kind,
  each a positive sum taken in logs. The bivariate Frank is in closed form for either sign of theta.
- **Rotations as a wrapper, families as names:** `clayton180`, `gumbel90`... are a family string each, so a list of
  families compares them like any other; the rotation maps the density, CDF, h-function, sampler, tau and tails.
- **Marshall-Olkin sampling where there is a frailty:** exact and fast in any dimension (gamma, positive stable,
  logarithmic series, Sibuya; BB1 and BB7 by compound frailties); conditional inversion of the h-function for the
  bivariate Frank (negative theta has no frailty) and as an independent check of the others.
- **Hansen's skewed t for margins:** pymetrix's other skewed t (growth at risk) is Azzalini-Capitanio's, whose CDF
  needs quadrature; Hansen's has its CDF and quantiles in closed form, which IFM, joint ML and simulation need at every
  observation and draw.
- **GARCH margins are the GARCH's own errors:** with `margins="t"` the t's degrees of freedom are the GARCH's,
  estimated with it, so the margin is the model that filtered the returns; skewed-t margins are fitted to the
  standardised residuals afterwards (the native GARCH has no skewed-t errors).
- **IFM standard errors without the sandwich:** the observed information of the copula alone, stated as ignoring the
  margins' estimation; `method="joint"` gives the full ones. The Godambe sandwich (and the Genest-Ghoudi-Rivest
  variance for rank margins) is left for later.
- **Goodness of fit on the ranks:** Genest, Remillard and Beaudoin's parametric bootstrap needs each bootstrap sample
  treated as the data were; doing it on the pseudo-observations whatever the margins keeps it one test, about the
  copula alone. The bivariate Gaussian and t CDFs are by Gauss-Legendre quadrature of the h-function, split where it
  turns: about 1e-8, against scipy's Genz algorithm at its default 1e-5.
- **Existing results unchanged:** the Gaussian, t, Clayton and Gumbel fits and samplers keep their code paths (the
  same numbers to the last digit); `portfolio_var` defaults to the fitted margins only when they are parametric.

### Vine copulas, time-varying copulas and goodness of fit in any dimension (31 October 2026)

- **A vine is a list of trees of edges, not a matrix:** each edge carries its conditioned pair (the copula's first and
  second argument), its conditioning set, family and parameters; the conditional pseudo-observations are a dictionary
  keyed by (variable, conditioning set), so the likelihood, the h-function cascade and the Rosenblatt inverse are the
  same short loop for R-, C- and D-vines. The lower-triangular R-vine matrix (VineCopula's convention, the copula of
  (F(M[k, i] | D), F(M[i, i] | D))) is derived from the trees, by Dissmann et al.'s construction, for simulation
  order and for interchange (`matrix=`, `to_vinecopula()`), and checked against RVineLogLik.
- **Pair families are the module's bivariate families:** the h-functions of `copulas` serve as hfunc1; hfunc2 is
  hfunc1 of the transposed family (the 90 and 270 degree rotations swap, the rest are exchangeable), so no family is
  written twice. Independence is a family with no parameters. The Gaussian and t pairs are estimated by full maximum
  likelihood (the t by its profile likelihood in nu), unlike the multivariate `copula`, whose estimators are kept.
- **Selection as VineCopula does it:** maximum spanning trees on |tau|, the C-vine root the largest sum of |tau|,
  families by AIC among those of the edge's sign, trees above a truncation level any valid tree (depth first); so
  RVineStructureSelect's choice is reproduced and differences are numerical. A D-vine's order maximises the sum of
  |tau| along the path, exactly up to 12 series (VineCopula needs the TSP package for this).
- **Standard errors per pair:** each edge's from its own observed information, inputs taken as known; the joint
  Hessian of every parameter (and the margins') would be the honest one but is costly, and is left out.
- **Patton's recursion, with a defined start:** before the sample the forcing variable is its sample mean and the
  parameter the recursion's fixed point there, so the path is a deterministic function of the parameters and data
  and alpha = beta = 0 is exactly the constant copula; the restricted (constant) likelihood is reported beside. The
  t copula's forcing variable uses its own quantiles (as Patton's code does). Tails are kept in [1e-6, 0.99] and the
  summary says when the path reaches the bounds (kinks in the likelihood). For many series' correlations the DCC
  multivariate GARCH already exists; a time-varying vine is not done.
- **Goodness of fit in more dimensions by an exact-enough CDF, not a new test:** the Gaussian and t CDFs by Genz's
  separation of variables averaged over one fixed scrambled Sobol sample (the t as a normal scale mixture), about
  1e-5 and smooth in the parameters, so the same parametric bootstrap applies in any dimension; the multiplier
  bootstrap (Kojadinovic and Yan) would avoid refitting but needs the estimator's influence function, and was not
  done then (it is now: see below).

### BB6, BB8 and Tawn pairs, vine goodness of fit, DCC copulas and the multiplier bootstrap (1 November 2026)

- **New families as objects of the same shape:** BB6 and BB8 are Archimedean generators like the others, but their
  densities, CDFs and h-functions are worked in log phi and log t (t = phi(u) + phi(v)): in plain arithmetic phi
  underflows at large parameters near the corners and the likelihood search fails on the first trial step. Tawn's
  copula is an extreme-value copula (a Pickands function, not a generator) with the same methods, so `Rotated`, the
  vine's pair functions and the parametric bootstrap take it unchanged. Kendall's tau is integrated at 1e-12 rather
  than R's default tolerance (BiCopPar2Tau's values are up to 1e-7 off; checked against R's integral at 1e-13).
- **Tawn is not exchangeable:** its transpose is the other type, so `transpose` swaps the types (and 90 with 270),
  and hfunc2 still comes from hfunc1 of the transposed family. VineCopula's 90 and 270 degree Tawn copulas turn the
  density rather than reflect one argument (the same thing for exchangeable families); `tawn1_90` is that turn, the
  reflection of type 2, so VineCopula's codes map one to one.
- **Rotated two-parameter families are named with an underscore** (`bb1_90`, `tawn2_180`): `bb190` would read as a
  number. The defaults stay as they were: `compare_copulas` and vine selection look among the earlier families unless
  the new ones are listed (or `families="all"` for a vine), so existing results do not move.
- **The vine test is VineCopula's ECP2 statistic with a parametric bootstrap that refits:** the Rosenblatt transform
  comes from the same h-function cascade as the likelihood (the conditional pseudo-observations are already keyed by
  conditioning set), so it is a lookup per variable. VineCopula's own ECP2 p-value resamples the data with the
  parameters fixed, which ignores their estimation; refitting each sample on the same structure and families is the
  Genest-Remillard-Beaudoin recipe applied to the vine. Selection is not repeated (it would multiply the cost by the
  number of candidate families); the known gaps say so.
- **The multiplier bootstrap follows the copula package's C code:** the empirical copula's derivatives by finite
  differences of width 2 / sqrt(n) without boundary adjustment, normal multipliers, the parametric part through the
  estimator's influence (maximum pseudo-likelihood's with its rank correction, Kendall's tau's for two series) with
  every derivative numerical, so it works for any family whose parameters are all estimated that way. The t copula
  is refused: its correlations come from Kendall's tau and its degrees of freedom from the likelihood. The Gaussian
  correlation here is the normal scores' (asymptotically the pseudo-likelihood estimator, so the same influence).
- **The DCC copula reuses mgarch's recursion:** `dcc_path_linear` in `mgarch` runs `dcc_path`'s recursion for every
  element of Q at once with a linear filter (about ten times faster; checked equal to `dcc_path` to 1e-12), and
  mgarch's own fits are left on `dcc_path`. Qbar is targeted (z'z / T of the copula's scores), so a = b = 0 is the
  constant copula, and for ranks exactly the Gaussian copula fit. The t copula's scores depend on nu, so Qbar moves
  with it.

### A result for Black-Litterman (1 November 2026)

- **`BlackLittermanPosterior` in place of a tuple,** so the window shows labelled tables and a chart rather than
  "Item 0" and "Item 1". It keeps the prior, posterior, covariance, views (labelled as the dialog writes them), q,
  omega, delta and tau, and computes the He-Litterman optimal weights on demand. Old code that unpacks
  `mu, sigma = black_litterman(...)` or indexes `[0]`, `[1]` keeps working: the result iterates over, and indexes, the
  posterior mean and covariance as NumPy arrays, the very arrays the tuple held. Asset names come from the
  covariance's labels (a DataFrame, or a pymetrix table such as `data.cov(...)`), else the market weights' index,
  else "asset 1", "asset 2"... Provenance records the inputs and a fingerprint of the covariance and weights; the
  plugin builds it from `pymetrix.metadata.Provenance`, a stable module, and shows it in its own table.

### Fat tails and asymmetry in GARCH (26 October 2026)

- **The estimator was already there:** the native GARCH estimates t errors and GJR terms, cross-checked against
  arch; the finance plugin only passes them through, so the VaR and the estimation cannot disagree.
- **Report what the data say:** fat tails and asymmetry halve the gap but monthly market returns still fail a 99%
  backtest; the tutorial says so instead of picking the configuration that passes.

### Strategy backtests and term structure (26 October 2026)

- **No look-ahead by construction:** a backtest's weights at each rebalance come from a function of the window
  before only, and a test compares them with the public optimiser on that window.
- **Costs where they arise:** turnover is measured against the drifted weights, so a strategy that rebalances
  often pays for it, which is how minimum variance can beat maximum Sharpe after costs.
- **The Kalman filter reused:** the dynamic Nelson-Siegel model is a `SystemMatrices` for the existing filter and
  smoother, and its likelihood is checked against statsmodels'.
- **A floor under measurement errors:** with real yields the likelihood grows without bound as one maturity's error
  goes to zero; the floor is stated in the docs rather than hidden.

### Realised volatility, finance datasets and the tutorial (25 October 2026)

- **Intraday data are ordinary datasets:** intraday periods (five-minute and so on) already work, so realised
  measures need no new data type; each calendar day's returns make one row of a daily dataset.
- **Real where redistribution is clear:** Kenneth French's returns through linearmodels' copy, as for the earlier
  classics, and Treasury yields from FRED, public domain; the rest simulated with their truths stated, so the
  tutorial can show the methods recovering them.
- **A tutorial that is a test:** its code blocks run in order on every test run, so it cannot drift from the code.

### Risk contributions and event studies (25 October 2026)

- **Components that add up:** Euler allocation, so the parts always sum to the portfolio's VaR and ES; the
  historical component VaR, which has no exact sample form, is estimated near the quantile and rescaled to add up.
- **Several tests, not one:** the cross-sectional t, Patell, BMP and rank tests answer different worries (variance
  changed by the event, fat tails); all are reported, and their size is checked by simulation.
- **A separate name:** `abnormal_returns`, since `event_study` is the treatment-effects estimator on panels.

### Options and yield curves (25 October 2026)

- **Closed forms where they exist, a tree where they do not:** European options by Black-Scholes-Merton; American
  ones by a binomial tree, whose Greeks come from its own first nodes (bumping the spot by less than the node
  spacing only measures the tree's jaggedness).
- **No implied volatility outside the bounds:** a price below intrinsic value or above the discounted spot has no
  volatility, and is shown as missing rather than forced to an edge.
- **One decay for the sample:** Diebold and Li's choice, so the factors are comparable over time and are ordinary
  series to model; Svensson's two decays are kept apart, as they are otherwise not identified.
- **Checked against the textbook:** QuantLib is not a dependency, so the checks are Hull's worked examples, parity,
  convergence and exact recovery from simulated curves.

### Returns, performance, factor models and portfolios (25 October 2026)

- **Per period in, annual out:** functions take returns per period as in the data; only the performance measures
  are annualised, by the data's frequency unless told otherwise, so nothing is guessed silently.
- **The exact GRS statistic:** covariances divided by T, so it is exactly F(N, T - N - L) under normal errors; with
  one asset it equals alpha's squared t statistic, which is the test.
- **scipy, not a solver dependency:** SLSQP handles budget, bounds and target constraints for the handful of
  assets a workbench sees; cvxpy remains an optional later choice for larger problems.
- **Covariance as an input:** a portfolio can take a fitted multivariate GARCH and use its forecast for next
  period, joining the two pieces without either knowing the other's internals.

### Multivariate GARCH (25 October 2026)

- **Two-step where it scales, one step where it is small:** CCC and DCC reuse the univariate GARCH and add a
  correlation stage, so many series stay tractable; BEKK estimates everything at once for a few series.
- **Variance targeting:** BEKK's long-run covariance is the sample's, leaving two (scalar) or 2N (diagonal)
  parameters, and its constant is checked to be a covariance.
- **Honest standard errors:** DCC's come from the second stage alone, which understates them a little, and the
  summary says so.
- **Checked without a reference library:** no maintained Python library estimates these, so the checks are
  recovery of known parameters, a plain-loop likelihood and the special cases.

### The finance plugin: value at risk (25 October 2026)

- **A first-party plugin, built as an outside one would be:** registered through the extension API only (dialogs,
  results, views, menu, workspace), so it proves API 1.1 and can move to its own package.
- **Out of sample, always:** each period's forecast uses only the returns before it, so the backtests judge
  forecasts that could have been made at the time.
- **Our own code, others as checks:** the methods are short and checkable, so they need no new dependency; the arch
  package checks the GARCH forecast in the tests, and statsmodels the dynamic quantile regression.
- **Losses as positive numbers:** VaR and ES are reported as losses, the convention of risk reports and the Basel
  rules, with the returns' own units kept.

### Extension API 1.1 and plugin decisions (25 October 2026)

- **Extension points that hand over an interface, not the window:** a panel gets `AppInterface` (session, selection,
  opening a dialog, running a step through the app's runner, a history notice) and nothing else, so the app can
  change around it; a step a plugin runs is recorded, cancellable and undoable like any other.
- **Results describe themselves:** a plugin's result class says its stage and next steps as class attributes, so
  badges and Next ▸ need no table in core.
- **Additive versioning:** 1.1 only adds; the major version is what is checked, so 1.0 plugins keep loading.
- **Decided for plugins** (the questions in the plugins design note):
  - credentials go through one small helper backed by the operating system's keychain (`keyring`, optional);
    projects and code name a credential, never hold it;
  - remote reads are cached under `~/.pymetrix` only when the vintage is fixed, so `latest` always reads fresh;
  - a plugin template (pyproject, entry points, `PluginInfo`, a setting, a reader, the conformance tests, and the
    1.1 points) comes before 0.1.

  Still open: a generic Data → Get data with catalogues (to be built with the first real data plugin), and the
  graphics route (exporters first, on demand).

### Autosave, the data preview and the family hint (25 October 2026)

- **Recovery without surprises:** the autosave is a copy, so the session keeps its own file and its unsaved state;
  a running app's heartbeat keeps two windows from taking each other's copies; the offer comes from the launcher
  once the window is up, never in the middle of a test or a script.
- **Guesses that stay out of the way:** Load data fills only the fields still empty, and replaces only its own
  earlier guess when the file changes; the family hint never changes the form without the user's click.
- **Rules a reader can check:** the guesses are simple, stated rules (years 1800-2200, period strings, the spacing
  of dates; counts are small whole numbers; a pile-up is over 10% at the lowest value), each said in words.

### Plugin management (25 October 2026)

- **A project outlives its plugins:** a step that cannot run here is still the project's, so it is kept, explained and
  saved back unchanged, as untrusted typed code already was; failing to open the whole project lost everything for
  want of one plugin.
- **Ask before running, from what can be read without running:** a plugin's code runs on import, so the question
  about a new plugin uses its package's metadata (name, author, licence, the groups it hooks into), never counts of
  what it registers.
- **One set of choices:** the app and scripts read the same enabled and disabled lists, so a project behaves the
  same in both; scripts never stop to ask, and load a new plugin with a warning.
- **Two ways to disable:** installed plugins are skipped before import; the bundled ones, imported with pymetrix,
  are hidden in the registries, so a direct import in a script still works.

### Example datasets, GMM and treatment effects (25 October 2026)

- **Data that come with their terms:** the real datasets are taken only from libraries that may redistribute them
  (linearmodels under NCSA, statsmodels' public-domain sets), each saying so in its metadata; the simulated ones
  state their true effects, so they serve as tests as well as examples.
- **GMM as moments of an equation:** the moments are instruments times the residual of an ordinary pymetrix
  equation, so a model, its formula and its derivatives are written once; its first step is two-stage least
  squares, so the familiar estimator is the special case.
- **Methods that state their assumptions:** each treatment-effect result says what it assumes (parallel trends,
  local linearity, no unobserved confounders, a convex hull) and shows the evidence for it: leads, other
  bandwidths, the overlap of propensity scores, placebos.

### Limited dependent variables and structural breaks (25 October 2026)

- **Our own likelihoods, checked against others':** the models are written in pymetrix so that they share one
  maximiser, one covariance (classic, sandwich, cluster) and one set of marginal effects; statsmodels and published
  textbook results are the independent checks, not the implementation.
- **Ranges handled by reparameterising:** alpha, sigma, rho and ordered cut points are estimated where any value is
  allowed and reported on their own scale by the delta method, so the optimiser never steps outside the model.
- **Effects that answer the question asked:** marginal effects are on E[y] or on the probabilities, and a dummy's
  effect is the change from 0 to 1, since the derivative of a 0/1 variable means nothing.
- **A search priced as a search:** Quandt-Andrews' p-value comes from the limit distribution of the largest F, not
  from the F distribution of one date; it is tabulated offline (1 to 20 coefficients, trimming 5% to 45%), simulated
  for any other case, and checked against Andrews' table.

### Forecasting benchmarks, named matrices and sign restrictions (25 October 2026)

- **Benchmarks that match the references:** seasonal ARIMA wraps statsmodels but reports standard errors from the
  numerical Hessian, as R does, so the airline model's table reads as in the textbooks; theta and STL forecasting
  are there to be beaten, so they backtest like everything else.
- **Names catch the errors numbers hide:** a product whose names do not meet is refused, since a matrix in the
  wrong order gives a plausible wrong answer; unnamed matrices still meet anything of the right size, so NumPy
  habits carry over.
- **A set reported as a set:** sign restrictions identify many models. The responses are pointwise medians with
  bands that say whether they span the set or the sampling uncertainty too; anything that needs one model
  (variance decompositions) uses the accepted model nearest the medians, never a mixture.

### Long steps in the background; the start page; badges (25 October 2026)

- **A live window, one step at a time:** a long step runs on a worker thread while the window stays usable for
  reading, but nothing else may change the session until it finishes; the dialog still returns when the step is done,
  so everything that follows a step behaves as before.
- **Cancel that is honest:** Python threads cannot be stopped from outside, so long computations check for Cancel
  where they report progress; a cancelled step leaves no trace, and a computation without checks finishes.
- **A way in, and files where they are dropped:** an empty session starts from data, a recent project or a demo,
  with the workspaces explained in two lines; a file dropped on the window opens as a project or through Load data.
- **Attention where the work is:** the stage bar counts each stage's results that are out of date or not computed,
  so a data revision shows which stages it reached; the Problems panel says which results. A result not computed yet
  counts under the stage of the menu its step comes from.
- **A dialog belongs to its running step** (review fixes): OK cannot run it twice, and Cancel on the dialog means
  stop the step, never hide it while it goes on. A step stopped only in the recalculation after it is kept: it ran,
  and what depends on it is out of date, as after any manual change.
- **Undo that never loses work:** a step is undone by putting back the objects it found, which is exact only while
  nothing else has changed them, so Undo checks that the objects are as the step left them and otherwise refuses.
  Typed code can change objects in place, which no saved reference undoes, so it ends the chain. Ctrl+Z belongs to
  the widget with the focus when that widget has an Undo of its own.
- **Cancel where the time goes:** optimisers check at each evaluation of their objective, which is where estimation
  spends its time, so Cancel takes effect within one evaluation.
- **Shortcuts read from the menus:** Help → Keyboard shortcuts walks the menus as built, so it cannot fall behind
  them, and a test checks no two commands share a key (Qt fires neither when they do). Symbol-only buttons have
  accessible names.

### Next steps (25 October 2026)

- **The stages connect through results:** the usual next steps live where the result is, so moving on is one choice
  with the result already filled in, not a trip through menus.
- **Leading, not forcing:** from a stage's workspace a step moves to its own stage, showing where the work has gone;
  in All nothing moves. The steps are ordinary dialogs, recorded as ever.
- **One table:** the steps are listed by result type in one place, and a test checks each names a dialog, the field
  that takes the result, and a workspace.

### The report outline (25 October 2026)

- **Collect as you go:** reporting goes on throughout the work, so Add to report is wherever a figure, table or result
  is shown, in every workspace; the Report workspace is where the outline is ordered, given headings and text, and
  exported, not the only place it grows.
- **Compose freely, record the export:** ordering and editing the outline are composition, like arranging windows,
  so they are not steps; the export is, and its code writes the whole report out item by item, so it can be written
  again without the app.
- **No new dialog:** the export reuses Export report, which takes the outline; a dialog keeps the hidden fields of the
  command it was opened on.

### Workspaces for the stages of the work (25 October 2026)

- **The natural order, not a pipeline:** projects run Explore → Estimate → Model → Forecast → Report, and the stage
  bar shows that order left to right; but work loops back (revisions, failed diagnostics, forecasts drawing on both
  estimation and models), so a workspace only shows and hides, switching keeps the session, and All and Find
  anything always reach the whole app.
- **Exploring is a stage of its own:** looking at the data (loading, checking, plotting distributions and
  correlations, stationarity and seasonality) comes first for everyone, and students most of all.
- **Code is a way of working, not a stage:** the Code toggle adds the code panels to any workspace, for those who
  script and those who read the code the dialogs write.
- **The layout is the user's, per workspace:** each workspace remembers how its panels were left, and the workspace
  in use is a user setting, not part of a project.
- **Next:** a report outline that collects figures and tables from every workspace, then actions on result windows
  that lead to the next stage with the result filled in.

### Find anything; the series browser (25 October 2026)

- **One search box for the whole app:** with commands in submenus, typing a few letters is the fastest way to any of
  them; the palette searches what the menus hold (so it never drifts from them), the workspace and the help, and
  shows each item's place, so it also teaches where things are.
- **Metadata to find series by:** in a large dataset, series are found by label, units, source or sample, not by
  scrolling names; the data dictionary becomes a table to filter and sort, and what is picked there can be plotted,
  edited, named as a selection or made the working set in one step.

### Working sets and searchable lists (25 October 2026)

- **Large datasets need a scope, not longer lists:** a working set says which of a dataset's series the app shows
  everywhere at once (dialogs, the spreadsheet, the dataset's window), set once by pattern or selection.
- **A view, never data:** commands still name every series they use, so a script or a replayed project does the same
  with or without the working set; it is saved with the project only to keep the view.
- **Never silent:** every scoped list says how many series it shows and from what, with Show all one click away; a
  series already chosen stays visible. Large datasets suggest a working set rather than having one imposed.
- **Search everywhere:** each series list can be filtered, shown ticked-only and counted, so even an unscoped list of
  thousands is usable.

### The spreadsheet's edit line; help contents that fold (25 October 2026)

- **An edit line, as spreadsheets have:** the selected cell's full value is easier to read and change in a line
  than in a narrow cell; ✓ and ✗ make accepting and abandoning explicit. Accepting makes the same pending edit as
  typing in the cell, so nothing new needs undoing or recording: Apply still records all edits as one step.
- **Contents a reader can scan:** one page per line, under headings that fold, rather than paragraphs of links.
  The open sections live in the link, so the viewer needs no state and Back simply goes back; readers that cannot
  click get every section open, and notebooks fold with their own `<details>`.

### Menus grouped by task (24 October 2026)

- **Order by the work, not by when a module was written:** each menu runs in the order the work is done (look at
  the data, make series, check and set the sample, then transform or reshape; estimate single equations before
  systems; define, solve, estimate a model, then analyse it), with separators between stages.
- **Submenus for families, not for everything:** a submenu holds a family whose members are used together (Panel,
  Transform, VARs and local projections, Policy); common steps stay at the top level.
- **One table, and nothing left out:** the layout is a single table; a command it does not name (a plugin's, or a
  new one) still appears, at the end of its menu, and a test fails until a built-in command has its place.

### One house look for charts (6 November 2026)

- **Settings, not global state:** the look is a dict of matplotlib settings (`figure_style.chart_rc`) applied with
  `rc_context` around the drawing of each pymetrix figure, so scripts and notebooks that also use matplotlib for
  their own plots are never changed by importing or drawing with pymetrix. The axes pin their tick and grid look
  explicitly too, because matplotlib makes some ticks lazily, after the context has closed.
- **Layered:** house look, then the theme's sizes and palette, then the theme's style sheet, then the figure's
  recorded style calls. Each later layer wins, so nothing a user set before changes meaning.
- **Okabe-Ito by default:** the palette was already the default theme's; it is now cited and tested against the
  published RGB values, in its fixed order.
- **Dark theme, the simplest robust way:** the chart takes the window's own background and text colours, the grid is
  a blend of the two, and neutral dark greys pymetrix or a note drew (an "actual" line, reference lines, the
  palette's black) take the opposite lightness. Hues and colour maps are left alone, so a series keeps its colour.
  This happens in the app's drawing only (`Figure.draw(surface=...)`); copies, saved files and reports are light.
- **Sharp without moving anything:** the canvas draws at the device pixel ratio (matplotlib's own mechanism, called
  at once rather than at first show), and zoom accounts for it; inches never change, so saved projects, report
  layouts and window sizes are as before.

### Mixed frequencies; a review of the charting code; the chart gallery (24 October 2026)

- **Each value over its own period:** when a panel mixes frequencies, values sit at the middle of their periods,
  so a quarter's bar covers its three months beside the monthly line; a panel of one frequency keeps the
  conventional starts. The axis, readout and notes use the finest frequency.
- **A period means the whole period:** a span or a band given as 2023Q4 covers October to December on a monthly
  axis, as a reader means it.
- **The script mirrors the drawing, not a copy of its numbers:** where pymetrix decides at draw time (shading and
  marks within the dates shown), the script makes the same decision against its own axis, so the two agree.
- **An independent review, then a test per finding:** fifteen bugs found by a separate reader, each reproduced,
  fixed and kept fixed by a test; the gallery's test runs every chart's script, which is how one more was found.

### Charts in results and reports; a house style; the value under the mouse (24 October 2026)

- **Charts where people look:** the Q-Q plot and the heatmaps appear as tabs of the results they explain, so they
  need no extra step; a table of numbers becomes a heatmap with one button, recorded like any step.
- **Report figures at their printed size:** a figure drawn at 7.5 inches and shrunk to 6 has smaller text than
  intended; drawing it at the width it appears keeps type in points. Each gets alt text from what the figure
  shows, so reports are accessible and searchable.
- **A house style that never surprises:** it applies only where its calls fit, the figure's own calls win, it is
  recorded as a step, and a saved result keeps what it was drawn with. Opening a project starts from its own
  house style, never one left from another project.
- **A readout, not a cursor:** the value under the mouse is shown in words beside the figure, with dates as
  periods, and changes nothing. Since November 2026 it is off unless Values is switched on: a running readout
  distracts while reading a chart, so a click identifies a series or a point instead (`ui/chart_click.py`), in
  a box drawn on the window's canvas only.

### Distributions (24 October 2026)

- **Distributions without choosing bins:** the empirical distribution shows every observation, so several series
  compare in one panel without the look depending on bin edges. Histograms keep their bins, now settable.
- **Counts where points pile up:** a two-dimensional histogram replaces a scatter plot when points overlap, with
  empty cells left blank so the shape of the data shows.
- **An x-y panel is not dated:** scatter plots and 2-D histograms take their x from a series, so their axes are
  numbers even when the series are dated.

### Coefficient plots and heatmaps (24 October 2026)

- **New layers, not new escape hatches:** an estimate plot and a heatmap show data, so they are layers with their
  numbers inside, saved, replayed and written as matplotlib code. `fig.call` still refuses methods that plot data.
- **Intervals come from the results:** a coefficient plot asks each result's parameters for their interval, so a
  Bayesian result shows its posterior interval and a classical one its normal interval, without the chart knowing
  which is which.
- **Scales that cannot mislead:** values of both signs get a diverging scale symmetric about zero, so equal colours
  mean equal distances from zero; correlations are always shown from −1 to 1.

### Macro charts and saving for publication (24 October 2026)

- **Macro charts from existing parts:** a contributions chart is stacked bars, a line and a zero line; a fan is
  bands, a line and a start line. Each is built from layers pymetrix already draws, saves, styles and writes as
  matplotlib code, so the new charts get all of that at once.
- **The bars add up to the line:** when a total is given and the parts do not explain it, the rest is drawn as
  other rather than left as a gap between the bars and the line. A chart never shows a sum that does not add up.
- **A fan is one layer:** it keeps its central path, coverages and bands together, so a saved result and a replay
  keep the fan, while drawing sees the bands and line, each of which can be styled.
- **Sizes in print terms:** journals ask for widths in millimetres, so the presets are journal columns and pages,
  slides and half an A4 page. Text stays in points, as in print.
- **Files that explain themselves and repeat:** a saved file records the versions and the recorded steps, and
  leaves out the date, so the same figure always writes the same bytes. That makes a figure in a paper checkable
  against the project that made it.

### Managing recorded styles (24 October 2026)

- **The list is the code:** each recorded style call is shown as the Figure method call that records it, and
  running those calls on the unstyled figure rebuilds it exactly. What the dialog lists is what a script would say.
- **Editing is recorded too:** removing, reordering and changing calls are steps of their own
  (`without_style`, `reorder_style`, `restyle`), so the history explains how a figure reached its look.
- **Order matters, as in matplotlib:** calls apply in order and the later one wins, which is why moving a call is
  offered next to removing it.
- **Copying applies where it fits:** a style is copied call by call, only where the target has the series, panel or
  right-hand axis the call names. Notes are left out unless asked for, since they describe the other chart's data.
  The calls left out can be listed, and a copy that would apply nothing is refused with the reasons.

### Styling figures: phase D, the rest of matplotlib (24 October 2026)

- **An escape hatch that stays data:** `fig.call` records a method's name and plain arguments, never code, so a
  project with styled figures is still safe to open and replays exactly.
- **Styling only:** the allowed methods style or annotate. What a figure shows comes from its layers, so `plot`,
  `bar` and the like are refused: data reach a chart through the series, with their provenance.
- **Checked when added:** the arguments are bound to the method's signature on a scratch axes, so a wrong keyword
  is reported in the dialog, not when the figure next draws.
- **Style sheets as themes:** a `.mplstyle` sheet maps onto the theme's own fields where it has them, and the rest
  is kept as matplotlib settings, so a house style carries over without pymetrix knowing every setting.
- **The standalone script is the check:** `matplotlib_code` writes each option and style as its matplotlib call,
  and the tests run the script and compare it with pymetrix's drawing (data, colours, limits, texts, patches). A
  matplotlib user can read exactly what the chart is.

### Styling figures: phase C, notes placed by clicking (24 October 2026)

- **The calls are matplotlib's:** `annotate`, `text`, `axhline`, `axvline`, `axhspan` and `axvspan`, with their own
  arguments. Positions are in the data's terms: periods for dates, values for values.
- **A band covers its last period:** a band from 2012Q1 to 2013Q4 runs to the end of 2013Q4, as shading and
  recession bands do.
- **Clicks are recorded as data, not pixels:** the click gives the panel, the period and the value. The note's text
  is offset in points (15, 20), matplotlib's own idiom, so it sits beside the point at any figure size and in any
  saved format.
- **Each axes knows its panel and frequency:** that makes a click on a right-hand twin land in the right panel,
  with the date written in the data's own periods.
- **Prefill, not instant recording:** a click opens the dialog with the position filled in, so the words, colour and
  kind are chosen before anything is recorded. One step per note, as with every other change.

### Styling figures: phase B, every property (24 October 2026)

- **The catalogue is matplotlib's own:** `ArtistInspector` lists each artist type's setters and their valid values,
  so pymetrix lists nothing by hand, and a newer matplotlib's properties appear on their own.
- **Two filters keep it honest:**
  - properties holding the plotted data or positions are left out, so the editor cannot change what is plotted;
  - so are those needing Python objects (transforms, colormaps, path effects), which a project could not save.
- **One form mechanism:** the editor is a settings sub-form, the same mechanism that shows an estimation method's
  settings, whose fields come from the catalogue for the chosen part. The values come back as option text, so the
  recorded call is the one that ran.
- **Typing:** colours, words and list choices are quoted. Everything else is read as written, so `(0, 10)` stays a
  tuple and `2` a number, while `0.5` as a colour and `steps-post` from a list stay text. The second of these was
  caught by the dialog's own validation, as matplotlib refused `'steps - post'`.
- **Tested exhaustively:** each catalogued property with a fixed kind of value is applied and drawn, so the editor
  never offers something matplotlib cannot use.
- **Tabs:** dialogs can ask for a tab per group (`group_tabs`). Figure style does, giving matplotlib's Series, Axes,
  Legend and Layout.

### Styling figures with matplotlib's vocabulary (phase A built, 24 October 2026)

- **Built (phase A):**
  - figures record matplotlib-style calls (`line`, `bars`, `band`, `set`, `ticks`, `grid`, `spines`, `legend`,
    `layout`) as data, checked by matplotlib when set and applied when drawn;
  - the Figure style dialog (Edit… → Style) records the chain of calls.
- **Choices made in building it:**
  - **Series styles before the legend:** they are applied as each panel is drawn, before its legend is made, so the
    legend shows the styled lines. Axes, ticks and legend calls are applied after every panel, in the recorded
    order.
  - **One list of calls, two uses:** the dialog's settings become a list of calls, and both `run` and `to_code` use
    that list, so the recorded code is exactly what ran.
  - **The kind in the label:** the series list shows how each series is drawn ("gdp (line)", "i (bars)"), so the
    recorded call can be `.line(...)` or `.bars(...)`. A command's code may not depend on the session.
  - **Validation by drawing:** the dialog draws the styled figure before the step runs, so matplotlib's own message
    appears in the dialog.
  - **One form for sequences:** tuples throughout, so a style compares equal after a JSON round trip.
- **Found on the way:**
  - dialogs without a menu item could not be built (Figure options from Edit… crashed);
  - saved results dropped every figure option added this week.

  Both are fixed.

- **The aim:** expose as much of matplotlib as possible within pymetrix's figure model, in the dialogs, with code
  that replays and can be verified.
- **The design:** `docs/notes/figure-styling.md`. Styles are recorded on the figure as matplotlib's own calls, held
  as data (`fig.line("gdp", color="C1")`, `fig.set(yscale="log")`, `fig.legend(loc=...)`, `fig.annotate(...)`), and
  each is checked by matplotlib when it is set.
- **The dialogs:** Figure options gets tabs that follow matplotlib's Figure → Axes → artist hierarchy, with a property
  editor generated from `ArtistInspector`.
- **The escape hatch:** any `Axes` method can be recorded as data.
- **Replication outside pymetrix:** `fig.matplotlib_code()` writes a standalone matplotlib script.
- **Phases:** four (A to D).

### Recession dates and line-end labels (24 October 2026)

- **Recession dates:** recession shading is the commonest addition to a macro chart, and needing a dummy series
  for it was friction.
  - The NBER (US) and CEPR (euro area) dates are built in, as small tables in the code. They are a few dozen dates
    that change rarely, so no download is needed.
  - Each committee's own monthly or quarterly dates are used at that frequency. They are dated separately, and do
    not always map onto each other (1980: January monthly, Q1 quarterly).
  - A recession runs from the period after the peak through the trough, as FRED's USREC and USRECQ define it, so
    the shading matches the series most users already know.
  - **Keywords are resolved when drawn:** `shade="nber"` is kept as a marker and turned into spans at the axis'
    frequency, so one figure option works for monthly, quarterly and annual charts alike.
  - **Shading is clipped to the data's dates.** Without that, 1948's recession stretched a 2000–2024 chart back to
    1948.
  - **Updates:** the dates are as announced by October 2026, and the module says so. New announcements (the next
    peak) are a one-line change.
- **Line-end labels:** with three or more lines, a legend makes the reader match colours. Naming each line at its
  end, in its colour, is the chart-pack convention.
  - The labels are stacked to keep a minimum gap, so lines ending close together stay readable.
  - The axis gets room on the right, and the date ticks stop at the data's end so the room has no dates in it.
- **Legends:** a labelled shading always gets a legend entry, even on a one-series chart that otherwise has none.
  Grey bands need explaining.

### Zoom and pan; conditions in formulas (24 October 2026)

- **Zoom and pan:** charts in the app were static pictures, so the window now has matplotlib's own Qt toolbar.
  - It is cut to Home, Back, Forward, Pan and Zoom; Save, Subplots and Customize are left out, since the window has
    its own buttons and figure options.
  - Zooming is a view, not a change to the figure. Nothing is recorded, and Home returns to the figure as drawn.
- **Conditions in formulas:** a recession dummy could not be written as a formula, which shading made obvious.
  - Comparisons and `and`/`or`/`not` join the formula language with Python's precedence, so `not u > 5 and x` reads
    as `(not (u > 5)) and x`. Chained comparisons (`0 < x < 1`) are refused rather than read the Python way, with
    a hint to use `and`.
  - **1 and 0, not booleans,** so conditions multiply and add like any series: dummies, regressors and switches.
  - **Missing stays missing,** unlike pandas, where NaN compares as False. A dummy should not turn a gap into a
    "no".
  - **Derivatives are zero,** as for `step()`: the jump is measure-zero for solvers.
  - **Sample conditions (`where=`) are unchanged.** They are evaluated by pandas and already had comparisons.

### Editing figures (24 October 2026)

- **Why:** a figure's window had Copy image, Save and Copy data but no way to change it. A result's own figure (a
  fit, a fan chart) could take a title, shading or events only in code.
- **One button, two paths.**
  - A figure made by Plot series reopens that dialog, because it can change everything, including the kind of
    chart and the series. It gained a subtitle and a footnote so that it covers every figure option.
  - Any other figure opens a new **Figure options** dialog, filled in with how the figure is drawn now.
  - The user never chooses between the two dialogs.
- **A result's figure becomes a new object** (`fit_graph`): re-estimating the result must not lose the edit, and
  editing must not change the result. A workspace figure is edited in place.
- **Replayable:** the step's code is `figure_of(source, tab).with_options(...)`, with every option written out.
  - Writing every option makes clearing one (removing shading, say) replay too.
  - `figure_of` returns a figure as it is, so the same code works whatever the source is.
- **Code that doesn't depend on the session:** a command's code cannot look at the session. So a Shade entry that
  is a plain name (`recession`) means a series of the dataset, and anything with a digit means periods.
- **Prefilling round-trips:** accepting the dialog unchanged gives an identical figure (tested). `shade_text`
  writes spans back in the form the parser reads.

### More kinds of series plot (24 October 2026)

- **Why:** the user asked for vertical, horizontal and stacked bars, and more. Looking into it showed that bars for
  several series were drawn at the same positions, each hiding the ones behind it. Areas for several series simply
  overlapped.
- **What:**
  - **Kinds:** `kind` gains `step`, `barh` and `box`.
  - **Stacking:** `stacked` stacks bars and areas, positives above zero and negatives below, the convention for
    contributions to growth.
  - **Right axis:** `right` draws named series against a second scale.
  - **Transforms:** `transform` gives an index (first period shown = 100), a change, growth on the previous period,
    growth on a year earlier, or the log.
  - **Zero line:** added automatically where bars, areas or changes cross zero.
- **Stacking is an option, not a kind.** It applies to two kinds and two orientations. The form shows the tick box
  only for those kinds, and the script reads `kind='bar', stacked=True`.
- **Transforms belong in the plot,** although they are expressible as derived series. Plotting growth or an index
  is the commonest chart in macro work. Making a new series for each chart clutters the dataset.
  - Changes are computed before the span is applied, so the first period shown has its value.
  - The axis label names the transform.
- **Horizontal bars use category positions,** not dates: periods (or a cross-section's entities) down the side,
  thinned to about 30 labels, the first at the top.
- **Right axis:** one per panel, through matplotlib's twin axes, with a combined legend. It is refused with
  stacking or one panel per series, where a second scale would mislead or mean nothing.
- **Shading** (added the same day) belongs to the figure, not to series plots, so any figure can have it: a fit,
  a forecast, impulse responses over dates.
  - It is a list of (start, end, label) periods, from text (`"2008Q3:2009Q2=GFC"`) or from a 0/1 series in the
    dataset.
  - Each span is drawn from the start of its first period to the end of its last, behind the data, in dated panels
    only.
- **Events, reference lines, log scales, value limits, one scale across panels, and values on bars** (also the
  same day) are figure options too, so they reach every figure and the dialog through one mechanism.
  - **Event and reference lines** are labelled in small type on a light box at the panel's edge, so a line passing
    through does not hide the label.
  - **Bar values** use one number format per chart, so that 4.3 is not shown beside 0.25, and white inside stacked
    segments.
  - **Log axes** are labelled in plain numbers at 1, 2 and 5 steps, not in powers of ten, as economists read them.
- **Not done:**
  - pie charts (rarely right for economic data);
  - dot plots and high-low charts;
  - heatmaps, scatter labels and colouring, seasonal subseries plots and chart-pack layouts (proposed next).

### An MCP server for AI assistants, stage 1 (18 October 2026)

- **Why:** the AI note's principle is that an AI proposes commands that run through the session. MCP is the
  standard way to give any assistant tools, so a server is that principle's first form, with no AI inside
  pymetrix. The design is in `docs/notes/mcp-server.md`.
- **General tools, not one per command:**
  - about 85 commands as separate tools would swamp most clients and duplicate the registry's schemas;
  - `list_commands`, `describe_command` and `run_command` expose the registry instead;
  - `form_spec` already carries each field's kind, choices and help.
- **Settings as the dialogs take them:** `run_command` builds the registry's command class, so validation, history,
  cautions and generated code are the session's own.
  - The one adaptation is coercing JSON values to the form's field types. Assistants send `2` and `["c", "i"]`
    where dialogs take `"2"` and `"c, i"`.
- **Bounded output:**
  - tables are cut to their first and last 15 rows past 40;
  - charts come back as PNG images;
  - `show_object` returns one tab at a time, listing the others, so a large result never floods the context.
- **Policy:** closed by default.
  - Python execution, file writes and network data each need a flag, and file paths are confined to a root folder.
  - The client's approval of each tool call is the human in the loop the AI note asks for.
  - The policy is enforced in the server, not left to the assistant.
- **Errors** are `ToolError`s carrying pymetrix's message, plus close matches for misspelt names. Without that, the
  SDK would hide the text of other exceptions, and the assistant couldn't correct itself.
- **stdio safety:** standard output carries the protocol, so every tool runs with stdout redirected to stderr.
- **SDK version:** written against MCP Python SDK 2.3. Version 2 renamed `FastMCP` to `MCPServer`, so the floor is
  2.3; CI's oldest-versions job installs 2.3.0. The SDK is optional, imported lazily, and blocked in the no-extras
  job.
- **Next:**
  - stage 2: prompts from the course labs, opening projects (with project trust), and a redaction setting;
  - stage 3: connecting to the running desktop app.

### The course pack in Jupyter: start page and reference cells (17 October 2026)

- **Why:**
  - JupyterLab opens `.md` files in its text editor, so the pack's README showed as plain text.
  - Students had no list of the pymetrix functions each lab uses.
- **Start page:** `00_start_here.ipynb` is generated from the README, since notebooks open formatted. It isn't
  maintained by hand: a test compares it with what the README generates. The help lists both, the README as a note
  and the start page as a notebook. That small duplication is accepted so that GitHub still shows the README.
- **Front matter in metadata:** `load_course` already read `metadata.pymetrix`. Moving the labs' front matter there
  removes the `---` block from their first cells.
- **Reference cells:** found by tracing, not by reading the code.
  - Reading the code can't tell what `bts["AR(1)"].compare` is. A profile hook, recording calls whose caller is a
    cell, gets the exact function, including methods on results and classes made through generated dataclass
    `__init__`s.
  - Tracing costs a second run of the notebook, so it happens only when writing the cell. The check re-renders from
    the stored `module:qualname` entries, which is cheap. It catches docstring and signature changes, renamed or
    removed names, and new imports. A method newly called but not imported is caught only when the cell is next
    rewritten.
  - Each entry shows the argument names (no defaults or types) and the first sentence of the docstring, so the
    table stays readable; `name?` gives the rest.
  - An entry with no docstring fails the check. That turned up 12 undocumented public names.
- **Outputs:** the labs are committed without outputs, and a test enforces it, because a student or instructor who
  runs a lab and saves it would otherwise commit its outputs. That happened once while this work was being done.
- **Public helpers:** `write_api_reference` and `check_api_reference` are public, in `pymetrix.testing` beside
  `check_course`, so instructors can use them on their own labs.

### A guide to wrapping libraries (17 October 2026)

- **Why:** the rules for wrappers were spread across adding-features.md (principles) and the Extension API section
  (the contract and the form rules), with no single worked template. The user asked for the general form to be
  documented.
- **What:** `docs/wrappers.md`, built around a working example, `examples/custom_wrapper.py`, rather than snippets.
  The example is tested, so the guide cannot drift from the API. It describes each wrapper as six parts:
  settings, lazy import, data in, the call, results out, registration.
- **Example choice:** both halves use statsmodels, which is a core dependency, so the example runs everywhere.
  - Huber robust regression (RLM): an estimation method, the commonest shape.
  - The Theta method (ThetaModel): a forecasting model. It shows the smallest interface backtests need (`spec`,
    `dependent`, `fit(data, span)`, `forecast(data, span)` with `se`), plus a menu command under `plugin_context`.
- **Test isolation:** the test restores the registries, the plugin list and the command tables after importing the
  example, as the custom-estimator test does.
- **Found on the way:** `adapters.EXTRAS` had no entry for scikit-learn, so its missing-library message named a core
  dependency and `pip install sklearn`. It is fixed, and a test now ties every bundled plugin's `requires` to an
  extra.

### A forecasting course pack (17 October 2026)

- **Why:** pymetrix now covers most of Elliott and Timmermann's *Economic Forecasting*. A teaching pack makes that
  usable in a course, and gives a reviewer of the forecasting tools one place to start.
- **Form:** `examples/forecasting/`. It is a course folder like `examples/course`, with its own course name, so
  **Help → Course material…** loads it and `check_course` checks it. It holds:
  - six notebooks;
  - a README with the lab list and a coverage map: each chapter, what pymetrix has, which lab uses it, and what is
    not covered yet.
- **Design of each lab:**
  - **Data:** simulated from a model the lab states, with a fixed seed, so the answers are known. For example:
    - the AR(1) is the true model in F2;
    - indicators are noisy measures of one factor in F3, the textbook case for combining;
    - the volatility doubles in 2008 in F4;
    - two factors drive 30 indicators in F5;
    - a mean break and a SETAR in F6.
  - **Checks:** at least one "check it yourself" cell recomputes a pymetrix number by another route, with an assert.
  - **Questions:** the closing questions send students to the API and to the menu items that do the same thing.
- **Seeds:** chosen so that each lab's lesson shows clearly in its draw (in F4 the pre-break coverage is about 10%).
  The tests assert the lessons, so a change to an estimator that alters them fails visibly. The labs invite
  students to change the seed and see which conclusions survive.
- **Runtime:** about 45 seconds in all, 23 of them for F5. Ridge and lasso in F5 choose their penalty once, by
  cross-validation on the data before the first origin, and keep it. Choosing it at every origin, as a backtest of
  `Ridge()` does, takes about a minute. F2's SPA, MCS and StepM use 500 bootstrap replications.
- **Optional packages:** F2's multiple-comparison cells need arch, and F5's ridge and lasso need scikit-learn. Each
  sits in a try/except ImportError that prints why it was skipped, so the no-extras environment runs every lab.
- **Diffusion index:** F5's diffusion index is a small class, `(data, origin) -> fit` with `forecast(data, span)`.
  It re-estimates the principal components at each origin. That makes it a real-time forecast, and it shows how
  any procedure can be backtested.
- **Limits it exposes:**
  - dynamic forecasts from equations with lags have no standard errors, so the density labs work one step ahead;
  - the lasso solver warns at the smallest penalties of its cross-validation grid (tolerance 1e-10), and F5
    silences those warnings.

### The scikit-learn plugin: shrinkage and machine learning (17 October 2026)

- **Shape:** these are estimation methods for equations, not models of their own, so formulas, lags, samples, the Estimate dialog and backtests apply unchanged. Linear methods return `FittedEquation`, whose coefficients act on the regressors as written (a cov of None: no standard errors; forecasts use the regression's sd). Nonlinear learners return `FittedLearner`.
- **Shrinkage:** regressors are standardized (population sd), the intercept is fitted unpenalized, and coefficients go back to the original scale. The objectives are scikit-learn's: ridge |y - Zb|^2 + a|b|^2; lasso and elastic net |r|^2/2n + a(l1|b|_1 + (1 - l1)|b|^2/2).
  - **Penalty choice:** time-series cross-validation, `TimeSeriesSplit` with `folds` expanding windows. Each window standardizes on its own training rows, then the smallest mean squared error over the validation blocks wins.
  - **Grids:** ridge n x logspace(-5, 2, 36); lasso and elastic net 40 points from max|Z'(y - ybar)|/(n l1) down by 1e-4.
  - **Reporting:** the chosen penalty is recorded (stats and provenance `resolved`), with the CV path in `extra`.
- **Bagging:** Inoue and Kilian (2008). For each of `reps` moving-block bootstrap samples (block n^(1/3) unless given), run OLS, keep the regressors with |t| > `critical` (the constant always stays), run OLS on those, and average the coefficients (zero when dropped). It is linear, so the averaged coefficients give the bagged predictions. Inclusion shares go in `extra`; it needs only NumPy and registers under core.
- **Learners:** `RandomForest`, `GradientBoosting` (boosted regression trees), `NeuralNetwork` (an MLP on standardized x and y, as a sieve) and `NearestNeighbours` (on standardized x). `FittedLearner` gives:
  - In-sample and cross-validated RMSE and R^2 (the same `TimeSeriesSplit`), with the out-of-fold errors.
  - Permutation importance.
  - Static predictions of the left-hand side.
  - Dynamic predictions of the target, period by period: later regressors and left-hand side use the earlier predictions, and a transformed left-hand side is solved for the target by a secant search, as equations do through the model solver.
  - The forecast se is the CV RMSE (times d target / d lhs when dynamic). It is missing beyond one step when the forecast feeds on itself.
- **Integration:**
  - `_refit` treats any object with `equation` and `method` like a fitted equation, so backtests re-estimate learners.
  - The Forecast and Backtest dialogs list `FittedLearner`. Views: Summary, Graph, Importance, Cross-validation, Residuals.
  - Packaging: the extra is `ml = ["scikit-learn>=1.3"]`, also in `all` and `dev`, with the oldest-versions pin 1.3.0. The gallery adds a learner only when scikit-learn is installed, and the window-coverage test excuses `FittedLearner` otherwise.
- **Checks** (tests/test_sklearn_plugin.py):
  - Ridge against (Z'Z + aI)^-1 Z'y back-transformed (1e-8). The lasso and elastic net meet their KKT conditions (1e-7), and the lasso zeroes coefficients.
  - The CV penalty and RMSE redone by hand with closed-form ridge in each window. Bagging redone with statsmodels' t-statistics and the same bootstrap draws.
  - Boosted trees' fit and CV RMSE against scikit-learn called directly.
  - Dynamic forecasts by hand, for c ~ c[-1] + y and for dlog(c) solved as c_(t-1) exp(prediction), with se = CV RMSE x c.
  - A backtest row against a refit at its origin; the dialog's code replays.

### Elliott and Timmermann phase 1: losses, binary forecasts, StepM, exponential smoothing, conditional forecasts, model averaging, threshold AR, factor criteria (17 October 2026)

- **Scope:** done unattended for review. The parts of phase 1 that need no new dependency; scikit-learn would be a download and a new optional dependency, so the user decides it.
- **Asymmetric losses** (`stats/evaluation.py`): `linlin_loss(e, tau)` is tau e for e >= 0 and (tau - 1) e otherwise, with e = outcome - forecast. `linex_loss(e, a)` is exp(a e) - a e - 1. `Backtest.losses` and `compare` take `"lin-lin"` (`tau`) and `"linex"` (`a`); the dialogs' advanced tab has `tau` and `linex_a`.
- **Binary forecasts:**
  - `brier_score`, `binary_log_score`.
  - `roc_auc`: the Mann-Whitney U over n1 n0, from ranks, with ties counted half.
  - `roc_curve`: thresholds at the distinct forecasts, events forecast at or above them.
  - `kuipers_score`: the hit rate less the false-alarm rate.
  - `binary_scores`: a table, with the base rate's Brier score as the benchmark.
  - `pesaran_timmermann`: the 1992 statistic with its exact variance terms (including the 4 py pf (1 - py)(1 - pf) / n^2 term), one-sided normal.
  - Backtests of probability forecasts are not built: Logit and Probit have `predict`, not `forecast`.
- **StepM:** wraps arch's `StepM` (stationary bootstrap, studentized), returning a table of mean loss differences and which models are superior, at the familywise level `alpha`.
- **Exponential smoothing** (`pymetrix/ets.py`):
  - `ExponentialSmoothing(series, trend, damped, seasonal, error)` wraps statsmodels' `ETSModel` (maximum likelihood, estimated initial states). Seasons are 4 or 12, from the frequency. The series must be unbroken; multiplicative parts need positive values.
  - `FittedExponentialSmoothing` gives `params_table`/`summary` (alpha, beta, gamma, phi named), `fitted`, `resid`, `states` (level, trend, seasonal), `plot`, and `forecast(data, span)` after the sample. Its columns are forecast, se, lower, upper and actual, as for equations; standard errors come from `forecast_variance` (analytic for additive models, simulated otherwise).
  - Views: Summary, Graph, States, Residuals.
  - Backtests re-estimate any fitted object with a `spec` that has `fit` (a new branch of `_refit`), from the fit's own start or on a rolling window.
  - Dialog: `FitExponentialSmoothing` (`pymetrix:fit_exponential_smoothing`). Forecast and Backtest list exponential smoothing fits.
- **Dialogs:** Combine forecasts and Compare many forecasts take `Widget("object", source="Backtest", multiple=True)` (comma-separated, as before), not typed names.
- **Checks:**
  - Losses: the check loss's optimal constant on 200,000 normal draws is the 0.8 quantile, and linex's is the mean plus a sd^2 / 2, both found numerically.
  - Binary forecasts: the AUC against brute-force pair counts and the trapezoid area under `roc_curve`.
  - Pesaran-Timmermann: its square within 1% of the Pearson chi-squared of the 2 x 2 table (the asymptotic equivalence) on 3,000 observations, and its size over 400 simulations.
  - StepM: picks the clearly better model and not a copy of the benchmark.
  - Exponential smoothing: simple smoothing's one-step fit and flat forecast from the level recursion by hand, and its variance ratios equal to 1 + (h - 1) alpha^2 (Hyndman et al.). The damped trend's forecasts follow from the error-correction recursion by hand, and Holt-Winters' from the final states. Backtest rows equal a fit to each origin from the fit's own start; a rolling window, the last 40 periods.
- **Conditional forecasts** (section 9.5; task 3, first part):
  - `FittedVAR.conditional_forecast(data, span, conditions)` follows Waggoner and Zha (1999). Write the future path as point + M e, with M block lower-triangular in Psi_(i-j) P (P the Cholesky factor of sigma) and e ~ N(0, I). With conditions R y = r, D = R M, the shocks are e* = D'(DD')^-1 (r - R point); the mean is point + M e*, and the covariance is M (I - D'(DD')^-1 D) M'.
  - The mean is E[y | conditions] under Gaussian shocks, so it does not depend on P; a test checks this by reordering the variables. Conditioned cells get exactly their value and a zero sd (rounding left about 1e-7).
  - Conditions that are not linearly independent are refused (rank of DD').
  - `FittedBVAR.conditional_forecast` does the same for each posterior draw, adding the free shocks (I - D'(DD')^-1 D) z, z ~ N(0, I), and summarises like `forecast`. The posterior is not updated by the conditions (the usual shortcut; documented).
  - The point path is now `var._point_path`, shared with the Bayesian VAR.
  - Dialog: `ConditionalForecast` (`pymetrix:conditional_forecast`, Forecast menu), with the scenario syntax parsed by `_paths`.
  - **Checks** (tests/test_conditional_forecast.py):
    - One condition in the first period against the regression formula from sigma (rel 1e-10), its sd too.
    - The mean and sd of three conditions over four quarters against an OLS regression on 200,000 paths simulated in the test (within 2% of the sd).
    - Invariance to variable order; conditions on everything give the conditions with zero sd; errors for unknown variables, periods outside the span and no conditions.
    - Bayesian VARs: exact conditions in every draw, reproducible with a seed, and with every cell conditioned the formula of the VAR at that draw's parameters.
    - The dialog's code replays.
  - **Not done:** ranges ("between 3 and 4%"), structural scenarios moving only chosen shocks, and conditions recorded in forecast rounds.
- **Bayesian model averaging** (section 14.6; `pymetrix/bma.py`):
  - **Model:** `BMA(outcome, candidates, always=(), g=0)` enumerates all 2^K subsets (K <= 15). The intercept and `always` are flat-prior terms, partialled out (Frisch-Waugh) before each model's R^2.
  - **Model probabilities:** log p(y | M) = ((n - q - k)/2) log(1 + g) - ((n - q)/2) log(1 + g (1 - R^2)) (Liang et al. 2008), with q the flat terms. Model priors are equal; g defaults to max(n, K^2) (Fernandez, Ley and Steel 2001).
  - **Coefficients:** within a model they are (g/(1+g)) times least squares, with Student-t posteriors on n - q degrees of freedom and scale (g/(1+g)) SST (1 - (g/(1+g)) R^2)/(n - q) (X'X)^-1. Averages and sds mix over models (0 where excluded). Inclusion probabilities sum the posteriors of the models that include a regressor.
  - **Forecasts:** regressors known in the span (like an equation's static forecasts). Each model's predictive mean has the flat terms re-estimated given the shrunk candidates; its variance is s^2 (1 + leverage) df/(df - 2), with the candidates' leverage shrunk by g/(1+g). The mixture's mean and sd give a normal approximation for intervals and scores.
  - **Views:** Summary, Graph (inclusion probabilities), Models (the top ten). Dialog: `FitBMA` (`pymetrix:fit_bma`). Forecast and Backtest accept it.
  - **Backtests:** the target can be an expression such as dlog(c), so backtests now take outcomes by evaluating the target on the latest data when it is not a series (`forecasting._outcomes`).
  - **Checks** (tests/test_bma.py):
    - The Bayes factor against the definition: alpha and sigma integrated analytically, beta by 1-D quadrature, agreeing to 1e-10 (2e-15 in practice). 3-D quadrature over all three was only good to about 2e-4, too loose to test with.
    - Inclusion probabilities and averaged coefficients against every subset fitted by statsmodels' OLS.
    - Partialling out against the R^2 of x2 after the intercept and x1.
    - Forecasts by hand as the posterior-weighted model forecasts.
    - A backtest row against a fit to its origin, with an expression target's outcome.
    - Errors, and the dialog's replay.
- **Number of factors** (section 10.3): `FittedPCA.factor_criteria(kmax)` computes Bai and Ng's IC_p1-3 from the stored eigenvalues. V(k) is (1/N) times the sum of the eigenvalues of Z'Z/T beyond k, which is the average squared residual of the standardized series on the first k components. **Checks:** V(k) against that regression done directly; the penalties by hand; and on simulated 2- and 4-factor panels (N = 50, T = 200), IC_p1 and IC_p2 pick the true number.
- **Threshold autoregressions** (chapter 8; `pymetrix/nonlinear.py`):
  - `ThresholdAR(series, lags, delay=0, trim=0.15)` is a two-regime SETAR. Conditional least squares runs over each delay (1 to the lag order, or the one given) and each threshold among the switching variable's values between the trim quantiles, with at least `trim` of the observations in each regime. The error variance is common to both regimes.
  - As for equations, the sample is the periods of y and the lags may come from earlier data. Refits in backtests then use the recorded sample consistently; the first version dropped lags inside the span, so refits drifted.
  - Standard errors treat the threshold and delay as known.
  - `linearity_test(reps)`: Hansen's sup-F, n (SSR_AR - SSR_TAR)/SSR_TAR, with the fixed-regressor bootstrap (y* = e u, u ~ N(0, 1), the same search for each replication). The window's Tests tab uses 199 replications.
  - `terasvirta(data, series, lags, delay)`: the LM3 F test, the AR residuals on the lags and the lags times q, q^2 and q^3.
  - Forecasts simulate the regimes with normal shocks (2,000 draws by default). `predictive_draws` has the Bayesian VAR's shape, so backtests score the draws.
  - Views: Summary, Graph (the series with the threshold), Tests. Dialog: `FitThresholdAR` (`pymetrix:fit_threshold_ar`); Forecast and Backtest accept it.
  - **Checks** (tests/test_threshold_ar.py):
    - The estimates, delay and threshold against a brute-force search fitted with statsmodels.
    - A threshold of 1.0 recovered within 0.1 on 3,000 simulated observations.
    - The sup-F from statsmodels' sums of squares; it rejects for a SETAR and not for a linear AR.
    - Terasvirta against statsmodels' `compare_f_test`.
    - The one-step simulated mean against the regime's exact prediction, with sd sigma; multi-step means against a separate simulation written in the test.
    - A backtest row against a refit at its origin; errors, and the dialog's replay.
- **CI:** the oldest-versions job (PySide6 6.6) failed on be0e1d8: `menuBar().actions()[0].menu()` returned an already-deleted QMenu in the save-result app test. The test now finds the actions with `findChildren(QAction)`.

### Forecast evaluation: Elliott and Timmermann chapters 14 to 18 (17 October 2026)

- **Why now:** a comparison with Elliott and Timmermann, *Economic Forecasting* (2016), found the model chapters well covered and evaluation, the book's emphasis, the thinnest. This closes chapters 14 (combination), 15 (efficiency), 16 (rolling windows), 17 (comparisons) and 18 (densities). Still missing from the book: asymmetric losses, Bayesian model averaging, binary-forecast scores, and the methods of chapters 5, 6, 8, 11, 13 and 19 (see known gaps).
- **Scoring rules** (`stats/density.py`): log score, CRPS and PIT for a normal (closed forms), a sample (CRPS in its energy form from the sorted draws, O(n log n); PIT with ties counted as half; log score from a Gaussian kernel density with Silverman's bandwidth) and a normal mixture (CRPS from Grimit et al. 2006). Berkowitz is the conditional-likelihood LR of an AR(1) with mean and variance on Phi^-1(PIT) against iid N(0, 1), 3 degrees of freedom.
- **Tests** (`stats/evaluation.py`):
  - Mincer-Zarnowitz: a Wald test of (0, 1) with a Newey-West sandwich, h - 1 lags.
  - Encompassing (HLN 1998): the DM statistic on (e1 - e2) e1, one-sided.
  - Giacomini-White: n Zbar' Omega^-1 Zbar, with instruments a constant and the differential h periods before (or the constant only); Omega is the outer product at h = 1, Newey-West beyond.
  - Clark-West: the adjusted MSPE differential, a Newey-West t, one-sided normal.
  - Christoffersen: the Kupiec LR plus the first-order Markov independence LR, with 0 log 0 = 0.
  - SPA and MCS wrap arch's `SPA` and `MCS` (stationary bootstrap; mean block length max(h, n^(1/3)) unless given). SPA's displayed statistic is the largest studentized mean improvement over the benchmark.
  - `mean_differential_test` is the old DM core, moved here unchanged and shared.
- **Backtests:** `_forecast_parts` returns means, standard deviations and (for fits with `predictive_draws`, the Bayesian VAR) the draws. `FittedBVAR.forecast` now summarises `predictive_draws`, which gives the same values as before. Equations' dynamic forecasts have no standard errors (stochastic simulation is planned); when the static and dynamic point forecasts agree (no lag of the target), the static standard errors are used. Otherwise there is no density, and the density methods say so. Rows add `sd`, `density` (normal, draws, pool), `log score`, `crps` and `pit`. `losses(variable, h, loss)` gives the series every comparison uses, with lower always better (the log score is negated). `window=` sets each origin's span to its last `window` periods.
- **Combination:** members are matched on (origin, target, horizon, variable). For each variable and horizon, the weights at an origin come from rows whose target is at or before that origin (outcomes known then), and are equal until `min_history`. The methods are inverse MSE, and constrained least squares through its KKT system. With non-negative weights and standard deviations for every member, the density is the linear pool, with its sd and scores. The result is a `Backtest` with `weights`, so every evaluation applies to it.
- **App:** the Backtest window adds Density scores, PIT, Tests and Weights tabs. There are three new dialogs (`pymetrix:combine_forecasts`, `pymetrix:compare_backtests`, `pymetrix:compare_many_forecasts`), and RunBacktest gains a `window` field (default 0, so older projects need no migration). Lists of backtests are typed as comma-separated names, as variables are in Compare rounds.
- **Checks** (tests/test_forecast_evaluation.py):
  - Scoring rules: the normal and mixture CRPS against numerical integration of the definition (1e-7); the sample CRPS against all n^2 pairs and, for 200,000 normal draws, the normal CRPS; the log score and PIT against scipy.
  - Berkowitz: equal to 2(llf - ll0) from statsmodels' OLS, and its size within 2 to 9% over 400 simulations.
  - Christoffersen: against hand counts of the transitions.
  - Mincer-Zarnowitz: against statsmodels' HAC Wald test. Clark-West: against statsmodels' HAC t on the adjusted series. Encompassing: by hand. Giacomini-White: equal to n times the uncentred R-squared of 1 on Z.
  - SPA and MCS: find a clearly best forecast.
  - Backtests: a VAR row's CRPS against `crps_normal` with the VAR's own standard error; a Bayesian VAR row against `crps_sample` on its own draws; the equation row's sd against the static forecast.
  - Old DM numbers are unchanged. A rolling window equals a fit on the last N periods.
  - Combination weights by hand: inverse MSE, and regression by substitution (y - f2 on f1 - f2), a different algorithm from the KKT solve. Pooled CRPS and sd by formula.
  - The dialogs' recorded code replays to the same results.

### Saving a single result (17 October 2026)

- **Decision: a record of the views, not the object.** The table above planned a folder per fitted object (spec, params, provenance, Parquet), which means a serializer for each of the 54 kinds and keeping them all in step. Pickle is ruled out for persistence (it breaks between versions and runs code on load), and a file sent to a colleague must be safe to open. Every kind already reduces to `view_sections`: tables and declarative figures. So `pymetrix.saved` writes those, which covers every kind, plugins' included, at once. The cost: a saved result cannot be forecast from or estimated again. Projects (or the producer command) do that.
- **Format** (`.pymxr`, a zip, format version 1): `manifest.json` (format and pymetrix versions, saved time, kind, name, views, any not saved), `summary.txt` (`views.summary_text`), `views/<n>.json` and `provenance.json` (`to_jsonable` of `obj.provenance`). A table is its columns, rows, row labels, title, notes and index name; values JSON can't hold (dates, periods) are written as text, NumPy scalars as numbers, and missing values as `NaN`. A figure is its options and panels; each layer (Line, Band, Bars, Scatter, Histogram, Rule) is its fields, and each series its values with its index (period with frequency, datetime, number, text, or multi-level). It reopens as a real `Figure`: redrawn, themable, `data()`. A view or layer of another type is listed under `not_saved`; a result with no view left is refused. A newer format version is refused with a message.
- **`SavedResult`**: kind, name, saved, package_version, views, text, provenance, source. `summary()` gives the text; its sections are the saved views plus a Saved tab (what, when, which version, which file); `to_frame(view)` and notebook display come from `attach_view_methods`. Saving one again writes its views as read, without the Saved tab.
- **App:** File → Save result… (not a step: like Export script, replaying it would write files) takes the active window's object, else the Objects selection; a dataset gets a pointer to its project or `write`. File → Open result… is a dialog command, `OpenResult(path, name)` (`name = load_result(path)` in the script), so it replays without trust. The session keeps the bytes it read (`Session.opened`), and projects store them as `opened/<n>.pymxr`. A reopened project reads its copies, not the files (plain JSON: no trust needed). `kind_of` says "Saved fitted equation".
- **Checks** (tests/test_saved_results.py): all 52 kinds from the gallery and demos (28 figures, 175 tables), a saved result among them are saved and read back. Every table shows identically and every cell comes back exactly (floats bit for bit, text as written); every figure gives the same `data()` frame, panel titles and layer types; summaries are equal; the archive holds only .json and .txt; a second save of the saved result keeps its views. Also: a saved figure redraws in another theme, `to_frame` matches the original's, bad and newer files are refused, and an Open result step survives saving a project, deleting the file and reopening it untrusted. App: test_ui_demo_polish saves the active window, opens it as a step with the Saved tab, and declines a dataset.

### X-13 failures as data errors (16 October 2026)

- `X13.decompose` catches statsmodels' `X13Error` (an `Exception`, not a `ValueError`, so `Dataset._recompute` let it through) and raises `DataError("X-13ARIMA-SEATS could not adjust {name}: {message}")`, the program's message on one line without its "ERROR:" prefix. Derived series then add their own "could not be recomputed" context, as for other recipes. **Checks:** a test with a stand-in that raises statsmodels' error (runs everywhere), and one with the real program on a series without noise, which X-13 refuses ("...mean square error of the residuals is zero").

### Small items: recent projects, Copy summary, a Newton step (16 October 2026)

- **Recent projects:** File > Open recent lists up to eight projects, newest first. A project is added when it is opened or saved (`remember`) and dropped when the file is gone; Clear recent empties the list. They are stored in the app's settings (`recent_projects`). `open_project(path=None)` opens a given path without the file dialog.
- **Copy summary:** a corner button beside Copy producer command. It copies `views.summary_text(obj)`: `str(obj.summary())`, or the first table view for a result without `summary()`. test_result_views checks it for every kind of result.
- **Test settings:** the app built its `QSettings("pymetrix", "desktop")`, which always uses the native store (the Windows registry), so tests wrote window layouts and now recent projects into the user's own settings. The app now passes `QSettings.defaultFormat()` explicitly. That is native for users, and conftest's `pytest_configure` sets it to INI files in a temporary folder.
- **Newton step** (`linear._newton_step`): one exact step from the starting values, kept when the residuals are within 1e-10. Otherwise the hybr/lm root search runs as before, from the same start. On Smets-Wouters a log posterior takes 16.7 ms against 19.9 ms (the difference is at most 4.5e-13), and the Dynare steady-state and decision-rule checks still pass at 1e-12. A test checks that a linear model skips the root search and a nonlinear one does not, with equal answers.
- **X-13 in provenance** (done the same day, with x13as 1.1 Build 62 from the Census Bureau kept in the untracked `.tools/x13as`): `seasonal.x13_notes` reads the version from the program's banner ("Version Number 1.1 Build 62"), the model from the output's "ARIMA Model:" line (else "Final automatic model choice"), and the outliers from "Automatically Identified Outliers" up to the rule below them, "none" when the spec echo shows `outlier{}` and none were listed. `X13.decompose(notes=)` fills them with `x13_program`; `seasonal_components(notes=)` merges a panel's runs, naming each entity only for a note that differs. Derived series get them through a new optional recipe method, `compute_noted(data, extra)`, which `Dataset._recompute` uses in place of `compute`: the x13_ notes are replaced on each run, so a revision that removes an outlier removes it from the notes, and switching to STL drops them. The data dictionary (Variables view) shows a variable's text and number `extra` values. **Check:** test_seasonal runs x13as directly on a spec written by the test (log, automdl, outlier, x11 saving D11), without statsmodels: model, version and an injected January 2010 outlier agree, and the adjusted series equals D11 to 1e-5; a panel test, a recompute test and a parser test on fixed text (which runs without the program). CI has no x13as, so the program tests skip there.

### Every result from Python: to_frame(view) and notebook display (14 October 2026)

- **Probe across the 54 kinds** (summary, plot, HTML display, to_frame): nothing failed, but coverage was uneven. `to_frame` existed on 8 kinds, and 13 had no notebook display, Dataset among them.
- **One definition, attached** (`views.attach_view_methods`, run for every registered type when pymetrix is imported, and by `register_view` for plugins):
  - `to_frame(view=None)` uses `view_sections`. With no name it gives the first table; otherwise a case-insensitive title, with a `NotFoundError` that lists the views, or says a figure is a figure.
  - `_repr_html_` gives the first table view, or for Dataset types the one-line `dataset_status` plus pandas' display.
  - Types with their own methods keep them. Figure and Table are left out: a figure's numbers are `.data()`, and a table is its own view.
  - `IRF`, `Posterior` and `Results` keep their default and pass a view name to the shared version. `FittedEquation.to_frame` uses it too.
  - Dataset's `to_frame` keeps its bridge meaning (series names).
  - Loading `pymetrix.views` at import costs about 12 ms.
- **Checks:** test_result_views (every table view of every result kind at any case, figures refused clearly, every display), runnable without Qt.

### Every kind of result opens in a window (14 October 2026)

- **Why:** each look at the app found something real, but only demo results were ever opened in tests: 15 of the 54 kinds registered in `views.sections`.
- **tests/_gallery.py:** a builder for each of the other kinds, from the public API on example or small synthetic data. It covers models, estimators, nowcasting (MIDAS, bridge, dynamic factor, news, disaggregation), model tools (check, calibration, solutions, systems), sweeps and scenarios, selections and loops, and plain values. `tests/test_result_windows.py` builds every object's window, tab by tab, figures drawn. `test_every_kind_with_a_window_is_covered` fails when a registered kind has no example.
- **Found and fixed:**
  - `FittedDisaggregation` could not be opened. `views._describe` treated `low_index` (a PeriodIndex) as a named collection, because it has `.names`, and joined `[None]`. Indexes are now described by length and span.
  - Empty tables showed a blank grid; they now show "Nothing to show: this table has no rows."
  - `_fit_height` measured only on Show, before the final width, so it reserved room for a horizontal scroll bar under most short tables. It now re-measures on Resize.
- **Screenshots** (.tools/ui_audit4.py, local) of the first two tabs of every gallery object were reviewed. The rest read well.

### Readability, third pass: panes, spreadsheet, editors (14 October 2026)

- **Screenshots** (.tools/ui_audit3.py, local): the start page, Script, Problems with results out of date, History, the spreadsheet, a code editor, the objects' right-click menu and the course dialog.
- **Fixed:**
  - The Script pane used the UI font; it now uses `QFontDatabase.systemFont(FixedFont)`, as `CodeEditor` does.
  - `sheet_qt._fmt` used six significant figures; it now uses `output._fmt` below 1e6. The sheet's Copy takes the EditRole (full precision), not the display text, so changing the display could not change what is copied.
  - `open_sheet` and `open_editor` call `fit_in_workspace`. Its cascade step counts every subwindow, not only result windows.
  - The History and Problems empty states use `setFirstColumnSpanned`, and History's Code column stretches.
  - The Problems empty text said "No problems_view:". The October rename of the `problems` variable had reached a string.
- **Seen and left as they are:** the objects' right-click menu and the course dialog read well. The start page's subwindow sits top left of an empty workspace, which is fine for a start page.
- **Checks:** `test_panes_and_editors_read_consistently` covers the Script font, the sheet format, cascading and the empty states.

### Readability, second pass: dialogs, graphs, help (14 October 2026)

- **A crash found on the way.** `SeasonalAdjust`'s method settings include `log: bool | None` declared as a choice with no `Literal`, so `_Editor` iterated `None`, and the dialog failed in every project.
  - `form_spec` now gives any `bool | None` field the choices (None, True, False), shown as automatic, yes and no.
  - `_Editor` tolerates a choice field without options.
  - `test_every_menu_dialog_opens` builds every dialog with a menu entry in an empty project, the consumption demo and the policy demo. The six commands without menus are recorded by other windows.
- **Dialogs:**
  - The form side is at least 420 px wide.
  - Spin boxes show the help's "0 = …" in words (`_zero_means`, via `setSpecialValueText`).
  - Dynamic dropdowns show a blank option as "(empty text)" or "(none)", and `get`/`set` map it back to "".
  - `CHOICE_LABELS` names the report template's and theme's blanks.
- **Graphs:**
  - Panels whose lines, bands and bars all have integer indexes use `MaxNLocator(integer=True, steps=[1, 2, 5, 10])`.
  - Sweep tabs and panels say "y, shock to ey", like the impulse-response panels. VAR frame columns keep "y <- ey": they are data, not labels.
  - The model's identification label is "rational-expectations solution".
  - Chains panels have the x label "draw".
- **Help:** the contents give each test's name from the first words of its docstring, or `_TEST_NAMES` for the nine without one, followed by the registered code.

### Demo polish: UI suggestions assessed (11 October 2026)

ChatGPT's demo-polish spec was checked against the code. Much already existed:

- `Selection` with DeriveEach, FitEach and TestEach;
- Add results to dataset;
- the Script pane, plus History (copy code, re-run, reopen dialog);
- out-of-date marking with Recompute;
- Excel paste recorded as an edit on Apply;
- Copy for Excel on tables.

**Adopted and built:**

- **Table renderers:**
  - `Table.to_csv(sep, header)` at full precision with blanks for missing values, and `Table.to_markdown(digits)`, now shared with the report writer.
  - Result windows add a Copy as menu: values only for Excel, Markdown, or the Python that gives the table (`view(fit, "Coefficients").to_frame()`).
  - Duplicate column names (a VAR's equations) are handled by position.
- **Fields tab** (`views.fields(obj, name, code)`): a result's named fields (dataclass fields, public attributes, properties) with type, size, a short value and the Python that reaches each; provenance expanded one level; methods and internal working objects left out; the producing command as a note. Results only, not data or tables. It is a UI tab, not a view, so reports and `titles()` are unchanged.
- **Zoom:** View → Zoom in, Zoom out, Actual size (Ctrl+=, Ctrl+-, Ctrl+0), from 70% to 200%, for the text and figures of the active window. It is view state, never saved in projects.
- **Text size** (6 November 2026): View → Text size (Smaller, Normal, Larger, Largest; 0.85, 1, 1.15, 1.3 of the font the app has at Normal, the system's) scales the application's font, so everything that inherits it follows; widgets that set a size of their own (fixed-width code fonts, a zoomed window, bold headers) are scaled by the same ratio, worked out before any change so a widget that sets its children's fonts from its own lands on the same sizes. `ui/text_size.py`. Kept in QSettings (`view/text_size`), applied before the main window builds anything, applied live from the menu (open result windows are drawn again so their tables and charts take it). Zoom multiplies it: `zoom.scale_font` keeps each widget's size at Normal text and actual size. Charts: `chart_look.look` puts the scale on the `Surface` (`text_scale`); `Figure.draw` scales the theme's two font sizes with it (`figure_style.scaled`); exported and copied files are drawn with `LIGHT` (scale 1), so they keep their sizes. Icons: `icons.view_size` is the style's small icon size times the factor, to even pixels (14, 16, 18, 20); the views that show the objects' icons (Objects panel, Find anything, the tabbed view's tabs) take it through `icons.size_icons` and `text_size.apply` sets it again (`icons.reapply`); the title bars paint the icon at that size, capped by the bar's icon room. No shortcut: the zoom has Ctrl+= / Ctrl++ / Ctrl+- / Ctrl+0, and Ctrl+Shift+= is Ctrl++ on many layouts.
- **Spacing tokens** (6 November 2026): `ui/metrics.py` holds the app's margins and spacing as named sizes on a 4-pixel grid (`hairline` 2, `tight` 4, `normal` 8, `section` 12, `indent` 12, `dialog_margin` 12), scaled with the text size and re-applied when it changes. Dialogs take `dialog_margin`; a section heading, a fold toggle (Advanced settings, Show the code) and a group box take `section` above; a page in the workspace (the Start window) `normal`. Qt's own spacing between a layout's widgets is left to the style. `tests/test_ui_spacing.py` finds any other bare number in a margin or spacing call in `pymetrix.ui`; the deliberate ones (the figure buttons' 3 pixels, a list entry's padding) are listed there.
- **Objects filter box:** matches name, type or status.
- **Data dictionary:** the Variables tab adds observations, first, last and gaps (missing values between first and last). `views.dataset_status(data)` gives the Data tab's line: series, frequency, sample, rows, kept up to date, missing values.
- **Demo projects:** `pymetrix.demo("forecast_round" | "consumption")` and File → Open demo project build a session by running the course's own commands, so its history, script and project replay are real.

**Declined, with reasons:**

- **`NamingSpec`, `ResolvedName`, `NamingPreview` and `BatchResult`:** `Selection.format` and `formulas` already do template naming. Commands record their resolved names in their code, so replay is already safe.
- **The increment conflict policy (`x_2`):** it silently creates names that scripts don't expect. Errors or explicit overwrite stay.
- **Project or user naming preferences:** hidden state changing a command's output.
- **Renaming the defaults:** this would break projects for no analytical gain.
- **Selection lineage helpers:** lineage lives on each series (`recipe`, `derived_from`).
- **A badge framework:** every command replays, and the AI badges are for features that don't exist. The status lines give the useful part.
- **Window pinning and a theme toggle:** windows are MDI children, and Qt follows the system theme.
- **Why-disabled tooltips:** dialogs open and say what is missing.

**Later:**

- a command palette;
- paste as new series, and paste into panel data;
- frequency conversion of a selection;
- compare with the previous result.

**Found on the way:** VAR results record no data fingerprint. This is part of next step 4, the results contract.

### CI: Qt teardown crash fixed on Linux (12 October 2026)

- **Problem:** the oldest-versions job (Linux, Python 3.11) crashed in the conftest Qt window clean-up after all tests passed. It happened on day 3, then twice in a row on the demo-polish commit, so re-running was no longer enough.
- **Fix:** on Linux CI with Qt loaded, the clean-up is skipped, and `pytest_unconfigure` (trylast) ends the process with `os._exit` and the session's exit status. On Linux this skips Qt's shutdown code.
- **Elsewhere:** Windows, macOS and local runs keep the clean-up. On Windows, skipping it brings back the crash in Python's final clean-up, and both `os._exit` and `TerminateProcess` crash because Qt's DLL shutdown still runs. This supersedes the earlier "re-run if it recurs" note.

### Closing the gap with IRIS and Dynare: revised next steps (12 October 2026)

**Assessment.** The solution methods are largely in place:

- compiled models and checks, steady state, linearisation;
- a first-order rational-expectations solution (QZ, Blanchard-Kahn) and perfect foresight;
- scenarios with set, shift, fix, hold and free;
- measurement maps, the Kalman filter and smoother, and maximum-likelihood or posterior-mode estimation;
- Dynare import.

The gap macro modellers will feel first is the forecasting and policy-analysis workflow of an IRIS or FPAS team:

- conditional forecasts with anticipated and unanticipated shocks;
- judgemental tunes in the filter;
- decompositions;
- models written in IRIS's language.

**Order** (replaces the earlier next steps from step 2 on):

1. Fold the new work into the demos.
2. **The forecasting core for linear models.**
   - Simulation plans: `Scenario(hold=, free=, unanticipated=)` on a first-order solution.
   - Forecasts from fitted models starting at the smoothed state.
   - Tunes on unobserved variables in the filter.
   - Historical and forecast decompositions, and contributions of observables.
   - Theoretical moments and variance decomposition.
3. **The IRIS `.model` bridge.** Log variables and log-linearisation, steady states with growth, and a fuller Dynare reader through Dynare's preprocessor JSON output, run as an optional external tool.
4. **Bayesian estimation.**
   - Random-walk Metropolis-Hastings, the marginal likelihood (Laplace and modified harmonic mean), prior-posterior plots, mode checks and Iskrev identification.
   - Validated against Dynare's `fs2000`.
5. **Nonlinear models and policy.** OccBin-style occasionally binding constraints (the zero lower bound), perfect foresight with expectation errors, second-order perturbation, optimal policy (discretion, commitment, optimal simple rules).
6. **Finish the results contract.**
7. **After collaborator feedback:** report templates, PDF, a command palette, parameter sweeps, and a large-model benchmark.

**Principles.**

- Import bridges and plain Python, not IRIS's object model or MATLAB syntax.
- No reimplementation of Dynare's preprocessor.
- Methods from published papers, never from IRIS or Dynare source.
- Validation against stored reference results generated once from Dynare under Octave, with versions recorded, so CI needs neither.

**Design for simulation plans (step 2, first piece).**

- **No new plan object.** `Scenario` gains `unanticipated`, the shocks that are surprises. All other shocks and exogenous paths are anticipated from the first period, as in IRIS's default.
- **Entry points:** `Model.solve(data, span, scenario, solver=FirstOrder())` linearises at the steady state with every exogenous variable as a shock. `LinearSolution.solve(data, span, scenario)` works on any solution, including a fitted model's.
- **Anticipated shocks** use the forward expansion `x(t) = P x(t-1) + Σ_k F^k Q a(t+k) + Q u(t)`, where `F = -(A0 P + A1)^{-1} A0`. It is computed by the backward recursion `w(t) = Q a(t) + F w(t+1)`.
- **Exogenize/endogenize:**
  - `hold` paths may be partial; missing values mean not held.
  - Each target is paired, in order, with a `free` shock that is solved in the target's periods.
  - Anticipated free shocks are known from the start; unanticipated ones arrive in their period.
  - Since the model is linear, the shocks come from one linear system of responses.
- **Independent checks:**
  - Fully anticipated plans against the stacked perfect-foresight solver on the same model.
  - Unanticipated shocks against impulse responses.
  - Unanticipated conditioning against a period-by-period closed form.

### Simulation plans built (12 October 2026)

- **API, as designed:**
  - `Scenario.unanticipated` names the surprise shocks; all other shocks are anticipated.
  - `LinearSolution.solve(data, span, scenario)` simulates any first-order solution.
  - `Model.solve(..., solver=FirstOrder())` linearises at the steady state, with every exogenous variable as a shock, then calls it.
  - Solver choices: Newton, Gauss-Seidel, Perfect foresight, First order.
  - The code is in `pymetrix.models.plans`, with the forward matrix `F` and the backward recursion for news.
- **Targets:**
  - Paths can be numbers, Series or `{period: value}` mappings; missing periods are free to move. The nonlinear solvers now accept mappings too.
  - Each target is paired in order with a free shock, solved in the target's periods by one linear system. A rank check explains impossible plans, such as a free shock that does not move its target.
- **Starting values:**
  - Taken from data in levels, but only for variables that enter lagged; aux lags `x_lagk` read `x` k periods further back.
  - Missing ones are named in the error.
  - With no data, the solution starts at the steady state.
  - `LinearSolution.endogenous` gives the model's own variables.
- **Solve dialog:** gains Target (e.g. `i=0.5` over the span), Free and Unanticipated fields. The code it records reproduces the result in a reopened project.
- **Checks** (tests/test\_plans.py):
  - Unanticipated shocks equal impulse responses (1e-12).
  - News matches stacked perfect foresight (5e-16).
  - Solved holding shocks, given to perfect foresight, hit the target (1e-9).
  - Unanticipated holds match a period-by-period closed form.
  - An AR(2) with a non-zero steady state matches the recursion by hand.
  - The nonlinear RBC model stays within 1% of the exact path for a small shock.
- **Not yet:** surprises in nonlinear models (perfect foresight with expectation errors); `fix` and `adjust` with first-order solutions (they raise and point to target/free); per-period targets in the dialog.

### Codespaces for collaborators (12 October 2026)

- **Why:** collaborators should see the desktop app, and many IMF and central-bank laptops cannot install software. A codespace needs only a browser.
- **`.devcontainer/`:**
  - **Base:** Microsoft's Python 3.12 dev container image, plus the `desktop-lite` feature (Fluxbox, TigerVNC and noVNC on port 6080; password `vscode`). Both are MIT-licensed, and neither is a pymetrix dependency.
  - **`setup.sh`:** installs Qt's system libraries and fonts, then `pip install -e .[all]`.
  - **`start-app.sh`:** waits for the display, then starts `pymetrix-gui --demo forecast_round --maximized` in the background.
  - **README:** what to click.
  - **Machine size:** a 4-core machine, so the free allowance gives about 30 hours a month per collaborator. Billing is to the person who creates the codespace.
- **`pymetrix-gui` command line:**
  - New options: a project file, `--demo NAME` and `--maximized`.
  - Qt's own single-dash options (`-platform offscreen`, `-style`, ...) are filtered out first. Otherwise a value like `offscreen` would be read as the project file, a bug the new test caught.
- **Line endings:** `.gitattributes` keeps `*.sh` LF on every checkout, since this machine converts to CRLF. A test checks the scripts and the configuration.
- **Not yet verified in a real codespace:** this needs the files pushed and a codespace created from the GitHub page. The local `gh` cannot create one without the `codespace` scope. Checked locally: script syntax, the JSON, and the app starting with `--demo` and `--maximized`.

### max and min in the model language (12 October 2026)

- **Meaning:**
  - `max(a, b, ...)` and `min(a, b, ...)` with two or more arguments give the larger or smaller value, period by period. This is what Dynare, IRIS and FINEX models write (`r = max(r_unc, r_ELB)`).
  - With one argument they remain the statistic over a series.
  - Grouped extremes are written `max(y, by=country)`. `by=` is now accepted by every group-wise function; positional grouping still works except for max and min.
  - This changes the earlier, untested and unused meaning of `max(x, group)`.
- **Representation:** nested binary `Call` nodes (`min(a, b, 0)` is `min(min(a, b), 0)`). `Call.elementwise` tells them apart from group-wise statistics, whose second argument is a `Level`. Printing writes `by=` so text round-trips.
- **Derivatives:**
  - A new primitive `step(x)` (Heaviside, 1 where x ≥ 0, missing stays missing).
  - d max(a, b) = step(a−b)·a′ + (1−step(a−b))·b′, and similarly for min. At a tie the first argument is used.
  - Newton's method therefore works on either side of a kink.
- **Evaluation:** `np.maximum` and `np.minimum`, so they work on floats, arrays and Series. Dynare parameter assignments may use max and min.
- **Checks** (tests/test\_maxmin.py):
  - Values and derivatives on each side of each kink, against central differences.
  - Data, and a grouped panel maximum.
  - A New Keynesian model with an effective lower bound under perfect foresight. The bound binds for three quarters, `i = max(iu, floor)` holds, and the path equals the piecewise-linear solution (the rate held at the floor by an anticipated wedge, solved by the first-order simulation plan) to 1e-8.
  - A Dynare file with max in the model block and in parameter values.
- **Behaviour to note:** a first-order solution sees only the side of a kink that holds at the steady state. Binding constraints need perfect foresight now, and OccBin-style solutions later. Nonlinear surprises (unanticipated shocks with a binding constraint) are part of next step 5.
- **The OccBin filter** (added 29 October 2026): Dynare 6's piecewise Kalman filter rather than a particle filter, since
  the model is linear within each regime. Each period's rule comes from the OccBin solver's backward recursion over the
  regimes expected from the filtered state (shared with `occbin` through `occbin._Piecewise`), with Q(1) the response to
  the period's shock. The regimes are checked from E[x(t-1) | data to t] and E[e(t) | data to t], which reproduce the
  filtered state exactly; checking from the predicted state instead misplaced them. Where the guesses cycle (data near
  the bound fit both readings), the regime under which the period is likelier is taken, and a rule exactly at its bound
  counts as binding (a rate observed at its floor). Observations the model fixes at a bound make the prediction-error
  variance singular: the density is that of the other directions, and the fixed ones must match. The smoother keeps
  the filter's regimes by default, and the constraint checked on the smoothed path is reported beside them. Estimation rebuilds the regimes at each trial value
  (`_Likelihood.build_occbin`), about 0.1 s a likelihood for the New Keynesian model over 200 periods.

- **Regimes with known dates** (added 30 October 2026, `models/regimes.py`): DSGE.jl's regime-switching parameters
  and temporary alternative policies, credible. A `Regime` is a start, parameter values and equations replaced by name,
  with `end` for a temporary one and `expected=False` for a surprise; regimes cumulate (each changes what it names), so
  a temporary regime's end restores the one before it. The regime in force before the first period (where the starting state comes from) is the reference: every
  other is linearised at its steady state with the constant its equations leave there (`linearize(constant=True)`, now
  also on the compiled path), so linear models are exact and a new target moves the path to its own steady state. The
  rules `P(t) = -(A0 P(t+1) + A1)^-1 A2`, `Q(t)`, `G(t) = -(A0 P(t+1) + A1)^-1 A0` and the constant `k(t)` run
  backwards from the last regime's own QZ solution and the constant `-(A0 P + A1 + A0)^-1 c` that is its fixed point,
  so, unlike OccBin, there is no horizon to choose. Anticipated shocks enter through `w(t) = Q(t) a(t) + G(t) w(t+1)`
  and surprises through `Q(t)`, so `solve_plan`'s holds and free shocks work unchanged (responses without the
  constants). Surprise regimes split the span: from each arrival the rules are those of the regimes then known. Regimes
  that change nothing are dropped first, and with no change left the ordinary solution is returned, which makes the
  equal-to-base results exact rather than equal to rounding. The filter (`tv_filter`) starts from the reference
  regime's stationary distribution for the state before the sample (OccBin's convention) and shares
  `piecewise_smoother`; estimation names regime values `<parameter>@<regime>`, whose specs fall back to the
  parameter's (`Model.spec`). Regimes filtered or estimated must start inside the sample after its first period, so the
  base regime is identified; forecasts refuse regimes starting after the span.
- **Regimes with OccBin, perfect foresight, posteriors and decompositions; correlated shocks with OccBin** (added 30
  October 2026):
  - *One schedule for the nonlinear solutions.* `regimes.regime_schedule` turns regimes into the model of each period
    from the first solved (`Schedule`: keys, each key's `regime_model`, names), with the regime before the first period
    as the reference; `_phases(freq=None)` numbers periods from 1 for `occbin`. Perfect foresight and OccBin take only
    regimes known in advance: a surprise would need the stacked system solved again at its arrival (perfect foresight
    already does that for shocks) and, for OccBin, beliefs that differ from the path; both are refused rather than
    approximated.
  - *Perfect foresight.* The stacked system's equations are each period's regime's own `model_equations` (parameters
    substituted, equations replaced by name and matched by name and dependent variable), with their derivatives taken
    per regime; the compiled path is kept for one regime throughout (that regime's model is solved as any other) and the
    slower expression path used when the equations switch. The terminal steady state is the last regime's.
  - *OccBin.* `_Piecewise` gains `offset`, the period a solution starts in: each period's matrices are
    `_regime(regime model, reference steady state, constraint choice)`, cached by (regime, choice). The regimes keep the
    model's max and min (matched by position and equation; their branches carry the regime's parameters, since
    `model_equations` substitute them), so each period's constraint is checked with its own regime's values
    (`_Piecewise.binding`, `branches`). After the horizon the constraints are slack and the rules are the regimes' own,
    computed once backwards from the last regime's stable solution and constant (`_known_rules`), so the horizon need
    not reach past the last regime. Without regimes every arithmetic step is the one before (the fast path included),
    so results are bit-identical.
  - *Correlated shocks with OccBin.* The filter already used Q in full; the smoothed shocks now solve
    `R(t) e = gap` for the likeliest `e`, `Q R'(R Q R')^+ gap`, which equals the old least squares when R has full
    column rank (nothing binds) and differs only where a binding regime mutes shocks; `_place_muted` measures the move
    in `z'N'Q^-1 N z`. The old least-squares arithmetic is kept when Q is diagonal. Forecast draws use the Cholesky
    factor (`covariance.shock_factor`), as the linear draws do. Decompositions (`decompose.path_decomposition`, shared
    by OccBin and the regime filter) orthogonalise by the same factor by default, as impulse responses and variance
    decompositions do; `orthogonal=False` gives the shocks' own contributions, additive either way.
  - *Posterior forecasts with regimes.* At each draw `with_values` sets the regimes' values from the draw, the state
    comes from `regimes.smooth` and the path from `plans._solve_plan` with the regimes (the fit's, then `ahead`, then
    those given); draws use that plan's period rules. *Moments.* A model estimated with regimes has no single set of
    moments; `FittedModel.moments(regime=)` and `fevd(regime=)` take one regime's own stationary solution
    (`regimes.regime_solution`, the regime's model at its own steady state), defaulting to the base model as before.
    Iskrev's identification check still refuses: its moments would have to come from the whole path of regimes.
- **Regime gaps closed: surprises in the nonlinear solutions, constraints that change, identification** (added 1
  November 2026):
  - *Surprises.* `regimes.regime_schedules` gives, for each date news arrives (period 0 and each surprise's start), the
    schedule believed from then on: the expected regimes plus the surprises started by then, each believed for good (a
    temporary surprise's end is a surprise too), built from the same phases and keys as `regime_plan`'s infos, so the
    first-order and nonlinear solutions agree on what is believed when. `regime_schedule` is that with one view and
    still refuses surprises for any caller that cannot take them. Perfect foresight (`_surprise_regimes`) solves each
    view from its arrival to the span's end, from the history so far, and keeps it until the next arrival, splicing as
    `simulate_with_surprises` does for unanticipated shocks (the two compose: each news segment solves its own surprise
    regimes). OccBin's `_Piecewise` keeps all the views; every method that takes `offset` (`rules`, `path`,
    `solve_from`, `check`) uses the view in force at its first period, and `binding`/`branches` use each period's
    actual regime (the last view by then) unless given a view. Surprise shocks already solve each period, so they see
    each surprise when it starts; anticipated shocks (`occbin(anticipated=True)`, OccBin forecasts) solve again at each
    arrival from the state reached, the shocks still known; the OccBin filter (each period solved from its own state)
    takes surprises as is. Holding targets by free shocks is solved for one path from the first period and refuses
    surprise regimes.
  - *Constraints that change.* A regime's max and min are matched to the others by label (the equation and `max` or
    `min`), not by position, over the union of all the regimes' constraints. A regime that lacks one has None there:
    its linearisation ignores that branch choice and `_binding` never reports it binding, so the guess-and-check over
    the union treats it as slack. One brought in by a regime takes as reference branch the one that holds at the
    reference steady state, with the regime's parameters (the same rule the model's own constraints follow); the
    choice tuple stays the union's, and each regime's own branch is its reference flipped where the union's binds, so a
    constraint's slack branch can differ between regimes. A model with no max or min of its own is taken when a regime
    has one. Rewriting (another floor) was already possible and is unchanged.
  - *Identification.* `bayes.regime_moments`: the regime filter's time-varying transition (`_matrices` of
    `regime_plan`) from the reference regime's stationary distribution (`system.start`), each period's observable
    means, variance (upper triangle) and covariances with the `lags` periods before it, stacked and divided by
    sqrt(periods). The alternative, the Jacobian of the likelihood's information matrix, needs data and second
    derivatives; the stacked moments are what Iskrev's test already differentiates, period by period, need no data and
    reduce to the same thing when nothing changes. With no regime changing in the sample the moments are
    `moments(system)` exactly, so the check is bit-identical to the one without regimes; when a trial value makes a
    regime change something that the point does not (a regime pinning a parameter being varied), all trial values are
    evaluated stacked so the Jacobian has one length. `Model.identification`/`Template.identification` take `regimes=`
    and `span=` (the sample's dates), leaving out regimes after it as estimation does, so a parameter of a regime that
    never occurs moves nothing and is reported not identified.

### Distribution functions and moving windows in the expression language (30 October 2026)

- **Why:** the model editor's checker and the Dynare reader accepted `normcdf`, `normpdf` and `erf`, but the
  evaluator had none of them, so a model using them checked and then failed when solved. They, and MATLAB's other
  distribution functions, are now part of the one expression language (`pymetrix.expr.functions`).
- **Names and parameterisations:** MATLAB's (`<family>pdf`, `<family>cdf`, `<family>inv`): gamma by shape and
  *scale*, exponential by its *mean*, lognormal by the mean and standard deviation of log x, uniform on [a, b];
  arguments with MATLAB defaults may be left out and are filled in at evaluation, so trees print as typed.
- **Evaluation:** `scipy.special` for the normal and erf (the common case in models, fast on scalars), `scipy.stats`
  otherwise; invalid parameters give NaN, missing stays missing, Series stay Series. The compiled solver functions
  call the same code through a fixed namespace (`_F.numeric("tcdf", (...))`), so the names in generated source are
  still never the user's.
- **Derivatives:** the chain rule over every argument, with each partial an expression (`partials`). In x always; in
  the parameters where the derivative is a closed form of the functions at hand (normal, lognormal, uniform,
  exponential, a gamma's scale, a Poisson's lambda, a binomial's p: d binocdf/dp = -n (binocdf(k; n-1) - binocdf(k-1;
  n-1)), so no floor function is needed). Where none is offered (degrees of freedom, shape parameters, a binomial's
  n) differentiation raises `ExpressionError` naming the argument, rather than a numerical derivative that would make
  some solvers' Jacobians quietly inexact. Discrete pmfs and cdfs are flat in k, as `step()` is.
- **Moving windows:** `movavg(x, n)` and the rest are stored with every lag explicit, `Call("movavg", (x, x[-1], ...,
  x[-n+1]))`, and print back as `movavg(x, n)`. So they take lags through the dataset's resolver (within entity, gaps
  respected), and everything that reads an expression's lags (model lag structure, Jacobians, compiled functions)
  sees the window without special cases. Windows are capped at 1000 periods.
- **Writers:** Dynare and IRIS get `normcdf`, `normpdf` and `erf`; the other functions are refused with a message,
  as `step()` is.
- **Integer quantiles (30 October 2026):** `poissinv` and `binoinv` are scipy's `ppf` (which is the smallest k with
  cdf(k) >= p) with two corrections to MATLAB's convention: p = 0 gives 0, not scipy's -1 below the support, and all
  mass at 0 (lambda or p = 0) gives 0 for every p in [0, 1]. They are steps in every argument, so `partials` offers
  none and linearising through one is refused with its own message, rather than a zero derivative that would be
  right almost everywhere and silently wrong where it matters.
- **Centred and expanding windows (30 October 2026):** the parser has no strings, so the variants are names, not a
  third argument: `cmov<stat>` for centred, `cum<stat>` for expanding (not `exp<stat>`, which would read as
  exponential smoothing next to `expinv`). A centred window is stored as a trailing one is, every period explicit,
  from its longest lead, (n - 1) // 2, down to its longest lag, n // 2 (pandas' `center=True`): so the model compiler,
  the checker's lead count, Jacobians and the compiled functions see the leads with no special case, the statistic
  and its derivatives are the trailing function's on the same values (`CENTRED` maps the name), and only printing
  needs to know which argument is the unshifted one (`window_lead`). An expanding window cannot be written with fixed
  lags: it is evaluated on the whole series by pandas' `expanding()` within each entity (every index level but the
  dataset's time axis, which the evaluator takes from the resolver's dataset), like the group-wise functions; on
  arrays (estimation) it is refused, `compilable` is false, `compile_model` refuses it with a message pointing to a
  derived series or a moving window, and it has no derivative. The model editor's completion leaves them out.
- **Moments:** `describe` (skewness and excess kurtosis bias-corrected, as pandas), `corr(method=)`, `cov`,
  `partial_corr` (residuals on a constant and `given`, listwise), `crosscorrelogram` and `autocov` (full-sample
  moments, sums over T, as `correlogram`'s autocorrelations). Dialogs: Data → Statistics → Descriptive statistics…
  and Correlations…, Test → Cross-correlogram….
- **Checks:** tests/test\_distributions.py and tests/test\_moments.py (see their docstrings and validation.md).

### Co-skewness, co-kurtosis and the random generator (31 October 2026)

- **Co-moments:** `Dataset.coskewness` and `cokurtosis(names, standardized=True, form="matrix"|"pairs")` and
  `systematic_comoments(names, market)`, in `data/dataset.py` beside `cov` and `partial_corr`, sharing `_deviations`
  (listwise over the active sample; means and, when standardized, standard deviations with the T denominator) and
  `_comoment`. Decisions: the finance convention (moments over T, as PerformanceAnalytics' `M3.MM`/`M4.MM`, Jondeau
  and Rockinger), so M3 and M4 are symmetric tensors flattened as E[e (e ⊗ e)'] with columns labelled `"j,k"` (and
  `"j,k,l"`), the diagonal g1 and b2 (not excess), stated in the notes with the conversion to `describe()`'s G1 and
  G2; the pairs form is mean(e_i^(r-1) e_j), row i and column j. Returned as `Table`s (names on rows and columns), as
  `cov` and `corr` are. The Correlations dialog gains `show="coskewness"|"cokurtosis"|"systematic"` with `form`,
  `standardized` and `market` (shown only for them); new fields have defaults, so saved dialogs open unchanged.
- **Random generator:** `pymetrix._random` holds the setting (`set_random_generator`, `get_random_generator`, the
  `random_generator` context, exported from `pymetrix`) and the only constructors: `generator(seed)` (exactly
  `np.random.default_rng` for PCG64, `np.random.Generator(BitGenerator(seed))` otherwise; a Generator passes through)
  and `seed_sequence(seed)`, from which chains, SMC blocks and Monte Carlo replications spawn, each child made a
  Generator by `generator`. Every `default_rng` and `SeedSequence` in the package and the finance plugin goes through
  them (the plugin through `pymetrix.extension.random_number_generator`); a test scans the source. The setting is
  mirrored in the `PYMETRIX_RANDOM_GENERATOR` environment variable, which processes started by `_processes` and the
  sweeps' process pool read on import, so parallel draws equal serial ones. `Provenance.random_generator` is filled
  in `__post_init__` (the generator in use) whenever a seed is recorded; `Posterior` and `MonteCarloResult` keep it
  from when they were made; `codec` asks a class's `_saved_fields` what missing fields meant, so seeded provenance
  saved before reads as PCG64. Each `CommandRecord` keeps the generator it ran with; projects write it only when not
  PCG64 (so files are unchanged by default), replay (`_replay_entries`), `compute` and `recompute` run each step with
  its own, and `Session.script` inserts `set_random_generator(...)` where it changes; the opening setting is restored
  afterwards. App: View → Random generator, saved in the desktop settings as `random_generator`.
- **Checks:** tests/test\_comoments.py and tests/test\_random\_generator.py (see their docstrings and validation.md).

### Speed: compiled model equations (12 October 2026)

- **OccBin forecasts** (added 29 October 2026): built on the linear forecast's `Solution` (its index, history, shells and
  provenance) with OccBin's paths put in its forecast periods and the observed series recomputed, so every view of a
  forecast works unchanged. The starting state comes from the smoothed levels, lagged copies included; draws cannot
  be laid over a baseline (the model is not linear across regimes), so each draw is a whole OccBin path with its
  shocks as surprises.
- **OccBin gaps closed** (added 30 October 2026):
  - *Iterated smoother.* `occbin_filter.iterated_smoother` runs `piecewise_filter` again with each period's expected
    regimes given (`regimes=`, one pass a period, no regime search), from regimes `_Piecewise.solve_from` works out
    from the smoothed state of the period before and the smoothed shock, until they repeat. Regimes are compared as
    whole expected paths (horizon x constraints), not just the period's. Without care it cycles at once: at the floor
    the policy shock has no effect, the smoother leaves it at zero, the rule then evaluates above the floor and the
    check says slack. `_place_muted` moves such shocks in the null space of R(t) (states unchanged) by the least
    change in standard deviations that levels the constraint's branches, which a tie counts as binding; that is the
    mode of their distribution given the data and the regime. A period whose re-derived regime makes an observation
    without error impossible keeps its earlier regime (`smoother_kept`). The likelihood stays the filter's. On
    simulated data the iterated regimes agree with the truth about as often as the filter's overall (better in some
    samples, worse in others: the filter's contradicted periods are often near misses either way); the gain is a
    smoothed path consistent with its regimes.
  - *Posterior forecasts.* `stochastic._occbin_draw`: the OccBin filter at the draw's model, then
    `estimation.occbin_solution` (factored out of `FittedModel._occbin_forecast`). About 0.1 s a draw for the New
    Keynesian model.
  - *Held targets.* `occbin_filter._hold`: under fixed regimes the path is affine in the shocks, so a step solves
    the targets exactly; the regimes of the new shock path are then checked, as OccBin's guess-and-verify. Holds with
    unanticipated shocks would need a nested solve per period and are refused, as are draws with holds.
  - *Decompositions.* Through the smoother's own T(t), c(t), R(t): contributions given the regimes, with the
    regime constants c(t) propagated as their own part, so additivity is exact and no "nonlinearity" residual is
    needed. The first period is the starting point, as in the linear decomposition, so the two are equal when nothing
    binds. Moments under OccBin would need long simulations and are left on the linear solution.
- **Benchmark first.** A synthetic model the size of FINEX: 270 equations in blocks of gap (with a lead and `exp`), trend growth and policy rate (with `max`), annual.
  - Before: 60 years of perfect foresight took 6.7 s.
  - An estimation step (rebuild, steady state, linearise, QZ) took about 5 s.
  - Profiling showed most of the time was spent walking expression trees per equation and period, re-taking symbolic derivatives, and two older inefficiencies.
- **`pymetrix.expr.compile`:**
  - Expression lists become one generated NumPy function `f(v, p)`, with variables as slots `v[k]` and parameters as `p[j]`.
  - The source is generated only from the parsed tree (no names, so no injection route) and cached by expressions.
  - Results match `evaluate` (same NumPy operations, division and missing values).
  - Group-wise functions are not compiled.
- **`pymetrix.models.fast`:**
  - Per model: compiled residuals, all non-zero derivatives (taken symbolically once per form), and the steady-state residuals and Jacobian.
  - Built from the symbolic residuals, so a model rebuilt with new parameter values reuses the code.
  - `ENABLED = False` switches back to the interpreter.
- **Solvers:**
  - The steady state and linearisation use the compiled functions.
  - Perfect foresight evaluates every equation for all periods at once. Where each value comes from (an unknown, data, a terminal value) is worked out once, and the sparse Jacobian's rows and columns are fixed in advance.
- **QZ:** the pencil is on `[x_L(t-1); x(t)]`, size n + (number lagged) instead of 2n, in the real Schur form (3.5× faster than complex at this size). Blanchard-Kahn counts the lagged variables.
- **Fixed on the way:**
  - `Model.exogenous` and `has_leads` were recomputed on every access, and linearisation read them inside a loop over every term.
  - Linearisation scanned all equations once per variable.
  - Both are now done once.
- **Result at 270 equations:**
  - Perfect foresight, warm: 0.13 s (from 6.7 s).
  - Estimation step: 0.6 s (from about 5 s); the QZ is now most of it.
  - The first solve of a new model pays about 2.7 s once for symbolic derivatives.
- **Checks** (tests/test\_speed.py):
  - Compiled expressions equal the interpreter.
  - Template solutions and perfect foresight with a binding `max` are identical either way (1e-10).
  - All model tests also pass with compilation off.
  - A speed test with generous budgets (3 s and 6 s, against 0.13 s and 0.6 s measured) guards against regressions.
- **Not done:** symbolic derivatives are still interpreted on first use; JAX and Numba are not used; reducing static variables further (Dynare-style) is left for when estimation of large models needs it.

### Speed: estimation without rebuilding the model (13 October 2026)

- **Profile first.** One Smets-Wouters (2007) log-posterior evaluation spent about 88% of its time rebuilding the model from its equations: `Model` construction 29%, steady state 27% (mostly re-deriving compiled forms), linearisation 32%. The Kalman filter was 6%.
- **`Model.with_parameters(values)`:** a copy at other parameter values. It shares, through a `_shared` cache, everything that does not depend on them:
  - the compiled functions (`fast.functions`, keyed by the equations chosen);
  - the linearisation's structure (`linear._compiled_system`: auxiliary equations for lags and leads beyond one, the variable order, the endogenous list), worked out once from the symbolic residuals. Substituting values never removes a reference (`0*x[-2]` stays), so the structure is the same at every value;
  - `endogenous`, `exogenous` and `has_leads`, which now read the symbolic equations.
  - `model_equations` (equations with values substituted) is lazy: the compiled solvers never ask for it.
  - `params` keeps the caller's type: a dict stays a dict, a ParameterSet stays a ParameterSet.
  - Pickling leaves `_shared` out (compiled code does not pickle; sweeps and parallel chains send models to other processes). It is worked out again on first use.
- **Estimation:** `_Likelihood` builds the base model once (`_base`, not pickled) and evaluates each trial value as a `with_parameters` copy. The steady state builds the interpreter's residuals only when no compiled functions exist.
- **Priors:** normal, beta, gamma and inverse-gamma log densities in closed form (`math.lgamma`). SciPy's still handles the edges and outside the support.
- **Result:** Smets-Wouters log posterior 12.0 ms per evaluation, against 24.5 ms when rebuilt, on the same machine and load. What is left: steady state, the filter, the root solve and the priors, in that order.
- **Checks** (tests/test\_model\_reparam.py):
  - At random values, a copy's steady state, decision rules, variables and substituted equations equal those of the model built again (1e-12). This covers the New Keynesian and RBC templates and a random model with two-period leads. Where the values have no stable solution, both fail.
  - Copies share compiled functions and leave the original's values alone; a solved model and its copies pickle and solve the same afterwards.
  - Closed-form priors equal SciPy (1e-10), including outside the support.
  - The Smets-Wouters log posterior from the cached path equals the rebuilt one at 30 points (3e-11), and all estimation, Bayes and Dynare tests pass.

**Speed test revised after CI.** The Windows installed-wheel runner took 6.3 s for the estimation step (0.6 s locally), so fixed time limits are unreliable on shared runners. Two checks now guard the speed work:

- **Machine-independent:** on the same machine, compiled linearisation and perfect foresight must beat the interpreter by more than 3×. They measured 6.6× and 8× on a 90-equation model.
- **Loose absolute backstops** at 270 equations: 3 s for 60 years of perfect foresight and 20 s for an estimation step.

A random dense QZ was rejected as a reference: it is much harder than a model's structured pencil, so it would not catch a regression.

### Growth paths and calibration; nowcasting next (12 October 2026)

**Order agreed:** steady states with growth and calibration (done, below), then a nowcasting toolkit, then the rest of the modelling plan. The nowcasting toolkit has three parts:

- **Ragged-edge data:** publication lags, `ragged_edge()`, `as_of(date)`, and a simulated monthly dataset with a cleaned `sample_macro`.
- **Models:** bridge equations; MIDAS with several indicators, models by month of the quarter, and combination; a mixed-frequency factor model on statsmodels' `DynamicFactorMQ`, with news.
- **Workflow:** a nowcast tracker, pseudo-real-time evaluation, dialogs and a lab.

**Steady states with growth** (`pymetrix.models.growth`):

- **Additive balanced growth**, `x(t) = level + growth·t`, as gap-trend models in logs times 100 need.
- **Solving:** levels and growth rates come from the residuals at two consecutive periods, by Levenberg-Marquardt with the analytic Jacobian from the compiled derivatives. A check at later periods rejects paths that are not balanced.
- **Undetermined levels** are found from the Jacobian's null space; `fix=` pins them.
- **`SteadyState`** gains `growth`, `undetermined` and `at(name, t)`.
- **Linearisation** evaluates each lag and lead at its own point on the path; auxiliary lags sit on the path too.
- **The QZ counts unit roots as stable** (within 1e-6 of one), as IRIS does for trending models. A random walk now solves, while explosive models still fail.
- **Simulation plans** add the trend back, anchored at the first simulated period. Results do not depend on the anchor, because shifting along a balanced path is itself a solution.
- **`FirstOrder(growth=True)`** makes `Model.solve` and the Solve dialog work for trending models.
- **The state-space form refuses trending models for now** (their measurement needs a time-varying intercept), rather than give wrong levels.

**Calibration** (`Model.calibrate`, Model → Calibrate):

- Free parameters are solved so the steady state hits targets. Targets are expressions of the variables (`k/y`, `c/y`, levels, spreads), one for each parameter, with or without growth.
- The steady state is re-solved at each trial, warm-started from the last; the solver is Powell hybrid, then Levenberg-Marquardt.
- The result is a `Calibration`: the new model, its steady state, values found, before and after, and notes on parameters outside their bounds. It has views.

**Checks** (tests/test\_growth.py):

- The analytic growth path of a FINEX-like gap-trend model (potential output, trend growth, a gap, forward-looking inflation, a policy rule).
- Every model equation holds in levels at every simulated period after demand and trend-growth shocks (1e-9).
- No shocks gives exactly the trend.
- Starting from data, potential output continues from its last value.
- A random walk solves; explosive and accelerating models are refused.
- Calibration matches the RBC closed form for β, and for β and δ together (1e-10). A trending model is calibrated on its growth path.
- The dialog's recorded code replays in a reopened project.

### Nowcasting tools (12 October 2026)

**Scope.** At the owner's request: the underlying tools only, with no tracker, evaluation layer, report templates or lab.

- **Example data:**
  - `sample_macro` is regenerated with a 1990s burn-in, so its start-up jump (growth up to 160% annualised) is gone. Only four tests had hard-coded the old numbers.
  - OLS on the consumption function now shows simultaneity bias (MPC 0.76), while IV recovers the true 0.6 and 0.2: a better teaching example.
  - New `sample_monthly` (2000-01 to 2025-03): industrial production, retail sales, employment and a survey. They follow a monthly activity path whose growth adds up to GDP growth, including 2025Q1, which is kept out of `sample_macro` as the quarter to nowcast.
- **`Variable.release_lag`:** days from a period's end to publication.
- **`pymetrix.nowcast`:**
  - **The ragged edge:** `Dataset.ragged_edge`, `Dataset.as_of` (it removes unpublished values; it does not restore vintages), `fill_edge` (AR forecasts of each series) and `realign` (vertical realignment).
  - **`Bridge`:** indicators averaged within quarters, an `Equation` fitted on them, and a nowcast that fills, averages and then forecasts with the equation.
  - **`DynamicFactor`:** a thin wrapper around statsmodels' `DynamicFactorMQ` (EM, Kalman filter, Mariano-Murasawa quarterly series), giving factors, loadings, `nowcast` from smoothed predictions, `update` (apply to new data), and `news` (the contributions of each new figure, as a `News` result, with `by_series`).
  - **`combine`:** weighted averages of nowcasts.
- **`pymetrix.factors.PCA`:** correlation or covariance matrix, signs fixed, `transform`. Static factor models with rotations remain planned.
- **MIDAS** needed no change: several indicators and `high_lag` already existed, and a differently ragged edge is handled by `fill_edge` or `realign` first.
- **Dialogs and views:** Estimate → Bridge equation, and Estimate → Dynamic factor model; views for the bridge, the factor model, news and PCA.
- **Checks** (tests/test\_nowcast.py):
  - publication days, day by day;
  - the AR fill by hand;
  - bridge coefficients equal to OLS on quarterly means, and its nowcast step by step;
  - factor-model parameters and nowcast equal to statsmodels called directly (1e-8);
  - news impacts plus revisions equal to the change in the nowcast (1e-10);
  - PCA against an eigendecomposition by hand;
  - both dialogs replaying from projects.
- **A design point found on the way:** a nowcast inside the data's sample must use smoothed estimates. One-step predictions leave out the period's own figures, and then the nowcast does not match the news.

### Demos brought up to date; a CI fix for the factor model (12 October 2026)

- **Lecture 1:** a third exercise on simultaneity. With the regenerated data, OLS gives an MPC of 0.756 and −0.010 on lagged consumption. IV with government spending and lagged income as instruments gives 0.602 and 0.197, against the true 0.6 and 0.2. The notebook asserts the claim, so it cannot silently stop being true.
- **Lab 3:**
  - The round is also reported from the `forecast_round` template in the policy theme, as Markdown.
  - A new section 8 nowcasts 2025Q1 from monthly indicators as of 15 February: the ragged edge, a bridge equation, and the factor model updated to 15 April with the news, which is asserted to add up.
  - The question asks whether to revisit the round's judgement. The nowcasts start at 6.5–7.4% against the VAR's −0.6%, then fall to 5.0%; the true value is 1.1%, and students see that nowcasts can be well off.
- **Lab 4:** a section estimating the New Keynesian template from 400 simulated quarters. The persistence and shock sizes are recovered within three standard errors (asserted), and the question explores the weak identification of kappa and phi\_pi.
- **Fixes made with the demos:**
  - `Dataset.keep(*names)`, and `select([...])` now explains that it chooses periods.
  - The Solve dialog's Set, Shift and Target fields take `name[2025Q1:2025Q4]=value` and `name[2026Q1]=value`.
  - `ForecastRound.history` keeps 16 periods of actual data from `from_forecast(data=...)`, so fan charts show history without data being passed. `ForecastStore` saves it.
- **CI failure on the nowcasting commit (9 of 14 jobs):**
  - Reproduced in a throwaway environment with CI's numpy 2.5.4: statsmodels' EM step failed a Cholesky factorisation when the first month had only the survey observed (the growth rates start a month later).
  - With numpy 2.4 the same case stopped after 2 iterations.
  - `DynamicFactor` now starts at the first month where every monthly series has data; later missing values are unaffected.
  - The news test now also checks for sensible magnitudes.
  - The core-only jobs needed a statsmodels skip on the dialog test.

### The forecasting core for models (12 October 2026)

The IRIS/FPAS workflow on a fitted model: filter (with tunes), decompose, forecast under a plan, and report moments.

- **Moments** (`pymetrix.models.moments`):
  - `LinearSolution.moments(shock_sd, lags)` solves the discrete Lyapunov equation for the state covariance, with autocovariances `P^k S`, and each shock's share from the same equation with only that shock.
  - `fevd(periods, shock_sd)` returns the existing `FEVD` result (one vocabulary with VARs).
  - Unit roots have no unconditional moments; they raise, and point to `fevd` by horizon.
  - `FittedModel.moments()` and `fevd()` use the estimated shock sizes; the fitted-model view gains Moments and Variance decomposition tabs.
- **Forecasts:** `FittedModel.forecast(span, scenario, data=, tunes=)` starts from the Kalman-smoothed state (refiltered when `data` is given), runs the simulation plan, and adds the observed series in data units through the measurement map. The result is a `Solution`.
- **Soft tunes and smoothed shocks** (added 29 October 2026): a tune with a standard deviation is one extra measurement row per tuned value, observed only in its own period, with variance sd**2; a variable whose tunes are all exact keeps the single error-free row, so exact results are unchanged. `filter(shocks=True)` adds e(t) from the smoothed state disturbances (projected from the smoothed states when R has full column rank, which keeps the identity at rounding error on both engines); e(0) is E[e(0) | y] through the states that start from their stationary distribution. A separate result type bundling states, shocks and the likelihood was considered and left out: the shocks are columns like a solution's, and the likelihood is `loglike`.
- **Least-norm plans** (added 29 October 2026): `Scenario.hold_sd` and `free_sd` (or more free shocks than holds) switch `solve_plan` from the paired square system to min ||u||^2 + ||(A_s u - g_s)/sd||^2 subject to A_h u = g_h, u the free shocks over the periods up to the last hold in units of `free_sd`: the minimum-norm solution of the exact holds, then the soft ones in the null space the exact ones leave. Weights default to 1 because a LinearSolution carries no shock sizes; anticipation follows `unanticipated`, as for every plan.
- **Tunes:** `StateSpaceSystem.filter(data, tunes=)` adds each tuned variable as an extra observation without error, the way IRIS imposes tunes. A tune pins a level the data leave unidentified: in the NK template, output is observed only through growth, so a tune shifts the whole output path by a constant, and the forecast of growth moves by exactly 4 per unit.
- **Shock decomposition** (`pymetrix.models.decompose`):
  - Shocks are recovered as `Q^+ (x(t) − P x(t−1))`, with each shock's contribution and the initial conditions `P^t x(0)`, and a `residual` for paths that do not follow the model.
  - Available as `LinearSolution.decompose(paths)` and `FittedModel.decompose()`.
  - `start="presample"` (added 29 October 2026): the first row of the paths is x(-1), so the first period's shocks are the shocks' and the initial condition is P x(-1); the default `"first"` keeps the first period's state, shocks included, as the initial condition.
  - The chart is stacked bars with the path as a line; figures gained `Bars(stacked=True)`, which stacks positive values above and negative below.
- **Checks:**
  - moments against the AR(1) closed form, a sum over impulse responses (1e-10), long-horizon FEVD and a 200,000-period simulation (2%);
  - known shocks recovered exactly, including with a second lag;
  - forecasts by hand;
  - tunes hit exactly while the observed series stay fitted.
- **Remaining in this area:** forecast decompositions, and contributions of observables to filtered states.

**Codespaces verified (12 October 2026).** The owner created a codespace from the repository. It opens as VS Code in the browser, and the pymetrix app runs on the browser desktop (the Ports tab, port 6080, password `vscode`), as designed. The README now links straight to a new codespace (`codespaces.new/GeoffHeenan/pymetrix`). For collaborators: invite them to the private repository; each codespace uses the free monthly allowance of the person who creates it.

### FINEX-type models completed: surprises and filtering along a trend (12 October 2026)

- **Perfect foresight with expectation errors** (`simulate_with_surprises`):
  - With `Scenario(unanticipated=...)` and `PerfectForesight`, the news dates are the first period plus each period with an unanticipated shock.
  - At each news date the stacked model is solved from the realised path so far. Anticipated paths are known; surprises are known only up to that date, later ones are unknown (set: the data's path; shift: nothing). Each solve is kept until the next news date.
  - Provenance lists the dates of expectation errors.
  - For models without leads, `unanticipated` is allowed and changes nothing; the earlier refusal is removed.
- **The state-space form along a trend:**
  - `StateSpaceSystem` gains `trend` (each state's growth) and `trend_lags`.
  - Observations are adjusted by `Z·(g·(t + lag))` (t = 0 at the first observation) in `filter`, `loglike` and the fitted series. Levels are reported with `g·t` added back. Tunes get the offset through the tuned system.
  - Solutions with unit roots declare no stationary states, and the filter and likelihood use `auto` initialisation, so unit-root states start diffuse. Stationary models are unchanged, because all their states are declared stationary.
  - `Model.estimate(..., growth=True)` estimates trending models; the refusal is removed.
- **Checks:**
  - Surprises on a linear model equal the first-order plan to 1e-9.
  - The path before a surprise equals the path without it.
  - A surprise demand shock hits the lower bound in its quarter, while the same shock announced moves the economy at once.
  - With output, potential output, inflation and the rate observed, the filter recovers a simulated trending path exactly (1e-6).
  - Gap persistence and a shock size are estimated within three standard errors.
- **Remaining for FINEX-type models:** multiplicative growth of variables in levels, and a growth terminal condition for perfect foresight.

### Quantile regression (built 11 October 2026)

- **`Quantile(q, vcov="robust"|"iid", kernel, bandwidth)`** wraps statsmodels' QuantReg (library rule), replacing the stage-3 placeholder. Registered in `methods`, so it is in the Estimate dialog.
- **Naming:** the field is `vcov`, not `cov`, because the Estimate dialog passes its own `cov` (classic, White, HAC) to every method. A test found the clash.
- **Results:** a `FittedEquation` with the pseudo R², q and a notes line.
- **`quantile_process(eq, data, quantiles, vcov, alpha)`** returns `QuantileProcess`, with `table()` (coefficients by quantile plus OLS) and `plot()` (each coefficient across quantiles with a band, and OLS dashed). It has a view.
- **Checks:** coefficients, standard errors and pseudo R² equal statsmodels'. Median regression on a constant lies between the middle order statistics. Simulated heteroskedastic data show slopes rising with q. Help topic `quantile-regression`, with growth-at-risk as the motivating case.
- **Not yet:** bootstrap standard errors, non-crossing constraints, and a growth-at-risk helper that fits skewed-t densities.

### Period spinners and Dynare import (built 11 October 2026)

- **Period spinners (the user's request of 11 October):** Start and End fields are spin boxes over the periods of the form's dataset. `Session.period_choices` reads them from the `data` field, else the first dataset named, and they follow a change of dataset. Typing works, and empty means no limit. When the start is after the end, the End box is outlined and OK is disabled. `span_errors` in `Session.run` refuses such steps from scripts and notebooks too.
- **Dynare `.mod` import:** `pymetrix.models.dynare.read_mod(path or text)` returns `DynareModel(endogenous, exogenous, params, equations, initval, shock_sd, commands, model)`. It reads:
  - `var`, `varexo` and `parameters`, without their LaTeX and long\_name decorations;
  - parameter values, which may be expressions;
  - the model block: timing `x(+1)` becomes `x[+1]`, `#` locals are substituted, tags are dropped, and an equation without `=` means `= 0`;
  - `initval` and `steady_state_model` (as guesses), and `shocks` (stderr or variance).
  - (added 29 October 2026) `stderr` or a variance on a `varobs` variable, in `shocks` or `estimated_params`: that series' measurement error (`measurement_sd`, or `sd_<variable>` when estimated), as Dynare reads it.

  Other commands are listed, not run, and macro directives are refused. Equations are paired with endogenous variables by bipartite matching, preferring each equation's own left-hand side and variables that are no other equation's left-hand side. An equation whose left side is not its variable is rewritten `v ~ v + (lhs) - (rhs)`. Equations with parameters are calibrated stochastic equations (`~`); others are identities. `.steady_state()`, `.linearize()` and `.solve()` follow.
- **Checks:** a New Keynesian model read from Dynare text gives impulse responses identical to the hand-built model. An RBC model's steady state matches the analytic one, and output's response to the technology shock equals steady-state output.
- **UI:** Model → Import Dynare file…, with a view of equations, parameters, shocks and the commands not run.
- **Not yet:** estimated\_params, varobs and estimation; the macro processor; writing .mod files; IRIS model files.
- **Later (30 October 2026): the macro processor and correlated shocks.**
  - **Macro processor** (`pymetrix.models.dynare_macros`): a text pass before the reader, as Dynare's preprocessor
    is. The directives are parsed into a tree first (so an unclosed `@#if` or `@#for` is reported at its opening line
    before anything runs), then run with one environment of macro variables; expressions go through a small Pratt
    parser with Dynare 6's precedence (range below `+`, `in` below comparisons...). Branches not taken are not
    evaluated, as in Dynare. Includes are run where they stand, in the same environment, so a calibration file's
    defines are seen after it. Each output line remembers the line of the file it came from (an included file's at
    its `@#include`), which is how the model editor puts problems of the expanded text back where they were typed.
    Text without a file refuses `@#include` rather than guessing a folder.
  - **Correlated shocks** (`pymetrix.models.covariance`): correlations are kept apart from the sizes,
    `{(a, b): r}` with the pair in the model's order of shocks, so sizes stay `sd_<shock>` and a correlation is its
    own parameter `corr_<a>_<b>`, bounded in (-1, 1), as Dynare's. The covariance is `D C D`, built only when some
    correlation is not zero: every uncorrelated computation keeps its old arithmetic exactly, which the existing
    Dynare comparisons rely on. Positive definiteness is checked where the covariance is made, so estimation's
    usual penalty for a model it cannot solve covers it. Responses and decompositions use the Cholesky factor
    `D chol(C)` (Dynare's `stoch_simul` order-dependence, said in the notes); unit-shock responses stay as they were.

### MIDAS (built 11 October 2026)

- **`pymetrix.midas.MIDAS(formula, lags=12, weights="exp_almon"|"beta"|"unrestricted", ar=0, high_lag=0)`** with `.fit(low, high)`, native: no mature Python MIDAS library is installed.
  - Indicators are high-frequency series names. Each low-frequency period takes `lags` values ending at its last high-frequency period, less `high_lag`.
  - Exponential Almon and Beta (shapes on the log scale) weights are normalised to sum to one. Parametric forms are fitted by nonlinear least squares (scipy `least_squares`), from flat weights and two alternative starting shapes; standard errors come from the Jacobian.
  - Unrestricted MIDAS is OLS.
- **`FittedMIDAS`:** params, `weights` per indicator, residuals and fitted values, summary, `plot()` of the weights, and `nowcast(low, high, through)` for every low-frequency period the indicators cover. AR terms are applied recursively, using data where available.
- **Checks:**
  - Exponential Almon recovers simulated weights (correlation above 0.99) and the slope.
  - U-MIDAS equals statsmodels' OLS (coefficients and standard errors).
  - With monthly data to October, a `high_lag=2` model nowcasts 2024Q4 and the full-quarter model cannot.
- **UI:** Estimate → MIDAS (mixed frequency)…, with a view (Summary, Graph, Weights). Help topic `midas`.
- **Not yet:** step-weighting and polynomial (PDL) forms, leads within the current quarter for daily data, MIDAS with mixed-frequency VARs, and a dynamic-factor nowcasting model.

### Defaults page (built 11 October 2026; review finding R7)

- **Help topic `defaults`, "Defaults, and other packages' defaults":** pymetrix's defaults where packages are known to differ, and the option that matches.
  - White standard errors are HC1 (as Stata's robust), with Stata's cluster factor.
  - HAC uses floor(4(n/100)^(2/9)) lags; Stata's newey needs an explicit lag.
  - ADF chooses lags by AIC, where EViews uses Schwarz (lags="bic") and Stata's dfuller uses none.
  - The VAR covariance is degrees-of-freedom adjusted, where Stata's var divides by T without dfk.
  - SUR divides by n (as Stata's sureg without dfk; df\_correction matches dfk).
  - HP lambdas are 100, 1,600 and 14,400 (as EViews).
  - Seasonal adjustment defaults to STL, with X-13 available.
  - BVAR hyperparameters follow GLP.
  - Tests use t for single-equation least squares, z elsewhere.
- **Only confident statements:** the page says to check other packages' documentation where no comparison is listed. The review's examples were mostly not real differences (HP matches EViews; SUR matches Stata). The page also records one real difference it did not mention: ADF lag selection versus EViews.
- **Not yet:** a column for IRIS conventions, and links from result summaries to this page.

### ARDL and Markov switching (built 11 October 2026)

- **`pymetrix.ardl.ARDL(formula, lags=1, order=1, trend="c", select=None, max_lags=4, max_order=4)`** with `.fit(data, span)`, wrapping statsmodels' `ARDL`, `ardl_select_order` and `UECM`.
  - `order` is one number for every regressor, or a mapping by name. `select="aic"|"bic"|"hqic"` chooses the lags and orders instead.
  - **`FittedARDL`:** params, residuals, summary, `long_run()` (multipliers with delta-method standard errors), `ecm()` (the error-correction form) and `bounds_test()`.
  - The bounds test is Pesaran-Shin-Smith: case 3 with a constant, case 5 with a trend, case 1 with neither. It returns a `TestResult` with both bounds' critical values, both p-values and a verdict at 5% (relationship, none, or inconclusive).
  - **Checks:** coefficients and standard errors equal statsmodels'; the long-run multipliers equal the hand formula (the sum of each regressor's coefficients over one less the sum of the autoregressive ones); two independent random walks show no level relationship.
  - **UI:** Estimate → ARDL and bounds test…, with a view (Summary, Graph, Long run, Bounds test, Error correction). Help topic `ardl`.
- **`pymetrix.markov.MarkovSwitching(formula, regimes=2, ar=0, switching_variance=False, switching_exog=True, search=20, seed=0)`** with `.fit(data, span)`, wrapping statsmodels' `MarkovRegression` (or `MarkovAutoregression` when `ar` > 0).
  - The likelihood has local maxima, so the fit tries `search` random starts. They come from `seed` through statsmodels' own `rng` argument, so results repeat and NumPy's global generator is untouched.
  - **`FittedMarkovSwitching`:** params (statsmodels' `x1`, `x2` renamed to the regressors' own names), `transition` (from-by-to), `durations`, `probabilities("smoothed"|"filtered")` by period, summary and `plot()` (the series and each regime's probability).
  - Regimes are numbered as estimated; the summary says to read their means before naming them.
  - **Checks:** simulated two-regime data are recovered (means, a stay probability near 0.95, over 95% of periods classified correctly); the log likelihood and parameters equal statsmodels' fitted directly; repeated fits agree; the dialog, view and project round trip work.
  - **UI:** Estimate → Markov switching…, with a view (Summary, Graph, Transition probabilities, Expected durations, Regime probabilities). Help topic `markov-switching`.
- **Not yet:** time-varying transition probabilities (statsmodels' `exog_tvtp`), ordering regimes by a chosen parameter, and NARDL (asymmetric ARDL).

### Library audit (11 October 2026)

- **Rule:** wrap a mature library when it does exactly what is needed; write our own only where the expression or model layer needs it (lags within entity, explicit form, shared parameters, models); every hand-written piece that has a library equivalent gets a cross-check test.
- **Wrapped:** VECM, Johansen, ADF, KPSS, Engle-Granger and the Kalman engine (statsmodels); LIML, IV-GMM and panel estimators (linearmodels); EGARCH and the other volatility models, plus PP, DF-GLS, Zivot-Andrews and variance ratio (arch); moving average (pandas rolling); pivot tables (pandas); linear algebra, optimisation, QZ, sparse solves (SciPy).
- **Hand-written and already cross-checked:** OLS with classic, White, HAC and one-way cluster errors; NLS; IV with first-stage F, Sargan and Durbin-Wu-Hausman; Logit, Probit, Poisson; ARMA errors (SARIMAX); native GARCH and GJR (arch); Wald, Breusch-Godfrey, Breusch-Pagan, White, ARCH, RESET, Chow, Jarque-Bera, Ljung-Box; correlograms; HP, Baxter-King, Christiano-Fitzgerald, Hamilton; recursive residuals and CUSUM; VAR estimates, impulse responses, FEVD, lag selection, forecasts and tests; state space; SUR and 3SLS.
- **Gaps closed (tests/test\_crosschecks.py):** two-way cluster covariance (statsmodels), NLS standard errors (SciPy curve\_fit), likelihood-ratio test (compare\_lr\_test), local projections (horizon-by-horizon OLS with Newey-West h+1 lags), rolling regressions (RollingOLS), one-sided HP (end points of expanding hpfilter), moving average (pandas rolling). All agree to 1e-8 or better.
- **No Python reference exists:** perfect foresight and linear rational expectations (Dynare and IRIS are not Python); checked analytically instead (closed-form solutions, the model equations holding on the solution path, perfect foresight equal to linear responses).
- **Decision:** do not replace the cross-checked native filters and tests with statsmodels calls. They are short, match to machine precision, handle panels and missing ends, and keep the core free of statsmodels; switching would make basic features depend on an optional package for no gain in accuracy. No linearmodels backend for SUR either: the cross-check gives the same assurance without a second code path.

This closes the most visible gap with EViews identified in the capability review; next is forward-looking model solution.

### UI review fixes: views down the side, OK that waits, plain failures (4 November 2026)

The pre-review UI audit's nine first-priority findings. **Result windows** list their views down the left side
(`pymetrix.ui.view_switcher.ViewSwitcher`) instead of tabs: a fitted equation's thirteen views ran behind scroll
arrows even maximised at 1366×768, and at 125% only three or four showed. Two designs were weighed: primary tabs plus a
More ▾ menu, and a list down the side. The list won: every view stays in view under its full name (a short window
scrolls the list rather than hiding its end), group headings (Summary, Tables, Graphs, Diagnostics, Provenance; from
six views) make a long list readable, Up and Down move through it, and it is the same for every kind of result; a
More menu would have hidden most of an equation's views, and which views count as primary differs by kind. It costs
the width of the longest name (at most 30% of the window). The switcher keeps `QTabWidget`'s calls, so code and tests
that chose views by index or title work unchanged; the view picked last is remembered per kind of result (only a
reader's pick, not one made in code) and kept when the result is recalculated. **Dialogs** hold OK through one gate
(`pymetrix.ui.dialog_checks.OkGate`): each check (fields marked * empty, start after end, a formula checked as typed, a
preview) holds or releases OK under its own key, so one check's all-clear no longer re-enables OK while another holds
it. The * is `pymetrix.forms.needs_value`: no default and no meaning for empty (`Widget(empty=...)`), so the marker
and what OK waits for cannot disagree. A step whose code does not parse as Python is refused in `Session.run` before
it runs, so it never reaches History or Problems. **Failures** are explained at the dialog boundary
(`pymetrix.ui.messages.explain`): common low-level exceptions become a plain message and a next step, pymetrix's own
messages are kept, and the technical text stays under Details; the box opens over the dialog without blocking it.

### UI review fixes, batch C: menus, panels and words (4 November 2026)

The audit's second- and third-priority findings that A and B had not covered. **Menus** keep command ids and change
only where an item is shown: a command's menu path is its decorator's, its place within the menu
`pymetrix.commands.menus.MENU_LAYOUT` (the Forecast menu now has one), and a plugin's grouping its `MenuLayout`
(Finance's submenus). Moving an item breaks no script or project (they record calls, not menu paths); the docs'
paths are checked by `tests/test_docs_menu_paths.py`. A workspace's Ctrl+number is fixed for the core six
(`pymetrix.ui.workspaces.workspace_numbers`), so the docs' Ctrl+1 to Ctrl+6 hold with plugins. **Panels** say what
they lack in one pattern (`pymetrix.ui.messages.EMPTY_STATES`: "No … yet: how to get some"), inside their own view.
**Long steps**: a demo is built by `run_blocking`, as a dialog's step is, so the window stays live with progress and
Cancel; the alternative, shipping the demos' results precomputed, would have meant a second copy of each demo to
keep in step with the code. Two findings were left: figures keep a white background under a dark palette (they are
what prints and exports), and the guide's step wording waits for the open guide pull request, which rewrites those
files.

### Regression without a formula, and Breusch-Godfrey's constant (5 November 2026)

`pymetrix.ols(y, X, ...)` is least squares on arrays passed directly, for code running many small regressions. It
is a function, not a new method or result type: it returns the `FittedEquation` a formula fit returns, its equation
written from the names, so views, tests, forecasts and saving need nothing new, and one result type per estimator
holds. Rather than a second implementation, `Equation.fit`'s least squares was split at the design
(`methods._ols_on_design`): both paths run the same code on the same arrays, so they agree to rounding. What `ols`
skips is the formula and the dataset: parsing (equations for the same names are cached; they are immutable),
evaluating terms over a dataset, pandas sample selection, and working out the rows used from a dataset; the checks
(missing values, collinearity, HAC only on time series) and the provenance (the same fields, fingerprint and
`dropped_rows` computed from the arrays) stay. The data's origin is recorded as `vintage_request={"policy":
"direct"}` rather than in a new provenance field. Profiling the formula path (40 observations, 3 to 6 regressors):
design building about 40%, provenance and the effective sample about 35%, dataset construction and parsing about
15%; `ols` takes about a sixth of the time. Views and diagnostics were already computed when asked for, so nothing
needed to be made lazy. Names that are not plain series names are refused rather than rewritten.

Breusch-Godfrey without a constant differed from statsmodels because statsmodels always puts a constant in the
auxiliary regression; pymetrix's uses the equation's own regressors, with zeros for the presample lags and n
R-squared about the mean (both conventions use all n observations and the chi-squared form). The default stays;
`auxiliary_constant=True` gives statsmodels' number.

### The window in front, and one set of colours (5 November 2026)

Asked for: make the active window more obvious, perhaps by tinting its title bar. Qt's styles disagree on a title
bar's palette (`CC_TitleBar` in Qt 6): Fusion fills the front bar with Highlight but writes titles in fixed white or
black; the Windows style follows Highlight and HighlightedText; windows11 fills every bar with Button; windowsvista
draws the system's caption and its colours and reads no palette at all, and that is the style most users see. A
palette set on a `QMdiSubWindow` also reaches its contents (a table's selection colours). So `pymetrix.ui.active_window`
paints the bar over the style's own from an event filter on each window (Qt paints first; the style's buttons are left
as they are), in palette roles: Highlight and HighlightedText for the window in front (the bar moved towards 4.5:1
contrast where the palette's pair is short of it), Window and the muted colour for the others, a two-pixel accent
frame round the front window; `subWindowActivated` repaints. The Objects panel already selected the front window's
object. The panel with the keyboard focus gets a two-pixel accent line (`QApplication.focusChanged`).

Colours: status and link colours were literals in eight modules, four different reds among them. `pymetrix.ui.theme`
has the roles (error, warning, success, link, muted), each a light and a dark value picked by the palette's window
lightness, moved darker or lighter until it reads at 4.5:1 on the window and the base colour. Muted text was
`palette(placeholder-text)`, half-opaque black on Qt's default palette: 3.9:1 on its window grey; it is now the text
colour mixed towards the window colour only as far as 4.5:1 allows. Syntax colours, the spreadsheet's cell tints and
the marks drawn on charts are kept as they were, as named constants in the same module, so a test can check that no
other module in `pymetrix.ui` writes a colour. Rejected: setting the application palette's PlaceholderText (it would
change every field's placeholder) and a custom-drawn title bar with its own buttons (the native buttons and their
hover states would be lost).

### Summary statistics grouped (5 November 2026)

A result's notes were one line each under its table, in one weight and colour. They are now split by
`pymetrix.summary_layout.split_notes`: a note "Label: value" whose value is a number, or short (a known label's up
to 90 characters, an unknown one's up to 24, not a sentence and not code), is a **statistic**; anything else
(warnings, explanations, a null hypothesis) stays **prose**, after the statistics. Statistics go under Estimation, Fit, Information criteria, Residual checks or
Other by the label's longest known prefix, ignoring case (`pymetrix.wording.STAT_GROUP_PREFIXES`, beside the other
display words); a single group shows no heading. Two renderers share the split. **HTML** (`Table._repr_html_`, so
notebooks and every result shown through a table, and the HTML report): inline styles only, a flex row of groups
that wraps to two or three across and stacks when narrow, labels muted, numbers right-aligned in tabular figures;
Qt's rich text ignores flex, so there the groups stack, which reads as well. The colours are neutral greys with
alpha, named in one place (`MUTED_COLOUR`, `STRIPE_COLOUR`, `RULE_COLOUR`), because `palette(...)` exists only in
Qt style sheets. **The desktop window** draws the Summary with a `QTableWidget` and, under it, a
`pymetrix.ui.stats_panel.StatsPanel`: a grid of groups re-laid on resize (the most columns, up to three, whose
widest groups fit), muted with `ui.theme.muted_css()`, labels and values word-wrapping so a narrow window
never scrolls sideways; the zoom re-measures it. The grid gains alternate-row shading (the palette's
AlternateBase), a bold header, 10 more pixels a column, and a width no more than its columns. The plain text,
CSV, Markdown and Copy summary are unchanged. Estimators that wrote their statistics as one compound note
("Log likelihood -280.58; AIC 565.17; BIC 570.38") now write one "Label: value" note each, with a fitted equation's
labels (Log likelihood, Akaike info criterion, Schwarz criterion...), so they group too: ARIMA, ETS, ARDL, MIDAS,
Markov switching, systems, VAR (its criteria and largest root), DSGE estimation, count series, spatial models,
conditional logit, dynamic factor models, the OccBin filter, and Finance's copulas (static, time-varying, DCC),
vines and dynamic Nelson-Siegel. Their plain text has one statistic a line; the values are as before. The
compound notes were changed where they are written rather than parsed: their wording varied by estimator.

### Simulated critical values for ADF and Engle-Granger (9 November 2026)

`ADF` and `EngleGranger` take `critical_values="mackinnon"` (the default, statsmodels as before) or `"simulated"`, with
`draws=2000` and `seed=0`, and Engle-Granger `exog=` (fixed levels regressors, simulated values only). The simulated
path does not call statsmodels: `pymetrix.stats.adf_arrays` computes the statistic on a stack of series, and the data
and every draw go through the same function, so a draw's statistic is computed exactly as the data's. It reproduces
`adfuller` and `coint` (the regressors in statsmodels' order, the lag search on the common sample of the largest
lag with the criteria from the profile log-likelihood counting the regressors, ties to the fewer lags, the t-stat rule
from the top down at 1.645, the refit on the chosen lag's own sample, coint's R² guard giving minus infinity, and a
signed zero when no degrees of freedom are left); a test holds them to the same lag and 1e-9. Speed comes from
batching: the search's regressions are nested, so one QR per draw gives every candidate's sum of squares (the tail
sums of Q'y squared) and last t-ratio, and the refits are grouped by chosen lag. The QR is CholeskyQR2 on unit-norm
columns (two Gram products and Cholesky factors), which NumPy computes on a stack many times faster than its batched
Householder QR and as accurately for these well-conditioned regressions; Householder is the fallback when a Gram
matrix is not positive definite, and is used for the levels regression, whose R² guard needs residuals near zero
accurately. Draws come from `pymetrix._random.generator(seed)` in blocks that continue one stream (the same numbers
as one call), so the results do not depend on the block size. The p-value is the share at or below the statistic
(not (count + 1)/(draws + 1) as the Perron test's), as asked, with its standard error `sqrt(p(1 - p)/draws)`; the
critical values are NumPy's default (linear) quantiles. Rejected: computing the data's statistic by statsmodels and
only the draws by the arrays (two implementations to keep in step, and no `exog` in `coint`).

### CI: the Python 3.11 worker crash was PySide6 6.12.0 dropping references to None (6 November 2026)

- **Problem:** from the merge of the window title painter, the statistics panel and the chart house style (5bd616b),
  the full Linux Python 3.11 job lost its test process gw0 on every run, near the end ("worker 'gw0' crashed", in
  `test_working_set.py` or `test_demo_rehearsal.py`), with no report. Python 3.13 and the oldest-versions job (3.11,
  PySide6 6.6) passed. Not reproduced locally in full runs, in a 1 MB stack, or in gw0's own file order.
- **Finding the cause:** each test worker now keeps a crash report (tests/conftest.py): its last steps, every thread's
  stack on a fatal signal, and its captured stderr in a named file, so Python's own fatal-error message outlives the
  process. CI prints them with the worker's exit status, Java's error file and the kernel's crash lines. The reports
  showed exit status -6 (an abort), no faulthandler stack (Python disables it before its own fatal-error abort), then
  "Fatal Python error: none_dealloc: deallocating None", in the garbage collection after a test module.
- **Cause:** PySide6 6.12.0 drops a reference to `None` on every Qt call that returns nothing (measured: exactly one
  per call, for QObject, QWidget and value types alike; none with 6.6, 6.9.3, 6.10.3 or 6.11.3). `None` is immortal
  from Python 3.12, so only 3.11 runs down its count, and the process that has made the most Qt calls dies first. The
  merge did not add a fault; its repainting (a title bar painted over on every paint of every window, fonts set per
  statistic) made many more such calls.
- **Fix:** the `qt` and `all` extras leave PySide6 6.12.0 out on Python 3.11; later Pythons keep it. A test checks that
  a Qt call returning nothing leaves `None`'s count alone, so a binding with the fault fails by name, not as a crash.
  The desktop app on 3.11 with 6.12.0 had the same exposure, after enough use.

### Robust regression, restricted least squares, and the cointegrating regression's sample (8 November 2026)

- **`RobustLS` is a method object like `Quantile`,** registered under "robust" (kind `"robust"`), in its own module
  (`estimation/robust.py`), and computed natively rather than through statsmodels: the core needs only NumPy, pandas
  and SciPy, and statsmodels' RLM is the test's reference instead. Iteratively reweighted least squares from the
  least-squares start, the MAD scale re-estimated each round (statsmodels' order: weights from the last scale, then a
  new scale from the new residuals), stopping when the coefficients and the scale change by less than `tol` relative
  to 1 + their size (statsmodels' default stops on the change in the objective at 1e-8, so it can end a step short:
  the test iterates it to convergence). H1 is the covariance, as RLM's default; H2 and H3 were left out. Fits of kind
  `"robust"` label their likelihood criteria as a Gaussian quasi-likelihood, as quantile regression's. The Estimate
  dialog lists it by its label; its own standard errors only, so a covariance other than conventional is refused.
- **Restrictions are a field of `OLS` and `IV`, not a new method,** so they combine with every covariance and appear in
  the dialog's Method settings without a new Estimate field (the command's schema is unchanged). Stored normalised
  (a tuple of the restrictions one by one, None for none), so `OLS() == OLS(restrictions="")` and scripts record a list.
  The estimate is the textbook projection `b - A R'(R A R')^-1 (R b - r)`; its covariance is `M V M'`, `V` the
  chosen kind's sandwich from the restricted residuals with `n - k + q` residual degrees of freedom, which equals the
  substitution estimate's delta-method covariance for every kind (classic, HC1, HAC, clusters) because `M A` equals
  `J (J'X'XJ)^-1 J'` for any `J` spanning the null space of `R`. The parser moved out of `System._restrictions` into
  `estimation/restrictions.py` (`parse_restrictions`, `linear_restrictions`), which now names coefficients by
  substitution: any label (`x[-1]`, `dlog(y)`, a system's `c.y`) is replaced by a placeholder before parsing, longest
  first, so term labels that are not identifiers work; systems use it unchanged in behaviour.
- **Cointegrating regressions take `where=`** through the shared `_frame` (also `gregory_hansen`), and the dialog
  gains Start, End and Rows where, laid out as Estimate → Equation's. The fields were added with defaults, which the
  registry's rule says needs no schema change: old steps load unchanged.

### Exercise packs (13 November 2026)

- **A pack is a folder; a plugin points at one.** The entry-point group `pymetrix.exercises` names a callable that
  returns the folder (`importlib.resources.files(...)`), so a plugin's pack and a folder are read by the same
  `load_pack`, and the plugin's enable/disable state and the app's question about new plugins apply (`_admit`), as
  for its other extensions; `plugins.pending()` reads the group too, so a plugin that only ships a pack is asked
  about. Rejected: registering packs in a registry (the `courses` route): it would import the plugin to list a folder.
- **Checks are data.** A check is `object` + dotted `value` path + `expected` (+ tolerances), resolved by key lookup
  (mappings, pandas objects), attribute lookup (properties count) and key lookup on other objects (`params.educ`),
  with no calls and no `__` names: a pack opened from any folder runs nothing. Estimates have no `bse`/`pvalues`
  attributes, so `bse`, `tvalues`, `pvalues` and `nobs` are read from the result's own coefficient table and `stats`
  when no attribute has the name; the check's language itself stays lookups. Rejected: expressions in checks (an
  evaluator to sandbox) and the guide plugin's sentence grammar (it cannot read numbers).
- **Code runs only from the command line.** `solution.py` (or, without one, the steps' code blocks) runs in a fresh
  `Session` through `run_code`, as the Immediate window runs typed code, after the data load as recorded File → Load
  data… / Load example data… steps (`data_commands`): a pack's CI checks it the way a user would do it. The app
  never runs a pack's code, so there is no trust question to ask (the project trust store, `trust.py`, stays for
  projects' typed code).
- **Validation lists everything.** `load_pack` collects every problem (file and reason) before raising one
  `SpecificationError` with them in `.problems`, so an author fixes a pack in one pass; unknown keys are errors, to
  catch misspellings. A pack written for a later pymetrix loads and says so (`requires_pymetrix`, pre-releases
  allowed), so the app can list it with the reason rather than hide it.
- **CSV needs its axes.** `Dataset.read` cannot read a CSV without a time or entity column, so `pack.toml` takes
  `[data.<name>]` tables with `time`, `entity` and `freq`, the fields of File → Load data… (an addition to the
  format the pack author was first given).
- **Models and where code runs** (after the first pilot pack). `models/` holds `.mod`, `.model` and `.pymodel`
  (pymetrix's syntax: an extension chosen here, since pymetrix had none) files; `load_model` reads them as the import
  dialogs do, and the panel's Import… opens those dialogs filled in rather than importing on start, since importing
  is often a step of the exercise. A solution runs with the pack folder as its working directory and `PACK_DIR`
  given through the session's objects (the namespace is rebuilt from them before typed code runs), removed after.
  `level` is one of beginner, intermediate, advanced; `classification` is free text, shown and never enforced: a
  label pymetrix cannot verify should not gate anything. `se` joins `bse`, `tvalues`, `pvalues` and `nobs`, read
  from the result's coefficient table because `params.se` is an unnamed array.
- **The panel.** A dock on the right, made the first time an exercise starts, over the plugin panel interface
  (`AppInterface`: the session, running steps, the status bar), so data load as the app runs any step. Each step
  shows everything at once (no paging); Show the code is the dialogs' folded code box with Copy, not their live
  `_code_panel`, which is built for a command's fields. Added folders are remembered in the app's settings, as course
  folders are.

### Exercise pack gaps (15 November 2026)

- **A model's session name is pack data.** `[models.<file stem>] name = "..."` in `pack.toml`, beside
  `[data.<name>]`, rather than `models: [{file, name}]` in each exercise's front matter: the name belongs to the
  file, not to an exercise, so two exercises using a model cannot import it under different names, and the front
  matter stays a list of names. Lookups take the stem first, then the session name, so a pack written before the
  table keeps working, and `load_model(PACK_DIR, "nk")` reads the way a solution names the object.
- **Show the code builds from the session.** The dialogs' code panel now uses the code the step will record
  (`_dialog_code`: the call built from the session, for dialogs whose call needs it). Check model had shown a call
  built in an empty session with a dataset `data` assumed, which differed from the recorded step.
- **Whole-number path parts: string, int, position.** TOML paths are text, so `2001` on a Series indexed by year was
  never found. The string is tried first (so no path that worked changes meaning), then the int label, then a
  position only for lists, tuples and arrays: a Series or DataFrame is read by label alone, since a fallback to
  position would make `s.1` mean the second row on one index and the label 1 on another.
- **A model's impulse responses get a dialog over the existing call.** Model → Analysis → Impulse responses… runs
  `LinearSolution.irf`, made a recorded call (`pymetrix:linear_solution.irf`) with one new argument, `shocks=`, so the
  dialog's one shock gives a result with one shock rather than all of them at size 1 but the one chosen. Selecting
  after computing every response keeps each response as it is among all, correlated shocks included. The shock
  field is a new dynamic kind, `"shock"`: the shocks of the object its `depends_on` field names. Rejected: reusing
  Estimate's Impulse responses… (a VAR's dialog: identification schemes and bands that a model's solution does not
  have), and a size per shock (the exercises asked for one shock at a time; code takes `shock_sd` for several).

### Programs reading and writing projects (8 November 2026)

- **Deferred steps from code.** `Session.add(cmd, deferred=True)` appends the record Manual mode's deferral on opening
  appends (`ok=False`, `deferred=True`, the outputs it will make, its code and call), so `pending()`, `compute()`,
  the script and the project's `"deferred": true` treat both alike; nothing new is saved. The outputs come from the
  step itself (a `Call`'s output, a dialog's call's output or its `name`, the top-level names typed code assigns), or
  `outputs=`. The step is validated except where it uses another pending step's results (which do not exist yet). It
  is an undoable step (the undo entry `run` makes), and observers hear a `step` event with `detail={"deferred": True}`
  rather than a new kind, so existing observers keep working. Rejected: running and discarding the result (the point
  is not to spend the time), and a separate "planned steps" list (two kinds of pending step for compute to merge).
- **The call in a saved result.** A session step that made a result with a call (a dialog's, or typed code recorded as
  calls) notes it in `pymetrix.calls` (`call_of`: by the result's identity, through a weak reference, so nothing is
  kept alive). `save_result` writes it into the provenance (`call`, the `Call` fields as plain JSON; `call_uses`, the
  objects it reads by name). The arguments are the recorded ones, so the span and row condition are as given, not
  as resolved (the resolved sample is already in the provenance). Rejected: recording every `@api` call outside a
  session (the cost lands on every call, and a script has no names for its datasets), and storing the call on the
  result object (results are frozen, and many types would need a new field).
- **The commit beside the version.** `pymetrix.build.build_info()` runs `git` only when a folder above the package holds
  a `.git` and git tracks the package's `__init__.py` (so a virtual environment inside some other repository does not
  report that repository's commit), each command with a two-second limit, `GIT_OPTIONAL_LOCKS=0` and no prompt, and
  caches the answer for the process. `dirty` is `git status --porcelain --untracked-files=no` on the package folder:
  changes to tests or documents do not mark the code dirty. `Provenance.build` is filled in `__post_init__` when a
  version is given (as `random_generator` is); `_saved_fields` gives an old saved provenance a "not recorded" note
  rather than the running commit. The save log and a saved result's manifest carry the same keys at their top level,
  beside `pymetrix` and `package_version`. Rejected: writing the commit into `_version.py` at build time (wheels are
  built by CI from a clean checkout, so it would say only what the version already says, and a development install
  would go stale).
- **A public interface for programs, provisional.** `Call`, `make_call`, the codec's JSON functions and
  `Session.attachments` were already used by the app, the MCP server and plugins; they are now listed in their
  modules' `__all__` and documented together (the guide's "Reading and writing projects from programs", whose
  examples a test runs), marked public but provisional before 1.0. No stability policy is promised: that is the
  project lead's decision. They stay in their own modules rather than the top-level `pymetrix` namespace, which is
  what typed code and scripts see (`from pymetrix import *`), so generic names such as `encode` do not crowd it.
- **Monte Carlo standard errors of posterior means.** `models.bayes.mcse(chains)` is the posterior s.d. (all draws,
  ddof 1) over the square root of `ess` (Stan's estimator, as the ESS column already reports), so the two columns are
  consistent: MCSE = s.d. / sqrt(ESS). Batch means were the alternative; they need a batch size, and the ESS was
  already computed. ArviZ splits each chain in two before its ESS; pymetrix's ESS does not, and the two agree within
  a fraction of a percent on long chains. A Bayes fit's `stats` gain `ess_<name>` and `mcse_<name>` per parameter
  (keyed by name, as the stats are a flat mapping of numbers); the estimates and the posterior's other columns are
  untouched.

### The default largest lag in short samples (14 November 2026)

- **Cause of the zero Engle-Granger statistic.** statsmodels' `adfuller` caps Schwert's `ceil(12 (n/100)^(1/4))` at
  `n//2 - terms - 1`, which leaves the largest search regression `n - 2(n//2) + terms` residual degrees of freedom:
  none for an even n without deterministic terms, the regression `coint` runs on the residuals. With no degrees of
  freedom the sum of squares is zero, the log-likelihood infinite, AIC/BIC pick that lag, and the t-ratio's standard
  error is infinite, so the statistic is a signed zero; simulated draws do the same, so the critical values were 0.
- **The rule.** `adf_arrays.default_max_lag(n, terms, used)` = `min(Schwert, n//2 - terms - 1, (n - 2 - terms - used
  - MIN_DF) // 2)`, at least 0, with `MIN_DF = 4` and `used` the columns of the levels regression for Engle-Granger
  (and Pedroni's cointegrating regression), whose residuals have already spent those degrees of freedom. The
  statsmodels path is given the result as `maxlag`, so the MacKinnon and simulated paths, the panel tests and the
  simulations share one rule. Rejected: arch's cap `(n - 1)//2 - 1 - terms` (it still leaves 1 to 3 degrees of
  freedom and ignores the levels regression), and a cap of regressors at half the rows (it changes results up to
  about 35 observations, where statsmodels' rule is not degenerate). The constant 4 is a judgement, recorded in
  known-gaps.
- **Refusal rather than a warning.** A user's lag or largest lag must leave at least one residual degree of freedom
  (`adf_arrays.check_lag`), and `degenerate_statistic` refuses a data statistic of exactly 0.0 or NaN (a t-ratio of
  exactly zero from data is a perfect fit, not evidence). Both raise `SpecificationError`; the old warning in the
  simulated result (which still reported 0.0) is gone. Minus infinity for perfectly collinear series keeps its own
  warning, as statsmodels returns it.

### One parser for coefficient hypotheses (14 November 2026)

- The Wald test parsed its hypothesis as a formula with the coefficient names as parameters, so a name that is not an
  identifier (`dlog(y)`, a term label) was read as an expression in series. It now calls
  `estimation.restrictions.parse_restrictions`, the parser restricted least squares and systems already used, which
  replaces each coefficient name by a placeholder (longest first) before parsing. That parser gains two things:
  spaces inside a name are optional (`L(x,1)` for `L(x, 1)`), and `` `name` `` quotes a name explicitly (an unknown
  quoted name is an error naming it). `FittedSystem.wald` no longer translates `c.y` into identifiers first (which
  broke on `c.dlog(y)`); the system restriction check flags a dotted reference only when no coefficient name starts
  there. Rejected: a second, Wald-only name handler (two parsers drift apart, which is how this arose).

### Seasonal dummies in formulas (14 November 2026)

- **Both spellings.** `C(quarter)` failed because `C()` takes a series or an axis and the period index has no
  `quarter` level. `Dataset.values_of` now falls back, after columns and index levels, to calendar fields of the time
  axis (`quarter`, `month`), so `C()` keeps one meaning and a user's own series called `quarter` still wins. The
  general form is a new marker, `seasonal()`, with no argument: the season is the quarter of quarterly data and the
  month of monthly data, so one formula serves both. It is recorded as the categorical `"seasonal()"` (not a valid
  series name, so no clash), and its dummies are named `seasonal[Q2]` / `seasonal[M2]`: short, readable in a summary,
  and usable in the Wald test now that it reads any coefficient name. Rejected: generating dummy series in the data
  (they would go stale when the sample changes and clutter the dataset), and centred seasonal dummies (a
  different parametrisation; plain 0/1 matches EViews' `@seas` and R's `factor(cycle(x))`).
- Calendar seasons are ordered by number (2 before 10); `C()` of an ordinary series keeps its text order.
  `calendar_names()` gives the names the dialog's formula check treats as known.

### The automatic Bartlett bandwidth of cointegrating regressions (15 November 2026)

- **Bug.** For a fractional bandwidth b, `long_run_covariance` summed the lags j = 1..floor(b) with the weights
  1 - j/(b+1), leaving out lag floor(b) + 1, whose weight is positive. The truncated weights are not a kernel: the
  spectral window goes negative for every fractional b tried (-0.20 at b = 1.5), and the long-run variance of an
  AR(1) with phi = -0.9 at b = 1.5 was -0.54 (the standard error NaN). Every lag j < b + 1 is now used, as in arch 9
  and R's sandwich; whole-number bandwidths (FMOLS and CCR always round up) were not affected.
- **Convention.** The default automatic bandwidth is now Newey and West's (1994) as arch 9.0 and sandwich
  (`bwNeweyWest` with `kernHAC`) compute it: the pilot lag is the integer part of 4 (T/100)^(2/9) (it was rounded up),
  S = 1.1447 (alpha T)^(1/3) is the bandwidth of the kernel k(j/S), so the weights 1 - j/(b+1) take b = max(S - 1, 0)
  (it was b = S, one lag longer than the bandwidth the constant 1.1447 is optimal for), and FMOLS and CCR use
  ceil(S) - 1 (it was ceil(S)), which is sandwich's `NeweyWest` lag floor(S). S is capped at T - 1, as in arch.
  Reasons: it is the papers' convention, the two reference implementations agree with it to 10-12 digits, and the
  tests can compare automatic bandwidths with arch 9 again rather than handing arch pymetrix's bandwidth.
- **The old rule, for one release.** `bandwidth_rule="legacy"` reproduces 0.1.0a18 exactly (pilot rounded up, b = S,
  the truncation included), so earlier results can be repeated; it is arch's rule before 9.0, which the tests use when
  arch 7 or 8 is installed. A separate switch for the truncation alone was rejected: one option that restores old
  numbers is easier to explain, and nobody needs a non-kernel estimate for new work. The name says what it is for
  rather than naming arch, whose version is not the point. To be removed after a19.
- **Saved steps.** The dialog's new Bandwidth rule field defaults to the new rule and is written to scripts only when
  legacy, so the command keeps schema 1 and steps saved before compute with the new rule, as scripts do. Migrating
  old steps to "legacy" was rejected: it would pin old projects to the non-kernel estimate.
- **Phillips-Ouliaris** is arch's own computation, so it follows the installed arch. Raising the arch floor to 9.0
  was rejected for now: it would change the oldest-versions pins (`.github/oldest-supported.txt`, `pyproject.toml`)
  and their environments for one test's default bandwidth, and a user's own arch would still decide. Instead the
  test's details record arch's version beside the bandwidth, and known-gaps says what changes.

### The EViews plugin, data route 1 (24 November 2026)

The bundle reader, the export program and File → Import EViews workfile… of [design/eviews-plugin.md](design/eviews-plugin.md)
(sections 1, 3 and 4), in `packages/pymetrix-eviews`.

- **One page, one Dataset, one call.** `Dataset.read(path, page=...)` through the `.evbundle` reader, so typed code
  and the dialog record the same `pymetrix:dataset.read` call and nothing new is needed in core. With several pages
  and no `page=`, the read is refused naming them: guessing the active page would make a script depend on how the
  workfile was last left in EViews.
- **What is not a column goes in `meta.extra["eviews"]`**, as JSON values, so Parquet files and projects keep it:
  groups, samples, scalars, strings, matrices, equations with EViews's estimates and model texts. `eviews_objects`
  turns them into `Selection`, `Sample`, `Matrix` and templates on demand. Putting them in the session as separate
  objects from the dialog was rejected for now: one recorded call per page is what the design asks of the dialog,
  and objects made outside a call would not replay.
- **Models through the core reader.** A model's text, with its `:eq` links, becomes a program (`equation eq.ls ...`
  on the equation's sample, `model m`, `m.merge`, `m.append`) that `read_eviews` reads with the page's data, so the
  plugin adds no second model parser. The page's scalars that the text names are its parameters.
- **Re-estimation is the core reader's fit.** `compare_estimates` refits only least-squares equations without options,
  since the core reader ignores options such as `cov=white` and would compare a different estimator; the rest are
  listed as not compared, with the reason. EViews's `c(i)` are matched to pymetrix's terms by the list's order
  (`c` is `const`), or `c_i` in an expression.
- **Strict, with places.** JSON with no NaN and no key given twice; a numeric field is a number, `NA` (the manifest's
  `na`) or empty; a date must start a period of the page's frequency; files must be inside the bundle's folder. Unknown
  keys are kept and noted, not refused, so a version 1 bundle from a later program still reads.
- **Weekly anchors come from the dates**, not from what EViews says about them, which the design left to verify.
- **Names.** Series are lower-cased and made valid Python names (`in` becomes `in_v`), with EViews's name kept in
  `Variable.source_key` and, when changed, `extra["eviews_name"]`. The id and date columns the program adds lose
  their `pmx_` prefix as axis names (`crossid`, `obs`); the time axis is `time`, as `Dataset.empty` names it.
- **Save the data as Parquet** reads each page from a Parquet copy written next to the bundle (rewritten while the
  bundle is there), so the script's lines are plain `Dataset.read("....parquet")`. The project still records the
  plugin's step: making one dialog step record several core calls would need a change in the session, not wanted for
  this.
- **The plugin's dialog ID** (`eviews:import_eviews`) is fixed in the plugin's own `tests/fixtures/command_ids.json`,
  as pymetrix's are in the core's. The plugin imports its dialogs outside its plugin context, because importing
  `pymetrix.commands` inside it would register pymetrix's own dialogs as the plugin's; a test runs that import first
  in a fresh interpreter.
- **The menu line is File → Import EViews workfile…**, not the design's Import EViews…: the File menu already uses
  every letter of "Import EViews" as an access key, and every menu line must have one; "workfile" brings the free k.
- **No writer.** A bundle writer would exist only for a round-trip test; the hand-made fixtures already pin every
  field, and the program, not pymetrix, is what writes bundles.

### Core additions for the EViews plugin (24 November 2026)

Decisions 4 and 5 of [design/eviews-plugin.md](design/eviews-plugin.md), built before the plugin.

- **The reader's text helpers are public, by name, not by a new layer** (extension API 1.10). `pymetrix.extension`
  maps `eviews_blank_comments`, `eviews_statements`, `parse_eviews_expression`, `render_eviews_expression`,
  `walk_eviews_expression`, the tree's node classes, `EViewsExpressionError`, `Diagnostic` and `line_and_column` to
  the reader's own functions and classes, as `figure_views` maps to `views._figure`. A plugin then splits and parses
  a program exactly as `read_eviews` does, and a change in the reader reaches it. Wrapping them in new public
  functions (offsets as lines and columns, a nicer node type) was rejected: two parsers to keep in step. The nodes'
  fields (`at`, `name`, `lag`, `fn`, `args`, ...) are now part of the API, so they change only additively. The version
  is 1.10, not 1.9: 1.9 was already `Session.proposing`. Versions compare as numbers.
- **An empty dataset is a classmethod and a dialog.** `Dataset.empty(freq, start, end)` is a recorded call
  (`pymetrix:dataset.empty`), as `Dataset.example` and `Dataset.read` are, and **File → New dataset…** records it, as
  Load data and Load example data record theirs. Only a dated time series: an empty panel or cross-section needs its
  entities, which `from_frame` already takes; the time axis is named `time`, as `from_frame` names an unnamed one.
- **Assignment within a sample is an argument of `with_series`**: `within=` (a `Sample`, a span or `"start:end"`),
  so the History and the script show one ordinary call. The active sample still does not limit generated series:
  that would make a dataset's values depend on a hidden setting, and every existing script would change. A separate
  method (`assign`) was rejected as a second way to add series; the cost is that a series cannot be named `within`
  through `with_series` (`add_series` makes one). A formula that uses the series it assigns is computed one period at
  a time inside the sample (leads and the series itself see the values before the call), as EViews computes it, since
  the common EViews idiom `y = y(-1)*1.02` would otherwise be translated into different numbers. Without `within`,
  `with_series` is unchanged: one pass over the values before the call. Keep updated (a derived series) and `within`
  are refused together: a derived series follows its formula in every period.
- **`with_series(y=None)`** makes a series of missing values (EViews' `series y`), and with `within`, missing values in
  the sample's rows (`y = na`). Formulas still have no missing-value literal.
- **Generate series gains Within** (a named sample, or a first and last period; under Advanced), step schema 3 with a
  migration from 2 that leaves the new fields empty. Adding fields with defaults needs no migration (the registry's
  rule), but schema 3 makes an older pymetrix say the step was saved by a newer version rather than list unknown
  fields.

### Removing the legacy bandwidth rule (24 November 2026)

- **What goes.** `bandwidth_rule="legacy"` (0.1.0a18's automatic bandwidth and truncated Bartlett lags) was kept
  for 0.1.0a19 and 0.1.0a20 and is removed, as planned (above). Its recorded 0.1.0a18 numbers leave the tests with it.
- **The argument stays, with one value.** `cointegrating_regression(..., bandwidth_rule=...)` (and
  `long_run_covariance`, `newey_west_bandwidth`, `bartlett_weights`) still take `bandwidth_rule`, accepting only
  `"newey-west"`. Removing the argument would have needed a call migration (version 2) that drops `"newey-west"`
  and fails on `"legacy"`, and scripts typed by hand with the default spelled out would stop with a TypeError;
  keeping it costs one documented argument and lets every such script run. `"legacy"` raises a
  `SpecificationError` that names the removal and the last version with the rule: silently computing other numbers
  for a script that asked for old ones was rejected.
- **The dialog field goes** (schema 2, with a migration): a saved step's `bandwidth_rule` of `"newey-west"` is
  dropped, and `"legacy"` raises a `MigrationError` (a project reports it as the step it cannot open), for the same
  reason. The field was only ever written to scripts when legacy, so scripts from the dialog are unaffected.
- **Tests with arch 7 and 8** (the oldest-versions job) can no longer compare automatic bandwidths, since those arch
  versions use the removed rule; they hand arch pymetrix's bandwidth rounded up to a whole number, where both use
  lags 1..b. The default rule is still checked against numbers recorded from arch 9.0.0 and R's sandwich whatever
  arch is installed.

### Preparing the public release: what may be redistributed (20 November 2026)

- **Fama-French data downloaded, not shipped.** The data library marks its files as Fama's and French's copyright and
  states no licence; linearmodels' NCSA licence covers linearmodels, not the data. `pymetrix_finance.datasets.french`
  downloads the five files on first use and caches the built dataset in `~/.pymetrix/cache/datasets/`. The plugin
  gives `ExampleDataset` an `os.PathLike` whose `__fspath__` does the download, so the core needs no new API and an
  older pymetrix still loads it. Kept: the same columns and months as the copy shipped before, so the tutorial and
  its text stay as they are. Rejected: downloading linearmodels' copy (the same data, from a third party), and
  serving the latest months (the tutorial's counts would change with every release of the library).
- **Tests without the network.** The finance tests serve a simulated stand-in in the library's own file layout, so
  the parser, the cache and the error path are all exercised; the tutorial's claims about the real returns are
  checked only with `PYMETRIX_FRENCH`, as the FRB/US and Brillet checks are opt-in.
- **Brillet's book.** Its excerpts left the repository; the tests read the reader's own copies through
  `PYMETRIX_BRILLET` (a folder, or `structmodel.txt` itself as before). The FRB/US excerpts stay: the Board's website
  terms put its information in the public domain, cited to the Board.


## File operations and persistence

Data is stored as Parquet, specifications and results as JSON, and models as plain text. All input and output goes through one plugin-based entry point.

```python
Dataset.read("data/na.parquet")
Dataset.read("warehouse://national_accounts", vintage="2026-09-30")   # a reader plugin
Dataset.read("sdmx://IMF/IFS/Q.WS.NGDP_XDC")
Dataset.read("survey.dta", entity="hhid")
ds.write("out.xlsx")
save_result(fit, "results/cons.pymxr"); load_result("results/cons.pymxr")   # a record of its views (see below)
```

| What | Format | Why |
| --- | --- | --- |
| Data | Parquet; variables, axes and flags stored as JSON in the file's metadata footer | Typed, compressed, fast, self-describing |
| Specifications, configuration, parameters | JSON (machine-written), TOML (hand-edited) | Frozen dataclasses serialise directly; diffable |
| Models | Plain-text `.model` file: variables, parameters, equations | Readable, diffable in git, editable without the UI |
| Fitted objects, solutions | Folder: `spec.json`, `params.json`, `provenance.json`, Parquet for residuals and paths | Reload without re-estimating |
| Large draws | Parquet, or Zarr when very large | Avoids huge JSON |

Pickle is never used for persistence: it breaks between versions and can execute code on load.

**Project layout.** The same structure works as a folder (for git) or zipped as one `.pymx` file (for UI save and for sharing):

```
project.toml          # name, format version, default frequency, data sources
data/*.parquet
models/*.model
results/<name>/...
scripts/session.py    # captured command log
```

The command log is the authoritative record; results are caches it can rebuild, and their provenance shows when they are out of date. `project.toml` carries a format version so newer releases can migrate older projects.

**Adapters to ship early:** CSV and Excel (openpyxl, including the wide layouts statistics offices use), Stata/SPSS/SAS (pyreadstat), MATLAB `.mat` (scipy), SDMX (`sdmx1`), FRED, and Dynare `.mod` model blocks. EViews `.wf1` has no reliable open reader, so migration is a one-off export from EViews to Excel or CSV. DuckDB is an optional route for querying large survey files without loading them.

## Existing packages and reuse

No existing package combines EViews-style estimation, MATLAB/IRIS-style solving and a forms UI. The gap is the integrating layer: the data and expression model, the uniform specification/result interface and the command registry. Build that; wrap the rest. This list is from memory and needs checking for current status and licences.

| Package | Covers | Role here |
| --- | --- | --- |
| irispie (OGResearch) | Python successor to the IRIS Toolbox: dated series, simultaneous and sequential models, Kalman filter, simulation plans | Candidate solver back end; closest overall match |
| ModelFlow (World Bank MFMod) | EViews-style macro model solving, add factors, scenarios, Jupyter widgets | Candidate solver back end; useful for users of World Bank MFMod models |
| pyfrbus (Federal Reserve) | FRB/US solving with Newton and mixed expectations | Reference implementation |
| statsmodels | OLS, GLM, ARIMA, VAR/SVAR, state space, many tests | Wrapped by methods and tests |
| linearmodels | IV, GMM, panel, SUR, 3SLS | Wrapped by methods |
| arch | GARCH, unit-root tests, bootstrap | Wrapped by methods and tests |
| dolo, pydsge, econpizza, gEconpy, linearsolve | DSGE solving and estimation in Python | References for the rational-expectations path |
| Dynare (MATLAB/Octave) | Standard DSGE toolkit | `.mod` model-block import only |
| gretl | EViews-like GUI econometrics (C) | UI reference, not a dependency |

The `Solver` interface lets irispie or ModelFlow serve as the first solving back end, replaced by native solvers only where they limit vintage handling, features users need or course licensing. (Superseded: native solvers were built; see Open decisions.)

## Extension API v1 (draft)

pymetrix grows through plugins that wrap other libraries or add methods. This contract guarantees that every plugin looks, scripts and behaves like core code: same dialogs, same result objects, same project files. Plugins declare `api="1"`; this section is the reference for that version.

**Principles**

- Our specification, data handling and results; the wrapped library does only the numerical work.
- Plugins describe their settings; they never contain UI code.
- Everything done in a dialog has an exact Python equivalent, and both replay identically from a project file.
- Every result records where it came from, including the library and its version.

### Extension points

| Extension point | Python contract | Entry-point group | Appears in the UI as |
| --- | --- | --- | --- |
| Estimation method | Frozen dataclass; `estimate(eq, data, *, span, where, weights, missing) -> FittedEquation`; class attributes `label`, `kind` (`"ls"`, `"iv"`, `"ml"`, `"panel"`) | `pymetrix.methods` | Estimate → Equation: Method list and Method settings |
| Model specification | Frozen dataclass with `fit(data, span=None, ...)` returning a result type (VAR-like systems) | `pymetrix.models` | Its own Estimate dialog |
| Error or variance specification | Frozen dataclass plus a backend `fit(method, eq, data, design) -> FittedEquation` | `pymetrix.variances` | Error-model choices in Estimate → Equation |
| Test | `Test` subclass dataclass; class attributes `name`, `null`, `applies_to`; `run(target) -> TestResult` | `pymetrix.tests` | Test → Run test, with a settings sub-form |
| Filter | Dataclass with `trend(y, f)` or `cycle(y, f)`, and `describe(f)` | `pymetrix.filters` | Data → Filter |
| Solver | Dataclass with `solve_period(system)` | `pymetrix.solvers` | Model → Solve |
| Data reader or writer | `read(source, **kw) -> Dataset`, `write(dataset, target, **kw)`, keyed by file extension or URI scheme | `pymetrix.io` | File → Load data |
| Result view | `sections(obj) -> list[(title, Table)]` for a result type | `pymetrix.views` | Tabs in the object window |
| Command (dialog) | `@command(menu=...)` dataclass with `run(session)` and `to_code()` | `pymetrix.commands` | A menu item |

### Declarative UI rules

Settings are dataclass fields. The frontends (Qt and Jupyter) build forms from the field types and `Widget` metadata, so a plugin author writes no interface code.

1. Every parameter of the wrapped function is exposed; none is silently fixed. Common settings go on the Main tab, rare ones are marked `advanced=True`.
2. Defaults equal the library's defaults, unless a different default is documented (for example `debiased=True`).
3. Every field has one line of help in economists' terms, with units where relevant. Fields with standard names (`cov`, `lags`, `trend`, `dist`, `seed`, `start`, `name`...) get the core's help automatically.
4. Settings use pymetrix vocabulary: `cov`, `lags`, `trend`, `span`, `dist`, `alpha`, `seed`. The wrapper maps them to the library's names.
5. Fixed choices are `Literal` types, so their labels are also valid values in scripts. The frontend decides the widget (three or fewer: radio buttons; more: dropdown).
6. Conditional fields use `show_if`; sections use `group`. Neither needs frontend code. `show_if` values may be a function, evaluated when the form is shown (for example the labels of all registered IV methods).
7. Numbers declare `min`, `max` and `step`. Small tolerances are typed as text, not spun. Optional numbers (`float | None`) are typed as text, empty meaning the default; text that is a number stays a number.
8. Field types are limited to `str`, `int`, `float`, `bool`, `None`, `Literal`, sequences of these, `Mapping[str, ...]` and nested settings dataclasses (such as `Cluster`), so every setting serialises to a project file. Settings made only in code (planned features, fitted objects) are marked `Widget("hidden")` and stay out of forms.
9. A command's settings sub-form (`Widget("options", source="variances")`) lists the chosen extension's fields, less those the main form sets (`exclude=(...)`). A registered entry may preset settings: GJR-GARCH is `GARCH` registered with `preset={"asym": 1}`.

### Results and reproducibility

- Return a core result type (`FittedEquation`, `TestResult`, `Dataset`, `Table`, `IRF`, `FEVD`). A new result type must register `sections` and document its attributes.
- Set `provenance.method["backend"]` to the library and version, for example `"linearmodels 7.0"`, and record the plugin name and version.
- Settings must round-trip: rebuilding an object from its generated code, or from its project-file JSON, gives an equal object.
- Randomness only through a `seed` setting; the same seed gives the same result.
- Use the core sample builders (`linear_design`, `explicit_design`) for formulas, lags, spans, weights and missing values; never reimplement them.
- Never change inputs; results are immutable.

### Help

Help is generated from the same declarations, so plugins need no separate help files. It has three levels:

| Level | Source | Shown as |
| --- | --- | --- |
| Usage | The class docstring (first line: what it does; the rest: how to use it) and each setting's help line | The dialog's Help page and tooltips |
| Reference | `wraps = "linearmodels.iv.IVLIML"` (the wrapped function or class, whose own docstring is shown) and `docs` (a URL) | A Reference section on the Help page |
| Background | `topics = ("instrumental-variables",)`, ids of help topics, and `references`, citations | Links to topic pages on the statistics and econometrics |

Topics are short pages added with `pymetrix.help.add_topic`; plugins can add their own. `pymetrix.help.search` searches every page, topic and course note. The conformance kit checks help: a docstring is required, `wraps` must resolve when its library is installed, topics must exist, and contrib plugins must link at least one topic or reference.

**In the UI.** Every generated dialog has a Help button (F1) that opens a pane beside the form: the dialog's page, then the page of each extension chosen in it, following the choices as they change. Links go to the wrapped library's documentation, to topics and to course notes; Back returns to the dialog's page. Help → Help contents is a browser with contents and search. Notebook forms have a Help toggle, and `show_help(...)` shows any page.

**Course material.** Instructors add a folder of lecture notes (`.md`) and example notebooks (`.ipynb`). Front matter (`title`, `course`, `links`, `references`; for notebooks also notebook metadata `"pymetrix"`) links each note to help pages: topic ids such as `garch`, or `group/name` such as `variances/GJR-GARCH` and `dialog/FitEquation`. A note then appears under Course material on those pages, in the contents and in search; notebooks are shown as their Markdown and code cells, with a link to open them. Relative links (slides, data) open the files. Folders are added in Help → Course material (remembered per user), listed in the `PYMETRIX_COURSES` environment variable (for labs), or installed as packages with a `pymetrix.courses` entry point. `pymetrix.testing.check_course` reports links that match no page, missing files, and files outside the course folder.

**Course material in projects.** A folder can be included in the session's project (`Session.attach_course`, or Include in project in the Course material dialog). Saving writes the whole folder (less checkpoints, caches and hidden files) under `course/<n>/` in the `.pymx` file and lists it in the manifest. Opening the project unpacks it to a cache folder named by its content hash (reused while the content is unchanged; paths that would escape the folder are refused) and adds it to the help. When the session is replaced, a project's courses leave the help unless they are also the user's own. An assignment's data, commands and notes therefore travel as one file.

### Registration, naming and packaging

Plugins register through entry points in their own `pyproject.toml`:

```toml
[project.entry-points."pymetrix.methods"]
liml = "pymetrix_linearmodels:LIML"
```

Each plugin package also exposes metadata the core checks on load:

```python
PLUGIN = PluginInfo(name="pymetrix-linearmodels", version="0.1.0", api="1",
                    requires={"linearmodels": ">=6"}, license="BSD-3-Clause",
                    maintainer="...")
```

- Package names follow `pymetrix-<library>` (for example `pymetrix-linearmodels`).
- Identifiers are qualified as `plugin:object` (for example `linearmodels:liml`). Menu labels stay short; the qualified name is shown only when two plugins use the same label.
- A plugin whose requirements are not installed is listed as unavailable, with its install command, rather than hidden.

### Conformance kit

`pymetrix.testing` provides pytest helpers that every plugin runs in its own test suite:

| Check | What it verifies |
| --- | --- |
| `check_form(cls)` | The form builds; defaults instantiate; every field has help; field types are allowed |
| `check_roundtrip(obj)` | Generated code and project JSON both rebuild an equal object |
| `check_result(result)` | Correct result type; provenance names the backend; `sections` render |
| `check_project(command)` | Save to a project, reopen, and get an identical result |
| `check_determinism(fn)` | The same seed gives the same result |
| `crosscheck(ours, reference, rtol)` | Agreement with the wrapped library called directly; required for contrib plugins |

### Versioning and stability

- The core follows semantic versioning and supports plugins written for the current and previous API major version.
- Deprecations warn for at least one minor release before removal.
- Plugins declare the library versions they support, and their CI tests the oldest and newest.

### Tiers

| Tier | Where it lives | Maintained by | Review standard |
| --- | --- | --- | --- |
| Core | `pymetrix` | The project | Full tests, cross-checks and documentation |
| Contrib | `pymetrix-*` packages in the `pymetrix` GitHub organisation | A named maintainer per package | Conformance kit, cross-check against the library, a documentation page |
| Third-party | Any author's package | The author | Conformance kit recommended; listed in a catalogue if it passes |

### Licence: BSD-3-Clause (decided 9 October 2026)

The core and the bundled reference plugins are BSD-3-Clause, matching the scientific Python ecosystem; every current dependency is compatible (NumPy, pandas, SciPy and statsmodels are BSD, pyarrow Apache-2.0, linearmodels and arch NCSA, PySide6 LGPL). Plugins choose their own licence and declare it in `PluginInfo`; wrappers for GPL libraries live in separate packages.

### Implementation status (9 October 2026)

- `pymetrix.plugins`: `PluginInfo`, one registry per extension point, entry-point loading with API checks, and `plugin_table()` (Help → Plugins), which lists plugins with missing requirements as unavailable, with their install command.
- Entry points naming a class register it; entry points naming a function call it (the route for readers, writers and result views).
- Menus, dialogs and settings sub-forms read the registries: Estimate → Equation (methods, variances), Data → Filter (filters), Model → Solve (solvers), Test → Run test (tests). Scripts import any plugin classes they use.
- `pymetrix.testing` (the conformance kit) runs on every built-in component in the test suite; the linearmodels and arch adapters (`pymetrix.ext`) pass at the contrib standard; `tests/sample_plugin.py` loads through entry points as a third-party package would.
- `pymetrix.help`: help pages and search built from the declarations, with the first background topics. The Help button on dialogs and the Help browser in the menu bar are the next UI step.

## Build plan and open decisions

Build in five stages, each usable on its own; the first proves the architecture end to end.

1. **Foundations:** `Dataset` with axes and metadata, the expression parser, `Equation.fit` with `OLS` (including fixed effects and clustered errors), `TestResult` and a few tests, the command registry, a minimal PySide6 form UI, Jupyter display and forms, Parquet and Excel input/output, pivot and aggregate.
2. **Macro models:** `Model` with block decomposition, `Newton` and `GaussSeidel`, `Scenario` including the exogenous/endogenous swap, stochastic simulation with draws, Monte Carlo and bootstrap.
3. **Multi-equation:** `VAR`/SVAR with identification schemes, `System`, `StateSpace` with Kalman filter and smoother, PCA and factor models (including nowcasting), quantile regression.
4. **Rational expectations:** steady state, linearisation, Klein/Gensys, perfect-foresight solving, `.model` and Dynare import.
5. **Bayesian and extensions:** `Bayes` with the JAX back end, panel GMM, discrete choice, mixed frequency (Chow-Lin, Denton, MIDAS).

**Deferred, with hooks in place:** rolling and real-time forecast evaluation, seasonal adjustment (X-13, STL), regime switching and time-varying parameters, optimal policy, occasionally binding constraints, higher-order perturbation and particle filters, moment matching.

**Excluded:** units algebra in equations, machine-learning estimators, spatial econometrics, a custom array type, distributed computing.

**Open decisions**

- [x] PySide6 desktop first, Jupyter notebooks for teaching (decided 9 October)
- [x] Native solvers rather than irispie or ModelFlow: Newton, Gauss-Seidel, perfect foresight and the QZ rational-expectations solution are built in (October)
- [ ] Align `Variable` and `Vintage` field names with common data-store schemas (SDMX, and institutional stores through plugins), so reader plugins map cleanly
- [x] Package name **pymetrix** and licence **BSD-3-Clause** (decided 9 October)
- [ ] Whether course participants need a no-install JupyterLite version
- [x] Keep the cross-checked native filters and tests rather than calling statsmodels, so the core needs only NumPy, pandas and SciPy (library audit, 11 October)

* [x] Standalone and for a broad audience (professional, training, academic), not tied to any institution's data systems or models; the built-in PSETS reader stub was removed, and institutional stores connect through reader plugins (decided 11 October)
* [ ] When to publish the repository as open source, and under which organisation name
