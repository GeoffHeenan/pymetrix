# User guide

A tour of pymetrix by topic, with short examples. The [README](../README.md) has the quick start;
model builders can start with [building and estimating models](modelling-quickstart.md);
the [course labs](../examples/course) work through complete exercises; the [How do I…](how-do-i/README.md) pages
take one task from start to finish, in the app and in code; the in-app help (F1) has a page for every dialog,
method and test.

## Example data and course material

`Dataset.example("sample_macro")` loads the simulated economy used in the
examples, quarterly 2000Q1-2024Q4, and `Dataset.example("sample_monthly")` monthly
indicators of the same economy for nowcasting (industrial production, retail sales,
employment and a survey, through 2025-03, each with its publication lag); 2025Q1 GDP is
left out, as the quarter to nowcast. Both are installed with the package
(`pymetrix.datasets.available()` lists them; `examples/make_sample_data.py` makes them). `examples/course/` is an example course: lecture notes and notebooks,
including:
- **Lab 2**, a student exercise from data to estimates, a model, a scenario and a saved and
  reopened project, with its instructor solution in `examples/solutions/`;
- **Lab 3**, a quarterly forecast round: a VAR forecast, judgement, revisions, a backtest, a
  structural fiscal scenario, a report and the project;
- **Lab 4**, pymetrix as a Python library: estimators side by side, a cross-check against
  statsmodels, a plugin, and dialog steps as plain Python. Every course notebook runs in the test suite.

`examples/forecasting/` is a forecasting course pack: six labs following the topics of Elliott and Timmermann,
*Economic Forecasting* (2016):
- loss functions and efficiency;
- comparing forecasts;
- combination and model averaging;
- density and probability forecasts;
- many predictors;
- breaks and nonlinearity.

Each lab simulates its own data, so the answers are known, and checks pymetrix by another route. Its
[README](../examples/forecasting/README.md) maps the book's chapters to pymetrix and to the labs.

## Continuous integration

GitHub Actions runs the tests on Windows, macOS and Linux with Python 3.11, 3.12
and 3.13, with every optional library installed, and on Linux with the core
dependencies only (tests that need an optional library are skipped). That full
set runs weekly and on demand; pull requests run Linux on Python 3.11 and 3.13,
the core-only set, the installed wheel and the oldest supported versions, and
each push to main adds Windows on Python 3.13.

## Defaults

Where pymetrix's defaults differ from EViews' or Stata's (robust and HAC standard
errors, ADF lag selection, VAR covariance, SUR, HP lambda...), the help page
**Defaults, and other packages' defaults** says how, and which option matches; [defaults](defaults.md) is the same
page here.

## Working with results

Every result is an ordinary Python object. The desktop window shows the same
contents as tabs (Summary, Coefficients, Statistics, Residuals, ...), each with
Copy (for Excel, with headers), Copy as (values only, Markdown, or the Python
that gives the table) and Export. Right-click a table to copy the selected
cells (with their row and column names) or the whole table for Excel, as
Markdown or as CSV, to sort the rows shown by the column under the mouse (smallest or largest first, numbers as
numbers, missing values last; **Back to the table's order** undoes it, and the result itself is unchanged), and,
where the table's buttons offer them, **Show as a heatmap** and **Add to report**. **Copy the step's code** copies the Python step that made
the result. **Data → Add results to dataset** turns residuals, fitted values,
volatility or forecast columns into series.

**The code behind a dialog.** Every dialog has **Show the code** under its fields: the Python the step will
record, as in History and the script, following the fields as they are filled in, with **Copy**. It is folded by
default; opened once, it stays open in every dialog until hidden. A figure's **Format…** form has the same: the
change as matplotlib's keyword arguments (`arrowprops={...}, color='navy'`), which an expert can also type over; what
is typed there is what OK runs. **Edit as code (every property)…** in the figure's right-click menu opens every
keyword of the item at once.

A result's last view, **Python access**, lists what it holds: each named field with its
type and size, the Python that reaches it (`fit.params`, `fit.stats["aic"]`,
`fit.provenance.warnings`, ...), and the command that made it; right-click a row
to copy its Python. In code:
`pymetrix.views.fields(fit, "fit")`. Tables render as text too:
`table.to_markdown()`, `table.to_csv(sep="\t")`.

A dataset's window opens with a status line (series, frequency, sample, rows,
series kept up to date, missing values), and its **Variables** tab is the data
dictionary: metadata, observations, first and last period, and gaps inside the
sample for each series.

```python
fit = Equation("c ~ y + c[-1]").fit(data)

fit.params.values, fit.params.se, fit.params.cov   # arrays, in fit.params.names order
fit.params["y"]                                    # one coefficient
fit.params.to_frame()                              # coefficients and standard errors (DataFrame)
fit.params.interval(0.05)                          # 95% intervals
fit.summary().to_frame()                           # the summary table as a DataFrame
fit.stats["r2"], fit.stats["aic"]                  # statistics: see fit.stats.keys()
fit.resid, fit.fitted                              # Series indexed by period (and entity)
fit.extra.keys()                                   # model-specific results (a GARCH fit's "volatility", IV first stages, ...)
fit.provenance.sample, fit.provenance.data_vintages

test = fit.test("breusch_godfrey", lags=4)
test.statistic, test.p_value, test.df, test.critical_values, test.details
test.reject(0.05)

var.coefs, var.sigma, var.irf(20).to_frame(), var.fevd(20).table("y").to_frame()
var.irf(20).paths                                 # the responses by name: columns "y <- shock", a row per horizon
var.residual_correlogram(12).plot()               # residual auto- and cross-correlations, ±1.96/sqrt(T) bands (VECMs too)
VAR(["oil", "y", "pi"], lags=2).fit(data, method=Restricted(block_exogenous=["oil"], zeros={"pi": ["y[-2]"]}))  # feasible GLS
hd = var.historical_decomposition()                # each shock's contribution to each variable, and the baseline
hd.contributions["y"], hd.baseline, hd.shocks; hd.plot(["y"])
data = data.with_series(u=fit.resid)               # keep a result as a series
```

Any `Table` converts with `.to_frame()`; any `Dataset` has `.frame`.

**Where the numbers come from.** Every result estimated from data has `provenance`: the
data vintages, the sample, the method and its settings, the seed, the pymetrix version,
the warnings raised while estimating, whether an iterative estimator converged, and a
fingerprint of the data used (the columns and periods, including those lags reached back
to). The same data give the same fingerprint on any machine, so two results can be
checked to come from the same numbers. Tests carry the provenance of what they tested.
It records what was used, not only what was asked for: `provenance.resolved` holds
choices made automatically (`{"hac_lags": 4}`; a VAR's lags chosen by AIC; the lag
an ADF test chose), and `sample_source` and `sample_definition` say how the rows
were chosen (`explicit`, the `active sample of the dataset` in full, conditions
included, or `all observations`). Each series records its source, its ID there, its
release date where the provider gives one, and when it was fetched; each load is
listed in `data.meta.extra["loads"]`.

Beside the version, `provenance.build` says which code that was: from a checkout of
pymetrix's source, the commit and whether files under the package differed from it
(`{"commit": "628beba5…", "dirty": False}`); from an installed package, `{"commit": None,
"note": "installed package (no git checkout): the version identifies the code"}`. The same
is written at each save in a project's `save_log` and in a saved result's manifest
(`pymetrix.build.build_info()` gives it; it is read once per process).

**Errors.** Everything pymetrix raises about its inputs is a `PyMetrixError`:
`DataError` (the data), `NotFoundError` (a series, object or test that does not exist),
`SpecificationError` (a formula or option that cannot be used), `ModelError`,
`ExpressionError` and `PlannedError` (a feature not built yet). Each is still the
built-in kind (`ValueError`, `KeyError`, `NotImplementedError`), so existing
`except ValueError:` code keeps working.

**Checking data.** `data.validate()` lists errors (rows that appear twice, periods
missing from the time axis, infinite values, text among numbers), warnings (missing
values inside a series' span, constant series, outliers) and notes (series without units
or a vintage, ragged edges); `report.ok`, `report.summary()`, and
`data.validate(strict=True)` raises a `DataError`. In the app: **Data → Check data…**.

## Matrices with names

```python
X, y = Matrix.design("c ~ y + c[-1]", data)        # regressors and dependent variable, rows named by period
b = (X.T @ X).inv() @ X.T @ y                       # rows: const, y, c[-1]
V = Matrix.cov(fit); V["y", "y"] ** 0.5             # a covariance matrix and a standard error, by name
A1 = Matrix.var_lag(var, 1); S = Matrix.var_sigma(var).chol()
U, s, V = X.svd()                                   # X = U @ s.diag() @ V.T, the names meeting
```

A `Matrix` keeps names for its rows and columns. `A @ B` needs A's columns to be B's rows, in the same order,
so a mistake in ordering is an error rather than a wrong number (a matrix without names meets anything of its
size). Element-wise `+ - * /`, `inv`, `pinv`, `solve`, `det`, `trace`, `rank`, `chol`, `eig`, `kron`, `vec`,
`diag` and `T` keep the names; `values` and `to_frame()` give NumPy and pandas. A matrix opens in a window as a
table and a heatmap.

**Powers:** `**` is element by element, as in NumPy, and `A.power(n)` is the matrix power (`A.power(-1)` the
inverse). For now `A ** n` on a square matrix with a whole n still gives the matrix power with a DeprecationWarning;
it will become element by element, so write `A.power(n)`, or `A * A` or `np.square(A)` for squares element by
element. `chol()` gives the lower factor L with `L @ L.T = A`; `chol(upper=True)` the upper R with `R.T @ R = A`, as
MATLAB's `chol`.

**Factors** name the dimension they make, so they multiply back: `svd()` gives U, s and V with columns s1, s2, ...;
`qr()` gives Q and R (q1, ...); `lu()` gives P, L and U with `A = P @ L @ U` (L's rows are A's rows in pivot order);
`schur()` gives T and Q with `A = Q @ T @ Q.T`; `qz(B)` gives AA, BB, Q and Z with `A = Q @ AA @ Z.T`. `eig()` works
on any square matrix: a symmetric one by `eigh` (real values, orthonormal vectors), any other by the general method,
whose eigenvalues may be complex (a matrix may hold complex numbers; it shows them as text, such as 0.4+0.84j).
`eigh()` and `eig(symmetric=True)` insist on the symmetric method, `eig(symmetric=False)` on the general one.

**Functions of a matrix:** `expm`, `logm`, `sqrtm` (named as the matrix), `norm("fro" | "nuc" | 1 | 2 | "inf")`,
`cond()`, `null()` (rows named by the columns), `orth()`, and `perp()`, the orthogonal complement used for a VECM's
alpha-perp and beta-perp: an m x (m - r) matrix with orthonormal columns, `perp().T @ A` zero and `[A perp]` of full
rank. Johansen defines it only up to an invertible factor, which cancels in the formulas it enters, such as
`b_perp @ (a_perp.T @ G @ b_perp).inv() @ a_perp.T`. Element by element, NumPy's functions keep the names
(`np.log(m)`, `np.exp(m)`, `np.sqrt(m)`, `abs(m)`, also `m.log()` and so on); `m.sum(axis=0)` and `m.mean(axis=1)`
sum or average each column or row.

```python
F = Matrix.vcat(Matrix.hcat(A1, A2), Matrix.hcat(I, Z))     # a VAR(2)'s companion form, names checked
Q = Matrix.zeros(F.rows, F.rows).set(["gy", "gc"], ["gy", "gc"], Sigma)
S = Matrix.lyapunov(F, Q)                                  # S = F S F' + Q: the unconditional covariance
K = Matrix.commutation(A.rows, A.cols)                     # K @ A.vec() is A.T.vec(), names and all
```

**Calculus and equations:** `vech()` and `Matrix.unvech(v)`, `Matrix.commutation(m, n)`, `Matrix.duplication(n)` and
`Matrix.elimination(n)` (given names instead of sizes, they are named to meet `vec` and `vech`);
`Matrix.lyapunov(A, Q)` solves X = A X A' + Q (`continuous=True`: A X + X A' + Q = 0), `Matrix.sylvester(A, B, C)`
solves A X + X B = C, and `Matrix.riccati(A, B, Q, R)` the discrete algebraic Riccati equation (the Kalman filter's
steady-state covariance with A = T', B = Z'). **Assembly:** `Matrix.hcat` and `Matrix.vcat` need the shared names to
meet and refuse clashing ones; `m.set(rows, cols, value)` replaces an element or a block by name in a new matrix;
`outer` and `inner`; `Matrix.from_dataset(data, ["c", "y"])` and `m.to_dataset()` (rows named by period) move between
matrices and datasets. A decomposition opens in a window with a table for each factor.

**In the app.** **Model → Matrix** makes a named matrix as a step, three ways:

- **Expression…**: a line in the syntax above, using the session's objects by name, such as
  `(X.T @ X).inv() @ X.T @ y`, `Matrix.cov(fit)`, `V.chol(upper=True)`, `A.kron(Matrix.identity(3))` or `A.svd()`
  (a decomposition). `Matrix`, `np` and the session's objects are there; the usual functions are methods
  (`A.inv()`, not `inv(A)`). The expression is checked before it runs: one Python expression of names, attribute
  access, calls, operators, indexing and literals. Imports, `lambda`, comprehensions, attributes starting with `_`,
  `exec`, `eval`, `open` and anything that reads or writes files are refused, with a message saying which, and an
  unknown name is reported with the nearest one there is. The dialog checks as you type (a moment after typing
  stops): the problem shows under the field, a parse error with its column, and OK stays disabled until it is
  fixed; when the expression is cheap to compute it is evaluated, and the result's shape and row and column names
  (or a decomposition's factors) are shown. Large inputs, large numbers and arithmetic on numbers alone are checked
  but not evaluated while typing.
- **From data or a result…**: a fit's coefficients (a column) or the covariance matrix of its estimates, a VAR's
  lag matrix or error covariance, a dataset's series over a span (rows named by period), or a formula's regressors
  and dependent variable (two matrices, `X, y = Matrix.design("c ~ y", data)`).
- **Enter values…**: a grid. Set the rows and columns, type the numbers, or paste them (Ctrl+V or Paste) from Excel
  or as text with tabs, semicolons, commas or spaces between the cells (decimal commas are read where commas do not
  separate the cells); a first row or column of names gives the names. Double-click a row or column header to
  rename it. It records the matrix written out: `M = Matrix([[1, 0.5], [0.5, 1]], rows=['a', 'b'], cols=['a', 'b'])`.

A matrix's window has a **Matrix ▸** menu: Transpose, Inverse, Pseudo-inverse, Cholesky factor (lower or upper),
Decompose (SVD, eigenvalues, QR, LU, Schur), Power… and Solve Lyapunov… (X = A X A' + Q, with Q chosen from the
session's matrices of the right size) each make a new named matrix (`S_inv = S.inv()`), asking for its name; Copy as
LaTeX gives a plain `bmatrix` of the numbers, Copy as LaTeX table a `tabular` with the row and column names (for a
matrix whose names matter); Copy as CSV, Copy as tab-separated (for Excel) and Export to Excel… take it elsewhere. A
decomposition's window makes each factor a matrix (`U = S_svd.U`). A fitted equation, system, VAR or VECM offers
its matrices from its right-click menu in the workspace (**As matrix**: coefficients, covariance; a VAR's lag
matrices and error covariance) and its window's **Next ▸ As a matrix…**.

Every one of these is a step whose code is the line it records, so the script gives the same matrix; it is saved
with the project, which opens without asking to trust it (the expression can only compute), and recomputed when the
objects it uses change. The help topic "Matrices with names" has the same in brief.

## NumPy, pandas, SciPy and statsmodels

**The short path.** A pandas DataFrame with a `PeriodIndex` (or dates with a frequency) goes straight into any
estimator, and `to_pandas()` is the way back out. Metadata, vintages and flags are optional:

```python
fit = Equation("y ~ x").fit(df)                    # a plain DataFrame works where a Dataset does
data = pm.dataset(df)                             # the same conversion, explicitly (pm.as_dataset)
df = data.to_pandas()                             # every series and row, plain pandas, no metadata
fit = data.over("1990Q1:2019Q4").fit(Equation("c ~ y"), method=OLS(cov="hac"))   # a sample, in a chain
fits = pm.fit_each(["c", "i"], "d{s} ~ d{s}[-1]", data)   # lists wherever series names go
fit.coefficients(); fit.statistics(); fit.observations()  # tidy tables, on every estimate
```

Spans can be written as text, `"1990Q1:2019Q4"` (`"1990Q1:"` and `":2019Q4"` leave one end open), wherever a
`(start, end)` pair is accepted.

```python
frame = data.to_frame(["c", "y"])                 # the active sample's rows, as commands use them; sample="all" for every row
X = data.to_numpy("c, y", missing="drop")         # an array; missing: keep, drop or raise
y, X = data.design("log(c)", ["log(y)", "c[-1]"]) # labelled y and X (const first) on the rows pymetrix's own OLS would use
res = sm.OLS(y, X).fit()                          # any outside estimator
fit = FittedEquation.from_statsmodels(res, data, command="sm.OLS(y, X).fit()")   # an ordinary pymetrix result
fit.summary(); fit.to_frame("coefficients")
post.to_frame("Summary"); irf.to_frame("Selected horizons")  # any result: any table view, as its window's tabs

data = data.add_series("gap", my_array)           # an array (one value per row) or a pandas Series; source "user:numpy"
data = data.apply_np("lc", np.log, "c", source_expr="np.log(c)")   # computed by your function, recorded as user code
data = data.transform_each(["c", "y"], np.log, prefix="l", source_expr="np.log({s})")
data = Dataset.from_frame(df, source="user:pandas")
```

Everything that leaves a dataset says in `.attrs["pymetrix"]` how its rows were chosen (an explicit
span, the active sample in full, or all observations) and how many were dropped. `design` builds `y` and
`X` with pymetrix's own formula language and rules, lags within entity included, so an outside
estimator sees exactly the rows pymetrix's estimators would. Coming back in, `add_series` refuses a
name the dataset already has unless `overwrite=True`, and a series of another frequency (convert it
with `to_freq` first); every series added records where it came from (`user:pandas`, `user:numpy`) and
when, and `apply_np`, `apply_pd` and `transform_each` record the function used, or `source_expr` when
given. Such series are computed once, like `with_series`; use `derive` for series that follow their
inputs. `from_statsmodels` takes regressions with `params` and `cov_params` (OLS, WLS, GLS, GLM,
Logit and Probit, ...): coefficients and their covariance, so standard errors, t or z and p values as
statsmodels reports them, fitted values and residuals, R-squared and information criteria, the
covariance type and convergence, and provenance (vintages and a fingerprint when `data` is given).
Its standard errors are statsmodels' own: its HAC covariance, for one, has no small-sample correction
unless `use_correction=True`, while pymetrix's has (they agree when it is on). To match statsmodels'
default instead, `OLS(cov="hac", hac_correction=False)` (and `IV(...)`) leaves the n / (n - k) factor out.

### Regressions without a formula

Code that runs many small regressions (in a loop, a simulation, a model-building routine) can skip the formula and
the dataset: `pymetrix.ols` takes the dependent series and a table of regressors. It is a Python function, for code;
the dialogs estimate equations by formula.

```python
import pymetrix as pm

fit = pm.ols(y, X)                                 # y a Series, X a DataFrame with the same index
fit = pm.ols(y, X, cov="hac", hac_lags=4)          # standard errors as OLS(cov=...): classic, white, hac
fit = pm.ols(y, X, cov=pm.Cluster("firm"), clusters=firms)   # clusters: a Series or DataFrame of groups (or index levels)
fit = pm.ols(y, X, constant=False, weights=w)      # no intercept; weighted least squares
fit = pm.ols(y_array, X_array, names=["k", "l"])   # numpy: y is "y", the columns x1, x2... unless named
fit.params, fit.summary(), fit.test("breusch_godfrey", lags=4), fit.diagnostics()
```

The result is the `FittedEquation` that `Equation("y ~ x1 + x2").fit(data, method=OLS(...))` gives on the same data,
computed by the same least-squares code: the same numbers, views, tests and diagnostics. Its equation is that formula,
written from the names (so the names must be plain series names: letters, digits and `_`), and it forecasts and
predicts on any dataset holding those series. Pandas inputs must share their index. A `PeriodIndex` (or dates with a
frequency) makes the data a time series, as in a dataset (Durbin-Watson, HAC, Breusch-Godfrey, Chow); anything else
is a cross-section. Rows are put in index order, and rows with a missing value are dropped (`missing="raise"`
refuses them).

Provenance is as complete as for a formula fit: a fingerprint of the data used, the rows used and why each other row
was not (`dropped_rows`: missing values), the method and backend. The one difference is
`provenance.vintage_request == {"policy": "direct"}`: the data were passed directly to the call, not taken from a
dataset. In a session (the Immediate or Code window) the call is recorded as the Python you typed; the arrays are
saved with the project only if you keep them there yourself (as objects of the session or series of a dataset).

Speed: 10,000 regressions of 40 quarterly observations on 3 to 6 regressors and a constant
(`benchmarks/ols_arrays.py`, Python 3.13, pandas 3.1, one BLAS thread, on a busy 4-core Linux machine):

| Path | Total | Each | Speed-up |
| --- | --- | --- | --- |
| `Equation(...).fit(as_dataset(frame))` | 60.6 s | 6.06 ms | 1x |
| `pm.ols(y, X)`, Series and DataFrame | 10.1 s | 1.01 ms | 6.0x |
| `pm.ols(y, X)`, numpy arrays | 7.8 s | 0.78 ms | 7.8x |

The formula path spends its time building the dataset, evaluating each term of the formula over it, selecting the
sample with pandas and working out which rows were used; `ols` does the same checks (missing values, collinearity,
HAC only on time series) and the same provenance on the arrays themselves.

### Breusch-Godfrey without a constant

`fit.test("breusch_godfrey", lags=p)` regresses the residuals on the equation's own regressors and their lags 1 to p
over all n observations, with the lags before the first one set to zero, and reports n R-squared, R-squared measured
about the residuals' mean (Breusch 1978; Godfrey 1978; Greene, *Econometric Analysis*, 8th ed., chapter 20). When the
equation has a constant, statsmodels gives the same number. When it has none they differ:
statsmodels' `acorr_breusch_godfrey` always adds a constant to the auxiliary regression. `auxiliary_constant=True`
does the same, and matches statsmodels; with a constant in the equation it changes nothing. In the app: Test → Run
test, Breusch-Godfrey, **Add a constant to the auxiliary regression** under Test settings.

```python
fit = Equation("c ~ y - 1").fit(data)
fit.test("breusch_godfrey", lags=4)                            # pymetrix's convention (unchanged)
fit.test("breusch_godfrey", lags=4, auxiliary_constant=True)   # statsmodels' convention
```

Every result gives any of its table views as a DataFrame with `to_frame(view)`: the tabs of its window in
the app, by name and in any case (`pymetrix.views.titles(result)` lists them). With no name you get the
result's own default: the coefficients of a fitted equation, the responses of an impulse response, the
draws of a posterior, and otherwise the first table. A figure's numbers come from `result.plot().data()`.
In a notebook, results show as their first table, figures as pictures, and datasets as their data with a
one-line description.

## Derived series

A derived series keeps the recipe that made it and is recomputed when a series
it uses changes; anything derived from it follows in turn.

```python
data = data.derive(dly="dlog(y)", share="c / y", pct="share * 100")
data = data.filter("y", HP())                  # y_trend and y_cycle follow y
data = data.with_series(y=revised_y)           # dly, share, pct, y_trend, y_cycle recomputed
data.derived                                   # {name: recipe}
data.freeze("pct")                             # keep the values, drop the recipe
```

Expressions have `log`, `exp`, `sqrt`, `abs`, `sign`, `step` (1 where the
argument is at least 0, else 0), the differences `d`, `dlog` and `pch`, and
`max(a, b, ...)` and `min(a, b, ...)`, which take the larger or smaller value
period by period: `max(c - 900, 0)` in data, or an effective lower bound
`i ~ max(i_rule, floor)` in a model. With one argument, `max(y)` and `min(y)`
are the statistic over the series; grouped, `max(y, by=country)`.

**Rounding, missing values and choices:**
- `round(x)` and `round(x, n)` round to n decimals (n negative rounds to tens, hundreds...), with halves away from
  zero, as MATLAB and Excel do (`round(2.5)` is 3; NumPy's `round`, like R's, takes halves to the even neighbour
  and would give 2). `floor(x)` and
  `ceil(x)` round down and up, and `log10(x)` is the base-10 logarithm.
- `na` is the missing value; `isna(x)` is 1 where x is missing and 0 where not (never missing itself);
  `fillna(x, v)` is v where x is missing.
- `iif(c, a, b)` is a where the condition c holds (is not 0), b where it is 0, and missing where c is missing:
  `iif(gy > 0, gy, na)` keeps the positive growth rates.

In a model, rounding is flat where it does not jump, as conditions and `step` are, so a model using it linearises;
`iif` takes the derivative of the branch it takes.

**The calendar and random draws** come from the data's rows, so they work on data (Generate series, derived series,
`data.eval`, regressors written in an equation) and models refuse them:
- `year()`, `quarter()`, `month()` and `day()` are those of each period's first day (`month()` of a quarter is 1, 4, 7
  or 10); `obsnum()` numbers the observations from 1, within each entity of a panel, in time order.
- `normrnd(mu, sigma, seed=s)` and `unifrnd(a, b, seed=s)` draw one number a row (the parameters default to 0 and 1).
  `seed=` is required: a formula gives the same draws each time it is computed and when a project is replayed. The
  draws come from the generator `set_random_generator` chose. Being values rather than a recipe, they are made with
  Generate series (`with_series`), not `derive`.

```python
data = data.with_series(q4="quarter() == 4", trend="obsnum()", noise="normrnd(0, 0.5, seed=1)")
fit = Equation("y ~ x + (year() >= 2009)").fit(data)
```

**Distribution functions**, named as in MATLAB and Dynare, work in series expressions, equations, measurement
maps and model text; arguments may be expressions and parameters, and parameters with defaults may be left out:

| family | density (mass), cdf, inverse cdf | parameters (MATLAB's parameterisation) |
|---|---|---|
| normal | `normpdf`, `normcdf`, `norminv` | `(x, mu = 0, sigma = 1)` |
| Student's t | `tpdf`, `tcdf`, `tinv` | `(x, df)` |
| chi-squared | `chi2pdf`, `chi2cdf`, `chi2inv` | `(x, df)` |
| F | `fpdf`, `fcdf`, `finv` | `(x, d1, d2)` |
| gamma | `gampdf`, `gamcdf`, `gaminv` | `(x, a, b)`: shape a, **scale** b (mean a*b) |
| beta | `betapdf`, `betacdf`, `betainv` | `(x, a, b)` |
| lognormal | `lognpdf`, `logncdf`, `logninv` | `(x, mu = 0, sigma = 1)` of log x |
| uniform | `unifpdf`, `unifcdf`, `unifinv` | `(x, a = 0, b = 1)` |
| exponential | `exppdf`, `expcdf`, `expinv` | `(x, mu = 1)`: the **mean**, not the rate |
| Poisson | `poisspdf`, `poisscdf`, `poissinv` | `(k, lambda)` |
| binomial | `binopdf`, `binocdf`, `binoinv` | `(k, n, p)` |

`erf` and `erfc` are the error function and its complement. The `inv` functions are quantiles (`norminv(0.975)` is
1.96); `poissinv(p, lambda)` and `binoinv(p, n, p)` are MATLAB's, the smallest integer k with cdf(k) >= p (0 at
p = 0; at p = 1 infinite for the Poisson and n for the binomial). Invalid parameters give missing values, as in MATLAB. Models using these functions are linearised with
their exact derivatives, in x and in the parameters where the derivative is simple (normal, lognormal, uniform,
exponential, a gamma's scale, a Poisson's lambda, a binomial's p); in the degrees of freedom of t, chi-squared and
F, the shapes of gamma and beta, or a binomial's n, linearisation stops with a message, as it does through `poissinv` or `binoinv` (steps, with no
derivative). A probit-style rule:

```python
Model([Equation("y ~ rho*y[-1] + 0.2*p[+1] + e", params=["rho"]),
       Equation("p ~ normcdf(a + b*y)", params=["a", "b"])], params=dict(rho=0.5, a=-0.5, b=2.0))
```

**Moving-window statistics** use the last n periods, within each entity of a panel, and are missing until a full
window is there (or when a value in it is missing, a gap in the dates included): `movavg(x, n)`, `movsum`,
`movsd`, `movvar` (n - 1 denominator), `movmin`, `movmax`, and between two series `movcov(x, y, n)`,
`movcorr(x, y, n)` and `movbeta(y, x, n)`, the slope of a rolling regression of y on x (with a constant):

```python
data = data.derive(vol="movsd(dlog(p), 20)", beta="movbeta(r_stock, r_market, 60)")
```

**Centred windows** put the window around t: `cmovavg(x, 5)` averages x[+2] ... x[-2] (an even n has one more lag
than leads, `cmovavg(x, 4)` is x[+1] ... x[-2], as pandas' `rolling(n, center=True)`), missing at both ends until the
window is full; every moving function has one (`cmovsum`, `cmovsd`, `cmovvar`, `cmovmin`, `cmovmax`, `cmovcov`,
`cmovcorr`, `cmovbeta`). In a model the leads are leads: the model is forward-looking and is solved as such.
**Expanding windows** use every period up to t, within each entity: `cumavg(x)`, `cumsum`, `cumsd`, `cumvar`,
`cummin`, `cummax`, `cumcov(x, y)`, `cumcorr(x, y)` and `cumbeta(y, x)`, as pandas' `expanding()` (missing values
skipped). They have no fixed lags, so they work on data (derived series, `data.eval`, regressors) and are refused in
models.

```python
data = data.derive(trend="cmovavg(y, 9)", record="cummax(y)", running_mean="cumavg(dlog(p))")
```

**Conditions** give 1 where they hold and 0 where they don't:
- comparisons: `<`, `<=`, `>`, `>=`, `==` and `!=`;
- joined with `and`, `or` and `not`;
- for example `recession = gy < 0` or `boom = gy > 5 and gy[-1] > 5`.

**Missing values:** where a value in the condition is missing, the result is missing too, not 0. So a dummy never
silently turns a gap into a "no".

**Uses:**
- **dummies:** made with **Data → Generate series**;
- **regressors** written in the equation, such as `c ~ y + (gy < 0)`;
- **switches in models**, such as `i ~ (infl > 2) * rule + (infl <= 2) * floor`;
- **shading in charts:** `shade="recession"`.

`data.keep("y", "c")` keeps some series and `data.drop(...)` removes some;
`data.select(where=..., span=...)` keeps some periods or rows.

### Samples

```python
precrisis = Sample(end="2007Q4")
core = Sample(("1995Q1", "2019Q4"), country=["DEU", "FRA", "ITA"])
large = Sample(where="pop > 1e6")
by_country = Sample(windows={"DEU": ("1991Q1", None), "*": ("1980Q1", None)})

data = data.with_sample(core & large)     # the active sample, as EViews' smpl
model.fit(data)                           # estimation, tests, tables and graphs use it
model.fit(data, span=precrisis)           # an explicit span or Sample replaces it for one call
data = data.with_sample(None)             # all the data again
core.describe(data)                       # "380 of 1200 rows, 3 country, 1995Q1 to 2019Q4"
```

A sample belongs to the dataset (it is saved with it) and the line that sets it is
recorded, so scripts and projects replay it; there is no hidden global sample. It only
selects rows: lags reach back before it and never cross entities, and generated or
derived series are computed over all the data, unless the sample is given to `with_series`
(below).
Conditions are kept as text and applied each time; results record the sample they
used. A condition (`where=`, here and in estimation) is pandas' row-condition syntax, checked before it runs: column
and axis names (in backticks if they have spaces), numbers, text and dates in quotes (`time >= '2000Q1'`), comparisons,
`in` and `not in` with a list (`country in ['DEU', 'FRA']`), `and`, `or`, `not` (or `&`, `|`, `~`), arithmetic, and
the functions `abs`, `sqrt`, `exp`, `log`, `log10`, `log1p`, `expm1`, `floor`, `ceil`, the trigonometric ones and
`arctan2`. Anything else, such as `@` references to Python variables, `.` attribute access or other functions, is
refused with an error saying what was found, so a project from someone else cannot run code through a condition. `&`, `|` and `~` combine samples. In the app: **Data → Sample → New sample…** names one,
**Data → Sample → Set sample…** makes it a dataset's active sample; the data window, its graphs
and the start and end fields of every dialog then show that sample. Results estimated
on the old sample are then out of date: **Recalculate** estimates them again on the new one.

**Assigning within a sample.** `with_series(..., within=...)` assigns only the rows a sample selects and keeps the
others' values, as EViews' `smpl 2000q1 2010q4` followed by `y = ...` does. `within` is a `Sample` (periods,
entities, `where=` conditions, windows per entity), a `(start, end)` span or `"start:end"`. A new series is missing
outside the sample; a value of `None` makes a series of missing values (EViews' `series y`), or missing values in the
sample's rows:

```python
data = data.with_series(y=None)                                          # a series of missing values
data = data.with_series(y="x / 2", within=("2000Q1", "2010Q4"))          # only 2000Q1 to 2010Q4
data = data.with_series(gdp="gdp[-1] * 1.02", within=Sample(start="2025Q1"))   # extend gdp by 2% a quarter
data = data.with_series(d="1", within=Sample(where="u > 8", country=["FRA"]))  # rows of a panel
```

Lags reach outside the sample. A formula that uses the series it assigns is computed one period at a time, each
period seeing the values assigned before it, as in EViews: the third line compounds from the last value before
2025Q1. A lead, or the series at the same period, sees the values before the call. Without `within` a formula is
computed once over every row from the values before the call, as before. In the app, **Data → Generate series…**
has **Within** under Advanced: a named sample, or a first and last period. **Keep updated** (a derived series) and
**Within** do not go together: a derived series follows its formula in every period. `with_each(..., within=...)` does
the same for each of several series.

**An empty dataset.** `Dataset.empty("Q", "1990Q1", "2030Q4")` (**File → New dataset…**; EViews' `wfcreate q 1990q1
2030q4`) has every period from the first to the last and no series; its time axis is named `time`. The frequency is
`Y`, `Q`, `M`, `W` or `D` (`Y-JUN` for a July-June year; `A` reads as `Y`). Fill it with `with_series`, or type
values in with the spreadsheet. The step is recorded, so the script and the project rebuild it.

`with_series` stays static (like EViews' `genr`); `derive` is dynamic. In code both add or
replace; a replaced series keeps its label and units but not its source or vintage, which
described the old values. The dialogs that add series (**Generate series**, **Derived
series**, **Add to dataset**, **Derive for each series**) refuse a name the dataset already
has unless **Replace existing series** is ticked. A series of another frequency is refused rather than
turned into missing values: a dataset has one frequency, so convert it first (`to_freq`;
**Data → Transform → Convert frequency…**) or keep it in a dataset of its own. A panel's entities
add up to totals with `aggregate` (**Data → Panel → Aggregate over entities…**). Recipes
are saved in Parquet files and projects. Taking a subset or converting the
frequency keeps the values but drops the recipes. Two-sided filters revise
their history when the series changes, as with a new release. A derived
series' vintage is that of its most recent input. In the app: **Data → Derived
series…**, and **Keep up to date** in **Data → Transform → Filter (trend and cycle)…** (on by default; the
trend and cycle are added to the dataset itself).

**One formula for several series.** `with_each` makes a series for each of several, from a name
template and a formula template:

```python
data = data.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))")   # gdp_g, cons_g, inv_g
data = data.with_each("ip_*", "{1}_growth", "100*dlog({x})")         # ip_us -> us_growth, ip_de -> de_growth
data = data.with_each("*_sa", "{x}_r", "{x} / cpi * 100", derived=True)
from pymetrix.data.each import preview_each
preview_each(data, "ip_*", "{1}_growth", "100*dlog({x})")   # the rows and their checks; nothing made
```

The series are listed (spaces or commas), matched by wildcards (`*` any run of characters, `?` one) or both;
`{x}` is each series and `{1}`, `{2}`... the parts its wildcard matched (each `*` as much as it can, left
first); `{{` and `}}` are literal braces. Names the dataset has already are refused unless `replace=True`. It
is recorded, and written in scripts, as the one `with_series` (or, with `derived=True`, `derive`) call it
makes, every new series written out:
`data = data.with_series(us_growth='100*dlog(ip_us)', de_growth='100*dlog(ip_de)')`, so the script replays
the same whatever series are added later. In the app: **Data → Generate series…** with **For each** filled
in (or pick series in a dataset's series list and press **Generate for each…**); Series and Formula are then
the templates, and a preview lists every new series, its formula and any problem (a name that is not valid,
a formula that does not parse or uses a missing series, a name made twice, a series that exists: tick
**Replace existing series**) before OK. On a panel each is computed within each entity, as any formula is.

### Descriptive statistics and correlations

```python
data.describe()                                     # count, mean, std, min, quartiles, max, skewness, kurtosis
data.describe(["dly", "dlc"], percentiles=(0.05, 0.5, 0.95))
data.corr(["dly", "dlc", "dli"], method="spearman")  # or "pearson" (the default), "kendall" (tau-b)
data.cov(["dly", "dlc"])                            # n - 1 denominator, each pair over the periods both have
data.partial_corr(["dlc", "dli"], given=["dly"])   # correlations of residuals on a constant and dly
data.crosscorrelogram("dly", "dlc", lags=12)        # corr(dly(t), dlc(t+k)), k = -12 ... 12; band ±1.96/sqrt(T)
data.autocov("dly", lags=12)                        # autocovariances (sums over T) and the autocorrelations
data.coskewness(["dly", "dlc"])                     # the n x n² matrix M3 of standardized co-skewness
data.cokurtosis(["dly", "dlc"], form="pairs")       # mean(e_i³ e_j) for each pair; also standardized=False
data.systematic_comoments(["dlc", "dli"], market="dly")  # beta, co-skewness, co-kurtosis and their betas
```

Skewness and kurtosis are the bias-corrected sample statistics (G1 and G2: pandas' `skew` and `kurt`, Excel's
SKEW and KURT, scipy's with `bias=False`); kurtosis is *excess* kurtosis, 0 for a normal. All use the active
sample. In the app: **Data → Statistics → Descriptive statistics…** and **Correlations…** (correlations,
covariances, partial correlations, co-skewness, co-kurtosis or systematic co-moments), and **Test →
Cross-correlogram…** (leave y empty for autocovariances); each records the call it stands for.

Co-skewness and co-kurtosis are the higher-moment counterparts of the covariance matrix used in portfolio choice
and asset pricing. With e the deviations from the means divided by the standard deviations, over the periods every
series has, co-skewness is S_ijk = mean(e_i e_j e_k) and co-kurtosis K_ijkl = mean(e_i e_j e_k e_l): the default
`form="matrix"` gives all of them as M3 = E[e (e ⊗ e)'] (n x n², columns labelled `"j,k"`) and M4 (n x n³);
`form="pairs"` the n x n tables mean(e_i² e_j) and mean(e_i³ e_j) usually reported for two assets. Moments divide by
T (the finance convention, as R's PerformanceAnalytics `M3.MM` and `M4.MM`), so the diagonal is each series'
skewness g1 and kurtosis b2 (not in excess: 3 for a normal) with the T denominator, scipy's with `bias=True`;
`describe()` shows the bias-corrected G1 = g1 √(T(T−1))/(T−2) and excess G2 = ((T+1)(b2−3)+6)(T−1)/((T−2)(T−3)).
The standardized versions do not change when a series is shifted or rescaled. `systematic_comoments` relates each
series to a market series: beta, the standardized co-skewness mean(d_i d_m²)/(σ_i σ_m²) and co-kurtosis
mean(d_i d_m³)/(σ_i σ_m³), and the co-skewness and co-kurtosis betas mean(d_i d_m²)/mean(d_m³) and
mean(d_i d_m³)/mean(d_m⁴) (Kraus and Litzenberger 1976; Harvey and Siddique 2000).

## Forecast rounds and backtests

Task pages: [Run a forecast round: baseline, judgement, scenarios, report](../src/pymetrix/howto/run-a-forecast-round.md) and [Check a forecast's record (backtest)](../src/pymetrix/howto/check-a-forecasts-record.md).


```python
rnd = ForecastRound.from_forecast("2026Q3 baseline", var.forecast(data, ("2026Q4", "2028Q4")), data=data)
rnd = rnd.adjust("gdp", {"2026Q4": -0.3}, reason="port strike")   # judgement, logged with the path before
rnd.plot(data)                                                     # fan chart: 30, 60 and 90 per cent
store = ForecastStore("forecasts"); store.save(rnd)
store.compare("2026Q2 baseline", "2026Q3 baseline")                # revisions between rounds
store.accuracy("gdp", data)                                        # the rounds' real-time track record

bt = backtest(var, data, ("2015Q1", "2022Q4"), horizon=8)          # re-estimate and forecast at each origin
bt.accuracy(); bt.compare(bvar_bt, "gdp", horizon=4)               # Diebold-Mariano (HLN)
```

`backtest(..., vintages={origin: dataset_as_released_then})` makes the evaluation
real-time (in **Forecast → Backtest…**, the **Vintages** field: `2015Q1=v2015q1, ...`, one
dataset for each origin), and `window=40` estimates on the last 40 periods at each origin
(a rolling window) instead of all of them. The **Forecast** menu has Make round, Adjust round,
Save round, Compare rounds, Backtest, Combine forecasts, Compare backtests,
Compare many forecasts, Superior predictive ability and Binary forecast scores.

### Evaluating forecasts

A backtest keeps each forecast's predictive standard deviation and scores its density
against what happened: the log score (higher is better), the CRPS (lower is better, in the
variable's units) and the PIT. Forecasts with standard errors are scored as normal
densities; Bayesian VARs from their simulated draws. Dynamic forecasts from equations with
a lag of their own target have no standard errors yet, so no density.

```python
bt.density_accuracy()                      # log score, CRPS, coverage of the 50% and 90% intervals
bt.evaluation()                            # p-values: efficiency, PIT calibration, interval coverage
bt.efficiency("gdp", 4)                    # Mincer-Zarnowitz: outcome = a + b forecast, a = 0, b = 1
bt.pit_test("gdp", 1); bt.plot_pit()       # Berkowitz; PIT histograms
bt.interval_test("gdp", 1, coverage=0.9)   # Christoffersen: coverage and independence of the misses
bt.compare(other, "gdp", 4, loss="crps")   # Diebold-Mariano on any loss: squared, absolute, crps, log score
bt.compare(other, "gdp", 1, test="giacomini-white")
small.compare(large, "gdp", 1, test="clark-west")    # nested models
bt.encompasses(other, "gdp", 4)            # Harvey-Leybourne-Newbold

combo = combine_forecasts({"var": var_bt, "bvar": bvar_bt, "ols": ols_bt}, method="inverse_mse")
combo.weights_table(); combo.density_accuracy()      # a combination is a backtest too

spa(ols_bt.losses("gdp", 1), {"var": var_bt.losses("gdp", 1)})      # does any model beat the benchmark?
ols_bt.spa({"var": var_bt}, "gdp", 1)      # the same from the backtests (Forecast → Superior predictive ability…)
model_confidence_set({n: b.losses("gdp", 1, "crps") for n, b in backtests.items()})
```

Combination weights (equal, inverse MSE, or regression weights summing to one) are
estimated in real time: at each origin only from outcomes known by then. With
non-negative weights the combination's density is the linear pool of the members'
densities. The tests are also functions in `pymetrix.stats` for any forecasts
(`mincer_zarnowitz`, `encompassing`, `giacomini_white`, `clark_west`, `christoffersen`,
`berkowitz`, and the scoring rules). SPA and the model confidence set need the arch
package. These follow Elliott and Timmermann, *Economic Forecasting* (2016),
chapters 14 to 18.

Asymmetric losses (chapter 2) price over- and under-forecasts differently:

```python
from pymetrix.stats import linex_loss, linlin_loss, stepm

bt.compare(other, "gdp", 1, loss="lin-lin", tau=0.8)   # under-forecasts cost four times as much
bt.compare(other, "gdp", 1, loss="linex", a=0.5)       # exponential in under-forecasts
linlin_loss(errors, tau=0.8); linex_loss(errors, a=0.5)   # for any errors
stepm(bt.losses("gdp", 1), {"var": var_bt.losses("gdp", 1), "ets": ets_bt.losses("gdp", 1)})   # Romano-Wolf
```

For probability forecasts of events, such as a recession next quarter from a Logit, and
for the direction of change:

```python
from pymetrix.stats import binary_scores, pesaran_timmermann, roc_curve

binary_scores(probabilities, outcomes)     # Brier (against the base rate's), log score, ROC area, Kuipers
binary_scores("p_rec", "rec", data=data)   # two series of a dataset (Forecast → Binary forecast scores…)
roc_curve(probabilities, outcomes)         # false and true positive rates by threshold
pesaran_timmermann(actual, forecast)       # are the signs right more often than chance?
```

### Conditional forecasts

```python
var.conditional_forecast(data, ("2025Q1", "2026Q4"), {"r": {"2025Q1": 4.0, "2025Q2": 3.75}})
bvar.conditional_forecast(data, ("2025Q1", "2026Q4"), {"r": 3.5})   # a number holds r every period
var.conditional_forecast(data, span, {"pi": Bounds(3, 4)})          # a range: between 3 and 4 every period
var.conditional_forecast(data, span, {"pi": {"2026Q1": Bounds(upper=2.5)}, "r": 4.0})   # one side, with an exact path
var.conditional_forecast(data, span, {"r": 4.0}, shocks=["r"])      # a structural scenario: only the policy shock moves
var.conditional_forecast(data, span, {"r": 4.0}, shock_conditions={"y": 0.0})           # ... and no demand shocks
bvar.conditional_forecast(data, span, {"r": 4.0}, update_posterior=True)                # the conditions inform the parameters
```

With exact conditions, the other variables follow their expected paths given the
conditions (Waggoner and Zha 1999): the most likely shocks that meet them. The result is
the conditional expectation, so it does not depend on how the shocks are identified. The
conditioned values have no uncertainty; the intervals of the others reflect the shocks
the conditions leave free, and, for Bayesian VARs, which impose the conditions draw by
draw, parameter uncertainty.

A condition can also be a range: `Bounds(3, 4)`, `Bounds(lower=3)`, `Bounds(upper=4)`,
`(3, 4)` or `{"lower": 3, "upper": 4}`. Ranges are met in every simulated path: given the
exact conditions, the bounded values are drawn from their normal distribution truncated
to the bounds, by Gibbs sampling one value at a time (Geweke 1991), and the rest of the
path from its distribution given them. A range wide enough never to bind gives the
unconditional forecast, and one that shrinks to a point gives the exact condition. With a
range the forecast is simulated (`draws`, default 5,000 paths for a VAR; `seed`), and its
columns are the mean, sd, median and quantile interval.

`shocks=` names the identified shocks that may move to meet the conditions (structural
scenario analysis, Antolín-Díaz, Petrella and Rubio-Ramírez 2021); the others keep their
usual distribution, so the answer now depends on the identification. When the named
shocks can meet the conditions in many ways the smallest (minimum-norm) combination is
the mean; when they cannot, the forecast is refused. `shock_conditions=` restricts the
structural shocks themselves, written as conditions (`0.0` switches a shock off,
`Bounds(lower=0)` keeps only positive ones).

A Bayesian VAR conditions each posterior draw in turn and does not update the posterior
itself. `update_posterior=True` runs Waggoner and Zha's Gibbs sampler instead: a path
given the parameters and the conditions, then the parameters given the data and that
path, under the fitted prior; the conditions then also inform the parameters.

The forecast records its conditions (`forecast.meta.extra["conditions"]`), and a forecast
round made from it keeps them in its record (`round.conditions`, `round.conditions_table()`),
saved in the store and replayed with the project. **Forecast → Conditional forecast…**
takes conditions as `r[2025Q1:2025Q4]=3.5, i=4`, ranges as `pi=3..4`, `pi>=3` or
`pi[2026Q1:2026Q4]<=4`, and, under Structural scenario, the shocks that may move and
conditions on the shocks.

### Bayesian model averaging

```python
bma = BMA("dlog(gdp)", ["dlog(gdp)[-1]", "spread[-1]", "pmi[-1]", "dlog(credit)[-1]"]).fit(data)
bma.summary()           # inclusion probabilities, averaged coefficients
bma.models_table()      # the most probable models
bma.forecast(data, ("2025Q1", "2025Q4"))   # averaged forecasts (the regressors must be known)
backtest(bma, data, ("2015Q1", "2022Q4"), horizon=1)
```

Every subset of the candidates is a model, with the intercept and any regressors given
in `always`, and every model is equally likely a priori. Zellner's g-prior gives each
model's posterior probability in closed form; g defaults to the benchmark of Fernandez,
Ley and Steel (2001), max(n, K^2). Up to 15 candidates (32,768 models).
**Estimate → Bayesian model averaging…** in the app.

### Nonlinear models and the number of factors

```python
tar = ThresholdAR("dgdp", lags=2).fit(data)     # two AR(2) regimes, split by dgdp[-d] at a threshold
tar.summary(); tar.linearity_test()             # Hansen's sup-F, bootstrap p-value
terasvirta(data, "dgdp", lags=2, delay=1)       # linearity against smooth transition
tar.forecast(data, ("2025Q1", "2025Q4"))        # simulated: multi-step forecasts follow the regimes
pca.factor_criteria()                           # Bai-Ng criteria for the number of factors
```

The threshold and the delay (which lag switches regimes) are chosen by least squares, with
at least 15% of the observations in each regime. **Estimate → Nonlinear → Threshold autoregression…**
in the app; its Tests tab has both linearity tests.

### Shrinkage and machine learning

With scikit-learn installed (`pip install pymetrix[ml]`), equations take these methods:

```python
eq = Equation("dgdp ~ dgdp[-1] + spread[-1] + pmi[-1] + dcredit[-1] + oil[-1]")
eq.fit(data, method=Lasso())              # the penalty by time-series cross-validation
eq.fit(data, method=Ridge(alpha=50))      # or given
eq.fit(data, method=ElasticNet(l1_ratio=0.5)); eq.fit(data, method=Bagging())
gb = eq.fit(data, method=GradientBoosting())     # also RandomForest, NeuralNetwork, NearestNeighbours
gb.summary(); gb.importance()                    # cross-validated error; permutation importance
gb.forecast(data, ("2025Q1", "2025Q4"), dynamic=True)
backtest(gb, data, ("2015Q1", "2022Q4"), horizon=1)
```

Cross-validation respects time: the sample is split into expanding windows, each
validated on the periods that follow it. Shrinkage methods return ordinary fitted
equations without standard errors; the learners report their cross-validated error,
which is also their forecast standard error one step ahead. Bagging needs no extra
package. The methods are in the Method list of **Estimate → Equation…**.

## Exponential smoothing

```python
ets = ExponentialSmoothing("sales", trend="add", damped=True, seasonal="mul").fit(data)
ets.summary(); ets.plot(); ets.states        # smoothing parameters, fit, level/trend/seasonal states
ets.forecast(data, ("2025Q1", "2026Q4"))     # forecast, se, lower, upper
backtest(ets, data, ("2015Q1", "2022Q4"))    # re-estimated at each origin, densities scored
```

No trend and no season is simple exponential smoothing, `trend="add"` Holt's method,
`damped=True` the damped trend, and `seasonal="add"` or `"mul"` Holt-Winters (four or
twelve seasons, from the data's frequency). Parameters and initial states are estimated
by maximum likelihood (statsmodels' ETS models); forecast standard errors are analytic
for additive models and simulated otherwise. **Estimate → Time series → Exponential smoothing…** in
the app.

## Value at risk and expected shortfall (the finance plugin)

```python
risk = value_at_risk(data, "r", level=0.99, method="garch", window=500)
risk.backtests()    # violations, Kupiec, Christoffersen, dynamic quantile, McNeil-Frey, the Basel zone
risk.plot()         # returns against -VaR and -ES, violations marked
risk.frame          # return, var, es, violation by period
```

Each period's VaR and ES are forecast from the returns before it only: from the last `window` of them
(`historical`, `normal`, `t`, `cornish_fisher`), or from a GARCH(1,1) estimated on everything before (`garch`,
and `fhs`, filtered historical simulation), re-estimated every `refit` periods. VaR and ES are positive losses:
at 99%, the return falls below `-var` one period in a hundred, and averages `-es` when it does. The backtests ask
whether the violations come at the right rate and at unpredictable times, and whether losses beyond VaR average
ES. In the app: **Finance → Risk → Value at risk…**, and the Finance workspace. The finance plugin is a package of its
own, `pymetrix-finance` (`pip install pymetrix[finance]`, or `pip install -e packages/pymetrix-finance` in a clone);
once installed it is listed in Help → Plugins and can be disabled there. Every finance section below needs it; its
names import from `pymetrix_finance`, or from `pymetrix` as before.

## Strategy backtests, bonds and dynamic Nelson-Siegel (the finance plugin)

```python
bt = backtest_strategy(ff, ["nodur", "manuf", "hlth", "money"], ["equal_weight", "min_variance", "risk_parity"],
                       window=60, rebalance=12, cost=0.002, rf="rf")
bt.performance(); bt.plot()                               # out of sample, net of costs; annual turnover

bl = black_litterman(ff.cov(["nodur", "manuf", "hlth", "money"]), market_weights, views=[[1, -1, 0, 0]], q=[0.002])
bl.summary(); bl.views_table(); bl.plot()                  # prior and posterior by asset, the views, optimal weights
mu, _ = bl                                                 # still unpacks as (posterior mean, covariance)
optimize_portfolio(ff, ["nodur", "manuf", "hlth", "money"], objective="max_sharpe", mu=mu)

yc = yield_curve(treasury, names, years)
yc.forwards([1, 5, 10]); yc.discount([5]); yc.bond(coupon=4, maturity=10)    # from the latest fitted curve
dns = dynamic_nelson_siegel(treasury, names, years)       # one step, by the Kalman filter; dns.forecast(12)
```

Each rebalance estimates from the window before it only, and the weights drift between rebalances. Curves are read
as continuously compounded zero rates, in percent unless `percent=False`. In the app: **Finance → Portfolios →
Strategy backtest…, Black-Litterman…** and **Finance → Fixed income → Dynamic Nelson-Siegel…, Bond measures…**.

## Realised volatility and HAR (the finance plugin)

```python
daily = realized_measures(Dataset.example("intraday_returns"), "r")   # one row a day: rv, bv, jump test, rvol
fit = har(daily, "rv", log=True)                                      # daily, weekly, monthly averages of RV
fit.summary(); fit.forecast(22); fit.value_at_risk(0.99)
```

The intraday returns are a dataset with intraday periods (five-minute, say); each calendar day's returns make one
row. In the app: **Finance → Risk → Realised volatility…, HAR model…**. The [finance tutorial](../packages/pymetrix-finance/docs/tutorial.md) walks
through the whole plugin on its example datasets (`french`, `treasury_yields` and simulated ones). `french`, the
Fama-French returns, is downloaded from Kenneth French's data library the first time it is loaded and kept in
`~/.pymetrix/cache/datasets/` (the library gives no licence to redistribute its files, so pymetrix does not ship
them); offline, that first load says so.

## Risk contributions and event studies (the finance plugin)

```python
rc = risk_contributions(data, ["a", "b", "c"], [0.5, 0.3, 0.2], level=0.99)   # or method="historical", cov=mgarch_fit
rc.table()                                                                    # marginal and component VaR and ES

ev = abnormal_returns(data, {"ibm": ["2015-03-02"], "ge": ["2016-05-10"]}, market="mkt", car_window=(-1, 1))
ev.tests(); ev.average(); ev.plot()                                           # four tests; AAR and CAAR; the path
```

Windows count rows (trading periods) from the event: estimation from -250 to -11 and the event window from -10 to
10 by default. In the app: **Finance → Risk → Risk contributions…** and **Finance → Portfolios → Event study of returns…**.

## Options and yield curves (the finance plugin)

```python
v = option_price(spot=42, strike=40, expiry=0.5, rate=0.1, vol=0.2)          # 4.76, with delta, gamma, vega, theta, rho
option_price(50, 50, 5 / 12, 0.1, 0.4, kind="put", american=True)             # a binomial tree
smile = implied_volatility(quotes, "price", "strike", "expiry", spot=100, rate=0.03, kind="cp")  # cp: 1 call, -1 put
smile.surface(); smile.plot()                                                 # by strike and expiry

yc = yield_curve(data, ["m3", "y1", "y2", "y5", "y10"], [0.25, 1, 2, 5, 10])  # or model="svensson", lam=0.7308
yc.factors; yc.curve("2020-06"); yc.forecast(12)                              # level, slope, curvature; a curve; ahead
```

Expiries and maturities are in years, rates continuously compounded, volatilities annual. A dataset of quotes has
one row per option (an entity column such as `quote`). The yield-curve factors are an ordinary table, so they can
also go to a VAR. In the app: **Finance → Options → Option price…, Implied volatility…** and **Finance → Fixed income → Yield curve…**.

## Returns, performance, factor models and portfolios (the finance plugin)

```python
data = returns(data, ["spx", "bonds"])                    # spx_ret, bonds_ret: log returns (kind="simple" for p/p[-1] - 1)
perf = performance(data, ["spx_ret", "bonds_ret"], rf="tbill", benchmark="spx_ret")
perf.table(); perf.plot()                                 # Sharpe, Sortino, drawdowns...; the growth of 1 and drawdowns

fm = factor_model(data, ["a", "b", "c"], ["mkt_rf", "smb", "hml"], rf="rf")
fm.summary(); fm.rolling_betas(60)                        # alphas and betas, the GRS test; betas through time

p = optimize_portfolio(data, ["a", "b", "c"], objective="max_sharpe", rf=0.0002)
p.weights; p.frontier(30); p.plot()                       # or "min_variance", or "target_return" with target_return=
optimize_portfolio(data, ["a", "b"], cov=mgarch_fit)      # with a multivariate GARCH's covariance for next period
```

Returns are fractions per period; annualising follows the data's frequency (252 trading days a year for daily
data) unless `periods_per_year` is given. Factors should be excess returns already (as the Fama-French ones are);
`rf` is taken from the assets. Portfolios are long only unless `long_only=False` or `bounds` say otherwise. In the
app: **Finance → Returns…, Performance…** and **Finance → Portfolios → Factor model…, Portfolio…**.

## Volatility surfaces (the finance plugin)

```python
surf = vol_surface(chain, "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, kind="cp", model="svi")
surf.summary(); surf.arbitrage(); surf.vol([95, 105], 0.75)   # parameters; arbitrage checks; volatility anywhere
vol_surface(chain, ..., model="sabr", beta=0.5)                # or model="heston" for one set of parameters
heston_price(100, [90, 110], 0.5, 0.03, 0.01, v0=0.04, kappa=2, theta=0.04, sigma=0.5, rho=-0.7)
```

In the app: **Finance → Options → Volatility surface…**, and **Finance → Options → Heston option price…** for one option.

## Credit risk (the finance plugin)

```python
m = merton(data, equity="mcap", debt="default_point", rate="rf", horizon=1)   # assets, DD and PD by period
merton_point(equity=30, equity_vol=0.5, debt=80, rate=0.03)                   # one firm, one date
cds = cds_curve([1, 3, 5, 7, 10], [0.008, 0.011, 0.014, 0.015, 0.016], recovery=0.4, rate=0.03)
cds.table(); cds.survival([2, 4])
```

The default point is often short-term debt plus half the long-term. Spreads are fractions a year (0.01 for 100 basis
points); the dialog takes basis points. In the app: **Finance → Credit → Merton distance to default…, Merton at a point…, CDS curve…**.

## Copulas (the finance plugin)

```python
cop = copula(data, ["a", "b", "c"], family="t", garch=True)   # t copula of GARCH-filtered returns
cop.summary(); cop.tail_dependence()                         # correlations, degrees of freedom, joint-tail chances
cop.portfolio_var([0.5, 0.3, 0.2], level=0.99)               # next period's VaR and ES by simulation
```

The families: `gaussian` and `t`; `clayton` (joint falls), `gumbel` and `joe` (joint rises), `frank` (no tail
dependence) for any number of series (for more than two they are exchangeable: one parameter for every pair); `bb1`
and `bb7` (both tails, unequal), the survival copulas `clayton180`, `gumbel180`, `joe180` (the tails swapped) and the
rotations `clayton90`, `clayton270`, ... (negative dependence) for two; `frank` also takes negative dependence for two.
Also for two: `bb6` (Joe-Gumbel, upper tail), `bb8` (Joe-Frank, no tail dependence), Tawn's asymmetric extreme-value
copulas `tawn1` and `tawn2` (one series' extremes drag the other's more than the reverse), and the rotations of the
two-parameter families, named with an underscore: `bb1_180`, `bb7_90`, `tawn1_270`, ... (VineCopula's codes 17-40,
104-134, 204-234). These are compared by `compare_copulas` only when listed.

```python
copula(data, ["a", "b"], family="bb1", margins="skewt")           # Hansen skewed-t margins, then the copula (IFM)
copula(data, ["a", "b"], family="gumbel", margins="t", method="joint")   # margins and copula together
copula(data, ["a", "b"], family="clayton", method="itau")         # the parameter from Kendall's tau
copula(data, ["a", "b"], family="t", garch=True, margins="t")     # GARCH(1,1) with t errors as each margin
cop = copula(data, ["a", "b"], family="clayton", gof=True)        # Cramer-von Mises test, bootstrap p-value
cop.aic, cop.bic, cop.se; cop.goodness_of_fit(n_boot=200, seed=1)
copula(data, ["a", "b"], family="gumbel", gof=True, bootstrap="multiplier", n_boot=1000)   # no refitting: fast
copula(data, ["a", "b"], family="tawn1")                          # theta and psi; the upper tail, asymmetric
compare_copulas(data, ["a", "b"])                                # every family on the same data: AIC, BIC, tails
compare_copulas(data, ["a", "b"], families=["t", "clayton", "bb1"], gof=True)
```

`margins` is `"empirical"` (the ranks: canonical maximum likelihood), `"normal"`, `"t"` or `"skewt"` (Hansen's
skewed t); `method` is `"ml"` (inference functions for margins when they are fitted), `"itau"` or `"joint"` (fitted
margins, no GARCH). Standard errors are from the observed information: with rank or IFM margins they treat the
uniforms as known, so they are somewhat too small; `joint`'s include the margins' estimation. The goodness-of-fit test
is on the ranks whatever the margins, for any family and number of series (the Gaussian and t CDFs in more than two
dimensions by quasi-Monte Carlo, about 1e-5). Its p-value is by parametric bootstrap (each sample refitted) or, with
`bootstrap="multiplier"`, by Kojadinovic and Yan's multiplier bootstrap, as R's `gofCopula(..., simulation = "mult")`:
no refitting, so a thousand replicates take about as long as a few parametric ones (not for the `t`, whose
correlations come from Kendall's tau). `portfolio_var` maps draws back through the fitted margins
(`margins="empirical"` or `"normal"` to override). In the app: **Finance → Copulas → Copula…** (family, margins, method,
goodness of fit), **Finance → Copulas → Compare copula families…** and **Finance → Copulas → Copula portfolio VaR…** (weights, level,
margins).

**Vine copulas** build the dependence of several series from bivariate copulas, a family for each pair:

```python
vine = vine_copula(data, ["a", "b", "c", "d"])                   # R-vine: trees and families selected
vine.summary()                                                   # tree, edge, family, parameters, tau, tails
vine_copula(data, ["a", "b", "c", "d"], structure="cvine")       # a root in each tree; "dvine": a path
vine_copula(data, ["a", "b", "c"], structure="dvine", order=["b", "a", "c"], method="joint")
vine_copula(data, ["a", "b", "c", "d"], criterion="bic", trunc_level=2, families=["gaussian", "t", "clayton", "indep"])
vine.portfolio_var([0.25] * 4); vine.simulate(10_000); vine.logpdf(u); vine.to_vinecopula()
vine_copula(data, ["a", "b", "c"], families="all")              # BB6, BB8, Tawn and rotated BB pairs as well
vine = vine_copula(data, ["a", "b", "c"], gof=True, n_boot=100)  # Rosenblatt-transform goodness of fit
vine.gof_result; vine.rosenblatt()
```

The trees are Dissmann et al.'s maximum spanning trees of |Kendall's tau|, each pair copula's family the lowest AIC
(or BIC) among the bivariate families, their rotations and independence (`indeptest=True` makes an edge independence
when Kendall's tau is not significant), estimated tree by tree by maximum likelihood and, with `method="joint"`, all
together. `trunc_level` makes the trees above it independence; `matrix=` takes a VineCopula R-vine matrix (series
numbered from 1). Margins as for `copula`. The goodness-of-fit test transforms the data by the vine's Rosenblatt
transform, which leaves independent uniforms if the vine is right, and measures their distance from independence
(VineCopula's ECP2 Cramer-von Mises statistic); its p-value is a parametric bootstrap that refits the vine's
parameters on each sample (structure and families kept), so it takes a while. In the app: **Finance → Copulas → Vine
copula…**.

**Time-varying copulas** (Patton 2006) let two series' dependence move:

```python
tv = tv_copula(data, ["a", "b"])                     # Gaussian: rho_t = L(omega + beta rho_{t-1} + alpha mean of z_u z_v)
tv.path; tv.forecast(); tv.simulate(horizon=10, n=5000)
tv_copula(data, ["a", "b"], family="t", lags=10)     # the t with constant degrees of freedom
tv_copula(data, ["a", "b"], family="sjc")            # symmetrised Joe-Clayton: both tails moving with mean |u - v|
```

`family` is `"gaussian"`, `"t"`, `"clayton"`, `"gumbel"` or `"sjc"`; the summary shows the constant copula's likelihood
beside the time-varying one's. In the app: **Finance → Copulas → Time-varying copula…**.

For **several series**, the DCC copula lets a Gaussian or t copula's correlation matrix move with Engle's DCC
recursion on the copula's scores (the normal or t quantiles of the uniforms), the margins left free:

```python
dcc = dcc_copula(data, ["a", "b", "c"], family="t")   # Q_t = (1 - a - b) Qbar + a z z' + b Q_{t-1}
dcc.summary(); dcc.path                               # a, b, nu; each pair's correlation by period
dcc.forecast(10); dcc.simulate(horizon=10, n=5000)    # correlation matrices ahead; forward paths
dcc.portfolio_var([0.4, 0.3, 0.3], level=0.99)        # next period's VaR and ES through the margins
```

With a = b = 0 it is the constant copula, whose likelihood the summary shows beside. The DCC multivariate GARCH
(`mgarch(..., model="dcc")`) runs the same recursion on GARCH residuals with normal errors. In the app: **Finance →
Copulas → DCC copula…**.

## Fat tails and asymmetry (the finance plugin)

```python
value_at_risk(data, "r", level=0.99, method="garch", dist="t", asymmetric=True)   # Student-t errors, GJR asymmetry
mgarch(data, ["r1", "r2"], model="adcc", asymmetric=True)                         # correlations that rise after joint falls
```

Both options work with `method="garch"` and filtered historical simulation (`"fhs"`). In the app they are on the
Value at risk and Multivariate GARCH dialogs.

## Multivariate GARCH (the finance plugin)

```python
fit = mgarch(data, ["r1", "r2", "r3"], model="dcc")      # or "ccc", "bekk", "diagonal_bekk"
fit.summary(); fit.plot()                                # parameters; correlations and volatilities over time
fit.correlations; fit.forecast(10)                       # each pair's correlation; covariances 1 to 10 periods ahead
fit.portfolio_var([0.5, 0.3, 0.2], level=0.99)           # next period's VaR and ES of a portfolio
```

CCC and DCC fit a GARCH(1,1) to each series, then a constant (CCC) or moving (DCC) correlation of their
standardised residuals; DCC's two parameters say how fast correlations react (a) and how long they last (b).
Scalar and diagonal BEKK model the whole covariance at once, with its long-run level set to the sample covariance.
In the app: **Finance → Risk → Multivariate GARCH…**.

## Seasonal ARIMA, the theta method and STL forecasting

```python
air = SARIMA("lair", order=(0, 1, 1), seasonal_order=(0, 1, 1)).fit(data)   # the airline model
air.summary(); air.forecast(data, ("1961-01", "1962-12"))
SARIMA("y", (1, 0, 0), exog=("x",)).fit(data)                             # regressors: their future values come from data
Theta("sales").fit(data).forecast(horizon=8)                               # smoothing plus half the trend's slope
STLForecast("sales", model="arima", order=(1, 1, 0)).fit(data)            # STL, then a model of the adjusted series
```

`SARIMA` differences the series d times and seasonally D times, at the season length (4 or 12 from the
frequency, or `period=`), and estimates the AR and MA terms by exact maximum likelihood; standard errors come
from the numerical Hessian, as in R's `arima`. With differencing there is no constant unless `trend=` asks for
one. `Theta` and `STLForecast` are simple, strong benchmarks. All three backtest like any other model and are in
**Estimate → Time series** in the app.

## Data from FRED, the World Bank and SDMX services

Task page: [Load data with Browse series (FRED, IMF, ECB, Eurostat, World Bank)](../src/pymetrix/howto/load-data-with-browse-series.md).


```python
Dataset.read("fred://GDPC1,GDP")                                   # ?vintage=2020-01-30 for ALFRED
Dataset.read("fred://GDPC1?start=1990Q1&end=2019Q4")               # the periods between
Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON;FJI&date=2000:2023")   # a panel by country
Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2015-01")
Dataset.read("sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M?entity=COUNTRY")      # a panel by a dimension
register_sdmx_provider("MYCB", "https://data.mycb.org/sdmx/data/{flow}/{key}")  # your own warehouse
```

Series are named by the dimensions that differ (`usd`, `gbp`), labelled with the
provider's titles, and record their source. When the provider says when the data were
released or updated (the World Bank, Eurostat, the IMF), that date is the series'
vintage. The connectors use only the standard library and the providers' public web
services (`pymetrix.ext.connectors`, a bundled reference plugin).

**File → Load from connector…** in the app: choose the provider (the connectors installed,
plugins' included: `pymetrix.io.connectors()`), give the series codes, and optionally the
first and last periods and, for FRED, a vintage date; the dialog writes the address, the
same text **File → Load data…** takes as its path. A plugin's reader describes its addresses for the
dialog with `register_reader("myscheme", reader, Connector(label, codes, start, end, vintage))`.

### Finding series

```python
from pymetrix import connectors

connectors.search("worldbank", "gdp per capita")             # the World Development Indicators by default
connectors.search("sdmx", "exchange rates", source="ECB")     # a provider's dataflows
connectors.search("fred", "unemployment rate")               # needs FRED_API_KEY (or connectors.set_api_key)
connectors.browse("worldbank", "topics")                     # sources, topics, countries; FRED categories by ID
exr = connectors.structure("sdmx", "ECB/EXR")                 # the dimensions and their codes
exr.address(FREQ="M", CURRENCY=["USD", "GBP"], CURRENCY_DENOM="EUR", EXR_TYPE="SP00", EXR_SUFFIX="A")
# 'sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A'
found = connectors.search("worldbank", "unemployment")
connectors.to_frame(found)                                    # code, title, frequency, units, dates, address
Dataset.read(found[0].address)
```

Every result is a `SeriesInfo`: code, title, frequency, units, first and last period, last
update and notes, where the provider gives them. **File → Browse series…** does the same in
the app: choose the provider, search or browse, select one or more series (for SDMX, a
dataflow, then codes for each dimension, with the key previewed), and **Load** them at once
or **Add to Load from connector…** to add periods, a vintage or options. Either way the step
recorded is `Dataset.read(address)`, so the script replays without the browser. A saved project keeps the data as
it was read and opens with it, whatever the provider has revised since; **File → Refresh data…**
(`session.refresh("fx")`) reads it again, runs again the steps that changed it and recomputes what uses it. An SDMX
provider's dataflows are in the provider's own folders where it publishes them (its SDMX
category schemes, such as the ECB's "Monetary and financial statistics › Balance sheet items";
dataflows in none are under "Other"). Where it publishes none, or they put nearly everything in
one folder, the dataflows are grouped by family, such as the IMF's "Monetary and Financial
Statistics (MFS)", with dated releases newest first, else by the first part of their ID
(Eurostat's `nama_`, `prc_`), else listed by name; the tree's header says which ("Browse · by
category"). A row's tooltip gives its full name and dataflow ID, and the search box's words
filter the tree as you type, opening the folders down to each match. A long World Bank topic
or source has a folder for each indicator-code prefix (`NY.`, `SL.`, `SP.`…). Click a column's
heading to sort the results by it (again for the reverse, a third time for the provider's own
order: FRED's relevance for a search).
Once a dataflow is chosen, the search box searches its codes' names and descriptions instead
(the "In WEO's codes" box beside Search shows this; untick it to search dataflows again), and
a code found is chosen by double-clicking it or pressing Enter. A dimension whose codes cannot
be loaded says why in its box and takes codes typed with `+` between them.

FRED's search uses its web service, which needs a free API key: set `FRED_API_KEY`, or type it
in the browser (kept for the session only, never saved). Reading FRED series by ID needs
none. Searches are kept for a day and lists of categories, indicators, dataflows and
structures for a week, in `~/.pymetrix/cache/connectors` (under `$PYMETRIX_HOME` if set);
`refresh=True` (the browser's "Ask again") asks the provider afresh, `connectors.clear_cache()`
empties it and `connectors.configure_cache(search_ttl=..., metadata_ttl=..., enabled=...)`
changes it. Requests are kept by their address with any key taken out. An institution's SDMX
service becomes searchable with `register_sdmx_provider(name, url, structure_url=..., agency=...)`.

## EViews workfiles (the EViews plugin)

pymetrix does not read EViews's binary workfiles (`.wf1`). The supported way in is to save the workfile from EViews as
CSV or Excel and load that with **File → Load data…**. The `pymetrix-eviews` plugin (`pip install -e
packages/pymetrix-eviews` in a clone) adds a second, **experimental** way: its export program, `pymetrix_export.prg`
(**File → Import EViews workfile…**, *Get the export program*), run in EViews on the open workfile, writes a *bundle*, which the
same dialog's *Workfile bundle* reads, a dataset per page, with the series' labels, units and sources:

```python
quarterly = Dataset.read("C:/work/macro-export/macro.evbundle", page="quarterly")

from pymetrix_eviews import compare_estimates, eviews_objects
eviews_objects(quarterly)      # the page's groups, samples, scalars, strings, matrices and models
compare_estimates(quarterly)   # least-squares equations refitted, beside EViews's estimates
```

The export program has never been run in EViews (the project has no licence); every point to check is marked
`(verify)` in it. Replaying a bundle's read needs the plugin; **Save the data as Parquet** in the dialog reads the
pages from Parquet copies instead, so the script needs only pymetrix. The plugin's README lists what each EViews
object becomes and what is refused.

### EViews programs

Task page: [Translate an EViews program](../src/pymetrix/howto/translate-an-eviews-program.md).


The plugin also translates EViews programs (`.prg`). This first phase covers data work: workfiles and pages,
samples, series and formulas, groups, labels, scalars, `!` and `%` variables, and matrices. A program's `wfopen`
needs the workfile's data: the bundle, or a CSV or Excel file saved from it in EViews.

```python
from pymetrix_eviews import translate_program

t = translate_program("C:/work/macro.prg", bundle="C:/work/macro-export/macro.evbundle")
print(t.script)            # Python, each EViews line as a comment above what it became
t.report.summary()         # one row per line: translated, translated with a note, left out or error, and why
out = t.run(session)       # the same work as steps in a session, each recorded as a dialog's would be
```

For a CSV file, give `Dataset.read`'s options: `translate_program(prg, bundle="macro.csv", read_options={"time":
"obs", "freq": "Q"})`. `smpl` becomes the dataset's active sample, and an assignment under it uses
`with_series(..., within=Sample(...))`, so only the sample's rows change, period by period when the formula uses its
own series, as in EViews. Scalars are Python variables: the script writes them into expressions as f-strings, and
the steps write their values.

Estimation, tests and models translate too (phase 2). An equation becomes a fitted `Equation` on the page's sample,
with the method its options map to, and keeps EViews's coefficient numbering, so later lines that read it work:

```
equation eq.ls(cov=white) cons c inc cons(-1)     eq = Equation("cons ~ inc + cons[-1]", name="eq").fit(q, method=OLS(cov="white"))
eq.wald c(2)=1                                    eq_wald = eq.test("wald", "inc=1")
scalar mpc = eq.@coefs(2)                         mpc = float(eq.params["inc"])
eq.makeresids r                                   q = q.with_series(r=eq.resid)
model m                                           (collected)
m.merge eq                                        (collected)
m.append @identity y = cons + inv                 (collected)
m.solve                                           m = read_eviews('model m ...', data=q); m_model = m.model
                                                  m_0 = m_model.solve(q, ("2020Q1", "2024Q4"))
                                                  q = q.with_series(cons_0=m_0['cons']) ...
```

A model's lines are collected and read by `read_eviews` before the model is first used; what the reader refuses is
reported at its line. Scenarios, overrides (from the alias series, `x_1`), excludes, add factors (`addassign`,
`addinit`) and the solver (`solveopt`) map to `Scenario`, `Model.add_factors` and `Newton` or `GaussSeidel`, and the
solution's series come back into the page as EViews names them (`cons_0`, `cons_1`). Where EViews's defaults differ
from pymetrix's (the HAC bandwidth, the ARMA estimator, information criteria per observation, the default solver),
the line is translated with a note saying what to check; an option that is not mapped leaves the line out, naming
it. Graphs, tables and the few EViews functions pymetrix has no equivalent of (`@date`, `@trend` and `@seas` in data
work) are left out with the reason ([known gaps](known-gaps.md)). Rounding, missing values, `@iif`, the calendar,
`@cumsum` and the by-group statistics translate to pymetrix's functions; random numbers become seeded draws, from
`rndseed`, marked to check, since they are never EViews's numbers; EViews's dates (`@dateval`, `@dateadd`,
`@datestr`) are worked out as the program's values. Nothing is dropped silently: every line with text has a row.

Control flow translates too (phase 3). Loops stay loops, an `if` block an `if`, and a subroutine a function, defined
at the top of the script:

```
for %c us uk                                   for c in ["us", "uk"]:
  equation eq_{%c}.ls y_{%c} c x_{%c}              globals()[f"eq_{c}"] = Equation(f"y_{c} ~ x_{c}", name=f"eq_{c}").fit(q)
  if eq_{%c}.@r2 > 0.5 then                         if globals()[f"eq_{c}"].stats["r2"] > 0.5:
    %good = %good + " " + %c                            good = good + " " + c
  endif
next
call suffixed(%name, "_aerr")                  name = suffixed(name, "_aerr")
subroutine suffixed(string %s, string %tail)   def suffixed(s, tail):
  %s = %s + %tail                                  s = s + tail
endsub                                             return s
```

Steps mode runs the loops, conditions and calls in Python, so the session records only flat steps, each one undoable,
and a condition is decided by the results made before it. EViews passes arguments by reference: a parameter the
subroutine assigns goes back to the caller's variable. A subroutine that is not `local` shares the program's names;
a `local` one shares none. `include lib` translates `lib.prg` in line, its rows in the report under its name. The
program's arguments are the script's parameters: `translate_program(prg, args="us 2020q1")` puts `arg0 = "us"` and
`arg1 = "2020q1"` at its top, and `t.run(session, args=...)` runs it with others. What the program uses but does not
make (a model an add-in reads, the series a database fills) is given with `objects={"stdver": model}`: the script
expects it under that name, and steps mode puts it in the session. A `while` loop in steps mode stops after 10,000
passes (`run(while_cap=...)`).

In the app (with `pymetrix-eviews`), **File → Translate EViews program…** asks for the program, the workfile data its `wfopen` opens and its
arguments. The translation opens in the **EViews translation** panel (View → EViews translation): a row for each
line, its status in words and in colour, and the reason; the program and the script side by side, both following
the row you select; a filter for each status, with its count; **Copy** and **Copy script**; and **Run as steps**,
which makes the program's work steps of the session, one at a time, each in the History and undoable. It stops at
the first step that fails and marks that row.

## Reports

Task page: [Make a report](../src/pymetrix/howto/make-a-report.md).


```python
rep = Report("Quarterly forecast, 2026Q3", author="Forecasting team")
rep.heading("Consumption").text("The consumption function is stable.")
rep.add(fit)                                  # its summary and graph; sections="all" for every tab
rep.add(rnd.plot(data), caption="GDP: fan chart")
rep.save("forecast.docx")                     # or .xlsx, .html, .md
```

An appendix lists where each number came from: data vintages, sample, method and
pymetrix version. Word needs python-docx (`pip install pymetrix[report]`). Markdown writes its
figures as PNG files in a folder beside it.
**File → Export report…** in the app.

### Templates, themes and chart options

```python
rep = Report.from_template("estimation_summary", {"Consumption": fit, "Investment": inv},
                           title="Equations", theme="publication")
rep = Report.from_template("forecast_round", {"Baseline": rnd}, theme="policy", span=("2020Q1", None))
report_templates()                            # estimation_summary, forecast_round, technical_appendix, policy_brief

set_theme("policy")                           # the default for this session; themes() lists them
fig = rnd.plot(data, coverages=(0.5, 0.7, 0.9)).with_options(
    theme="monochrome", span=("2021Q1", None), subtitle="Percent", footnote="Source: staff projections.")
```

A template is a list of views to include for each result, in order: `estimation_summary`
includes the summary, graph and diagnostics; `forecast_round` the fan charts, paths, judgement
and accuracy; `technical_appendix` everything, with provenance; `policy_brief` a scenario set: which scenario is
the baseline, how far each other scenario departs from it (the Impact table), the paths and the departures.

Themes (`default`, `publication`, `teaching`, `policy`, `monochrome`) change colours, line
styles and widths, font sizes, grids, band shading and the decimals in report tables. They do not
change the numbers. Every theme starts from the same house look (`pymetrix.figure_style`): a light grey grid
of major lines behind the data, no top or right spines, short outward ticks, DejaVu Sans (the app's own font
in the app), and the default theme's Okabe-Ito colours, which stay distinct with colour-vision deficiencies.
It is applied to pymetrix's figures only, never to matplotlib's global settings, so your own plots are as
you set them. In the app, a dark palette gives dark charts; what is copied, saved or put in a report is light. Chart options (`span`, `subtitle`, `footnote`, `theme`) work the same way:
`fig.data()` gives the numbers drawn.

Reports are built from results' views, and views from named fields, so a report never reads
printed text. The Export report dialog has a template, a theme and the dates for figures. Like
every dialog, it records code that writes the same report again.

## Temporal disaggregation

Task page: [Reconcile national accounts components to a total](../src/pymetrix/howto/reconcile-components-to-a-total.md).


```python
fit = TemporalDisaggregation("gdp ~ ip", method="chow-lin").fit(annual, quarterly)
quarterly = quarterly.with_series(gdp=fit.series)   # adds up to the annual data; extends with ip
TemporalDisaggregation("gdp ~ 1", method="denton").fit(annual, quarterly)   # smooth, no indicator
```

`conversion="average"` for rates and indices, `"first"` or `"last"` for stocks.
**Data → Transform → Temporal disaggregation…** in the app.

Methods: `chow-lin` (rho by maximum likelihood) and `chow-lin-minrss`, `fernandez`, `litterman` and
`litterman-minrss` (a random walk with AR(1) increments), `denton` and `denton-additive`. Rho is searched in
[0, 0.999], as in R's tempdisagg; `rho_bounds=(-0.999, 0.999)` allows negative values, and the summary notes an
estimate at an end of the range. `rho=0.5` fixes it.

```python
C = aggregation_matrix(quarterly, [pd.Period("2018", "Y"), pd.Period("2019", "Y-JUN"), pd.Period("2020", "Y-JUN")])
fit = TemporalDisaggregation("gdp ~ ip").fit(low, quarterly, aggregation=C)   # calendar and fiscal years together
TemporalDisaggregation("gdp ~ ip", soft=50).fit(annual, quarterly)            # may miss each total (sd 50)
TemporalDisaggregation("gdp ~ ip", soft={"2023": 50}).fit(annual, quarterly)  # only 2023's, the rest exact
TemporalDisaggregation("gdp ~ ip", method="denton", differences="seasonal").fit(annual, quarterly)
```

An aggregation matrix has a row per low-frequency value (named as the low-frequency data's rows: here a
cross-section dataset whose rows are named `2018`, `2019 Y-JUN`...) and a column per high-frequency period; C·y = Y
then holds exactly, and rows with no value are dropped. `soft=` is a standard deviation in the data's units: the
regression methods treat it as measurement error on the totals, Denton as a penalty. Seasonal differences keep the
indicator's seasonal pattern; `level_penalty=` (0.001) is the light weight on first differences that pins it.

Components disaggregated one by one need not add up to their total. `reconcile` moves them as little as possible
(Di Fonzo and Marini's multivariate Denton, or least-squares balancing) so that they do, every quarter, while each
keeps its annual totals; fixed components stay as they are:

```python
rec = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", low=annual, fixed=["gov"])
rec.diagnostics()        # each component's adjustments, and the gaps before and after
quarterly = quarterly.with_series(cons=rec.series["cons"], inv=rec.series["inv"])
```

`method="least-squares"` moves each value in proportion to its size. `method="weighted"` instead gives each
component's adjustments a constant variance, its weight, whatever its level and sign: a component that enters with a
negative sign, such as imports, moves as one with a positive sign would, and `weights=` says what share of each gap
a component takes (equal weights give the smallest adjustments in the ordinary sense, the minimum-norm one):

```python
rec = reconcile(quarterly, ["cons", "inv", "gov", "imp"], "gdp", low=annual, fixed=["gov"],
                method="weighted", weights={"cons": 1, "inv": 4, "imp": 2})
```

Weights are positive; a component that should not move goes in `fixed`.

**Data → Transform → Reconcile components…** in the app.

## MIDAS and nowcasts

```python
m = MIDAS("gdp ~ ip", lags=12, weights="exp_almon").fit(quarterly, monthly)
m.weights["ip"]; m.nowcast(quarterly, monthly)        # includes quarters past the GDP data
MIDAS("gdp ~ ip", lags=12, high_lag=2)                 # as if only each quarter's first month were known
```

**Estimate → Mixed frequency → MIDAS…** in the app.

### Nowcasting tools

```python
from pymetrix.nowcast import combine, fill_edge, realign

monthly.ragged_edge()                    # each series' last observation, release lag and publication day
known = monthly.as_of("2025-02-15")      # what was known that day (from each series' release_lag)
fill_edge(known, lags=2)                 # missing months forecast by AR(2), as bridge equations do
realign(known)                           # each series moved so its last value is in the latest month

bridge = Bridge("gy ~ gr + pmi").fit(quarterly, monthly)          # indicators averaged within quarters
bridge.nowcast(quarterly, known)                                  # 2025Q1, months completed by AR forecasts

dfm = DynamicFactor(monthly=["gr", "gip", "gemp", "pmi"], quarterly=["gy"]).fit(quarterly, known)
dfm.nowcast("gy"); dfm.factors; dfm.loadings
later = dfm.update(quarterly, monthly.as_of("2025-04-15"))        # same model, newer data
later.news(dfm, "gy").by_series                                   # how each series' new figures moved it

combine({"bridge": b, "dfm": d}, weights={"bridge": 1, "dfm": 2})  # weighted average of nowcasts
PCA(["gr", "gip", "gemp", "pmi"], n=2).fit(monthly).scores         # principal components
FactorModel(["gr", "gip", "gemp", "pmi"], 1).fit(monthly).loadings  # maximum-likelihood factor analysis
```

`FactorModel(..., rotation="varimax" | "promax")` rotates the loadings; `fit.scores("bartlett")` gives Bartlett's
scores, `fit.test` whether that many factors are enough, and `fit.factor_correlation` the rotated factors'
correlations (promax). **Estimate → Factors → Principal components…** and **Factor analysis…** in the app; their
result windows' Next ▸ adds the components or scores to a dataset (**Data → Add results to dataset…**).

Each series carries its publication lag in days (`Variable.release_lag`), so
`as_of` rebuilds the ragged edge of any day; it removes unpublished values but does
not restore earlier vintages. The three usual treatments of the edge are all
available: leave the gaps for the factor model (its Kalman filter handles them),
fill them by autoregressions (bridge equations), or realign the series (MIDAS works
on realigned data). The dynamic factor model is statsmodels' `DynamicFactorMQ`
(Bańbura and Modugno; quarterly series as in Mariano and Murasawa); series must be
stationary, so transform them first. Its news adds up exactly to the change in the
nowcast between two data sets. The tools stop there: tracking a nowcast through a
quarter or scoring nowcasts by horizon is left to the user (`backtest` helps).
**Estimate → Mixed frequency → Bridge equation…** and **Estimate → Mixed frequency → Dynamic factor model (nowcasting)…** in the app,
and in the same menu the data tools: **Ragged edge…**, **Data as known on a day…**, **Fill the ragged edge…**
(`known.fill_edge(lags=2)`), **Realign the ragged edge…** (`known.realign()`) and **Combine nowcasts…**
(`nowcast.combine(...)` of bridge and dynamic factor models' nowcasts).

## Seasonal adjustment

Task page: [Seasonally adjust a series](../src/pymetrix/howto/seasonally-adjust-a-series.md).


```python
data = data.seasonal_adjust("sales")                  # STL; adds sales_sa and sales_seasonal
data = data.seasonal_adjust(["c", "i"], X13(trading=True))
data.seasonal_adjust("c", X13(mode="add", seasonal_filter="s3x9", trend_filter=13, easter=8))
data.seasonal_adjust("c", X13(adjustment="seats", arima="(0 1 1)(0 1 1)", outlier_types="ao ls tc"))
data.seasonal_adjust("c", TramoSeats())                # TRAMO-SEATS, JDemetra+'s RSAfull
data.seasonal_adjust("c", TramoSeats(spec="rsa3", trading="working", easter=True, critical=3.5))
data.seasonal_adjust("c", TramoSeats(interventions="ls 2020Q2, ao 2008Q4",      # outliers known in advance
                                     regressors="holidays, strike=irregular"))   # series of the dataset as regressors
data = data.calendar(trading_days="one", easter=8)    # td_weekdays, easter_8, leap_year
fit = Equation("dlog(c) ~ dlog(y) + seasonal()").fit(data)            # seasonal[Q2], seasonal[Q3], seasonal[Q4]
fit.test("wald", "seasonal[Q2] = 0, seasonal[Q3] = 0, seasonal[Q4] = 0")  # deterministic seasonality
Equation("dlog(c) ~ dlog(y) + C(quarter) - 1").fit(data)              # every quarter, no constant
```

In a formula, `seasonal()` adds seasonal dummies: one per quarter of quarterly data or month of monthly data, the
first left out when the equation has a constant, named `seasonal[Q2]` ... or `seasonal[M2]` ...; with `- 1` every
season has one. `C(quarter)` and `C(month)` give the same dummies named `C(quarter)[2]` ..., the quarter or month
taken from the time axis when the data have no series of that name (a fiscal axis such as `Q-JUN` gives fiscal
quarters). Each row's dummy is its own period's, whatever the sample's start or gaps. Annual data have none, and
saying so is an error.

Positive series are adjusted multiplicatively (the seasonal factor is around 1)
unless `log=False`. Adjusted series are derived series, so they are recomputed when
the data are revised. X-13ARIMA-SEATS and TRAMO-SEATS come with the seasonal plugin,
`pymetrix-seasonal` (`pip install "pymetrix-seasonal[jdemetra]"`, or `-e packages/pymetrix-seasonal` from the
repository), because they need programs outside Python; installed, they appear in the same dialog and
`from pymetrix import X13` works as before. X-13ARIMA-SEATS needs the Census Bureau's `x13as` program on
the PATH or in `X13PATH`. Series it adjusts record its version, the ARIMA model it
chose and the outliers it found (`data.variables["sales_sa"].extra`, and the dataset's
Variables tab). TRAMO-SEATS runs Eurostat's JDemetra+ 3 in Java: install JPype
(the plugin's `[jdemetra]` extra) and Java 21, then fetch JDemetra+'s
libraries once with `pymetrix_seasonal.jdemetra.install()` (or point `JDPLUS_PATH` at a folder that
has them). `spec` picks one of JDemetra+'s specifications, RSA0 to RSAfull, and the other
settings change it; series record the model, outliers and calendar effects TRAMO found.
`interventions` are outliers known in advance (`ao`, `ls`, `tc` or `so` with a period), kept without
testing; `regressors` are series of the same dataset (holiday counts, strike dummies, policy changes),
each with the component its effect goes to: `calendar` (the default; removed from the adjusted
series, as trading days are), `irregular`, `trend` or `seasonal`. A regressor needs values over the
whole sample of the series adjusted; the effects are recorded as `tramo_regression`. **Data → Transform → Seasonal adjustment…** and **Data → Transform →
Calendar regressors…** in the app.

## State space

Models are written as equations with declared parameters:

```python
model = StateSpace(
    states=["potential", "growth", "gap"],
    transition=["potential = potential[-1] + growth[-1] + e_pot",
                "growth = growth[-1] + e_growth",
                "gap = rho1*gap[-1] + rho2*gap[-2] + e_gap"],
    measurement=["ly = potential + gap"],
    shocks={"e_pot": "sd_pot", "e_growth": "sd_growth", "e_gap": "sd_gap"},
    params=["rho1", "rho2", "sd_pot", "sd_growth", "sd_gap"],
    stationary=["gap"],
)
fit = model.fit(data)                  # maximum likelihood; posterior mode if parameters have priors
fit.smoothed["gap"], fit.filtered["gap"]   # two-sided and real-time estimates (with _se columns)
fit.forecast(data, ("2025Q1", "2026Q4")), fit.irf(20), fit.simulate(100, seed=1)
UnobservedComponents("ly", trend="local linear trend", cycle=2).fit(data)
```

Lags beyond one are handled automatically; trends start diffuse (exactly, as in
Durbin and Koopman), stationary states from their distribution. `fixed={...}`
calibrates parameters; `Param(..., bounds=, prior=Beta(0.7, 0.1))` sets bounds
and priors. The statsmodels engine is used when installed (fast, exact); a
native filter is the fallback, and the two are cross-checked in the tests. The
HP filter is the smoothed trend of a smooth-trend model, and the tests check
that too. In the app: **Estimate → Time series → Unobserved components…** and **Estimate → Time series →
State space model…**.

**The diffuse log likelihood, and which constants it counts.** With diffuse states, the log likelihood is the exact
diffuse likelihood of Durbin and Koopman (2012, section 7.2.2). With `d` diffuse periods (until every diffuse state
is pinned down; `d` is `fit.stats["diffuse periods"]`), `p_t` series observed in period `t`, innovations `v_t` with
variances `F_t` afterwards, and the diffuse part `F_inf` of a diffuse period's innovation variance (observations
taken one at a time):

```text
log L = -1/2 sum_{t<=d} [ p_t log 2pi + sum_i log F_inf,t,i ]                  the diffuse periods
        -1/2 sum_{t>d}  [ p_t log 2pi + log|F_t| + v_t' F_t^-1 v_t ]           the rest
```

(an observation of a diffuse period whose `F_inf` is zero counts `log F_* + v^2/F_*` instead). Every observation
counts `log 2pi`, the diffuse ones included; nothing is concentrated out. statsmodels' exact diffuse filter gives the
same number. statsmodels' `UnobservedComponents` and `SARIMAX` by default report another convention (Harvey 1989,
section 3.4.3): the likelihood of the observations after the first `d`, given them (their approximately diffuse start
leaves the first `d` periods out, `loglikelihood_burn`). Theirs is higher by the diffuse periods' terms, for one
series `d/2 log 2pi + 1/2 sum_t log F_inf,t = d/2 log 2pi + 1/2 log det(X X')`, with `X` the rows `Z T^(t-1)` on
the diffuse states: 0.919 for a local level, 1.838 for a local linear trend, 7.367 for a local linear trend with a
quarterly seasonal. `diffuse_terms="exclude"` reports theirs:

```python
fit = uc.fit(data, diffuse_terms="exclude")      # also StateSpace.fit/.loglike/.filter, Model.estimate
fit.criteria()                                   # loglike, aic, bic, hq as fitted; fit.criteria("include") the other
```

Where the diffuse part of the system (`Z` and `T` on the diffuse states) has no parameters, as in every
unobserved-components model, the difference is the same at every parameter value: the estimates and standard errors
do not change, only the level of the log likelihood and of AIC, BIC and HQ (by twice the constant). Differences of
criteria between models with the same diffuse states are the same in both conventions; between models with
different diffuse states (a local level against a local linear trend), or between numbers from packages with
different conventions, they are not comparable. In the dialogs: **Diffuse periods in the likelihood** under
Advanced.

## Figures

```python
data.plot(["c", "y"])                          # lines; kind="step", "bar", "barh", "area", "scatter" (x first), "histogram", "box"
data.plot(["c", "i", "g"], layout="each", span=("2010Q1", None))
data.plot(["gc", "gi", "gg"], kind="bar", stacked=True)        # contributions: negatives stacked below zero
data.plot(["y", "u"], right="u")                               # u against a second scale on the right
data.plot(["c", "i"], transform="yoy")         # or "index" (first period = 100), "change", "growth", "log"
data.plot("gdp_growth", shade="recession")     # shade the runs of a 0/1 series, or periods: "2008Q3:2009Q2=GFC, 2020Q1:2020Q2"
fit.plot().with_options(shade="2008Q3:2009Q2") # any figure: every dated panel is shaded
data.plot("gdp_growth", shade="nber")          # NBER recessions (cepr: the euro area's), at the chart's frequency
data.plot(["c", "i", "g"], line_labels=True)   # each line named at its end, instead of a legend
recessions("nber", "Q")                        # the spans; recession_series(data, "nber") a 0/1 series
```

**Styling in matplotlib's terms.** A figure takes matplotlib's own calls and keyword arguments, recorded on the
figure so that they replay ([docs/notes/figure-styling.md](notes/figure-styling.md)):

```python
fig = (data.plot(["c", "i"])
       .line("c", color="C3", linestyle="--", linewidth=2)      # as Line2D.set; aliases (lw, ls, c) work
       .bars("x", hatch="//")                                   # as Rectangle.set, for a bar series
       .set(yscale="log", ylim=(10, 1000), xlim=("2010Q1", "2020Q4"), ylabel="levels")   # as Axes.set
       .ticks(axis="y", format="{x:,.0f}", labelsize=7)         # tick_params, and a StrMethodFormatter format
       .grid(axis="y", linestyle=":")                           # Axes.grid
       .spines(which="top", visible=False)                      # the frame's sides
       .legend(loc="upper left", ncols=2, frameon=False)        # Axes.legend
       .layout(figsize=(8, 4), suptitle="Spending"))            # the whole figure
```

- **Which panel:** `ax=` picks a panel: its position, its title, or `"all"`. `side="right"` addresses a panel's
  right-hand axis.
- **Which series:** found by name, wherever it is drawn.
- **Checking:** each value is checked by matplotlib when it is set, so `linestyle="dottted"` fails at once, with
  matplotlib's own suggestion.
- **The data:** styling never changes what is plotted (`fig.data()` is the same).

**In the app:** a figure's **Edit… → Style** opens **Figure style**, with tabs for a series, the axes, the legend
and the size. Each change is recorded as the chain of calls above.

**Every property matplotlib has.** **Edit… → All properties…** opens **Figure properties**: pick a series or a
panel, and the form lists every style property matplotlib can set on it.
- **Where the list comes from:** matplotlib's own description of its artists (`ArtistInspector`): 22 for a line, 16
  for bars, 13 for a band and 23 for an axes.
- **Help:** each property is listed with matplotlib's note on its valid values.
- **Left out:** properties holding the plotted data or positions.
- **Recording:** the step is recorded as one call, such as
  `figure_of(fit).line('actual', drawstyle='steps-post', markevery=4)`.

In code, `pymetrix.figure_properties.properties("line")` lists the same properties, and `property_table("axes")`
shows them as a table.

**Notes, lines and bands.** Things added to a figure take `Axes.annotate`'s, `text`'s, `axhline`'s, `axvline`'s,
`axhspan`'s and `axvspan`'s arguments, with dates as the data's periods:

```python
fig = (data.plot("gdp_growth")
       .annotate("Lehman", xy=("2008Q3", -2.1), xytext=(15, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"))
       .text("2010Q1", 6, "recovery", style="italic")
       .axhline(0, linestyle=":").axvline("2020Q1", color="C3")
       .axvspan("2012Q1", "2013Q4", alpha=0.15))            # through the end of 2013Q4
```

**In the app:** click **Add a note** under a figure, then click on the figure where the note goes. The **Add a note**
dialog opens with the panel, the date (as a period) and the value filled in.
- **What it adds:** a note with an arrow, words, a line at a date or a value, or a band between dates or values.
- **Without clicking:** **Edit… → Add a note…** opens the same dialog.
- **Recording:** the step is recorded as the matplotlib call, in the data's terms, so it replays at any size.

**Any other Axes method.** `fig.call` records any `Axes` method that styles or annotates a panel, with matplotlib's
own arguments:

```python
fig = (data.plot(["gdp", "investment"], right="investment")
       .call("minorticks_on")
       .call("ticklabel_format", axis="y", style="sci", scilimits=(0, 0))
       .call("set_ylabel", "investment", side="right"))   # on the right-hand axis
```

`pymetrix.figure_properties.axes_methods()` lists the methods. Those that plot more data (`plot`, `bar`), clear
things or need Python objects are not among them. The arguments are checked against the method's signature when
the call is added. In the app: **Edit… → matplotlib method…**, with the arguments typed as in Python.

**Style sheets.** A matplotlib style sheet becomes a theme:

```python
from pymetrix import Theme

Theme.from_mplstyle("house.mplstyle")       # registered as "house"
data.plot(["c", "i"], theme="house")
```

The sheet's colour cycle becomes the palette; its font, sizes, line width and grid set the theme's; every other
setting in it (background, spines, ticks...) applies while drawing, as matplotlib would apply the sheet.

**The chart in matplotlib alone.** `fig.matplotlib_code()` writes a standalone script, using only NumPy, pandas and
matplotlib, that draws the same chart from the same numbers:
- the data as arrays, the dates as pandas periods;
- each layer as `plot`, `bar`, `fill_between`, `scatter`, `hist` or `boxplot`;
- each option (shading, events, reference lines, scales, values on bars, line names) and each style call as the
  matplotlib call it is.

Use it to check a chart, or to reuse it without pymetrix. In the app, **Copy plotting code** under a figure copies
the script. `fig.to_matplotlib()` returns the drawn `matplotlib.figure.Figure`, to finish by hand; changes made to it
are not recorded.

**The recorded styles.** `fig.style_steps()` lists a figure's style calls in the order they apply, each written as
the call that records it:

```python
fig.style_steps()
# ["line('gc', color='C3', linestyle='--')", "set(ax=0, ylabel='% a year')", "annotate(text='dip', xy=('2020Q2', -10), ax=0)"]
fig.without_style(2)                 # remove the note
fig.reorder_style([1, 0])            # the label first, then the line; the note removed
fig.restyle(0, color="C2")           # change a keyword argument; None removes one; replace_all=True replaces them all
```

Calls apply in order, so where two set the same thing the later one shows. A change is checked by matplotlib as when
the call was first made.

**In the app:** **Edit… → Recorded styles…** lists the calls.
- **Up and Down** move a call; **Remove** drops it. OK records the new order as one step.
- **Change…** opens **Change a style** on the selected call, with its keyword arguments filled in.

**Copying a style.** `fig.with_style_of(house_chart)` adds another figure's recorded style calls after this figure's
own, where they apply:
- a series' call (`line`, `bars`, `band`), if this figure draws that series that way;
- a panel's call, if this figure has that panel (by position or title) and, for a right-hand axis, if the panel has
  one; `layout` always;
- notes, text, lines and bands only with `notes=True`, since they mark the other chart's data.

`fig.style_skipped(house_chart)` lists the calls left out. In the app: **Edit… → Apply style from…**.

```python
data.plot("infl", events="2016Q2=Brexit vote", reference="2=target")  # labelled lines at dates and at values
data.plot(["c", "i"], kind="bar", span=("2024Q1", None), bar_labels=True)  # each bar's value written on it
data.plot(["c", "i", "g"], layout="each", sharey=True, log=True, ylim=(10, None))  # one log scale, from 10
data.plot("growth", ybreak="6:12")            # a split value axis: 6 to 12 left out, so a spike squashes nothing
data.plot("u", xbreak="1986Q1:2004Q4")        # a split time axis: those periods left out, two spans side by side
fit.plot()                                     # actual, fitted and residuals (±2 s.e.)
fc.plot()                                      # forecasts with intervals
var.irf(20).plot(), lp.plot(), ssm.plot()      # responses with bands; estimated states
fig = data.plot("i", kind="bar"); fig.save("investment.png")   # or .svg, .pdf
```

**Split axes.** `ybreak=(low, high)` leaves out the values between: each panel whose values span them is drawn
in two parts, one above the other, each as tall as its range of values, with break marks where the axis is cut, so
one spike (2020's growth) no longer squashes the rest. `xbreak=(first, last)` leaves out those periods, both
included: each dated panel that spans them is drawn in two parts side by side, on one value scale ("1986:2004" is
whole years, on a quarterly axis too). A panel the break does not reach is drawn whole. One split per figure; a split
value axis is for panels without a right-hand axis (split the dates of one that has). The title and legend go on the
first part; recorded styles, notes and lines reach both parts; the data are unchanged. In the app: **Split values**
and **Split dates** under Plot series' Advanced settings, and in **Edit… → Titles, shading and marks**.
`matplotlib_code()` writes the same split.

**Mixed frequencies.** A panel can mix frequencies: a monthly indicator beside quarterly GDP. Its date axis
takes the finest one, and each value is drawn at the middle of its own period, so a quarter's bar sits over its
three months. A span given in quarters covers whole quarters of a monthly series (`span=("2023Q1", "2023Q4")` is
January to December).

**The chart gallery.** `examples/gallery/chart_gallery.ipynb` shows every kind of chart pymetrix draws, each with
its code and the matplotlib it comes down to; run it from top to bottom.

**Macro charts.** Three kinds draw what macroeconomists chart most:

```python
from pymetrix.figures import Fan, contributions_figure, vintages_figure   # for charts built from tables or series

data.plot(["dc", "di", "dg"], kind="contributions", total="dy")   # stacked parts, the total as a line
data.plot("gdp", kind="fan", coverages=(0.3, 0.6, 0.9))           # bands from gdp_lo90/gdp_hi90 or gdp_se columns
data.plot(["gdp_jan", "gdp_apr", "gdp_jul"], kind="vintages")    # releases of the same data, oldest first
```

- **Contributions:** each part is a stacked bar, positive above zero and negative below, and the total is a dark
  line. Without `total`, the parts' sum is drawn. With it, whatever the parts leave unexplained is drawn as
  **other**, so the bars always add up to the line. `contributions_figure(parts, total)` does the same from a
  table, and shock and data decompositions draw their charts this way.
- **Fans:** a `Fan` layer is a central path with a band at each coverage. The inner bands are darker, since each
  one overlaps the wider ones. A dashed line marks where the bands, the forecast, begin. The bands come from
  `{name}_lo{pct}` and `{name}_hi{pct}` columns (any coverages present), or are normal bands from `{name}_se`.
  `Fan.normal(center, se)` and `Fan.from_frame(frame, name)` make one for any figure's panel, and a forecast
  round's fan chart is one: it starts at the last actual value, its bands opening from there, with one legend in a
  row above the first panel (`Panel(..., legend_loc="above")`; `""`, the default, puts a legend in the best place
  inside the panel). Each band and the line can be styled (`fig.band("90%", alpha=0.1)`).
- **Vintages:** one line per release, from light to dark, the latest thickest, to show how the data were revised.
  `vintages_figure({"January": jan, "April": apr})` does the same from a mapping.

In the app, **Data → Plot series…** has the three kinds. Contributions take a **total**, and fans their
**coverages** in per cent.

**Coefficient plots and heatmaps.**

```python
coefficient_figure({"short": short, "long": long}, level=0.9)        # estimates with 90% intervals, side by side
coefficient_figure(fit, terms=["dlog(y)", "dlog(c)[-1]"])           # some coefficients, in this order
data.plot(["gy", "gc", "gi"], kind="correlation")                   # the correlations, as a heatmap
heatmap_figure(table, label="response")                             # any table of values
heatmap_figure(posterior, view="Summary", scale="auto")             # each column its own scale where units differ
```

- **Coefficient plots** use matplotlib's `errorbar`: a marker at each estimate and a line across its interval, one
  row per coefficient, the first at the top, with a dashed line at zero. Several results stand side by side for
  each coefficient. The intervals come from each result's `params.interval`, so Bayesian estimates show their
  posterior intervals. An `Estimates` layer with dated estimates runs along the dates instead, for a coefficient
  over time. The markers style as a line: `fig.line("long", marker="s")`.
- **Heatmaps** use `imshow`, with a colour bar. Values of both signs get a red–blue scale symmetric about zero (or
  `center`); others a single-hue one. Each value is written in its cell, in white where the colour is dark.
  `cmap`, `center`, `vmin`, `vmax`, `annotate` and `fmt` change that. Correlations always run from −1 to 1.

In the app: **Estimate → Coefficient plot…** (tick the equations; the coefficients and the level are optional),
and **correlation** among Plot series' kinds.

**Distributions.**

```python
data.plot("gdp_growth", kind="histogram", bins=20)      # a histogram per series, with the normal curve
data.plot(["gy", "gc", "gi"], kind="ecdf")              # empirical distributions, in one panel
data.plot(["gy", "gc"], kind="hist2d", bins=15)         # a two-dimensional histogram of gc against gy
```

- **Empirical distributions** use matplotlib's `ecdf`: at each value, the share of observations at or below it.
  They compare distributions without choosing bins; an `Ecdf` layer with `complementary=True` shows the share
  above. Each styles as a line.
- **Two-dimensional histograms** use `hist2d`: the plane cut into bins by bins cells, coloured by how many
  observations fall in each, with a colour bar. Use one in place of a scatter plot when points would hide each
  other. Tick the x series first.
- **Violin plots** use `violinplot`: each series' kernel density (Scott's bandwidth), mirrored, from its lowest
  to its highest value, with the median marked: a box plot that shows the shape.
- **Hexagonal bins** use `hexbin`: as a 2-D histogram, with hexagons, which tile the plane more evenly; for many
  observations. Tick the x series first.
- **Bins:** `bins` sets a histogram's or 2-D histogram's number of bins (30 by default), or how many hexagons
  across. In the app, **Plot series** shows **bins** for the three kinds.

**Surfaces and contours.**

```python
surface_figure(yields, label="yield")                    # a table over a grid as a 3-D surface (dates x maturities)
surface_figure(yields, kind="contourf", clabel=True)     # or filled contours, the levels written on lines
data.plot(["m3", "y1", "y2", "y10"], kind="surface")     # series side by side by date: a yield curve over time
fig.view_init(elev=20, azim=-130)                        # turn a 3-D panel; recorded like any style call
```

- **Surfaces** use `plot_surface` on a 3-D panel, coloured by height with a colour bar; `kind="wireframe"` draws
  `plot_wireframe`. The table's columns run along x, its rows along y, and either may be dates (labelled as
  periods), numbers or names. Every cell is drawn. `Line3d` and `Scatter3d` put lines and points in a 3-D panel;
  a series of periods on any axis is drawn as dates.
- **A 3-D panel** holds only 3-D layers. It takes `view_init`, `set` (`zlim`, `zlabel`...), `ticks(axis="z")`,
  `grid`, `legend` and `call`; a figure's `ylim` limits the heights. Shading, events, reference lines and notes are
  for flat panels.
- **Contours** use `contourf` (filled) or `contour` (lines): `levels` (how many, or which), `clabel=True` to write
  each level, `color` for lines in one colour over a filled contour, and a colour bar.
- **In results:** a volatility surface's **Surface** tab (`surf.plot_surface()`, by strike or `x="moneyness"`), and
  a TVP-VAR's **Responses over time** (`tvp.irf_surface()`: each response by horizon and date).

**The new charts in results.**
- **Equations** have a **Q-Q plot** tab: the residuals, standardised and sorted, against the normal distribution's
  quantiles, with the 45-degree line they lie along if normal (`fit.qq_plot()`). Fitted to a time series, they
  have a **Correlogram** tab and its chart (`fit.correlogram(20)`), and with a GARCH variance one for the squared
  residuals. Their statistics include AIC, BIC and Hannan-Quinn (`fit.stats["hq"]`), as totals.
- **Impulse responses** have a **Heatmap** tab: a row per response and shock, a column per horizon
  (`irf.heatmap()`). **Variance decompositions** have one too: each variable's shares by shock, in %, at the last
  horizon (`fevd.heatmap(horizon)`).
- **Any table of numbers** opens as a heatmap: the **Heatmap** button under a table in the app, or
  `heatmap_figure(granger, view="Results")` in code (recorded as that step).
- **Columns in different units** (a posterior summary's means beside its effective draws; estimates beside t
  statistics) each get their own scale with `scale="column"`: each column's smallest value at one end of the colour
  map and its largest at the other, the cells still showing their values, and a note in place of the colour bar.
  `scale="auto"` (what the Heatmap button uses) picks that when one column's typical size (the median of its
  absolute values) is more than ten times the largest value in another, and keeps one scale for correlation and
  covariance matrices, responses by horizon or by shock, and shares such as variance decompositions; `"table"`, one
  scale, is the default in code. The button also leaves out a posterior's R-hat and effective draws
  (`diagnostics=False`), which describe the sampler. The numbers in the cells are dark on light cells and white on
  dark ones, at a contrast of at least 4.5:1.

**A house style.** One figure's recorded style can start every figure:

```python
set_house_style(figure_of(chart))      # every figure drawn from now on starts from chart's style calls
set_house_style(None)                  # and stops
fig.with_options(house=False)          # one figure without it
```

- **Where it applies:** call by call, as `with_style_of` copies a style: a series' call where that series is
  drawn, a panel's where the panel is. A figure's own calls come after, so they win. Notes, text, lines and bands
  stay with their chart unless `notes=True`.
- **Recorded and kept:** in the app, **Edit… → Use as house style…** records `set_house_style(figure_of(chart))`;
  opening the project sets it again, and a project without one clears it. A saved result keeps the calls it was
  drawn with, so it looks the same anywhere. The standalone matplotlib script writes them out too.

**Arrows, text boxes and changes.** **Edit… → Add an arrow…** draws an arrow from one point to another, with
optional words at its tail: its style (->, -|>, <->, fancy, wedge), straight or curved (the curvature is
matplotlib's `rad`), its colour, width and head size. **Add a text box…** puts words in a box: round, square,
sawtooth or no box, its fill and edge colours, padding, font size and alignment. Both are recorded as `annotate` or
`text` calls with matplotlib's `arrowprops` and `bbox`, so they replay, appear in reports and in the figure's
matplotlib code. To draw one as in a spreadsheet's chart, press **Arrow** on the figure's toolbar (or **Edit… →
Draw an arrow on the figure**), then press where it starts and drag to where it points: the arrow is drawn as you
drag, added when you let go (two clicks work too), and left selected to adjust. Draw as many as you like; on a
result's graph each one joins those already added.

**Mark a change** (Edit… → Mark a change…, or Mark a change on the figure and click two points of a series, each
snapping to its nearest data point) draws a double-headed arrow, or a bracket, from one point to the other,
labelled with the change, the percent change, or both, and optionally the rate a year (compound for the percent
change). The change is in percentage points (pp) where the panel's axis is a rate or a percentage. In Python,
`fig.change("u", "2019Q4", "2020Q2", show="both", units="auto", annualise=False)`; the values are read from the data
each time the figure is drawn, so the label stays right when the data change, and `matplotlib_code()` writes it as
the arrow and label it draws.

**Editing on the figure.** Click a note, an arrow, a text box, a line at a date or value, a band or a change marker
to select it: small squares mark its handles. Drag it to move it (Shift: only across, or only up and down), or drag
a handle to move an arrow's head or tail, a band's edge or a change marker's end; dates snap to periods, and a change
marker's ends to the series' data points, its label recomputed. Delete removes it, a double-click on words edits
them, a double-click elsewhere (on an arrow's line, a line or a band) opens its **Format…** form, and Esc
puts a drag back or clears the selection. An arrow is picked by its line, within a few pixels, words or not. Each edit is recorded as the step's new arguments (a Change
a style or Recorded styles step, in the data's terms), so it survives a resize, exports, reports and replay, and
Undo and Redo apply. The handles are never recorded or exported.

**Format… and the right-click menu.** The Format… form sets an added item's look in words and lists: an arrow's
colour, line width, head (open, filled, at both ends, none, fancy, wedge), head size and curve; a text box's words,
colour, size, box shape, fill and edge; a line's colour, width and style; a band's colour and opacity; a change
marker's colour, shape (arrow or bracket) and label. Colours are typed (navy, #1f77b4) or chosen with **Choose…**.
A right-click on the figure opens a menu for what is under the mouse: on an added item, **Format…**, **Edit the
words…**, **Bring to front**, **Send to back**, **Edit as code (every property)…** (every keyword, as Change a style) and
**Delete**; on a series, **Format series…** (Figure style, filled in with the series); anywhere in a panel, **Format
axes…** (limits, scale, labels, ticks, grid and legend), **Draw an arrow** and **Add a note here…**; and the figure's
**Titles, shading and marks…**, **Copy image** and **Save…**. On a Mac, right-click is a two-finger click on the
trackpad or Ctrl+click; the keyboard's Menu key opens it too. Every choice is a recorded step, as above.

**What is on the chart.** In the app, click on a series (a line, step, area, band, bar or scatter points) and a
small box gives its name; click on or near one of its data points (within a few pixels) and the box gives the
period as the axis shows it (2020Q1) and the value, with the point circled. A bar gives its period or category and
value, and a heatmap's cell its row, column and value; 3-D panels turn when dragged instead. A click on empty space,
or Esc, takes the box away. It is only on the screen: nothing is recorded, and saved and copied images and reports
are drawn without it. **Show values** on the figure's toolbar shows the value under the mouse as it moves, in the line
under the figure (off unless switched on).

**Reports.** Figures in reports are drawn at the size they appear (`Report(figure_size=(6, None))` by default, 6
inches wide; any size `save` takes, such as `"column"`), so their text prints at its size in points. Each figure
gets alt text saying what it shows (its title, panels, series and footnote; `fig.describe(technical=False)`): in
Word documents, HTML and Markdown.

**Saving for publication.** `save` sizes a figure for print or slides:

```python
fig.save("gdp.pdf", size="column")                  # 89 mm wide (a journal column); "page" 183 mm, "slide", "half-a4"
fig.save("gdp.png", size=("120mm", "80mm"), dpi=300)
fig.save("gdp.svg", size=(6, None), editable_text=True)   # inches; None keeps the figure's shape
fig.save("logo.png", transparent=True)
```

- **Text keeps its size in points,** so a narrower figure has relatively larger type, as print needs.
- **Fonts:** PDF and EPS embed theirs as TrueType, which journals ask for. SVG draws text as outlines unless
  `editable_text=True` keeps it as text.
- **The file says what made it:** the versions of pymetrix and matplotlib, the title, the series and the recorded
  style calls (`fig.describe()`), in the PNG's text, the PDF's Subject or the SVG's Description. Dates are left
  out, so the same figure writes the same file.

In the app, **Save…** under a figure opens **Save figure**: the file, a size, the resolution and the background,
recorded as a step (`figure_of(fit).save('fit.pdf', size='column')`).

Results show their figures in a **Graph** tab of their windows, with **Edit…**,
**Copy image** (paste into Word or PowerPoint), **Save…** and **Copy data**; notebooks
show figures inline; **Data → Plot series…** makes one in the app.

**Edit…** changes a figure:
- **A Plot series figure:** it reopens that dialog as you left it.
- **Any other figure:** it opens **Figure options**, filled in with how the figure is drawn now. It covers the
  title, subtitle, footnote, shading, events, reference lines, log scale and limits.
- **Where the result goes:** a result's figure is kept as a new object (`fit_graph`), so the result itself is
  unchanged.
- **The script:** the edit is a step like any other, so it replays.

In code:

```python
fig = figure_of(fit).with_options(title="Consumption", shade=data["recession"], events="2020Q1=COVID")
```

**Data → Plot series…** takes the same choices as `data.plot`:
- the kind of chart;
- **Stacked**, for bars and areas;
- a **Transform**;
- **Shaded periods**, for periods such as recessions;
- **Events**, for labelled lines at dates;
- **Reference lines**, for labelled lines at values;
- **Label bars with values**, for bar charts;
- **Same scale for every panel**, for one panel each;
- under Advanced settings: **Right-hand scale**, **Log scale**, the lowest and highest values shown, and
  **Split values** or **Split dates** to split an axis.

Bars for several series stand side by side. Stacked bars and areas put negative values below zero, as contributions
to growth do. Horizontal bars list the periods, or a cross-section's entities, down the side. A zero line is added
where bars, areas or changes cross zero.

A figure is data (panels of lines, bands, bars, scatter points, histograms and box plots) drawn with
matplotlib (installed with pymetrix); dates are
labelled in the data's own periods (2023Q1) and colours stay distinguishable
with colour-vision deficiencies.

## Local projections

```python
lp = LocalProjection("100*log(gdp)", "mp_shock", horizons=12, lags=4, controls=["infl"]).fit(data)
lp.irf                     # response, se, 68% and 90% bands, observations, by horizon
lp.fits.coefficients()     # the regressions side by side
```

Options: `contemporaneous=` (controls ordered before the shock, recursive
identification), `long_difference=True` (log levels), `cov="hac"` (Newey-West
with h+1 lags) or `"white"` (lag-augmented), `state=` (state-dependent
responses with a test of the difference), `scale=`, `instruments=` (LP-IV: the shock instrumented by external
instruments, such as a high-frequency surprise, by two-stage least squares at each horizon, with its first-stage F).
Panels get entity fixed effects and clustered standard errors. **Estimate → VARs and local projections → Local
projections…** in the app.

## Smooth-transition autoregressions

```python
star = SmoothTransitionAR("dgdp", lags=2, delay=1).fit(data)               # logistic (LSTAR)
SmoothTransitionAR("dgdp", lags=2, transition="exponential").fit(data)     # exponential (ESTAR)
star.summary(); star.transition_path(); star.forecast(data, ("2025Q1", "2025Q4"))
terasvirta(data, "dgdp", lags=2, delay=1)                                  # is a transition needed at all?
```

The two regimes blend by a logistic or exponential function of the lagged series; the transition's speed is scaled by
the lag's standard deviation, so it does not depend on units. **Estimate → Nonlinear → Smooth-transition autoregression…** in the
app, and **Forecast → Forecast…** forecasts it.

## Smooth local projections

```python
lp = LocalProjection("100*log(gdp)", "mp_shock", horizons=20, lags=4, controls=["infl"])
smooth = lp.fit_smooth(data)            # penalised spline in the horizon; the penalty by cross-validation
smooth.irf; smooth.plot()               # the smooth response, its bands, and the ordinary LP beside it
lp.fit_smooth(data, penalty=0)          # no penalty: the ordinary local projection on the same periods
```

All horizons are estimated together on the periods where every horizon is observed. `order=2` (the default) shrinks
towards a straight line in the horizon, `order=1` towards a constant.

## Growth at risk

```python
gar = growth_at_risk(data, "dlog(gdp)*400", ["nfci", "dlog(gdp)*400"], horizons=(1, 4))
gar.summary()                      # the quantile regressions: how conditions move each quantile of future growth
gar.at_risk(4)                     # by period: growth at risk (5%), expected shortfall and the median, a year ahead
gar.quantiles(4); gar.density("2008Q4", 4); gar.plot()
```

Future growth (averaged over the next h periods) is regressed on today's conditions at the 5%, 25%, 50%, 75% and 95%
quantiles; each period's predicted quantiles are matched by a skewed t (Adrian, Boyarchenko and Giannone 2019), which
gives the whole predictive distribution. **Estimate → VARs and local projections → Growth at risk…** in the app.

## Bayesian single equations

```python
from pymetrix.estimation.priors import InvGamma1, Normal
fit = Equation("c ~ y").fit(data, method=Bayes(prior={"y": Normal(0.8, 0.1), "sigma": InvGamma1(0.5, 1)}))
fit.summary()                                   # posterior mean, s.d., median, 90% HPD, R-hat, effective sample size, MCSE
fit.draws                                       # one row per draw, with its chain
Equation("d ~ x", family=Logit()).fit(data, method=Bayes(prior={"x": Normal(0, 2)}, chains=4))
```

Coefficients without a prior are flat; with no prior on the error, 1/sigma^2. Gaussian equations with normal priors
(and an inverse gamma, or none, on the error) are sampled by Gibbs; others by random-walk Metropolis-Hastings.
`Bayes(..., sampler="nuts")` uses the No-U-Turn sampler for any of them, with analytic gradients: fewer, less
correlated draws (compare the ESS column), its burn-in the warmup that tunes its step size and mass matrix; the notes
report divergent transitions, which mean the draws near them may be biased. `metric="dense"` adapts a dense mass
matrix, which undoes correlations between coefficients (a regressor far from zero makes the intercept and slope
correlate near -1: a diagonal one then needs many more gradients a draw); `metric="mode"` keeps the inverse Hessian
at the mode and tunes the step size alone.
The MCSE column is the Monte Carlo standard error of each posterior mean, s.d. / sqrt(ESS): how far the mean of
the draws may be from the posterior mean itself. Draw more until it is small beside the s.d. (and beside the precision
you report). `fit.stats["mcse_y"]` and `fit.stats["ess_y"]` give it and the ESS by parameter.
In the app: **Estimate → Equation…** with method Bayes, the family (Gaussian, logit, probit, Poisson) and, under
Method settings, the priors typed as `y=Normal(0.8, 0.1); sigma=InvGamma1(0.5, 1)` (`Bayes(prior="...")` reads the
same text in code), the draws and chains.

## Bayesian VARs

```python
bvar = VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=Minnesota(tightness=0.2)))
best = VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=Minnesota(
    sum_of_coefficients=1, initial_observation=1, optimize=True)))   # hyperparameters by marginal likelihood
best.hyperparameters, best.log_ml
best.irf(20).plot()                         # posterior median with credible bands
best.forecast(data, ("2025Q1", "2026Q4"))   # predictive mean, median and interval
```

The conjugate form gives exact posterior draws and marginal likelihood. Giannone, Lenza and Primiceri's
hierarchical prior gives the hyperparameters (tightness, sum-of-coefficients, single-unit-root and the scales psi)
priors of their own and estimates them, at their posterior mode or integrated over by MCMC:

```python
glp = VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=GLP()))              # at the mode
glp.hyperparameters                                   # tightness, sum_of_coefficients, initial_observation, psi ...
full = VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=GLP(mcmc=True), draws=5000))
full.hyperparameter_draws.describe()                  # their posterior; bands and forecasts integrate over them
```

The independent form shrinks other variables' lags harder than a variable's own, by Gibbs sampling; its log
marginal likelihood is Chib's estimate from the draws, and `optimize=True` chooses its hyperparameters by it:

```python
VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=NormalWishart(tightness=0.2, cross_tightness=0.5), draws=5000))
VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=NormalWishart(optimize=True), draws=5000))   # slower: a Gibbs run per trial
```

Use
`own_mean=0` (or a mapping per variable) for growth rates. A fitted Bayesian VAR
works wherever a VAR does: identification, variance decompositions, tests. Tick
**Bayesian (Minnesota prior)** in **Estimate → VARs and local projections → VAR…** in the app.

## DSGE-VARs

A DSGE-VAR(lambda) (Del Negro and Schorfheide 2004) is a VAR in a linearised model's observed series whose prior is
the model itself: its population moments, weighted as lambda T observations. Large lambda holds the VAR to the
model's restrictions, small lambda leaves it close to least squares; lambda is at least (k + n) / T.

```python
from pymetrix import priors, templates

nk = templates.new_keynesian()
dv = nk.dsge_var(data, lags=4, lambda_="grid", estimate=["kappa", "rho"],   # lambda-hat over a grid
                 priors={"kappa": priors.Gamma(0.15, 0.05), "rho": priors.Beta(0.7, 0.1)})
dv.lambdas(); dv.lambda_plot()          # the marginal data density by lambda: the hump, its top lambda-hat
dv = dv.mcmc(draws=10000, seed=1)       # the model's parameters by Metropolis-Hastings
dv.irf(20).plot()                       # responses identified by the model's rotation, with posterior bands
dv.dsge_irf(20).plot()                  # the model's own, to compare
dv.forecast(("2025Q1", "2026Q4"))       # predictive mean, median and interval
dv.var_draws(500)                       # the VAR's coefficients and covariance, given theta's draws
```

`lambda_=1.0` fixes lambda; `lambda_=priors.Gamma(1, 0.5)` estimates it with the parameters (as `lambda`). Any
`Model` takes the same (`model.dsge_var(data, measurement, ...)`); a Dynare file's `estimation(dsge_var=...,
dsge_varlag=...)` runs with `read_mod(path).dsge_var()`. The observables must be stationary (growth rates, rates,
gaps), and the responses need as many shocks as observables. In the app: **Model → Estimation → Estimate DSGE-VAR…**, then
**Model → Estimation → Metropolis-Hastings…** on its result.

## GMM flavours and weak-instrument tests

```python
m = GMM(instruments=["w", "z1", "z2"], steps="cue")       # continuously updated: S re-evaluated at every theta
fit = Equation("y ~ x + w").fit(data, method=m)
c_test(Equation("y ~ x + w"), data, GMM(instruments=["w", "z1", "z2", "z3"]), ["z3"])   # is z3 a valid instrument?
anderson_rubin(Equation("y ~ x + w"), data, "x", ["z1", "z2"], 2.0)   # is the coefficient on x 2? valid if z is weak
anderson_rubin_set(Equation("y ~ x + w"), data, "x", ["z1", "z2"], np.linspace(0, 4, 401))   # the robust confidence set
stock_wright(Equation("y ~ x + w"), data, m, {"const": 1, "x": 2, "w": -0.5})   # all parameters, any GMM model
md = minimum_distance(pi, V, lambda th: [th[0], th[0] * th[1]], [0.5, 0.5])   # structural parameters from reduced form
```

With `cov="classic"`, CUE in a linear model is LIML. The C test's statistic is the fall in Hansen's J when the
suspect instruments are dropped. The Anderson-Rubin set may be unbounded or empty: with weak instruments, that is
the honest answer. **Estimate → Equation…** with GMM offers continuously updated weighting; **Test → Weak-instrument
tests…** runs Anderson-Rubin (a test or the set), Stock-Wright and the C test on an IV or GMM fit.

## Estimation by simulation

```python
def simulate(theta, shocks):                       # shocks: T + 1 standard normals, drawn once and held fixed
    return shocks[1:] + theta[0] * shocks[:-1]      # an MA(1)

ar3 = lambda z: ar_coefficients(z, 3)[1:4]          # the auxiliary model: an AR(3)'s lag coefficients
fit = indirect_inference(y, simulate, ar3, [0.2], paths=20)
fit = simulated_moments(y, simulate, lambda z: np.column_stack([z[1:], z[1:]**2, z[1:] * z[:-1]]), [0.2])  # per-period moments
fit = efficient_method_of_moments(y, simulate, lambda z: ar_coefficients(z, 3), ar_score(3), [0.2], length=20)
fit.summary()                                       # estimates, standard errors inflated by (1 + 1/S), the J test
```

For a model that can be simulated but has no likelihood in closed form: indirect inference matches an auxiliary
model's estimates, simulated moments match chosen moments, and the efficient method of moments sets the auxiliary
score to zero over a long simulation (`length` times the sample). A continuous-time model enters through its
discretisation, simulated finely enough. `ar_coefficients` and `ar_score` come from
`pymetrix.simulation_estimators`.

## Structural VARs by A-B restrictions

```python
S = "*"                                                       # a free entry; numbers are fixed
svar = var.identify(AB(A=[[1, 0, 0], [S, 1, 0], [0, S, 1]], B=[[S, 0, 0], [0, S, 0], [0, 0, S]]))
svar.structural()          # A and B with standard errors; the LR test of the over-identifying zero in A
svar.irf(20).plot()        # responses, decompositions and bootstrap bands as for any identification
mixed = var.identify(AB(B=[[S, S, S], [S, S, 0], [S, S, S]], long_run=[[S, 0, 0], [S, S, S], [S, S, S]]))
```

A u = B e with independent unit-variance shocks e. A defaults to the identity and B to all free; `long_run` sets
zeros of the long-run effects (with A the identity), so impact and long-run zeros combine (Galí). **Estimate → VARs and
local projections → Structural VAR (A-B)…** in the app takes the matrices as text: `1 0 0; * 1 0; * * 1`.

## Max share, news shocks, narrative restrictions and heteroskedasticity

```python
var.identify(MaxShare("tfp", horizon=40)).irf(40).plot()                 # the shock behind most of tfp's variance
var.identify(MaxShare("tfp", horizon=40, zero_impact=True))              # a news shock (Barsky and Sims)
var.identify(SignRestrictions({"monetary": {"ff": "+", "cpi": "-"}},
                              narrative_signs={"monetary": {"1979Q4": "+"}},   # Volcker: a tightening
                              narrative_major={"monetary": {"1979Q4": "ff"}}))  # and the main reason ff rose
hsvar = var.heteroskedasticity_svar(["1984Q1"]); hsvar.summary(); hsvar.irf(20).plot()
```

Narrative restrictions need the VAR's residuals, so their bands span the identified set at the estimates. With two
variance regimes the impact matrix is exactly identified if the shocks' variance ratios differ (the summary shows them).
In the app, **Estimate → VARs and local projections → Impulse responses…**, **Variance decomposition…** and
**Historical decomposition…** offer MaxShare and News (Variable, Horizon and, under Advanced, the first horizon) and,
for responses and variance decompositions, Heteroskedasticity (Breaks: `1984Q1`, or several dates for more regimes);
Impulse responses also offers Generalised, `var.girf(...)`, Pesaran and Shin's ordering-free responses.

## FIML and VECM restrictions

```python
klein = System([Equation("c ~ p + p[-1] + w"), Equation("i ~ p + p[-1] + k[-1]"), Equation("w ~ x + x[-1] + t"),
                Equation("x = c + i + g")], endogenous=["p"])
klein.fit(data, method=FIML())                       # full-information maximum likelihood, identities included
vecm.test_beta([[1, 0], [-1, 0], [0, 1], [0, -0.8]])  # do the cointegrating vectors lie in these columns?
vecm.test_weak_exogeneity(["b"])                     # does b fail to adjust to the relations?
vecm.irf(20, draws=500).plot()                       # responses of the levels with residual-bootstrap bands
vecm.historical_decomposition().plot()               # each shock's contribution to the levels, and the baseline
vecm.relations_plot(); vecm.relations()              # beta' y(t-1) (with restricted terms): stationary if they cointegrate
vecm.roots_plot(); var.roots_plot()                  # companion roots and the unit circle (K - rank on it for a VECM)
vecm.recursive_eigenvalues_plot()                    # Hansen-Johansen: the eigenvalues as the sample grows; steady if stable
vecm.fluctuation_test(draws=199); vecm.fluctuation_plot()   # their fluctuation statistic, bootstrap p-values
```

FIML needs as many equations, identities included, as endogenous variables, and no instruments. `test_beta`'s
H has a row for each variable, then one for a restricted constant or trend (cases "ci" and "coli"). In the app,
**Test → VECM restrictions and stability…** runs the three tests on a fitted VECM (H as text: `1 0; -1 0; 0 1; 0 -0.8`);
a VECM's result window offers it under Next ▸.

## More unit-root tests

```python
data.test("ng_perron", "y", trend="ct")           # MZa (the statistic), MZt, MSB, MPT with their critical values
data.test("ers_point_optimal", "y")                # ERS's PT
data.test("perron_break", "y", date="1973Q1", model="C")   # a break in level and slope at a known date
```

## Choosing an equation's lags

```python
search = Equation("dlog(c) ~ dlog(y)").select_lags(data, max_lags=8)          # lags of dlog(c), 0 to 8
search = Equation("dlog(c) ~ dlog(y)").select_lags(data, max_lags=4, of=["dlog(c)", "dlog(y)"])  # distributed lags
search.table; search.best["bic"]; search.equation("bic").fit(data)
```

Every candidate is fitted on the sample the largest allows, by the equation's own method (least squares,
or maximum likelihood with ARMA errors or GARCH), and compared by AIC, BIC and Hannan-Quinn as totals.
Lags already in the equation are not added again. In the app: **Estimate → Time series → Equation lag
search…**. VARs choose their lags with `VAR(..., lags="bic")` or `select_order`; ARDL models with `select="bic"`.

## Robust regression and restricted least squares

```python
fit = Equation("y ~ x1 + x2").fit(data, method=RobustLS())                  # Huber's norm, scale by the MAD
fit = Equation("y ~ x1 + x2").fit(data, method=RobustLS(norm="bisquare"))   # Tukey's bisquare: outliers get no weight
fit.extra["robust_weights"].sort_values().head()                            # the observations weighted least
Equation("y ~ k + l").fit(data, method=OLS(restrictions=["k + l = 1"]))     # constant returns imposed
Equation("y ~ x1 + x2").fit(data, method=IV(endog=["x1"], instruments=["z"], restrictions="x1 = 2 * x2"))
```

Robust regression (M-estimation) down-weights observations with large residuals, by iteratively reweighted least
squares from the least-squares start, the scale re-estimated each round by the MAD (or held at `scale=`); the
standard errors are Huber's H1, as statsmodels' RLM. The tuning constant (`tuning=`, default 1.345 for Huber and
4.685 for the bisquare) sets how far out a residual is down-weighted.

`restrictions=` imposes linear restrictions on OLS's and IV's coefficients, named as the summary names them, the
same as substituting them into the equation (the Wald test reads its hypotheses with the same parser: `dlog(y)`,
`x[-1]`, `log(P)` or a name in backticks, `` `dlog(y)` ``, are each one coefficient); the standard errors are of the chosen kind, with one residual degree of
freedom more per restriction, and a coefficient a restriction fixes has none. In the app: **Estimate → Equation…**,
method **Robust regression (M)** (the norm in Method settings, the rest under Advanced), and **Restrictions** in the
Method settings of OLS and IV.

## AR(1) errors, multiplicative heteroskedasticity, frontiers and non-nested tests

```python
Equation("c ~ y").fit(data, method=CochraneOrcutt())          # or PraisWinsten(); exact ML: errors=ARMA(1, 0)
Equation("y ~ x").fit(data, method=MultiplicativeHeteroskedasticity(variance=["z"]))
frontier = Equation("lq ~ lk + ll").fit(data, method=StochasticFrontier())
frontier.extra["efficiency"]                                  # each firm's E[exp(-u) | e]
j_test(Equation("c ~ y"), Equation("c ~ c[-1]"), data)        # is the first right, given the second? run both ways
cox_test(Equation("c ~ y"), Equation("c ~ c[-1]"), data)      # in the app: Test → Non-nested tests (J, Cox)…
```

ARMA errors and GARCH variance together, as fGarch and rugarch estimate them (conditional
likelihood, the first p periods conditioned on; normal or t errors):

```python
fit = Equation("r ~ 1", errors=ARMA(1, 1), variance=GARCH(1, 1, dist="t")).fit(data)
fit.extra["volatility"]                                       # the conditional standard deviation
fit.forecast(data, span=("2025-01", "2025-12"), dynamic=True)  # mean from the ARMA, variance from GARCH and the ARMA weights
```

With lags of the dependent variable among the regressors, a dynamic forecast feeds each period's
forecast into the next period's lags, and its standard errors include those lags' dynamics
(`Equation("dlog(y) ~ dlog(y)[-1] + x", errors=ARMA(1, 0))`). Other functions of the dependent
variable, such as `d(y)[-1]` in an equation for `y`, are solved as a one-equation model, without
standard errors: `Model([fit]).solve(data, span, draws=1000)` gives their distribution.

## Nonparametric estimation

```python
kde(data, "r", bandwidth="cv").plot()                         # a kernel density beside the normal
kreg = kernel_regression(data, "y", ["x1", "x2"])             # local linear, bandwidths by cross-validation
kreg.predict([[0.5, 1.0]]); kreg.plot("x1")
partially_linear(data, "y", linear=["x"], nonparametric=["z"])
nonparametric_autoregression(data, "y", lags=2)
semiparametric_garch(data, "r", draws=200)                    # GARCH with the shocks' density by kernel
```

**Estimate → Nonparametric** in the app has the density and the regression.

## Counts, durations and GARCH-in-mean

```python
inar(data, "claims"); poisson_autoregression(data, "claims"); acd(data, "duration", distribution="weibull")
ach(data, "rate_change")                                      # a 0/1 event: the hazard as the inverse of an ACD
garch_in_mean(data, "r")                                      # the variance in the mean: a risk premium
Equation("d ~ d[-1] + x", family=Probit()).fit(data, method=ML())   # a dynamic probit
var.girf(20)                                                  # generalised impulse responses: no ordering
fit.test("neural_network")                                    # neglected nonlinearity
```

## Threshold VARs, bilinear models and VARMA

```python
tvar = threshold_var(data, ["gy", "pi", "r"], lags=2, threshold="gy", delay=1)
tvar.summary(); tvar.regime_table(); tvar.test(draws=500)   # the threshold, each regime's VAR, Tsay's test
bilinear(data, "y")                                         # y_t = mu + phi y_{t-1} + b y_{t-1} e_{t-1} + e_t
fit = varma(data, ["gy", "pi"], p=1, q=1); fit.irf(20).plot()
Equation("d ~ x", family=Logit()).fit(data, method=ML(optimizer="bhhh", cov="opg"))
Equation("d ~ a + b*x^c", params=["a", "b", "c"], family=Logit()).fit(data, method=ML(start={"c": 1}))  # a nonlinear index
```

In the finance plugin, `mgarch(data, [...], model="deco")` gives every pair the same dynamic correlation (Engle and
Kelly), and `model="diagonal_vech"` a free GARCH for each variance and covariance. In the app: **Estimate → Nonlinear
→ Threshold VAR…** (then Next ▸ or **Test → Threshold VAR linearity (Tsay)…** for the test), **Estimate → Nonlinear →
Bilinear model…** and **Estimate → VARs and local projections → VARMA…**.

## Monte Carlo experiments

```python
def dgp(rng):                                     # one simulated dataset from a numpy Generator
    x = rng.normal(size=100)
    return Dataset.from_frame(pd.DataFrame({"y": 1 + 0.5 * x + rng.normal(size=100), "x": x}, index=periods))

mc = MonteCarlo(dgp, lambda d: Equation("y ~ x").fit(d), reps=1000, truth={"x": 0.5}).run()
mc.summary(); mc.draws                            # mean, s.d., bias and RMSE; one row per replication
MonteCarlo(dgp, lambda d: Equation("y ~ x").fit(d).test("wald", "x = 0.5"), reps=1000).run().rejection()
```

Each replication has its own random stream, so `workers=4` gives the same draws faster (for functions defined at
the top level of a module; lambdas run here). A replication that fails is counted, not fatal.

### The random number generator

Every random draw pymetrix makes (simulations, bootstraps, Metropolis-Hastings and SMC, Monte Carlo replications,
copula draws) comes from a NumPy `Generator` built from the seed you give. The bit generator under it is a
setting:

```python
set_random_generator("mt19937")        # the Mersenne Twister; also "philox", "sfc64"; "pcg64" is the default
get_random_generator()                 # 'mt19937'
with random_generator("philox"):       # only inside the block
    post = fit.mcmc(draws=5000, seed=1)
post.provenance.random_generator       # 'philox': results record the generator with the seed
```

The default, PCG64, is NumPy's own (`np.random.default_rng(seed)`), so results are those of earlier versions bit for
bit. The setting reaches parallel chains, SMC blocks and Monte Carlo replications in other processes, so they still
give the same draws in any number of processes. A project saves the generator with each step that ran with another
one and replays the step with it whatever is chosen when it is opened (its script calls `set_random_generator`
before those steps); a generator the opening pymetrix does not have is warned about and PCG64 used. In the app:
**Calculate → Random generator**, remembered between sessions.

Choosing the Mersenne Twister does **not** reproduce MATLAB's, Dynare's or Stata's draws: NumPy seeds MT19937
through `SeedSequence` rather than MATLAB's `rng(seed)`, and makes normal draws with its own ziggurat (MATLAB's
`randn` uses a different ziggurat, Stata's `rnormal` the inverse CDF), so the numbers differ even with the same
seed; simulations agree in distribution, not draw for draw. Paths that go through a library's legacy
`np.random.RandomState` are Mersenne Twister already and do not change with the setting: the `arch` package's
bootstraps, statsmodels' Markov-switching random starts and scikit-learn's `random_state`.

## Cointegrating regressions and heterogeneous panels

```python
cointegrating_regression(data, "lc", ["ly"], method="fmols")    # or "ccr", "dols" (leads and lags by BIC)
cointegrating_regression(data, "lc", ["ly"], span=("1960Q1", "2007Q4"), where="strike == 0")   # rows kept: consecutive
gregory_hansen(data, "lc", ["ly"], model="regime")              # cointegration with one break in level and slope
mean_group(panel, "y", ["x"]); cce(panel, "y", ["x"], pooled=True)
pmg = pooled_mean_group(panel, "y", ["x"], lags=1); pmg.summary(); pmg.hausman()
panel.test("pedroni", "y, x")                   # seven statistics; also "kao" and "westerlund" (Gt, Ga, Pt, Pa)
panel.test("breitung", "y", robust=True)        # a panel unit-root test robust to a common factor
panel.test("cips", "y", lags=1)                 # unbalanced panels too
```

In the app: **Estimate → VARs and local projections → Cointegrating regression…, Panel mean group and CCE…** (the
pooled mean group's window has its Hausman test) and **Test → Gregory-Hansen cointegration test…**. The cointegrating
regression's dialog has Start and End, and under Advanced **Rows where**, a condition on the rows; the rows kept are
taken as consecutive periods, as rows with missing values are.

The long-run covariance behind FMOLS, CCR and DOLS uses the Bartlett kernel with Newey and West's (1994) automatic
bandwidth, as arch 9 and R's sandwich choose it (pymetrix 0.1.0a18's rule, `bandwidth_rule="legacy"`, was
removed after 0.1.0a20; use 0.1.0a20 to repeat older numbers). The automatic bandwidth is a noisy choice, so check a few given bandwidths (`bandwidth=4`) before relying on a borderline z statistic.

### Critical values for short samples

```python
data.test("engle_granger", "lc, ly", critical_values="simulated")                   # 2,000 draws, seed 0
data.test("engle_granger", "lc, ly", max_lags=2, critical_values="simulated", exog="d2008", draws=5000, seed=1)
data.test("adf", "ly", trend="ct", critical_values="simulated")
```

ADF and Engle-Granger report MacKinnon's response-surface critical values and p-values, which are asymptotic: with
13 to 30 observations they are far off, and the lag search moves them further. `critical_values="simulated"`
simulates the null for this sample instead: Gaussian random walks of the sample's length (one for ADF, one per
variable for Engle-Granger), each tested exactly as the data are (the same trend terms, the same ADF regression on
the residuals, the same lag rule, searched and refitted). The table gives the 1, 5 and 10% critical values, the
p-value (the share of draws at or below the statistic) with its Monte Carlo standard error, the draws and the seed;
the seed, draws and random generator go into the provenance. `exog=` names regressors that are not I(1), such as
impulse or break dummies: they enter the levels regression of the data and of every draw as given. Left empty,
the largest lag searched is Schwert's `ceil(12 (n/100)^(1/4))` (statsmodels' default), capped in short samples so the
largest search regression keeps 4 residual degrees of freedom after the levels regression's columns: 4 lags at 16
annual observations with one regressor and a constant, where statsmodels' own default (7) leaves none and reports a
statistic of 0.0. A largest lag that would leave none is refused, with the largest allowed in the message. In the app, **Run test…** with
the test chosen: Critical values under Test settings, Draws, Seed and Exog under Advanced settings.

## Synthetic difference-in-differences and diffusions

```python
sdid = synthetic_did(data, "packs", "treated")     # a balanced panel; the treated all switch on together
sdid.summary(); sdid.plot(); sdid.unit_weights.sort_values()
diffusion(data, "r", model="cir", dt=1/12)          # exact maximum likelihood: vasicek, cir or gbm
```

The summary shows DiD and the synthetic control beside synthetic DiD. **Estimate → Treatment effects → Synthetic
DiD…** in the app.

## BVARs with stochastic volatility, FAVARs and Markov-switching VARs

```python
sv = BVARSV(["gy", "pi", "r"], lags=4).fit(data, draws=2000, burn=1000); sv.plot(); sv.irf(20, period="2008Q4")
fv = favar(data, panel=many_series, observed=["ff"], factors=3, lags=7, slow=slow_series)
fv = favar(data, many_series, ["ff"], factors="icp2", lags="bic")   # Bai-Ng's criterion; the lags by BIC
fv.irf(["ip", "cpi"], 48, draws=500).plot()                  # any panel series' response, in its own units, with bands
fv.fevd(["ip", "cpi"], 60)                                   # their common components' variance, shock by shock
ms = MSVAR(["gy", "pi"], lags=1, switching="intercept_variance").fit(data)
ms.summary(); ms.plot(); ms.durations                        # regimes, their probabilities, expected durations
sv.fevd(20, period="2008Q4"); sv.historical_decomposition()  # posterior means, with bands over the draws
ms.fevd(regime=1, periods=20); ms.historical_decomposition() # within a regime; along the most likely regime path
ms.irf(regime=1, periods=20, draws=500, coverage=0.9)        # bands from draws of the parameters
ms.historical_decomposition(draws=500)                       # bands with the regime path held fixed
```

A BVAR-SV's variance decomposition uses one date's volatilities over the whole horizon, as its responses do; its
historical decomposition takes each period's shocks through that period's impact matrix, draw by draw, and its
point is the mean over the draws (each draw's parts, and the mean's, add up to the data). An MS-VAR's historical
decomposition follows the most likely regime each period (`ms.regime_path()`, from the smoothed probabilities): that
regime's intercepts, dynamics and impact matrix, so a shift in the mean between regimes shows up in the baseline,
not in the shocks.

An MS-VAR's bands come from draws of its parameters from the maximum-likelihood estimates' asymptotic normal
distribution (`ms.parameter_covariance()` is the inverse of minus the Hessian of the likelihood; `ms.parameter_draws()`
the draws, each with valid covariance and transition matrices). They reflect the parameters' uncertainty only: the
historical decomposition's bands keep the most likely regime path, and a transition probability estimated at 0 or 1
is held there. Regimes that barely differ leave the likelihood flat, and the bands are refused.

The likelihood of a Markov-switching model has local maxima. `fit` starts EM from regime paths read off the data (each
variable split at its quantiles and by its rolling mean, the linear VAR's residuals split by their rolling variance,
k-means, a one-variable fit of each variable) and from `starts - 1` random splits; each start gets a short EM run,
starts whose regimes collapse into one are restarted, and the best few are run to convergence and polished on the
exact likelihood. Clearly separated regimes are found from the default starts in a few seconds. If the estimates
change with `starts` or `seed`, keep the fit with the highest `loglike`.

A FAVAR's bands bootstrap the whole two-step procedure, as Bernanke, Boivin and Eliasz do: the factors and
observed variables simulated from the VAR, the panel rebuilt from its loadings, everything estimated again
(`bootstrap="var"` holds the factors and loadings fixed). Shocks to the factors depend on how they are
rotated, so read the bands for shocks to the observed variables, which are ordered last. In the app:
**Estimate → VARs and local projections → Factor-augmented VAR…**, **BVAR with stochastic volatility…** and
**Markov-switching VAR…** (what switches: intercept, intercept_variance or all; the number of regimes); the BVAR-SV's
and MS-VAR's responses, at a period's volatilities or within a regime, come from Next ▸ → Impulse responses…, and
their decompositions from Next ▸ → Variance decomposition… and Historical decomposition….

## Spectra and long memory

```python
spectrum(data, "gy", bandwidth=5).plot()          # log spectral density by frequency
cross_spectrum(data, "gy", "gc").summary()         # coherence and gain by band: long run, business cycle, short run
gph(data, "vol"); local_whittle(data, "vol")       # d, with its standard error in details
arfima(data, "vol", p=1, q=0).summary()
```

In the app: **Test → Spectrum…** (periodogram, smoothed spectrum or cross-spectrum) and **Estimate → Time series →
Long memory…** (GPH, local Whittle or ARFIMA).

## Quantile VARs and spatial models

```python
qv = QuantileVAR(["gdp", "fci"], lags=2, quantile=0.1).fit(data); qv.irf(12).plot()   # the 10% quantile system
qv.fevd(12)                                       # quantile connectedness: variance shares of the quantile system
W = knn_weights(regions, ["lon", "lat"], k=5)
spatial_diagnostics(regions, "crime", ["income", "housing"], W)                       # Moran's I, LM lag and error
spatial_regression(regions, "crime", ["income", "housing"], W, model="lag").summary()
```

In the app, **Estimate → VARs and local projections → Quantile VAR…**, its responses by Next ▸ → Impulse responses…
and its variance decomposition by Next ▸ → Variance decomposition…. A quantile VAR has no historical decomposition:
away from the median its residuals are not centred, so there is no clean split into a baseline and shocks.
In the finance plugin, `mgarch(data, ["r1", "r2"], model="vech")` is the full VECH.

## TVP-VARs with stochastic volatility

```python
tvp = TVPVAR(["infl", "unemp", "rate"], lags=2).fit(data, draws=5000, burn=2000)   # Gibbs sampling: minutes
tvp.volatilities(); tvp.plot()                          # each shock's standard deviation over time, with bands
tvp.irf("1975Q1", 20).plot(); tvp.irf("2000Q1", 20)     # responses at two dates: did they change?
```

Priors come from least squares on the first `training` periods (40 by default), which the estimation sample then
starts after. `kq`, `ks` and `kw` set how much the coefficients, the contemporaneous relations and the volatilities
may drift (Primiceri's 0.01, 0.1 and 0.01). **Estimate → VARs and local projections → TVP-VAR with stochastic volatility…** in the app.

## Proxy SVARs

```python
var = VAR(["ff", "ip", "cpi"], lags=12).fit(data)
psvar = var.proxy_svar(data, "ff4", normalise_on="ff", periods=36)   # the surprise instruments the ff shock
psvar.impact; psvar.irf.plot(); psvar.first_stage                     # impact, responses with bands, first-stage F
var.proxy_svar(data, "ff4", normalise_on="ff", scale="unit")          # ff moves by one on impact
```

`normalise_on` names the variable whose shock the instrument identifies (default: the VAR's first); with
`scale="unit"` that shock moves it by one on impact. (It was called `target`, which still works with a
DeprecationWarning.) The instrument must be correlated with that variable's shock and no other. At horizon 0 the
relative responses equal LP-IV's with the same lags (`LocalProjection(..., instruments=["ff4"])`); beyond it the
VAR's dynamics extrapolate.
**Estimate → VARs and local projections → Proxy SVAR…** in the app.

## Staggered difference-in-differences

```python
cs = staggered_did(panel, "y", "d")                                   # Callaway and Sant'Anna, never-treated controls
cs.event_study(); cs.overall(); cs.plot()                             # by time since adoption; the overall ATT
staggered_did(panel, "y", "d", control="notyet")                      # not-yet-treated controls
staggered_did(panel, "y", "d", method="sun_abraham")                  # interaction-weighted event study
bacon_decomposition(panel, "y", "d").summary()                        # what two-way fixed effects averages
```

`d` is 0 before a unit adopts and 1 from then on; the panel must be balanced. With `covariates=["x"]` (each unit's
value in the first period), parallel trends need hold only among units alike in them: `estimator="dr"` (doubly
robust, the default), `"or"` (outcome regression) or `"ipw"`. **Estimate → Treatment effects →
Staggered DiD…, Bacon decomposition…** in the app.

## Panel VARs

```python
pvar = VAR(["inv", "sales", "cash"], lags=2, by="firm").fit(panel)           # GMM on forward orthogonal deviations
pvar.summary(); pvar.granger()                                                # clustered errors; Wald tests
pvar.irf(10).plot(); pvar.fevd(10)                                            # Cholesky by default; identify() as for a VAR
VAR([...], lags=2, by="firm").fit(panel, instrument_lags=4)                   # over-identified: two-step GMM, Hansen's J
VAR([...], lags=2, by="firm").fit(panel, transform="within")                  # least squares on demeaned data
VAR([...], lags="bic", max_lags=4, by="firm").select_order(panel)             # Andrews and Lu's MBIC, MAIC, MQIC, Hansen's J, CD
VAR([...], lags="bic", max_lags=4, by="firm").fit(panel)                      # the lags MBIC picks
```

Lag lengths are compared as Abrigo and Love's pvarsoc does: each estimated by GMM with the same instruments (the
levels' lags 1 to max_lags + 1) on the same rows, by Andrews and Lu's criteria built from Hansen's J (`"bic"` is
MBIC, `"aic"` MAIC, `"hq"` MQIC). With `transform="within"` the criteria are the usual ones of the residual
covariance, which inherit that estimator's bias when entities have few periods.

`by` names the panel's entity axis. Each variable and its lags are transformed by forward orthogonal deviations,
which remove the fixed effects without the bias demeaning brings when entities have few periods, and the lags are
instrumented by their levels. Response bands come from draws of the coefficients. In the app: **Estimate → VARs and
local projections → Panel VAR…**.

## Example cross sections and panels

`Dataset.example(name)` loads, besides the simulated macro economy:

| Dataset | What it is | For |
| --- | --- | --- |
| `mroz` | married women's labour supply, 1975 (Mroz 1987) | Tobit, Heckman, probit |
| `card` | returns to schooling and college proximity (Card 1995) | instrumental variables |
| `wage_panel` | young men's wages, 1980-1987 (Vella and Verbeek 1998) | panel models |
| `jobtraining` | Michigan job-training grants to firms, 1987-1989 | panels, difference-in-differences |
| `nile` | the Nile's annual flow, 1871-1970 | structural breaks |
| `randhie` | RAND Health Insurance Experiment doctor visits | count models |
| `anes96` | 1996 American National Election Study | ordered and multinomial models |
| `grunfeld` | Grunfeld's investment data, 11 firms | panels and systems |
| `policy_panel` | simulated: a policy in half of 50 states from 2010, effect 2.0 | difference-in-differences, event studies |
| `rd_sample` | simulated: a programme for scores of 0 or more, effect 1.5 | regression discontinuity |
| `synth_panel` | simulated: one region's policy from 2005, effect -3.0 | synthetic control |

Each dataset's `meta.license` says where it comes from: linearmodels' copies of Wooldridge's datasets, statsmodels'
public-domain datasets, or pymetrix's own simulations.

## GMM with your own moments

```python
euler = Equation("1 ~ beta * (c / c[-1])^(-gamma) * R", params=["beta", "gamma"])
fit = euler.fit(data, method=GMM(instruments=["c[-1]/c[-2]", "R[-1]"], cov="hac", start={"beta": 0.95, "gamma": 1}))
fit.stats["J"], fit.stats["J_pvalue"]                       # Hansen's test of the over-identifying restrictions
Equation("y ~ x + w").fit(data, method=GMM(instruments=["w", "z1", "z2"]))   # linear: two-step efficient IV-GMM
```

The moments are each instrument times the equation's residual (lhs minus rhs; write an Euler equation with `~`, since
`=` makes an identity). List every instrument, including regressors that are their own; a constant is added. The
first step is two-stage least squares; `steps="two"` (the default) re-weights once by the inverse of the moments'
covariance, `"iterated"` until the estimates settle. `cov` is that covariance: `"white"` (robust, the default),
`"classic"` or `"hac"`. In the app: GMM among the methods in **Estimate → Equation…**, with the instruments field.

Moments that are not an equation's residual times instruments (several equations at once, moments of a distribution,
a model's simulated moments written by hand) can be a Python function:

```python
def moments(p, d):                                          # each row's contributions, an n x L array or DataFrame
    u = p["beta"] * (d.c / d.c.shift(1)) ** -p["gamma"] * d.R - 1
    return pd.DataFrame({"u": u, "u_c1": u * d.c.shift(1) / d.c.shift(2), "u_r1": u * d.R.shift(1)})

fit = MomentGMM(moments, start={"beta": 0.95, "gamma": 1.0}, cov="hac").fit(data)   # steps, weights as GMM's
fit.summary(); fit.J, fit.stats["J_pvalue"]; fit.contributions               # the moments at the estimates
```

The function gets the whole frame (so `shift` reaches before the span); the rows used are those whose moments are
all finite. The first step weights the moments equally (W = I), as statsmodels' GMM does; derivatives are numerical.
`cov` is `"white"` or `"hac"`; `center=True` demeans the contributions in S. Code only: a function cannot be a dialog
step.

## Treatment effects

```python
did(panel, "y", "d", controls=["x"])                       # two-way fixed effects, errors clustered by unit
event_study(panel, "y", "d", window=(-4, 4)).plot()         # effects by period around adoption
rdd(data, "y", "score", cutoff=0)                           # the jump at the cutoff; the bandwidth by cross-validation
treatment_effect(data, "y", "treated", ["age", "educ"], method="aipw")   # selection on observables
synthetic_control(panel, "y", treated=1, start=2005)        # a weighted mix of untreated units
```

Difference-in-differences assumes parallel trends; the event study's effects before adoption check it. With
adoption at different dates and effects that grow, two-way fixed effects can mix comparisons, which the summary
notes. The RD estimate is local linear with a triangular kernel and a conventional robust standard error, shown
at other bandwidths too. Propensity weighting assumes nothing unobserved drives both treatment and outcome; the
window shows the overlap of the propensity scores. Synthetic control's p-value ranks the treated unit's post- to
pre-treatment fit among placebos in which each donor is treated in turn. In the app: **Estimate → Treatment
effects**.

## Double machine learning, shift-share IV, interactive fixed effects and conditional logit

```python
double_ml(data, "y", "d", controls)                                   # partially linear, OLS learners, 5-fold cross-fitting
double_ml(data, "y", "d", controls, learner="forest", repeats=5)      # random forests; five splits, the median
double_ml(data, "y", "d", controls, model="pliv", instrument="z")     # partially linear IV
double_ml(data, "y", "d", controls, model="irm", score="ATTE")        # 0/1 treatment: doubly robust ATT
ss = shift_share(regions, "dy", "dx", shares, shocks, controls=["total_share"])
ss.summary(); ss.rotemberg_table(); ss.shock_level()                  # AKM, AKM0 and BHJ errors; Rotemberg weights
interactive_fe(panel, "y", ["x1", "x2"], factors="auto", effects="twoway")   # Bai (2009), r by Bai and Ng
conditional_logit(panel, "union", ["married", "exper"])               # Chamberlain's fixed-effects logit
choice_logit(trips, "chosen", ["cost", "time"], case="trip", alternative="mode", alternative_constants=True)
me = clogit_effects(choice)            # or choice.marginal_effects(at="mean")
me.table("cost"); me.table("cost", "elasticity")                      # dP(row)/dcost(column), alternatives by alternatives
clogit_effects(clogit, fixed_effect=0)                                # panel logit: odds ratios, effects for a unit with fixed effect 0
me = clogit_effects(choice, dummy=True, changes={"cost": 1})          # discrete changes: 0/1 variables from 0 to 1, cost by 1
me.table("wifi", "discrete change")                                   # P(row) with the column's wifi 1, less with it 0
```

Double machine learning learns the controls' part of the outcome and the treatment by cross-fitting and solves a
Neyman-orthogonal score; with least-squares learners and no cross-fitting (`folds=1`) the partially linear estimate
is ordinary least squares. The learners `"lasso"`, `"forest"` and `"boosting"` need scikit-learn (`pip install
pymetrix[ml]`); `cluster=` keeps clusters whole in folds and clusters the errors. A shift-share IV builds the
instrument from shares and shocks; its AKM and BHJ errors allow regions with similar shares to share shocks, and the
Rotemberg weights say which industries drive the estimate. Interactive fixed effects need a balanced panel with many
units and periods; with no factors they are pooled least squares or two-way fixed effects. After a choice logit,
`clogit_effects` gives each alternative's probability's derivatives and elasticities in each attribute (own on the
diagonal, cross off it; each column sums to zero) and in each characteristic of the case, averaged over the cases or
at the means. The panel logit's probabilities depend on the fixed effects it conditions away, so it has no marginal
effects: `clogit_effects` gives odds ratios, and effects for a unit with a fixed effect you choose, labelled as such.
For a 0/1 variable a derivative is a poor guide: `dummy=True` gives the change in the probabilities when each one
goes from 0 to 1 (for an attribute, one alternative's at a time), and `changes={"cost": 1}` the change when a
variable rises by a chosen amount, with delta-method standard errors.
In the app: **Estimate → Microeconometrics**, and **Next ▸ Marginal effects…** on a conditional logit.

## Limited dependent variables and marginal effects

```python
Equation("visits ~ age + income", family=NegativeBinomial()).fit(data, method=ML())        # overdispersed counts
Equation("visits ~ age", family=ZeroInflatedPoisson(inflate="age")).fit(data, method=ML())  # extra zeros
Equation("rating ~ x1 + x2", family=OrderedProbit()).fit(data, method=ML())                 # cut points, no constant
Equation("mode ~ cost + time", family=Multinomial()).fit(data, method=ML())                  # against the first outcome
tob = Equation("hours ~ educ + kidslt6", family=Tobit(left=0)).fit(mroz, method=ML())       # censored at zero
heck = Equation("lwage ~ educ + exper").fit(mroz, method=Heckman("inlf ~ educ + kidslt6"))  # sample selection
tob.marginal_effects(); tob.marginal_effects(at="mean")
```

The likelihoods are pymetrix's own (checked against statsmodels, and against Wooldridge's Mroz examples for Tobit
and Heckman). Parameters with a range are reported on their own scale: the negative binomial's alpha, Tobit's
sigma, the ordered cut points, Heckman's rho and sigma. `predict` gives E[y], or each outcome's probability for
ordered and multinomial models. Marginal effects are the change in E[y] or in the probabilities for a unit change
in each regressor (0 to 1 for a dummy), averaged over the sample or at the means, with delta-method standard
errors; logit, probit and Poisson fits have them too, and the result window shows them. In the app: the family
choice under ML in **Estimate → Equation…**, and Heckman among the methods.

## Structural breaks at unknown dates

```python
fit.test("quandt_andrews")        # the largest Chow F over the middle 70% of the sample, and its date
b = fit.breaks(max_breaks=5)      # Bai-Perron: the best dates for 0 to 5 breaks; the number chosen by BIC
b.dates; b.segments(); b.plot()
```

A Chow test needs the date; Quandt-Andrews searches for it, and its p-value allows for the search (Andrews'
limiting distribution). Bai-Perron finds the dates of several breaks at once, each regime at least `trim` of the
sample, and BIC chooses how many. In the app: **Test → Run test…** with quandt_andrews, and **Test → Structural
breaks (Bai-Perron)…**.

## Sign restrictions

```python
sr = var.identify(SignRestrictions({"supply": {"gy": "+", "pi": "-"}, "demand": {"gy": "+", "pi": "+"}}, horizon=2))
sr.irf(20).plot()       # median of the accepted rotations; bands with a rotation for each bootstrap draw
sr.irf(20, draws=0)     # bands across the identified set at the estimates
sr.fevd(20)             # at the median-target rotation
```

Each named shock's responses must have the given signs on impact and for `horizon` periods after; shocks not
named stay unidentified. Rotations of the Cholesky factor are drawn at random (uniformly) and kept when they
satisfy the signs, until `rotations` (1,000) are accepted. Sign restrictions identify a set of models, not one:
the median response at each horizon may come from different models, so variance decompositions use the
accepted model closest to the medians (Fry and Pagan's median target). A Bayesian VAR takes one accepted
rotation for each posterior draw. In the app: SignRestrictions in **Impulse responses…**, **Variance
decomposition…** and **Historical decomposition…**, with the signs written as `supply: gy+, pi-; demand: gy+, pi+`.

### Zero and sign restrictions

```python
zs = var.identify(SignRestrictions({"monetary": {"r": "+", "pi": "-"}}, zeros={"monetary": ["gy"]}))
zs.irf(20).plot()                       # the monetary shock does not move gy on impact, on every draw
var.identify(SignRestrictions({"supply": {"gy": "+"}, "demand": {"gy": "+", "pi": "+"}},
                              long_run_zeros={"demand": ["gy"]}))    # demand has no long-run effect on gy (in levels)
```

Zero restrictions say a shock does not move a variable on impact (`zeros`) or in the long run (`long_run_zeros`, the
cumulated response, for variables in differences); a shock may have zeros and no signs. Rotations are then drawn
as Arias, Rubio-Ramírez and Waggoner (2018) do: column by column, each in the space the zeros (and the columns
before it) leave, signed by the shock's first sign and kept when all its signs hold. That draw is not uniform given
the zeros, so each kept draw carries ARW's importance weight and the draws are resampled by it; with a Bayesian VAR
each posterior draw gets its rotations in turn, so the result is ARW's posterior. Shocks keep their columns (there is
no search over columns as with signs alone), and the zeros can at most just identify the VAR: ordered by their
number, n - 1 for the first shock, n - 2 for the next, and so on. Zeros alone that just identify it (recursive zeros)
give the Cholesky model. In the app: **Zero restrictions** (and, under Advanced settings, **Long-run zero restrictions**) beside **Sign restrictions**, written
`monetary: gy, pi`.

A historical decomposition (`var.historical_decomposition()`) splits each variable over the estimation
sample into the identified shocks' contributions and the baseline, the path from the first observations and
the deterministic terms with no shocks; the parts add up to the data. Under sign restrictions it uses the
median-target model. With `draws=500` it adds bands to each contribution: quantiles over a Bayesian VAR's posterior
draws (each draw decomposes the data with its own coefficients and impact matrix, and its parts add up to the data),
or over the accepted rotations of a sign-identified set; `hd.plot_bands()` and `hd.bands_table("gy")` show them, and
the point decomposition is unchanged.

A panel VAR is decomposed entity by entity: `pvar.historical_decomposition()` gives one decomposition per entity,
each adding up to that entity's data, with the baseline starting from the entity's first lags and carrying its fixed
effect (the entity's mean of the data less the lags' effect). `hd["firm 3"]` is one entity's; `hd.select("firm 3")`
turns the window's views to it (or give **Entity** in the dialog); `hd.summary()` lists the entities with their
samples and fixed effects. `draws=500` adds bands over draws of the coefficients, as the panel VAR's impulse
responses have. A proxy SVAR identifies one shock, so `psvar.historical_decomposition()` splits each variable into
that shock's contribution, the other shocks' together and the baseline; the shock is the residuals' part along its
impact column, whatever its scale. Its bands (`draws=500`) come from the same moving-block bootstrap as its
responses' bands.

## Dynare models

```python
dyn = read_mod("nk.mod")                      # a pymetrix Model with the file's calibration
dyn.steady_state(); sol = dyn.solve()         # first-order solution, Blanchard-Kahn checked
sol.n_stable, sol.n_predetermined; sol.roots() # the counts the conditions compare
sol.irf(20, shock_sd=dyn.shock_sd).plot()
```

Commands in the file (`steady`, `stoch_simul`...) are listed, not run.
**Model → Import → Dynare file…** in the app.

Dynare's macro processor runs first, as Dynare's preprocessor does: `@#define`, `@{...}` substitutions, `@#if`,
`@#ifdef`, `@#for` loops over arrays (`@#for c in ["us", "ea"]`, `@#for (a, b) in ...`), `@#include` (next to the
model file, so read the file by its path), `@#echo` and `@#error`. A directive the reader does not know, an unclosed
`@#if` or `@#for` and an include not found are errors with their line; the model editor shows them there, and the
problems of the expanded model at the lines they come from.

Correlated shocks are read from the shocks block (`corr e1, e2 = 0.3;`, or a covariance `var e1, e2 = 0.01;`) into
`dyn.shock_corr`, and pass wherever the shock sizes do:

```python
sol.irf(20, shock_sd=dyn.shock_sd, shock_corr=dyn.shock_corr)   # orthogonalised by Cholesky, as stoch_simul
sol.moments(dyn.shock_sd, shock_corr=dyn.shock_corr)            # variance with the full covariance
sol.system(["y", "pi"], dyn.shock_sd, shock_corr=dyn.shock_corr) # the filter's Q, off its diagonal
model.estimate(data, measurement, shock_corr={("ey", "ei"): 0.3})  # corr_ey_ei estimated, within (-1, 1)
```

With correlated shocks, impulse responses and variance decompositions are of the shocks orthogonalised by the
Cholesky factor of their covariance, in the order of `varexo`, as Dynare's: the first shock carries the part it shares
with the later ones. Without `shock_corr`, responses are to each shock alone, as before.

Files set up for estimation are estimated as Dynare would:

```python
dyn = read_mod("fs2000.mod")
dyn.priors, dyn.estimated, dyn.varobs, dyn.options   # estimated_params, varobs, estimation(...)
fit = dyn.estimate()                          # posterior mode, on the datafile named in estimation(...)
post = fit.mcmc(draws=10000, chains=2)        # as mh_replic, mh_nblocks
fit.laplace(); post.marginal_likelihood()     # Dynare's two log data densities
dyn.measurement_sd                            # measurement errors: stderr on a varobs variable in shocks
dyn.log_posterior(dyn.mode_file())            # at a mode file's values, as mode_compute=0 reports it
```

`estimated_params` is read in all three forms (initial values and bounds, priors, or both);
`stderr e` is the shock size `sd_e`, and `corr e1, e2` the correlation `corr_e1_e2` (within (-1, 1); a correlation
matrix that is not positive definite is ruled out). Dynare's priors map to pymetrix's, including the inverse
gamma of type 1 (`inv_gamma_pdf` -> `InvGamma1`, the usual prior on a standard deviation, with
an infinite standard deviation allowed). With `loglinear`, every variable is written `exp(x)`,
so the model is solved in logs, and the data are logged unless `logdata` is given. Data files
may be `.csv`, MATLAB `.mat` files, or simple MATLAB `.m` files (matrices, column selection,
`diff`, `log`...); `first_obs` and `nobs` select the sample, `presample` leaves the first periods
out of the likelihood, and `lik_init` sets the filter's start (2: variance 10 I, as Dynare).
Prior shapes may be written in any case (`BETA_PDF`), and a shape the reader does not know is an
error. A parameter without a calibrated value takes its `estimated_params` initial value.
Dynare's own `fs2000` example gives the same posterior mode, log posterior and Laplace density
as Dynare. Smets and Wouters (2007), from Johannes Pfeifer's DSGE_mod replication, gives Dynare's
decision rules (to 1e-12) and log posterior (to 1e-9 at Dynare's mode) for the 40-variable model.

## IRIS models

```python
qpm = read_iris("qpm.model", params={"b1": 0.7, "std_shock_y": 0.5})   # values missing from the file
qpm.solve().irf(20).plot()
system = qpm.system()                         # measurement equations from the file
qpm.estimate(data)                            # like any template model
```

The file is read as text; IRIS is not needed. Transition and measurement equations, `x{-1}`
timing, `diff` and `difflog`, `!log_variables` (those variables are solved in logs, so
responses are log deviations, as in IRIS), `!substitutions` and `!for` loops are read.
Shock sizes come from `std_` parameters; measurement shocks become measurement errors.
Steady-state references `&x` are read: the steady state is found with `&x` as `x`, then the
model holds it as a fixed parameter `ss_x` (re-read the file for other parameter values).
`!if`/`!elseif`/`!else` blocks choose by parameter values from `params=` or the file.
`!dtrends` (`obs += g*ttrend`) become deterministic trends of the observed series, taken out by
the filter and the likelihood and added to forecasts; give `trend_base="2020Q1"` (IRIS's
`baseYear`, where `ttrend` is 0) when they use `ttrend`. In code, any measurement map takes
them: `MeasurementMap([...], trends={"rate": "0.3 + 0.05*ttrend"}, trend_base="2001Q1")`.
`!reporting_equations` (`"Output gap, percent" gap = 100*y_gap;`) become the measurement map's reported series
(`qpm.measurement.reported`): shown with the smoothed history, forecasts and draws, never fitted; one may use those
before it. A file with reporting equations and no measurement equations gets a map that only reports (no state-space
form until measurement equations are given). Parameter descriptions become the model's parameter metadata
(`qpm.model.spec("b1").label`), and those of `std_<shock>` the labels of the shock sizes (`sd_<shock>`).
**Model → Import → IRIS file…** in the app.

## Systems of equations

```python
system = System([Equation("c ~ y + c[-1]"), Equation("i ~ y[-1] + i[-1]"), Equation("y = c + i + g")])
fit = system.fit(data, method=ThreeSLS())     # or SUR(), SUR(iterate=True), SystemOLS(), TwoSLS()
fit["c"]                                      # each equation as a FittedEquation
fit.wald("c.y = 0.5"), fit.test_diagonal()
fit.model().solve(data, span)                 # the estimated equations and the identity, as a model
```

Identities are not estimated but make their left-hand sides endogenous; the
instruments are the constant and every regressor without a current endogenous
variable, plus `instruments=`. Linear-form coefficients are `equation.coefficient`;
explicit-form parameters shared between equations, and `restrictions="c.y = i.y"`,
impose cross-equation restrictions. **Estimate → System of equations…** in the app.

## Forward-looking models

```python
model = Model([Equation("y ~ y[+1] - s*(i - pi[+1]) + ey", params=["s"]),
               Equation("pi ~ b*pi[+1] + k*y + epi", params=["b", "k"]),
               Equation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", params=["r", "phi"])],
              params=dict(s=1, b=0.99, k=0.1, r=0.8, phi=1.5))   # calibrated
model.steady_state()
re = model.linearize(shocks=["ey", "epi", "ei"]).solve()       # x(t) = P x(t-1) + Q e(t)
re.rules(); re.irf(20).plot(); re.simulate(200, {"ei": 0.25})
re.filter(data, ["pi", "i"], {"ey": 0.5, "ei": 0.2})          # unobserved variables by the Kalman smoother
model.solve(data, span, solver=PerfectForesight(terminal="steady_state"))   # anticipated, non-linear paths
```

Model equations are compiled to NumPy the first time a model is solved, and the code is reused for new parameter
values, so a 270-equation annual model simulates 60 years of perfect foresight in about a tenth of a second.
Equations with group-wise functions fall back to the interpreter, which gives the same results.

First-order solutions use Dynare's reduced form: variables that appear only in the current period are taken out
before the generalised Schur (QZ) step, which then works only on the lagged and led variables. The likelihood
filters only the states that the observed series depend on. On a 300-equation model, one trial value of an
estimation (steady state, linearise, solve, filter) takes about 0.3–0.6 s, depending on how many
variables are identities (on an Intel Core Ultra 7 256V laptop). Metropolis-Hastings on the New Keynesian template takes about 9 s for two chains of
1000 draws, each after 1000 of burn-in.
`python benchmarks/model_speed.py` times each step by model size.

At these sizes the linear-algebra library's threads cost more than they save, and on a busy laptop far
more (on the machine above, a 300 x 300 solve took 2-6 ms on one thread and 40-190 ms on all eight). So
estimation, Metropolis-Hastings, identification and sweeps run with one BLAS thread, and restore the
library's setting afterwards; `pymetrix.set_blas_threads(4)` changes that, and `set_blas_threads(None)`
leaves the library alone. Parallel chains get their speed from processes, one thread each. On a Mac,
whose NumPy uses Apple's Accelerate, one thread means one thread there too (macOS 15 or later); Accelerate
takes no thread count, so a limit above one leaves it to decide.

Perfect foresight takes surprises too: `Scenario(unanticipated=[...])` with
`solver=PerfectForesight()` solves the model again each time a surprise arrives, from the
path so far, with what is known by then (expectation errors, as in Dynare). On a linear model
this equals the first-order plan; with a lower bound, a surprise can make it bind. Models
without leads have no expectations, so there it changes nothing.

Models may use `max` and `min`, e.g. an effective lower bound on the policy rate,
`i ~ max(i_rule, floor)`. Perfect foresight solves them exactly, the bound binding
where it should; a first-order solution sees only the side that holds at the
steady state. Derivatives use `step`, so Newton's method works on either side of
a kink.

**Second order.** `solve(order=2)` on a template or a Dynare or IRIS model gives the
second-order approximation of Schmitt-Grohé and Uribe (Dynare's `stoch_simul(order=2)`),
with the model's shock sizes: risk now matters.

```python
s = templates.rbc().solve(order=2)
s.ghxx, s.ghxu, s.ghuu, s.ghs2   # the second-order terms, in Dynare's names
s.mean()                         # the ergodic mean: above the steady state for capital (precaution)
s.irf(20, shock_sd={"e": -0.05}) # responses from the stochastic steady state: size and sign matter
s.simulate(200, seed=1)          # pruned, so it stays stable
s.rules()
```

In the app: **Model → Solution → Second-order solution…**.

**Third order.** `solve(order=3)` adds the cubic terms and the corrections for risk to the responses to states and
shocks (Dynare's `stoch_simul(order=3)`; ghxxx, ghxxu, ghxuu, ghuuu, ghxss, ghuss).

```python
s = templates.rbc().solve(order=3)
s.simulate(200, seed=1)                        # pruned (Andreasen et al.); pruning=False for the plain rule
s.irf(20, replications=500)                    # generalised responses, averaged over the other shocks
s.irf(20, state={"k": 40})                     # from a given state rather than the stochastic steady state
s.moments()                                    # mean, standard deviation, skewness, kurtosis: a long simulation
```

In the app: **Model → Solution → Second-order solution…**, Order 3.

**The particle filter, and estimating second-order solutions.** A second-order solution is
not linear, so the Kalman filter cannot give its likelihood. The particle filter can: particles
follow the pruned decision rules with drawn shocks (exactly as `simulate`), are weighted by how
likely each makes the period's data under the measurement equations (evaluated exactly, with
transforms, trends and lags) and their measurement errors, and are resampled (systematic) when
the effective sample size falls below half of them.

```python
s = rbc.solve(order=2)
pf = s.particle_filter(data, measurement, particles=5000, seed=1)   # every observed series needs an error
pf.loglike; pf.filtered; pf.quantiles; pf.ess                    # likelihood, filtered means, bands, diagnostics
s.particle_filter(data, measurement, proposal="guided")          # shocks drawn given the data: less noise
fit = model.estimate(data, measurement, estimate=["rho", "sd_e"], order=2, particles=1000, seed=0)
fit.particle; fit.nonlinear                                      # the filter and the solution at the estimates
fit.smoothed                                                     # the particle smoother's means (all the data)
pf = s.particle_filter(data, measurement, smooth_paths=200)      # the smoother too: pf.smoothed, pf.smoothed_shocks
post = fit.mcmc(draws=5000)                                      # particle marginal Metropolis-Hastings
```

The likelihood is an estimate. Each evaluation uses the same random numbers (from `seed`), so it
moves smoothly enough with the parameters for a derivative-free mode search (Nelder-Mead on
transformed parameters, restarted), but it stays noisy: check its spread over seeds
(`pymetrix.models.particle.loglike_spread`) and raise `particles` until it is well below one.
Standard errors come from a Hessian taken over about one standard error of each parameter.
`mcmc` then draws fresh random numbers at every proposal and keeps the current draw's estimate
(pseudo-marginal), so its draws are from the exact posterior; more noise means slower mixing.
`filter="particle"` at `order=1` gives the same filter on the first-order solution, a check
against the Kalman filter. In the app: **Model → Estimation → Estimate model…**, Advanced, Particle filter.

The particle smoother uses all of the data for each period, as the Kalman smoother does. A pruned
second-order model moves deterministically once its shocks are known, so it works in the shocks
(backward simulation, Godsill, Doucet and West 2004): from a particle drawn at the end of the
sample, it draws the particle of the period before in proportion to its filter weight times the
likelihood of the later data when the model runs on from that particle with the path's shocks,
and so on back to the start. Each path is a start and its shocks, re-simulated, so it is one the
model can produce; `smoothed` holds the means over `smooth_paths` paths (100 for an estimate,
Smooth paths in the dialog; 0 keeps the filtered means), `smoothed_quantiles` their quantiles and
`smoothed_shocks` the shocks' means. Each period's particle is chosen among `candidates` (200)
drawn from the filter's; the cost grows as the sample's length squared, a few seconds for a
hundred periods.

**Optimal policy.** In place of the model's policy rule, the instrument can be set to
minimise a quadratic loss, the discounted sum of weighted squares:

```python
nk = templates.new_keynesian()
com = optimal_policy(nk, "i", {"pi": 1.0, "y": 0.25}, kind="commitment")   # timeless perspective
dis = optimal_policy(nk, "i", {"pi": 1.0, "y": 0.25}, kind="discretion")
com.loss(), dis.loss()          # weighted unconditional variances: commitment does better
com.irf(20).plot(); com.summary(); com.solution.rules()
best = osr(nk, ["phi_pi", "phi_y"], {"pi": 1.0, "y": 0.25}, bounds={"phi_pi": (1.01, 10), "phi_y": (0, 5)})
best.values; best.template.solve()                                          # the best simple rule
```

The equation that determines the instrument is dropped (`rule=` names it otherwise).
Commitment solves the constraints with the first-order conditions in Lagrange multipliers,
which carry the history of policy; discretion is the Markov-perfect equilibrium (Dennis
2007); `osr` chooses a rule's coefficients, as Dynare's `osr`. In the app: **Model → Policy → Optimal
policy…** and **Model → Policy → Optimal simple rule…**.

**Occasionally binding constraints (OccBin).** `occbin` gives the piecewise-linear
solution of Guerrieri and Iacoviello (2015), as Dynare's OccBin does, using the model's
own `max` and `min` as the constraints:

```python
nk = templates.new_keynesian(lower_bound=0.0)          # i ~ max(rule, floor): the rate at least 0%
path = occbin(nk, {"ey": {1: -4.0, 2: -4.0}}, periods=40)   # shocks by period, counted from 1
path.summary()             # when each constraint binds
path.frame; path.binding   # levels with the constraints; True where they bind
path.plot(["y", "pi", "i"])  # with the unconstrained path, dashed
occbin(nk, shocks, anticipated=True)   # the whole shock path known from period 1
```

Each constraint gives two regimes, linearised around the steady state where it is
slack; the periods in which it binds are found by iterating on a guess, checking the
`max` or `min` in levels along the path. Shocks are surprises period by period (Dynare's
default), or anticipated. On a model that is linear apart from its `max`, the result is
the exact perfect-foresight solution. In the app: **Model → Solution → Occasionally binding
constraints…**.

**The OccBin filter and estimation.** With data, the OccBin filter estimates the model's
variables and shocks and the periods at the bound, and gives the likelihood of the data:

```python
result = occbin_filter(nk, data)             # the template's measurement equations and shock sizes
result.smoothed; result.filtered             # levels (the smoothed ones with the shocks)
result.binding                               # the regimes used: True where each constraint binds
result.ex_post                               # the constraint checked on the smoothed path
result.loglike; result.summary()
fit = nk.estimate(data, ["kappa", "rho"], occbin=True)   # maximum likelihood or posterior mode with it
fit.mcmc(draws=2000)                                      # Metropolis-Hastings on the same likelihood
fit.forecast(span, Scenario(set={"ey": {"2025Q1": -3.0}}))  # forecasts that respect the floor (OccBin)
fit.forecast(span, draws=500)                              # draws, each solved by OccBin: the fan stops at the floor
fit.forecast(span, occbin=False)                           # the linear forecast instead
fit.forecast(span, Scenario(hold={"y": {"2025Q1": -2.0}}, free=["ey"]))  # a held target, with the floor respected
post = fit.mcmc(draws=2000); post.forecast(span, draws=200)   # each posterior draw filtered and forecast by OccBin
result = occbin_filter(nk, data, smoother="iterated")    # regimes worked out again on the smoothed path
result.regimes                                           # the regimes the smoother settled on
result.decompose()                                       # shock decomposition with the regimes held at the smoothed ones
```

As Dynare 6's piecewise Kalman filter: each period uses the decision rule of the regimes
expected from the filtered state, solved again with the shock the update estimates, until
the regimes settle; near the bound, where the guesses can cycle, the regime under which the
period's data are likelier is taken. A rate at its floor observed without measurement error is
allowed: the likelihood is that of the other series. Forecasts from a model estimated this way
run OccBin from the smoothed state at the end of the data (any fitted model, with `occbin=True`):
the scenario's shock paths are anticipated unless all are named unanticipated, and draws are
surprises in each period. Targets held by free shocks are solved for by Newton's method through
the piecewise solution, with the shocks anticipated and without draws; a target a shock cannot reach
while a constraint binds (a rate below its floor, held with the policy shock) is refused. Posterior
forecasts filter and forecast with OccBin at each draw, by default when the model was estimated with
the OccBin filter (`occbin=` chooses).

The default smoother keeps the regimes the filter chose with data up to each period, and its
smoothed path can contradict them (a rate smoothed below its floor). `smoother="iterated"` works
the regimes out again from the smoothed state and shock of each period, and smooths again with
them, until they settle, as Dynare 6's OccBin smoother: the smoothed path then satisfies the
constraints period by period. A shock a binding regime mutes (the policy shock at the floor) is
not seen in the data; it takes the value nearest zero consistent with the regime. The likelihood
is the filter's either way. `decompose()` splits the smoothed path into the shocks' contributions
through the smoother's rules, with the regimes held at the smoothed ones; the constants the binding
regimes carry (the floor itself) are a part of their own, "binding constraints", so the parts add
up to the path. A model's response to a shock depends on the regimes, so this is the decomposition
given them, not one of a linear model. Moments still use the linear solution. In the app: **Model → Solution → OccBin filter…**, and OccBin under
Advanced in **Model → Estimation → Estimate model…** and **Model → Analysis → Forecast from estimated model…**.

Correlated shocks and known-date regimes work with OccBin too:

```python
occbin_filter(nk, data, shock_corr={("ey", "ei"): 0.3})        # the filter's Q in full
nk.estimate(data, ["rho", "corr_ey_ei"], shock_corr={("ey", "ei"): 0.3}, occbin=True)
fit.forecast(span, draws=500)                                   # draws correlated as estimated, each solved by OccBin
occbin(nk, shocks, regimes=[Regime(5, {"phi_pi": 2.0})])        # a new rule from period 5, with the floor
occbin_filter(nk, data, regimes=[Regime("2009Q1", {"phi_pi": 1.2})])
nk.estimate(data, ["phi_pi@2009Q1"], regimes="2009Q1: phi_pi=1.2", occbin=True)
fit.forecast(span, regimes="2025Q1..2025Q4: rho=1, phi_pi=0", occbin=True)
occbin(nk, shocks, regimes=[Regime(5, equations={"policy": "i = rule"}, end=8)])  # the floor switched off, periods 5-8
occbin(nk, shocks, regimes="5 surprise: phi_pi=2")              # unknown until period 5: solved again from there
```

The filter takes the shocks' covariance in full, and the smoothed shocks are the likeliest given the smoothed states
(least squares without correlations, as before). Decompositions with correlated shocks are of the shocks
orthogonalised by the Cholesky factor of their covariance in the model's order, as impulse responses are
(`decompose(orthogonal=False)` for the shocks' own contributions). With regimes, each period's matrices come from both
its known regime and the constraint's regime, and each period's `max` is checked with its regime's parameters (a floor
that changes is a regime too). A regime's equations may rewrite a `max` or `min`, leave one out (switched off for a
while: it never binds then) or bring one in: OccBin works over the constraints of all the regimes, matched by the
equation each is in, and a constraint a period's regime lacks is slack in it. A surprise regime (`expected=False`,
or `surprise` after its start in text) is unknown until it starts: the path is solved again from the state it finds,
with the regime believed to hold for good, as perfect foresight does. `occbin` numbers its
periods from 1, so its regimes start at period numbers (`Regime(5, ...)`, or `"5..8: rho=1"`).

**Regimes with known dates.** Parameters that change at known dates, and alternative policy
rules, as in DSGE.jl's regime-switching parameters and temporary alternative policies:

```python
from pymetrix import Regime, regime_filter

post = Regime("2009Q1", {"phi_pi": 1.2}, name="post")            # a new policy response, for good
peg = Regime("2025Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2026Q4")  # a peg for two years; then the rule before it
alt = Regime("2027Q1", equations={"policy": "i = 0.9*i[-1] + 0.2*pi + ei"})   # another rule, by equation name
nk.solve().solve(None, ("2025Q1", "2030Q4"), regimes=[peg, alt])   # the path, the regimes known in advance
fit = nk.estimate(data, regimes=[post])                            # phi_pi@post estimated with phi_pi
fit.forecast(("2025Q1", "2027Q4"), regimes="2025Q1..2025Q4: rho=1, phi_pi=0")   # text works too
regime_filter(fit, data).regime                                    # the regime of each period
Regime("2026Q1", {"phi_pi": 2.0}, expected=False)                  # a surprise: believed permanent when it comes
```

Regimes go in order of their starts. Each changes the parameters and equations it names and
keeps the rest as they were before it; the base model holds before the first. Agents know the
whole path of regimes, so a peg announced for two years moves inflation now. The decision
rules are found backwards from the last regime, which holds for good, and every regime is
linearised at the steady state of the one in force before the first period, with the constant its
equations leave there: a regime with a new inflation target takes the path to its own steady
state, and a peg for a while needs no stable solution of its own. With nothing changing in the
periods solved, the result is the ordinary solution. In estimation each regime's values are
estimated as `<parameter>@<regime>` (`phi_pi@2009Q1` when the regime has no name), with the
parameter's bounds and prior unless `specs=`, `priors=` or `bounds=` name them; regimes
filtered or estimated start inside the sample, after its first period. In the app: Regimes
under Advanced in **Model → Estimation → Estimate model…**, **Model → Analysis → Forecast from estimated model…** (an
estimated model or its posterior) and **Model → Solution → Solve model…** (first-order solver or perfect foresight),
written as `2009Q1: phi_pi=1.2; 2025Q1..2026Q4: rho=1, phi_pi=0` (`2026Q1 surprise: phi_pi=2` for a surprise).

```python
model.solve(data, span, solver=PerfectForesight(), regimes=[post, peg])  # the nonlinear model, its equations switching
post_draws = fit.mcmc(draws=5000); post_draws.forecast(span, regimes="2025Q1..2025Q4: rho=1, phi_pi=0")
fit.decompose()                       # through the regimes' rules; their constants a part of their own, "regimes"
solution.decompose(path, regimes=[post, peg])   # any path, through the rules of the regimes over it
fit.moments(regime="post"); fit.fevd(20, regime="post")   # one regime held for good, at its own steady state
```

Perfect foresight with regimes solves the nonlinear model itself: each period's equations are those of the regime in
force then, known in advance, and the terminal steady state is the last regime's. A surprise regime is solved as an
unanticipated shock is: the path believed before it is kept until it starts, and the model is solved again from the
state reached, the surprise believed to hold for good (a temporary one's end is another surprise); on a model whose
equations are linear this is the first-order solution with the same surprise. A posterior's forecasts solve
each draw with that draw's values of the regimes' parameters (`phi_pi@post`), filtered and forecast through their
rules. Decompositions go through the time-varying rules; the regimes' constants (a new inflation target, a peg away
from the rule) are a part of their own, so the parts add up to the path, and with nothing changing it is the linear
decomposition. Moments and variance decompositions are of one regime at a time, held for good at its own steady state
(default: the base model). The identification check of a model estimated with regimes differentiates the moments of
every period of the sample, which change with the rules: from the reference regime's stationary distribution before
the sample, each period's mean, variance and autocovariances follow from the time-varying rules, stacked over the
sample (divided by the square root of its length, so moments that never change weigh as one period's). A regime's
parameter is identified only if its regime is in the sample (or foreseen in it); with nothing changing in the sample
the check is the one without regimes. Before any data, `Model.identification(..., regimes=, span=)` and
`Template.identification(params, span=)` (the template's own regimes) check the same over a sample's dates.

```python
fit.identification()                                                     # through the fit's regimes
nk.identification(["phi_pi", "phi_pi@post"], regimes=[post], span=("1990Q1", "2019Q4"))
```

The rational-expectations solution uses the QZ decomposition and checks the
Blanchard-Kahn conditions, explaining failures (e.g. a Taylor rule with a
coefficient on inflation below one is indeterminate). Leads and lags beyond one
become auxiliary variables. Perfect foresight solves every period at once, with
the last value, the data or the steady state beyond the span. **Model →
Solution → Rational expectations…** and **Model → Solution → Solve model…** (with Params for calibrated values and the
Perfect foresight solver) in the app. Rational expectations also takes a template model (Model →
Template model…) or a Dynare model as it is, with its own shocks and parameters (`re = nk.solve()`),
and picks the newest one when the dialog opens. **Model → Analysis → Impulse responses…** traces a solution's
responses to one shock: choose the solution, the shock (its list follows the solution), its size (one unit by
default; the shock's standard deviation for a one-standard-deviation shock) and the horizon. It records
`irf = re.irf(periods=20, shock_sd={'ei': 0.25}, shocks=['ei'])`; `shocks=` keeps only those shocks' responses,
each as it is among all of them (correlated shocks orthogonalised over all).

### Writing a model as text: the model editor

```python
src = ModelSource("""
parameters
    beta = 0.99        "discount factor"
    kappa = 0.1
    sigma = 1
    rho = 0.8
shocks
    ey = 0.5           "demand"
    epi = 0.3
    ei = 0.2
equations
    IS: y = y[+1] - (1/sigma)*(i - pi[+1]) + ey
    Phillips: pi = beta*pi[+1] + kappa*y + epi
    policy: i = rho*i[-1] + (1 - rho)*(1.5*pi + 0.5*y) + ei
measurement
    infl = 4*pi  error 0.1  "inflation, annual"
""")
src.check()                     # problems at their line and column, and the symbols: nothing is solved
nk = src.build()                # a Template: the model, shocks and sizes, measurement equations
ModelSource(mod_text, syntax="dynare").build()   # a .mod file's text: a DynareModel, as read_mod gives
```

A model can be written in pymetrix's syntax or in Dynare's. In pymetrix's, each section (`parameters`,
`shocks`, `exogenous`, `variables`, `equations`, `measurement`) is a keyword on its own line followed by
one entry a line; a description in double quotes may end an entry and `#` starts a comment. Each
equation determines the variable on its left, `x[-1]` is a lag and `x[+1]` a lead, and an equation may
run over several lines while a bracket is open. Parameter values may use the parameters above them;
a shock's value is its standard deviation, an exogenous variable's its steady-state value.

**Model → Model editor…** in the app checks the text as it is typed: each problem is underlined where it
is (red for errors, amber for warnings) and listed beside the text with its line and column, and a
click goes there. An unknown name comes with the nearest declared one. The Symbols list shows each name's
role, value, description and the lines that use it. Apply builds the model into the session as a step
that keeps the text as written, so a saved project keeps your comments and layout and builds the model
from the text when it opens.

Typing two characters of a word (or Ctrl+Space) offers the names the text declares, the equations' names, the
functions and the keywords of the syntax, never inside a comment or a description; typing a function's bracket
shows its signature. Hovering a name shows its role, value, description, estimation settings and the lines that use
it; hovering a function, its signature and what it does; hovering an underlined problem, its message. The same help
is there in code: `src.completions(line, column)` and `src.hover(line, column)`.

#### Correlated shocks and regimes in the text

```python
src = ModelSource("""
...
shocks
    ey = 0.5           "demand"
    ei = 0.2
    corr ey, ei = 0.3  prior uniform(-0.9, 0.9)     # estimated as corr_ey_ei
...
regimes
    2009Q1 post: phi_pi = 1.2                  "post-crisis rule"
    2025Q1..2026Q4 peg: rho = 1, phi_pi = 0    # for a while; the rule before it resumes after 2026Q4
    2030Q1 alt: policy: i = 0.9*i[-1] + 0.2*pi + ei
    2032Q1 hawk surprise: phi_pi = 2           # not known until it starts
""")
nk = src.build()
nk.shock_corr, nk.regimes          # {("ey", "ei"): 0.3}, the four Regime objects
nk.solve().solve(data, ("2025Q1", "2035Q4"))   # the path through the regimes
fit = nk.estimate(data)            # regimes inside the sample estimated with (phi_pi@post); later ones for forecasts
fit.forecast(("2025Q1", "2035Q4"))
```

A correlation is a line `corr a, b = r` in `shocks`, with estimation settings after it as for a shock's size; all of
them together must make a valid correlation matrix, and the editor says on which line they stop doing so. A regime
is a line in `regimes`: its start (or `start..end` for a temporary one), optionally a name and `surprise`, a colon,
then parameter values and equations replaced by name (`label: lhs = rhs`), separated by `;`. The editor checks the
periods, their order and overlaps, and every name, where they are typed. Writing the model in pymetrix's syntax
writes both back; IRIS's syntax has the correlations as `corr_ey__ei` parameters, Dynare's as `corr` in the shocks
block, and neither has a place for regimes (the translation's notes say so).

#### IRIS's syntax, and writing a model in another syntax

```python
ModelSource(iris_text, syntax="iris").build()   # a .model file's text: a Template, as read_iris gives
out = src.export("dynare")      # a Translation: the text, its notes, and whether it solves the same
print(out.text)
out.notes                       # what did not carry across: comments, loops, measurement as observed variables...
out.checked, out.difference     # the written text built again and compared: responses and steady state
export_model(templates.new_keynesian(), "iris")   # any template, Dynare model or Model with named parameters
```

The editor reads IRIS's syntax too: blocks such as `!transition_variables` and `!transition_equations`, a
description in quotes before a name, `!!` steady-state parts. **Write as** in the editor writes the model in
pymetrix's syntax, Dynare's, IRIS's or EViews's (below) and opens it in a second editor, with the notes beside it. The text
is written from the model, so comments do not carry across, `!for` loops are written out, Dynare's
estimation settings and commands are left out, and in Dynare measurement equations become equations for
observed variables with their errors as shocks. Reported series are written to IRIS as `!reporting_equations`
(Dynare has no equivalent). Each written text is built and solved again and compared
with the original; a difference larger than 1e-8 is reported in the notes. Equations with comparisons or
`step()` have no equivalent in Dynare or IRIS and are refused, as are the distribution and moving-window functions
other than `normcdf`, `normpdf` and `erf`; estimated equations are written once their
coefficients are named parameters.

#### EViews's syntax: model text and programs

Task page: [Bring an EViews model across, then check and solve it](../src/pymetrix/howto/bring-in-an-eviews-model.md).


```python
src = ModelSource(eviews_text, syntax="eviews")    # model text or a .prg: checked as typed, built into a Template
read_eviews("macro.prg", data=data)                # its "equation eq.ls" specifications fitted to the data first
export_model(templates.gap_model(), "eviews")      # a program: model m, m.append ..., the coefficients' values
read_eviews_xml("frbus_package/mods/model.xml")    # the XML model format of the Fed's FRB/US package
read_eviews_sspace(sspace_text)                    # an EViews state-space object as a StateSpace, where it maps
```

The editor reads an EViews model as its Text view shows it, or as a program builds it:

```
' a small macro model
model macro
macro.append cons = c(1) + c(2)*inc + c(3)*cons(-1)
macro.append dlog(inv) = c(4) + c(5)*dlog(gdp(-1))
macro.append @identity gdp = cons + inv + gov
macro.append @identity inc = gdp - tax
macro.append @add cons cons_a
macro.append @innov inv 0.02
c(1) = 10
c(2) = 0.6
c(3) = 0.2
param c(4) 0.01 c(5) 0.3
```

Names are read in lower case, as EViews treats them. Lags are `x(-1)`, leads `x(+1)` or `x(1)`; `'` starts a comment
and a line ending in ` _` continues. The variable an equation determines is the first series of the current period
on its left (`dlog(inv) = ...` determines `inv`; `y - y_aerr = ...` determines `y`, the add factor moved to the
right). Coefficients `c(1)` and those of vectors declared with `coef(n) beta` become parameters `c_1` and `beta_1`,
with values from `c(1) = ...`, `param`, `scalar`, or `read_eviews(params=...)`; one without a value is an error at
its place. `@identity` marks an identity (no coefficients, no add factor); `@add y y_a` adds the series `y_a` to the
equation for `y` (EViews's intercept shift; `@add(v)` only where `y` is alone on the left); `@innov y sd` gives it a
shock `y_innov` of that size, the template's shocks. In a program, `equation eq.ls cons c inc cons(-1)` is a least-
squares specification (c the intercept, its coefficients c(1), c(2) ... in the order listed), merged with `m.merge
eq` or linked as `:eq`; `read_eviews(text, data=...)` fits it on its `smpl` (or `span=`), and without data it uses the
values given to `c`. Constant series (`series g = 20`) are exogenous values or starting guesses. The functions read
are `log`, `exp`, `abs`, `sqr` (the root), `@sqrt`, `@inv`, `d(x, n, s)`, `dlog(x, n, s)`, `@pch` (a fraction,
`x/x(-1) - 1`) and `@pc`, `@movav`, `@movsum`, `@movmax`, `@movmin`, `@cnorm`/`@dnorm`/`@qnorm` and the t,
chi-squared and F families, `@recode(a > b, a, b)` as `max(a, b)` (or `min`) and any other `@recode` as
`(c)*(a) + (1 - (c))*(b)`, and comparisons in brackets, 1 or 0, as in `(t <= 2004)`. With the data's frequency
(`read_eviews(frequency="Q")`, `wfcreate q`, a dated `smpl`, or the data) `@pchy`, `@pcy`, `@pcha` and `@pca` read
too. `@trend(60:1)`, `@trend`, `@seas(2)` and `@elem(x, "2004q4")` become series (`trend_1960_1`, `trend_start`,
`seas_2`, `x_at_2004q4`) that `eviews_series(model, data)` makes from the data's dates; `@elem(@trend(60:1),
"2004q4")`, the book's way of stopping a trend, is the number itself (179). Refused where they are: `@max` (a sample's
maximum), moving variances and standard deviations, `and`/`or`, `-a^b` and `a^b^c` (written differently in different
languages: give the brackets), EViews's programming language (`for`, `if`, `!n`, `%s`, `{%s}`, `subroutine`,
`include`) and estimation methods other than `.ls`. Workfile, data and display commands (`wfcreate`, `fetch`,
`m.solve`, `.append assign`, series generation) are ignored with a note.

**Write as → EViews** writes a program: the coefficients and scalars, constant series for exogenous values and
guesses, `model m` and an `m.append` line for each equation (identities as `@identity`, a shock that is the last term
added to one equation as its `@innov`, `max`/`min` as `@recode`, `pch(x)` as `100*@pch(x)`). Names EViews would read
differently are renamed (`c` → `c_v`: EViews's coefficient vector), and the notes say what does not carry across:
measurement equations, priors and bounds, regimes, reported series, descriptions, the sizes of other shocks.
`export_model(..., "eviews", form="text")` writes the model's text alone. The subset was confirmed against
J.-L. Brillet's *Structural Econometric Modelling: Methodology and Tools with Applications under EViews* (its model
listings read: the very basic model of chapter 5, the estimated small model of section 6.3.7 and the larger model of
chapter 11, which needs `frequency="Q"` for its `@pchy`) and the Federal Reserve's FRB/US packages; EViews itself
was not run. What neither settles (moving variances, `@add(v)` inside a function, the programming language) stays
refused.

**FRB/US.** `read_eviews_xml` reads the XML model format of the Federal Reserve Board's public FRB/US package
(`mods/model.xml`, as its own `read_xml_model` add-in reads it): each variable's EViews equation through the reader
above, its coefficients (`y_ecd(1)` as `y_ecd_1`, the names of the Fed's Python version), definitions as labels, kinds
and sectors; `mce=True` (or a list of groups) takes the model-consistent-expectations versions. The standard
(VAR-expectations) model has no leads. With tracking add factors from the package's LONGBASE
(`Scenario(adjust=model.add_factors(data, span))`, as EViews's `addinit`), `Model.solve` reproduces the baseline
projection by Newton or Gauss-Seidel, and a scenario's shocks read as deviations from it. The model-consistent version
is solved by `PerfectForesight` over 240 quarters at once. [The FRB/US tutorial](tutorials/frbus.md) runs the Fed's
example programs this way: a funds-rate shock, a federal-purchases shock, a policy-rule switch, and model-consistent
against VAR expectations. The format, equations and data are the Federal Reserve's; the packages are not part of
pymetrix. `read_eviews_sspace` reads an
EViews state-space object (`@signal`, `@state`, `@ename`, `@evar var(e) = beta(3)^2`, `[var = ...]`) as a
`StateSpace`, data in a signal moved to the observed side; regressions in signals (data with estimated coefficients),
covariances of errors and variances of other forms are refused where they are.

In the app: **Model → Import → FRB/US XML file…** and **Model → Import → EViews state-space object…**;
**Model → Solution → Add factors…** computes the tracking add factors over a span and **Model → Solution → Solve model…** takes them under
Scenario (advanced, Add factors), its Equations box taking the model read; **Model → Policy → Optimal control…**
runs `optimal_control` (instruments, loss weights, desired values, the control span, add factors for the baseline).
**Model → Export model…** writes a template, Dynare model or Model in another syntax, with the notes and the check.

#### The equations in algebraic notation, and the solution check

```python
for eq in src.algebraic():      # name, variable, and the equation: yₜ = yₜ₊₁ − (1 / σ) (iₜ − πₜ₊₁) + eyₜ
    print(eq.name, eq.text)
print(src.latex())              # align blocks, each equation tagged with its name, for a paper or slides
found = src.check_solution()    # the steady state, the roots, and a unique stable solution or not
found.verdict                   # "unique", "no stable solution", "indeterminate" or "no steady state"
found.roots_table()             # the roots' moduli: as many stable ones as predetermined variables
```

The editor's **Equations** tab shows the equations as an economist writes them (a click goes to the
equation's line), and **Copy LaTeX** puts them on the clipboard. **Check solution** finds the steady
state and the roots of the model linearised there and says whether the Blanchard-Kahn conditions hold:
too few stable roots means no stable solution (an explosive model), too many means many solutions (often
a policy rule that responds too weakly to inflation). The roots are listed either way, so you can see
how far a root is from the unit circle. When the solution is unique, the Solution tab also draws the impulse
responses (20 periods, one-standard-deviation shocks, correlated shocks orthogonalised as `LinearSolution.irf`
does): choose the shock, and tick the variables (the first six endogenous ones to start with). A model with regimes
shows its base regime's responses, and says so. In code: `ModelSource(text).check_solution().irf(20)`.

### Trending models: steady states with growth, and calibration

```python
ss = model.steady_state(growth=True)          # a balanced growth path: levels and growth per period
ss.growth                                     # e.g. {"l_y": 0.5, "l_y_bar": 0.5}: log output x100, 2% a year
ss.undetermined                               # levels the model does not pin down (only their growth)
model.steady_state(growth=True, fix={"l_y": 460})
sol = model.linearize(ss).solve()             # unit roots allowed, as trending models need
sol.solve(data, ("2025Q1", "2030Q4"), scenario)     # levels follow the trend; plans work as before
model.solve(data, span, scenario, solver=FirstOrder(growth=True))

cal = model.calibrate({"k/y": 10, "c/y": 0.8}, ["beta", "delta"])   # parameters for steady-state targets
cal.values, cal.model, cal.steady, cal.summary()
```

Gap-trend models write levels as logs times 100, so trends are additive: on a balanced
growth path each variable moves by a constant amount each period. The steady state
solves for levels and growth rates together and checks that the path is balanced;
levels of variables with unit roots are reported as undetermined and can be fixed.
Simulations add the trend back, so results are in levels; every model equation holds
on the simulated path. The state-space form and the Kalman filter work along the trend
(`solution.system(...)`, `filter`, `loglike`; states with unit roots start diffuse), so
trending models can be filtered, tuned and estimated (`model.estimate(..., growth=True)`).
Variables in levels that grow by a factor each period: `model.in_logs("y", "k")` (or all)
writes them as `exp(x)`, so their logs grow by `log(1 + g)` on the balanced growth path, as
IRIS's log variables do; levels the model does not pin down stay at their starting values
(`guess=`) or `fix=`. Perfect foresight on a growing path: `PerfectForesight(terminal="growth")`
continues each variable beyond the last period on the balanced growth path, so the end of a
simulation does not bend.

Calibration chooses parameters so the steady state hits targets: any expression of
the variables (a level, a ratio, a spread), one target for each free parameter, with
or without growth. **Model → Estimation → Calibrate model…** in the app.

### Simulation plans: anticipated and unanticipated shocks, exogenize and endogenize

```python
sol = nk.solve()                                   # any first-order solution: a template's, a fitted model's, re above
sol.solve(None, ("2025Q1", "2034Q4"), Scenario(
    set={"ei": {"2026Q1": 0.25}},                  # announced now, happens in 2026Q1: news moves things early
    hold={"i": {"2025Q1": 0.5, "2025Q2": 0.5}},    # hold the policy rate for two quarters (exogenize i) ...
    free=["ei"],                                   # ... with the policy shock (endogenize ei)
    unanticipated=["ey"],                          # surprises, known only when they happen
))
model.solve(data, span, scenario, solver=FirstOrder())   # the same from a Model: linearised at its steady state
```

Shocks and exogenous paths are **anticipated** unless named in `unanticipated`:
announced in the first simulated period and known from then on, as with perfect
foresight. An unanticipated shock is known only in its own period. Each `target`
is paired, in order, with one `free` shock, which takes the values that put its
variable on the target in the periods given; missing periods are free to move.
An anticipated free shock is an announced policy path; an unanticipated one is
a period-by-period conditioning. Starting values come from `data` in levels (only
variables that appear lagged need them; a fitted model's smoothed variables have
them all), or the steady state when `data` is `None`. Results are a `Solution`
in levels, with the shock paths actually used. `fix=` and `adjust=` need the
nonlinear solvers. In the app: **Model → Solution → Solve model…** with the First order solver and
the Hold, Free and Unanticipated fields; Set, Shift and Hold take `i=0.5` for every
period, `i[2025Q1:2025Q4]=0.5` for a range, or `ey[2025Q1]=-1` for one period.

Checks: unanticipated shocks reproduce impulse responses; announced shocks match
the stacked perfect-foresight solver to machine precision; the shocks that hold a
target, given to the perfect-foresight solver, hit it; unanticipated holds match a
period-by-period closed form.

**Least-norm plans and soft holds.** With more free shocks than holds, or with
`free_sd` or `hold_sd`, a plan is a least-norm conditional forecast (as Waggoner and
Zha's): every free shock is free in each period up to the last held one, and the
plan takes the smallest shocks, in units of `free_sd` (default 1), that meet the holds.
A hold with a standard deviation in `hold_sd` is soft, met more closely the smaller it
is; none, or 0, is exact. First-order solutions only.

```python
sol.solve(None, span, Scenario(hold={"i": {"2025Q1": 0.5, "2025Q2": 0.5}, "y": {"2025Q4": -1.0}},
                               free=["ei", "ey", "epi"], free_sd=nk.shock_sd,      # weighted by the shocks' sizes
                               hold_sd={"y": {"2025Q4": 0.25}}, unanticipated=["ei", "ey", "epi"]))
```

**Ranges.** `hold_lower` and `hold_upper` keep a variable within bounds in the periods
given; the free shocks (least-norm) move only as far as needed, so a range that does not
bind leaves the plan as it was and one that binds puts the variable on its bound. This is
the deterministic counterpart of a VAR's range conditions (which draw from the truncated
distribution). First-order solutions only.

```python
sol.solve(None, span, Scenario(set={"ey": {"2025Q1": -1.0}}, hold_lower={"y": {"2025Q1": -0.2, "2025Q2": -0.2}},
                               free=["ei"]))           # policy cushions the fall in output to 0.2 at most
```

Checks: the shocks equal numpy's weighted minimum-norm and soft solutions on the
responses from separate simulations; square plans give the paired result; a binding range
gives the minimum-norm shocks of its active bound, and a slack one changes nothing.
In the app, Model → Solution → Solve model and Model → Analysis → Forecast from estimated model take `hold_sd`,
`free_sd`, `hold_lower` and `hold_upper` under Advanced.

### Fixing in some periods, missing inputs and convergence by period

```python
sc = Scenario(fix={"c": {"2026Q3": 1.2, "2026Q4": 1.3}}, partial=True)   # c fixed in two quarters, solved after
sol = model.solve(data, ("2026Q3", "2028Q2"), sc)                     # Newton, Gauss-Seidel or PerfectForesight
Scenario(hold={"y": {"2026Q3": 1.5}}, free=["g"], partial=["y"])      # a hold in one quarter: g at its data after
model.missing_inputs(data, ("2026Q3", "2028Q2"), sc)                  # period, equation, input: what the solve lacks
sol.convergence                                                       # by period: iterations, residual, equation, converged
```

A fixed (`fix=`) or held (`hold=`) variable needs a value in every period of the span, unless the scenario says
`partial=True` (every fixed and held variable) or `partial=["c"]` (some). Then in a period with a value (not NaN) the
variable is exogenous at it, its equation dropped (or, held, its free variable solved for); in a period without one,
its equation determines it as usual and a held variable's free partner keeps its data. A fix at the data's values
(`fix={"c": None}`) applies where the data have values. The period-by-period solvers (Newton, Gauss-Seidel, compiled
or interpreted) solve each period's own set of equations, so a fix over two quarters gives what solving those two
quarters fixed and then the rest free gives; a fix covering the whole span gives what it gave without `partial`.
`PerfectForesight` takes partial fixes (in a fixed period the variable's equation in the stacked system is
`x = value`; solved by the interpreted stacked system, slower for a large model) but not partial holds, which it
refuses. A first-order solution holds variables only in the periods their paths give in any case. Optimal control
refuses partial fixes. Model → Solution → Solve model: **Hold only where given** (Advanced settings) applies the holds only where given.

`model.missing_inputs(data, span, scenario)` lists, before solving, each period, equation and input it cannot
evaluate (`x`, `c[-1]`): the exogenous series and the starting values before the span. Endogenous values the solve
determines in the span, and leads past it, are not inputs; the scenario's set paths, fixes (partial ones too), holds
and free variables are applied first. A fixed variable without a value is listed against its equation. `model.check(data,
span)` keeps it as `ModelCheck.missing` (its rows are as before), shown as Missing inputs in the result window.

`Solution.convergence` is a DataFrame by solved period: the solver's `iterations`, the largest absolute `residual`
at the end and the `equation` it belongs to, and `converged`; `converged` and `iterations` on the solution stay the
overall values. Perfect foresight has one row per period with its residuals there and the stacked system's
iterations in each. A period that does not converge stops the solve with a `ModelError` whose `convergence` holds the
periods solved before it and that period, marked `converged=False` (the missing-values error too, its row with no
residual). With draws, the table is the deterministic solution's; the draws that fail are counted in
`simulation.info["failed"]`. A first-order solution has none (`None`). Shown as Convergence in the result window,
and saved with the solution in a project.

### Solver options: convergence test, fallback, line search, Jacobian reuse and blocks

```python
from pymetrix import Fallback, GaussSeidel, Newton

model.solve(data, span, solver=Newton(convergence="step"))          # stop on the scaled step, whatever the units
model.solve(data, span, solver=Newton(line_search=True))            # Armijo backtracking from a far start
model.solve(data, span, solver=Newton(refresh="period"))            # one Jacobian a period, again if it slows
solver = Fallback(GaussSeidel(convergence="step"),                   # a period Gauss-Seidel cannot solve is
                  GaussSeidel(damping=0.5, max_iter=400),            # solved again, from its starting values,
                  Newton(line_search=True, refresh="period"))        # by the next solver
sol = model.solve(data, span, solver=solver, ordering="blocks")      # recursive equations once, iterate on blocks
sol.convergence[["iterations", "test", "solver", "jacobians", "blocks", "recursive"]]
sol.convergence.attrs["blocks"]                                      # the blocks by equation, in solution order
```

Newton and Gauss-Seidel stop by default when the largest absolute residual is below `tol` (`convergence="residual"`),
a test in the equations' units: an equation written a million times larger needs its residual a million times
smaller relative to its terms. `convergence="step"` stops instead when the largest scaled step,
`|x_new - x_old| / (1 + |x_old|)`, is below `tol`, which does not depend on how the equations are scaled (the same
iterations either way); `"both"` asks for both. The step test needs one step past the last that moved, so it can take
one more iteration than the residual's.

`Fallback(a, b, c)` tries `a` on every period and, where it fails, `b` and then `c`, each from the period's starting
values; the periods `a` solves are exactly as `a` alone solves them. `Solution.convergence`'s `solver` column names
the solver that solved each period and `iterations` counts every attempt's; when every solver fails the error
lists each attempt with its worst equation. `Newton(line_search=True)` halves the step (Armijo's sufficient
decrease of the squared residuals) until the residuals fall, where plain Newton would overshoot; near the solution it
takes Newton's full steps. `Newton(refresh="period")` keeps the period's first Jacobian (factorised once) while each
step cuts the largest residual to a quarter or less, and evaluates it again when convergence slows: fewer Jacobian
evaluations (the `jacobians` column) for the same solution.

`ordering="blocks"` splits each period's equations into blocks, the strongly connected components of "this equation
needs that variable in the same period" (with holds, each equation is paired with an unknown by matching first), in
an order where each block needs only those before it. An equation that is a block of its own and linear in its own
variable (identities, fitted equations written for their variable) is solved once, by one evaluation; every other
block is solved by the solver given, alone (a `Fallback` tries its solvers block by block). The solution is the
joint solution; a model whose equations are mostly recursive evaluates far fewer equations. The `residual` column is
then the largest left in the blocks iterated on (the equations solved once hold to rounding), and `blocks` and
`recursive` count the blocks iterated on and the equations solved once. With draws, the blocks give the
deterministic solution; the draws are solved by Newton as before.

These options add the columns `test`, `solver` and `jacobians` (and `blocks`, `recursive`) to
`Solution.convergence`; without them the table, and every number, are as before. In the app: Model → Solution →
Solve model, Advanced settings, Solver settings: **Ordering** and **Fallback** (retry a failed period with damped
Gauss-Seidel, then Newton with a line search) and, under the solver's name, its own settings, shown for the solver
chosen (Newton's Convergence, Line search and Refresh).

Every period-by-period solve also records the seconds each period's solve took, in
`Solution.convergence.attrs["seconds"]` (one a row, by `time.perf_counter` around the solver's call), outside the
table's columns. The task page
[Solve a large backward-looking model](../src/pymetrix/howto/solve-a-large-backward-looking-model.md) says when to use
each option, how they combine, a starting configuration for a model of mostly identities with a few simultaneous blocks, and how to
read the table; `python benchmarks/solve_options.py` times the options on a generated model of 200 equations, or on
your own (`--model`, `--data`, `--span`).

### Stochastic simulation

```python
sol = model.solve(data, ("2025Q1", "2027Q4"), draws=1000, seed=1)   # the fitted equations' errors drawn
sol.quantiles([0.05, 0.5, 0.95]); sol.mean(); sol.sd(); sol.fan(["c", "y"])
model.solve(data, span, draws=1000, errors="bootstrap")             # resampled residuals, a period at a time
model.solve(data, span, draws=1000, coefficients=True)              # parameter uncertainty too
model.solve(data, span, draws=1000, shock_sd={"rule": 0.25})        # an equation's size, or a calibrated one's
nk.solve().solve(None, span, draws=1000, shock_sd=nk.shock_sd)      # a first-order solution: shocks drawn
fit.forecast(("2025Q1", "2027Q4"), draws=1000)                      # an estimated model: its shocks, and observables
```

A backward-looking model draws each fitted equation's error in every period
(normal, with its standard error and the residuals' correlations, or the residuals
resampled), adds it to the equation as an add factor, and solves again: nonlinear
equations are solved exactly in each draw. Fixed variables and targets hold in
every draw. A first-order solution adds the response to shocks drawn in each
period to the deterministic path. The `Solution` is the deterministic one, with
the draws in `sol.simulation`; draws that cannot be solved are dropped and
counted in `sol.simulation.info`. Equations with ARMA errors enter the model with
their AR terms and, for MA terms, their innovations (`e_<equation>`, zero ahead);
with GARCH variance, each draw follows its own variance path. A sample of residuals given (`errors=` a DataFrame,
periods by equation; a Dataset of them, over its active sample; or `add_factors` over a historical span) is resampled
a period at a time, as the Fed's stochsim.prg draws FRB/US's historical errors; `error_scale={"2026Q2": 0.5}` scales
some periods' errors and `workers=4` solves the draws in four processes. In the app: **Model → Solution → Solve model…**, Stochastic
simulation under Advanced settings (Error draws: The data's residuals, with the residuals, their series and periods,
Error scale and Parallel processes), and **Model → Analysis → Forecast from estimated model…**.

### Solving for each country of a panel

```python
model = Model([cons, inv, gdp], over="country")                          # the same equations for each country
sol = model.solve(panel, ("2025Q1", "2027Q4"))                            # a PanelSolution: every country's solution
sol = model.solve(panel, span, entity_params={"UK": {"cons.y": 0.6}},     # a country's own parameters
                  scenario={"FR": Scenario(set={"g": 120})})              # and its own scenario (others: baseline)
sol.solved(); sol.solution("UK"); sol.solution("UK").simulation           # the panel, one country, its draws
```

Each country is solved on its own series. In the app: **Model → Solution → Solve model…**, the entity axis under Advanced.

#### Linking the countries: foreign variables and world aggregates

```python
trade = Matrix([[0, .6, .4], [.5, 0, .5], [.3, .7, 0]], rows=["DE", "FR", "UK"], cols=["DE", "FR", "UK"])
eq = Equation("y ~ a + b*y[-1] + c*ys + d*ys[-1] + e*oils + g", params=["a", "b", "c", "d", "e"])
model = Model([eq], over="country", params={...},
              links=[Link("ys", "y", trade),                                   # each country's foreign output
                     Link("oils", "oil", {"DE": .4, "FR": .35, "UK": .25})])  # one weighted oil price for all
sol = model.solve(panel, ("2025Q1", "2027Q4"), scenario={"FR": Scenario(shift={"g": 5})})
sol.solved().frame.loc["DE", ["y", "ys"]]        # DE's output responds to FR's spending through trade
sol.rounds                                       # rounds until the links settled
```

A link is a variable each country takes from the others: `ys` for country i is the sum over j of
`trade[i, j] * y[j]`. The weights are a matrix with a row for each country (usually trade shares, zero on the
diagonal), or one set of weights for every country, a world aggregate. Before the span the links are worked out
from the data, so `ys[-1]` is there. Over the span the countries are solved jointly: each with its links as given
paths, then the links worked out again from every country's solution, until they settle. This is the solution of
the whole multi-country model at once. A scenario shifts the variables the links are made from, never the links
themselves. In the app: **Model → Solution → Solve model…**, Advanced → Panel → Links, written `ys = trade @ y`, where `trade` is a
matrix in the session.

With `draws=` (and `seed=`), a backward-looking linked model is simulated stochastically: each country's errors are
drawn as when solved alone (`shock_sd` by equation, seed plus the country's position), and each draw is solved
jointly, so a shock in one country reaches its partners within the draw. `sol.solution("DE").simulation` holds the
draws, links included. Coefficient draws, GARCH errors and first-order solutions are not drawn with links.

## Inspecting and checking a model

```python
system = model.compile()        # equations, variables by role, parameters, lead and lag structure
system.parameters               # values with status: estimated (named by equation, e.g. cons.y), calibrated, fixed
system.incidence                # where each variable appears: {"c": (-1, 0), ...}
model.check(data, ("2025Q1", "2026Q4"))                 # before solving
model.check(shocks=["ey", "epi", "ei"])                  # forward-looking: steady state and Blanchard-Kahn too
```

Every way of solving a model (simulation, perfect foresight, rational expectations, Dynare import)
works from the compiled model. The check reports, one row each:
- whether every equation contains the variable it determines;
- the recursive and simultaneous blocks;
- parameters without values or outside their bounds;
- the timing of lags, leads and shocks;
- with data and a period, whether the data supply the starting values and exogenous paths. An
  exogenous path a scenario will set is a warning, not an error.

With data and a period, `check.missing` also lists each period, equation and input the solve would lack
(`model.missing_inputs`; `scenario=` to apply one first), and the result window shows it as Missing inputs.

Parameters can be given as a `ParameterSet` with `Param` metadata (bounds, priors, symbols), and
`fix(...)` marks those to hold when a model is estimated. **Model → Check model…** in the app.

## Measurement, state space and template models

```python
from pymetrix import MeasurementMap, templates

nk = templates.new_keynesian()            # also templates.rbc(), templates.gap_model(); nk.model is an ordinary Model
nk.check(); solution = nk.solve()
m = MeasurementMap(["infl = pi_ss + 4*pi", "rate = r_ss + pi_ss + 4*i", "growth = g + 4*(y - y[-1])"],
                   params={"pi_ss": 2, "r_ss": 1, "g": 2}, errors={"growth": 0.5})
system = solution.system(m, nk.shock_sd)  # or nk.system(): the template's own measurement equations
system.filter(data)                       # the model's variables from the data, in levels, and fitted observables
system.loglike(data)                      # the likelihood
m.apply(solution_paths)                   # model paths in the units of the data
```

Measurement equations link the data to the model in the model's own language. Each has an
observed series on the left (`sources` names the data series if different) and model variables
and their lags on the right, with the map's own parameters. Leads are not allowed: an
observation depends on the present and the past.

For the Kalman filter, measurement equations are linearised at the steady state (exact when they
are linear), and lags in them add lagged copies of the states. The state-space system shows its
measurement, transition and shock matrices and the shock sizes. The likelihood agrees with
statsmodels' Kalman filter.

In the app: **Model → Template model…**, **Model → Analysis → State-space form…** (from a template or a
rational-expectations solution) and **Model → Analysis → Kalman filter…**.

#### From raw data and back, and reported series

```python
m = MeasurementMap(["infl = pi_ss + 4*pi", "rate = r_ss + pi_ss + 4*i", "growth = g + 4*(y - y[-1])"],
                   params={"pi_ss": 2, "r_ss": 1, "g": 2},
                   transforms={"growth": "400*dlog(gdp)", "infl": "400*dlog(cpi)"},   # made from raw levels
                   reported=["gap = 100*y", "real_rate = 4*(i - pi)"],                 # reported, not fitted
                   labels={"gap": "output gap, percent"})
fit = replace(nk, measurement=m).estimate(raw)      # raw: gdp, cpi and rate, as published
fc = fit.forecast(("2025Q1", "2027Q4"))            # growth and infl, gdp and cpi levels, gap and real_rate
fit.smoothed["gap"]                                # the output gap over the history
m.observed(raw); m.reverse(forecast_growth, raw)   # either direction by hand
```

A **transform** says how an observed series is made from raw data, in the same expression language:
`400*dlog(gdp)` for annualised growth, `100*(cpi/cpi[-4] - 1)` for year-on-year inflation,
`400*dlog(gdp/pop)` for per-capita growth (with `sources={"growth": "gdp"}`, the series it is rebuilt
into). The filter and estimation then take the raw data, and forecasts come back as raw series too:
each period beyond the data, the transform is solved for the raw value given its past, so no inverse
needs writing. Every draw of a stochastic forecast is rebuilt the same way, so fan charts of GDP
levels are there.

A **reported** series is a quantity of interest in the model's variables (the output gap, the real
rate, potential output) that is not fitted to data. It is reported with the smoothed history,
forecasts and their draws, with its label and units. In the model text: `growth = g + 4*(y - y[-1])
from 400*dlog(gdp)` in the measurement section, and a `reported` section of `name = expression` entries.

### Estimating a linearised model

```python
from pymetrix.estimation.priors import Beta, Gamma, InvGamma, Normal

fit = nk.estimate(data)                                    # every calibrated parameter and shock size, by maximum likelihood
fit = nk.estimate(data, estimate=["rho", "sd_ei"], span=("1990Q1", None))
fit = nk.estimate(data, estimate=["kappa", "rho"], priors={"kappa": Gamma(0.1, 0.05), "rho": Normal(0.8, 0.1)})  # posterior mode
fit = model.estimate(data, measurement, shocks=["ey", "epi", "ei"])                     # any model, with a measurement map
fit = nk.estimate(data, estimate=["kappa", "rho"], priors=priors, starts=8, seed=1)     # from 8 points: is there another mode?
fit = nk.estimate(data, cov="none")                                                    # the mode only: no Hessian, no standard errors

fit.summary()                      # estimates, standard errors, priors, bounds; log likelihood, AIC, BIC
fit.params, fit.parameters         # the estimates with their covariance; every parameter with its status
fit.shock_sd, fit.solution, fit.system, fit.smoothed, fit.provenance
```

At each trial value the model takes the new parameter values (`model.with_parameters(values)`,
which reuses its compiled equations rather than building it again), its steady state is found,
and it is linearised, solved and filtered. Values with no steady state or no unique stable solution are ruled out. Free by
default: calibrated parameters and shock sizes (`sd_<shock>`). Measurement-error sizes
(`sd_<series>`) are free only if named. Fixed parameters (`ParameterSet.fix`) and coefficients
fitted equation by equation are held. Bounds come from each `Param` (or `bounds=`). Priors
come from `Param.prior` or `priors=`; with any prior, the estimate is the posterior mode.
Standard errors come from the inverse Hessian; a parameter at a bound gets none.

**A model that carries its estimation set-up.** Each parameter, shock size and measurement error can
carry its prior, bounds, starting value, description, symbol and whether it is fixed, so `estimate()`
needs nothing more:

```python
model = Model(eqs, params={...}, roles=..., specs=[
    Param("rho", prior=Beta(0.7, 0.1), bounds=(0, 1), label="policy smoothing", symbol="ρ"),
    Param("kappa", fixed=True),                         # held at its value
    Param("sd_ey", prior=InvGamma1(0.5, 2), start=0.4), # a shock's size
    Param("phi_y", transform="log")])                   # positive: bounds (0, inf)
fit = replace(nk, model=model).estimate(data)            # the free parameters, priors and bounds from the specs
model.spec("rho"); model.parameters.table()             # the metadata, and a table with priors and descriptions
```

The same can be written in a model's text (`rho = 0.8  bounds 0 1  prior beta(0.7, 0.1)  symbol ρ
"policy smoothing"`; on a shock, the settings are for its size) and is read from Dynare files: `$\rho$`
names, `long_name` descriptions and `estimated_params`. Explicit `priors=`, `bounds=` and `estimate=`
still take precedence.

The optimiser is BFGS on transforms of the parameters that keep them within their bounds (a full
quasi-Newton picture of the curvature, like Dynare's csminwel), settled by L-BFGS-B. Medium-sized
models can have more than one mode: Smets and Wouters (2007) has one 0.54 log points below the
best (a persistent risk-premium shock with little habit, against a short-lived one with strong
habit), which a limited-memory optimiser reached from the file's starting values. `starts=n`
optimises from the starting values and from `n - 1` draws from the priors (within the bounds,
where the model solves), keeps the best, and says in the summary how many reached it and where
the others stopped; each start's origin and result are in `fit.provenance.method["starts"]`.
When the first start takes more than a few seconds, the others run in parallel processes
(`workers=` sets how many). In the app:
**Model → Estimation → Estimate model…** (priors can be typed there, e.g. `kappa=Gamma(0.3, 0.1); rho=Beta(0.8, 0.1)`).

### The posterior: Metropolis-Hastings, marginal likelihood, identification

```python
fit = nk.estimate(data, estimate=["kappa", "rho", "sd_ei"],
                  priors={"kappa": Gamma(0.3, 0.1), "rho": Beta(0.8, 0.1), "sd_ei": InvGamma(0.2, 0.1)})
post = fit.mcmc(draws=20000, chains=2, seed=1)     # random-walk Metropolis-Hastings from the mode; chains run in parallel
post.summary()                  # mode, mean, median, HPD intervals, R-hat, effective draws, acceptance
post.marginal_likelihood()      # log marginal data density (modified harmonic mean); post.laplace() at the mode
post.irf(20).plot()             # impulse responses: posterior median and 90% band
post.forecast(("2025Q1", "2027Q4"), draws=500).fan()   # forecasts with parameter and shock uncertainty
post.forecast(("2025Q1", "2027Q4"), draws=500, shocks=False)   # parameter uncertainty alone
post.plot(); post.trace()       # prior and posterior densities; the chains
post["kappa"], post.to_frame()  # the draws
post = fit.smc(particles=2000, seed=1)   # sequential Monte Carlo: from the prior, any number of modes; the same Posterior
post = fit.mcmc(draws=1000, chains=2, seed=1, sampler="nuts")   # the No-U-Turn sampler; burn = its warmup
post.marginal_likelihood()      # from the same run (method="mhm": the harmonic mean of its draws)
fit.identification().summary()  # Iskrev: can the data's moments tell the parameters apart?
nk.identification(["kappa", "rho"])          # the same before any data: at the calibration (or at=)
fit.identification(elasticity=True).pairwise # each parameter's closeness to the one most like it
```

As in Dynare, every estimated parameter needs a prior; priors are given by mean and standard
deviation (`InvGamma1` is Dynare's `inv_gamma_pdf` for standard deviations, `InvGamma`/`InvGamma2`
its `inv_gamma2_pdf`). Each parameter is kept where its prior is positive. The proposal is normal with the inverse Hessian at the mode, its scale tuned during
burn-in (as many draws again as are kept) towards an acceptance rate of about 0.3. Each chain
starts from an overdispersed point near the mode. R-hat near 1 and large effective sample
sizes suggest the chains have converged; compare models by their log marginal data densities.

Sequential Monte Carlo (Herbst and Schorfheide, as Dynare 6's) starts from particles drawn from the
prior and moves them to the posterior through the likelihood raised to rising powers: each step keeps
the weights' effective sample size near `ess_fraction` of the particles (half by default; a higher share
means more, smaller steps; `target=` is its old, deprecated name), resamples when it
falls below half, and moves the particles by `steps` Metropolis-Hastings steps. It needs no mode to
start from and finds posteriors with several modes or long ridges, and its marginal data density
comes from the same run. Its effective sample size is the particles', at the end; there is one
"chain", so no R-hat. **Model → Estimation → Sequential Monte Carlo…** in the app.

The No-U-Turn sampler (`sampler="nuts"`; Hoffman and Gelman, as Stan runs it) moves along the posterior by its
gradient, on the parameters mapped off their bounds (logistic between two, log above one), and a draw takes
2^depth leapfrog steps (the summary reports the mean tree depth). The gradient is analytic: the steady state, the
linearised equations, the solution, the measurement equations, the shocks' covariance and the Kalman filter are
differentiated exactly (`pymetrix.models.kalman_score`; the filter in reverse mode), at the cost of about two
log posteriors on a small model and three to six on Smets and Wouters (2007), however many parameters there are. Where that does not apply (known-date regimes, the OccBin filter, DSGE-VARs, diffuse starts, growth
models), or with `gradient="numerical"`, it is by central differences, 2k + 1 solutions of the model a step for k
parameters. The same gradient drives the posterior mode's BFGS search, and the Hessian behind the standard errors
is the difference of exact gradients (`gradient=` in `estimate` too); `fit.gradient()` shows it at the estimates
or elsewhere (`at={...}`), with `method="numerical"` for comparison. Timings on the New Keynesian template are in
the validation notes. Its burn-in is Stan's warmup (step size by dual averaging towards `target_accept`,
0.8, and the mass matrix in windows); `metric="dense"` (the default) adapts the posterior's whole covariance from the
inverse Hessian at the mode, so linearly correlated parameters cost about what independent ones do, `metric="diag"` only their
variances, and `metric="mode"` keeps the inverse Hessian and tunes the step size alone (then a burn-in of 150 can do;
good when the posterior is near Gaussian, poor when it is not). Divergent transitions are counted and noted, and a point where the
model cannot be solved stops a trajectory. Its chains run at once in other processes as Metropolis-Hastings chains do
(`workers=`, by default when the run is long enough), with the same draws as one after another. **Model →
Estimation → Metropolis-Hastings…** with sampler NUTS in the app.

Which sampler? A gradient costs two to six log posteriors (the filter is differentiated backwards, once for all
parameters), and a NUTS draw takes 2^depth of them, against one log posterior a Metropolis-Hastings draw. With numba
installed (`pip install pymetrix[fast]`) the filter's loops run compiled, which roughly halves both on small models.
Measured (validation notes, one chain): on the New Keynesian template (7 parameters) NUTS with its dense metric gives
2.5 times Metropolis-Hastings's smallest effective draws a second (10.5 against 4.1; 16 with `metric="mode"`), and on
Smets and Wouters (36 parameters) about four times (0.023 against 0.005), with posterior means that agree with
Dynare's long run where an equal-time Metropolis-Hastings chain's do not. Try NUTS first for a model whose gradient
is analytic, with several chains on several cores (`workers=`); keep Metropolis-Hastings where the gradient is by
differences (regimes, OccBin, diffuse starts), and compare `post.summary()`'s effective draws per second of run time
before long runs.

Identification needs no data: `Template.identification` and `Model.identification(measurement, params,
shocks=, shock_sd=)` test at the calibration, as fitting and testing at the same point would. Collinearity
is each parameter's multiple correlation with all the others, near one for most parameters of a large
model; `pairwise` says which single parameter each is most like, and how much (the |cosine| of their
effects, as Dynare reports). `elasticity=True` measures each moment's change relative to its size, so
the largest variances do not make every pair look alike. `pymetrix.models.bayes.moments(system, lags)`
is the moments vector the test differentiates.
Each chain has its own random stream from `seed`, and long runs put each chain in its own process
(`workers=` sets how many at once; `workers=1` runs them here, one after another): the draws are the
same either way. Halfway through burn-in each chain's proposal takes the shape of its own
draws so far (`adapt=False` turns this off): on a long curved ridge, such as Smets-Wouters'
risk-premium shock against habit, the curvature at the mode is a poor guide. The summary
names any parameter whose chains disagree (R-hat above 1.1).

A posterior with several modes needs more than a random walk: one scaled to one mode rarely
crosses to another far away. Estimate with several starts, and the sampler jumps between the
modes they found:

```python
fit = sw.estimate(starts=8, seed=1)              # the starts' ends are in fit.provenance.method["starts"]; about half an hour
post = fit.mcmc(draws=40000, chains=4, seed=1)   # jumps between the distinct modes found
post = fit.mcmc(draws=40000, chains=4, modes=[other])   # or give other modes: {name: value} for every estimated parameter
```

The modes jumped between are the starts' ends within 5 of the best log posterior, highest first;
lower ones hold little of the posterior. On 5% of steps (`jump=`) the sampler proposes moving the
draw by the difference between two of the modes. The reverse move is as likely, so the draws keep each mode's share of the posterior
exactly; chains start around the modes in turn, and the summary reports the jumps accepted.
`modes=False` turns jumps off. Smets and Wouters (2007) is such a posterior: two modes of almost
equal height, with the risk-premium shock's persistence at 0.14 or 0.82. Dynare finds the same
two, and without jumps the chains cross only about once in 10,000 draws. Each draw starts the model's solution from the previous draw's (a few cheap
iterations instead of the generalised Schur form), with Blanchard-Kahn checked on the result and QZ
as the fallback.
Identification uses the observables' means and autocovariances: a parameter that moves no
moment, or moves them only as others do, is flagged. In the app: **Model → Estimation → Metropolis-Hastings…**
and **Model → Analysis → Identification…**.

### Parameter sweeps and scenario lists

```python
s = sweep(nk, {"phi_pi": [1.2, 1.5, 2.0], "kappa": [0.05, 0.1]}, "irf", periods=20)  # every combination
s.table("y", "ei", horizon=4); s.plot("y", "ei"); s[{"phi_pi": 1.5, "kappa": 0.1}]
sweep(nk, {"sd_ei": [0.1, 0.2]}, "moments").table("i")                          # shock sizes too
sweep(nk, {"phi_pi": [1.2, 2.0], "kappa": [0.05, 0.1]}, "steady_state", grid="zip")  # pairwise
c = scenarios(fit, {"baseline": Scenario(), "tight": tight}, ("2025Q1", "2027Q4"))
c.table("y"); c.difference("tight", "baseline"); c.plot(["y", "pi"])
compare_solutions(baseline=sol1, tight=sol2)                                      # already solved
c.impact(["y", "pi"], at="average")      # a scenario per row: its departure from the baseline ("end", "peak", a period)
c.plot(["y", "pi"], deviations=True)     # each scenario less the baseline
c.with_baseline("tight")                 # the same scenarios, compared with another
c = c.with_loss({"pi": 1, "ygap": 0.5}, desired={"pi": 2}, discount=0.99)   # a loss to rank them by
c.losses()                               # each scenario's loss, by variable, and less the baseline's
Report.from_template("policy_brief", {"Monetary policy options": c}).save("brief.docx")
```

A sweep runs a template (or a Dynare or IRIS model) once per set of parameter values and
computes impulse responses, moments, variance decompositions, steady states, solutions or
anything a function returns; a run that fails is listed with its error and the others are
kept. `scenarios` solves an estimated model or a solution under each scenario. Both take
`workers=n` to run in parallel processes, which pays when each run takes a while. In the
app: **Model → Scenarios → Parameter sweep…** and **Model → Scenarios → Compare scenarios…**.

**A scenario set** is the scenarios against one baseline: the scenario named in `baseline=` (Compare scenarios'
Baseline), else one named "baseline", else the first. Its window has the scenarios with the baseline marked, an
**Impact** table (each scenario less the baseline, on average over the periods solved, a scenario per row and a
variable per column), the paths, the paths **Less the baseline**, a table per variable and each scenario less the
baseline period by period. **A loss** ranks the scenarios: `with_loss(loss, desired, discount)` (Compare scenarios'
Loss, Desired values and Discount) sets, as optimal control's, the sum over the periods solved and the variables
weighted of discount^t × weight × (value − desired)², and `losses()` (the window's **Loss** tab) gives each scenario's,
split by variable, and less the baseline's: negative is better than the baseline. The `policy_brief` report template
puts the scenarios, the Impact table, the loss (when one is set) and both graphs in a report.

### Forecasts, decompositions, moments and tunes

```python
fit.forecast(("2025Q1", "2026Q4"))                         # from the Kalman-smoothed state; observed series in data units
fit.forecast(span, Scenario(hold={"i": {"2025Q1": 0.0}}, free=["ei"]))  # with a simulation plan
fit.forecast(span, data=newer, tunes={"y": {"2024Q4": -1.0}})              # judgement imposed in the filter
fit.system.filter(data, tunes={"y_gap": {"2024Q4": -1.0}})                 # the filter with tunes
fit.system.filter(data, tunes={"y_gap": {"2024Q4": (-1.0, 0.25)}})         # a soft tune: value, standard deviation
fit.system.filter(data, shocks=True)                                       # the smoothed shocks as extra columns
fit.decompose().plot(["y", "pi"])                           # each shock's contribution to the filtered history
fit.contributions(horizon=8).plot(["y"])                    # each observed series' contribution, history and forecast
fit.moments().summary(); fit.moments().table(); fit.fevd(20).table("y")   # theoretical moments, variance decompositions
solution.moments(shock_sd); solution.fevd(20, shock_sd); solution.decompose(paths)
solution.decompose(paths, start="presample")               # the first row is x(-1): its shocks count as the shocks'
```

In the app: **Model → Analysis → Shock decomposition…** (`fit.decompose`, with the start and, advanced, OccBin)
and **Model → Analysis → Data contributions…**.

A tune is an exact observation of a model variable in a period: the filter makes
everything else consistent with it. With a standard deviation (or `tune_sd=`) it is
soft: an observation with that error, in its own period only, met more closely the
smaller the standard deviation. `shocks=True` adds the smoothed shocks e(t) of
x(t) = P x(t-1) + Q e(t), from the smoothed state disturbances, so the shocks need not
be carried as states. Where a variable is known only up to a constant
(output observed only through its growth), a tune sets that constant. The shock
decomposition recovers the shocks from the smoothed path and adds up to it exactly,
with a column for the starting point. Moments solve the Lyapunov equation of the
solution, as Dynare's `stoch_simul` reports them; models with unit roots have none
(use `fevd` by horizon).

Weak identification is common: in the New Keynesian template, the slope of the Phillips curve
and the Taylor-rule response to inflation trade off against each other in 100 years of
quarterly data. Fix them, give them priors, or check the standard errors before trusting the
estimates.

## Panel data tools

```python
data.pivot("region", "time", "infl", agg=["mean", "count"], margins=True)   # pandas pivot_table
data.wide("gdp")                       # gdp_AUS, gdp_NZL, ...: time series per country
sheet.long(entity="country", name="gdp")   # a column per country -> a panel
data.entity("WSM"); data.cross_section("2020Q1")
data.coverage("gdp"); data.balance(keep="periods"); data.complete()
data.describe_panel("gdp")             # overall, between and within variation
data.derive(gap="y - mean(y, time)", dev="demean(y, country)", w="share(gdp, time)")
```

Panels are long pandas frames with a MultiIndex (entity, time); `data.frame` gives
the frame itself for any pandas operation. Group-wise functions (`mean`, `sum`, `sd`,
`min`, `max`, `median`, `count`, `demean`, `share`, `rank`) take an axis or a
category series to group by, as `mean(y, country)` or `mean(y, by=country)`, and
work in expressions, derived series and regressors. `max` and `min` group only
with `by=`: `max(a, b)` is the larger of two values, period by period. **Data → Panel → Pivot table…, Panel to wide…, Wide to panel…, One entity or
period…, Balance panel…, Panel statistics…** in the app.

## Selections and loops

A selection is an ordered set of series names, chosen like the column selectors
of R's tidyselect, polars and Julia; templates use `{s}` for the series name.

```python
s = data.series("c", "i", "g")                # or data.series("*_cycle"), data.series(units="USD bn")
data = data.derive(**s.formulas("d{s}", "dlog({s})"))
fits = s.fit("d{s} ~ d{s}[-1] + dlog(y)", data, method=OLS(cov="hac"))
fits.coefficients()                           # side by side; rows line up as d{s}[-1]
adf = s.test(data, "adf", lags=4)             # one table of statistics and p-values
for name in s:                                # or an ordinary Python loop
    show(data[name].describe())
```

Results are a `Results` collection (by series, dict-like) with `coefficients()`,
`stats()`, `tests()` and `failures()` (with `errors="collect"`). The menus have
**Data → Select series**, **Data → Derived series for each selected series**,
**Estimate → Equation for each selected series** and **Test → Test each
selected series**. (`data.group(...)` still works, as an older name.)

## Your own estimators and solvers

NumPy, pandas and SciPy are ready in the Immediate and Code windows as `np`,
`pd` and `sp` (scripts get the import lines when they use them), and arrays and
tables made there open as tables in the app. `pymetrix.toolkit` has the
building blocks pymetrix's own estimators use:

```python
from pymetrix import toolkit

d = toolkit.design("c ~ y + c[-1]", data)        # y, X, names, rows used (lags, fe(), missing values done)
beta, u = toolkit.least_squares(d.y, d.X)
V = toolkit.covariance("hac", d.X, u, d.df_resid)  # classic, white, hac, Cluster(...)
fit = toolkit.result(eq, data, d, method, beta, V, u)   # a FittedEquation with statistics and provenance
```

Also `arrays`, `sandwich`, `hessian`, `gradient`, `fit_stats`, the Kalman filter
and smoother, and the solver interface (`PeriodSystem`). An estimator is a
frozen dataclass of settings with `estimate(eq, data, ...)`; a solver has
`solve_period(system)`. After `toolkit.register("methods", ...)` (or
`"solvers"`) it appears in the dialogs and works in scripts, projects, loops
over selections and `Results` tables. See `examples/custom_estimator.py` (ridge
regression) and `examples/custom_solver.py` (damped Newton);
`pymetrix.testing.check_entry` checks yours.

## Programs

A pymetrix program is an ordinary Python file starting `from pymetrix import *`.
It runs anywhere: the command line, VS Code, Jupyter, or the app's Code window.
In the Code window each top-level statement is a step in the history (a loop
or a function definition is one step), and a run stops at the first failing
statement, which is highlighted. `show(obj)` opens a result's window in the
app, displays it in a notebook, and prints it in a terminal; it stays quiet
while a project is reopened or results are recomputed.

## Workspaces: the app arranged for each stage of the work

The bar under the menus picks a workspace: **All** · **Explore** · **Estimate** · **Model** · **Forecast** ·
**Report** (Ctrl+1 to Ctrl+6, or **View → Workspace**). Each shows the menus, panels and shortcuts of one stage
and hides the rest; the row under it holds that stage's most used commands. Shortcuts of the app itself (Undo,
Redo, Save, Open, Find anything...) work in every workspace, whichever menus it shows.

| Workspace | Menus (with File, Edit, Data, View, Calculate, Help) | Panels | Shortcuts |
|---|---|---|---|
| All | every menu | as you arrange them | none |
| Explore | Test | Objects, History | Load data, Spreadsheet, Plot series, Check data, Select series, Run test, Correlogram, Seasonal adjustment, Filter |
| Estimate | Estimate, Test | Objects, History, Problems | Equation, VAR, Local projections, Run test, Coefficient plot, Forecast |
| Model | Model | Objects, History, Problems | Define equation, Template model, Check, Solve, Estimate model, Parameter sweep, Compare scenarios |
| Forecast | Forecast, Estimate | Objects, History | Make round, Adjust round, Save round, Compare rounds, Conditional forecast, Combine forecasts, Backtest |
| Report | none more | Objects, Report, History | Export report, Plot series, Save result, Export script |

- **A lens, not a gate.** Work rarely runs in a line (a data revision sends you back, forecasting draws on
  estimation, reporting goes on throughout), so switching is instant, the session stays as it is, everything is
  still one search away (**Find anything**, Ctrl+K, reaches hidden menus too) and **All** shows the whole app.
- **Code alongside.** **Code** (Ctrl+\`) adds the Script, Immediate and Problems panels to any workspace.
- **Each workspace keeps its arrangement.** Move or close panels in a workspace and it is that way when you come
  back; **View → Reset layout** sets the current one as it first shows. The workspace you were in, and Code, are
  remembered for the next session.
- **The window in front stands out.** Its title bar is in your theme's accent colour, with a thin frame of the same
  colour, and the other windows' titles are greyed; the panel you are typing in has an accent line along its top.
  The app takes its colours from the system's light or dark theme: errors red, warnings amber, successes green,
  links blue and secondary text grey, each chosen to read clearly (a contrast of at least 4.5:1) on that theme.
- **An icon for each kind of object.** Rows of the Objects panel, the windows' title bars (and their tabs in tabbed
  view) and the objects in **Find anything** carry a small icon of what the object is: a dataset (a cylinder), a
  series (a line on axes), an equation or fitted equation (points and a line), a system or VAR result (three linked
  nodes), a model, its solution or scenarios (a cube), a forecast (a line opening into a fan), a figure (bars), a
  report (a page), a test result (a tick), a table or matrix (a grid), and anything else (a diamond). The shapes
  differ, not only the colours, and the colours follow the light or dark theme. Plugins' objects show the diamond
  unless they derive from one of pymetrix's own kinds.

## Views of a result

A result window lists its views down its left side: a fitted equation's thirteen (Summary, Graph, Coefficients,
Residuals, Diagnostics, Provenance, Fields...) all in view, each name in full, under the headings **Summary**,
**Tables**, **Graphs**, **Diagnostics** and **Provenance** (a window of a few views lists them without headings; a
single view, such as a figure, fills the window). Click a view, or move with Up and Down in the list (Ctrl+PgDown and
Ctrl+PgUp from anywhere in the window). The view you pick is the one the next window of that kind of result opens on,
and a window keeps its view when its result is recalculated. Copy summary, Next ▸ and Add to report sit above the view.

## Next: from a result to the next stage

The stages connect through their results. Each result window (and a dataset's) has **Next ▸** in its corner: the
steps that usually follow, each opening its dialog with the result filled in.

| A window of | Next |
|---|---|
| a dataset | Plot series, Check data, Select series, Seasonal adjustment, Test each series, Estimate an equation |
| a fitted equation | Test, Coefficient plot, Forecast, Backtest, Solve as a model |
| a VAR (or VECM) | Impulse responses, Variance decomposition, Historical decomposition, Test, Forecast, Conditional forecast, Backtest |
| an equation | Check, Solve, Rational expectations |
| a model template, Dynare model | Estimate model, State-space form, Parameter sweep |
| an estimated model | Data contributions, Identification, Metropolis-Hastings |
| a state-space system | Kalman filter |
| a solution | Make a forecast round, Compare scenarios |
| a forecast round | Adjust round, Save round |

In a stage's workspace, choosing a step moves to that step's workspace (a forecast from Estimate goes to
Forecast); in All, the app stays as it is. The steps are recorded as the dialogs record them.

## The report outline: collect as you go

Reports are put together throughout the work, not only at the end. **Add to report** is under every figure and
every table, and in the corner of every result window (its summary and graph); each adds an item to the **report
outline**, from any workspace. The **Report** panel (shown in the Report workspace) lists the outline:

- **Up** and **Down** order it; **Remove** takes an item out; **Heading…** and **Text…** add headings and paragraphs;
  double-click an item to give it a caption or change its text.
- **Export…** opens **Export report** on the outline: Word, HTML, Markdown or Excel, with a title, a theme and the
  appendix of where the numbers come from. The export is the recorded step, and its code writes every item out, so
  the report can be written again without the app:

```python
report = Report('Consumption')
report.heading('Results', 2)
report.add(fit, caption='Consumption fit', sections=['Graph'])
report.text('Consumption follows income.')
report.save('consumption.docx')
```

Putting the outline together is not recorded; the outline is saved with the project. In code:
`session.add_to_report("fit", "Graph")`, `add_report_heading`, `add_report_text`, `move_report_item`,
`remove_report_item`.

## Long steps, the start page and the stage bar's badges

- **Long steps run in the background.** A step started from a dialog runs while the window stays live: other
  windows can be read and moved. After a moment the status bar shows how far it is, what it is doing and
  **Cancel**. MCMC chains, bootstrap bands, backtests, sweeps and rolling windows stop at their next check when
  cancelled, and nothing is recorded; a computation without checks finishes. While a step runs, another cannot be
  started and the Immediate window waits, so the session never changes under it. In code, a long computation reports
  with `pymetrix.progress.report(fraction, message)`.
- **The start page** of an empty session offers example data, a data file, a project, the recent projects and the
  demos, and says how the workspaces and Ctrl+K work. It opens centred in the workspace's free area, never under a
  panel, and keeps fitting it as panels change until you move it; run from a checkout of the repository, its
  **Chart gallery** link opens `examples/gallery/chart_gallery.ipynb`. **Drop a file on the window** to open it: a `.pymx` project
  opens; a data file (CSV, Excel, Parquet, Stata, SPSS, JSON) opens **Load data** with it filled in.
- **Badges on the stage bar** count each stage's results that are out of date or not computed, after a data
  revision for instance: "Estimate · 2". Their names are in the tooltip, All counts them all, and choosing a stage
  with a badge opens the Problems panel, which lists them; **Calculate → Recalculate** brings them up to date.
- **While a step runs**, its dialog waits: OK does nothing more, and Cancel, Esc or closing it stops the step (as the
  status bar's Cancel does). If Cancel stops the automatic recalculation after a step rather than the step itself,
  the step is kept and the dialog says the results after it are out of date. A step with no point at which to stop
  finishes anyway, and the dialog says it was recorded. Dropping several files opens one; the
  status bar says how many were left. Cancel stops model estimation, state-space fits and optimal-rule searches
  within one evaluation of the likelihood or loss, seasonal adjustment between series, and a long Kalman filter.

## Autosave and recovery

While a session has unsaved changes, the app writes a copy of it every 2 minutes to `recovery` in `~/.pymetrix`, as
an ordinary project file; the session itself is not saved, renamed or marked as saved. If pymetrix stops without
closing (a crash, a power cut), the next start offers the copy back: it opens with its original file name, marked
unsaved, until you save it. Saving, closing normally or opening another project deletes the copy. **File →
Autosave for recovery** turns it off and on.

## Help in the dialogs

**Load data** shows the file's first rows and what they suggest: the column of periods and their frequency (years,
quarters like 2001Q1, months, dates and their spacing) and, for a panel or cross section, the column that
identifies the units. The guesses go into the fields that are still empty; a field you set yourself is never
changed. **Estimate → Equation…** looks at the dependent variable and suggests a likelihood that suits it: logit or
probit for 0/1, Poisson or negative binomial for counts, ordered models for a few ranked values, Tobit for a
pile-up at the lowest value. **Use it** sets the method and family; nothing changes without it.

Each field is labelled in words (**Exogenous series**, **Error distribution**, **Replications**); its tooltip, and its
label's, ends with the Python argument it sets (*Python: exog=*), so a script can be written from the dialog. Fixed
choices show their names (**FM-OLS (fully modified)**, **Drop rows with missing values**) while the step records the
code (`method='fmols'`, `missing='drop'`); each choice's tooltip gives it. Settings most steps leave alone sit in one
collapsed section at the bottom of every dialog, **▸ Advanced settings**, with the advanced settings of a chosen
method or test (Method settings, Test settings) under it too. A dialog shows the essentials: the data or object it
acts on, the formula, variables or series, the method, the sample (Start and End) and the name of the result.
Tolerances, iteration limits, optimisers, starting values, seeds and draw counts, rarer covariance variants, display
and precision options and plugins' tuning go in the section. A setting the step cannot run without is never there.
The section opens by itself when a reopened step changed one of its settings or one holds a value that is not valid;
while closed, its header counts the settings changed from their defaults (**▸ Advanced settings (2 changed)**). It
has an access key of its own, a hidden field's access key opens it and goes to the field, and whether you left it
open is remembered for each dialog (it starts closed). Help pages list every field, the advanced ones marked so.

## Terms used in pymetrix

The app uses one word for each idea, in menus, dialogs, results and help. Scripts keep the argument names of the
Python API (`fit`, `span=`, `exog=`): each field's tooltip gives its argument, e.g. *Python: exog=*.

| Term | Meaning | Not |
|---|---|---|
| **series** | A column of data: one variable observed over periods (and, in a panel, over entities). Dialogs say Series for the columns they use. | variable (for a column of data) |
| **variable** | A model's variable: endogenous (explained by the model's equations) or exogenous (given to it). A model's variables are usually filled from series of the same name. | series (for a model's variables) |
| **dataset** | A table of series with its time (and entity) axes, metadata and active sample. Fields that pick one are labelled Data. | data table, frame |
| **sample** | The active window of a dataset: the periods (and entities, or a condition on the rows) that estimation, tests and graphs use unless a step says otherwise. Data → Sample → Set sample… sets it. | span, range |
| **Start and End** | The first and last period a step uses; left empty, the sample's first and last. | first and last, from and to |
| **period** | One date of a time series: 2020Q1, 2024-03. Time series have periods; cross-sections have rows. | row (for a time series), observation date |
| **estimate** | Fit a model to data: the Estimate menu, Model → Estimation → Estimate model…. Code calls it fit. | fit (in the app) |
| **estimated model** | The result of estimating: a fitted equation, VAR, VECM, BVAR... Fields that pick one are labelled Estimated model, Estimated VAR or Estimated equation. | fit, source, result (as labels) |
| **horizon** | How many periods ahead: of a forecast, of impulse responses, of a decomposition. | periods, steps (for steps ahead) |
| **draws** | Simulations from a Bayesian posterior (or a model's random errors). | reps |
| **replications** | Bootstrap or Monte Carlo repetitions, for standard errors, bands and p-values. | reps, draws (for a bootstrap) |
| **seed** | The number that starts the random-number generator: the same seed gives the same draws. |  |
| **step** | One thing done in the session (a dialog run, a command typed), recorded in History, undone by Undo and written to the session's script. | action |
| **command** | A menu item and its dialog: Estimate → Equation… is a command; running it records a step. | dialog (in lists of them) |
| **Advanced settings** | Settings most steps leave at their defaults: in every dialog, a collapsed section at the bottom of the form, opened with ▸ Advanced settings. | Advanced tab |
| **judgement** | A forecaster's adjustment to a model's forecast, recorded with its reason (Forecast → Adjust round…). |  |

Spelling is British (normalise, colour, annualise), as in the rest of the documentation. Labels are in sentence case,
with units in brackets where they help: Expiry (years), Width (inches).

In the app: **Help → Help contents**, *Terms used in pymetrix*.

## Undo and Redo

**Edit → Undo** (Ctrl+Z) takes back the last step made from a dialog: the objects are as they were before it (a
replaced result comes back, a new one goes), and the step leaves the History and the script, with any recalculation
it caused. It can go back 20 steps. Undo refuses, and says why, once objects have changed since the step, by typed
code or by hand, since that work would be lost, and after Clean up history or a freeze; typed code and opening a
project start afresh. If a spreadsheet of a dataset Undo takes away has edits not applied, it asks first. In a text field, the
spreadsheet or a Code window, Ctrl+Z undoes typing there while there is typing to undo. In code, `session.undo()` does the same.

**Edit → Redo** (Ctrl+Y or Ctrl+Shift+Z, and the platform's own Redo keys) runs the step undone again, as it was
recorded, with the recalculation after it; the menu names it ("Redo fig = figure_of(fig).restyle(…)"). Several
steps undone are redone in order, last undone first. A new step clears what can be redone, and Redo refuses, and
says why, once objects have changed since the Undo. In code, `session.redo()`. Plugins that observe the session
(the lab book) hear of a step redone as of any step run. Edits to a figure, from its Edit… dialogs or made on the
figure itself (dragging a note, deleting a band), are steps like any other, so Undo and Redo take them back and make
them again; there is no separate undo inside a figure window.

## Keyboard shortcuts

**Help → Keyboard shortcuts** lists every shortcut: the menus' (read from the menus themselves, so the list is always
current), the workspaces' (Ctrl+1 to Ctrl+6, Ctrl+\` for Code), and those of dialogs and panels (F1 for a dialog's
help, Enter and Esc in the spreadsheet's edit line, Ctrl+F in Code windows). Type in its box to narrow the list.

In a dialog, Alt and the underlined letter of a field's label goes to that field (each dialog's letters are its
own, none of them OK's, Cancel's or Help's). A formula is checked as you type: a mistake shows in red under the field
and OK waits for it; a name the data do not have shows in grey, with the nearest name.

## Find anything (Ctrl+K)

**View → Find anything…** (Ctrl+K) is one search box over the whole app: every menu item (shown with its
place, "Data → Transform → Seasonal adjustment…"), the workspace's objects and the help pages. Type a few letters
of any words ("seas adj", "var lag", "gdp") and press Enter, or choose another with the arrows. It opens what the
menu would; nothing more is recorded.

It knows EViews', IRIS's and Dynare's names too: `genr` offers **Generate series… (EViews: genr)**, `smpl` **Set
sample…**, `ls` **Equation…**, `sstate` **Check model…**, `stoch_simul` **Rational expectations…** and
`estimated_params` **Estimate model…**. They come after whatever pymetrix's own names match, so they never push
those down. The names come from the tables of the help pages **Coming from EViews** and **Coming from MATLAB/IRIS
and Dynare** (`pymetrix.coming_from`), so the palette and the pages agree.

## Large datasets: working sets and searchable lists

A dataset with hundreds or thousands of series makes every list of series long. Two things keep the app manageable:

- **A working set.** Right-click a dataset in the workspace → **Working set…**, and give names and patterns
  (`gdp_*, cpi*, u`) or a selection (**Data → Select series…**; or right-click the selection → **Use as working
  set…**). Every dialog's series lists, the spreadsheet and the dataset's window then show only those series, and
  say so ("150 of 2,140 series (working set: gdp_*)"); **Show all** in a list, or **Working set only** in the
  spreadsheet, shows every series for a moment. **Clear working set** goes back to all of them. A series already
  ticked stays ticked when it is outside the working set.
- **Search in every series list.** Type in **Filter series…** to see only the series that match; **Ticked only**
  shows what is chosen; the count says how many are ticked. A list choosing one series narrows as you type.

- **The series browser.** A dataset window's **Variables** tab lists its series with their label, units, source,
  first and last observation, observations and gaps. Filter it by any of those, sort by any column, pick series
  (Ctrl- or Shift-click), then **Plot…**, **Edit in spreadsheet** (only those series shown), **New selection…**
  or **Use as working set**, from the buttons or a right-click.

A working set is a view, not a step: commands still record the series they use by name, so scripts run the same
with or without it. It is saved with the project. In code: `session.set_working_set("data", "gdp_*")`. A dataset
of more than 300 series without one suggests it in its window's header.

## Spreadsheet editor

**Data → Spreadsheet editor…** (Ctrl+E), or right-click a dataset → **Edit as
spreadsheet**. Periods are rows and series columns; for panels, choose an
entity or show all rows. Type values or paste blocks from Excel (thousands
separators and NA are understood), with Undo/Redo. The **edit line** above the
table shows the selected cell (its series and period) and its value in full: type
a new value there, then **✓** (or Enter, which also moves down a row) to accept
it or **✗** (or Esc) to put the old one back; empty means missing. A derived
series' cell is shown but not editable. Edits stay pending, with
derived series already showing their effect, until **Apply** (Ctrl+Enter, which
also takes a value still in the edit line) records them as one step:

```python
data = data.edit({'c': {'2020Q1': 1.2, '2020Q2': None}})   # panels: {('FRA', '2020Q1'): ...}
data = data.rename(i='inv')                                 # recipes that use i follow
data = data.drop('x')
data = data.with_info('c', units='USD bn', measure='flow')
data = data.freeze('share')                                 # derived -> plain, so it can be edited
```

Shading shows pending edits, cells edited earlier (flagged `edited`, saved with
the data), derived series (read-only) and other flagged observations. An edit
keeps the series' vintage, because a vintage is a release date. New series
come from a formula, fixed or kept updated (**Data → Generate series…** in
the menus). If the dataset changes elsewhere while edits are pending, the
editor says so and Apply puts the edits on the new version.

**Right-click in the spreadsheet.** On the cells: **Copy**, **Paste**, **Fill down** (each column's first selected
value into the cells selected below it) and **Clear (make missing)**, each one change for Undo and pending until
Apply. Then, for the selected series (or the cell's, when the right-click is outside the selection): **Plot…**,
**Describe…** (descriptive statistics), **New series from it…** (Generate series, a formula for each), **Keep … as a
new dataset…**, **Rename…**, **Remove**, **Freeze** for a derived series, and **Series info**. A right-click on a
series' heading offers the same for that series (or the columns selected with it), and **Copy the column**.

## History, Immediate and Code windows

Task page: [Run a script in the app](../src/pymetrix/howto/run-a-script-in-the-app.md).


- **Immediate** (bottom panel) is a Jupyter console sharing the session's
  objects: dialog results are there to use, and objects made there appear in
  the Objects panel. It needs `pip install pymetrix[qt]` (qtconsole).
- **Code windows** (File → New script, New script from session, Open script)
  edit `.py` files. They offer syntax colouring, line numbers, auto-indent,
  Tab/Shift+Tab block indent, Ctrl+/ to comment, bracket matching, and find
  and replace. Completion (Ctrl+Space or `.`) and signature hints come from
  Jedi and see the live objects, so `fit.` offers `params`. pyflakes marks
  problems as you type. F5 runs the script and F9 the selection or current
  line, through the Immediate window; a failing line is highlighted.
- **Dialogs** mark with * the fields a step needs; OK waits until they are
  filled, and the line above the buttons says which are missing. A field that
  picks a result (an estimated model, a dataset) says when there is none yet, with
  a button to the dialog that makes one. When a step fails, the message is titled
  after its dialog, says what went wrong and what to try, and keeps the technical
  text under **Show Details…**; the dialog stays open to correct.
- **History** lists every step, from dialogs and code alike, with failed steps
  in red. Select steps to copy their code (Ctrl+C, imports included), send them
  to the Immediate or Code window, drag them, re-run them, or reopen a step's
  dialog filled in as it was. It shows From (the dialog; its tooltip gives the
  menu path, the step's number and time), Code and Made; right-click the header
  to show the step number (#) and Time columns too.

- **Objects** has a filter box (**View → Find object…**, Ctrl+Shift+F): type part
  of a name, a type (`fitted`, `round`) or a status (`out of date`). Right-click
  an object for **Show the step in History** and **Copy the step's code**.
- **Problems** (next to Script and Immediate) lists what needs attention:
  out-of-date or uncomputed results, steps that failed or used out-of-date
  results, and results' own warnings.
  Right-click a line for what it is about: a step's **Show the step in History**,
  **Open its dialog, filled in…** (to correct and run it again) and **Copy its code**;
  an out-of-date result's **Open**, **Recalculate** and **Recalculate everything out of
  date**; a plugin's **Plugins…**; and **Copy the problem**. A double-click does the first.
- **Right-click elsewhere.** In the **Report** panel: edit an entry's caption or words,
  open its result, move it (up, down, to the top or the end), add a heading or a
  paragraph after it, remove it. In a **Code** window, besides cut, copy and paste:
  **Run this line** (or the selection), **Run the whole script**, **Comment or
  uncomment**, **Find…** and **Help on "name"** for the name under the mouse (its
  documentation, from the session's live objects). In the **Script** panel: **Copy the
  whole script** and **Open in a Code window**. In the **Lab book** panel, on an entry:
  **Add a note on entry N…** (the note records the entry's number), **Copy the entry**
  and **Copy its full record (JSON)**. On a window's title bar, or its tab in tabbed
  view: **Copy the window as an image**, **Add to the report**, **Close the other
  windows**, **Tile** and **Cascade**. The Immediate window keeps its console's own menu.
- The **status bar** shows the dataset in focus (series, frequency), its sample,
  the selected object and whether the last step worked.
- A dataset window shows a one-line summary above its views: series, frequency,
  periods, sample, series kept up to date and with a source, missing values. Its
  **Variables** tab is the data dictionary, sources and vintages included.
- **View → Zoom in / Zoom out / Actual size** (Ctrl+= or Ctrl++, Ctrl+-, Ctrl+0)
  enlarges the active window only (tables, figures, the spreadsheet, Code
  windows), for projectors and screen sharing; menus and panels keep their size.
  Zoom is view state: it is never saved in projects.
- **View → Text size** (Smaller, Normal, Larger, Largest) sets the size of all the
  app's text at once: menus, dialogs, panels, tables, help, the Code windows, the
  Immediate window, the charts on screen and the objects' icons. Normal is your system's font; the
  others are 85%, 115% and 130% of it. The change applies at once and is kept for
  next time. Zoom works on top of it, window by window. Saved and copied figures
  keep their own sizes whatever the text size, so a report looks the same on every
  machine. The text size has no keyboard shortcut (the zoom has Ctrl+=, Ctrl+- and
  Ctrl+0).
- **File → Open demo project…** lists the demos, each with what it shows and the windows it opens (Up and Down
  move, Enter or a click opens one), and opens a session with the demo's steps already
  run (`forecast_round`, Lab 3; `consumption`, Lecture 1; `nk_estimation`, a New
  Keynesian model estimated with its posterior, identification and data
  contributions; `policy_and_nonlinear`, the lower bound, optimal policy, the best
  simple rule, a sweep and second order), with its history and script, and its
  headline results open (the round's fan chart, the dynamic consumption function,
  the posterior, the zero-lower-bound episode), as the status bar says. In code:
  `pymetrix.demo("forecast_round")`. The window title names the demo; **File →
  Reload demo project** opens it again as it first was, after trying things in
  it; its steps are in Script, and **File → Export script…** saves them as a
  Python file. Each demo's script is tested in a fresh Python process with no
  display: it reproduces the session's results and loads no Qt or plotting
  library.

Everything typed is recorded, so the script and projects replay sessions that
mix dialogs and code. Cells in IPython-only syntax (`%timeit`, `x?`) are
recorded but appear in the script as comments.

**Out-of-date results.** Each step records which objects it read and wrote.
When an input changes, results built on it are marked **out of date** in the
Objects panel, with the reason (for example "c in data has changed"), and
**Recompute** re-runs the steps that made them. For datasets the check is per
series: dialog steps record the series they use, so a fit of `c ~ y` goes out
of date when `c` or `y` changes, but not when another series changes or one is
added. (Typed code is checked against every series of the dataset.) In code:
`session.stale()` and `session.recompute()`.

**Calculation, as in a spreadsheet.** The **Calculate** menu sets the mode:
**Manual** (the default) marks results out of date when their inputs change, and
**Recalculate** (F9) brings them up to date in dependency order; **Automatic**
recomputes them after every step. Right-click a result to **Freeze** it: it is then
never recomputed, even when its inputs change (it shows as "frozen; inputs changed"),
until you unfreeze it. **Clean up history…** removes dead ends: steps whose results
were later replaced or deleted and that nothing current uses. In code:
`session.set_mode("automatic")`, `session.freeze("fit")`, `session.dead_ends()`,
`session.prune()`. A step that uses results that are out of date, frozen while their
inputs changed, or not computed says so: the app warns after it runs, the step records it
(`record.cautions`), and a report made from them prints it at the top.

## Projects

Task page: [Save, reopen and share a project, and replay its script](../src/pymetrix/howto/save-and-share-a-project.md).


**File → Save project** writes one `.pymx` file: the data the session started
from (embedded, with metadata), the command history, and the Python script.
**File → Open project** restores the data and replays the commands, rebuilding
every result exactly. In code: `save_project(session, "work.pymx")` and
`load_project("work.pymx")` from `pymetrix.commands`.

Opening does only the work needed. Dead ends stay in the history and the script but
are not run. Results of slow steps (two seconds or more, such as Metropolis-Hastings
or a long backtest) are saved in the project and taken from it, without running the
step again, when the project is yours (or trusted) and opened by the same version of
pymetrix. Otherwise, in Manual mode, slow steps and the steps that use their results
are listed as **not computed** until you compute them (Calculate → Compute results
not computed, or double-click one); a slow step whose inputs a later step changes
runs in its place, so it gives the result it had. The calculation mode and frozen
results are saved with the project.

A program can record a step the same way without running it: `session.add(cmd,
deferred=True)` (any dialog's command, or a `Call`) adds it as **not computed**, as
Manual mode defers a slow step on opening, and `session.compute()` runs it in its
place when asked. History lists it as not computed, the script marks it, and a saved
project keeps it deferred (Automatic mode runs it when the project is opened).
`session.add(cmd)` without `deferred` runs the step now, exactly as `session.run(cmd)`.

What a deferred step uses is read from its command when it is added (the objects its
fields name, its call's references, the names typed code reads), so the steps that make
them are never dead ends, and `session.compute()` runs it against them as they were at
its place, even if a later step has made one of them again since: a model defined, an
estimate of it added deferred, the model redefined; computing the estimate uses the first
model, and the estimate is then out of date, as it would be had it run at once. A
project whose deferred step names an object that is missing when it is opened still
opens, with the step not computed and a note saying what is missing (in History and the
Problems list): opening never fails because of a pending step.

```python
s.add(FitEquation("c ~ y + c[-1]", name="slow"), deferred=True)   # recorded; s.pending() == {"slow": ...}
s.add(RunCode("slope = float(slow.params.values[1])"), deferred=True)   # it uses a pending result: waits too
s.compute(["slope"])   # runs both, in their places
```

The objects a deferred step will make are read from it (a call's `output`, a dialog's
`name`, the names typed code assigns); `outputs=["a", "b"]` says them when they cannot
be.

**Compatibility.** Each step is saved as its command's stable ID (for example
`pymetrix:fit_equation`; a plugin's commands carry the plugin's prefix), a schema
version and its fields. When a command's fields change, its schema version rises
and a migration (`pymetrix.commands.registry.migration`) turns old steps into new
ones, so older projects keep opening. A project from a newer pymetrix, or one that
needs a plugin that is not installed, is refused with a message that says what to
install. Projects from before October 2026 (format 1) still open.

**Trust.** Code typed in the Code or Immediate windows is Python, and Python can do
anything a program can. So a project's typed code runs on opening only when the
project is trusted: projects you saved yourself, or ones you trust after reading
their code. Otherwise the app shows the code and asks. **Open without running the
code** restores the data and dialog steps, and keeps the code, marked *not run*.
Trust is remembered by a fingerprint of the code (in `~/.pymetrix`, or
`PYMETRIX_HOME`), so any change to the code asks again. In code:
`project_code(path).steps` to read it, `load_project(path, trust=True)` to run it.

A project can also carry course material: in **Help → Course material…**
choose a folder and **Include in project** (in code,
`session.attach_course("week3")`). The folder's notes, notebooks, slides and
data files are saved inside the `.pymx` file. Whoever opens the project gets
them in the help, unpacked to a cache folder (`PYMETRIX_CACHE` to move it), so
an assignment's data, commands and notes travel as one file.

### Saving a single result

To send one estimate, test, forecast or solution to someone, or keep it outside a project,
**File → Save result…** writes the active window's result (or the one selected in Objects) to a
`.pymxr` file. **File → Open result…** reads one back: the same tabs, a **Saved** tab saying what
it was and when it was saved, and Copy summary. In code:

```python
save_result(fit, "consumption.pymxr")
theirs = load_result("consumption.pymxr")
theirs.summary(); theirs.to_frame("Coefficients"); theirs.provenance
```

A saved result is a record of the result as it was, not the live object: its tables and figures
can be read, copied and redrawn (in any theme), but it cannot be forecast from or estimated again.
The project that made it can. The file holds only JSON and text, so opening one runs nothing,
whoever sent it. Projects keep a copy of each result they open, so they still open when the file
has moved.

A result made by a step of a session (a dialog, or typed code recorded as calls) keeps the call
that made it in its provenance: `provenance["call"]` holds the function's ID, its arguments as
recorded (the span and row condition as they were given, settings in full) and the name the result
was kept under; `provenance["call_uses"]` names the datasets and results it read. Given those
objects, the call makes the result again:

```python
from pymetrix.calls import Call

saved = load_result("consumption.pymxr")
fit = Call(**saved.provenance["call"]).invoke({"data": data})   # the same coefficients as saved
```

A result made in a plain script has no recorded call; give it one with `save_result(fit, path,
call=make_call("pymetrix:equation.fit", Equation("c ~ y"), data=Ref("data")))`.

## Reading and writing projects from programs

Another program (a pipeline, a notebook tool, a review script) can build a session's steps,
save them as a project the app opens, and read projects and saved results back, through a small
set of names. They are a **public interface, provisional before 1.0**: they are documented and
tested as public, but their names and details may still change before version 1.0. Saved projects
keep opening as they do (each step is versioned and migrated), but no further stability is
promised yet.

| Name | Where | What it is |
| --- | --- | --- |
| `Call` | `pymetrix.calls` | one call of a public function: `function` (its stable ID), `version`, `receiver`, `arguments` (encoded as below), `output` (the name its result is kept under). `call.invoke(objects)` runs it on named objects and returns the result; `call.to_code()` is its Python |
| `make_call(call_id, receiver, output=..., **arguments)` | `pymetrix.calls` | the `Call` of a function, defaults applied, as recording it would give; name workspace objects with `Ref("data")` |
| `call_of(result)`, `call_record(call)` | `pymetrix.calls` | the call a session step made a result with; a call as plain JSON (`Call(**record)` rebuilds it) |
| `encode(value)`, `decode(value, refs)`, `Ref`, `SerialisationError` | `pymetrix.codec` | settings and specifications as JSON and back (`{"$type": "pymetrix:OLS", ...}`, `{"$ref": "data"}`) |
| `to_json(command)`, `from_json(value)` | `pymetrix.commands.registry` | a dialog's command as JSON and back, migrated |
| `Session`, `CommandRecord`, `save_project`, `load_project` | `pymetrix.commands` | a session: `run`, `add` (with `deferred=True`, a step not computed yet), `compute`, `pending`, `history` (each record's `call`), `script`, `save_log`, and `attachments` |
| `Session.attachments` | `pymetrix.commands` | a program's own files in the project: `{key: {file name: bytes}}`, saved under `attachments/<key>/`, reopened as they were |
| `save_result`, `load_result`, `SavedResult` | `pymetrix.saved` | one result in a `.pymxr` file; its provenance keeps its call |
| `build_info()` | `pymetrix.build` | the pymetrix commit (and whether it was changed) recorded beside the version |

**Writing.** A program builds calls and runs them in a session, or records slow ones without
running them, then saves the project. The app opens it like any other: the steps, their code
and their results.

```python
from pymetrix import Dataset, Equation, OLS
from pymetrix.calls import make_call
from pymetrix.codec import Ref
from pymetrix.commands import Session, save_project

s = Session()
s.objects["data"] = Dataset.example("sample_macro")
s.run(make_call("pymetrix:equation.fit", Equation("c ~ y + c[-1]"), output="fit", data=Ref("data"),
                method=OLS(cov="hac")))
s.add(make_call("pymetrix:equation.fit", Equation("c ~ y"), output="later", data=Ref("data")), deferred=True)
s.attachments["mytool"] = {"run.json": b'{"pipeline": "nightly"}'}
save_project(s, "work.pymx")
```

**Reading.** A project is a zip file: `commands.json` lists the steps. A step recorded as a call
has the id `pymetrix:call` and the call's fields; a dialog built on a call also carries it, as
`"call"`. Neither needs the session to be opened, so nothing is run:

```python
import json, zipfile
from pymetrix.calls import Call
from pymetrix.codec import decode

with zipfile.ZipFile("work.pymx") as z:
    steps = json.loads(z.read("commands.json"))
    log = json.loads(z.read("manifest.json"))["save_log"]
calls = [Call(**(step["fields"] if step["id"] == "pymetrix:call" else step["call"])) for step in steps
         if step["id"] == "pymetrix:call" or "call" in step]
calls[0].to_code()                       # "fit = Equation('c ~ y + c[-1]').fit(data, method=OLS(cov='hac'))"
decode(calls[0].arguments["method"])     # OLS(cov='hac')
[step.get("deferred", False) for step in steps]   # [False, True]: the second is not computed yet
log[-1]["pymetrix"], log[-1]["commit"]   # the version that saved it, and its commit (None from a wheel)
```

Opening it replays the steps (`load_project`), with the program's files as they were saved:

```python
from pymetrix.commands import load_project

s = load_project("work.pymx")
s.attachments["mytool"]["run.json"]      # b'{"pipeline": "nightly"}'
s.pending()                              # {"later": ...}: s.compute() runs it
s.history[0].call.invoke(s.objects)      # the fit again, from its call
```

## Plugins (Extension API v1)

Estimation methods, models, variance models, tests, filters, solvers, data
readers, result views and dialog commands are all registered extensions.
Menus, dialogs, scripts and projects read the registries, so a plugin's
components appear and behave exactly like built-in ones.

```python
from pymetrix import registry, plugin_table

registry("methods").labels()          # ['OLS', 'IV', 'ML', 'LIML', 'IV-GMM', ...]
registry("variances").get("GJR-GARCH").build(dist="t")
plugin_table()                        # installed plugins, licences, status (Help → Plugins)
```

A plugin is a package with a `PLUGIN = PluginInfo(...)` and entry points:

```toml
[project.entry-points."pymetrix.filters"]
demean = "my_package:Demean"
```

Settings are frozen dataclass fields with `Annotated[..., Widget(help=...)]`;
plugins write no UI code. Help comes from the same declarations: the docstring
(usage), `wraps` and `docs` (the wrapped library's reference), and `topics` and
`references` (background; see `pymetrix.help`). `pymetrix.testing` is the
conformance kit (`check_plugin`, `check_entry`, `check_result`,
`check_project`, `check_determinism`, `crosscheck`); `tests/sample_plugin.py`
is a minimal example. The linearmodels and arch adapters in `pymetrix.ext`
are the bundled reference plugins. [Wrapping another library](wrappers.md) gives the general form of a
wrapper, with a worked example in `examples/custom_wrapper.py`.

**The app's extension points (API 1.2).** Besides methods, tests, readers, views and dialogs, a plugin can add a
workspace, lay out its own menu, dock a panel and add a report template (1.2):

```python
from pymetrix.plugins import register
from pymetrix.extension import AppPanel, MenuLayout, Workspace   # plain classes: no window or Qt needed to register

register("workspaces", Workspace("risk", "Risk", ("Risk",), ("Objects", "History", "Risk monitor"), ("ValueAtRisk",), "market risk"))
register("menus", MenuLayout("Risk", ["ValueAtRisk", None, ("Backtests", ["KupiecTest"])]))
register("panels", AppPanel("Risk monitor", lambda app: RiskMonitor(app), area="right"))

from pymetrix.report import ReportTemplate                       # the views to report, by tab title, in order
register("report_templates", ReportTemplate("risk_pack", "VaR and its backtests", ("Backtests", "Graph")))
```

A panel's `create(app)` gets an `AppInterface`: `app.session`, `app.selected()`, `app.open_dialog(cls, initial)`,
`app.run(command)` (as a dialog runs it: recorded, in the background with Cancel, undoable),
`app.on_history_changed(callback)`, `app.status(text)` and (API 1.7) `app.open_demo(name)`, which opens a demo
project (`pymetrix.demos()`) as File → Open demo project does: the app asks to save unsaved work first, and an
unknown name raises `NotFoundError` with the demos there are. An interface made without the app (in a plugin's own
tests) raises `NotSupportedError`, an `AttributeError`; a panel that must also run on an earlier API looks the method up
with `getattr(app, "open_demo", None)` and catches that error. A result class sets `stage = "forecast"` and
`next_steps = (("Plot it…", "PlotSeries", "data", "explore"),)` to place its badges and its Next ▸ menu.

A panel that proposes steps, such as an assistant, opens each one inside `with app.session.proposing(CommandClass,
"Claude"):` (API 1.9). The step the user then runs from that dialog has `CommandRecord.proposed_by` set. It is saved
with the project, History shows "proposed" after the dialog's name, and the exported script has a comment line above
it.

**Observing sessions (API 1.3).** A plugin that keeps its own record of the work, such as a lab book or an activity
log, registers a `SessionObserver`. Every session then tells it what happens, in scripts, notebooks, the app and the
MCP server alike:

```python
from pymetrix.extension import SessionObserver
from pymetrix.plugins import register

def handle(event):                  # a SessionEvent
    if event.kind == "step":        # also "failed", "cancelled", "undone", "removed", "opened", "saving"
        print(event.during, [r.code for r in event.records])     # during: "work", "recalculation" or "opening"
    elif event.kind == "saving":    # the project is about to be written: keep your files in it
        event.session.attachments["mylog"] = {"log.jsonl": b"..."}

register("observers", SessionObserver("mylog", handle))
```

`session.attachments[key]` (file name -> bytes) is saved in the project under `attachments/<key>/` and reopened
with it. It is kept as it is when the plugin is missing. The project's `save_log` notes, at each save, which observers
keeping a record (`records=True`, the default) were active. An observer that raises is reported once, as a warning,
and the session carries on.

**What a plugin imports.** Everything a plugin needs from pymetrix's inside comes from `pymetrix.extension`, under
names that stay put as pymetrix changes: `PluginInfo`, `plugin_context`, `register` (registration); `api` (a recorded,
replayable call: `@api("myplugin:fit")`), `make_call` and `Ref` (a dialog's call); `Command`, `Widget`, `command`
(dialogs); `register_view`, `figure_views`, `frame_table`, `views_frame` (a result's window and `to_frame`);
`report_progress`, `check_cancelled`; `add_help_topic`; `ReportTemplate`, `ExampleDataset`, `Workspace`,
`MenuLayout`, `AppPanel`; `SessionObserver`, `SessionEvent`; `numerical_hessian`, `point_estimate`; and (API 1.4)
`migration`, `call_migration` and `renamed_arguments`, for renaming a dialog's fields or a call's arguments without
breaking saved projects and scripts; and (API 1.10) the core EViews reader's text helpers, for a plugin that
translates EViews programs: `eviews_blank_comments` and `eviews_statements` (a program's statements with their
offsets), `parse_eviews_expression` (an expression as a tree of `EViewsNumber`, `EViewsName`, `EViewsCoefficient`,
`EViewsCall`, `EViewsNegative`, `EViewsBinary` and `EViewsBrackets` nodes, each with its offset), `walk_eviews_expression`,
`render_eviews_expression` (a tree as `convert_expression` writes it), `EViewsExpressionError` (what does not read,
with `.at` and `.length`), and `Diagnostic` and `line_and_column` to report it at a line and column. Its other imports come from the stable modules,
`pymetrix.extension.STABLE`: `pymetrix` itself, `errors`, `data`, `output`, `figures`, `estimation`, `statespace`,
`metadata` and `commands`, and no name starting with `_`. The finance plugin is held to this by a test.

**Starting a plugin.** `pymetrix-new-plugin pymetrix-myname "What it does"` (or `pymetrix.plugin_template.new_plugin`)
writes a working plugin laid out as the finance plugin is: a `pyproject.toml` with its entry points, a recorded call
returning a result with a summary, a graph and `to_frame`, a dialog under its own menu, a help topic, tests (an
independent check, the dialog's recorded code, the public-surface guard, the tutorial run as written), a tutorial,
a validation table and a BSD-3-Clause `LICENSE` in the author's name, which `license-files` puts in the sdist and
wheel (change the licence there, in `LICENSE` and in `PLUGIN` together). A plugin that bundles data or code from
elsewhere adds a `THIRD-PARTY-NOTICES.md` to `license-files`, as the finance plugin does. `cd pymetrix-myname`, `pip install -e .` and `python -m pytest` pass at once; then replace the
example with the plugin's own work, keeping the pattern.

**Managing plugins.** **Help → Plugins…** lists each plugin with its status, what it provides (methods, tests,
dialogs and their menus, data readers and their address schemes) and its licence, and can enable or disable it or
copy the command that installs it; **Check plugin** runs the conformance kit on it. A plugin's dialogs and History
steps say which plugin they come from, and Problems lists any plugin that could not load. The app asks before a
newly installed plugin first runs, from its package's
metadata alone; scripts load a new plugin with a warning. A disabled plugin stays disabled in the app and in
scripts (`pymetrix.plugins.enable(name)` and `disable(name)` do the same from code): a bundled one's methods
leave the menus and lookups by name at once, an installed one is not loaded from the next start. A class imported
directly in a script (`from pymetrix import LIML`) still works. A project that needs a plugin that is missing or
disabled still opens: those steps, and the ones using what they made, stay in the History marked as not run,
with what to install, and saving keeps them.

## Help and course material

Every dialog has a **Help** button (or F1). It opens a pane beside the form
with the page for the dialog and for whatever is chosen in it (the method,
variance model, filter or test), and follows the choices as they change. Links lead to the
wrapped library's own documentation and to background topics on the
econometrics. **Help → Help contents** is a searchable browser of every page. Its contents page lists one page
per line, under headings that open and close (**Open all**, **Close all**); in notebooks `show_help()` shows the
same sections folded.

For users of other packages, **Help → Coming from EViews** and **Help → Coming from MATLAB/IRIS** (linked at the top
of the contents too) say where the familiar things are: the objects (a workfile is a project, a page a dataset, an
equation object a fitted equation, a model object a Model, a spool the report; an IRIS model a template, a databank
a dataset), a table of the everyday commands and menu items with the menu path and the Python call for each, the
defaults that differ, what is different by design (every step recorded, scripts, Undo, projects) and how to bring
models across. In code, `pymetrix.help.topic("coming-from-eviews")` and `topic("coming-from-iris")`.

In notebooks, forms have a Help toggle, and `show_help(OLS)` or
`show_help("garch")` from `pymetrix.ui.jupyter` show a page.

Instructors can add their own material: a folder of lecture notes (`.md`)
and example notebooks (`.ipynb`). Front matter links each note to the help
pages it belongs to, and the note then appears on those pages under
**Course material**, in the contents and in search:

```markdown
---
title: Lecture 6. Volatility
course: ECON 5010 Applied Time Series
links: [garch, variances/GJR-GARCH, dialog/FitEquation]
references:
  - Tsay, R. (2010). Analysis of Financial Time Series, chapter 3.
---
```

Add folders with **Help → Course material…**, or for a lab list them in the
`PYMETRIX_COURSES` environment variable. A course can also be published as a
package with a `pymetrix.courses` entry point. `pymetrix.testing.check_course(folder)`
reports links that match no page and missing files. `examples/course` is a
small example.

A course can also limit the AI assistant (the `pymetrix-assistant` plugin) while it is loaded. Add
`assistant: explain` to any note's front matter, and the assistant explains methods and results but proposes no
steps; "Allow connections" for an outside assistant is off too. `assistant: off` turns the assistant off
altogether. The strictest rule any loaded course sets applies, and the Assistant panel says which course set it.

**Guided case studies** (the `pymetrix-guide` plugin, `pip install pymetrix[guide]`; experimental): a note with
`pymetrix-step` blocks becomes a case study the **Guide** panel (View → Guide) takes a trainee through, one step at
a time, opening each step's dialog filled in and ticking it when its result appears. Lab 3 ships as the first.
[Writing a case study](case-studies.md) shows how to make one.

A notebook can keep its front matter in its metadata, `{"pymetrix": {"title": ..., "links": [...]}}`, instead of in
its first cell. Then it opens on its own title in Jupyter, with no `---` block showing.

**A reference cell for a lab:** `pymetrix.testing.write_api_reference(notebook)` runs a notebook and adds, as its
last cell, a table of every pymetrix function, class and method the notebook calls. The table gives each one's
arguments and the first line of its documentation. `check_api_reference(notebook)` fails, in your own tests, when
that table is out of date with the code, describes something that has no docstring, or misses a name the notebook
imports. The forecasting course pack (`examples/forecasting`) uses both.

## Writing an exercise pack

**Help → Exercises…** lists guided exercises: each step says what to do in the app, with hints, the same step as
code (**Show the code**), and a **Check** that reads the session's results and marks each check green or red with a
message. Exercises come in **packs**: a folder you open with **Open folder…**, or a plugin. When an exercise starts,
its data load into the session once you agree, as File → Load data… steps (recorded, and Undo takes them back), and
the **Exercise** panel docks on the right. The format and `pymetrix.exercises` are provisional before 1.0; the full
reference is [the exercise pack format](exercise-packs.md).

Start a pack with the command line, then check it:

```
pymetrix-exercises new my-pack
pymetrix-exercises check my-pack
```

A pack is `pack.toml` (`[pack]` with `name`, `title`, `version`, `description`, `requires_pymetrix` and, optionally,
`audience`, free text shown in the list, and `classification`, free text such as "internal" that the app and the CLI
show and nothing enforces), an optional `data/` folder of `.parquet` or `.csv` files, an optional `models/` folder
(`.mod` Dynare, `.model` IRIS, `.pymodel` pymetrix's model syntax), and `exercises/01-<slug>/` folders. A model is
imported under its file's name without the extension unless `pack.toml` names it: `[models.nk_2019]` with
`name = "nk"` imports `models/nk_2019.model` as `nk`. Each exercise has an `exercise.md` (`level` is `beginner`,
`intermediate` or `advanced`; `models: [rbc]` lists the models it uses, which the Exercise panel offers to import
with their dialogs filled in):

~~~markdown
---
title: A wage equation with robust standard errors
level: beginner
minutes: 15
data: [mroz]
---

## Step 1: Fit the wage equation

Choose Estimate → Equation…, with Data: mroz, Formula: `lwage ~ educ + exper + expersq`, Method: OLS and
Save as: fit.

```python
fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)
```

> Hint: `lwage` is the log of the hourly wage.
~~~

and a `checks.toml`, one `[[check]]` per value to check:

```toml
[[check]]
id = "educ"
step = 1
object = "fit"
value = "params.educ"
expected = 0.1075
abs_tol = 0.0005
message = "The formula is lwage ~ educ + exper + expersq."
```

`data` names a file of the pack's `data/` (a CSV file also needs a `[data.<name>]` table in `pack.toml` naming its
time or entity column) or one of pymetrix's example datasets. `value` is a dotted path into the object: `params.educ`,
`se.educ`, `tvalues.educ`, `pvalues.educ`, `nobs`, `stats.r2`, `method.cov` for an estimate; `statistic`,
`p_value`, `critical_values.5%`, `details.lags used` for a test (the reference lists each, checked on real results). Checks are data: a path is followed by attribute and key lookup only, never calling
anything, and names starting with `__` are refused, so opening a pack from any folder runs nothing.

Write the in-app instructions with the menu paths and the dialogs' own field labels, as the How do I… pages do, and
choose tolerances from the numbers' meaning (a coefficient to its third decimal, a statistic to its second) so a
library's next release does not fail the check.

An optional `solution.py` completes the exercise in a fresh session, with the pack's folder as its working directory
and `PACK_DIR` (that folder) defined, so `load_model(PACK_DIR, "rbc")` reads a model of the pack. It is code, so only
`pymetrix-exercises check` runs it, on the pack you name: it loads each exercise's data, runs the solution (or, without one, the steps' code
blocks in order) and every check, and exits 1 when one fails (2 when the pack cannot be read). Run it in the pack's
own CI to catch drift between the pack and a new pymetrix.

To ship a pack in a plugin, put the folder in the package and point the entry-point group `pymetrix.exercises` at a
function returning it:

```toml
[project.entry-points."pymetrix.exercises"]
intro = "pymetrix_intro.exercises:pack_folder"
```

```python
from importlib.resources import files

def pack_folder():
    return files("pymetrix_intro") / "intro-pack"
```

A plugin's pack follows the plugin: disabled, it is not listed; new, it waits for your answer in Help → Plugins.

In code:

```python
from pymetrix.commands import Session
from pymetrix.exercises import check, data_commands, load_pack

pack = load_pack("my-pack")                 # every problem in one SpecificationError
ex = pack.exercise("first-regression")
session = Session()
for cmd in data_commands(ex):               # the data, as recorded steps
    session.run(cmd)
session.run_code('fit = Equation("c ~ y").fit(sample_macro)')
[(r.id, r.passed, r.actual) for r in check(ex, session, step=1)]
```

## Using pymetrix from an AI assistant (MCP)

`pip install pymetrix[mcp]` adds `pymetrix-mcp`, a Model Context Protocol server. With it, an AI assistant that
speaks MCP (Claude Desktop, Claude Code, VS Code and others) can work in a pymetrix session for you:
- load data;
- estimate, forecast, backtest and test;
- read the results as pymetrix's own tables and charts.

Every step is a recorded pymetrix command, so at the end `export_script` gives the Python that reproduces the
session, and `save_project` gives a project that replays in the app.

**Setup:** add it to the assistant's MCP settings. In Claude Desktop, that is the `mcpServers` entry of its
configuration file:

```json
{"mcpServers": {"pymetrix": {"command": "pymetrix-mcp", "args": ["--root", "C:/work/forecasts"]}}}
```

**What it may do:** the server is closed by default.
- **Files:** it reads only inside the `--root` folder.
- **Writing:** it writes nothing unless started with `--allow-writes`.
- **Python:** it runs no Python unless started with `--allow-code`.
- **Network data:** it fetches nothing (`fred://`, `worldbank://`, `sdmx://`) unless started with `--allow-network`.
- **Approval:** the assistant's client asks you before each step.

**The tools:**

| Tool | What it does |
|---|---|
| `list_commands`, `describe_command` | The app's dialogs and their settings |
| `run_command` | Run one, e.g. `FitEquation` with `{"formula": "c ~ y + c[-1]", "cov": "hac"}` |
| `list_objects`, `show_object` | What the session holds, and a result's tabs as tables or charts |
| `export_script`, `save_project` | The session as Python, or as a project |
| `search_help`, `read_help` | The in-app help |

In code, `pymetrix.mcp_server.build_server(session, Policy(...))` gives the server over a session of your own. The
design, and what comes next, is in [docs/notes/mcp-server.md](notes/mcp-server.md).
