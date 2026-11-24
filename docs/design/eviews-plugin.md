# EViews work into pymetrix: the `pymetrix-eviews` plugin

Design note, 5 November 2026, decided 24 November 2026 (see "Decisions") · the core additions (decisions 4 and 5) are built, and the plugin's data route 1 (the bundle reader, the experimental export program and File → Import EViews workfile…, named so that it has an access key, since the File menu uses every letter of "Import EViews": [`packages/pymetrix-eviews`](../../packages/pymetrix-eviews/README.md)); programs phase 1 (data work) is built
(`translate_program`); programs phases 2 to 4, the panel and route 2 are not · builds on the core EViews reader
(`src/pymetrix/models/eviews.py`), the writers (`src/pymetrix/models/writers.py`) and the extension API
([plugins.md](plugins.md), `pymetrix.plugins.API` 1.10)

## The question

Users who move from EViews bring two things: workfiles (`.wf1`) and programs (`.prg`). How should pymetrix take
both, so the user ends with Datasets, fitted equations and models in a session, and a script and a project that
replay?

## Short answer

- **A first-party plugin, `pymetrix-eviews`.** It is a bridge to another product, as `pymetrix-psets` is. The core
  reader stays in core, unchanged: `read_eviews`, `check_eviews`, `read_eviews_xml`, `read_eviews_sspace`,
  `convert_expression` and `eviews_series`. The plugin builds on them.
- **Data through EViews itself.** The plugin ships an export program. The user runs it once in their own EViews,
  and it writes a *bundle*: each page's series as CSV, and a JSON manifest with the metadata, equations and models.
  The plugin reads the bundle in one step. The binary `.wf1` is not read. The project has no EViews licence, so
  the export program ships marked *experimental* until someone runs it in EViews; until then, a workfile saved
  from EViews as CSV or Excel, which core already reads, is the supported way in.
- **Programs translated, not emulated.** A `.prg` becomes a readable pymetrix Python script and a translation
  report. Every line is either translated or listed with a reason. Nothing is dropped silently, as the model reader
  already does. The work comes in four phases: data, estimation, control flow, and what is left in the report.
- **No EViews command line in the app.** The user rejected one. pymetrix programs are Python (`docs/design.md`,
  "Programs are Python"). A separate help page, "Coming from EViews / IRIS", is wanted. It is related to this
  note but is not part of it.

## Conventions in this note

- *(verify)* marks EViews syntax or behaviour that must be checked against EViews's own *Command and Programming
  Reference* before it is used. Nothing here has been run in EViews. Syntax not marked is long-standing and
  widely documented, but the export program is still checked line by line in a real EViews before release.
- The plugin is written from EViews's public documentation and from programs the user owns. No EViews code, and
  no non-public program, is used to build it (`docs/notes/ai-assisted-development.md`, "Constraints").
- **Gap** marks a construct pymetrix has no equivalent for. Each one becomes a `docs/known-gaps.md` item and a
  report reason. None is a silent drop.

## 1. Data: three routes

### Route 1 (primary): an export program the user runs once

The plugin ships `pymetrix_export.prg`. **File → Import EViews… → Get the export program** saves a copy. The user
opens their workfile in EViews and runs it, for example `exec pymetrix_export.prg C:\work\macro-export`
(`exec` and the program arguments `%0`, `%1` are standard). The program writes the bundle, then the user imports
it in pymetrix with **File → Import EViews…**.

**Why this route.** EViews is the only reliable reader of its own workfiles. The program runs where the data
already are. It needs no licence on the pymetrix side and no network. The bundle is plain text, so the user can
read what leaves EViews, and a maintainer can make test fixtures by hand.

What the program does, page by page:

1. Lists the pages (`@pagelist` *(verify)*), selects each with `pageselect`, and records its frequency
   (`@pagefreq` *(verify)*), its range and the current sample (`@pagesmpl` *(verify)*).
2. Writes the page's series as CSV, under `smpl @all`, with `pagesave(type=text)` *(verify: the options for the
   delimiter, the NA string, numeric precision, and keeping only series and alpha series)*. It adds a date column
   made with `@datestr(@date, "YYYY-MM-DD")` *(verify the format string)*, so the importer never has to parse
   EViews's own date display.
3. Loops over objects by type with `@wlookup("*", "<type>")`, `@wcount` and `@word`, for the types `series`,
   `alpha`, `group`, `scalar`, `string`, `vector`, `matrix`, `sym`, `coef`, `valmap`, `equation`, `model` and
   `sample`. Every other object is listed in the manifest's `skipped` array with its type.
4. For each series, reads the descriptive data members `@description`, `@units`, `@source`, `@remarks` and
   `@displayname` *(verify which exist in which versions)*, and user attributes through `.@attr("name")`
   *(verify)*.
5. For each equation, writes its estimation command (`.@command` *(verify)*), its method (`.@method` *(verify)*),
   its sample (`.@smpl`), and its estimates `@coefs` and `@stderrs` with `@ncoef`. The estimates are not imported
   as results. They are kept so the plugin can re-estimate in pymetrix and compare (section 5).
6. For each model, writes its text, as its Text view shows it, to a file *(verify how: freezing the view into a
   text object and saving it is one way)*.
7. Writes scalars and strings into the manifest. It writes vectors, matrices and value maps as small CSV files,
   one each *(verify the command: a table frozen from the object's sheet and saved, or a direct write)*.
8. Writes the manifest last, so a bundle without one is an incomplete export and is refused.

The program changes nothing in the workfile. It uses its own temporary object names, with a prefix, and deletes
them at the end. It writes an `export.log` with one line per object: exported, or skipped and why.

### Route 2 (deferred): the same program through COM

When EViews is installed on Windows, the plugin can drive it through EViews's COM automation server *(verify: the
`EViews.Manager` ProgID, `GetApplication`, `Run`, and from which version COM is offered)*. It opens the workfile,
runs the same `pymetrix_export.prg` into a temporary folder, closes EViews, and imports the bundle. It does
nothing that route 1 does not do, so there is one export to maintain and test. Its only Python dependency is
`pywin32`, an optional extra (`pip install pymetrix-eviews[com]`). Without it, or off Windows, the dialog shows
route 1 only, with the reason.

**Deferred (decision 8).** Without EViews it cannot be tested at all, so it is not built now.

### Route 3 (not pursued): reading `.wf1` directly

- The format is undocumented. Any reader would be reverse-engineered.
- It varies across EViews versions, and nothing announces a change.
- Errors would be silent: a misread column of numbers looks like data. This breaks pymetrix's rule that what does
  not read is refused at its place.
- `docs/design.md` already records that `.wf1` has no reliable open reader, and the roadmap chose "EViews data
  through EViews' own interface".

### The bundle, format version 1

```
macro-export/
  macro.evbundle             the manifest (JSON, UTF-8); the file pymetrix opens
  pages/<page>.csv           one per page: date and id columns, then every series and alpha series
  text/<page>/<object>.txt   long texts: model text, equation commands longer than one line
  objects/<page>/<name>.csv  vectors, matrices, sym, coef; each equation's coefs and stderrs
  valmaps/<page>/<name>.csv  value maps: value,label
  export.log                 the program's own record of what it exported and skipped
```

The manifest holds only short strings. Long texts go in files, so the program never escapes long text inside
EViews. In every JSON string the program escapes `\` and `"` with `@replace` *(verify that EViews string literals
can hold a double quote, for example through `@chr(34)`)*. The reader parses strictly and reports an error with
its line and column.

```json
{
  "format": "pymetrix-eviews-bundle",
  "format_version": 1,
  "exporter": {"program_version": "1.0", "eviews_version": "…", "exported": "2026-11-05T14:03:00"},
  "workfile": {"name": "macro", "path": "C:\\work\\macro.wf1", "active_page": "quarterly"},
  "na": "NA",
  "pages": [{
    "name": "quarterly",
    "structure": "dated",
    "frequency": "Q",
    "range": ["1960-01-01", "2030-10-01"],
    "sample": "1980q1 2024q4",
    "file": "pages/quarterly.csv",
    "date_column": "pmx_date",
    "id_columns": [],
    "series": [{"name": "gdp", "kind": "numeric", "display_name": "", "description": "Real GDP",
                "units": "bn 2017 prices", "source": "…", "remarks": "", "attributes": {}, "value_map": null}],
    "groups": [{"name": "g1", "members": ["gdp", "cons"]}],
    "samples": [{"name": "pre", "text": "1980q1 2007q4"}],
    "scalars": [{"name": "alpha", "value": 0.3}],
    "strings": [{"name": "region", "value": "north"}],
    "arrays": [{"name": "w", "type": "vector", "rows": 4, "cols": 1, "file": "objects/quarterly/w.csv"}],
    "value_maps": [{"name": "vm_sector", "file": "valmaps/quarterly/vm_sector.csv"}],
    "equations": [{"name": "eq_cons", "command": "ls dlog(cons) c dlog(inc) dlog(cons(-1))", "method": "LS",
                   "sample": "1980q1 2019q4", "ncoef": 3, "estimates": "objects/quarterly/eq_cons_est.csv"}],
    "models": [{"name": "m", "text": "text/quarterly/m.txt"}],
    "skipped": [{"name": "gr1", "type": "graph", "reason": "graphs are not exported"}]
  }]
}
```

How each case is handled:

| Case | In the bundle | In pymetrix |
| --- | --- | --- |
| Pages | one CSV and one `pages` entry each | one `Dataset` each, named after the page (made a valid Python name; renames noted) |
| Several frequencies | one page per frequency, as in EViews | one Dataset per frequency: a Dataset has at most one time axis. `Dataset.to_freq` converts between them |
| Dates | `pmx_date`, the first day of each period, ISO | a `TimeAxis` of the page's frequency: A as `"Y"`, Q, M, D5 as `"B"`, D7 as `"D"`, W with its anchor day *(verify how EViews states it)* |
| Semi-annual, multi-year, intraday | written, with the frequency as EViews gives it | **gap:** pandas has no half-year period. Refused with the reason; the user can import the page as undated |
| Panel pages | the cross-section id and date columns, named in `id_columns` and `date_column` | `Dataset.from_frame(..., time=, entity=)`: a panel Dataset, balanced or not |
| Undated pages | `structure: "undated"`, an `obs` column | a cross-section Dataset with an `obs` entity axis |
| NA | the token in `na` (default `NA`); an empty numeric field is also missing | `NaN` |
| Alpha series | quoted text; an empty string is EViews's missing alpha | a text column (pandas string dtype). Estimation refuses it; `C(x)` turns it into dummies |
| Value maps | a CSV of value and label, and `value_map` on the series | the codes stay as data; the map goes in `Variable.extra["value_map"]`. **Gap:** core has no value labels |
| Labels and units | `description`, `units`, `source`, `remarks`, `display_name`, `attributes` | `Variable.label` (the display name, else the description), `description`, `units`, `source`; the rest in `extra` |
| Groups and samples | members; the sample's text | a `Selection` (`data.series(...)`) and a `Sample`, each a named object |
| Scalars, strings | in the manifest | parameters for `read_eviews(params=)` where a model uses them; otherwise listed, kept in `meta` |
| Vectors, matrices | one CSV each | `Matrix` objects |
| Equations | the command, method, sample and EViews's estimates | translated as a one-line program (section 2) and fitted; compared with EViews's estimates |
| Models | the text | `read_eviews(text, data=...)` |
| Links, `frml` series | their values *(verify that `pagesave` writes link and formula values)* | plain series, with a note. A `frml` can become a derived series only when its formula translates |

**Versioning.** `format_version` is a whole number. The reader accepts the versions it lists and refuses a newer
one with "update pymetrix-eviews". Within a version, changes are additive: an unknown key is kept in
`meta["eviews"]` and noted, never an error. The export program states its own version, and the plugin warns when
the bundle came from an older program than the one it ships. A JSON Schema file ships with the plugin. The tests
check every fixture against it.

**Encoding.** The reader expects UTF-8. If that fails, it falls back to Windows-1252 and adds a note to the
import's report *(verify what encoding EViews writes text files in)*.

**Recorded steps.** The plugin registers a reader for the extension `.evbundle` (`pymetrix.io.registry`, as
`pymetrix-psets` registers its scheme). One import writes one line per page in the History and the script:

```python
quarterly = Dataset.read("C:/work/macro-export/macro.evbundle", page="quarterly")
```

Replaying the project therefore needs the plugin, as a PSETS project needs `pymetrix-psets` (decision 3).

## 2. Programs: translation

### What comes out

`translate_program(source, bundle=None)` returns a `ProgramTranslation` with three parts, all from one parse:

- `.script`: readable Python that uses the public API only. Loops stay as loops, `{%s}` becomes an f-string, and
  subroutines become functions. Each block carries the EViews line it came from as a comment.
- `.report`: one row per source line (included files too). Each row has a status (`translated`, `translated with
  a note`, `left out`, `error`), the EViews text, the Python it became, and the reason, with the pymetrix
  equivalent where there is one. `report.summary()` is a `Table`, as `Translation.summary()` is for models.
- `.run(session)`: steps mode. The plugin walks the program itself. It evaluates control flow in Python, and it
  runs each data or estimation statement as a recorded call (`make_call`) through the session, or through
  `AppInterface.run` in the app. A condition on a result (`if eq.@r2 > 0.5`) is decided by the result just made.

**Recommendation: both.** The History needs flat, exact steps: a typed loop is recorded only as `RunCode`
(`docs/design/commands.md`, decision 4). People need a readable script that looks like their program. In the
app, *Run* makes steps, and the readable script and the report are saved with the project as attachments
(`session.attachments["eviews/<program>.py"]`). In code, the user chooses. A test runs each fixture both ways and
requires equal objects. Generated scripts import only `pymetrix`. The one exception is reading a bundle, which
needs the plugin's reader.

### Names

EViews ignores case, so names are lower-cased, as the core reader does. A name that is a Python keyword or a
pymetrix public name (`lambda`, `in`, `data`, `Model`) gets `_v` after it, as `eviews_names` does for EViews's
reserved words, and the report lists the change. The current page is a variable named after it. `pageselect`
changes which variable later lines use.

### Phase 1: data work

| EViews | pymetrix | Status |
| --- | --- | --- |
| `wfopen macro.wf1` | `Dataset.read(path, ...)` for the bundle, or the CSV or Excel file saved from EViews, given to the translator | translated; without either, an error saying how to make one |
| `wfcreate q 1990q1 2030q4`, `pagecreate` | `Dataset.empty("Q", "1990Q1", "2030Q4")` (File → New dataset) | translated (built in core, decision 5) |
| `pageselect p` | later lines use the Dataset `p` | translated (no step) |
| `wfsave`, `wfclose`, `pagesave` | the project is saved by the user | left out: "save the project (File → Save)" |
| `read`, `import` of a CSV or Excel file | `Dataset.read(path, ...)` | translated when the options map; otherwise left out, naming `Dataset.read` |
| `fetch`, `store`, `dbopen` (EViews databases) | none | left out: "export the database's series to the bundle" |
| `smpl 1990q1 2007q4`, `smpl @all`, `smpl @first 2000q4` | `data = data.with_sample(Sample(...))` (Set sample) | translated |
| `smpl if x > 0`, `sample s ...`, `smpl s` | `Sample(where="x > 0")`, a named `Sample` | translated |
| `series y = <expr>`, `genr y = <expr>`, bare `y = <expr>` under `smpl @all` | `data = data.with_series(y="<expr>")` (Generate series), the expression through `convert_expression` | translated |
| the same under a smaller sample | `data.with_series(y="<expr>", within=Sample(...))` (Generate series → Within): only the sample's rows, the others kept; a formula using its own series period by period, as EViews | translated (built in core, decision 5) |
| `frml y = <expr>` | `data.derive(y="<expr>")` (a derived series, *Keep updated*) | translated |
| `series y` (declared, all NA); `y = na` | `data.with_series(y=None)`, with `within=` under a sample | translated (built in core); `na` inside a larger expression: **gap** |
| `group g a b c`, `g.add d` | `g = data.series("a", "b", "c")`: a `Selection` | translated |
| `x.label`, `x.displayname`, `x.setattr(units)` *(verify the forms)* | `data.with_info("x", label=..., units=...)` | translated |
| `rename`, `delete` of series | `data.rename(...)`, `data.drop(...)` | translated |
| `copy(c=a) q\gdp a\gdp` (between pages) *(verify the `c=` letters)* | `Dataset.to_freq(freq, how=...)` (Convert frequency) | translated where the method maps; else left out with it |
| `scalar s = 0.3`, `!x = 1` | a Python variable | translated (code, not a step) |
| `scalar s = @mean(x)` and other sample statistics | `data.eval("mean(x)")` (a group-wise function, the same value in every row), or `data.describe` | translated with a note where the function exists; else left out, naming `describe` |
| `%s = "abc"`, `string s = "abc"` | a Python string | translated |
| `vector`, `matrix`, `sym`, `coef` declared or filled | `Matrix`, `Matrix.zeros` (Model → Matrix) | translated where the operation exists in `Matrix` |
| expression functions in `convert_expression` (`d`, `dlog`, `@pch`, `@movav`, `@recode`, the distributions ...) | the same functions in pymetrix's syntax | translated |
| `@trend`, `@seas`, `@elem` in series | `eviews_series` makes them from the dates | translated |
| `@year`, `@quarter`, `@month`, `@date`, `@obsnum`, `@round`, `@floor`, `@nan`, `@isna`, `nrnd`, `rnd`, `@iif` | none in pymetrix's expressions | **gap**, each listed by name. The phase starts with an audit of `pymetrix.expr` against the EViews functions found in the validation programs |

### Phase 2: estimation, tests and models

A list specification `y c x y(-1)` becomes `y ~ x + y[-1]` (`- 1` without `c`), as `read_eviews` already does. An
expression specification becomes an explicit-form equation with parameters `c_1`, `c_2`, ... The report keeps a
coefficient map, EViews's `c(i)` against pymetrix's term or parameter. Later lines need it (`@coefs(2)`,
`wald c(2)=0`).

| EViews | pymetrix | Status |
| --- | --- | --- |
| `equation eq.ls y c x`, `eq.ls ...`, `ls ...` | `eq = Equation("y ~ x").fit(data)` (Estimate equation) | translated |
| `ls(cov=white)`, `ls(cov=hac)`; older `ls(h)`, `ls(n)` *(verify the letters by version)* | `method=OLS(cov="white")`, `OLS(cov="hac")`. `white` is HC1, EViews's White (`docs/design/definitions.md`) | translated; HAC with a note on the bandwidth defaults |
| `ar(1)`, `ma(1)` terms | `Equation(..., errors=ARMA(ar=1))` with `method=ML()` | translated with a note: EViews's default ARMA estimator differs by version *(verify)*, so estimates may differ |
| `tsls y c x1 x2 @ c z1 z2` | `IV(endog=[regressors not among the instruments], instruments=[...])`; also `GMM`, `LIML` for `gmm`, `liml` | translated |
| `arch(1,1) y c x`, with `tdist` and similar options *(verify)* | `Equation(..., variance=GARCH(arch=1, garch=1))`, `method=ML()`; `EGARCH`, `TARCH` | translated where the variance model maps |
| `binary(d=l)`, `binary(d=n)` *(verify)* | `family=Logit()`, `Probit()` | translated; the extreme-value link (`d=x`): **gap** |
| `ordered`, `censored`, `count`, `qreg`, `heckit`, `ardl` | `OrderedProbit` or `OrderedLogit`, `Tobit`, `Poisson` or `NegativeBinomial`, `Quantile`, `Heckman`, `ARDL` | translated where options map; each option not mapped is listed |
| `stepls`, `breakls`, `threshold`, `glm`, `switchreg`, `nls` options | partly `bai_perron`, `ThresholdAR`, `MarkovSwitching` | left out, naming them; phase 2b if wanted |
| `var v.ls 1 4 y1 y2 @ c x`, `v.ec` | `VAR(["y1", "y2"], lags=4, exog=["x"])`, `VECM` | translated |
| `system s`, `s.sur`, `s.3sls`, `s.fiml` | `System([...])` with `SUR`, `ThreeSLS`, `FIML` | translated |
| `eq.wald c(2)=0, c(3)=0` | `eq.wald("x = 0, z = 0")`, through the coefficient map | translated. Wald refers only to plain names and `const` (`c[-1]` is refused), so a specification whose restrictions name a lag or a function is fitted in explicit form with `c_i` parameters |
| `eq.auto(4)`, `eq.hettest(type=...)`, `eq.archtest`, `eq.reset(2)`, `eq.chow 1990q1` *(verify the forms)* | `eq.test("breusch_godfrey", lags=4)`, `"breusch_pagan"`, `"white"`, `"arch"`, `"reset"`, `"chow"` | translated |
| `eq.hist`, `eq.rls(q)` *(verify)* | `eq.test("jarque_bera")`, `eq.recursive_residuals()` (CUSUM) | translated |
| `eq.testdrop x` | `eq.wald("x = 0")`, the same F statistic under least squares with classic errors | translated with a note |
| `eq.testadd z` (omitted variables) | none | **gap** |
| `x.uroot(adf, ...)` *(verify the options)* | `data.test("adf", "x", ...)`; also `kpss`, `phillips_perron`, `dfgls`, `ers_point_optimal`, `ng_perron` | translated |
| `g.coint(...)` | `data.johansen([...])` | translated |
| `eq.forecast yf`, `eq.fit yf` | `eq.forecast(data, span, dynamic=True)`, `dynamic=False`, then the forecast added to the Dataset (Add to dataset) | translated |
| `eq.makeresids r` | `data.with_series(r=eq.resid)` (Add to dataset) | translated |
| `eq.@coefs(i)`, `@stderrs`, `@r2`, `@se`, `@dw`, `@regobs` | `eq.coefficients()`, `eq.statistics()`, through the coefficient map | translated (code) |
| `@aic`, `@schwarz`, `@hq` | per observation, as EViews reports them | translated with a note: pymetrix's criteria are totals (`definitions.md`) |
| `model m`, `m.append`, `m.merge eq`, `@identity`, `@add`, `@innov` | collected into one text and read by `read_eviews(text, data=..., params=...)` | translated: the core reader's subset, with its refusals passed into the report at their lines |
| `m.solve`, `m.solveopt(...)` *(verify the option letters)* | `m.model.solve(data, span, solver=Newton() or GaussSeidel())`; stochastic options to `draws=` | translated where the options map |
| `m.addassign`, `m.addinit` | `Model.add_factors(data, span)` (Model → Add factors) | translated |
| `m.scenario "s1"`, `m.override x`, `m.exclude y`, the `_1` alias series | `Scenario(set=..., fix=...)`, `solve(..., scenario=...)`, then Compare scenarios | translated where the alias series are in the data |

### Phase 3: control flow

| EViews | Script | Steps mode |
| --- | --- | --- |
| `for !i = 1 to n [step k] ... next` | `for i in range(1, n + 1, k):` | the loop run in Python; each statement inside a step |
| `for %s a b c ... next` | `for s in ["a", "b", "c"]:` | as above |
| `while ... wend` | `while ...:` | as above, with a cap on iterations that the user can change |
| `if ... then ... else ... endif` | `if ...: ... else: ...` | decided as it runs |
| `{%s}`, `{!i}` in names | f-strings: `data.with_series(**{f"y_{s}": f"log(x_{s})"})` | the name made, then a plain step |
| `subroutine f(scalar !a, string %b) ... endsub`, `call f(...)` | `def f(a, b): ...` | the body run at each call |
| `subroutine local` | a function with no access to the caller's variables | the same |
| `include file.prg` | the included file translated with it, as a function or in line | in line; its rows in the report under its path |
| `exitloop`, `return`, `stop` | `break`, `return`, an exception that ends the script | the same |
| `%0`, `%1` (program arguments), `@equaloption` *(verify)* | the script's parameters | asked for in the dialog |
| `@wlookup`, `@wcount`, `@word` | `data.names` with a pattern, `len`, indexing | evaluated against the Dataset |
| `@isobject("x")` | `"x" in data.names` or the session's objects | evaluated |

### Phase 4: left in the report

These are listed, each with a reason and the pymetrix equivalent where there is one.

| EViews | Reason | pymetrix equivalent |
| --- | --- | --- |
| `graph`, `x.line`, `x.bar`, `x.scat`, `freeze(g) x.line`, `g.addtext`, `g.setelem`, `g.axis`, other graph formatting | graphs are not translated: their formatting has no one-to-one mapping | `data.plot(...)`, `Figure`, Data → Plot series |
| `table`, `t(1, 1) = ...`, `t.setformat` | tables are made by results, not cell by cell | `Table`, a result's `summary()` |
| `spool`, `sp.append` | output collections | `Report`, File → Export report |
| `freeze(name) view` | a view becomes an object in EViews; in pymetrix, results already are objects | the result itself, by name |
| `show`, `print`, `output`, `close` | display only | the Results area; `print(eq.summary())` in a script |
| `logmode`, `tic`, `toc`, `statusline`, `setmaxerrs`, `pon`, `poff`, `cd` | program settings and display | none needed |
| `@uiprompt`, `@uidialog`, `@uiedit` *(verify)* | interactive prompts | the import dialog asks for the arguments |
| add-ins and user-defined commands | another program's code | none; named in the reason |
| `shell`, `spawn`, `xopen`, `xrun` | external programs | none |

## 3. In the app

The plugin follows the extension API (1.10) and the existing first-party plugins.

- **File → Import EViews…** A dialog, `@command(menu="File/Import EViews...")`, so it lands at the end of the File
  menu. It has three choices. *Workfile bundle* takes a `.evbundle` file and the pages to read. *Program* takes a
  `.prg`, with an optional bundle for its `wfopen`. *Run EViews for me* (route 2, Windows only) takes a `.wf1`.
  *Run EViews for me* is not built while route 2 is deferred. The dialog also offers *Get the export program*,
  marked experimental. For a bundle, the dialog's step is one recorded command, and
  its code is one `Dataset.read(...)` line per page. Its ID is fixed in `command_ids.json`.
- **The translation report**, a docked panel (`AppPanel("EViews translation", create, area="bottom")`):
  - a table, one row per line: line number, status (with a colour and a text label, so it does not rely on colour),
    the EViews text, the Python, and the reason;
  - the `.prg` and the script side by side. Selecting a row scrolls both to that line;
  - filters by status, and a count of each;
  - **Copy** (the selected rows, or the whole report, as text) and **Copy script**;
  - **Run as steps**, which runs the steps one at a time through `AppInterface.run`, so each is recorded and can
    be undone. It stops at the first failure and marks that row.

  The panel uses only `AppInterface`, never the window's own parts.
- **Help.** A help topic for the plugin (`add_help_topic`), linked from the dialog. It explains the export program
  and the statuses. The general "Coming from EViews / IRIS" page is separate.
- **Help → Plugins** lists the plugin. When `pywin32` is missing, it lists route 2 as unavailable, with
  `pip install pymetrix-eviews[com]`.

Package layout, as `pymetrix-psets` and `pymetrix-labbook`:

```
packages/pymetrix-eviews/
  pyproject.toml          entry points: pymetrix.plugins, pymetrix.io, pymetrix.commands, pymetrix.panels
  src/pymetrix_eviews/
    __init__.py           PLUGIN = PluginInfo("pymetrix-eviews", ...); register()
    bundle.py             read_bundle, the .evbundle reader, the JSON Schema
    export/pymetrix_export.prg
    com.py                route 2
    program/              parse.py, ir.py, data.py, estimation.py, control.py, left.py, script.py, steps.py
    report.py             ProgramTranslation, the report rows
    app.py                the dialog and the panel
  tests/                  fixtures/ (bundles by hand, programs), test_bundle.py, test_program_*.py, test_panel.py
```

The plugin needs a few things from core that are now private: the statement splitter that keeps offsets
(`_blank`, `_statements`), the expression parser behind `convert_expression`, and the `Diagnostic` type. They would
be exposed under public names through `pymetrix.extension`, as an additive API (decision 4). No behaviour in
core changes. Built as API 1.10 (1.9 was already taken): `eviews_blank_comments`, `eviews_statements`,
`parse_eviews_expression` and its node classes, `render_eviews_expression`, `EViewsExpressionError`, `Diagnostic` and
`line_and_column` ([plugins.md](plugins.md)).

## 4. Validation

- **Bundles by hand.** The fixtures are written by hand to the format above, independently of the importer:
  quarterly and annual pages in one workfile, a panel page, an undated page, NA and empty fields, alpha series
  with an empty value, a value map, a label with a quote and a backslash, a matrix, an equation with its EViews
  estimates, and a model. The tests read each one and check every value and every metadata field. A round trip
  writes a Dataset back out as a bundle and reads it again, and requires identical data and metadata. Malformed
  fixtures (no manifest, a newer `format_version`, a CSV with a missing column) must be refused with their place.
- **Re-estimation against EViews's own numbers.** For every equation in a bundle, the translated equation is
  fitted and its coefficients and standard errors are compared with the `@coefs` and `@stderrs` EViews exported.
  The report says *matches* (to 1e-6 relative) or gives the difference. This check works on any user's workfile.
- **Programs.** The Brillet book programs and the public FRB/US programs (`example1.prg` to `example4.prg`,
  `pings.prg`, `stochsim.prg`, `ocpolicy.prg`, and the master library they include) are already used for the model
  reader (`docs/validation.md`; `PYMETRIX_BRILLET`, `PYMETRIX_FRBUS`). For each program:
  - every line has a status, and a test fixes the counts, so a change shows;
  - the translated script and the steps give equal objects;
  - where EViews output is published, the translated results are compared with it: Brillet's printed estimates
    (to their printed digits), and the FRB/US responses in Brayton's figures, which `tests/test_frbus_example.py`
    already checks for the hand-written equivalent. The translated run must give that test's numbers (1e-8).
- **Where results are recorded.** Each comparison gets a row in `docs/validation.md`. Every gap gets an item in
  `docs/known-gaps.md`.
- **The export program itself** can only be run in EViews. Before each release someone with an EViews licence
  runs it on the fixture workfiles, and the bundle must match the hand-made fixtures (decision 7). The project has
  no such person now, so the program ships marked experimental, and the import says so.

## 5. Size, by phase

Rough figures for one developer who knows the codebase. "Lines" are Python and EViews code, without tests. Tests
add about as much again.

| Phase | Lines | Days |
| --- | --- | --- |
| Data route 1: the export program, the bundle reader, the schema, the fixtures | 300 `.prg`, 600 Python | 8–12, plus 2 in EViews |
| Data route 2: COM | 200 | 3–5, Windows only, outside CI |
| Programs, phase 1 (the parser, the report, data work) | 1,200 | 10–15 |
| Programs, phase 2 (estimation, tests, models) | 1,500 | 15–20 |
| Programs, phase 3 (control flow, steps mode) | 800 | 8–12 |
| Programs, phase 4 (the catalogue of what is left) | 300 | 3–5 |
| The app (dialog, panel, help) | 700 | 6–8 |
| Validation on the Brillet and FRB/US programs | 300 | 5–10 |
| **Total** | **about 5,900** | **about 60–90** |

Phases 1 and 2 of programs depend on data route 1. Phase 3 can start after phase 1. The core additions (decisions
4 and 5) are each a few days and come first.

## 6. Out of scope

- Reading `.wf1`, `.wf2` or EViews databases (`.edb`) directly.
- An EViews-style command line or console in the app.
- Running a translated program inside EViews, or comparing with EViews automatically, except through the estimates
  carried in the bundle.
- Translating pymetrix back into EViews programs. `export_model` already writes models as EViews text.
- Graph and table formatting, spools, add-ins and EViews's user interface commands (phase 4 lists them).
- Every EViews estimator and option. Phase 2 covers the common ones; the rest are listed.
- Intraday and multi-year frequencies.
- The "Coming from EViews / IRIS" help page, which is separate work.

## 7. Risks

- **The export program cannot be tested in CI.** No EViews is available to the project. Mitigation: hand-made
  fixtures, a schema, and a release check by someone with a licence.
- **EViews versions.** Data members and options changed across versions (`@command`, `@attr`, the `ls` options,
  COM). The export program checks the version (`@ver` *(verify)*), falls back where it can, and records what it
  could not read in `export.log` and `skipped`.
- **Precision.** If EViews's text export rounds, comparisons fail and estimates move. The program must write full
  precision *(verify the option)*. A fixture test catches a CSV with too few digits.
- **Text in EViews strings.** Labels with quotes, backslashes or non-ASCII characters can break the JSON. Keeping
  long texts in files and parsing strictly limits the damage. A TSV attributes file is the fallback (decision 6).
- **Assigning within a sample.** It is common in EViews programs, and was the largest single gap. Core now has it
  (`with_series(..., within=...)`, decision 5); the translator must still pass each line's sample, and keep EViews's
  period-by-period order where a line uses its own series.
- **Different defaults.** HAC bandwidths, information criteria per observation, and the ARMA estimator differ. A
  translated line that runs but differs is worse than one left out. Each known difference is a "translated with a
  note" reason, and the re-estimation check catches the rest.
- **Two outputs drifting apart.** The script and steps mode come from one representation, and a test requires
  equal objects.
- **Scope creep.** EViews's command set is large. The phases list what is in; everything else is reported, not
  built.

## Decisions

Decided 24 November 2026.

1. **The plan:** yes. Order of work: the core additions (4 and 5), then the bundle reader with hand-made fixtures,
   then programs phases 1 and 2. Programs are the main deliverable, since they need no EViews to build or test.
2. **Script and steps:** both, from one parse. The app runs steps and attaches the script and the report to the
   project.
3. **Replay needs the plugin:** accepted, as with PSETS, with *Save the data as Parquet* on the import to cut the
   dependency.
4. **Core helpers public** (extension API 1.9, additive): yes. Built as 1.10, since 1.9 had been used meanwhile.
5. **Two core additions** (an empty dated Dataset; assignment within a sample): yes.
6. **Metadata format:** JSON, with the TSV of attributes as the fallback.
7. **The EViews check:** the project has no EViews licence. The export program ships marked *experimental*, tested
   only against the hand-made fixtures and the schema, and the import says so. A workfile saved from EViews as CSV
   or Excel is the supported way in until someone with a licence has run the program on the two most recent major
   versions; then the label comes off.
8. **Route 2 (COM):** deferred. It cannot be tested without EViews.
