# R scripts into pymetrix: the `pymetrix-r` plugin

Design note, 28 November 2026 · a proposal for the owner to approve or change; **phase 0 was approved and is
done** (`pymetrix.translation` in core, extension API 1.11, section 1); the R front end is not built · builds on
the EViews program translator ([eviews-plugin.md](eviews-plugin.md), section 2, and
[`packages/pymetrix-eviews`](../../packages/pymetrix-eviews/README.md)), the core expression language (with
`round`, `iif`, `isna`, `fillna` and `log10` since 0.1.0a21), the canonical definitions
([definitions.md](definitions.md)) and the extension API ([plugins.md](plugins.md), `pymetrix.plugins.API` 1.10)

## The question

Many economists who come to pymetrix bring R scripts: data cleaning in dplyr, regressions with `lm`, `fixest` or
`plm`, robust standard errors from `sandwich`, unit-root tests, a VAR. The EViews plugin already turns a program
into a readable pymetrix script, a report with a row per line, and the same work as recorded steps. How much of
that machinery serves R too, and what does an R front end need?

## Short answer

- **Extract the translator's framework into core, then add an R front end as a plugin, `pymetrix-r`.** About half
  of the EViews translator is not about EViews: the intermediate representation, the report, the script writer,
  steps mode, the translation panel and the tests that tie them together. It moves to a core module,
  `pymetrix.translation`, exposed through `pymetrix.extension` (API 1.11, additive). The EViews plugin then uses it,
  with no change in what it does. The R plugin mirrors `pymetrix-eviews`.
- **R parsed by a parser written for the plugin, in Python, checked against R's own parser.** Translation needs no
  R installed. R's parser (`getParseData`) is the reference in a CI workflow, as rjd3tramoseats is for
  TRAMO-SEATS. tree-sitter's R grammar would be a dependency with native code fetched at run time; rpy2 needs R
  and is GPL.
- **Phase 1 covers the common applied subset:** data frames with dplyr's verbs and base subsetting; `lm`, `glm`,
  `fixest::feols`, `plm`, `ivreg`; `sandwich` and `lmtest`; `ts`, `zoo` and `xts`; `urca` and `tseries` tests;
  `vars::VAR`; `forecast`; loops and functions; `library()` lines. Every line gets a status, as in EViews:
  translated, translated with a note, left out, or a gap, each with its reason.
- **R's defaults differ from pymetrix's in places that change numbers:** `sandwich::vcovHC` is HC3, R's `round`
  rounds half to even, factors sort numbers as numbers, `stats::lag` shifts the other way. A translated line that
  runs but gives other numbers is worse than one left out, so each known difference is a note, or a gap until core
  has the equivalent. Three small core additions would close the commonest (section 6).
- **Validation:** R's numbers recorded from a pinned R and pinned packages, committed so ordinary CI needs no R, and
  a workflow that runs R live, as `jdemetra-reference.yml` does.

## Conventions in this note

- *(verify)* marks R behaviour or a package default that must be checked against the package's documentation, at
  the version the validation workflow pins, before it is relied on. Package defaults change between versions.
- Statuses, as the EViews report has them: **translated**; **translated with a note** (runs, and the note says
  what to check, usually a default that differs); **left out** (display, settings, I/O with no pymetrix
  equivalent needed, with the reason and the equivalent where there is one); **gap** (pymetrix has no equivalent
  yet: reported as left out, and listed in `docs/known-gaps.md`); and **error** (the line does not parse).
- The plugin is written from R's and the packages' public documentation. No package source is copied, and
  GPL-licensed scripts used as test inputs stay outside the repository, read from where R installs them.

## 1. The EViews translator: what is EViews and what is a framework

The EViews translator is `packages/pymetrix-eviews/src/pymetrix_eviews/program/`, about 5,900 lines, with the
core reader's expression conversion (`src/pymetrix/models/eviews.py`, made public as `parse_eviews_expression`,
`render_eviews_expression` and the rest in API 1.10).

| File | Lines | What it does | EViews-specific? |
| --- | --- | --- | --- |
| `ir.py` | 395 | The intermediate representation: `Text` (an expression with the program's variables in it, an f-string in the script, their values in steps), `Step` (a recorded call: function ID, receiver, output, arguments), `Cmd`, `Code`, `New`, `Attr`, `RefTo`, `Py`, and the control-flow ops `For`, `While`, `If`, `Def`, `Invoke`, `Jump` | **No.** Two traces: `Spread` (EViews's `{%names}` words) and `Def.returns`, which models EViews's arguments passed by reference; R functions return a value |
| `report.py` | 155 | `ReportRow` (line, status, source text, Python, reason, file), `TranslationReport` (counts, `summary()` as a `Table`, `to_text`), `ProgramTranslation` (`script`, `report`, `run`), the four statuses | **No**, apart from the column named "EViews" and the field `eviews` |
| `script.py` | 258 | The readable script: each source line a comment above what it became; blocks as Python blocks; functions first; included files in line; script line numbers for the panel | **No**, apart from the comment's form and the header's words |
| `steps.py` | 234 | Steps mode: each `Step` a `make_call`, control flow run in Python, the `while` cap, `StepFailed` with its line; the script and report attached to the project | **No**, apart from `RECORDED_BY = "eviews:program"` and the attachment key |
| `left.py` | 89 | The catalogue of what is left out and why (graphs, tables, display, settings, external programs), gap functions, `left_reason` | the catalogue is EViews's; `left_reason` and the pattern (a table of reasons, each with pymetrix's equivalent) are not |
| `parse.py` | 124 | Statements, options in brackets, `obj.method`, assignments, through the core reader's splitter | **Yes** |
| `data.py` | 1,479 | Phase 1 and the `Translator`: pages, samples, series, scalars and strings, names; `Outcome`; the bookkeeping of what is known before the program runs (`unset`, `forget`) | mostly; the name allocator (Python keywords and pymetrix names get `_v`, clashes a suffix) and the status mechanics are not |
| `estimation.py`, `models.py` | 1,595 | Phase 2: EViews's estimation commands and options, the coefficient map, models | **Yes** |
| `control.py` | 1,368 | Phase 3: EViews's blocks, `{%s}` names, program functions, subroutines | the block walker's pattern carries over; the words do not |
| `app.py` | 374 | The dialog and the "EViews translation" panel: a row per line, status filters, the source and the script side by side, Copy, Run as steps | the panel is not, apart from its labels |

The validation approach is also a framework:

- every line with text has exactly one row, a test checks it, and every left-out or error row has a reason;
- the count of each status is fixed per fixture program, so any change in the translation shows;
- the script and steps mode give equal objects, from one representation;
- results are compared with numbers the source program's own software produced (the bundle's `@coefs`, the
  published FRB/US responses).

### Proposal: `pymetrix.translation` in core

| Moves to core | Changes on the way |
| --- | --- |
| `ir.py` whole | `Def` gains `result`, the value a function returns (R's last expression or `return(x)`); `Spread` stays, as "the words of a string" |
| `report.py` | `ReportRow.source` replaces `eviews` (the EViews plugin keeps `eviews` as a property); `TranslationReport` takes the language's name for its column |
| `script.py` | the comment's form and the header come from the front end |
| `steps.py` | `recorded_by` and the attachment key are parameters |
| `left_reason`, the name allocator, `Outcome` and its statuses | none |
| the translation panel | one panel class; each front end registers an instance ("EViews translation", "R translation") with its attachment key |
| test helpers in `pymetrix.testing` | `assert_rows_cover(translation, source)`, `assert_steps_equal_script(translation, ...)`, `status_counts(translation)` |

What stays in each plugin: the parser, the statement handlers, the catalogue of what is left out, the
language's control-flow words, and its help.

**Why core, not a shared plugin.** Plugins depend on pymetrix and nothing else, and this is the same step the
EViews work took for the reader's helpers (API 1.10, decision 4 of [eviews-plugin.md](eviews-plugin.md)). A third
package would add a version to keep in step for about 1,200 lines. A Stata front end later would use the same
module.

**Phase 0 checks it:** the EViews plugin moves onto `pymetrix.translation`, and its seven test files pass
unchanged, the FRB/US status counts included.

**Phase 0: done.** `src/pymetrix/translation/` holds the IR (`ir.py`), the report and `Translation` (`report.py`),
`Source`, `Statement`, `Outcome`, `Untranslated` and `report_rows` (`outcome.py`), the name allocator (`names.py`),
the script writer (`script.py`), steps mode (`steps.py`), the panel (`panel.py`) and the test checks
(`testing.py`); extension API 1.11 makes it a stable module, with its main names in `pymetrix.extension`
([plugins.md](plugins.md)). The EViews plugin's modules keep their names as thin re-exports, and its test files
pass unchanged, the FRB/US validation included. Where the build differs from the table above:

- `Translation` is the base of `ProgramTranslation`; a front end gives `language`, `recorded_by`, `attachments`
  and `while_cap` as class attributes and the names its steps' Python needs through `step_names()` (the EViews
  plugin's `eviews_dates`). `ProgramTranslation` adds `pages`.
- The script's header stays with each front end; `render` takes the comment's form (`comment=`), and
  `pymetrix_imports` finds the names the header imports.
- The panel is one class made from a `PanelSpec` (the language, where the session's translations are, the
  translate dialog, its menu path), registered with `translation_panel(spec)`.
- The test checks are in `pymetrix.translation.testing`, not `pymetrix.testing` (which plugins' tests also use,
  but which imports more): `assert_rows_cover(translation)` reads the sources from the translation itself.
- `Def.result` is a Python expression worked out in the function's own variables when it ends or reaches a
  `return`; an R function is `local=True`, as its variables are its own. A `Def` has `returns` or `result`, not
  both.
- `report_rows` and `flatten` (the report's rows from the outcomes) moved too: they had nothing of EViews in them.
  `ops_of` (the block walker) stays in the EViews plugin, as the note says.
- `examples/translation_toy.py`, a toy language of two statements, is the worked example, run by
  `tests/test_extension_translation.py`.

## 2. Parsing R

R is a programming language, not a command language: a line is an expression, and the meaning comes from which
functions it calls. The parser must give the tree with each node's line and column, so the report and the panel
can point at the source.

| Option | What it is | For | Against |
| --- | --- | --- | --- |
| **A. A parser in the plugin** (recommended) | a Pratt parser for R's grammar, in Python, written from *R Language Definition* and `?Syntax` (operator precedence) | no dependency; works without R; positions as the report needs them; errors recovered at the next top-level line, so every line still gets a row | about 700 lines to write; R's grammar has corners (below) |
| B. R's own parser | `Rscript` with base R's `parse(keep.source = TRUE)` and `getParseData`, written as CSV, read in Python | exact, by definition | needs R installed, and the right version; many users who want to leave R have it, but CI and colleagues may not |
| C. tree-sitter | `tree-sitter` 0.27.0 (MIT, wheels) with r-lib's R grammar (MIT) | fast, error-tolerant, positions | the R grammar has no wheel of its own on PyPI (28 November 2026); `tree-sitter-language-pack` 1.28.0 has it, but downloads the compiled grammar into the user's cache on first use, so it fails offline and loads native code fetched at run time; the grammar may differ from R's in corners |
| D. rpy2 | R embedded in Python | full R | GPL-2.0-or-later; needs R; it runs R, and the plugin only needs to read it |

Checked for option C: `tree-sitter-language-pack` parsed a seven-line script (`|>` and `%>%`, a formula with
`factor()` and `:`, a `for` loop, `function`, `coeftest(m, vcov = vcovHC(...))`) without error, after fetching
`libtree_sitter_r.so` from the network.

**Recommendation: A, with B as the reference.** The validation workflow runs R's parser on every fixture and
compares token by token: each token's type, text, line and column, and the parent of each expression. The
recorded parse data is committed, so ordinary CI checks the parser without R. A user with R can also ask for
the check (`translate_r(..., check_parse="R")`).

The corners the parser must get right, each with a fixture:

- newlines end an expression unless a bracket is open or the line ends in a binary operator;
- `if (...) x` then `else` on the next line is an error at the top level and fine inside braces;
- `->` and `->>` assign to the right; `<<-` assigns in an enclosing scope;
- `|>` (R 4.1), its placeholder `_` (R 4.2), and `\(x)` for `function(x)` (R 4.1);
- raw strings `r"(...)"` (R 4.0), backticked names, `%op%` operators, `::` and `:::`;
- `~` as a unary and a binary operator; `?`; `-2^2` is −4 and `-1:2` is (−1):2;
- `x[i]`, `x[[i]]`, `x$y`, `x@y`, and empty arguments (`d[, 1]`).

## 3. What the plugin gives

```python
from pymetrix_r import translate_r

t = translate_r("analysis.R", data={"macro.csv": "C:/work/macro.csv"})
print(t.script)           # Python that uses pymetrix's public API, each R line as a comment above what it became
t.report.summary()        # a Table: line, status, R, Python, reason
out = t.run(session)      # steps mode: each data and estimation statement a recorded step
```

The rules carried over from EViews:

- **Names.** R allows dots (`log.gdp`); they become underscores (`log_gdp`), and the report lists the change. A
  Python keyword or pymetrix name gets `_v`, as in EViews. Backticked names are read as written.
- **Data frames are Datasets.** A data frame read from a file is a cross-section Dataset; with `ts`, `zoo`,
  `xts` or a `pdata.frame` it has a time axis, an entity axis or both. A column `d$x` is the series `x` of the
  Dataset `d`.
- **One step per verb.** A pipe `d %>% filter(...) %>% mutate(...)` is a chain of methods in the script and one
  recorded step per verb, so the History can undo each.
- **Values worked out when known.** A number or string known before the script runs (`n <- 4`) is put into the
  steps; one known only as it runs is used by name, as `{%s}` values are in EViews.

## 4. The phase 1 subset, and what each becomes

### Reading data and packages

| R | pymetrix | Status |
| --- | --- | --- |
| `library(dplyr)`, `require(plm)` | nothing: the packages this table covers | translated with a note: "pymetrix covers dplyr's verbs listed in the report" |
| `library(x)` for a package not covered | nothing | left out: "x is not translated: its functions are reported where they are used" |
| `install.packages(...)`, `setwd(...)`, `options(...)`, `Sys.setenv(...)` | none needed | left out |
| `read.csv(f)`, `readr::read_csv(f)`, `data.table::fread(f)` | `Dataset.read(path)`, the file given to the translation | translated; options mapped (`sep`, `na.strings`, `skip`) or named in the note |
| `readxl::read_excel(f, sheet=)` | `Dataset.read(path, sheet=...)` | translated |
| `haven::read_dta(f)`, `haven::read_sav(f)` | `Dataset.read(path)` (`.dta`, `.sav`) | translated; value labels with a note |
| `readRDS(f)`, `load(f)` | none | **gap**: the report gives the line of R that writes the data as Parquet or CSV (`arrow::write_parquet(d, "d.parquet")`) |
| `data(Grunfeld, package = "plm")` | none | **gap**, with the same advice |
| `write.csv(d, f)`, `saveRDS(...)` | `data.write(path)` | translated for CSV and Parquet; otherwise left out: "save the project" |
| `set.seed(n)` | seeds for the draws after it | translated with a note: pymetrix's generator, never R's numbers |
| `source("f.R")` | the file translated with it, in line | translated (phase 3), its rows under its name |

### Data frames: dplyr and base R

| R | pymetrix | Status |
| --- | --- | --- |
| `mutate(d, y = expr)`, `d$y <- expr`, `d[["y"]] <- expr`, `transform(d, y = expr)` | `d = d.with_series(y="expr")` (Generate series), the expression converted (next table) | translated |
| `filter(d, cond)`, `subset(d, cond)` | `d = d.select(where="cond")`, with `&`, `\|`, `!`, `%in%` and `between` as `and`, `or`, `not`, `in [...]` | translated |
| `d[d$x > 0, ]` (base) | `d.select(where="x > 0")` | translated with a note: base R keeps a row of NAs where the condition is NA; dplyr and pymetrix drop it |
| `select(d, a, b)`, `d[, c("a", "b")]` | `d.keep("a", "b")` | translated |
| `select(d, -a)`, `d$a <- NULL` | `d.drop("a")` | translated |
| `rename(d, new = old)` | `d.rename(old="new")` | translated |
| `group_by(g) %>% mutate(m = mean(x))` | `with_series(m="mean(x, by=g)")`: pymetrix's group-wise functions | translated with a note where R would give NA (below) |
| `group_by(g) %>% summarise(x = mean(x))` | `d.aggregate("g", how={"x": "mean"})` | translated when each summary is one function of one column, keeping its name; otherwise **gap** |
| `arrange(d, x)` | none needed: a Dataset's rows follow its axes | left out, saying why |
| `left_join`, `inner_join`, `full_join`, `merge(d1, d2)`, `bind_rows`, `rbind` | none | **gap**: core has no join (section 6) |
| `pivot_wider` over an entity, `pivot_longer` of columns named `<series>_<entity>` | `d.wide(...)`, `d.long(...)` | translated where the shape is a panel's; otherwise **gap** (`d.pivot` makes a summary table, not a Dataset) |
| `distinct`, `slice`, `count`, `n()`, `head(d, n)`, `d[1:10, ]` | `count(1, by=g)` for `n()`; rows by position through a series of row numbers first (`with_series(row="obsnum()")`, then `select(where="row <= 10")`: a row condition takes no functions of the rows) | translated with a note (R counts rows from 1, as `obsnum()` does) where they map; otherwise **gap** |
| `str(d)`, `summary(d)`, `head(d)`, `View(d)`, `print`, `cat`, `glimpse` | `d.describe()` for `summary(d)` | left out as display; `summary(d)` translated with a note |
| `data.table`'s `dt[i, j, by]` | none | **gap**: a different language inside R; listed for a later phase |

### Expressions inside `mutate`, `filter` and formulas

| R | pymetrix | Status |
| --- | --- | --- |
| `+ - * / ^`, `log`, `exp`, `sqrt`, `abs`, `log10`, `floor`, `ceiling` | the same; `ceil` | translated |
| `log(x, base = b)`, `log1p(x)` | `log(x)/log(b)`, `log(1 + x)` | translated |
| `ifelse(c, a, b)`, `dplyr::if_else(c, a, b)` | `iif(c, a, b)`, NA where `c` is NA, as R | translated |
| `case_when(c1 ~ a, c2 ~ b, TRUE ~ z)` | nested `iif` | translated |
| `is.na(x)`, `!is.na(x)` | `isna(x)`, `not isna(x)` | translated in `mutate`; in `filter`, a flag series first (conditions take no functions of missing values), with a note |
| `coalesce(x, v)`, `tidyr::replace_na(x, v)` | `fillna(x, v)` | translated |
| `NA`, `NA_real_` | `na` | translated |
| `round(x)`, `round(x, n)` | `round(x)`, `round(x, n)` | translated with a note: R rounds a half to the even digit (`round(2.5)` is 2), pymetrix away from zero (3) |
| `dplyr::lag(x)`, `dplyr::lead(x)`, `lag(x, n)` | `x[-1]`, `x[+1]`, `x[-n]` | translated with a note: dplyr shifts by row order (within groups), pymetrix by the time axis (within entity); equal when the rows are sorted by time |
| `stats::lag(x, k)` on a `ts` | `x[+k]` | translated with a note: `stats::lag` with positive k gives the *next* value (the series starts earlier) |
| `c(NA, diff(x))`, `diff(x, lag = s)` | `d(x)`, `d(x, 1, s)` | translated (a bare `diff(x)` is one shorter in R: **gap** unless padded) |
| `cumsum(x)` | `cumsum(x)` | translated with a note: R gives NA from the first missing value, pymetrix skips it |
| `mean(x, na.rm = TRUE)`, `sum`, `sd`, `min`, `max`, `median` | `mean(x)` ... (group-wise, the same value in every row; `sd` divides by n − 1 in both) | translated; without `na.rm = TRUE`, with a note: R gives NA when any value is missing |
| `scale(x)` | `(x - mean(x))/sd(x)` | translated |
| `zoo::rollmean(x, k, align = "right")`, `"center"` | `movavg(x, k)`, `cmovavg(x, k)` | translated; `rollapply` with another function: **gap** unless it maps to a `mov*` function |
| `rnorm(n)`, `runif(n)` | `normrnd(0, 1, seed=s)`, `unifrnd(0, 1, seed=s)` | translated with a note: pymetrix's draws, never R's |
| `as.numeric`, `as.integer`, `as.character` | none, or the series as it is | translated with a note where the type matters |
| `factor(x)` in a formula | `C(x)` | translated with a note on the reference level (section 5) |
| `paste`, `paste0`, `sprintf`, `substr`, `gsub` on columns | none for series | **gap** (on the script's own strings, phase 3 translates them as Python) |

### Time series: `ts`, `zoo`, `xts`

| R | pymetrix | Status |
| --- | --- | --- |
| `ts(d$y, start = c(1990, 1), frequency = 4)` (and 1, 12) | the Dataset given a quarterly time axis starting 1990Q1 | translated once core can date a Dataset from a start (section 6); until then **gap** unless the file has a date column |
| `ts(...)` with frequency 52, 7, 24 or other | none | **gap**: as EViews's semi-annual pages |
| `window(x, start = c(2000, 1), end = c(2019, 4))` | `with_sample(Sample("2000Q1", "2019Q4"))` (Set sample) | translated |
| `zoo(x, order.by = dates)`, `as.yearqtr`, `as.yearmon`, `xts(x, order.by = dates)` | a Dataset with a time axis of that frequency | translated where the dates are regular |
| `x["2000/2010"]` (xts) | `with_sample(Sample("2000", "2010"))` | translated |
| `na.locf`, `na.approx` | none | **gap** |
| `aggregate(x, nfrequency = 1, FUN = mean)`, `apply.yearly` | `Dataset.to_freq("Y", how="mean")` (Convert frequency) | translated where `FUN` maps |
| `decompose(x)`, `stl(x, s.window = ...)` | `Classical`, `STL` | translated with a note on their options |

### Estimation

A formula is converted term by term:

| R formula | pymetrix | Status |
| --- | --- | --- |
| `y ~ x1 + x2`, `y ~ x - 1`, `y ~ 0 + x` | `"y ~ x1 + x2"`, `"y ~ x - 1"` | translated |
| `log(y) ~ log(x)`, `I(x^2)` | `"log(y) ~ log(x)"`, `x^2` as a term | translated |
| `x1:x2` (both numeric) | `x1*x2`, one term | translated |
| `x1 * x2` (both numeric) | `x1 + x2 + x1*x2` | translated |
| `factor(g)`, a character or factor column | `C(g)` | translated with a note on levels |
| `factor(g):x`, `factor(g) * x` | none | **gap**: interactions with categorical terms |
| `poly(x, 2)`, `splines::ns(x)`, `bs(x)`, `offset(z)` | none | **gap** |
| `lag(x, 1)` inside a formula (dynlm's `L(x, 1)`) | `x[-1]` | translated with the notes above |

| R | pymetrix | Status |
| --- | --- | --- |
| `lm(f, data = d)`, `summary(lm(...))` | `Equation(f).fit(d)` (Estimate equation); classic standard errors, as R | translated |
| `lm(f, data = d, subset = cond)`, `weights = w` | `fit(d.select(where=...))`, `weights="w"` | translated |
| `lm(cbind(y1, y2) ~ x)` | none | **gap** |
| `glm(f, family = binomial)`, `binomial(link = "probit")` | `Equation(f, family=Logit())`, `Probit()` | translated |
| `glm(f, family = poisson)` | `family=Poisson()` | translated |
| `glm(f, family = gaussian)` | `Equation(f).fit(d)` | translated |
| `glm(..., family = Gamma)`, `inverse.gaussian`, `quasibinomial`, `quasipoisson` | none | **gap** |
| `MASS::glm.nb(f)` | `family=NegativeBinomial()` | translated with a note on the dispersion's parameterisation *(verify)* |
| `fixest::feols(y ~ x \| fe1 + fe2, data, cluster = ~g)` | `Equation("y ~ x + fe(fe1) + fe(fe2)").fit(d, method=OLS(cov=Cluster("g")))` | translated |
| `feols(...)` with no `vcov` or `cluster` | the same, with `Cluster("fe1")` | translated with a note: fixest's default covariance depends on its version (clustered by the first fixed effect in the versions this note's author knows) *(verify against the pinned version)*; its small-sample adjustment (`ssc`) may differ |
| `feols(y ~ x \| fe \| endog ~ z)` | `IV(endog=["endog"], instruments=["z"])` with the fixed effects | translated |
| `fixest::feglm`, `fepois` | `family=Logit()`, `Poisson()` with `fe(...)` | translated where pymetrix's fixed effects support the family *(verify)*; else **gap** |
| `plm(f, data = pd, model = "within")` (the default), `effect = "twoways"` | `fe(id)`, and `fe(t)` | translated |
| `plm(..., model = "random")` | `method=RandomEffects()` | translated with a note: plm's default variance components are Swamy and Arora's; linearmodels' may differ *(verify)* |
| `plm(..., model = "fd")`, `"between"`, `"pooling"` | `FirstDifference()`, `Between()`, `OLS()` | translated |
| `AER::ivreg(y ~ x + w \| z + w)`, `ivreg::ivreg(...)` | `method=IV(endog=["x"], instruments=["z"])`: the regressors not among the instruments are endogenous | translated |
| `forecast::Arima(y, order = c(p, d, q), seasonal = ...)`, `stats::arima(...)` | `SARIMA(order=(p, d, q), seasonal_order=...)` | translated with a note: R's default method is CSS-ML; pymetrix (statsmodels) uses exact ML; `include.mean` and `include.drift` mapped to `trend` |
| `forecast::auto.arima(y)` | none | **gap**: no automatic ARIMA; the note names `data.select_order` and `SARIMA` (section 6) |
| `forecast::ets(y)` with a model given (`model = "AAN"`) | `ExponentialSmoothing(...)` | translated; automatic choice: **gap** *(verify pymetrix's ETS options)* |
| `forecast(fit, h = 8)`, `predict(m, newdata)` | `fit.forecast(...)`, `fit.predict(...)` | translated |
| `vars::VAR(y, p = 2, type = "const")`, `"both"`, `"none"`, `exogen = x` | `VAR(["y1", "y2"], lags=2, ...)` with the trend and `exog` mapped | translated |
| `vars::VARselect(y, lag.max = 8)` | `VAR(...).select_order(data, max_lags=8)` | translated |
| `irf(v, impulse, response, n.ahead = 10)` | `fit.irf(10)` with `Cholesky()` | translated with a note: vars bootstraps bands by default (100 runs); pymetrix's bands are made when asked, and differ in their draws |
| `fevd(v)`, `causality(v, cause = "x")`, `serial.test(v)`, `normality.test(v)`, `arch.test(v)` | `fit.fevd()`, `granger`, `portmanteau`, `normality`, `arch` | translated |
| `vars::SVAR(...)` | `AB(...)` | translated where the restrictions map |

### Covariances and tests

| R | pymetrix | Status |
| --- | --- | --- |
| `coeftest(m, vcov = vcovHC(m, type = "HC1"))` | the equation fitted again with `OLS(cov="white")` (HC1) | translated |
| `vcovHC(m)` (default type HC3), `type = "HC0"`, `"HC2"`, `"HC3"` | none | **gap** until core has HC0, HC2 and HC3 (section 6) |
| `vcovCL(m, cluster = ~g)` | `OLS(cov=Cluster("g"))` | translated *(verify that vcovCL's defaults, type HC1 with its cluster adjustment, equal pymetrix's G/(G−1) · (n−1)/(n−k))* |
| `vcovCL(m, cluster = ~g + h)` | `Cluster("g", "h")` | translated |
| `NeweyWest(m, lag = L, prewhite = FALSE)` | `OLS(cov="hac", hac_lags=L)` | translated |
| `NeweyWest(m)` (automatic bandwidth, prewhitened), `vcovHAC(m)`, `kernHAC(m)` | none for regressions | **gap**: pymetrix's regressions take a lag count, not Newey and West's automatic bandwidth or prewhitening |
| `lmtest::bptest(m)` (studentised, Koenker) | `fit.test("breusch_pagan")` (Koenker's, as R's default) | translated |
| `bgtest(m, order = k)` | `fit.test("breusch_godfrey", lags=k)` | translated *(verify that the zero-filled lags match `fill = 0`)* |
| `dwtest(m)` | the Durbin-Watson statistic in `fit.statistics()` | translated with a note: no exact p-value; **gap** for it |
| `resettest(m, power = 2:3, type = "fitted")` | `fit.test("reset", ...)` | translated where the powers map |
| `car::linearHypothesis(m, "x = 0")`, `lmtest::waldtest` | `fit.test("wald", "x = 0")` | translated with a note: R reports an F test for `lm`; pymetrix's Wald test is χ² by default |
| `AIC(m)`, `BIC(m)`, `logLik(m)` | `fit.statistics()` | translated with a note: R counts the error variance (k + 1), so its AIC is 2 more and its BIC ln n more ([definitions.md](definitions.md)) |
| `tseries::adf.test(x)` | `data.test("adf", "x", trend="ct", lags=floor((n - 1)^(1/3)))` | translated with a note: tseries always includes a trend, and interpolates its p-value from a table, truncated to [0.01, 0.1] |
| `urca::ur.df(x, type = "none", lags = 1)` (its defaults), `"drift"`, `"trend"`, `selectlags = "AIC"` | `data.test("adf", "x", trend="n" / "c" / "ct", lags=k)`, or `lags="aic", max_lags=k` | translated with a note on critical values |
| `ur.kpss`, `kpss.test`, `ur.pp`, `pp.test`, `ur.ers`, `ur.za` | `"kpss"`, `"phillips_perron"`, `"dfgls"` or `"ers_point_optimal"`, `"zivot_andrews"` | translated where the options map |
| `urca::ca.jo(y, type = "trace", K = 2)` | `data.johansen([...], lags=K - 1)`: ca.jo's K is the VAR's order in levels *(verify that pymetrix's `lags` counts differences)* | translated with a note |

### Control flow and functions (phase 3)

| R | Script | Steps mode |
| --- | --- | --- |
| `for (i in 1:n)`, `for (v in c("a", "b"))`, `seq_len(n)`, `seq_along(x)` | `for i in range(1, n + 1):`, `for v in ["a", "b"]:` | the loop run in Python; each statement inside a step |
| `while (cond)`, `repeat { ... break }`, `next` | `while ...:`, `while True:`, `continue` | as above, with the `while` cap |
| `if (cond) ... else ...` | `if ...: ... else: ...` | decided as it runs |
| `f <- function(x, k = 2) { ... }` | `def f(x, k=2): ...`, returning the last expression or `return(...)` | the body run at each call |
| `d[[paste0("y_", i)]] <- ...`, `assign(paste0("m_", i), lm(...))`, `get(...)` | f-strings: `with_series(**{f"y_{i}": ...})`; `globals()[f"m_{i}"]` | the name made, then a plain step |
| `lapply(vars, function(v) lm(...))`, `purrr::map(...)` | a loop over the list | translated for a list of names and a body that maps; otherwise **gap** |
| `tryCatch(...)`, `stop(...)`, `warning(...)` | `try`/`except`, an exception | translated (code) |
| `x[1]`, `coef(m)[2]`, `x[-1]` (drop the first) | a value by name where R indexes coefficients (`coef(m)["x"]`); positions from 1 converted | translated with a note on positions |
| `coef(m)`, `se`, `summary(m)$r.squared`, `nobs(m)`, `residuals(m)`, `fitted(m)` | `fit.coefficients()`, `fit.statistics()`, `fit.resid`, `fit.fitted` | translated |

### Left in the report

| R | Reason | pymetrix equivalent |
| --- | --- | --- |
| `plot`, `lines`, `abline`, `hist`, `par`, `ggplot2`'s functions, `dev.off`, `png`, `pdf` | graphs are not translated: their formatting has no one-to-one mapping | `data.plot(...)`, `Figure`, Data → Plot series |
| `stargazer`, `modelsummary`, `texreg`, `knitr::kable`, `xtable` | tables made from results | `Table`, a result's `summary()`, `Report` |
| `print`, `cat`, `message`, `View`, `str`, `glimpse` | display only | the Results area; `print(...)` in a script |
| `Sys.time`, `proc.time`, `system.time`, `invisible` | display or timing | none needed |
| `system`, `system2`, `shell` | external programs | none |
| Shiny, S4 and R6 classes, environments | another kind of program | none |

## 5. R's defaults that differ from pymetrix's

| Area | R | pymetrix | In the translation |
| --- | --- | --- | --- |
| `lm` standard errors | classic, s² (X'X)⁻¹ | `cov="classic"` | the same |
| `lm` information criteria | `AIC` counts the error variance | `aic.total` counts the coefficients only | a note with the difference |
| `sandwich::vcovHC` | default type HC3 | HC1 (`white`) only | **gap** until HC3 exists |
| `NeweyWest` | automatic bandwidth, prewhitened | `hac_lags`, default ⌊4(n/100)^(2/9)⌋, not prewhitened | translated only with `lag =` and `prewhite = FALSE` |
| `glm` families | binomial with logit; Poisson and binomial with dispersion 1; quasi-families estimate it | `Logit`, `Probit`, `Poisson`, `NegativeBinomial` | quasi-families a gap |
| factor contrasts | treatment contrasts, the first level the reference; levels sorted, numbers as numbers (2 before 10), strings in the locale's order; `relevel` and `factor(levels = )` change it; ordered factors use polynomial contrasts | `C(x)`: dummies, the first level in string order left out (`"10"` before `"2"`) | translated with a note when the order can differ; `relevel`, `contrasts` and ordered factors: **gap** until `C(x, base=...)` exists |
| NA in arithmetic and summaries | propagates; `mean(x)` is NA unless `na.rm = TRUE`; `cumsum` NA from the first NA | group-wise statistics and `cumsum` skip missing values | a note on lines without `na.rm = TRUE` |
| NA in models | `na.omit`: rows with a missing value dropped | the same, recorded in provenance as `missing` | the same |
| NA in base subsetting | `d[d$x > 0, ]` keeps a row of NAs where `x` is NA | `select(where=...)` drops it | a note |
| indexing | from 1; a negative index drops | rows by `obsnum()`, from 1; coefficients by name | converted, with a note |
| `round` | a half to the even digit (`round(0.5)` is 0) | away from zero (`round(0.5)` is 1) | a note |
| `stats::lag` | `lag(x, 1)` is the next value | `x[-1]` is the previous value | `x[+k]`, with a note |
| `dplyr::lag` | by row order | by the time axis, within entity | a note |
| unit-root tests | tseries: constant and trend, table p-values; urca: `type = "none"`, one lag | `trend`, `lags` given explicitly | translated with the values written out |
| `vars::irf` | orthogonalised, bootstrapped bands | `Cholesky()`; bands when asked | a note |
| `fixest::feols` | clustered by the first fixed effect *(verify by version)* | `cov` given explicitly | written out, with a note |

## 6. Core additions that would close the commonest gaps

As the EViews work added an empty dated Dataset and assignment within a sample before the plugin, three additions
would turn frequent gaps into translated lines. Each is small and has its own check:

1. **HC0, HC2 and HC3 covariances** (`cov="hc0"`, `"hc2"`, `"hc3"`, with `"white"` staying HC1). `vcovHC`'s default
   is the commonest robust covariance in applied R. Checked against statsmodels' `cov_type="HC0"`, `"HC2"` and
   `"HC3"` called directly, and recorded `sandwich` output. Reference: MacKinnon and White (1985); Long and Ervin
   (2000) on why HC3 is R's default.
2. **A time axis from a start**, for data with no date column: `ts(x, start = c(1990, 1), frequency = 4)`.
   *(verify whether `Dataset.read` or `Dataset.from_frame` can already be given a first period.)*
3. **A reference level for categorical terms** (`C(g, base="b")`), and levels of numbers sorted as numbers.

Later, each with its own design: joins (`Dataset.join`, for `left_join` and `merge`), and an automatic ARIMA
(wrapping a library, such as statsforecast's AutoARIMA, which [plugin-split.md](plugin-split.md) lists as a
candidate).

## 7. Validation

- **The parser against R's.** The workflow runs R's `getParseData` on every fixture script and compares each
  token's type, text and position, and each expression's parent, with the plugin's parser. The parse data is
  committed (`tests/fixtures/r/parse/*.json`), so the ordinary test run checks the parser without R.
- **R's numbers, recorded.** `tests/fixtures/r/make_r_reference.R` runs each fixture script in R and writes the
  results the script makes, by name (coefficients, standard errors, statistics, p-values, responses, forecasts),
  with `sessionInfo()`'s R and package versions, as JSON. The JSON is committed: "another package's recorded output,
  with its version", CONTRIBUTING.md's second kind of check. The tests translate each fixture, run it in steps mode,
  and compare: 1e-8 for least squares, panel and IV estimates and their standard errors; 1e-5 for maximum
  likelihood and ARIMA; test statistics as their estimates. Bootstrap bands are not compared, only their presence.
- **R live, in CI.** A workflow, `r-reference.yml`, made from `jdemetra-reference.yml`: Ubuntu named, R named
  (4.6.2, as there), `r-lib/actions/setup-r`, the packages installed by `pak` at pinned versions, the reference
  script run, then pymetrix's comparison against R's fresh numbers. It runs on pull requests that touch the plugin,
  the core translation module or the workflow, weekly, and by hand; it uploads R's JSON, and a difference fails it.
  When a package's new version changes a number, the committed JSON is refreshed by hand, with a line in
  `docs/validation.md`. A user's machine needs no R.
- **The framework's own checks**, from the EViews plugin: every line with text has one row; left-out, gap and
  error rows have a reason; the status counts of each fixture are fixed; the script and steps give equal objects.
- **The notes stay true.** Each default difference in section 5 has a test on the recorded R output showing that
  the difference exists (`vcovHC`'s HC3 differs from HC1 on the fixture), so a note cannot outlive its reason.
- **Fixtures.**
  - Hand-written scripts covering every row of section 4, written for the project (BSD).
  - Public scripts read where R installs them, never copied into the repository: the demonstration scripts that
    ship with AER (`system.file("demo", package = "AER")`, GPL) *(verify their licence and what they cover)*, run
    in the workflow and skipped elsewhere, as the Brillet and FRB/US programs are skipped without their
    environment variables.
  - The data: pymetrix's own example datasets written out as CSV, so R and pymetrix read the same file.

Each comparison gets a row in `docs/validation.md`; each gap an item in `docs/known-gaps.md`.

## 8. In the app, and where it lives

- **File → Translate R script…** A dialog, as File → Translate EViews program… is: the `.R` file, the data files
  its `read.csv` lines read (each line's path matched to a file the user picks), and *Run as steps*. Its ID is
  fixed in `command_ids.json`.
- **The R translation panel:** the core panel class, registered by the plugin. One row per line with its status
  (colour and text), the R and the Python side by side, filters, Copy, Copy script, Run as steps.
- **Help:** a topic for the plugin, with the statuses and the defaults table, linked from the dialog; and "R" rows
  in `coming_from.py`'s table, which feeds a "Coming from R" help topic and the command palette's synonyms (typing
  `lm` offers Estimate equation).

```
packages/pymetrix-r/
  pyproject.toml          dependencies: pymetrix (API 1.11); entry points: pymetrix.plugins, pymetrix.commands,
                          pymetrix.panels; no R, no native code
  src/pymetrix_r/
    __init__.py           PLUGIN = PluginInfo("pymetrix-r", ...); translate_r; register()
    parse.py              the R parser: tokens, the Pratt parser, positions, error recovery
    names.py              R names to Python names
    expressions.py        R expressions to pymetrix expressions and conditions
    data.py               data frames, dplyr, base subsetting, reading files
    timeseries.py         ts, zoo, xts
    estimation.py         formulas; lm, glm, fixest, plm, ivreg; forecast, vars; covariances and tests
    control.py            loops, if, functions, source()
    left.py               the catalogue of what is left out, and why
    app.py                the dialog, the panel's registration, the help topic
  tests/                  fixtures/r/ (scripts, parse data, R's numbers), test_parse.py, test_*.py
```

## 9. Phases and size

Rough figures for one developer who knows the codebase. Tests add about as much again.

| Phase | What | Lines | Days |
| --- | --- | --- | --- |
| 0 (**done**) | `pymetrix.translation` in core (API 1.11); the EViews plugin moved onto it, its tests unchanged | 300 changed, 1,200 moved | 4–6 |
| Core additions | HC0/HC2/HC3; a time axis from a start; a reference level | 400 | 5–8 |
| 1 | The parser; data work (reading, dplyr, base, `ts`/`zoo`/`xts`, expressions); `library()` lines; the catalogue of what is left out | 2,000 | 15–20 |
| 2 | Estimation and tests: formulas, `lm`, `glm`, `feols`, `plm`, `ivreg`, `sandwich`, `lmtest`, `car`, `urca`, `tseries`, `vars`, `forecast` | 1,500 | 15–20 |
| 3 | Control flow and functions; `source()` | 800 | 8–12 |
| App and help | the dialog, the panel's registration, the help topics, `coming_from` rows | 400 | 4–6 |
| Validation | the R workflow, the reference script, fixtures, recorded numbers | 500 R and Python | 6–10 |
| **Total** | | **about 5,600** | **about 57–82** |

Phase 0 and the core additions come first. Phase 1 and the validation workflow go together, since the parser is
checked against R from the start.

## 10. Out of scope

- Running R, or comparing with R on the user's machine, except the optional parse check.
- Translating pymetrix back into R.
- Graphics (base and ggplot2) and table packages: phase-4-style rows in the report.
- `data.table` syntax, purrr beyond simple maps, S4, R6, environments and Shiny.
- Reading `.rds` and `.RData` files: pyreadr, the Python reader, is AGPL-3.0, and the report gives the one line of
  R that writes Parquet or CSV instead.
- R Markdown and Quarto files. Their R chunks could be extracted and translated in order; it is cheap and could be a
  later phase.
- Every package and option. The tables list what is in; everything else is reported, not built.

## 11. Risks

- **R's breadth.** Any line can call any function. Mitigation: only recognised calls translate; anything else is a
  row that says so, as in EViews.
- **A translated line that gives other numbers.** The defaults in section 5 are the known ones; the recorded R
  numbers catch the rest, on every fixture.
- **Defaults that change with package versions** (fixest, sandwich). The workflow pins versions and records them; a
  note names the version the translation assumes.
- **Parser drift.** R adds syntax now and then (`|>`, `\(x)`, raw strings). The comparison with R's parser on a
  pinned R catches a mismatch, and a new R version in the workflow shows new syntax.
- **Two outputs drifting apart.** As in EViews, one representation and a test that the script and steps agree.
- **The extraction breaks EViews.** Phase 0 is a refactor with the EViews tests as its guard; nothing in the
  EViews plugin's behaviour changes.

## 12. Open questions for the owner

1. **Where the shared framework lives** (decided: core, built in phase 0). Core (`pymetrix.translation`, API 1.11), a shared plugin, or copied into
   each plugin? *Recommendation:* core. It is small, has no dependencies, and plugins should depend on pymetrix
   only.
2. **The parser.** Written in the plugin (A), R required (B), or tree-sitter (C)? *Recommendation:* A, with R's
   parser as the reference in CI and an optional check for users who have R.
3. **The core additions in section 6.** *Recommendation:* HC0/HC2/HC3 and a time axis from a start now, before
   phase 1; the reference level with them if cheap; joins and an automatic ARIMA as separate designs later.
4. **A live comparison with R on the user's machine** (`compare_with_r=True` when `Rscript` is found)?
   *Recommendation:* not in phases 1 to 3. The recorded numbers and the CI workflow carry the validation; the
   optional parse check is enough at first.
5. **fixest's default covariance**, which depends on its version. *Recommendation:* always write the covariance out
   explicitly in the script, with a note naming the fixest version assumed, and a dialog option to say which
   version the script was written for.
6. **`data.table`, purrr and R Markdown.** *Recommendation:* not in phases 1 to 3; listed as gaps, and R Markdown
   chunks as a cheap later phase.
7. **Order against other work.** *Recommendation:* phase 0 (the extraction) soon, since it also tidies the EViews
   plugin; the R front end after the first reviewers say whether R or Stata users are the larger group.

## References

For the core additions (help topics cite them; the R packages compared against are named in
`docs/validation.md` with their versions, not cited):

- Long, J. S. and Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression
  model. The American Statistician 54(3), 217-224.
- MacKinnon, J. G. and White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with
  improved finite sample properties. Journal of Econometrics 29(3), 305-325.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for
  heteroskedasticity. Econometrica 48(4), 817-838. doi:10.2307/1912934

DOIs other than White's (reused from the help) are left out until `scripts/check_references.py` confirms them.
