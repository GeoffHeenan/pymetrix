# Exercise pack format (reference)

An exercise pack is a folder of guided estimation exercises that ships on its own: as a folder a user opens in the
app (Help → Exercises… → Open folder…), or inside a plugin. pymetrix reads it with `pymetrix.exercises.load_pack`,
which validates every file below and lists all the problems it finds in one `SpecificationError`. The format and the
`pymetrix.exercises` interface are **provisional before pymetrix 1.0**. How to write one is in the guide,
[Writing an exercise pack](guide.md#writing-an-exercise-pack).

```
<pack>/
  pack.toml
  data/                      optional
    <name>.parquet | <name>.csv
  models/                    optional
    <name>.mod | <name>.model | <name>.pymodel
  exercises/
    01-<slug>/
      exercise.md
      checks.toml            optional (an exercise without it has nothing to Check)
      solution.py            optional
```

All text files are UTF-8.

## pack.toml

```toml
[pack]
name = "intro-econometrics"            # required: lower case letters, digits, "-" and "_"
title = "Introductory econometrics"    # required
version = "1.0.0"                      # required: the pack's own version (any non-empty string)
description = "Ten exercises ..."      # required
requires_pymetrix = ">=0.1.0a18"       # required: a PEP 440 version specifier
audience = "first-year students"       # optional, free text: who it is for, shown in the app's list of packs
classification = "internal"            # optional, free text: shown in the app and by the CLI; never enforced

[data.wages]                           # optional, one table per CSV file of data/ (see data/)
entity = "id"                          # the column naming the rows: entities (cross-sections and panels)
time = "year"                          # the column of dates or periods (time series and panels)
freq = "Y"                             # "", "Y", "Y-JUN", "Q", "M", "W" or "D"

[models.nk_2019]                       # optional, one table per file of models/ to rename (see models/)
name = "nk"                            # the session name it is imported under (default: the file stem, nk_2019)
```

- Unknown keys in `[pack]`, and tables other than `[pack]`, `[data.<name>]` and `[models.<file stem>]`, are errors
  (they catch misspellings).
- `requires_pymetrix` is compared with pre-releases allowed (`>=0.1.0a18` admits `0.1.0a18`). A pack whose
  requirement this pymetrix does not meet still loads: `Pack.compatible` is False, the app shows why, and
  `pymetrix-exercises check` refuses it (exit code 2).
- `audience` is free text, shown in Help → Exercises… beside the pack (and in its details).
- `classification` is free text (`"internal"`, `"public"`, `"course use only"`...): informational only. The app
  shows it in the pack list and the Exercise panel's header, and `pymetrix-exercises check` and `list` print it.
  Nothing restricts a pack because of it.
- `[data.<name>]` must name a file of `data/`; its keys are `time`, `entity` and `freq`, as File → Load data… asks
  for them.
- `[models.<file stem>]` must name a file of `models/` by its stem (`[models.nk_2019]` for `models/nk_2019.model`);
  its one key, `name`, required, is the name the model is imported under in a session (a Python name: letters,
  digits and `_`, not starting with a digit, not a keyword). Two models may not end up with the same name.

## data/

Optional. Each `.parquet` or `.csv` file is a dataset, named by its file name without the extension
(`data/wages.csv` is `wages`). Files starting with `.` and `README`, `LICENSE`, `LICENCE` or `NOTICE` files are
ignored; any other file is an error, as are two files with the same name.

- A Parquet file written by pymetrix (`Dataset.write`) carries its own axes: nothing more is needed.
- A CSV file needs a `[data.<name>]` table in `pack.toml` naming its `time` and/or `entity` column (and `freq`
  where the periods need it); without one the pack does not load.

An exercise's `data` may also name one of pymetrix's example datasets (`Dataset.example`: `mroz`, `nile`,
`sample_macro`, ... and those plugins add); a file of the pack's `data/` with the same name comes first.

In code, `load_data(pack, name)` reads one (`pack` is a `Pack` or the pack's folder, so `load_data(PACK_DIR, "wages")`
in a solution).

## models/

Optional. Model files, each named by its file name without the extension (its **stem**), read by their extension:

| Extension | Syntax | Read with | Gives | In the app |
| --- | --- | --- | --- | --- |
| `.mod` | Dynare | `read_mod` | a `DynareModel` | Model → Import Dynare file… |
| `.model` | IRIS | `read_iris` | a `Template` | Model → Import IRIS file… |
| `.pymodel` | pymetrix's model syntax (`pymetrix.models.source`) | `ModelSource(text).build()` | a `Template` | Model → Model editor… (the file's text) |

Other files (but `.`-files and `README`/`LICENSE`/`NOTICE`) are errors, as are two models with the same name.

- **The session name.** A model is imported under its stem (`models/rbc.mod` as `rbc`) unless `pack.toml` gives it
  another name: `[models.nk_2019]` with `name = "nk"` imports `models/nk_2019.model` as `nk`, so the exercise's
  steps and checks can say `nk`. `Pack.model_name(stem)` gives it.
- In code: `load_model(pack, name)` (`pack` a `Pack` or its folder: `load_model(PACK_DIR, "rbc")` in a solution),
  or the reader itself on the file (`read_mod(PACK_DIR / "models" / "rbc.mod")`, which is what the import dialog
  records). `name` is the stem or the session name (`load_model(PACK_DIR, "nk")` and `load_model(PACK_DIR,
  "nk_2019")` read the same file; a stem is looked up first). A `.pymodel` is built under the session name.
- In the app: an exercise lists the models it uses in its front matter (`models: [rbc]`, by stem or session name);
  the Exercise panel shows each with an **Import…** button that opens its import dialog (above) filled in with the
  pack's file and the model's session name (`model_command(pack, name)` makes that step). The user runs it (it is
  then a recorded step like any other); nothing is imported unasked.

## exercises/

One folder per exercise, named `NN-<slug>`: a number (digits; it orders the exercises and must be unique), `-`, then
a slug of lower case letters, digits and `-` (`01-ols-mroz`). The folder name is the exercise's **id**; the CLI and
`Pack.exercise()` also take the slug alone (`ols-mroz`) or the number (`1`). Folders starting with `.` or `_` are
ignored; anything else that does not match is an error.

### exercise.md

~~~markdown
---
title: A wage equation with robust standard errors     (required)
level: beginner                                        (optional: beginner, intermediate or advanced)
minutes: 15                                            (optional, a whole number)
data: [mroz]                                           (optional: datasets loaded when the exercise starts)
models: [rbc]                                          (optional: models of models/ the exercise uses, by stem or session name)
---

# A wage equation with robust standard errors          (optional: dropped, the front matter's title is shown)

Introduction (optional): everything before the first step.

## Step 1: Fit the wage equation

Choose Estimate → Equation…, with Data: mroz, Formula: `lwage ~ educ + exper + expersq`, Method: OLS and
Save as: fit.

```python
fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)
```

> Hint: `lwage` is the log of the hourly wage.

> Hint: a second hint, shown after the first.

## Step 2: Make the standard errors robust

...

## What next

Optional closing sections: any "## " heading after the last step.
~~~

- **Front matter** between `---` lines: `key: value`, lists as `[a, b]` or `- item` lines. Keys other than `title`,
  `level`, `minutes`, `data` and `models` are errors.
  - `level` is one of `beginner`, `intermediate`, `advanced` (shown in the list and the panel's header).
  - `data`: each name must be a file of `data/` (by its stem) or an example dataset. When the user starts the
    exercise in the app, they are asked once; if they agree, each is loaded into the session as a `Dataset` named
    after the name (the file's stem): File → Load data… for a file of the pack (with its `[data.<name>]` options),
    File → Load example data… for an example dataset. These are recorded steps (History, the script; Undo takes
    them back), and an object of the same name is replaced (the question says so). If they decline, nothing loads,
    and the panel says the data are not loaded. `pymetrix-exercises check` loads them the same way, without asking.
  - `models`: each name must be a file of `models/`, by its stem or the session name `[models.<stem>]` gives it;
    the panel offers to import each under its session name (above).
- **Steps** are `## Step N: <title>` headings, numbered 1, 2, 3… in order, at least one. A `## Step` heading not in
  that form is an error. Other `## ` headings before step 1 belong to the introduction; after the last step they
  are the closing text, and no step may follow them. Headings inside fenced code are not headings.
- **Within a step:**
  - the instructions in the app (Markdown: menu paths, dialog fields as the dialogs label them);
  - at most one fenced block with the info string `python` (or `py`): the same step as code, shown by
    **Show the code**. It runs as typed code in the session's namespace (pymetrix's names, `np`, `pd`, the session's
    objects, so the datasets by name). Other fenced blocks stay in the instructions;
  - hints: a blockquote whose first line starts `> Hint:`; following `>` lines continue it; a new `> Hint:` line
    starts the next. Hints show one at a time, in order. Other blockquotes stay in the instructions.

### checks.toml

```toml
[[check]]
id = "robust-se"          # required: unique within the exercise
step = 2                  # required: the step it checks (1 to the number of steps)
object = "fit_white"      # required: a session object's name (a Python identifier)
value = "se.educ"         # required: a dotted path into the object (below)
expected = 0.013219       # required: a number, a string, or true/false
rel_tol = 0.001           # optional, numbers only, >= 0
abs_tol = 0.0001          # optional, numbers only, >= 0
message = "Choose Standard errors: White (HC1)."   # optional: shown when the check fails
```

Unknown keys are errors.

**Comparison.**

- number: the value must be a number (not true/false, not NaN), and `math.isclose(value, expected, rel_tol=,
  abs_tol=)` must hold. Only `abs_tol` given: `rel_tol` is 0. Neither given: `rel_tol = 1e-6`. Integers and floats
  compare alike (`expected = 428` matches `428.0`).
- string: the value must be a string equal to `expected` (case and spaces count).
- true/false: the value must be a boolean (NumPy's too) equal to `expected`.

A check whose object does not exist yet, or whose path leads nowhere, fails and says why.

**Value paths.** `value` is resolved part by part, by **attribute and key lookup only**:

1. a part starting with `__` is refused (when the pack loads), and so are empty parts;
2. on a mapping (a dict such as `stats`) or a pandas Series/DataFrame: the key (a column of a DataFrame, a label of
   a Series);
3. otherwise an attribute (properties count);
4. failing that, for an estimate (a result with `params`): `se` (also `bse`), `tvalues`, `pvalues`, mappings by
   parameter name read from the result's own coefficient table (`coefficients()`: the standard errors the fit
   chose, the t or z statistic, the p-value), and `nobs` (from `stats`, as a whole number when it is one);
5. failing that, a key (`params.educ` reads the parameter `educ`) or, for a list or array, a whole-number index
   (`ok.a.1`).

A part that is a whole number (digits, optionally after `-`) is tried as a key in this order: the **string** (`"1"`,
a dict's string key or a Series' string label), then the **int** (a label of a Series or a column of a DataFrame
with an integer index, such as one indexed by year or by person, or a dict's int key), then, for a list, tuple or
NumPy array only, the **position** (`-1` the last). A Series or DataFrame is never read by position: `mroz.wage.2`
is woman 2's wage, and a Series whose labels are strings has no `1` (use `values.1` for a position).

Nothing is ever called: a part that resolves to a method or function is an error (`summary`, `test`, `reject`).
Keys may contain spaces (`details.lags used`) but not dots.

`params.se` is the standard errors as a plain array (no names): use `se.<name>` for one by name.

**Paths on real results.** Each row below is checked by the test suite against the objects these lines make
(`tests/test_exercises.py`), with the value it gives (rounded here):

```python
sample_macro, nile = Dataset.example("sample_macro"), Dataset.example("nile")
fit = Equation("c ~ y + g").fit(sample_macro, method=OLS(cov="white"))   # Estimate → Equation…, White (HC1)
eg = sample_macro.test("engle_granger", "c, y")                           # Test → Run test…, engle_granger
wald = fit.wald("y = 0.7")                                                # a Wald test of a restriction
adf = nile.test("adf", "volume")                                          # Test → Run test…, adf
mroz = Dataset.example("mroz")                                            # a cross-section indexed by woman: 1 ... 753
irf = templates.new_keynesian().solve().irf(8)                            # Model → Analysis → Impulse responses…
```

<!-- paths:start -->
| Object | Path | Value | What it is |
| --- | --- | --- | --- |
| fit | `params.y` | 0.86904 | the coefficient on `y` |
| fit | `params.const` | -1.1349 | the constant |
| fit | `se.y` | 0.0079239 | its standard error (White, as chosen) |
| fit | `bse.y` | 0.0079239 | the same (statsmodels' name) |
| fit | `tvalues.y` | 109.672 | its t statistic |
| fit | `pvalues.g` | 3.4272e-27 | the p-value on `g` |
| fit | `nobs` | 100 | observations used (a whole number) |
| fit | `stats.nobs` | 100 | the same, from the fit statistics (100.0) |
| fit | `stats.r2` | 0.999959 | R² (`stats` also has `r2_adj`, `ssr`, `se_regression`, `loglike`, `aic`, `bic`, `hq`, `k`, `df_resid`) |
| fit | `stats.aic` | 133.674 | Akaike's criterion |
| fit | `method.cov` | white | the standard errors chosen (`classic`, `white`, `hac`) |
| eg | `statistic` | -7.3004 | the Engle-Granger statistic |
| eg | `p_value` | 1.6252e-09 | its p-value |
| eg | `critical_values.5%` | -3.39854 | the 5% critical value (also `1%`, `10%`) |
| wald | `statistic` | 455.07 | the Wald chi-squared statistic |
| wald | `p_value` | 5.6852e-101 | its p-value |
| wald | `df` | 1 | restrictions |
| wald | `details.F` | 455.07 | the F form |
| wald | `details.F p-value` | 2.0901e-38 | its p-value |
| adf | `statistic` | -4.04871 | the ADF statistic |
| adf | `p_value` | 0.00117589 | its p-value |
| adf | `details.lags used` | 1 | the lags chosen (AIC) |
| adf | `critical_values.5%` | -2.89152 | the 5% critical value |
| sample_macro | `y.2000Q1` | 244.735 | one value of a series, by its period |
| mroz | `wage.2` | 1.3889 | one value of a series with an integer index, by its label (woman 2), not its position |
| mroz | `hours.753` | 0 | the last woman's (label 753; there is no position 753) |
| irf | `values.2.0.2` | -0.57588 | a response by position, `values[horizon, response, shock]`: `y` (the first response) to `ei` (the third shock) at horizon 2 |
| irf | `paths.y <- ei.2` | -0.57588 | the same by name: the column `y <- ei` (response, then shock) of `paths`, at horizon 2 |
| irf | `paths.pi <- ei.0` | -0.42391 | inflation's response to `ei` on impact |
<!-- paths:end -->

Test results generally have `statistic`, `p_value`, `df`, `null`, `name`, `critical_values.<level>` and
`details.<key>` (the keys a test's table shows). `reject` is a method (`reject(0.05)`), so it cannot be checked:
check `p_value` against a bound with a tolerance, or `statistic`.

### solution.py

Optional: a script that completes the exercise. It is **code**, so it runs only from the command-line checker,
`pymetrix-exercises check`, on a pack the user names; the app never runs it, and opening a pack never runs anything.

How it runs, for each exercise:

1. a fresh `Session`;
2. the exercise's `data` loaded into it, each as a `Dataset` under its name, exactly as the app loads them (File →
   Load data… for a file of `data/`, with its `[data.<name>]` options, equivalent to `load_data`; File → Load
   example data… for an example dataset);
3. `solution.py` run as the Immediate window runs typed code: in the session's namespace (pymetrix's public names,
   `np`, `pd`, `sp`, and the session's objects, so the datasets by name), with
   - the **working directory** set to the pack's folder (relative paths such as `"models/rbc.mod"` or
     `"data/extra.csv"` work), and put back afterwards;
   - **`PACK_DIR`**, the pack's folder as a `pathlib.Path`, defined (so `load_model(PACK_DIR, "rbc")`,
     `load_data(PACK_DIR, "wages")` after `from pymetrix.exercises import load_data, load_model`, or
     `read_mod(PACK_DIR / "models" / "rbc.mod")`); it is removed from the session afterwards;
   - the objects it leaves (its top-level names) become the session's objects, which the checks read;
4. every check of the exercise.

Without a `solution.py`, step 3 runs the steps' `python` blocks, in order, joined, the same way.

The steps' `python` blocks, shown by Show the code in the app, are copied by the user into the Immediate window, where
neither `PACK_DIR` nor the pack's folder as working directory exists: a step that needs a pack's file is better
written as its dialog in the instructions (the panel's Import… for a model; the data are loaded at the start).

## Plugins

A plugin ships packs through the entry-point group `pymetrix.exercises`. Each entry point's value is a callable
(taking no arguments) that returns the pack's folder; a path or string works too.

```toml
[project.entry-points."pymetrix.exercises"]
intro = "pymetrix_intro.exercises:pack_folder"
```

```python
from importlib.resources import files

def pack_folder():
    return files("pymetrix_intro") / "intro-pack"     # a folder inside the installed package
```

The folder must be on disk (an ordinary install, not a zipped one). `installed_packs()` lists the packs of the
plugins that are enabled: a disabled plugin's packs are left out, and a new plugin's wait for the user's answer in the
app (Help → Plugins), as its other extensions do. A plugin's pack and a folder are the same thing; packs that cannot
be read are left out, with the reason in `pymetrix.exercises.problems()`.

## The interface (`pymetrix.exercises`)

| Name | What it does |
| --- | --- |
| `load_pack(path)` | the `Pack` in a folder, validated; `SpecificationError` lists every problem (`.problems`), `NotFoundError` if there is no folder |
| `installed_packs()` | the packs plugins ship (`problems()`: those that could not be read) |
| `check(exercise, session, step=None)` | `CheckResult`s (`id`, `step`, `passed`, `actual`, `expected`, `message`, `detail`) for one step or all; `session` is a `Session` or a mapping of names to objects |
| `load_data(pack, name)` | a dataset of the pack's `data/`, or an example dataset; `pack` is a `Pack` or its folder |
| `load_model(pack, name)` | a model of the pack's `models/` (`.mod`, `.model`, `.pymodel`), by stem or session name; `pack` is a `Pack` or its folder |
| `model_command(pack, name)` | the import dialog step for that model, filled in with its file and session name (what the panel's Import… opens) |
| `LEVELS`, `MODEL_SUFFIXES` | the allowed `level` values; the model file extensions and their syntax |
| `data_commands(exercise)` | the File → Load data… / Load example data… steps that load an exercise's data (recorded and replayable) |
| `resolve(obj, path)` | a value path's value, as checks read it |
| `Pack`, `Exercise`, `Step`, `Check`, `CheckResult` | the parts, as frozen dataclasses; `Pack.exercise(id_slug_or_number)`, `Pack.model_name(stem)`, `Exercise.step(n)`, `Exercise.checks_for(step)` |

## Command line

```
pymetrix-exercises check <pack-folder-or-installed-pack-name> [--exercise ID]
pymetrix-exercises new <folder> [--name NAME] [--title TITLE]
pymetrix-exercises list
```

(`python -m pymetrix.exercises ...` is the same.) `check` prints the pack's title, version, folder and classification
(if any), then each exercise and each check (`ok` or `FAIL`, the value found against the value expected, and the
check's message on failure). It exits **0** when every check passes, **1** when a check fails
or a solution raises, and **2** when the pack cannot be read, does not suit this pymetrix, or `--exercise` names no
exercise. `new` writes a pack with one exercise that passes its own checks.
