# pymetrix-eviews

EViews work into pymetrix. This first version reads EViews **workfiles** through a *bundle*: a folder that an
export program, run once in your own EViews, writes from the open workfile. Each page becomes a pymetrix `Dataset`,
with its series' labels, units and sources, and the page's groups, samples, scalars, strings, matrices, value maps,
equations (with EViews's own estimates) and models. The binary `.wf1` file is not read: its format is not
published, and a misread column of numbers looks like data.

It also translates EViews **programs** (`.prg`), for data work so far (phase 1; see "Programs" below). Running
EViews for you through COM is later work (see pymetrix's `docs/design/eviews-plugin.md`).

## The export program is experimental

`pymetrix_export.prg` was written from EViews's public documentation and has **never been run in EViews**: the
project has no EViews licence. Every line whose syntax or behaviour must be checked is marked `(verify)` in it. It is
tested only through the reader, against bundles written by hand to the format and against the manifest's JSON
Schema.

**The supported way in, until the program has been checked**, is to save the workfile from EViews as CSV or Excel
(File → Save As, or `wfsave(type=...)`) and load that with pymetrix's **File → Load data…**. The label comes off once
someone with a licence has run the program on the two most recent major versions of EViews and its bundles match
the hand-made fixtures. If you have EViews, running it and reporting what happens is the most useful help there is.

## Use

1. In pymetrix: **File → Import EViews workfile…**, *Get the export program (experimental)*, and save `pymetrix_export.prg`
   (or `save_export_program("C:/work")` in code).
2. In EViews: open the workfile, then `exec pymetrix_export.prg C:\work\macro-export`.
3. In pymetrix: **File → Import EViews workfile…**, *Workfile bundle*, and choose `C:\work\macro-export\macro.evbundle`.
   Leave *Pages* empty for every page.

Each page is one line of the script, and of the History:

```python
quarterly = Dataset.read("C:/work/macro-export/macro.evbundle", page="quarterly")
annual = Dataset.read("C:/work/macro-export/macro.evbundle", page="annual")
```

Replaying them needs this plugin, as a PSETS project needs `pymetrix-psets`. Tick **Save the data as Parquet** to
write each page next to the bundle as `<page>.parquet` and read it from there: the script then needs only pymetrix
(a project still names the plugin's step, so reopening the project needs the plugin; load the Parquet files with
File → Load data to drop it).

In code:

```python
from pymetrix_eviews import bundle_pages, compare_estimates, eviews_objects

bundle_pages("C:/work/macro-export/macro.evbundle")   # each page: structure, frequency, how many series
objects = eviews_objects(quarterly)   # groups as selections, samples, scalars, strings, matrices, models
compare_estimates(quarterly)          # each least-squares equation refitted, beside EViews's estimates
```

## What is read, and where it goes

| EViews | pymetrix |
| --- | --- |
| a page | a `Dataset`, named after the page (made a valid Python name) |
| dated pages: A, Q, M, W, D5, D7 | a time axis of `Y`, `Q`, `M`, `W-<day>` (the week's last day, from the dates), `B`, `D` |
| semi-annual, multi-year, intraday, bimonthly, fortnightly, ten-day | refused with the reason; `undated=True` reads the page undated, its dates kept as the text series `date` |
| panel pages | a panel dataset (balanced or not), with the cross-section id as its entity axis |
| undated pages | a cross-section with an `obs` axis |
| NA, an empty numeric field | missing |
| alpha series | text (pandas `string`); an empty value is missing |
| value maps | the codes stay as data; the map goes in `Variable.extra["value_map"]` (pymetrix has no value labels) |
| display name, description, units, source | `Variable.label` (the display name, else the description), `description`, `units`, `source` |
| remarks, attributes | `Variable.extra` |
| groups, samples | `meta.extra["eviews"]`; `eviews_objects` gives a `Selection` and a `Sample` |
| scalars, strings | `meta.extra["eviews"]`; a model's scalars are its parameters in `bundle_model` |
| vectors, matrices, sym, coef | `meta.extra["eviews"]`; `eviews_objects` gives `Matrix` objects |
| equations | the command, method, sample and EViews's estimates, in `meta.extra["eviews"]`; `compare_estimates` refits the least-squares ones |
| models | the text, in `meta.extra["eviews"]`; `bundle_model(data, name)` reads it with `read_eviews` |
| other objects (graphs, tables, spools...) | listed in `meta.extra["eviews"]["skipped"]` |

What does not read is refused at its place, never misread: a folder without its manifest (an incomplete export:
the program writes the manifest last), a bundle of a newer `format_version` ("update pymetrix-eviews"), a CSV
without a column the manifest lists, a value that is not a number, a date that does not start a period of the
page's frequency, a file outside the bundle's folder. Each error names the file, line and column, or the manifest's
key. Keys the reader does not know are kept in `meta.extra["eviews"]["unknown"]` with a note, as the format's
version 1 grows only by adding. Files are read as UTF-8; one that is not falls back to Windows-1252 with a note.

## The bundle, format version 1

```
macro-export/
  macro.evbundle             the manifest (JSON, UTF-8): the file pymetrix opens
  pages/<page>.csv           a page's series: the date (pmx_date) and id columns, then every series and alpha series
  text/<page>/<object>.txt   long texts: a model's text, a command over several lines
  objects/<page>/<name>.csv  vectors, matrices, sym, coef (one line per row, no header); <equation>_est.csv
                             (coefficient,estimate,std_error)
  valmaps/<page>/<name>.csv  value maps: value,label
  export.log                 the program's record of what it exported and skipped
```

The manifest's JSON Schema is `src/pymetrix_eviews/bundle-1.schema.json` (`bundle_schema()`); pymetrix's
`docs/design/eviews-plugin.md` describes each key.

## Programs

`translate_program` turns an EViews program into a readable pymetrix script, a report with a row for every line,
and the same work as recorded steps:

```python
from pymetrix_eviews import translate_program

t = translate_program("C:/work/macro.prg", bundle="C:/work/macro-export/macro.evbundle")
print(t.script)          # each EViews line as a comment, above what it became
t.report.summary()       # line, status (translated, translated with a note, left out, error), EViews, Python, reason
out = t.run(session)     # steps mode: each data statement a recorded step; scalars, strings, matrices returned
```

`bundle` is what the program's `wfopen` opens: the bundle, or a CSV or Excel file saved from the workfile in EViews
(with `read_options={"time": "obs", "freq": "Q"}`, `Dataset.read`'s options). A program that starts with `wfcreate`
needs neither. `run` keeps the script and the report in the project (`session.attachments["pymetrix-eviews"]`).

This is phase 1, data work:

| EViews | pymetrix |
| --- | --- |
| `wfopen`, `wfcreate`, `pagecreate`, `pageselect` | `Dataset.read(...)`, `Dataset.empty("Q", "1990Q1", "2030Q4")`, a variable per page |
| `smpl 1990q1 @last`, `smpl @first+4 2000q4 if x > 0`, `sample s ...`, `smpl s` | `data.with_sample(Sample(...))`, a `Sample` variable |
| `series y = ...`, `genr`, `y = ...` | `data.with_series(y="...", within=Sample(...))` under a sample; period by period when the formula uses `y` |
| `series y`, `y = na` | `data.with_series(y=None)` |
| `frml y = ...` | `data.derive(y="...")` |
| `group g a b`, `g.add c` | `g = data.series("a", "b")` |
| `x.label(d)`, `x.label(u)`, `x.label(s)`, `x.displayname`, `x.setattr(units)` | `data.with_info("x", ...)` |
| `rename a b`, `delete a b`, `x.fill(o=2000q1) 1, 2` | `data.rename(a="b")`, `data.drop("a", "b")`, `data.edit({...})` |
| `scalar s = 0.3`, `!i = 2`, `%s = "a"`, `string s = "a"` | Python variables; `{%s}` and `{!i}` put in where known before the program runs |
| `scalar m = @mean(x)` (also `@sum`, `@min`, `@max`, `@stdev`, `@median`, `@obs`) | `float(data.eval("x")[data.sample_mask()].mean())` |
| `vector(3) v`, `matrix(2,2) m`, `m.fill 1, 2, 3, 4` | `Matrix.zeros(3, 1)`, `Matrix([[1, 3], [2, 4]])` (by column, as EViews) |

Left out, each with its reason and pymetrix's way where there is one: estimation, tests and models (phase 2),
loops, `if` blocks, subroutines and `include` (phase 3), graphs, tables, spools and display, databases, add-ins, and
expression functions pymetrix has no equivalent of (`@year`, `@round`, `@nan`, `@iif`, `nrnd`... ; pymetrix's
known gaps lists them). Translating in the app is later work.

## Installing

```bash
pip install -e packages/pymetrix-eviews    # from the pymetrix repository
```

It needs nothing beyond pymetrix. Its tests (`pip install -e "packages/pymetrix-eviews[test]"`) check the hand-made
fixtures against the schema with `jsonschema`.

## Tests

`tests/fixtures/` holds bundles written by hand, independently of the reader: a workfile of four pages (quarterly,
annual, a panel, an undated cross-section) with every case above, and malformed bundles that must be refused. The
"EViews estimates" in them were computed with NumPy's least squares on the fixture's data, and a test recomputes
them. The tests check every value and metadata field read, the refusals and their places, the dialog against the
`Dataset.read` calls it stands for, and that the scripts replay.

`tests/fixtures/programs/` holds EViews programs written for the tests; `test_program.py` checks every value they
make against values worked out by hand, and that the script and the steps give equal objects.
`test_program_validation.py` translates Brillet's and the FRB/US programs (`PYMETRIX_BRILLET`, `PYMETRIX_FRBUS`;
skipped without them) and fixes the count of each status.
