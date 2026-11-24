# pymetrix-eviews

EViews work into pymetrix. This first version reads EViews **workfiles** through a *bundle*: a folder that an
export program, run once in your own EViews, writes from the open workfile. Each page becomes a pymetrix `Dataset`,
with its series' labels, units and sources, and the page's groups, samples, scalars, strings, matrices, value maps,
equations (with EViews's own estimates) and models. The binary `.wf1` file is not read: its format is not
published, and a misread column of numbers looks like data.

Translating EViews **programs** (`.prg`) into pymetrix scripts, and running EViews for you through COM, are later
work (see pymetrix's `docs/design/eviews-plugin.md`).

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
