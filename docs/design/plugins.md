# Plugins: institutional data stores, data APIs, quantitative and graphics libraries

Design note, 25 October 2026 · Extension API v1 (`pymetrix.plugins`) · describes what is built and where it falls
short; nothing here is built by this note

## The question

How does a package from outside pymetrix extend it in both the app and in code? This note looks at four cases:

- an institution's statistical data store, such as PSETS;
- a public data API;
- an additional quantitative library;
- an additional graphics library.

## One mechanism for code and the app

A plugin is an ordinary Python package. It declares entry points in its own `pyproject.toml`, and pymetrix loads
every installed one when the group is first used:

```toml
[project.entry-points."pymetrix.io"]
psets = "pymetrix_psets:register"         # a function: called, and it registers itself

[project.entry-points."pymetrix.methods"]
liml = "pymetrix_linearmodels:LIML"       # a class: registered under the entry point's name
```

- **Groups.** There are twelve: `methods`, `systems`, `models`, `variances`, `tests`, `filters`, `seasonal`,
  `solvers`, `io`, `views`, `commands` and `courses` (`plugins.GROUPS`). A thirteenth entry point,
  `pymetrix.plugins`, is for a package that only describes itself.
- **Who provides it.** The package has a module-level `PLUGIN = PluginInfo(name, version, requires={...},
  license=..., maintainer=..., install=...)`. If a library it needs is missing or the wrong version, its entries
  stay listed, marked unavailable, with the command that fixes it. **Help → Plugins** lists every plugin, its
  version and licence, and any problem.
- **Settings are forms.** A plugin's settings are a frozen dataclass whose fields are
  `Annotated[type, Widget(...)]`. The same class is:
  - the Python object (`LIML(cov="hac")`);
  - the dialog's form, or its sub-form under *Method settings*;
  - a value the codec saves (`{"$type": "pymetrix_linearmodels:LIML", ...}`).

  A docstring and `topics` give it a Help page.
- **Calls are recorded.** A plugin's public functions and methods can carry `@api`. Their calls are then
  recorded with the plugin's own ID prefix (`linearmodels:...`), saved in projects and replayed exactly as
  built-in calls are ([commands.md](commands.md)). A project that uses a plugin that is not installed says
  `no call ... is installed` and does not run the step.
- **Dialogs.** A plugin may add a dialog with `@command(menu="Estimate/My model...")`; it appears in that menu.
  Dialogs should build the call they run (`to_call`), as the built-in ones do.
- **Safe reading.** The codec rebuilds classes only from pymetrix and from installed plugins
  ([serialisation.md](serialisation.md)), so a project file cannot make pymetrix import arbitrary code.
- **Trust.** Installing a plugin runs its code, as installing any Python package does (`SECURITY.md`).
  pymetrix does not sandbox plugins.

The reference plugins (`pymetrix.ext.connectors`, `ext.linearmodels`, `ext.arch`, `ext.sklearn`) are
written against this API only, and `tests/sample_plugin.py` is a minimal outside package. They are the
worked examples.

The first-party plugins in `packages/` are packages of their own, written against the same API and listed in
`plugins.FIRST_PARTY`: pymetrix-finance, -labbook, -guide, -assistant, -psets, -seasonal, -eviews and -ha
(heterogeneous-agent models by sequence-jacobian; [heterogeneous-agents.md](heterogeneous-agents.md)).

### What a plugin imports, and the API's versions

A plugin imports pymetrix's inside only through `pymetrix.extension`, under names that stay put as pymetrix changes,
and otherwise from the stable modules (`pymetrix.extension.STABLE`). The finance plugin's public-surface test, and
the one `pymetrix-new-plugin` writes, hold a plugin to this. `pymetrix.plugins.API` is the extension API's version.
Its major number is checked (`SUPPORTED_APIS`); a minor version only adds names, so a plugin written for 1.4 runs on
1.10. The comment beside `API` in `plugins.py` and section 6 of the reference (`docs/reference/reference.md`,
`scripts/make_api_reference.py`) list what each minor version added. Compare versions as numbers, never as text:
`1.10` follows `1.9`.

**1.10 (additive): the core EViews reader's text helpers**, for a plugin that translates EViews programs
([eviews-plugin.md](eviews-plugin.md), decision 4). Each name is the reader's own helper, so the plugin reads a
program exactly as `read_eviews` and `check_eviews` do; nothing in core changed.

| Name | What it is |
| --- | --- |
| `eviews_blank_comments(text)` | the text with comments (`'` outside double quotes) as spaces: every offset stays |
| `eviews_statements(clean)` | `[(statement, offset)]` of that text; a line ending in ` _` continues, with the same length |
| `parse_eviews_expression(text, base, vectors, comparison=False)` | one expression as a tree; offsets are `base` plus the text's own; `vectors` the coefficient vectors, `{"c"}` at least |
| `EViewsNumber`, `EViewsName`, `EViewsCoefficient`, `EViewsCall`, `EViewsNegative`, `EViewsBinary`, `EViewsBrackets` | the tree's nodes: frozen dataclasses, each with its offset `at` |
| `walk_eviews_expression(node)` | the tree's nodes, the root first |
| `render_eviews_expression(node, scalars=(), frequency=None)` | the tree as pymetrix text, as `convert_expression` writes it |
| `EViewsExpressionError` | what does not read: a `SpecificationError` with `.message`, `.at` and `.length` |
| `Diagnostic`, `line_and_column(text, offset)` | a problem at a line and column (both from 1), as `ModelSource.check` reports them |

## Case 1: an institutional data store (PSETS)

pymetrix is not tied to any institution's data systems; the built-in PSETS reader stub was removed (decided 11
October 2026). A store connects through a reader plugin.

### PSETS's own API

PSETS (Pretty Simple Economic Time Series, the `Econ_timeseries_DB` repository) already has a host API for
programs like pymetrix: the `psets` package, API version 1.0, specified in its `docs/plugin_api.md`. It is
read-only on purpose; loading figures stays with PSETS's CLI and Excel add-in, where its rules are enforced.

```python
import psets

db = psets.open("forecasts")                 # a registry alias, a path, or (no argument) the user's default
db.search("consumer price", freq="M")        # a frame: series_id, title, freq, unit, source, dataset, first/last period
db.facets(freq="Q")                          # what could narrow the search next, with counts
db.series("PCPI_LOC.M")                      # one series' metadata (psets.Series)
db.frame(["PCPI_LOC.M", "NGDP.A"], start="2010-01-01", as_of="2020-03-29")   # figures, wide
db.vintages(code), db.revisions(code, period), db.triangle(code)            # what was known when
db.provenance(code), db.issues(code)         # what a canonical code is built from; known publisher problems
```

A PSETS vintage is the publisher's release date, never the load date, which is pymetrix's rule too. A missing
code raises `UnknownSeries` rather than returning an empty frame.

### The `pymetrix-psets` plugin

The plugin is a thin reader over that API. It is built: [`packages/pymetrix-psets`](../../packages/pymetrix-psets/README.md),
a package of its own (`pip install "pymetrix[psets]"`), outside pymetrix's core. It is tested against a stand-in for
`psets` in CI, and against PSETS itself, with a real database built by PSETS's migrations, wherever PSETS is installed.

- **The address.** `psets://<alias>/<code>,<code>`, with an empty alias for the user's default database
  (`psets:///PCPI_LOC.M`). An alias rather than a file path keeps projects portable between machines.
- **Reading.** `Dataset.read("psets://forecasts/PCPI_LOC.M,NGDP_R.Q", vintage="2020-03-29")` becomes:
  - `db.frame(codes, as_of=...)`, where pymetrix's `vintage` is PSETS's `as_of`;
  - `long=True` when the codes' frequencies differ;
  - PSETS's periods (end-of-period dates) read as pandas periods.

  PSETS's own `freq` with `agg` (downward conversion only) may be passed through as reader options.
- **Metadata.** `db.series(code)` fills each series' `Variable`:

  | `psets.Series` | pymetrix `Variable` |
  | --- | --- |
  | `title` | `label` |
  | `notes` | `description` |
  | `unit` | `units` |
  | `multiplier` (a factor, e.g. 1e-06) | `scale`, where it maps; otherwise kept in `extra` |
  | `seasonal_adj` | `adjustment` (`nsa`, `sa`, ...) |
  | `source_name` | `source`, in words |
  | `series_id` | `source_key` |
  | the latest of `db.vintages(code)`, or the `as_of` asked for | `vintage` |
  | `freq`, `first_period`, `last_period`, `dataset`, `sector`, `topic`, `geo` | `extra` |

  `retrieved` is the time of the read. `external_ref` is `psets://<alias>/<code>@<vintage>`, which the plugin
  can resolve back to the series and its release.
- **Trust.** `db.provenance(code)` (the source series behind a canonical code, with link factors) and
  `db.issues(code)` (publisher problems PSETS knows about) go in each series' `extra`. Open issues should also
  appear as a caution when a step uses the series, which is where they would stop someone.
- **Real-time data.** `db.as_of(date)` gives the data as released on a date. That is exactly what
  `backtest(..., vintages={origin: dataset})` takes, so a backtest can use the data available at each origin.
  The plugin can offer a helper that builds that mapping.
- **No credentials, no cache.** PSETS is a local or shared-drive SQLite file found by alias, so decisions 2 and
  3 below do not arise for it.
- **Installing PSETS.** PSETS is now a package (`python/pyproject.toml`, everything under `psets`), and `import psets`
  loads none of its Qt console, xlwings or openpyxl. It is not on PyPI, so it installs from its repository:
  `pip install "PSETS @ git+https://github.com/GeoffHeenan/Econ_timeseries_DB#subdirectory=python"`. Its base install
  still brings the Excel add-in's packages, so the plugin does not depend on it. It is listed as a requirement, and
  **Help → Plugins** shows that command until PSETS is installed.

### What pymetrix provides any store's plugin

- **Provenance.** For each series the reader fills the minimum pymetrix keeps itself
  ([adding-features.md, 9a](../adding-features.md#9a-data-sources-and-provenance)):
  - `source`, in words;
  - `source_key`, the series' ID in the store;
  - `vintage`, the store's release date, never the load date;
  - `retrieved`, when the values were fetched.

  It does not copy the store's own, richer record into pymetrix. It sets `external_ref`
  (`psets://forecasts/PCPI_LOC.M@2026-09-30`), which pymetrix keeps in datasets, Parquet files and projects,
  shows in the data dictionary and never interprets; the plugin can resolve it. A series whose values are
  replaced loses these fields, since they described the old values.
- **Recording.** The read is recorded as a `Dataset.read` call. Generated code always writes the vintage asked
  for (`vintage='latest'` or a date), and every result made from the data carries that request in its
  provenance ([provenance.md](provenance.md)). Replaying a project reads the store again.
- **What is missing.**
  - *Finding series in the app.* The Load data dialog picks files; a store's address is typed by hand. There is
    no *Data → Get data* dialog that lists a store's series.
  - *Credentials.* pymetrix has no credentials mechanism, so each plugin handles sign-in itself.
  - *Caching.* Every read goes to the store.

## Case 2: a public data API

Public APIs use the same pattern: one reader per URI scheme. The bundled `pymetrix.ext.connectors` reads FRED
(`fred://GDPC1?vintage=2020-01-30` for ALFRED), the World Bank (`worldbank://...`) and SDMX providers (ECB,
Eurostat, OECD, IMF, BIS) with the standard library only. An institution's own SDMX data warehouse is one line:

```python
register_sdmx_provider("MYCB", "https://data.mycb.org/sdmx/data/{flow}/{key}?format=csvdata")
data = Dataset.read("sdmx://MYCB/NA/Q.GDP.SA")
```

So a store that speaks SDMX needs no plugin at all. The gaps are those of Case 1: no browser in the app, no API
keys (a FRED key would give per-observation vintages), no caching, and no SDMX code-list look-ups.

## Case 3: a quantitative library

This is the most complete extension point. [wrappers.md](../wrappers.md) gives the general form and three
shapes:

1. **An estimation method** for `Equation.fit`, such as LIML, ridge or a quantile regression. Registered under
   `methods`, it appears in the estimation dialogs' method picker with its settings as a sub-form, and in code
   as `Equation(...).fit(data, method=LIML())`.
2. **A model with its own `fit`**, such as a new time-series model. Registered under `models`:
   - it returns its own result type;
   - `register_view(MyResult)` gives that type its result-window tabs (tables and figures);
   - the generic verbs (`coefficients()`, `statistics()`, `observations()`) attach automatically;
   - `@api` on `fit` records it, and a `@command` dialog puts it in a menu.
3. **A test or a function**, registered under `tests` or another group. Tests appear in Run test and
   `fit.test("name")`.

Each wrapper reaches its library through one backend adapter ([adapters.md](adapters.md)). The adapter holds:
- how pymetrix's settings map to the library's arguments, shown on the help page;
- what pymetrix fixes, and what it refuses.

Users can pass further library arguments as `backend_options`. Results come back on pymetrix's time index and
names ([time-index.md](time-index.md)), with the backend and its version in their provenance. Library objects
stay in `backend_result` and are never saved ([results-record.md](results-record.md)).

Variances (GARCH-type), filters, seasonal adjustment, solvers and systems estimators each have their own group,
with the same pattern.

## Case 4: a graphics library

There is no extension point for another drawing library.

- A pymetrix `Figure` is a data model: panels of lines, bars and bands, plus recorded style calls (`line`, `set`,
  `annotate`, `call`...). It is drawn by matplotlib (`Figure.draw`, `to_matplotlib`, `matplotlib_code`).
- A result window's tabs hold a `Table` or a `Figure` (`views.Section`).

A plugin can already add new kinds of chart built from pymetrix's figure parts (as `coefficient_figure` and
`heatmap_figure` are), and they are styled, recorded and replayed like any other. A chart drawn by plotly,
Altair or Bokeh cannot be shown in a tab, styled in the app or recorded.

Because a `Figure` keeps its data and every style call, three routes are open:

| Route | What it gives | Cost |
| --- | --- | --- |
| (a) **Exporters:** a `Figure` → plotly or Vega-Lite (`figure.to_plotly()`, an `exporters` group) | interactive charts for web pages and reports; the app is unchanged | small: map panels and series; style calls that have no equivalent are dropped, and the exporter says which |
| (b) **Renderers:** a `renderers` group; a plugin draws a `Figure` its own way, in the app too | another look throughout, still recorded | large: every style call needs a meaning in the other library, and the app needs a widget for it |
| (c) **HTML tabs:** views may return HTML, shown in an embedded viewer | any library's output in a result window | medium: a web view in the Qt app; such charts cannot be styled or recorded |

**Recommendation:** (a) first, as a separate plugin to prove the API. Then (b) only if there is demand.

## Proposal for decision 1: browsing series, and requests kept for reuse

*Proposed, not built.* Readers today only read. This adds an optional catalogue for browsing, and a request
object for reading the same series again.

### A catalogue next to the reader

A data plugin registers a catalogue for its scheme beside its reader:

```python
def register() -> None:
    register_reader("psets", read_psets)
    register_catalogue("psets", PsetsCatalogue())   # optional
```

A catalogue answers up to five questions. A plugin implements only those its store supports: a plain list of
codes can offer just `search`; PSETS has `search`, `facets` and `describe`; an SDMX service has `dimensions`
too.

| Method | Answers | Returns |
| --- | --- | --- |
| `children(path="")` | what is under this node (store → topic → table → series) | nodes: key, label, and whether it is a series or a folder |
| `search(text, filters={}, limit=100)` | which series match | series descriptions |
| `facets(filters={})` | what could narrow the search next, and how many series each choice leaves | facet → (value, count) pairs |
| `describe(keys)` | a series' full metadata, before any data are read | series descriptions |
| `dimensions(table)` | an SDMX-like table's dimensions and their code lists, so a key can be built from choices | codes with their labels |

**A series description is the `Variable` the series would arrive with**:
- `label`, `units`, `measure`, `adjustment`, `prices`, `source_key` and `external_ref`;
- its frequency, first and last period, when it was last updated, and the vintages available.

What the browser shows is then exactly what lands in the dataset; there is no second metadata model. The
bundled SDMX connector would gain a catalogue from its providers' structure queries, which also fills the
code-list gap in Case 2.

For PSETS each method is one call: `search` is `db.search(text, **filters)`, `facets` is
`db.facets(**filters)` (freq, source, dataset, sector, topic, geo, publisher, concept), `describe` is
`db.series(code)` mapped as in Case 1, and the vintages are `db.vintages(code)`. PSETS's own Find Series dialog
is a faceted search, so the browser should offer facets as well as a tree; a tree alone would not suit it.

**The dialog.** *Data → Get data* holds:
- a source picker, listing the registered catalogues;
- a tree (for stores with one), a search box, and facets with their counts;
- a results table of label, units, frequency and last observation;
- a metadata pane filled by `describe`;
- a basket of the chosen series.

Catalogue calls run off the interface thread, are paged, and are cached for the session, so a slow store does
not freeze the app; errors are shown as messages. *Read* builds an ordinary recorded read and runs it, with a
code preview as every dialog has. Browsing creates no new kind of step: it is a way to compose a request.

### Keeping a request for reuse

There are three layers.

1. **The read step (built).** A project already records `data = Dataset.read("psets://...", vintage='latest')`
   as a call.
   - Reopening or re-running the project fetches the newest release.
   - A fixed vintage reproduces exactly.
   - Steps that use `data` are marked out of date when it changes.
2. **A named request: `DataRequest`.** A URI grows long and loses structure, such as which code becomes which
   short name. A request is a setting like any other:

   ```python
   gdp_pack = DataRequest("psets://forecasts",                       # the store, and which database
       series={"y": "NGDP_R.Q", "y_n": "NGDP.Q", "cpi": "PCPI_LOC.M"},  # short name: code
       span=("2000Q1", None), vintage="latest",
       options={"freq": "Q", "agg": "avg"})                          # applied by the store as it reads
   data = gdp_pack.read()                    # a recorded call; Dataset.read(gdp_pack) does the same
   data = gdp_pack.read(vintage="2026-09")   # the same request, as released then
   ```

   Because it is a setting, a request:
   - is saved by the codec: in projects, in a personal library (`~/.pymetrix/requests/gdp_pack.json`), or in a
     shared folder for a team;
   - opens in a form, appears in *Get data* under *Saved requests*, and compares cleanly between versions;
   - keeps opening as pymetrix changes, since the codec versions it;
   - holds no credentials: it names the source, and signing in stays with the plugin or the shared keychain
     helper of decision 2;
   - still has a URI form, `request.to_uri()`.

   Transformations stay out of a request. `dlog` and the like belong in derived series, so one request feeds
   many analyses. Options the store itself applies when reading, such as PSETS's downward frequency conversion
   (`freq="Q", agg="avg"`), may be kept in the request's `options` and passed to the reader.

   **PSETS's precedent.** PSETS already keeps requests of this kind: a model register
   (`registers/models/<model>_mapping.csv`) lists each model code, the PSETS code it comes from, a transform
   and a scale, and `model_pull` reads it as of a date. A `DataRequest` is the same idea as a setting. The
   PSETS plugin should read a register directly (`psets_request("registers/models/<model>_mapping.csv")`),
   so one register serves the model, PSETS and pymetrix.
3. **Refresh and snapshot.** A project may keep both the request and a Parquet snapshot of what it returned.
   - On opening, the user chooses *Use saved data* (works offline, reproduces the session) or *Refresh*.
   - A refresh reports what changed: the series revised, the new observations, and the vintage before and after.
     It reuses the revision comparison forecast rounds already have (`ForecastStore.compare`).
   - Results that use the data are marked out of date.

### Open questions

- **Where saved requests live:** a personal library, the project, or both. *Suggested:* both, with *Save to
  library* as an explicit action.
- **Short names for series.** Codes such as `PCPI_LOC.M` are not valid series names, so a request needs a
  short name for each. Either the user always gives one, or pymetrix makes one from the code. *Suggested:*
  pymetrix makes one (`pcpi_loc_m`), and the user can change it; a PSETS model register already gives one (its
  `model_code` column).
- **Snapshots:** whether a project keeps its data by default (bigger files, reproducible offline) or only when
  asked.

## Decisions needed

1. **Finding data in the app.** Either a generic *Data → Get data* dialog driven by optional reader hooks, or
   one dialog per plugin (each a `@command`).
   - *Recommended:* the generic dialog, with catalogues and `DataRequest`, as proposed above.
   - The proposal's open questions need answers too: where saved requests live, short names, and snapshots.
2. **Credentials.** Either a shared store for data plugins, or each plugin handles its own.
   - *Recommended:* a small shared helper backed by the operating system's keychain (the `keyring` package, an
     optional dependency).
   - It would never write secrets into projects or generated code; code would name the credential, not hold it.
3. **Caching** of remote reads. Either none, or a cache keyed by source and vintage, under `~/.pymetrix`.
   - *Recommended:* the cache, used only when a vintage is fixed, so `latest` always reads fresh.
4. **The graphics route:** (a), (b) or (c) above.
5. **A plugin template.** A cookiecutter or template repository with `pyproject.toml`, entry points, a
   `PluginInfo`, one setting, one reader, and the conformance tests every plugin should pass.
   - *Recommended:* yes, before 0.1, so that outside authors start from the tested shape.

## Tests that already cover plugins

- `tests/test_conformance.py`: every built-in and reference extension passes the conformance kit; the sample
  plugin loads through simulated entry points and conforms; unsupported API versions, missing libraries and two
  plugins using the same label are reported; plugin results record their backend and plugin, and round-trip
  through projects.
- `tests/test_plugin_versions.py`: missing and unsupported library versions, and the install hints.
- `tests/test_connectors.py`: readers by scheme, provenance fields, `external_ref` kept through Parquet and
  projects (with a `psets://` reference), vintages.
- `tests/test_sklearn_plugin.py` and the linearmodels and arch tests: whole wrappers through the adapters.
