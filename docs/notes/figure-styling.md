# Styling figures: matplotlib's power, recorded and replayable

Design note, 24 October 2026. **Status: built, phases A to D** (the calls of layer 1 and the Figure style dialog; the property editor of layer 2, `pymetrix.figure_properties`; annotations and click-to-place, layer 3; `call`, style sheets and standalone matplotlib code, layer 4). It follows the user's request to expose as
much of matplotlib as possible within pymetrix's own objects, in the app's dialogs, with code that can be verified
and replayed. It is written so that it makes sense to someone who knows matplotlib.

## The problem

pymetrix draws with matplotlib's object-oriented interface (never `pyplot`). A pymetrix `Figure` is a description of
a chart: panels holding layers (`Line`, `Bars`, `Band`, `Scatter`, `Box`, `Rule`), plus options such as titles,
shading and events. It is drawn into matplotlib only when it is shown or saved. That is what makes figures
replayable, saveable in projects, and independent of matplotlib's global state. But it also means a matplotlib
user cannot simply write `ax.set_yscale("log")` or `line.set_linestyle("--")` and have it last: the matplotlib
objects are rebuilt every time the figure is drawn.

So matplotlib's options need to enter the description as data, in a form that:

1. **reads like matplotlib**, so its users already know the names, the arguments and where they apply;
2. **is checked by matplotlib itself**, when it is set;
3. **replays exactly**, in scripts and in projects;
4. **appears in the app's dialogs** without a hand-written form for each option.

## The principle: matplotlib's vocabulary, pymetrix's record

Every customisation is a call that matplotlib already has, recorded as data on the figure and replayed when the
figure is drawn.

- **matplotlib's names:** the method names and keyword arguments are matplotlib's own: `set`, `legend`,
  `annotate`, `color`, `linestyle`, `yscale`, `loc`, `ncols`, `arrowprops`. Aliases are accepted and stored under
  their full names, through matplotlib's own alias table (`cbook.normalize_kwargs`): `lw=2` is kept as
  `linewidth=2`.
- **matplotlib's structure:** a matplotlib user thinks in the hierarchy Figure → Axes → artists (lines, patches,
  texts). pymetrix's own structure is Figure → panels → layers. The two are matched:
  - a panel is an Axes, addressed by its position or its title;
  - a layer is the artist it draws, addressed by its label (the series' name).
- **pymetrix's record:** each call returns a new `Figure` (figures are values, as the rest of pymetrix's results
  are), so the step that made it replays, and a project saves it as data, not as code.

A matplotlib user would write:

```python
fig, ax = plt.subplots()
ax.plot(x, gdp, label="gdp", color="C1", linestyle="--", linewidth=2)
ax.set(yscale="log", ylim=(50, 500), ylabel="index")
ax.legend(loc="upper left", ncols=2, frameon=False)
ax.annotate("Lehman", xy=(lehman, 3.2), xytext=(lehman, 6), arrowprops=dict(arrowstyle="->"))
```

The pymetrix equivalent says the same things with the same words, applied to a figure pymetrix made:

```python
fig = data.plot(["gdp", "cons"])
fig = fig.line("gdp", color="C1", linestyle="--", linewidth=2)       # Line2D.set(...)
fig = fig.set(yscale="log", ylim=(50, 500), ylabel="index")          # Axes.set(...); ax=0, the first panel, by default
fig = fig.legend(loc="upper left", ncols=2, frameon=False)           # Axes.legend(...)
fig = fig.annotate("Lehman", xy=("2008Q3", 3.2), xytext=("2008Q3", 6), arrowprops=dict(arrowstyle="->"))
```

Dates are given as the data's own periods (`"2008Q3"`). Plain dates and matplotlib's date numbers are accepted
too, and converted.

## The four layers

### Layer 1: the parts pymetrix draws (phase A)

| Call | Addresses | Mirrors |
|---|---|---|
| `fig.line(label, **props)`, `fig.bars(label, **props)`, `fig.band(label, **props)` | the artist a layer draws | `Line2D.set`, `Rectangle.set` (each bar), `PolyCollection.set` |
| `fig.set(ax=0, **props)` | a panel's Axes | `Axes.set`: `xlim`, `ylim`, `xscale`, `yscale`, `xlabel`, `ylabel`, `title`, `aspect`, ... |
| `fig.ticks(ax=0, axis="y", **props)` | a panel's ticks | `Axes.tick_params`, and tick-label formats as format strings (`"{x:.0%}"`, as `StrMethodFormatter` takes) |
| `fig.grid(ax=0, **props)`, `fig.spines(ax=0, **props)` | grid lines, spines | `Axes.grid`, `Spine.set` |
| `fig.legend(ax=0, **props)` | the legend | `Axes.legend` |
| `fig.layout(**props)` | the whole figure | figure size, dpi, `suptitle`, `supxlabel`, constrained layout |
| `fig.view_init(elev=, azim=, roll=, ax=0)` | a 3-D panel's view | `Axes3D.view_init` (added 27 October 2026, with 3-D panels; `set(zlim=...)` and `ticks(axis="z")` reach their z axis) |

`ax` is the panel: a position (0, 1, ...) or a panel's title. `ax="all"` applies to every panel.

### Layer 2: any property of any artist (phase B)

matplotlib describes its own artists: `ArtistInspector(Line2D).get_setters()` lists the 43 settable properties of a
line, and `get_valid_values("linestyle")` describes their values. The app's property editor is generated from
that, so it covers every property of every artist type pymetrix draws, with matplotlib's own help text, and no
form code per property. The same calls (`fig.line(...)`, `fig.set(...)`) record the changes.

### Layer 3: things added to the chart (phase C)

- **The calls:** `fig.annotate(...)`, `fig.text(...)`, `fig.axhline(...)`, `fig.axvline(...)`, `fig.axhspan(...)`
  and `fig.axvspan(...)`, each with `Axes.annotate`'s or `Axes.text`'s own signature, positions in data
  coordinates.
- **In the app:** a text or an arrow can be placed by clicking on the chart. The click is recorded as data
  coordinates (`"2008Q3"`, 3.2), not screen pixels, so the figure replays at any size.

### Layer 4: the rest of matplotlib (phase D)

`fig.call("set_aspect", "equal", ax=0)` records any other `Axes` method that styles or annotates a panel as
data: the method's name and its arguments, as matplotlib takes them.

- **Restricted:** only `matplotlib.axes.Axes` methods that style or annotate (`set_*`, `tick_*`, `invert_*`,
  `minorticks_*`, `ticklabel_format`, `axline`, `hlines`, `margins`...; `pymetrix.figure_properties.axes_methods()`
  lists them). Methods that plot more data, clear or remove things, or need Python objects (`plot`, `clear`,
  `set_prop_cycle`, `set_transform`...) are refused: the data a figure shows come from its layers.
- **Checked:** the arguments are checked against the method's signature when the call is added.
- **Data only:** arguments must be plain data (numbers, text, lists, dicts), so a project never holds code.

This is the escape hatch: nothing in matplotlib is out of reach, but everything stays replayable and safe to open.

**Style sheets.** matplotlib's `.mplstyle` files are imported as pymetrix themes (`Theme.from_mplstyle(path)`), so
a house style written for matplotlib carries over.

**The house look underneath.** Before any theme or style call, every figure is drawn inside
`rc_context(figure_style.chart_rc(theme, font, surface))`: the grid (major lines, light grey, behind the data),
spines (left and bottom), ticks (outwards, 3 points), type (DejaVu Sans, or the app's font; the theme's sizes), the
theme's palette as the colour cycle, and the surface (white, or the app's dark window colour). The theme's own
settings (a style sheet's) are applied last and win; recorded style calls come after that. Ticks and grid are also
pinned on each axes, so ticks matplotlib makes later (a resize, a zoom) keep the look whatever the global settings.
Nothing is set globally. The figure's size in inches never depends on the look, the screen or the surface.

## Verification

- **Checked when set:** each property is applied to a scratch artist of the right type, through matplotlib's own
  setter, as soon as it is recorded. "`linestyle='dottted'` is not a valid value" appears in the dialog or at the
  call, not later when the figure draws.
- **Round trip:** a test that every style call survives description → generated code → re-run → identical
  description → identical image bytes (fixed metadata; same matplotlib version).
- **The data are untouched:** styling cannot change what is plotted, and `fig.data()` is the same before and after.
- **Provenance:** the matplotlib version is recorded with each drawn figure. The same style can render slightly
  differently in another version, and a report should say which one drew it.

## Replication outside pymetrix

`fig.to_matplotlib()` returns the drawn `matplotlib.figure.Figure`, for anyone who wants to finish a chart by hand.
That copy is not recorded, as anything done by hand isn't.

`fig.matplotlib_code()` writes a standalone script (NumPy, pandas and matplotlib only) that draws the same chart
from the same numbers, with every style call translated into its matplotlib form. A matplotlib user can then check
the chart, or reuse it, without pymetrix.

## In the app

- **Figure options** (a figure's Edit…) gets tabs that follow matplotlib's hierarchy:
  - **Figure:** size, titles, layout and theme;
  - **Axes:** one per panel, with scales, limits, labels, ticks, grid and spines;
  - **Series:** one per layer, with its line, bars or band;
  - **Legend**;
  - **Annotations**.
- **Field names:** each field shows matplotlib's name next to a plain one, e.g. "Line style (`linestyle`)". Its
  help is matplotlib's own text, from `ArtistInspector`.
- **Choices:** fixed sets come from matplotlib: line styles, markers, legend locations, named colours and `C0`–`C9`.
- **Advanced:** an "All properties" view per object (layer 2), and a "Method call" row (layer 4).
- **Recording:** each accepted change is one recorded step (`fig = figure_of(fit).line("gdp", color="C1")`), so
  the History shows exactly what was done, and replays it.

## The data model

A figure gains one field, `style`: a tuple of recorded calls, applied in order when the figure is drawn, after
pymetrix has drawn its layers:

```python
@dataclass(frozen=True)
class StyleCall:
    kind: str        # "line", "bars", "band", "axes", "ticks", "grid", "spines", "legend", "layout", "annotate", "text", "call"
    where: tuple     # which panel and which layer: (panel, label), (panel,), or ()
    props: Mapping[str, Any]   # matplotlib keyword arguments, aliases resolved; plain data only
```

- **Order:** later calls override earlier ones, as successive `set` calls do in matplotlib.
- **Saving:** the tuple serialises to JSON, so projects and saved results keep it.
- **Interaction:** the figure options already built (shading, events, reference lines, line labels, log, ylim) stay
  as they are. Where they overlap, `fig.set(yscale="log")` and `with_options(log=True)` set the same thing, and
  the later call wins.
- **Addressing by label:** a series is found by its label, so a style survives re-estimation and new data. If no
  layer has that label, the drawing says so rather than silently styling nothing.

## Phases

| Phase | Content | Size |
|---|---|---|
| A | `style` on Figure; `line`, `bars`, `band`, `set`, `ticks`, `grid`, `spines`, `legend`, `layout`; checking at set time; Figure options tabs (Figure, Axes, Series, Legend); round-trip and image tests | 2 sessions |
| B | Property editor from `ArtistInspector` (all properties of each artist type) | 2 sessions |
| C | `annotate`, `text`, `axhline`/`axvline`/spans; click-to-place in the app | 1–2 sessions |
| D | `call` (any Axes method, as data); `.mplstyle` themes; `to_matplotlib` and `matplotlib_code` | 1–2 sessions |

## Not intended

- **pyplot's global state:** `plt.gca()`, `plt.rcParams` changes. Themes do that job, per figure.
- **Arbitrary Python in figures:** layer 4 records method calls with plain arguments, not code.
- **Interactive figures in reports and saved files:** they stay static images. The app's zoom and pan toolbar is a
  view, not part of the figure.
