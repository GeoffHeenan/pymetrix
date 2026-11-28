"""A pymetrix figure as a standalone matplotlib script: ``Figure.matplotlib_code()``.

The script uses NumPy, pandas and matplotlib only. It holds the figure's numbers, draws each layer with the
matplotlib call that pymetrix uses (``plot``, ``fill_between``, ``bar``, ``axhline``, ``scatter``, ``hist``,
``boxplot``, ``hexbin``, ``violinplot``, ``contourf``, ``plot_surface``...), and writes each figure option and recorded style call as the matplotlib call it is, so a matplotlib user
can check the chart, or reuse it, without pymetrix. Colours and line widths are those pymetrix drew (the figure is
drawn once to read them). Date ticks use matplotlib's own date labels rather than pymetrix's period labels (2023Q1).
A 3-D panel is made as matplotlib makes one, ``fig.add_subplot(..., projection='3d')``, in the panel's place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["matplotlib_code"]


def _value(v: Any) -> str:
    """A number as code: nan for missing, plain floats otherwise."""
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        v = float(v)
        return "nan" if np.isnan(v) else repr(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    return repr(v)


def _when(text: Any, dated: bool = True) -> str:
    """A period or date written as a timestamp: pd.Period('2008Q3').to_timestamp(); on an undated axis, text that is a
    number is that number, as pymetrix reads it."""
    if isinstance(text, str) and not dated:
        try:
            return _value(float(text))
        except ValueError:
            pass
    return f"pd.Period({str(text)!r}).to_timestamp()" if isinstance(text, str) else _value(text)


def _hex(color: Any) -> str:
    from matplotlib.colors import to_hex

    return to_hex(color, keep_alpha=False)


class _Script:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.index_names: dict[tuple[str, ...], str] = {}
        self.data: list[str] = []
        self.count = 0

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def name(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}{self.count}"

    def x(self, index: pd.Index, mid: bool = False) -> str:
        """The x values of a series, defined once per distinct index: periods at their starts, or at their middles
        in a panel that mixes frequencies."""
        key = tuple(str(i) for i in index) + (str(getattr(index, "freqstr", "")), mid)
        if key not in self.index_names:
            name = f"x{len(self.index_names) + 1}"
            if isinstance(index, pd.PeriodIndex):
                periods = f"pd.PeriodIndex({[str(p) for p in index]!r}, freq={index.freqstr!r})"
                self.data.append(f"{name} = {periods}.to_timestamp()" if not mid else
                                 f"{name} = {periods}.to_timestamp() + ({periods}.to_timestamp(how='end') - {periods}.to_timestamp()) / 2"
                                 "  # the middle of each period: this panel mixes frequencies")
            else:
                self.data.append(f"{name} = {[i if isinstance(i, (int, float)) else str(i) for i in index]!r}")
            self.index_names[key] = name
        return self.index_names[key]

    def values(self, s: pd.Series) -> str:
        name = self.name("y")
        self.data.append(f"{name} = np.array([{', '.join(_value(v) for v in s.to_numpy(float))}])")
        return name

    def coordinates(self, s: pd.Series) -> str:
        """A 3-D layer's coordinates: periods as matplotlib's date numbers, numbers as they are."""
        if isinstance(s.dtype, pd.PeriodDtype):
            name = self.name("d")
            periods = pd.PeriodIndex(s)
            self.data.append(f"{name} = mdates.date2num(pd.PeriodIndex({[str(p) for p in periods]!r}, freq={periods.freqstr!r}).to_timestamp())")
            return name
        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            name = self.name("d")
            self.data.append(f"{name} = mdates.date2num(pd.to_datetime({[t.isoformat() for t in pd.DatetimeIndex(s)]!r}))")
            return name
        return self.values(s)

    def grid(self, index: pd.Index) -> tuple[str, str]:
        """A grid's rows or columns, where they are drawn, and what they are: "date", "number" or "names"."""
        name = self.name("g")
        if isinstance(index, pd.PeriodIndex):
            self.data.append(f"{name} = mdates.date2num(pd.PeriodIndex({[str(p) for p in index]!r}, freq={index.freqstr!r}).to_timestamp())")
            return name, "date"
        if isinstance(index, pd.DatetimeIndex):
            self.data.append(f"{name} = mdates.date2num(pd.to_datetime({[t.isoformat() for t in index]!r}))")
            return name, "date"
        if pd.api.types.is_numeric_dtype(index) and not pd.api.types.is_bool_dtype(index):
            self.data.append(f"{name} = np.array([{', '.join(_value(v) for v in index)}], float)")
            return name, "number"
        self.data.append(f"{name} = np.arange({len(index)}, dtype=float)  # " + ", ".join(str(v) for v in index))
        return name, "names"

    def table(self, frame: pd.DataFrame) -> str:
        name = self.name("Z")
        self.data.append(f"{name} = np.array({[[float(v) for v in row] for row in frame.to_numpy(float)]!r})".replace("nan", "np.nan"))
        return name

    def label_axis(self, ax: str, axis: str, kind: str, index: pd.Index, positions: str) -> None:
        """Dates with matplotlib's date ticks; names at their positions, at most twelve."""
        if kind == "date":
            self.add(f"{ax}.{axis}axis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))")
            self.add(f"{ax}.{axis}axis.set_major_formatter(mdates.ConciseDateFormatter({ax}.{axis}axis.get_major_locator()))")
        elif kind == "names":
            names = [str(v) for v in index]
            step = max(1, int(np.ceil(len(names) / 12)))
            self.add(f"{ax}.{axis}axis.set_ticks({positions}[::{step}], {names[::step]!r})")


def _kwargs(props: dict[str, Any], dates: tuple[str, ...] = ()) -> str:
    parts = []
    for k, v in props.items():
        if k in dates and isinstance(v, (tuple, list)):
            v_code = "(" + ", ".join(_when(x) for x in v) + ("," if len(v) == 1 else "") + ")"
        else:
            v_code = _value(v) if isinstance(v, (int, float)) else repr(v)
        parts.append(f"{k}={v_code}")
    return ", ".join(parts)


def matplotlib_code(figure: Any) -> str:
    """The script (see the module's description)."""
    from .figures import (QQ, Band, Bars, Box, Contour, Ecdf, Estimates, Heatmap, Hexbin, Hist2d, Histogram, Line, Line3d, Rule, Scatter,  # noqa: F401
                          Scatter3d, Surface, Violin, _estimate_names, _heatmap_mode, _heatmap_scale, _panel_indexes, _runs)
    from .figures import _BLACK_TEXT, _DARK_TEXT, _WHITE_TEXT, COLUMN_SCALE_NOTE
    from .recessions import recessions
    from .themes import get_theme

    from dataclasses import replace

    figure = figure._housed()  # the house style's calls are written out like the figure's own
    _, drawn = replace(figure, style=())._draw()  # what pymetrix drew before the styles: colours and widths, by layer
    theme = get_theme(figure.theme)
    panels = figure._panels()
    cols = max(1, min(figure.columns, len(panels)))
    rows = figure.rows
    height = figure.drawn_height()
    sc = _Script()
    from .figure_style import chart_rc

    # the house look (figure_style), as plain settings; the script sets them for itself, as a script of its own may
    rc = {k: v for k, v in chart_rc(theme).items()
          if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, str) for x in v))}
    sc.add(f"plt.rcParams.update({rc!r})")
    sc.add(f"fig, axes = plt.subplots({rows}, {cols}, figsize=({_value(figure.width)}, {_value(height)}), squeeze=False, "
           f"layout='constrained', sharey={figure.sharey!r})")
    artists: dict[int, dict[str, list[tuple[str, str]]]] = {}  # panel -> label -> (kind, variable)
    for i, panel in enumerate(panels):
        sc.add()
        sc.add(f"# panel {i}" + (f": {panel.title}" if panel.title else ""))
        ax = f"ax{i}"
        split = drawn[i].get("split")
        if split:  # two parts in the panel's place, the panel drawn in each (below), then each cut at the break
            start, cut0, cut1, end = split["edges"]
            sc.add(f"spec = axes[{i // cols}][{i % cols}].get_subplotspec()")
            sc.add(f"axes[{i // cols}][{i % cols}].remove()")
            if split["axis"] == "y":
                sc.add(f"grid = spec.subgridspec(2, 1, hspace=0.08, height_ratios=[{_value(end - cut1)}, {_value(cut0 - start)}])")
            else:
                sc.add(f"grid = spec.subgridspec(1, 2, wspace=0.06, width_ratios=[{_value(cut0 - start)}, {_value(end - cut1)}])")
            sc.add(f"{ax}_parts = (fig.add_subplot(grid[0]), fig.add_subplot(grid[1]))")
            if drawn[i]["right"] is not None:
                sc.add(f"{ax}r_parts = []")
            body_start = len(sc.lines)
        else:
            sc.add(f"{ax} = axes[{i // cols}][{i % cols}]")
        if panel.projection:  # a 3-D panel, in the flat one's place
            sc.add(f"{ax}.remove()")
            sc.add(f"{ax} = fig.add_subplot({rows}, {cols}, {i + 1}, projection={panel.projection!r})")
        twin = any(getattr(layer, "axis", "left") == "right" for layer in panel.layers)
        if twin:
            sc.add(f"{ax}r = {ax}.twinx()")
            if split:
                sc.add(f"{ax}r_parts.append({ax}r)")
        artists[i] = {}
        made = drawn[i]["artists"]
        seen: dict[tuple[str, str], int] = {}
        kinds = {Line: "line", Band: "band", Bars: "bars", Estimates: "line", Ecdf: "line", Line3d: "line", Surface: "wireframe",
                 Scatter3d: "points"}
        for layer in panel.layers:
            target = f"{ax}r" if getattr(layer, "axis", "left") == "right" else ax
            label = getattr(layer, "label", "")
            key = (kinds.get(type(layer), ""), label)
            k = seen.get(key, 0)  # this label's how-many-th artist of its kind
            seen[key] = k + 1
            if isinstance(layer, Line):
                x, y = sc.x(layer.y.index, drawn[i]["mid"]), sc.values(layer.y)
                line = [a for kind, a in made.get(label, []) if kind == "line"][k] if label in made else None
                color = _hex(line.get_color()) if line is not None else theme.palette[0]
                width = line.get_linewidth() if line is not None else theme.line_width
                style = line.get_linestyle() if line is not None else "-"
                var = sc.name("line")
                extra = ", drawstyle='steps-post'" if layer.step else ""
                sc.add(f"({var},) = {target}.plot({x}, {y}, linestyle={style!r}, color={color!r}, linewidth={_value(width)}, "
                       f"label={label!r}{extra})")
                artists[i].setdefault(label, []).append(("line", var))
            elif isinstance(layer, Band):
                x, lo_v, hi_v = sc.x(layer.lower.index, drawn[i]["mid"]), sc.values(layer.lower), sc.values(layer.upper)
                band = [a for kind, a in made.get(label, []) if kind == "band"][k] if label in made else None
                color = _hex(band.get_facecolor()[0]) if band is not None else theme.palette[0]
                alpha = band.get_alpha() if band is not None and band.get_alpha() is not None else layer.alpha
                var = sc.name("band")
                step = ", step='post'" if layer.step else ""
                sc.add(f"{var} = {target}.fill_between({x}, {lo_v}, {hi_v}, color={color!r}, alpha={_value(alpha)}, linewidth=0, "
                       f"label={label or None!r}{step})")
                artists[i].setdefault(label, []).append(("band", var))
            elif isinstance(layer, Bars):
                container = [a for kind, a in made.get(label, []) if kind == "bars"][k] if label in made else None
                var = sc.name("bars")
                patches = list(container) if container is not None else []
                color = _hex(patches[0].get_facecolor()) if patches else theme.palette[0]
                if layer.horizontal:
                    ys = [p.get_y() for p in patches]
                    sc.add(f"{var} = {target}.barh([{', '.join(_value(v) for v in ys)}], [{', '.join(_value(p.get_width()) for p in patches)}], "
                           f"height={_value(patches[0].get_height() if patches else 0.8)}, left=[{', '.join(_value(p.get_x()) for p in patches)}], "
                           f"align='edge', color={color!r}, label={label or None!r})")
                else:
                    from matplotlib import dates

                    dated = isinstance(layer.y.index, pd.PeriodIndex)
                    lefts = [p.get_x() for p in patches]
                    left_code = (f"pd.to_datetime({[dates.num2date(v).replace(tzinfo=None).isoformat() for v in lefts]!r})" if dated
                                 else f"[{', '.join(_value(v) for v in lefts)}]")
                    sc.add(f"{var} = {target}.bar({left_code}, [{', '.join(_value(p.get_height()) for p in patches)}], "
                           f"width={_value(patches[0].get_width() if patches else 0.8)}, bottom=[{', '.join(_value(p.get_y()) for p in patches)}], "
                           f"align='edge', color={color!r}, label={label or None!r})")
                artists[i].setdefault(label, []).append(("bars", var))
            elif isinstance(layer, Estimates):
                group = [e for e in panel.layers if isinstance(e, Estimates)]
                n, g = len(group), group.index(layer)
                drawn_line = [a for kind, a in made.get(label, []) if kind == "line"][k] if label in made else None
                color = _hex(drawn_line.get_color()) if drawn_line is not None else theme.palette[0]
                est, lo_v, hi_v = sc.values(layer.estimate), sc.values(layer.lower), sc.values(layer.upper)
                var = sc.name("estimates")
                errors = f"np.vstack([{est} - {lo_v}, {hi_v} - {est}])"
                style = f"fmt={layer.marker!r}, color={color!r}, markersize=4, capsize=2, linewidth=1.2, label={label or None!r}"
                if isinstance(layer.estimate.index, pd.PeriodIndex):
                    x = sc.x(layer.estimate.index, drawn[i]["mid"])
                    if n > 1:
                        gap = float(np.median(np.diff(layer.estimate.index.to_timestamp().to_numpy()).astype("timedelta64[D]").astype(float))) if len(layer.estimate) > 1 else 30.0
                        x = f"{x} + pd.Timedelta(days={_value((g - (n - 1) / 2) * 0.6 * gap / n)})"
                    sc.add(f"{var} = {target}.errorbar({x}, {est}, yerr={errors}, {style})")
                else:
                    names = _estimate_names(group)
                    places = [names.index(str(p)) + (g - (n - 1) / 2) * 0.6 / n for p in layer.estimate.index]
                    sc.add(f"{var} = {target}.errorbar({est}, np.array([{', '.join(_value(r) for r in places)}], float), xerr={errors}, {style})")
                    if g == 0:
                        sc.add(f"{target}.axvline(0, color='#888888', linewidth=0.8, linestyle='--')")
                artists[i].setdefault(label, []).append(("line", f"{var}.lines[0]"))
            elif isinstance(layer, QQ):
                v = sc.values(layer.values.dropna())
                sc.add(f"sd = {v}.std(ddof=1); sd = sd if sd > 0 else 1.0  # a constant series: all at zero")
                sc.add(f"z = np.sort(({v} - {v}.mean()) / sd)  # standardised, sorted")
                sc.add("p = (np.arange(1, len(z) + 1) - 0.5) / len(z)")
                sc.add("from scipy.stats import norm")
                sc.add(f"theory = norm.ppf(p)  # the normal quantiles at the same probabilities")
                sc.add(f"{ax}.scatter(theory, z, s=12, alpha=0.75, color={theme.palette[0]!r}, label={label or None!r})")
                sc.add(f"{ax}.axline((0, 0), slope=1, color='#888888', linewidth=0.9, linestyle='--')  # where normal values lie")
                sc.add(f"{ax}.set_xlabel({panel.xlabel or 'normal quantiles'!r}); {ax}.set_ylabel({panel.ylabel or 'standardised values'!r})")
            elif isinstance(layer, Ecdf):
                if not layer.values.notna().any():  # nothing to draw, as in pymetrix
                    continue
                found = [a for kind, a in made.get(label, []) if kind == "line"]
                drawn_line = found[k] if k < len(found) else None
                color = _hex(drawn_line.get_color()) if drawn_line is not None else theme.palette[0]
                width = drawn_line.get_linewidth() if drawn_line is not None else theme.line_width
                var = sc.name("ecdf")
                v = sc.values(layer.values.dropna())
                sc.add(f"{var} = {target}.ecdf({v}, complementary={layer.complementary!r}, color={color!r}, linewidth={_value(width)}, "
                       f"label={label or None!r})")
                sc.add(f"{target}.set_ylabel({panel.ylabel or ('share above' if layer.complementary else 'share at or below')!r})")
                artists[i].setdefault(label, []).append(("line", var))
            elif isinstance(layer, Hist2d):
                x, y = sc.values(layer.x), sc.values(layer.y)
                sc.add(f"ok = np.isfinite({x}) & np.isfinite({y})")
                sc.add(f"*_, image = {target}.hist2d({x}[ok], {y}[ok], bins={layer.bins}, cmap={layer.cmap!r}, cmin=1)")
                sc.add(f"fig.colorbar(image, ax={target}, shrink=0.9).set_label({layer.label!r})")
                sc.add(f"{target}.set_xlabel({panel.xlabel or str(layer.x.name or '')!r}); {target}.set_ylabel({panel.ylabel or str(layer.y.name or '')!r})")
            elif isinstance(layer, Hexbin):
                x, y = sc.values(layer.x), sc.values(layer.y)
                sc.add(f"ok = np.isfinite({x}) & np.isfinite({y})")
                sc.add(f"cells = {ax}.hexbin({x}[ok], {y}[ok], gridsize={layer.gridsize}, cmap={layer.cmap!r}, mincnt=1)")
                sc.add(f"fig.colorbar(cells, ax={ax}, shrink=0.9).set_label({layer.label!r})")
                sc.add(f"{ax}.set_xlabel({panel.xlabel or str(layer.x.name or '')!r}); {ax}.set_ylabel({panel.ylabel or str(layer.y.name or '')!r})")
            elif isinstance(layer, Violin):
                keep = [j for j, s in enumerate(layer.series) if s.notna().any()]
                if not keep:  # nothing to draw, as in pymetrix
                    continue
                names = [sc.values(layer.series[j].dropna()) for j in keep]
                sc.add(f"parts = {ax}.violinplot([{', '.join(names)}], positions={[j + 1 for j in keep]!r}, widths=0.7, "
                       f"showmedians={layer.medians!r})")
                sc.add(f"for body, color in zip(parts['bodies'], {[theme.palette[j % len(theme.palette)] for j in keep]!r}):")
                sc.add("    body.set_facecolor(color); body.set_edgecolor('#444444'); body.set_alpha(0.45)")
                sc.add("for name in ('cmins', 'cmaxes', 'cbars', 'cmedians'):  # the lines: the range and the median")
                sc.add("    if name in parts:")
                sc.add("        parts[name].set_color('#222222'); parts[name].set_linewidth(1.6 if name == 'cmedians' else 0.9)")
                labels = list(layer.labels) or [str(s.name or j + 1) for j, s in enumerate(layer.series)]
                sc.add(f"{ax}.set_xticks(range(1, {len(labels) + 1}), {labels!r})")
            elif isinstance(layer, Contour):
                (gx, xkind), (gy, ykind) = sc.grid(layer.values.columns), sc.grid(layer.values.index)
                z = sc.table(layer.values)
                levels = layer.levels if isinstance(layer.levels, (int, np.integer)) else [float(v) for v in layer.levels]
                colors = f"colors={layer.color!r}" if layer.color else f"cmap={layer.cmap!r}"
                sc.add(f"contours = {ax}.{'contourf' if layer.filled else 'contour'}({gx}, {gy}, {z}, levels={levels!r}, {colors})")
                if layer.clabel and layer.filled:  # the levels written on thin lines, as matplotlib advises for filled contours
                    sc.add(f"lines = {ax}.contour({gx}, {gy}, {z}, levels=contours.levels, colors='#333333', linewidths=0.5)")
                    sc.add(f"{ax}.clabel(lines, fmt='%.3g', fontsize='small')")
                elif layer.clabel:
                    sc.add(f"{ax}.clabel(contours, fmt='%.3g', fontsize='small')")
                if layer.colorbar:
                    sc.add("bar = fig.colorbar(contours, ax=" + ax + ", shrink=0.9)" + (f"; bar.set_label({layer.label!r})" if layer.label else ""))
                sc.add(f"{ax}.grid(False)")
                sc.label_axis(ax, "x", xkind, layer.values.columns, gx)
                sc.label_axis(ax, "y", ykind, layer.values.index, gy)
                sc.add(f"{ax}.set_xlabel({str(layer.values.columns.name or '')!r}); {ax}.set_ylabel({str(layer.values.index.name or '')!r})")
            elif isinstance(layer, Surface):
                (gx, xkind), (gy, ykind) = sc.grid(layer.values.columns), sc.grid(layer.values.index)
                z = sc.table(layer.values)
                sc.add(f"X, Y = np.meshgrid({gx}, {gy})  # the columns along x, the rows along y")
                if layer.wireframe:
                    found = [a for kind, a in made.get(label, []) if kind == "wireframe"]
                    color = _hex(found[k].get_color()[0]) if k < len(found) else theme.palette[0]
                    var = sc.name("wireframe")
                    sc.add(f"{var} = {ax}.plot_wireframe(X, Y, {z}, rstride=1, cstride=1, color={color!r}, linewidth=0.7, "
                           f"alpha={_value(layer.alpha)}, label={label or None!r})")
                    artists[i].setdefault(label, []).append(("wireframe", var))
                else:
                    sc.add(f"surface = {ax}.plot_surface(X, Y, {z}, rstride=1, cstride=1, cmap={layer.cmap!r}, linewidth=0, "
                           f"antialiased=False, alpha={_value(layer.alpha)})")
                    sc.add(f"bar = fig.colorbar(surface, ax={ax}, shrink=0.6, pad=0.1)" + (f"; bar.set_label({label!r})" if label else ""))
                    sc.add(f"{ax}.set_anchor('C')  # centred: a 3-D box cannot widen to fill its space")
                sc.label_axis(ax, "x", xkind, layer.values.columns, gx)
                sc.label_axis(ax, "y", ykind, layer.values.index, gy)
            elif isinstance(layer, (Line3d, Scatter3d)):
                x, y, z = (sc.coordinates(getattr(layer, a)) for a in ("x", "y", "z"))
                kind = "line" if isinstance(layer, Line3d) else "points"
                found = [a for knd, a in made.get(label, []) if knd == kind]
                artist = found[k] if k < len(found) else None
                var = sc.name(kind)
                if isinstance(layer, Line3d):
                    color = _hex(artist.get_color()) if artist is not None else theme.palette[0]
                    width = artist.get_linewidth() if artist is not None else theme.line_width
                    sc.add(f"({var},) = {ax}.plot({x}, {y}, {z}, color={color!r}, linewidth={_value(width)}, label={label or None!r})")
                else:
                    color = _hex(artist.get_facecolor()[0]) if artist is not None else theme.palette[0]
                    sc.add(f"ok = np.isfinite({x}) & np.isfinite({y}) & np.isfinite({z})")
                    sc.add(f"{var} = {ax}.scatter({x}[ok], {y}[ok], {z}[ok], s=12, alpha=0.8, color={color!r}, label={label or None!r})")
                artists[i].setdefault(label, []).append((kind, var))
                for a in ("x", "y", "z"):
                    if isinstance(getattr(layer, a).dtype, pd.PeriodDtype) or pd.api.types.is_datetime64_any_dtype(getattr(layer, a).dtype):
                        sc.label_axis(ax, a, "date", pd.Index([]), "")
            elif isinstance(layer, Heatmap):
                cmap, lo, hi = _heatmap_scale(layer)
                frame = layer.values
                by_column = _heatmap_mode(layer) == "column"
                sc.add(f"cells = np.array({[[float(v) for v in row] for row in frame.to_numpy(float)]!r})".replace("nan", "np.nan"))
                if by_column:
                    sc.add("lo, hi = np.nanmin(cells, axis=0), np.nanmax(cells, axis=0)  # each column on its own scale")
                    sc.add("shades = np.where(hi > lo, (cells - lo) / np.where(hi > lo, hi - lo, 1), np.where(np.isnan(cells), np.nan, 0.5))")
                sc.add(f"image = {ax}.imshow({'shades' if by_column else 'cells'}, cmap={cmap!r}, vmin={_value(lo)}, vmax={_value(hi)}, "
                       "aspect='auto', interpolation='nearest')")
                rows_, cols_ = [str(r) for r in frame.index], [str(c) for c in frame.columns]
                long = max(map(len, cols_), default=0) * len(cols_) > 60
                sc.add(f"{ax}.set_yticks(range({len(rows_)}), {rows_!r})")
                sc.add(f"{ax}.set_xticks(range({len(cols_)}), {cols_!r}" + (", rotation=45, ha='right')" if long else ")"))
                sc.add(f"{ax}.grid(False); {ax}.tick_params(length=0)")
                if by_column:
                    sc.add(f"{ax}.set_xlabel({COLUMN_SCALE_NOTE!r}, style='italic', color='#555555')  # in place of a colour bar")
                else:
                    sc.add(f"bar = fig.colorbar(image, ax={ax}, shrink=0.9)" + (f"; bar.set_label({layer.label!r})" if layer.label else ""))
                if layer.annotate and frame.size <= 400:
                    sc.add("def luminance(rgba):  # WCAG relative luminance")
                    sc.add("    return sum(w * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) "
                           "for w, c in zip((0.2126, 0.7152, 0.0722), rgba[:3]))")
                    sc.add("def contrast(a, b):")
                    sc.add("    a, b = sorted((luminance(a), luminance(b)), reverse=True)")
                    sc.add("    return (a + 0.05) / (b + 0.05)")
                    sc.add("def text_colour(cell):  # dark grey or white, whichever reads better; black where dark grey falls short of 4.5:1")
                    sc.add(f"    best = max(({_DARK_TEXT!r}, {_WHITE_TEXT!r}), key=lambda t: contrast(mcolors.to_rgba(t), cell))")
                    sc.add("    if contrast(mcolors.to_rgba(best), cell) >= 4.5:")
                    sc.add("        return best")
                    sc.add(f"    return max(({_BLACK_TEXT!r}, {_WHITE_TEXT!r}), key=lambda t: contrast(mcolors.to_rgba(t), cell))")
                    sc.add("for (r, c), v in np.ndenumerate(cells):  # each value in its cell")
                    sc.add("    if np.isfinite(v):")
                    sc.add(f"        text = {layer.fmt!r}.format(v)")
                    sc.add("        text = text[1:] if text.startswith('-') and not text.strip('-0.,% ') else text  # no -0.00")
                    shade = "shades[r, c]" if by_column else "v"
                    sc.add(f"        {ax}.text(c, r, text, ha='center', va='center', fontsize='small', "
                           f"color=text_colour(image.to_rgba({shade})))")
            elif isinstance(layer, Rule):
                func = "axvline" if layer.vertical else "axhline"
                if isinstance(layer.y, str) and drawn[i]["freq"] is not None:  # a date: its period's start, if it is shown
                    at = f"pd.Period({layer.y!r}).start_time"
                    sc.add(f"if {ax}.get_xlim()[0] <= mdates.date2num({at}) <= {ax}.get_xlim()[1]:")
                    sc.add(f"    {target}.{func}({at}, color={layer.color!r}, linewidth=0.8, linestyle={layer.style!r})")
                else:
                    sc.add(f"{target}.{func}({_value(layer.y)}, color={layer.color!r}, linewidth=0.8, linestyle={layer.style!r})")
            elif isinstance(layer, Scatter):
                x, y = sc.values(layer.x), sc.values(layer.y)
                sc.add(f"ok = np.isfinite({x}) & np.isfinite({y})")
                sc.add(f"{ax}.scatter({x}[ok], {y}[ok], s=12, alpha=0.7, color={theme.palette[0]!r}, label={label or None!r})")
                if layer.fit:
                    sc.add(f"slope, intercept = np.polyfit({x}[ok], {y}[ok], 1)")
                    sc.add(f"grid = np.linspace({x}[ok].min(), {x}[ok].max(), 50)")
                    sc.add(f"{ax}.plot(grid, intercept + slope * grid, color='#444444', linewidth=1.2, label=f'fit: slope {{slope:.3g}}')")
                sc.add(f"{ax}.set_xlabel({str(layer.x.name or '')!r}); {ax}.set_ylabel({str(layer.y.name or '')!r})")
            elif isinstance(layer, Histogram):
                if not layer.values.notna().any():  # nothing to draw, as in pymetrix
                    continue
                v = sc.values(layer.values.dropna())
                sc.add(f"{ax}.hist({v}, bins={layer.bins}, density=True, color='#56B4E9', edgecolor='white', linewidth=0.4, label={label or None!r})")
                if layer.normal and layer.values.notna().sum() > 1:  # as pymetrix: a curve needs two values
                    sc.add(f"grid = np.linspace({v}.min(), {v}.max(), 200); m, sd = {v}.mean(), {v}.std(ddof=1)")
                    sc.add(f"{ax}.plot(grid, np.exp(-0.5 * ((grid - m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color='#D55E00', linewidth=1.4, label='normal')")
            elif isinstance(layer, Box):
                names = [sc.values(s.dropna()) for s in layer.series]
                sc.add(f"boxes = {ax}.boxplot([{', '.join(names)}], widths=0.5, patch_artist=True, medianprops={{'color': '#222222', 'linewidth': 1.6}})")
                sc.add(f"for j, patch in enumerate(boxes['boxes']):")
                sc.add(f"    patch.set_facecolor({list(theme.palette)!r}[j % {len(theme.palette)}]); patch.set_alpha(0.45)")
                labels = list(layer.labels) or [str(s.name or j + 1) for j, s in enumerate(layer.series)]
                sc.add(f"{ax}.set_xticks(range(1, {len(labels) + 1}), {labels!r})")
        categories = drawn[i]["categories"]
        if categories is not None:  # once per panel, as pymetrix: at most about 30 labels
            step = max(1, int(np.ceil(len(categories) / 30)))
            positions = list(range(0, len(categories), step))
            shown = [categories[j] for j in positions]
            if drawn[i]["horizontal"]:
                sc.add(f"{ax}.set_yticks({positions!r}, {shown!r})")
                sc.add(f"{ax}.invert_yaxis()  # the first at the top, as a table reads")
            else:
                wide = max(map(len, shown), default=0) * len(shown) > 60
                sc.add(f"{ax}.set_xticks({positions!r}, {shown!r}" + (", rotation=45, ha='right')" if wide else ")"))
        if panel.title:
            sc.add(f"{ax}.set_title({panel.title!r}, loc='left')")
        if panel.projection:  # the axes' names, as pymetrix gave them
            made_ax = drawn[i]["ax"]
            names = {a: getattr(made_ax, f"get_{a}label")() for a in ("x", "y", "z")}
            sc.add(f"{ax}.set_xlabel({names['x']!r}); {ax}.set_ylabel({names['y']!r}); {ax}.set_zlabel({names['z']!r})")
        elif panel.ylabel:
            sc.add(f"{ax}.set_ylabel({panel.ylabel!r})")
        if panel.xlabel:
            sc.add(f"{ax}.set_xlabel({panel.xlabel!r})")
        if twin and panel.ylabel_right:
            sc.add(f"{ax}r.set_ylabel({panel.ylabel_right!r})")
        _options(sc, figure, i, ax, panel, drawn[i], recessions, artists[i])
        _series_styles(sc, figure, i, artists[i])  # before the legend, as pymetrix styles them, so the legend shows them
        if drawn[i]["ax"].get_legend() is not None:
            texts = [t.get_text() for t in drawn[i]["ax"].get_legend().get_texts()]
            sc.add(f"handles, names = {ax}.get_legend_handles_labels()")
            if twin:
                sc.add(f"more, more_names = {ax}r.get_legend_handles_labels(); handles, names = handles + more, names + more_names")
            sc.add(f"keep = {texts!r}")
            from .figures import legend_place

            place = "".join(f", {k}={v!r}" for k, v in legend_place(panel.legend_loc, len(texts)).items())
            sc.add(f"{ax}.legend([h for h, n in zip(handles, names) if n in keep], [n for n in names if n in keep], frameon=False{place})")
        if split:
            body = sc.lines[body_start:]
            sc.lines[body_start:] = [f"for {ax} in {ax}_parts:  # the panel, drawn in each part", *(f"    {line}" if line else line for line in body)]
            sc.add(f"{ax} = {ax}_parts[0]  # the first part: its title and legend")
            if twin:
                sc.add(f"{ax}r = {ax}r_parts[0]")
    splits = {i: drawn[i]["right"] is not None for i in range(len(panels)) if drawn[i].get("split")}
    _styles(sc, figure, {i: drawn[i]["freq"] is not None for i in range(len(panels))}, {i for i, p in enumerate(panels) if p.projection},
            {i: drawn[i]["mid"] for i in range(len(panels))}, splits)
    for i in splits:
        _split_code(sc, f"ax{i}", drawn[i])
    for j in range(len(panels), rows * cols):
        sc.add(f"axes[{j // cols}][{j % cols}].set_visible(False)")
    if figure.title or figure.subtitle:
        sc.add(f"fig.suptitle({chr(10).join(x for x in (figure.title, figure.subtitle) if x)!r})")
    if figure.footnote:
        sc.add(f"fig.supxlabel({figure.footnote!r}, x=0.01, ha='left', style='italic', color='#555555')")
    body = "\n".join(sc.lines)
    head = ['"""' + (figure.title or "A pymetrix figure") + ", drawn with matplotlib alone (from Figure.matplotlib_code()).",
            "",
            "Every number drawn is below; each option and style is the matplotlib call it is.", '"""',
            "import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd",
            *(["from matplotlib import dates as mdates"] if "mdates." in body + "\n".join(sc.data) else []),
            *(["from matplotlib import colors as mcolors"] if "mcolors." in body else []),
            *(["from matplotlib.ticker import StrMethodFormatter"] if "StrMethodFormatter" in body else []),
            "", "nan = float('nan')", "", "# the data", *sc.data, "", "# the figure"]
    return "\n".join([*head, body, "", "if __name__ == '__main__':", "    plt.show()", ""])


def _options(sc: _Script, figure: Any, i: int, ax: str, panel: Any, drawn: dict[str, Any], recessions: Any,
             made: dict[str, list[tuple[str, str]]]) -> None:
    """The figure options on one panel: shading, events, reference lines, scale and limits, values on bars, line names."""
    freq = drawn["freq"]
    horizontal = drawn["horizontal"]
    if panel.projection:  # a 3-D panel: no dates to shade or mark, nor lines across; limits on the heights
        if figure.ylim is not None:
            sc.add(f"{ax}.set_zlim({_value(figure.ylim[0]) if figure.ylim[0] is not None else None}, "
                   f"{_value(figure.ylim[1]) if figure.ylim[1] is not None else None})")
        return
    if figure.shade and freq is not None:
        spans = []
        for start, end, label in figure.shade:
            spans += list(recessions(start[1:], freq)) if start.startswith("@") else [(start, end, label)]
        bounds = [(f"mdates.date2num(pd.Period({str(pd.Period(s, freq=freq))!r}).to_timestamp())",
                   f"mdates.date2num(pd.Period({str(pd.Period(e, freq=freq) + 1)!r}).to_timestamp())", label) for s, e, label in spans]
        sc.add(f"lo, hi = {ax}.get_xlim()  # the data's dates: shading outside them is left out, and never widens the axis")
        sc.add("shaded = set()")
        sc.add(f"for left, right, label in [{', '.join(f'({a}, {b}, {c!r})' for a, b, c in bounds)}]:")
        sc.add("    if right <= lo or left >= hi:")
        sc.add("        continue")
        sc.add(f"    {ax}.axvspan(max(left, lo), min(right, hi), color='#7F7F7F', alpha=0.18, linewidth=0, zorder=0, "
               "label=label if label and label not in shaded else None)")
        sc.add("    shaded.add(label)")
        sc.add(f"{ax}.set_xlim(lo, hi)")
    if freq is not None:
        for period, label in figure.events:
            when = f"pd.Period({str(pd.Period(period, freq=freq))!r}).to_timestamp()"
            sc.add(f"if {ax}.get_xlim()[0] <= mdates.date2num({when}) <= {ax}.get_xlim()[1]:  # only where shown, as pymetrix")
            sc.add(f"    {ax}.axvline({when}, color='#555555', linewidth=0.9, linestyle='--')")
            if label:
                sc.add(f"    {ax}.text({when}, 0.98, {label + ' '!r}, transform={ax}.get_xaxis_transform(), rotation=90, "
                       f"ha='right', va='top', fontsize='small')")
    for value, label in figure.reference:
        if horizontal:  # values run across a horizontal panel
            sc.add(f"{ax}.axvline({_value(value)}, color='#555555', linewidth=0.9, linestyle=':')")
            if label:
                sc.add(f"{ax}.text({_value(value)}, 1.0, {' ' + label!r}, transform={ax}.get_xaxis_transform(), ha='left', "
                       f"va='top', fontsize='small')")
        else:
            sc.add(f"{ax}.axhline({_value(value)}, color='#555555', linewidth=0.9, linestyle=':')")
            if label:
                sc.add(f"{ax}.text(1.0, {_value(value)}, {label + ' '!r}, transform={ax}.get_yaxis_transform(), ha='right', "
                       f"va='bottom', fontsize='small')")
    bars = [var for items in made.values() for kind, var in items if kind == "bars"]
    if figure.bar_labels and bars:
        from .figures import Bars

        stacked = any(isinstance(layer, Bars) and layer.stacked for layer in panel.layers)
        group = "(" + ", ".join(bars) + ",)"
        sc.add(f"values = np.abs(np.concatenate([np.asarray(bars.datavalues, float) for bars in {group}]))")
        sc.add("fmt = '{:,.0f}' if np.nanmax(values) >= 100 else '{:,.1f}'  # one format: the decimals the largest value needs")
        sc.add(f"for bars in {group}:")
        sc.add(f"    {ax}.bar_label(bars, labels=[fmt.format(v) if np.isfinite(v) and v != 0 else '' for v in bars.datavalues], "
               + ("label_type='center', color='white', fontsize='small')" if stacked else "padding=2, color='#333333', fontsize='small')"))
    if figure.log:
        sc.add(f"{ax}.set_{'x' if horizontal else 'y'}scale('log')")
    if figure.ylim is not None:
        sc.add(f"{ax}.set_{'x' if horizontal else 'y'}lim({_value(figure.ylim[0]) if figure.ylim[0] is not None else None}, "
               f"{_value(figure.ylim[1]) if figure.ylim[1] is not None else None})")
    if figure.line_labels:
        sides = f"({ax}, {ax}r)" if drawn["right"] is not None else f"({ax},)"
        sc.add(f"left, right = {ax}.get_xlim()")
        sc.add(f"{ax}.set_xlim(left, right + 0.16 * (right - left))  # room for the names")
        sc.add(f"for side in {sides}:  # each line named at its last value, in its colour")
        sc.add("    for line in side.get_lines():")
        sc.add("        xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata(), float)")
        sc.add("        ok = np.flatnonzero(np.isfinite(ys))")
        sc.add("        if line.get_label() and not line.get_label().startswith('_') and len(ok):")
        sc.add("            side.annotate(line.get_label(), (xs[ok[-1]], ys[ok[-1]]), xytext=(4, 0), textcoords='offset points', "
               "va='center', color=line.get_color(), fontsize='small')")


def _series_styles(sc: _Script, figure: Any, i: int, made: dict[str, list[tuple[str, str]]]) -> None:
    """The recorded line, bars and band calls that reach panel ``i``, in their order."""
    from .figures import _panel_indexes

    for call in figure.style:
        if call.kind not in ("line", "bars", "band"):
            continue
        ax_spec, label = call.where
        if i not in _panel_indexes(figure, ax_spec):
            continue
        p = dict(call.props)
        for kind, var in made.get(label, []) + made.get(f"{label} (right)", []):
            if kind != call.kind:
                continue
            if kind == "bars":
                sc.add(f"for bar in {var}:")
                sc.add(f"    bar.set({_kwargs(p)})")
            else:
                sc.add(f"{var}.set({_kwargs(p)})")


def _change_code(sc: _Script, figure: Any, i: int, ax: str, props: dict[str, Any], mid: bool) -> None:
    """A ``Figure.change`` call as matplotlib draws it: the values read from the data now, written as numbers."""
    from .figures import CHANGE_STYLES, _period_x, change_between

    panel = figure._panels()[i]
    if props.get("series") not in [getattr(layer, "label", None) for layer in panel.layers]:
        return
    c = change_between(panel, props["series"], props["start"], props["end"], props.get("show", "both"),
                       props.get("units", "auto"), bool(props.get("annualise", False)))

    def at(when: str) -> str:
        if not c["dated"]:
            return _value(float(when))
        return f"pd.Timestamp({str(_period_x(pd.PeriodIndex([pd.Period(when)]), mid)[0])!r})"

    color, lw = props.get("color", "#333333"), float(props.get("linewidth", 1.2))
    sc.add(f"# the change in {props['series']} from {c['start']} to {c['end']}: {c['label']!r}")
    sc.add(f"cx0, cy0, cx1, cy1 = {at(c['start'])}, {_value(c['y0'])}, {at(c['end'])}, {_value(c['y1'])}")
    sc.add(f"{ax}.annotate('', xy=(cx1, cy1), xytext=(cx0, cy0), arrowprops={{'arrowstyle': {CHANGE_STYLES[props.get('style', 'arrow')]!r}, "
           f"'color': {color!r}, 'lw': {lw!r}, 'shrinkA': 0, 'shrinkB': 0}})")
    sc.add(f"{ax}.plot([cx0, cx1], [cy0, cy1], linestyle='none', marker='o', markersize=4, color={color!r})")
    sc.add(f"{ax}.annotate({c['label']!r}, xy=(cx0 + (cx1 - cx0) / 2, (cy0 + cy1) / 2), xytext=(6, 6), textcoords='offset points', "
           f"fontsize={float(props.get('fontsize', 8.5))!r}, color={color!r}, ha='left', va='bottom', "
           "bbox={'boxstyle': 'round,pad=0.25', 'fc': 'white', 'ec': 'none', 'alpha': 0.85})")


def _each_part(sc: _Script, indexes: list[int], splits: dict[int, bool], kind: str) -> Any:
    """The panels a call reaches; a split panel's lines (written by the loop's body) wrapped to run on both its parts,
    as pymetrix styles them (a legend: the first part's only)."""
    for i in indexes:
        mark = len(sc.lines)
        yield i
        if i in splits and kind != "legend":
            ax, twin = f"ax{i}", splits[i]
            head = f"for {ax}, {ax}r in zip({ax}_parts, {ax}r_parts):" if twin else f"for {ax} in {ax}_parts:"
            sc.lines[mark:] = [head, *(f"    {line}" if line else line for line in sc.lines[mark:]),
                               f"{ax} = {ax}_parts[0]", *([f"{ax}r = {ax}r_parts[0]"] if twin else [])]


def _split_code(sc: _Script, ax: str, drawn: dict[str, Any]) -> None:
    """A split panel's two parts cut at the break, as pymetrix finishes them (``figures._finish_split``)."""
    from .figures import BREAK_MARK, BREAK_MARK_X

    start, cut0, cut1, end = drawn["split"]["edges"]
    sc.add()
    sc.add(f"# {ax}: split at the break, the inner edges taken away, break marks where the axis is cut")
    sc.add(f"a, b = {ax}_parts")
    sc.add("if b.get_legend() is not None:")
    sc.add("    b.get_legend().remove()")
    sc.add("for loc in ('left', 'center', 'right'):")
    sc.add("    b.set_title('', loc=loc)")
    if drawn["split"]["axis"] == "y":
        sc.add(f"a.set_ylim({_value(cut1)}, {_value(end)}); b.set_ylim({_value(start)}, {_value(cut0)})")
        sc.add("a.spines['bottom'].set_visible(False); a.tick_params(axis='x', bottom=False, labelbottom=False); a.set_xlabel('')")
        sc.add("b.set_ylabel('')")
        sc.add(f"mark = {BREAK_MARK!r}")
        sc.add("a.plot([0], [0], transform=a.transAxes, **mark); b.plot([0], [1], transform=b.transAxes, **mark)")
    else:
        sc.add(f"a.set_xlim({_value(start)}, {_value(cut0)}); b.set_xlim({_value(cut1)}, {_value(end)})")
        sc.add("a.spines['right'].set_visible(False); b.spines['left'].set_visible(False)")
        sc.add("b.tick_params(axis='y', left=False, labelleft=False); b.set_ylabel('')")
        if drawn["right"] is not None:
            sc.add(f"ar, br = {ax}r_parts")
            sc.add("ar.tick_params(axis='y', right=False, labelright=False); ar.spines['right'].set_visible(False); ar.set_ylabel('')")
            sc.add("br.spines['left'].set_visible(False)")
        sc.add(f"mark = {BREAK_MARK_X!r}")
        sc.add("a.plot([1], [0], transform=a.transAxes, **mark); b.plot([0], [0], transform=b.transAxes, **mark)")


def _styles(sc: _Script, figure: Any, dated: dict[int, bool], solid: set[int] = frozenset(),  # type: ignore[assignment]
            mids: dict[int, bool] | None = None, splits: dict[int, bool] | None = None) -> None:
    """The recorded axes, legend, layout and added calls, as the matplotlib calls they are, in their order."""
    from .figures import _panel_indexes

    rest = [call for call in figure.style if call.kind not in ("line", "bars", "band")]
    if rest:
        sc.add()
        sc.add("# the recorded styles")
    for call in rest:
        p = dict(call.props)
        if call.kind == "layout":
            if "figsize" in p:
                sc.add(f"fig.set_size_inches({_value(p['figsize'][0])}, {_value(p['figsize'][1])})")
            if "dpi" in p:
                sc.add(f"fig.set_dpi({_value(p['dpi'])})")
            if "facecolor" in p:
                sc.add(f"fig.set_facecolor({p['facecolor']!r})")
            if "figwidth" in p:
                sc.add(f"fig.set_figwidth({_value(p['figwidth'])})")
            if "figheight" in p:
                sc.add(f"fig.set_figheight({_value(p['figheight'])})")
            for key in ("suptitle", "supxlabel", "supylabel"):
                if key in p:
                    sc.add(f"fig.{key}({p[key]!r})")
            continue
        for i in _each_part(sc, _panel_indexes(figure, call.where[0]), splits or {}, call.kind):
            p = dict(call.props)  # each panel its own copy: the branches below take arguments out
            on_dates = dated.get(i, True)
            ax = f"ax{i}" + ("r" if call.kind in ("set", "call") and call.where[1] == "right" else "")
            if call.kind == "set":
                sc.add(f"{ax}.set({_kwargs(p, dates=('xlim', 'xbound'))})")
            elif call.kind == "ticks":
                fmt = p.pop("format", None)
                if p:
                    sc.add(f"{ax}.tick_params(axis={call.where[1]!r}, {_kwargs(p)})")
                if fmt is not None:
                    for name in ("x", "y") if call.where[1] == "both" else (call.where[1],):
                        sc.add(f"{ax}.{name}axis.set_major_formatter(StrMethodFormatter({fmt!r}))")
            elif call.kind == "grid":
                sc.add(f"{ax}.grid({_kwargs(p)})")
            elif call.kind == "spines":
                sides = ("left", "right", "top", "bottom") if call.where[1] == "all" else (call.where[1],)
                sc.add(f"for side in {sides!r}:")
                sc.add(f"    {ax}.spines[side].set({_kwargs(p)})")
            elif call.kind == "legend":
                if p.pop("visible", True) is False:
                    sc.add(f"if {ax}.get_legend() is not None:")
                    sc.add(f"    {ax}.get_legend().remove()")
                else:  # as pymetrix: the legend's own entries (both axes'), restyled
                    sc.add(f"old = {ax}.get_legend()")
                    sc.add(f"handles, names = (list(old.legend_handles), [t.get_text() for t in old.get_texts()]) if old is not None "
                           f"else {ax}.get_legend_handles_labels()")
                    sc.add("if handles:")
                    sc.add(f"    {ax}.legend(handles, names, {_kwargs({'frameon': False, **p})})")  # the call's own frameon wins
            elif call.kind == "change":
                _change_code(sc, figure, i, ax, p, (mids or {}).get(i, False))
            elif call.kind == "annotate":
                text, xy = p.pop("text"), p.pop("xy")
                xytext = p.pop("xytext", None)
                textcoords = p.get("textcoords", "data")
                xt = "" if xytext is None else (f", xytext=({_when(xytext[0], on_dates)}, {_value(xytext[1])})" if textcoords == "data"
                                                else f", xytext={tuple(xytext)!r}")
                rest = f", {_kwargs(p)}" if p else ""
                sc.add(f"{ax}.annotate({text!r}, ({_when(xy[0], on_dates)}, {_value(xy[1])}){xt}{rest})")
            elif call.kind == "text":
                x, y, s = p.pop("x"), p.pop("y"), p.pop("s")
                sc.add(f"{ax}.text({_when(x, on_dates)}, {_value(y)}, {s!r}" + (f", {_kwargs(p)})" if p else ")"))
            elif call.kind == "axhline":
                y = p.pop("y")
                sc.add(f"{ax}.axhline({_value(y)}" + (f", {_kwargs(p)})" if p else ")"))
            elif call.kind == "axvline":
                x = p.pop("x")
                sc.add(f"{ax}.axvline({_when(x, on_dates)}" + (f", {_kwargs(p)})" if p else ")"))
            elif call.kind == "axhspan":
                a, b = p.pop("ymin"), p.pop("ymax")
                sc.add(f"{ax}.axhspan({_value(a)}, {_value(b)}" + (f", {_kwargs(p)})" if p else ")"))
            elif call.kind == "axvspan":
                a, b = p.pop("xmin"), p.pop("xmax")
                end = f"(pd.Period({str(b)!r}) + 1).to_timestamp()" if isinstance(b, str) and on_dates else _when(b, on_dates)
                sc.add(f"{ax}.axvspan({_when(a, on_dates)}, {end}" + (f", {_kwargs(p)})" if p else ")"))
            elif call.kind == "view_init":
                if i in solid:  # turns 3-D panels; pymetrix leaves the flat ones as they are
                    sc.add(f"{ax}.view_init({_kwargs(p)})")
            elif call.kind == "call":
                args = ", ".join([*(repr(a) for a in p.get("args", ())), *(f"{k}={v!r}" for k, v in p.get("kwargs", {}).items())])
                sc.add(f"{ax}.{p['method']}({args})")
