"""The house look of pymetrix's charts: one set of matplotlib settings every pymetrix figure is drawn with.

Every ``Figure`` is drawn inside ``matplotlib.rc_context(chart_rc(...))``, so the look reaches the app, scripts,
notebooks, reports and exported files alike, and a user's own matplotlib plots are never touched: the global
``rcParams`` are the same after a pymetrix figure is drawn as before.

- Type: the app passes its own UI font family (when matplotlib knows it); elsewhere DejaVu Sans, which matplotlib
  always ships, so a chart looks the same on every machine. Sizes come from the figure's theme; on screen in the app
  they follow its text size (``Surface.text_scale``, ``scaled``), and every exported or copied file keeps the theme's.
- Grid: major lines only, light grey, behind the data. No top or right spines. Ticks point outwards, short.
- Series colours: the theme's palette in a fixed order; the default theme's is Okabe-Ito (see ``themes.OKABE_ITO``),
  which stays distinguishable with the common colour-vision deficiencies.
- Surface: light (white, dark text) by default and for every exported file. The app draws on its own window colours
  (``surface_from``): a dark palette gives a dark chart, with dark greys in the chart (an "actual" line, reference
  lines, notes) turned light so they still read (``on_surface``).

Figure sizes are in inches and never change with the look: sharpness on high-resolution screens comes from the
app's canvas drawing at the screen's device pixel ratio, and from the dpi a PNG is saved at.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = ["DARK", "FALLBACK_FONT", "LIGHT", "SCREEN_DPI", "Surface", "chart_rc", "known_font", "on_surface", "scaled",
           "surface_from"]

FALLBACK_FONT = "DejaVu Sans"  # shipped with matplotlib: always there
SCREEN_DPI = 100.0  # matplotlib's figure.dpi: inches to logical screen pixels, before the device pixel ratio


@dataclass(frozen=True)
class Surface:
    """What a chart is drawn on: its background, the colour of its text, axes and ticks, and of its grid."""

    background: str = "#FFFFFF"
    foreground: str = "#262626"  # titles, labels, tick labels, spines and ticks
    grid: str = "#E3E3E3"  # light grey: there when looked for, never competing with the data
    dark: bool = False
    text_scale: float = 1.0  # the app's text size (View → Text size) on screen; 1 for every exported file


LIGHT = Surface()
DARK = Surface("#1E1F22", "#E3E3E3", "#3A3C40", dark=True)


def _rgb(color: Any) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return to_rgb(color)


def _mix(a: Any, b: Any, share: float) -> str:
    """``share`` of the way from colour ``a`` to colour ``b``."""
    from matplotlib.colors import to_hex

    return to_hex(tuple(x + share * (y - x) for x, y in zip(_rgb(a), _rgb(b))))


def surface_from(background: Any, foreground: Any) -> Surface:
    """The surface of a window's colours (an app palette's window and text colours): dark when the background is darker
    than mid-grey; the grid a little way from the background towards the text."""
    from matplotlib.colors import to_hex

    _, light, _ = colorsys.rgb_to_hls(*_rgb(background))
    dark = light < 0.5
    return Surface(to_hex(_rgb(background)), to_hex(_rgb(foreground)), _mix(background, foreground, 0.16 if dark else 0.12), dark)


def scaled(theme: Any, scale: float) -> Any:
    """``theme`` with its font sizes at ``scale`` of its own: the app's text size, for a chart on screen (every size a
    chart's text takes comes from the theme's two). ``theme`` itself at 1."""
    if scale == 1 or not scale:
        return theme
    from dataclasses import replace

    return replace(theme, font_size=float(theme.font_size) * scale, title_size=float(theme.title_size) * scale)


@lru_cache(maxsize=64)
def known_font(family: str | None) -> str | None:
    """``family`` if matplotlib can find a font of that name (so asking for it never logs "font not found"), else None."""
    if not family:
        return None
    from matplotlib import font_manager

    wanted = family.strip().lower()
    return family.strip() if any(f.name.lower() == wanted for f in font_manager.fontManager.ttflist) else None


def chart_rc(theme: Any, font: str | None = None, surface: Surface = LIGHT) -> dict[str, Any]:
    """The matplotlib settings (rcParams) a pymetrix figure is drawn with, for use in ``rc_context``: the house look,
    then the theme's sizes and palette, then the theme's own settings (a style sheet's), which win.

    ``font`` is a font family to use first (the app's UI font); the theme's ``font_family``, when it names one, wins
    over it. ``surface`` is what the chart is drawn on (``LIGHT``, or the app's own colours)."""
    from cycler import cycler

    families = [f for f in (theme.font_family or None, known_font(font), FALLBACK_FONT) if f]
    size, title = float(theme.font_size), float(theme.title_size)
    ink, grid = surface.foreground, surface.grid
    rc: dict[str, Any] = {
        # type
        "font.family": list(dict.fromkeys(families)),
        "font.size": size,
        "axes.titlesize": title,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size - 0.5,
        "legend.title_fontsize": size,
        "figure.titlesize": title + 1.5,
        # (panel titles are placed on the left by the drawing code itself; axes.titlelocation is left at matplotlib's
        # default so a title set by a style call or a plugin stays where get_title() finds it)
        "axes.titlepad": 5.0,
        # surface and ink
        "figure.facecolor": surface.background,
        "axes.facecolor": surface.background,
        "savefig.facecolor": "auto",  # the figure's own: light for every file unless the app's surface was asked for
        "text.color": ink,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.edgecolor": _mix(ink, surface.background, 0.25),
        "xtick.color": _mix(ink, surface.background, 0.25),
        "ytick.color": _mix(ink, surface.background, 0.25),
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
        "legend.labelcolor": ink,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        # spines: left and bottom only
        "axes.spines.top": False,
        "axes.spines.right": False,
        # ticks: outwards and short, major only
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
        # grid: major lines, light grey, behind the data
        "axes.grid": bool(theme.grid),
        "axes.grid.which": "major",
        "axes.grid.axis": "both",
        "grid.color": grid,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        # series
        "axes.prop_cycle": cycler(color=list(theme.palette)),
        "lines.linewidth": float(theme.line_width),
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.5,
        # margins: a little room at the ends, nothing clipped (constrained layout makes room for the labels)
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.05,
        "figure.dpi": SCREEN_DPI,
        "figure.constrained_layout.h_pad": 0.05,
        "figure.constrained_layout.w_pad": 0.05,
    }
    rc.update(dict(theme.rc))  # a theme's style sheet has the last word
    return rc


def on_surface(fig: Any, surface: Surface) -> None:
    """Fit what pymetrix drew in fixed dark greys (an "actual" line, reference lines and their labels, notes, the black of
    a palette) to a dark surface: each neutral grey darker than mid-grey takes the lightness opposite it, and the
    white boxes behind labels take the surface's colour. Colours (the palette's hues) and colour maps are left as they
    are. Nothing to do on a light surface."""
    if not surface.dark:
        return
    from matplotlib.colors import to_rgba

    background = to_rgba(surface.background)

    def turned(color: Any, boxes: bool = False) -> Any:
        try:
            r, g, b, a = to_rgba(color)
        except (ValueError, TypeError):
            return None
        hue, light, saturation = colorsys.rgb_to_hls(r, g, b)
        if saturation > 0.12 and 0.04 < light < 0.96:  # a colour, not a grey: keep it
            return None
        if light < 0.5:
            r, g, b = colorsys.hls_to_rgb(hue, min(0.92, 1.0 - light), saturation)
            return (r, g, b, a)
        if boxes and light > 0.9:  # a white box behind a label: the surface's colour
            return (*background[:3], a)
        return None

    def stronger(art: Any) -> None:
        """A light wash of colour (a band, a fan) all but vanishes on a dark surface: each layer covers more, 1 - (1 -
        alpha)^1.6, which keeps overlapping layers (a fan's) and their legend keys in step."""
        try:
            faces = art.get_facecolor()
            face = faces if isinstance(faces, str) or not hasattr(faces[0], "__len__") else faces[0]
            rgba = to_rgba(face)
            _, light, saturation = colorsys.rgb_to_hls(*rgba[:3])
        except (IndexError, ValueError, TypeError):
            return
        alpha = art.get_alpha()
        baked = alpha is None and isinstance(art, Patch)  # a legend key's shading can be in its face colour instead
        if baked:
            alpha = rgba[3]
        if alpha is None or not 0 < alpha < 0.6 or not (saturation > 0.12 and 0.04 < light < 0.96):
            return
        if baked:
            art.set_facecolor((*rgba[:3], 1 - (1 - alpha) ** 1.6))
        else:
            art.set_alpha(1 - (1 - alpha) ** 1.6)

    def fix(get: Any, put: Any, boxes: bool = False) -> None:
        try:
            value = get()
        except (AttributeError, TypeError, ValueError):
            return
        if isinstance(value, str) or (hasattr(value, "__len__") and len(value) in (3, 4) and not hasattr(value[0], "__len__")):
            new = turned(value, boxes)
            if new is not None:
                put(new)
        elif hasattr(value, "__len__") and len(value):  # a collection's colours, one per item
            news = [turned(v, boxes) for v in value]
            if any(n is not None for n in news):
                put([n if n is not None else to_rgba(v) for n, v in zip(news, value)])

    from matplotlib.axis import Axis
    from matplotlib.collections import Collection
    from matplotlib.spines import Spine
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    for ax in fig.get_axes():
        if getattr(ax, "name", "") == "3d":  # a 3-D box's panes: a shade of the surface, not light grey
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((*to_rgba(surface.grid)[:3], 0.35))
        legend = ax.get_legend()
        kinds = (Line2D, Text, Collection, Patch)
        # what was drawn in the axes, not its axes (ticks, grid, tick labels) or spines: those have the surface's colours
        found = [a for a in ax.get_children() if isinstance(a, kinds) and not isinstance(a, (Axis, Spine))]
        if legend is not None:
            found += legend.findobj(lambda a: isinstance(a, kinds))
        for art in found:
            if art is ax.patch or (legend is not None and art is legend.legendPatch):
                continue
            if isinstance(art, Line2D):
                fix(art.get_color, art.set_color)
                fix(art.get_markerfacecolor, art.set_markerfacecolor)
                fix(art.get_markeredgecolor, art.set_markeredgecolor)
            elif isinstance(art, Text):
                fix(art.get_color, art.set_color)
                box = art.get_bbox_patch()
                if box is not None:
                    fix(box.get_facecolor, box.set_facecolor, boxes=True)
                arrow = getattr(art, "arrow_patch", None)  # a note's arrow
                if arrow is not None:
                    fix(arrow.get_edgecolor, arrow.set_edgecolor)
                    fix(arrow.get_facecolor, arrow.set_facecolor)
            elif isinstance(art, Collection):
                if getattr(art, "get_array", lambda: None)() is not None:  # coloured by a colour map: as it is
                    continue
                fix(art.get_edgecolor, art.set_edgecolor)
                fix(art.get_facecolor, art.set_facecolor)
                stronger(art)
            elif isinstance(art, Patch):
                fix(art.get_edgecolor, art.set_edgecolor)
                fix(art.get_facecolor, art.set_facecolor)
                stronger(art)
    for text in fig.texts:
        fix(text.get_color, text.set_color)

