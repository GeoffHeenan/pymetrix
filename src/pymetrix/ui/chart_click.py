"""Click on a chart to identify what is there: a small box with the series' name, or, on one of its data points,
the series' name, the period (as the axis shows it, 2020Q1) and the value, with the point marked.

On a figure window's canvas (``install(canvas)``): a click on a line, step, area, band or set of scatter points
(within ``REACH`` pixels, through matplotlib's pick events) names the series; a click within ``POINT_REACH``
pixels of one of its data points gives that point; a bar gives its category or period and value; a heatmap cell its
row, column and value. 3-D panels are left out (a click there turns the panel). A click on empty space, or Esc,
takes the box away. One box at a time per figure, kept inside the panel. The box and the marker live on the
window's canvas only: they are never recorded as a style step, and saved images, copies and reports are drawn
afresh from the figure, without them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .theme import CHART
from .text_size import factor as text_factor

REACH = 5  # pixels: a click this close to a series picks it
POINT_REACH = 6  # pixels: and this close to one of its data points gives the point
GID = "pymetrix-identify"  # the box's and the marker's gid: never a series, never picked


def _name(artist: Any) -> str:
    label = artist.get_label() if hasattr(artist, "get_label") else ""
    return "" if not label or str(label).startswith("_") else str(label)


def _number(value: float) -> str:
    """A value as people read it: 1,234.5, 0.0123, 2.5e+06."""
    if not np.isfinite(value):
        return "no value"
    size = abs(value)
    if size != 0 and (size >= 1e6 or size < 1e-3):
        return f"{value:.4g}"
    if size >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4g}"


def _x_words(ax: Any, x: float) -> str:
    from ..figures import _x_text

    return _x_text(x, getattr(ax, "pymetrix_freq", None))


def _pixel_points(ax: Any, xs: Any, ys: Any) -> tuple[Any, Any]:
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    return ok, ax.transData.transform(np.column_stack([xs[ok], ys[ok]])) if ok.any() else np.empty((0, 2))


def _nearest(ax: Any, xs: Any, ys: Any, px: float, py: float) -> tuple[float, float, float] | None:
    """The data point of (xs, ys) nearest the pixel (px, py): (distance in pixels, x, y)."""
    ok, pixels = _pixel_points(ax, xs, ys)
    if not len(pixels):
        return None
    distance = np.hypot(pixels[:, 0] - px, pixels[:, 1] - py)
    k = int(np.argmin(distance))
    x, y = np.asarray(xs, dtype=float)[ok][k], np.asarray(ys, dtype=float)[ok][k]
    return float(distance[k]), float(x), float(y)


def identify(artist: Any, px: float, py: float) -> dict[str, Any] | None:
    """What a click at pixel (px, py) on ``artist`` identifies: {"text", "ax", "point" (x, y in data, or None),
    "distance"}; None when the artist is not a series (no name) or not one this handles."""
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    ax = getattr(artist, "axes", None)
    if ax is None or getattr(ax, "name", "") == "3d" or artist.get_gid() == GID:
        return None
    if isinstance(artist, AxesImage):  # a heatmap: the cell, its row and column, and its value
        x, y = ax.transData.inverted().transform((px, py))
        cells = getattr(artist, "pymetrix_values", None)  # the values, where the colours are shades (by column)
        cells = artist.get_array() if cells is None else cells
        row, col = int(round(y)), int(round(x))
        if not (0 <= row < cells.shape[0] and 0 <= col < cells.shape[1]):
            return None
        rows = [t.get_text() for t in ax.get_yticklabels()]
        cols = [t.get_text() for t in ax.get_xticklabels()]
        value = cells[row, col]
        shown = "no value" if np.ma.is_masked(value) or not np.isfinite(value) else _number(float(value))
        text = f"{rows[row] if row < len(rows) and rows[row] else row}, {cols[col] if col < len(cols) and cols[col] else col}\n{shown}"
        return {"text": text, "ax": ax, "point": (float(col), float(row)), "distance": 0.0, "mark": False}
    if isinstance(artist, Rectangle):  # a bar: its series, its category or period, and its value
        for container in ax.containers:
            patches = list(getattr(container, "patches", []) or [])
            if artist not in patches:
                continue
            values = getattr(container, "datavalues", None)
            k = patches.index(artist)
            value = float(values[k]) if values is not None and k < len(values) else float(artist.get_height())
            horizontal = getattr(container, "orientation", "vertical") == "horizontal"
            if horizontal:
                from ..figures import _tick_text

                where = _tick_text(ax, artist.get_y() + artist.get_height() / 2, "y")
                tip = (artist.get_x() + artist.get_width(), artist.get_y() + artist.get_height() / 2)
            else:
                where = _x_words(ax, artist.get_x() + artist.get_width() / 2)
                tip = (artist.get_x() + artist.get_width() / 2, artist.get_y() + artist.get_height())
            name = _name(container)
            lines = [name] if name else []
            lines.append(f"{where}: {_number(value)}" if where else _number(value))
            return {"text": "\n".join(lines), "ax": ax, "point": tip, "distance": 0.0, "mark": False}
        return None
    name = _name(artist)
    scatter = isinstance(artist, PathCollection) and getattr(ax, "pymetrix_freq", None) is None
    if not name and scatter and len(ax.collections) == 1:  # one unnamed scatter: named by its axes, y against x
        name = " against ".join(t for t in (ax.get_ylabel(), ax.get_xlabel()) if t) or "points"
    if not name:
        return None
    if isinstance(artist, Line2D):
        near = _nearest(ax, artist.get_xdata(orig=False), artist.get_ydata(orig=False), px, py)
    elif isinstance(artist, PathCollection):  # scatter points
        offsets = np.asarray(artist.get_offsets(), dtype=float)
        near = _nearest(ax, offsets[:, 0], offsets[:, 1], px, py) if len(offsets) else None
    elif isinstance(artist, PolyCollection):  # an area or a band: its name only
        near = None
    else:
        return None
    if near is not None and near[0] <= POINT_REACH:
        _, x, y = near
        if scatter:  # both values, each by its axis' name
            where = f"{ax.get_xlabel() or 'x'} {_number(x)}, {ax.get_ylabel() or 'y'} {_number(y)}"
        else:
            where = f"{_x_words(ax, x) or _number(x)}: {_number(y)}"
        return {"text": f"{name}\n{where}", "ax": ax, "point": (x, y), "distance": near[0], "mark": True}
    x, y = ax.transData.inverted().transform((px, py))
    return {"text": name, "ax": ax, "point": (float(x), float(y)), "distance": near[0] if near else float(REACH),
            "mark": False}


def make_pickable(figure: Any) -> None:
    """Every series in ``figure``'s panels answers a click within ``REACH`` pixels."""
    from matplotlib.collections import PathCollection, PolyCollection

    for ax in figure.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        for line in ax.get_lines():
            if _name(line):
                line.set_picker(True)
                line.set_pickradius(REACH)
        for collection in ax.collections:
            if isinstance(collection, (PathCollection, PolyCollection)):
                collection.set_picker(True)
                collection.set_pickradius(REACH)
        for container in ax.containers:
            for patch in getattr(container, "patches", []) or []:
                patch.set_picker(True)
        for image in ax.get_images():
            image.set_picker(True)


class Identifier:
    """The click-to-identify box on one canvas. ``enabled()`` says whether clicks are for it (not while Add a note
    waits for a click, or the toolbar pans or zooms)."""

    def __init__(self, canvas: Any, enabled: Any = None, skip: Any = None) -> None:
        self.canvas = canvas
        self.enabled = enabled or (lambda: True)
        self.skip = skip or (lambda event: False)  # a click something else used (selecting a note to edit)
        self.box: Any = None
        self.marker: Any = None
        self.shown: dict[str, Any] | None = None
        self._picks: list[tuple[Any, Any]] = []  # (mouse event, artist) picked by the click being handled
        make_pickable(canvas.figure)
        self._ids = [canvas.mpl_connect("pick_event", self._picked),
                     canvas.mpl_connect("button_press_event", self._pressed),
                     canvas.mpl_connect("key_press_event", self._key)]

    def _picked(self, event: Any) -> None:
        self._picks.append((event.mouseevent, event.artist))

    def _pressed(self, event: Any) -> None:
        """After the figure's own pick for this click (it is connected first): the best pick, or dismiss."""
        picks = [artist for mouse, artist in self._picks if mouse is event]
        self._picks.clear()
        if event.button != 1 or event.dblclick or not self.enabled():
            return
        if self.skip(event):
            self.dismiss()
            return
        if event.inaxes is not None and getattr(event.inaxes, "name", "") == "3d":
            self.dismiss()
            return
        found = [f for f in (identify(a, event.x, event.y) for a in picks) if f is not None]
        if not found:
            self.dismiss()
            return
        points = [f for f in found if f["mark"]]  # a data point beats a series' name, the nearest point first
        best = min(points, key=lambda f: f["distance"]) if points else min(found, key=lambda f: f["distance"])
        self.show(best)

    def _key(self, event: Any) -> None:
        if event.key == "escape":
            self.dismiss()

    def show(self, found: dict[str, Any]) -> None:
        """The box for ``found`` (from ``identify``), near the point and inside its panel; the point marked."""
        self.dismiss(redraw=False)
        ax, (x, y) = found["ax"], found["point"]
        px, py = ax.transData.transform((x, y))
        box = ax.get_window_extent()
        right = px > box.x0 + box.width * 0.6  # near the right or top edge: the box goes the other way
        top = py > box.y0 + box.height * 0.6
        offset = (-12 if right else 12, -12 if top else 12)
        self.box = ax.annotate(found["text"], xy=(x, y), xycoords="data", xytext=offset, textcoords="offset points",
                               ha="right" if right else "left", va="top" if top else "bottom", fontsize=8.5 * text_factor(),
                               bbox={"boxstyle": "round,pad=0.35", "fc": CHART["tip face"], "ec": CHART["tip edge"], "alpha": 0.95},
                               zorder=1000, annotation_clip=False)
        self.box.set_gid(GID)
        self.box.set_in_layout(False)  # never moves the panels
        if found.get("mark"):
            (self.marker,) = ax.plot([x], [y], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=CHART["picked point"],
                                     markeredgewidth=1.8, linestyle="none", zorder=999, label="_identify", scalex=False, scaley=False)
            self.marker.set_gid(GID)
            self.marker.set_in_layout(False)
        self.shown = found
        self.canvas.draw_idle()

    def dismiss(self, redraw: bool = True) -> None:
        had = self.box is not None or self.marker is not None
        for artist in (self.box, self.marker):
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    pass
        self.box = self.marker = self.shown = None
        if had and redraw:
            self.canvas.draw_idle()

    def text(self) -> str:
        """The box's text now ("" when none is shown)."""
        return self.box.get_text() if self.box is not None else ""


def install(canvas: Any, enabled: Any = None, skip: Any = None) -> Identifier:
    return Identifier(canvas, enabled, skip)
