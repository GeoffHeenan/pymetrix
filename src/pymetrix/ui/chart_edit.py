"""Edit what was added to a chart by dragging it: notes, arrows, text boxes, lines at a date or value, bands and
change markers, each drawn from a recorded style step (``Figure.annotate``, ``text``, ``axvline``, ``axhline``,
``axvspan``, ``axhspan``, ``change``).

On a figure window's canvas (``install(canvas, figure, on_restyle, on_remove, on_text)``):

- a click on one selects it and shows its handles (small squares: an arrow's head and tail, a band's edges, a
  line, a text's anchor, a change marker's two ends); a click on empty space or Esc clears the selection;
- a drag moves it (Shift: only across or only up and down); a drag on a handle moves that end or edge. Positions
  snap to periods on a dated axis, and a change marker's ends to the series' data points (its label is recomputed
  when the figure is drawn again). Esc during a drag puts it back;
- Delete (or Backspace) removes it; a double-click on words edits them, and on anything else opens its format
  dialog (``on_format``). An arrow is picked by its line, within a few pixels, words or not.

Each edit is handed back as the recorded step's new keyword arguments, in the data's terms (periods, values):
``on_restyle(step, props)``, ``on_remove(step)``; the app runs them as Change a style and Recorded styles steps, so
they are in the history, undone and redone like any step, and replay at any size, in exports and reports. The
handles are drawn on the window's canvas only: never recorded, never exported.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .theme import CHART

GID = "pymetrix-edit"  # handles: never a series, never picked, never saved
HANDLE_REACH = 7  # pixels: a press this close to a handle drags it
LINE_REACH = 5  # pixels: a press this close to an arrow's line picks it
EDITABLE = ("annotate", "text", "axvline", "axhline", "axvspan", "axhspan", "change")


def _num(value: Any, freq: Any, end: bool = False) -> float:
    from ..figures import _x_number

    return float(_x_number(value, freq, end=end))


def _period(ax: Any, x: float, edge: str = "start") -> Any:
    """``x`` (matplotlib's number) in the data's terms: the nearest period on a dated axis (``edge="end"``: the period
    whose end is nearest, for a band's last date), else the number, to six significant figures."""
    freq = getattr(ax, "pymetrix_freq", None)
    if freq is None:
        return float(f"{x:.6g}")
    from matplotlib import dates

    when = pd.Timestamp(dates.num2date(x).replace(tzinfo=None))
    p = pd.Period(when, freq=freq)
    start = float(dates.date2num(p.to_timestamp()))
    following = float(dates.date2num((p + 1).to_timestamp()))
    if edge == "end":  # the band reaches the end of this period
        return str(p if following - x <= x - start else p - 1)
    return str(p if x - start <= following - x else p + 1)


def _value(y: float) -> float:
    return float(f"{y:.6g}")


def near_arrow(artist: Any, x: float, y: float, reach: float = LINE_REACH) -> bool:
    """Whether the display point (x, y) is within ``reach`` pixels of an annotation's arrow (its line and head, as
    last drawn): matplotlib's own test misses a thin, unfilled arrow."""
    patch = getattr(artist, "arrow_patch", None)
    if patch is None or not patch.get_visible():
        return False
    try:
        polygons = patch.get_path().to_polygons(closed_only=False)
    except Exception:  # noqa: BLE001 - not drawn yet: nothing to pick
        return False
    p = np.array([x, y], dtype=float)
    for poly in polygons:
        pts = np.asarray(poly, dtype=float)
        if len(pts) == 1 and np.hypot(*(pts[0] - p)) <= reach:
            return True
        for a, b in itertools.pairwise(pts):
            d = b - a
            t = 0.0 if not d.any() else float(np.clip(np.dot(p - a, d) / np.dot(d, d), 0, 1))
            if np.hypot(*(a + t * d - p)) <= reach:
                return True
    return False


def _on_text(artists: list[Any], event: Any) -> bool:
    from matplotlib.text import Text

    for a in artists:
        try:
            on = Text.contains(a, event)[0]
        except Exception:  # noqa: BLE001 - not hit-testable: not on its words
            on = False
        if on:
            return True
    return False


class Handles:
    """The draggable points of one recorded step, in data coordinates, and the step's new arguments for a move."""

    def __init__(self, call: Any, artists: list[Any]) -> None:
        self.call, self.artists = call, artists
        self.ax = artists[0].axes
        self.freq = getattr(self.ax, "pymetrix_freq", None)
        p = call.props
        kind = call.kind
        self.points: dict[str, list[float]] = {}
        if kind == "annotate":
            a = artists[0]
            self.points["head"] = [float(a.xy[0]), float(a.xy[1])]
            self.data_text = p.get("textcoords", p.get("xycoords", "data")) == "data" and "xytext" in p
            if self.data_text:
                self.points["tail"] = [float(a.xyann[0]), float(a.xyann[1])]
        elif kind == "text":
            x, y = artists[0].get_position()
            self.points["at"] = [float(x), float(y)]
        elif kind in ("axvline", "axhline"):
            (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
            if kind == "axvline":
                self.points["line"] = [_num(p["x"], self.freq), (y0 + y1) / 2]
            else:
                self.points["line"] = [(x0 + x1) / 2, float(p["y"])]
        elif kind == "axvspan":
            y0, y1 = self.ax.get_ylim()
            self.points["from"] = [_num(p["xmin"], self.freq), (y0 + y1) / 2]
            self.points["to"] = [_num(p["xmax"], self.freq, end=True), (y0 + y1) / 2]
        elif kind == "axhspan":
            x0, x1 = self.ax.get_xlim()
            self.points["from"] = [(x0 + x1) / 2, float(p["ymin"])]
            self.points["to"] = [(x0 + x1) / 2, float(p["ymax"])]
        elif kind == "change":
            arrow = next(a for a in artists if getattr(a, "arrow_patch", None) is not None)
            self.points["from"] = [float(arrow.xyann[0]), float(arrow.xyann[1])]
            self.points["to"] = [float(arrow.xy[0]), float(arrow.xy[1])]
        self.start = {k: list(v) for k, v in self.points.items()}

    def moved(self, which: str | None, dx: float, dy: float) -> None:
        """Move one handle (``which``) or, with None, the whole thing, by (dx, dy) in data units from where it was."""
        for name, (x, y) in self.start.items():
            if which is None or name == which:
                self.points[name] = [x + dx, y + dy]

    def props(self, which: str | None) -> dict[str, Any]:
        """The step's changed arguments, in the data's terms, for the handles as they are now."""
        kind, p, ax = self.call.kind, self.call.props, self.ax
        pt = self.points
        if kind == "annotate":
            out: dict[str, Any] = {"xy": (_period(ax, pt["head"][0]), _value(pt["head"][1]))}
            if self.data_text:
                out["xytext"] = (_period(ax, pt["tail"][0]), _value(pt["tail"][1]))
            return out
        if kind == "text":
            return {"x": _period(ax, pt["at"][0]), "y": _value(pt["at"][1])}
        if kind == "axvline":
            return {"x": _period(ax, pt["line"][0])}
        if kind == "axhline":
            return {"y": _value(pt["line"][1])}
        if kind == "axvspan":
            a, b = _period(ax, pt["from"][0]), _period(ax, pt["to"][0], edge="end")
            return {"xmin": a, "xmax": b}
        if kind == "axhspan":
            return {"ymin": _value(pt["from"][1]), "ymax": _value(pt["to"][1])}
        if kind == "change":  # each end to the series' nearest data point
            line = self._series_line(p["series"])
            ends = {}
            for name, key in (("from", "start"), ("to", "end")):
                ends[key] = self._snap(line, pt[name]) if line is not None else _period(ax, pt[name][0])
            return ends
        return {}

    def _series_line(self, label: str) -> Any:
        return next((l for l in self.ax.get_lines() if l.get_label() == label), None)

    def _snap(self, line: Any, point: list[float]) -> Any:
        xs = np.asarray(line.get_xdata(orig=False), dtype=float)
        ys = np.asarray(line.get_ydata(orig=False), dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        k = int(np.argmin(np.abs(xs[ok] - point[0])))
        return _period(self.ax, float(xs[ok][k]))

    def preview(self) -> None:
        """The artists moved as far as matplotlib lets them be without redrawing the figure."""
        kind, a = self.call.kind, self.artists[0]
        pt = self.points
        if kind == "annotate":
            a.xy = tuple(pt["head"])
            if self.data_text:
                a.xyann = tuple(pt["tail"])
        elif kind == "text":
            a.set_position(tuple(pt["at"]))
        elif kind == "axvline":
            a.set_xdata([pt["line"][0]] * 2)
        elif kind == "axhline":
            a.set_ydata([pt["line"][1]] * 2)

    def restore(self) -> None:
        self.points = {k: list(v) for k, v in self.start.items()}
        self.preview()


class Editor:
    """Direct editing on one canvas. ``figure`` is the pymetrix Figure drawn there (its ``style`` numbers the steps);
    ``enabled()`` says whether clicks are for editing now."""

    def __init__(self, canvas: Any, figure: Any, on_restyle: Any, on_remove: Any, on_text: Any = None,
                 enabled: Any = None, on_format: Any = None) -> None:
        self.canvas, self.figure = canvas, figure
        self.on_restyle, self.on_remove, self.on_text, self.on_format = on_restyle, on_remove, on_text, on_format
        self.enabled = enabled or (lambda: True)
        self.selected: Handles | None = None
        self.markers: list[Any] = []
        self.drag: dict[str, Any] | None = None
        self.took: Any = None  # the last mouse event this editor used (the click-to-identify box leaves it alone)
        self._ids = [canvas.mpl_connect("button_press_event", self._pressed),
                     canvas.mpl_connect("motion_notify_event", self._moved),
                     canvas.mpl_connect("button_release_event", self._released),
                     canvas.mpl_connect("key_press_event", self._key)]

    # what can be edited

    def step_of(self, call: Any) -> int:
        return next(k for k, c in enumerate(self.figure.style) if c is call)

    def _groups(self) -> dict[int, tuple[Any, list[Any]]]:
        """Each recorded step drawn on the canvas, by its step number: (call, its artists)."""
        groups: dict[int, tuple[Any, list[Any]]] = {}
        for ax in self.canvas.figure.axes:
            for artist in [*ax.texts, *ax.lines, *ax.patches, *ax.collections]:
                call = getattr(artist, "pymetrix_style", None)
                if call is None or call.kind not in EDITABLE:
                    continue
                try:
                    k = self.step_of(call)
                except StopIteration:  # a house style's call: not this figure's to change
                    continue
                groups.setdefault(k, (call, []))[1].append(artist)
        return groups

    def artist_at(self, event: Any) -> tuple[int, Any, list[Any]] | None:
        """The recorded step under the mouse, topmost (latest) first."""
        for k, (call, artists) in sorted(self._groups().items(), reverse=True):
            for artist in artists:
                try:
                    hit = artist.contains(event)[0] or near_arrow(artist, event.x, event.y)
                except Exception:  # noqa: BLE001 - an artist matplotlib cannot hit-test is simply not picked
                    hit = False
                if hit:
                    return k, call, artists
        return None

    # selection and handles

    def select(self, step: int | None) -> None:
        self._clear_markers()
        self.selected = None
        if step is not None:
            call, artists = self._groups()[step]
            self.selected = Handles(call, artists)
            self.selected.step = step  # type: ignore[attr-defined]
            self._draw_markers()
        self.canvas.draw_idle()

    def _draw_markers(self) -> None:
        h = self.selected
        for name, (x, y) in h.points.items():
            (marker,) = h.ax.plot([x], [y], marker="s", markersize=7, markerfacecolor="white", markeredgecolor=CHART["edited point"],
                                  markeredgewidth=1.5, linestyle="none", zorder=1001, label="_handle", scalex=False, scaley=False)
            marker.set_gid(GID)
            marker.set_in_layout(False)
            marker.handle = name
            self.markers.append(marker)

    def _update_markers(self) -> None:
        for marker in self.markers:
            x, y = self.selected.points[marker.handle]
            marker.set_data([x], [y])

    def _clear_markers(self) -> None:
        for marker in self.markers:
            try:
                marker.remove()
            except ValueError:
                pass
        self.markers = []

    def handle_at(self, event: Any) -> str | None:
        if self.selected is None:
            return None
        for name, (x, y) in self.selected.points.items():
            px, py = self.selected.ax.transData.transform((x, y))
            if np.hypot(px - event.x, py - event.y) <= HANDLE_REACH:
                return name
        return None

    # the mouse and keys

    def _pressed(self, event: Any) -> None:
        if event.button != 1 or not self.enabled():
            return
        handle = self.handle_at(event)
        if event.dblclick:
            found = self.artist_at(event)
            if found is None:
                return
            step, call, artists = found
            on_words = call.kind == "text" or (call.kind == "annotate" and str(call.props.get("text", "")).strip() != ""
                                               and _on_text(artists, event))
            if on_words and self.on_text is not None:
                self.took = event
                key = "text" if call.kind == "annotate" else "s"
                self.on_text(step, key, call.props.get(key, ""))
            elif self.on_format is not None:
                self.took = event
                self.drag = None
                self.on_format(step)
            return
        if handle is None:
            found = self.artist_at(event)
            if found is None:
                if self.selected is not None:
                    self.select(None)
                return
            if self.selected is None or self.selected.step != found[0]:  # type: ignore[attr-defined]
                self.select(found[0])
        self.took = event
        if self.selected is None:
            return
        self.drag = {"handle": handle, "x": event.x, "y": event.y, "moved": False}

    def _moved(self, event: Any) -> None:
        if self.drag is None or self.selected is None:
            return
        h = self.selected
        dxp, dyp = event.x - self.drag["x"], event.y - self.drag["y"]
        if "shift" in str(getattr(event, "key", "") or ""):  # only across, or only up and down
            if abs(dxp) >= abs(dyp):
                dyp = 0
            else:
                dxp = 0
        if abs(dxp) + abs(dyp) < 2 and not self.drag["moved"]:
            return
        self.drag["moved"] = True
        inverse = h.ax.transData.inverted()
        (x0, y0), (x1, y1) = inverse.transform((self.drag["x"], self.drag["y"])), inverse.transform((self.drag["x"] + dxp, self.drag["y"] + dyp))
        h.moved(self.drag["handle"], x1 - x0, y1 - y0)
        if h.call.kind == "change" and self.drag["handle"] is None:  # a change marker moves by its ends only
            h.restore()
            return
        h.preview()
        self._update_markers()
        self.canvas.draw_idle()

    def _released(self, event: Any) -> None:
        drag, self.drag = self.drag, None
        if drag is None or not drag["moved"] or self.selected is None:
            return
        h = self.selected
        props = h.props(drag["handle"])
        old = {k: h.call.props.get(k) for k in props}
        if props and props != old:
            self.on_restyle(h.step, props)  # type: ignore[attr-defined]
        else:
            h.restore()
            self._update_markers()
            self.canvas.draw_idle()

    def _key(self, event: Any) -> None:
        if event.key == "escape":
            if self.drag is not None and self.selected is not None:  # the drag is put back
                self.drag = None
                self.selected.restore()
                self._update_markers()
                self.canvas.draw_idle()
            elif self.selected is not None:
                self.select(None)
        elif event.key in ("delete", "backspace") and self.selected is not None and self.drag is None:
            step = self.selected.step  # type: ignore[attr-defined]
            self.select(None)
            self.on_remove(step)


def install(canvas: Any, figure: Any, on_restyle: Any, on_remove: Any, on_text: Any = None, enabled: Any = None,
            on_format: Any = None) -> Editor:
    return Editor(canvas, figure, on_restyle, on_remove, on_text, enabled, on_format)
