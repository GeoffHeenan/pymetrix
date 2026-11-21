"""Small icons that say what kind of object a row, a window or a palette entry is.

One icon per kind (:data:`KINDS`): a dataset (a cylinder), a series (a line on axes), an equation or a fitted
single-equation result (points and the line through them), a system or a VAR family result (three linked nodes), a
model, its solution or scenarios (a cube), a forecast (a line opening into a fan), a figure (bars), a report (a page),
a test result (a tick in a circle), a table or matrix (a grid), and anything else (a diamond). The shapes differ, not
only their colours, so they tell apart in grey too.

They are drawn here, in code, on a 16 by 16 grid (no picture files, no icon set), at each size a view may ask for
(14 to 64 pixels), so a screen at twice the density gets a 32-pixel drawing, not a blown-up 16-pixel one. Each is
two-tone: an outline in the palette's text colour and an accent in one of :mod:`theme`'s roles, so it reads on a
light or a dark theme.

Which kind an object is is decided in one place, :func:`kind_of`: by its class (and its base classes) in
:data:`CLASS_KINDS`, then, for pymetrix's own classes not listed there, by the class's name ("Fitted..." is an
estimate of one equation, "...Forecast..." a forecast). A saved result takes the kind of the result it was. Anything
else, plugins' objects included, gets the generic icon. Icons are decoration only: no text anywhere depends on them.

They scale with View → Text size: a view that shows them (the Objects panel, Find anything, the tabs of the tabbed
view) takes :func:`view_size` through :func:`size_icons`, which ``text_size.apply`` sets again (:func:`reapply`), and
the title bars are painted at that size too (``active_window``), as far as the bar has room.
"""

from __future__ import annotations

from typing import Any, Callable

from . import theme

__all__ = ["CLASS_KINDS", "KINDS", "LABELS", "SIZES", "icon", "kind_of", "kind_of_name", "object_icon", "pixmap", "reapply", "size_icons", "view_size"]

KINDS = ("dataset", "series", "equation", "system", "model", "forecast", "figure", "report", "test", "table", "generic")
LABELS = {"dataset": "Dataset", "series": "Series", "equation": "Equation", "system": "System", "model": "Model",
          "forecast": "Forecast", "figure": "Figure", "report": "Report", "test": "Test result", "table": "Table",
          "generic": "Object"}
SIZES = (14, 16, 18, 20, 24, 32, 40, 48, 64)  # pixels: 16 at 100%, 20 and 24 at 125% and 150%, 32 at 200%, ...;
# 14, 18 and 20 for the text sizes Smaller, Larger and Largest (view_size)
_SIZED = "pymetrixIconSize"  # dynamic property: a view whose icon size follows the text size (size_icons)

# The accent of each kind: a role of theme (readable on the window and list backgrounds of either theme).
_ACCENT = {"dataset": "link", "series": "link", "equation": "success", "system": "success", "model": "warning",
           "forecast": "warning", "figure": "link", "report": "muted", "test": "success", "table": "link", "generic": "muted"}

_GROUPS = {
    "dataset": ("Dataset", "Vintage"),
    "series": ("Series", "Variable", "Selection", "SeasonallyAdjusted", "Filtered"),
    "equation": ("Equation", "FittedEquation", "FittedBayesEquation", "RollingFit", "Results", "OrderSelection"),
    "system": ("System", "FittedSystem", "VAR", "FittedVAR", "FittedBVAR", "FittedBVARSV", "FittedVECM", "FittedVARMA",
               "FittedPanelVAR", "FittedQuantileVAR", "FittedTVPVAR", "FittedThresholdVAR", "FittedMSVAR", "FittedFAVAR",
               "FittedDSGEVAR", "FittedDynamicFactor", "FittedFactorModel", "FittedLocalProjection",
               "FittedSmoothLocalProjection", "IRF", "FEVD", "ImpulseResponse", "VarianceDecomposition",
               "HistoricalDecomposition", "LagSelection", "VARLagOrder", "VECMLagOrder"),
    "model": ("Model", "Template", "DynareModel", "ModelSystem", "FittedModel", "Solution", "LinearSolution", "LinearModel",
              "SecondOrderSolution", "PanelSolution", "StateSpace", "StateSpaceSystem", "FittedStateSpace", "Decomposition",
              "SweepResult", "Scenario", "ScenarioSet", "ParticleFilter", "OptimalPolicy", "OSRResult", "OptimalControl",
              "Posterior", "Identification", "OccBinPath", "ModelCheck", "Translation", "EViewsXmlModel", "SteadyState",
              "Calibration", "MonteCarloResult"),
    "forecast": ("Forecast", "ForecastRound", "Backtest", "ForecastStore", "Nowcast"),
    "figure": ("Figure",),
    "report": ("Report", "ReportTemplate"),
    "test": ("TestResult", "JohansenResult"),
    "table": ("Table", "DataFrame", "ndarray", "Matrix", "PivotTable"),
}
CLASS_KINDS: dict[str, str] = {name: kind for kind, names in _GROUPS.items() for name in names}
"""Class name -> kind, for the classes named outright; a class takes the kind of the first of its classes (its own,
then its bases) found here."""


def kind_of_name(name: str, core: bool = True) -> str:
    """The kind of a class called ``name``: :data:`CLASS_KINDS`, else (for pymetrix's own classes, ``core``) by the
    words of its name, else "generic"."""
    if name in CLASS_KINDS:
        return CLASS_KINDS[name]
    if core:
        if "Forecast" in name or "Nowcast" in name or name.startswith("Backtest"):
            return "forecast"
        if name.endswith(("Test", "TestResult")):
            return "test"
        if name.endswith(("VAR", "VECM")) or (name.startswith("Fitted") and name.endswith(("VAR", "System"))):
            return "system"
        if name.startswith("Fitted"):
            return "equation"
    return "generic"


def kind_of(obj: Any) -> str:
    """The kind of ``obj`` (one of :data:`KINDS`): see the module's docstring."""
    from ..saved import SavedResult

    if isinstance(obj, SavedResult):
        return kind_of_name(obj.kind or "", core=True)
    for cls in type(obj).__mro__:
        if cls.__name__ in CLASS_KINDS:
            return CLASS_KINDS[cls.__name__]
    cls = type(obj)
    return kind_of_name(cls.__name__, core=(cls.__module__ or "").split(".")[0] == "pymetrix")


# -- drawing ---------------------------------------------------------------------------------------------------------
#
# Each painter draws on a 16 by 16 grid, the painter already scaled to the pixmap. ``ink`` is the outline (the
# palette's text colour), ``accent`` the kind's colour. Lines of width 1 lie on half units, so they fall on whole
# pixels at 16 and at 32 pixels.

def _pen(QtGui: Any, colour: Any, width: float = 1.0, dash: bool = False) -> Any:
    from PySide6.QtCore import Qt

    pen = QtGui.QPen(colour, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap if not dash else Qt.PenCapStyle.FlatCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if dash:
        pen.setDashPattern([1.5, 1.5])
    return pen


def _dataset(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QRectF

    body = G.QPainterPath()
    body.moveTo(2.5, 4.0)
    body.lineTo(2.5, 12.0)
    body.arcTo(QRectF(2.5, 10.0, 11.0, 4.0), 180, 180)
    body.lineTo(13.5, 4.0)
    p.setPen(_pen(G, ink))
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawPath(body)
    p.drawArc(QRectF(2.5, 6.0, 11.0, 4.0), 180 * 16, 180 * 16)
    p.setBrush(accent)
    p.drawEllipse(QRectF(2.5, 1.5, 11.0, 5.0))


def _series(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF

    p.setPen(_pen(G, ink))
    p.drawPolyline([QPointF(1.5, 1.5), QPointF(1.5, 14.5), QPointF(14.5, 14.5)])
    p.setPen(_pen(G, accent, 1.75))
    p.drawPolyline([QPointF(3.5, 11.5), QPointF(6.0, 6.5), QPointF(8.5, 9.5), QPointF(11.0, 3.5), QPointF(13.5, 6.0)])


def _equation(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    p.setPen(_pen(G, accent, 1.75))
    p.drawLine(QPointF(1.5, 13.5), QPointF(14.5, 2.5))
    p.setPen(G.QColor(0, 0, 0, 0))
    p.setBrush(ink)
    for x, y in ((3.5, 8.5), (6.0, 12.5), (8.0, 5.5), (10.5, 9.5), (12.5, 3.0)):
        p.drawEllipse(QRectF(x - 1.25, y - 1.25, 2.5, 2.5))


def _system(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    nodes = (QPointF(4.0, 4.0), QPointF(12.0, 4.0), QPointF(8.0, 12.0))
    p.setPen(_pen(G, ink))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p.drawLine(nodes[a], nodes[b])
    p.setBrush(accent)
    for n in nodes:
        p.drawEllipse(QRectF(n.x() - 2.5, n.y() - 2.5, 5.0, 5.0))


def _model(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF

    top = G.QPolygonF([QPointF(8.0, 1.5), QPointF(14.0, 4.5), QPointF(8.0, 7.5), QPointF(2.0, 4.5)])
    p.setPen(_pen(G, ink))
    p.setBrush(accent)
    p.drawPolygon(top)
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawPolyline([QPointF(2.0, 4.5), QPointF(2.0, 11.5), QPointF(8.0, 14.5), QPointF(14.0, 11.5), QPointF(14.0, 4.5)])
    p.drawLine(QPointF(8.0, 7.5), QPointF(8.0, 14.5))


def _forecast(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF

    fill = G.QColor(accent)
    fill.setAlphaF(0.4)
    p.setPen(_pen(G, accent))
    p.setBrush(fill)
    p.drawPolygon(G.QPolygonF([QPointF(8.0, 8.0), QPointF(14.5, 2.5), QPointF(14.5, 13.5)]))  # the fan
    p.setPen(_pen(G, accent, 1.5))
    p.drawLine(QPointF(8.0, 8.0), QPointF(14.5, 7.0))
    p.setPen(_pen(G, ink, 1.5))  # the history, up to the forecast's start
    p.drawPolyline([QPointF(1.5, 11.5), QPointF(3.5, 9.0), QPointF(5.5, 10.5), QPointF(8.0, 8.0)])


def _figure(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    p.setPen(G.QColor(0, 0, 0, 0))
    p.setBrush(accent)
    for x, top in ((2.0, 8.0), (6.0, 3.0), (10.0, 6.0)):  # whole pixels, a pixel apart
        p.drawRect(QRectF(x, top, 3.0, 13.0 - top))
    p.setPen(_pen(G, ink))
    p.drawLine(QPointF(1.0, 13.5), QPointF(15.0, 13.5))


def _report(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF

    page = G.QPolygonF([QPointF(2.5, 1.5), QPointF(10.0, 1.5), QPointF(13.5, 5.0), QPointF(13.5, 14.5), QPointF(2.5, 14.5)])
    p.setPen(_pen(G, ink))
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawPolygon(page)
    p.drawPolyline([QPointF(10.0, 1.5), QPointF(10.0, 5.0), QPointF(13.5, 5.0)])
    p.setPen(_pen(G, accent, 1.25))
    for y, end in ((7.5, 11.0), (10.0, 11.0), (12.5, 8.5)):
        p.drawLine(QPointF(5.0, y), QPointF(end, y))


def _test(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    p.setPen(_pen(G, ink))
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawEllipse(QRectF(1.5, 1.5, 13.0, 13.0))
    p.setPen(_pen(G, accent, 2.0))
    p.drawPolyline([QPointF(4.5, 8.0), QPointF(7.0, 10.5), QPointF(11.5, 5.0)])


def _table(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    p.setPen(G.QColor(0, 0, 0, 0))
    p.setBrush(accent)
    p.drawRect(QRectF(1.5, 2.5, 13.0, 3.0))
    p.setPen(_pen(G, ink))
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawRect(QRectF(1.5, 2.5, 13.0, 11.0))
    for y in (5.5, 9.5):
        p.drawLine(QPointF(1.5, y), QPointF(14.5, y))
    for x in (5.5, 10.0):
        p.drawLine(QPointF(x, 2.5), QPointF(x, 13.5))


def _generic(p: Any, G: Any, ink: Any, accent: Any) -> None:
    from PySide6.QtCore import QPointF, QRectF

    p.setPen(_pen(G, ink))
    p.setBrush(G.QColor(0, 0, 0, 0))
    p.drawPolygon(G.QPolygonF([QPointF(8.0, 1.5), QPointF(14.5, 8.0), QPointF(8.0, 14.5), QPointF(1.5, 8.0)]))
    p.setPen(G.QColor(0, 0, 0, 0))
    p.setBrush(accent)
    p.drawEllipse(QRectF(5.5, 5.5, 5.0, 5.0))


_PAINTERS: dict[str, Callable[..., None]] = {
    "dataset": _dataset, "series": _series, "equation": _equation, "system": _system, "model": _model,
    "forecast": _forecast, "figure": _figure, "report": _report, "test": _test, "table": _table, "generic": _generic,
}


def _colours(kind: str, palette: Any, selected: bool = False) -> tuple[Any, Any]:
    from PySide6 import QtGui

    pal = theme._palette(palette)
    G, R = QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole
    if selected:  # on the selection (Highlight): all in its text colour, the shape alone telling the kind
        return pal.color(G, R.HighlightedText), pal.color(G, R.HighlightedText)
    return pal.color(G, R.Text), theme.colour(_ACCENT.get(kind, "muted"), pal)


def pixmap(kind: str, size: int = 16, ratio: float = 1.0, palette: Any = None, selected: bool = False) -> Any:
    """``kind``'s icon drawn ``size`` pixels square on a screen of device pixel ratio ``ratio`` (a pixmap of
    ``size * ratio`` pixels), in the colours of ``palette`` (the app's): two-tone, or, ``selected``, in the
    selection's text colour (HighlightedText) alone. An unknown kind draws the generic icon."""
    from PySide6 import QtCore, QtGui

    pixels = max(1, round(size * ratio))
    image = QtGui.QPixmap(pixels, pixels)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    ink, accent = _colours(kind, palette, selected)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.scale(pixels / 16.0, pixels / 16.0)
        _PAINTERS.get(kind, _generic)(painter, QtGui, ink, accent)
    finally:
        painter.end()
    image.setDevicePixelRatio(ratio)
    return image


_cache: dict[tuple[str, ...], Any] = {}


def icon(kind: str, palette: Any = None) -> Any:
    """``kind``'s QIcon, with a drawing for each of :data:`SIZES` (Qt picks the one a view's size and the screen's
    density need), in ``palette``'s colours (the app's), and one in the selection's text colour for a selected row
    or the front window's title bar (QIcon.Mode.Selected). Made once per kind and palette."""
    from PySide6 import QtGui

    kind = kind if kind in _PAINTERS else "generic"
    pal = theme._palette(palette)
    G, R = QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole
    key = (kind, *(pal.color(G, role).name() for role in (R.Window, R.Base, R.Text, R.HighlightedText)))
    made = _cache.get(key)
    if made is None:
        made = QtGui.QIcon()
        for size in SIZES:
            made.addPixmap(pixmap(kind, size, 1.0, pal))
            made.addPixmap(pixmap(kind, size, 1.0, pal, selected=True), QtGui.QIcon.Mode.Selected)
        _cache[key] = made
    return made


def object_icon(obj: Any, palette: Any = None) -> Any:
    """The icon of ``obj``'s kind (:func:`kind_of`)."""
    return icon(kind_of(obj), palette)


def view_size(widget: Any = None) -> int:
    """The size, in pixels, an icon is shown at beside text: the style's small icon size (16 on most) at the text size
    applied now (``text_size.factor``), to whole even pixels (14, 16, 18 and 20 for Smaller to Largest). Exactly
    the style's size at Normal."""
    from PySide6 import QtWidgets

    from .text_size import factor

    style = widget.style() if widget is not None else QtWidgets.QApplication.style()
    small = style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_SmallIconSize, None, widget)
    return max(2, 2 * round(small * factor() / 2))


def size_icons(view: Any) -> Any:
    """``view`` (a list, a tree, a tab bar: anything with ``setIconSize``) showing its icons at :func:`view_size`, now
    and after a text size change (:func:`reapply`). Returns the view."""
    from PySide6 import QtCore

    view.setProperty(_SIZED, True)
    size = view_size(view)
    view.setIconSize(QtCore.QSize(size, size))
    return view


def reapply(app: Any) -> None:
    """Every view given :func:`size_icons` at the text size applied now (``text_size.apply`` calls this)."""
    from PySide6 import QtCore

    for widget in app.allWidgets():
        try:
            if widget.property(_SIZED):
                size = view_size(widget)
                widget.setIconSize(QtCore.QSize(size, size))
        except RuntimeError:  # deleted while being listed
            continue
