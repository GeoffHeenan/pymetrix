"""Screenshots of the desktop app in some of its larger workflows, rendered off-screen from the real app.

    python scripts/make_screenshots.py out/                  # all six, as PNG files in out/
    python scripts/make_screenshots.py out/ irf event        # only the scenes named

Each scene opens a fresh window on one of the built-in demo projects (``pymetrix.demo``) or on example data, runs
what the scene needs, arranges the result windows and saves the window as it would appear on screen, with the
Objects, History and Script panels in view. It needs the app's extras (PySide6, qtconsole) but no display: Qt runs
off-screen. Settings go to a temporary home folder, so the screenshots never show recent projects or a saved layout.

The scenes:
- ``round``: a quarterly forecast round, the VAR beside the forecast fan chart;
- ``dsge``: Bayesian estimation of a New Keynesian model, prior and posterior beside the parameters;
- ``hd``: a structural VAR's historical decomposition into identified shocks;
- ``irf``: structural VAR impulse responses with bootstrap bands;
- ``event``: an event study of a staggered policy, with the dialog that set it up;
- ``record``: an estimated model's provenance above the Script panel, the analysis as Python that replays.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HOME"] = tempfile.mkdtemp()  # fresh settings: no recent projects, the default layout

SIZE = (1600, 1000)
app: Any = None


def pump(n: int = 30) -> None:
    for _ in range(n):
        app.processEvents()


def window(session: Any, bottom: int = 150) -> Any:
    """The main window on ``session``: the start page closed, the Report panel hidden, the Script panel ``bottom``
    pixels high."""
    from PySide6 import QtCore, QtWidgets

    from pymetrix.ui.qt import main_window

    w = main_window(session)
    w.resize(*SIZE)
    w.show()
    pump()
    for sub in w.mdi.subWindowList():
        if sub.windowTitle() == "Start":
            sub.close()
    docks = {d.windowTitle(): d for d in w.findChildren(QtWidgets.QDockWidget)}
    if "Report" in docks:
        docks["Report"].hide()
    docks["Script"].raise_()
    w.resizeDocks([docks["Script"]], [bottom], QtCore.Qt.Orientation.Vertical)
    w.resizeDocks([docks["Objects"]], [330], QtCore.Qt.Orientation.Horizontal)
    pump()
    return w


def area(w: Any) -> tuple[int, int]:
    """The width and height of the window's working area, where result windows open."""
    r = w.mdi.viewport().rect()
    return r.width(), r.height()


def show(w: Any, name: str, tab: str, geometry: tuple[int, int, int, int]) -> Any:
    """Open ``name`` in its window, on ``tab``, at ``geometry`` (x, y, width, height) in the working area."""
    before = set(w.mdi.subWindowList())
    w.show_object(name)
    pump()
    sub = next(iter(set(w.mdi.subWindowList()) - before), None) or w.mdi.activeSubWindow()
    view = sub.widget()
    for i in range(view.count()):
        if view.tabText(i) == tab:
            view.setCurrentIndex(i)
    sub.setGeometry(*geometry)
    pump(40)
    return sub


def full(w: Any, name: str, tab: str) -> None:
    show(w, name, tab, (0, 0, *area(w)))


def side_by_side(w: Any, left: tuple[str, str], right: tuple[str, str], split: float = 0.5) -> None:
    width, height = area(w)
    cut = int(width * split)
    show(w, *left, (0, 0, cut, height))
    show(w, *right, (cut, 0, width - cut, height))


def forecast_round() -> Any:
    from pymetrix import demo

    w = window(demo("forecast_round"), bottom=170)
    side_by_side(w, ("var", "Summary"), ("forecast_round", "Graph"), split=0.42)
    return w


def dsge() -> Any:
    from pymetrix import demo

    w = window(demo("nk_estimation"))
    side_by_side(w, ("posterior", "Graph"), ("estimated", "Parameters"))
    return w


def historical_decomposition() -> Any:
    from pymetrix import demo
    from pymetrix.commands import HistoricalDecomposition

    session = demo("forecast_round")
    session.run(HistoricalDecomposition(var="var", name="hd"))
    w = window(session)
    full(w, "hd", "Graph")
    return w


def impulse_responses() -> Any:
    from pymetrix import demo
    from pymetrix.commands import ImpulseResponse

    session = demo("forecast_round")
    session.run(ImpulseResponse(var="var", periods=20, draws=500, name="irf"))
    w = window(session)
    full(w, "irf", "Graph")
    return w


def event_study() -> Any:
    from PySide6.QtGui import QPainter

    from pymetrix import Dataset
    from pymetrix.commands import FitEventStudy, Session
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("policy_panel")
    session.run(FitEventStudy(outcome="y", treatment="d", name="event"))
    w = window(session)
    width, height = area(w)
    dialog = command_dialog(FitEventStudy, session, parent=w, initial=session.history[-1].command)
    dialog.show()
    pump(40)
    left = max(dialog.width() - 300, 0)  # the chart starts where the dialog ends
    show(w, "event", "Graph", (left, 0, width - left, height))
    dialog.move(w.geometry().x() + 12, w.geometry().y() + 300)
    pump(20)
    shot = w.grab()  # a dialog is a window of its own: painted onto the main window's image
    painter = QPainter(shot)
    painter.drawPixmap(dialog.pos() - w.pos(), dialog.grab())
    painter.end()
    dialog.close()
    return w, shot


def record() -> Any:
    from pymetrix import demo

    w = window(demo("nk_estimation"), bottom=330)
    full(w, "estimated", "Provenance")
    return w


SCENES: dict[str, tuple[str, Callable[[], Any]]] = {
    "round": ("1-forecast-round.png", forecast_round),
    "dsge": ("2-dsge-estimation.png", dsge),
    "hd": ("3-historical-decomposition.png", historical_decomposition),
    "irf": ("4-impulse-responses.png", impulse_responses),
    "event": ("5-event-study.png", event_study),
    "record": ("6-history-script-provenance.png", record),
}


def main(argv: list[str]) -> int:
    global app
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    out, wanted = argv[0], argv[1:] or list(SCENES)
    unknown = [n for n in wanted if n not in SCENES]
    if unknown:
        print(f"unknown scene {', '.join(unknown)}; the scenes are {', '.join(SCENES)}")
        return 2
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")  # convergence and library chatter from the demos
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font = app.font()
    font.setPointSizeF(10)
    app.setFont(font)
    kept = []  # the windows made so far, kept alive until the end
    for name in wanted:
        file, make = SCENES[name]
        made = make()
        w, shot = made if isinstance(made, tuple) else (made, None)
        pump(40)
        (shot or w.grab()).save(os.path.join(out, file))
        w.state["session"].dirty = False
        w.hide()  # not closed: closing saves the window's layout, which the next window would then restore
        kept.append(w)
        print(f"{name}: {os.path.join(out, file)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
