"""Python's cyclic garbage collector runs on the GUI thread only, while the desktop app (or any window of it) runs.

Python collects cycles on whichever thread happens to be running when a collection is due, and JPype starts one on a
Java thread whenever Java's collector has run (once TRAMO-SEATS has started the JVM: ``JPGarbageCollection`` calls
``PyGC_Collect``). A cycle holding a Qt widget, freed there, can destroy the widget on that thread. Qt cannot stop the
widget's timers from it ("QObject::~QObject: Timers cannot be stopped from another thread"), so the GUI thread's
event dispatcher keeps them, and the next tick delivers a timer event to freed memory: an access violation in
``QCoreApplication::notifyInternal2`` from ``QEventDispatcherWin32Private::sendTimerEvent`` (Windows CI) or
``QTimerInfoList::activateTimers`` (Linux). Shown with PySide6 6.11.3 and JPype 1.7.1, and again on purpose in
``tests/test_ui_collector.py``.

So while Qt runs, automatic collection is off (``gc.disable``), which also makes ``PyGC_Collect`` do nothing, whoever
calls it, and the GUI thread collects instead: a timer on the application checks the collector's counts a few times a
second and collects the generations that are due, as Python itself would have. A library that turns automatic
collection back on (parso's cache does) finds it off again at the next check. ``gc.collect()`` called explicitly
still runs where it is called. Background steps also freeze what exists when they start (``qt.run_blocking``).
"""

from __future__ import annotations

import gc
from typing import Any

INTERVAL_MS = 250
_STATE: dict[str, Any] = {"timer": None}


def active() -> bool:
    """Whether the GUI thread is the collector's (``install`` has run, and its application still exists)."""
    timer = _STATE["timer"]
    if timer is None:
        return False
    import shiboken6

    return shiboken6.isValid(timer)


def install(app: Any) -> None:
    """Make the GUI thread of ``app`` (a QApplication, on its own thread) the only one that collects automatically.

    Safe to call more than once."""
    if active():
        return
    from PySide6.QtCore import QTimer

    gc.disable()
    timer = QTimer(app)  # the application's: made on, and ticking on, the GUI thread
    timer.setInterval(INTERVAL_MS)
    timer.timeout.connect(collect_due)
    timer.start()
    _STATE["timer"] = timer


def collect_due() -> int:
    """On the GUI thread: collect the generations Python would have collected by now; how many objects were freed."""
    if gc.isenabled():  # turned back on (a library): off again, so no other thread collects
        gc.disable()
    counts, thresholds = gc.get_count(), gc.get_threshold()
    if not thresholds[0]:
        return 0
    generation = -1
    for g in range(len(thresholds)):
        if counts[g] >= thresholds[g] > 0:
            generation = g
    return gc.collect(generation) if generation >= 0 else 0
