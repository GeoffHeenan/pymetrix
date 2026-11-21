"""Run a callable once, a moment from now, unless the object it belongs to is gone by then.

``QTimer.singleShot(msec, context, callable)`` does this, but only on newer PySide6: 6.6.0, the oldest version
pymetrix supports, takes a context object only with a slot's signature as bytes and raises ``TypeError`` for a
callable. ``single_shot`` uses that form where it works and otherwise a single-shot ``QTimer`` that is a child of
``context``: Qt destroys the timer with its parent, so the callable never runs once ``context`` is gone, as with
Qt's own form.

Call it on the thread ``context`` lives on (in the app, the GUI thread): a timer cannot be given a parent on another
thread, and timers belong on the GUI thread (``pymetrix.ui.collector``).
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["single_shot"]

_STATE: dict[str, Any] = {"native": None}  # whether QTimer.singleShot takes (msec, context, callable); None: not tried


def single_shot(msec: int, context: Any, callback: Callable[[], Any]) -> None:
    """Call ``callback`` once after ``msec`` milliseconds, on ``context``'s thread, unless ``context`` (a QObject) has
    been destroyed by then."""
    from PySide6.QtCore import QThread, QTimer

    if context.thread() is not QThread.currentThread():
        raise RuntimeError("single_shot must be called on the thread its context object lives on")
    if _STATE["native"] is not False:
        try:
            QTimer.singleShot(msec, context, callback)
        except TypeError:  # e.g. PySide6 6.6.0: no (msec, context, callable) form; nothing was scheduled
            _STATE["native"] = False
        else:
            _STATE["native"] = True
            return
    timer = QTimer(context)  # destroyed with context, and with it the connection: callback cannot outlive it
    timer.setSingleShot(True)
    timer.setInterval(max(0, int(msec)))
    timer.timeout.connect(callback)
    timer.timeout.connect(timer.deleteLater)  # done: not left on context for its lifetime
    timer.start()
