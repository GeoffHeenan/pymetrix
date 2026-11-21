"""pymetrix.ui.timers.single_shot: a callable run once a moment later, never after its context object is gone, on
every supported PySide6 (6.6.0 has no ``QTimer.singleShot(msec, context, callable)``; the helper's own timer is
tested on every version by forcing it)."""

import os
import sys

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix.ui import timers  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _native_works(app) -> bool:
    try:
        QTimer.singleShot(0, QObject(), lambda: None)
    except TypeError:
        return False
    return True


@pytest.fixture(params=["as found", "own timer"])
def path(request, app, monkeypatch):
    """Each test runs with the form this PySide6 takes, and again with the helper's own timer forced."""
    monkeypatch.setitem(timers._STATE, "native", None if request.param == "as found" else False)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, tb: errors.append(f"{kind.__name__}: {value}"))
    yield request.param
    assert not errors


def _settle(app, ms=60):
    QTest.qWait(ms)
    app.processEvents()


def test_the_callable_runs_once_after_the_delay(app, path):
    """Timed, not assumed: a slow machine may reach any check after the delay has passed."""
    import time

    context = QObject()
    calls = []
    delay = 200
    start = time.monotonic()
    timers.single_shot(delay, context, lambda: calls.append(time.monotonic() - start))
    app.processEvents()
    if time.monotonic() - start < delay / 1000 * 0.8:
        assert calls == []  # not yet
    waited = time.monotonic()
    while not calls and time.monotonic() - waited < 5:
        QTest.qWait(20)
    assert calls
    _settle(app, 2 * delay)
    assert len(calls) == 1  # once, not repeating
    assert calls[0] >= delay / 1000 * 0.8  # not before the delay (Qt's coarse timers may run up to 5% early)
    if path == "own timer":  # its timer is gone once it has fired
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert context.findChildren(QTimer) == []


def test_the_callable_does_not_run_once_its_context_is_deleted(app, path):
    calls = []
    for delay in (0, 20):
        context = QtWidgets.QWidget()
        timers.single_shot(delay, context, lambda: calls.append(delay))
        shiboken6.delete(context)  # destroyed before the timer fires
        _settle(app, 80)
    assert calls == []


def test_a_bound_method_of_the_context_runs_and_a_deleted_one_does_not(app, path):
    alive, gone = QtWidgets.QWidget(), QtWidgets.QWidget()
    alive.resize(300, 200)
    gone.resize(300, 200)
    timers.single_shot(0, alive, lambda: alive.resize(10, 10))
    timers.single_shot(0, gone, gone.close)
    shiboken6.delete(gone)
    _settle(app)
    assert alive.size().width() == 10
    shiboken6.delete(alive)


def test_the_form_this_pyside_takes_is_found_and_remembered(app, monkeypatch):
    monkeypatch.setitem(timers._STATE, "native", None)
    context = QObject()
    timers.single_shot(0, context, lambda: None)
    assert timers._STATE["native"] is _native_works(app)
    _settle(app)


def test_it_refuses_a_context_on_another_thread(app):
    import threading

    context = QObject()
    raised = []

    def other():
        try:
            timers.single_shot(0, context, lambda: None)
        except RuntimeError as exc:
            raised.append(str(exc))

    worker = threading.Thread(target=other)
    worker.start()
    worker.join()
    assert raised and "thread" in raised[0]
