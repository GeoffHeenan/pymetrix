"""Long steps: progress, Cancel, and the window staying live while a step runs."""

import os
import time

import pytest

from pymetrix import VAR, Dataset, Equation
from pymetrix.commands import FitVAR, ImpulseResponse, Session
from pymetrix.progress import Cancelled, check, report, task


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(gy="400*dlog(y)", gc="400*dlog(c)")


def test_progress_does_nothing_outside_a_task_and_cancel_stops_inside_one():
    report(0.5, "anything")  # outside a task: nothing, no error
    check()
    with task() as t:
        report(0.25, "a quarter done")
        assert (t.fraction, t.message) == (0.25, "a quarter done")
        report(7)
        assert t.fraction == 1.0  # kept between 0 and 1
        t.cancel()
        with pytest.raises(Cancelled):
            report(0.5)
        with pytest.raises(Cancelled):
            check()
    report(0.9)  # the task is over


def test_the_long_loops_report_and_stop_when_cancelled(data):
    from pymetrix.estimation.rolling import fit_windows
    from pymetrix.forecasting import backtest

    var = VAR(["gy", "gc"], lags=1).fit(data)
    with task() as t:
        var.irf(4, draws=20)  # bootstrap draws report as they go
        assert t.fraction is not None and "bootstrap draw" in t.message
    fit = Equation("gc ~ gy").fit(data, span=(None, "2018Q4"))
    runs = [lambda: var.irf(4, draws=20), lambda: backtest(fit, data, ("2019Q1", "2020Q4"), horizon=2)]
    runs.append(lambda: fit_windows(Equation("gc ~ gy"), data, 40, None, None, None, 1))
    for run in runs:
        with task() as t:
            t.cancel()
            with pytest.raises(Cancelled):
                run()


def test_a_cancelled_step_leaves_no_trace(data):
    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("gy, gc", lags="1", name="var"))
    steps = len(s.history)
    with task() as t:
        t.cancel()
        with pytest.raises(Cancelled):
            s.run(ImpulseResponse("var", periods=4, draws=50, name="irf"))
    assert len(s.history) == steps and "irf" not in s.objects  # not recorded, nothing made


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_step_runs_in_the_background_with_progress_and_cancel(app, data, monkeypatch):
    from PySide6.QtCore import QTimer

    from pymetrix.ui import qt
    from pymetrix.ui.qt import busy, main_window, run_blocking

    assert run_blocking(lambda: 6 * 7) == 42  # a quick step: its result at once
    with pytest.raises(ZeroDivisionError):
        run_blocking(lambda: 1 / 0)  # its error, where it was asked for
    s = Session()
    s.objects["data"] = data
    window = main_window(s)
    try:
        ticks = []
        QTimer.singleShot(150, lambda: ticks.append("the window is live"))  # runs while the step does

        seen = []  # (progress shown, busy), looked at every 50 ms while the step runs
        look = QTimer()
        look.setInterval(50)
        look.timeout.connect(lambda: seen.append((window.progress_box.isVisibleTo(window), busy())))
        look.start()

        def slow() -> str:
            for k in range(12):
                report(k / 12, f"step {k}")
                time.sleep(0.06)
            # on a busy machine (several test processes) the window's timers can run late: the step lasts until the
            # bar has been seen, rather than the bar being looked for once at a fixed time (600 ms)
            deadline = time.monotonic() + 20
            while (True, True) not in seen and time.monotonic() < deadline:
                time.sleep(0.02)
            return "done"

        assert run_blocking(slow, "Slow") == "done"
        look.stop()
        assert ticks == ["the window is live"] and (True, True) in seen
        assert not window.progress_box.isVisibleTo(window) and not busy()  # gone when done

        def endless() -> None:
            while True:
                report(None, "working")
                time.sleep(0.02)

        QTimer.singleShot(500, window.cancel_step)  # Cancel, from the status bar
        with pytest.raises(Cancelled):
            run_blocking(endless, "Endless")

        def until_cancelled() -> str:  # gives up after 3 s, so a lost Cancel fails here rather than hanging
            for _ in range(150):
                report(None, "working")
                time.sleep(0.02)
            return "never cancelled"

        QTimer.singleShot(100, window.cancel_step)  # before the bar shows (0.4 s): it still stops the step
        with pytest.raises(Cancelled):
            run_blocking(until_cancelled, "Early cancel")
        said = []
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[1]))
        qt._BUSY["running"] = True  # while a step runs, another is refused
        try:
            run = [a for a in window.command_menus["Data"].actions() if a.text().replace("&", "") == "Plot series…"][0]
            run.trigger()
            assert said == ["A step is running"]
        finally:
            qt._BUSY["running"] = False
    finally:
        s.dirty = False
        window.close()


def test_the_garbage_collector_does_not_run_on_the_steps_thread(app):
    """A collection run on the worker thread would finalize Qt objects left over from earlier windows there, off the
    GUI thread (seen as a segmentation fault and as timers that stopped firing): it waits for the GUI thread."""
    import gc
    import threading

    from PySide6.QtWidgets import QWidget

    from pymetrix.ui.qt import run_blocking

    class Holder:
        pass

    def leftovers() -> None:  # Qt widgets in reference cycles: only the collector frees them
        for _ in range(50):
            widget, holder = QWidget(), Holder()
            holder.widget, widget.holder = widget, holder

    collected_on = []

    def note(phase, info):
        if phase == "stop":
            collected_on.append(threading.current_thread().name)

    def allocating() -> int:  # far more new objects than the collector's first threshold: a collection is due here
        return len([[k] for k in range(20 * max(gc.get_threshold()[0], 700))])

    gc.callbacks.append(note)
    try:
        leftovers()
        assert run_blocking(allocating, "Allocating") > 0
    finally:
        gc.callbacks.remove(note)
    from pymetrix.ui import collector

    # automatic collection back as it was, unless the GUI thread is the collector (pymetrix.ui.collector): then off
    assert gc.isenabled() is not collector.active() and "pymetrix step" not in collected_on
    gc.collect()  # the leftovers, here


def test_a_dialog_says_when_its_step_was_cancelled(app, data, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from pymetrix.ui import qt
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("gy, gc", lags="1", name="var"))
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[1:3]))

    def cancelled(fn, label=""):
        raise Cancelled("cancelled")

    monkeypatch.setattr(qt, "run_blocking", cancelled)
    dialog = command_dialog(ImpulseResponse, s, initial=ImpulseResponse("var", periods=4, draws=50, name="irf"))
    dialog.findChild(__import__("PySide6.QtWidgets", fromlist=["QDialogButtonBox"]).QDialogButtonBox).accepted.emit()
    assert said == [("Cancelled", "The step was cancelled; nothing was recorded.")] and dialog.result() == 0  # still open
    assert "irf" not in s.objects


def test_no_collection_during_a_step_reaches_what_existed_before_it(app):
    """However a collection starts during a step (JPype runs one on a Java thread after Java's collector; a library can
    turn automatic collection on again), objects that existed before the step (earlier windows' Qt objects) are frozen:
    collected only once the step is over, on the GUI thread. A collector turned on meanwhile is turned off again."""
    import gc
    import weakref

    from pymetrix.ui.qt import run_blocking

    class Holder:
        pass

    old = Holder()
    old.me = old  # garbage in a reference cycle, from before the step
    alive = weakref.ref(old)
    del old

    def step() -> tuple[bool, bool]:
        gc.enable()  # as parso's cache does
        gc.collect()  # a full collection off the GUI thread (here the step's own)
        survived = alive() is not None
        deadline = time.monotonic() + 20
        while gc.isenabled() and time.monotonic() < deadline:  # the GUI thread turns it off again
            time.sleep(0.02)
        return survived, not gc.isenabled()

    assert run_blocking(step, "Collecting") == (True, True)
    from pymetrix.ui import collector

    assert gc.isenabled() is not collector.active()  # turned on by the library; off for good with the GUI collector
    gc.collect()
    assert alive() is None  # collected after the step, here


def test_an_older_window_closing_leaves_the_open_windows_progress_and_menus(app):
    """Closing a window clears only its own progress bar and menus: a newer window keeps showing its progress (in a
    test process, an earlier test's window closing used to leave the next window with no progress bar at all)."""
    from PySide6.QtCore import QTimer

    from pymetrix.ui import qt

    old = qt.main_window(Session())
    new = qt.main_window(Session())
    old_menus = [m for m in qt._MENUS if qt._belongs_to(m, old)]
    new_menus = [m for m in qt._MENUS if qt._belongs_to(m, new)]
    assert old_menus and new_menus
    try:
        old.state["session"].dirty = False
        old.close()
        assert qt._PROGRESS.get("window") is new
        assert all(m in qt._MENUS for m in new_menus) and not any(m in qt._MENUS for m in old_menus)
        seen = []
        look = QTimer()
        look.setInterval(50)
        look.timeout.connect(lambda: seen.append(new.progress_box.isVisibleTo(new)))
        look.start()

        def slow() -> str:
            deadline = time.monotonic() + 20
            while True not in seen and time.monotonic() < deadline:
                report(None, "working")
                time.sleep(0.05)
            return "done"

        assert qt.run_blocking(slow, "Slow") == "done"
        look.stop()
        assert True in seen  # the open window showed its progress
    finally:
        new.state["session"].dirty = False
        new.close()
    assert "window" not in qt._PROGRESS  # the last window closed: nothing left registered


@pytest.mark.parametrize("first", ["old", "new"])
def test_show_goes_to_the_newest_open_window_and_prints_again_once_all_have_closed(app, data, capsys, monkeypatch, first):
    """show() opens a result in the newest window still open, whichever window closes first, and prints again once
    every window has closed, also after Qt deletes them. When the older window closed first, the newer window's close
    used to put back the handler it found on opening, the closed older window's, so show() printed nothing for the rest
    of the process (tests/test_selection.py's printing test failed after this file in the same worker)."""
    from PySide6.QtCore import QCoreApplication, QEvent

    from pymetrix import output
    from pymetrix.ui import qt

    monkeypatch.setattr("IPython.get_ipython", lambda: None)  # a plain script: show() prints
    before = output._show_handler
    windows = {"old": qt.main_window(Session()), "new": qt.main_window(Session())}
    try:
        assert output._show_handler is windows["new"].show_value
        windows[first].state["session"].dirty = False
        windows[first].close()
        other = "new" if first == "old" else "old"
        assert output._show_handler is windows[other].show_value  # the window still open
    finally:
        for window in windows.values():
            window.state["session"].dirty = False
            window.close()
    assert output._show_handler is before
    for window in windows.values():  # Qt deleting them later changes nothing
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()
    assert output._show_handler is before
    fit = Equation("c ~ y").fit(data)
    capsys.readouterr()
    output.show(fit)
    assert "const" in capsys.readouterr().out
