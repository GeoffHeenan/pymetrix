"""Python's cyclic collector runs on the GUI thread only, while Qt runs (``pymetrix.ui.collector``).

JPype runs Python's collector on a Java thread (its "Python Reference Queue" thread) whenever Java's collector has
run. A cycle holding a widget freed there destroys the widget on that thread, with its timers still registered on the
GUI thread ("QObject::~QObject: Timers cannot be stopped from another thread"), and the GUI thread's next timer event
goes to freed memory: the access violation in ``notifyInternal2`` from ``sendTimerEvent`` that crashed Windows test
workers (runs 42467298210 and 42479717551, in test_demo_chooser). SCRIPT below, without the collector, reproduced it on
Linux: with PySide6 6.6.0 and JPype 1.5.0 a SIGSEGV in ``QCoreApplication::notifyInternal2`` on the main thread with
the Reference Queue thread in native code (the JVM's crash report), and with PySide6 6.11.3 and JPype 1.7.1 Qt's two
"Timers cannot be stopped from another thread" warnings from the Java thread, then a segmentation fault in
``processEvents``: rarely (two runs in about ninety), but in every run without the collector most of Python's
collections started on the Java thread, and with it none did. ``python tests/test_ui_collector.py none`` runs it
without the collector.
"""

import ctypes
import gc
import os
import subprocess
import sys
import textwrap
import threading

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from pymetrix.ui import collector  # noqa: E402


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    collector.install(app)
    return app


def _widget_cycle(app, destroyed_on):
    """A parentless widget with a running timer, held only by a reference cycle."""
    gui = threading.get_native_id()
    w = QtWidgets.QWidget()
    t = QtCore.QTimer(w)
    t.setInterval(5)
    t.start()
    w.destroyed.connect(lambda *_: destroyed_on.append(threading.get_native_id() == gui),
                        QtCore.Qt.ConnectionType.DirectConnection)
    w._cycle = w


def test_no_other_thread_collects(app):
    assert collector.active() and not gc.isenabled()
    destroyed_on = []
    _widget_cycle(app, destroyed_on)
    collect = ctypes.pythonapi.PyGC_Collect  # what JPype calls on a Java thread
    collect.restype = ctypes.c_ssize_t
    freed = {}
    other = threading.Thread(target=lambda: freed.update(n=collect()))
    other.start()
    other.join()
    assert freed["n"] == 0 and destroyed_on == []  # nothing collected there
    gc.collect()  # on the GUI thread, as collect_due does
    app.processEvents()
    assert destroyed_on == [True]


def test_the_gui_thread_collects_what_is_due(app):
    destroyed_on = []
    kept = gc.get_threshold()
    try:
        gc.set_threshold(1, 1, 1)
        _widget_cycle(app, destroyed_on)
        assert collector.collect_due() > 0  # due at once with these thresholds
    finally:
        gc.set_threshold(*kept)
    assert destroyed_on == [True]


def test_a_library_turning_collection_on_finds_it_off_again(app):
    gc.enable()  # as parso's cache does
    collector.collect_due()
    assert not gc.isenabled()


SCRIPT = textwrap.dedent("""
    import collections, gc, os, sys, threading, time
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import jpype
    from PySide6 import QtCore, QtWidgets
    from pymetrix.ui import collector

    messages = []
    QtCore.qInstallMessageHandler(lambda mode, context, message: messages.append(message))
    app = QtWidgets.QApplication([])
    if sys.argv[1] == "collector":
        collector.install(app)
    gui = threading.get_native_id()
    started_on, destroyed_on = collections.Counter(), collections.Counter()
    gc.callbacks.append(lambda phase, info: started_on.update(["gui" if threading.get_native_id() == gui else "other"])
                        if phase == "start" else None)
    jpype.startJVM()
    Array = jpype.JArray(jpype.JByte)
    for _ in range(12):
        for _ in range(25):  # widgets with running timers, freed only by the cyclic collector
            w = QtWidgets.QWidget()
            t = QtCore.QTimer(w)
            t.setInterval(3)
            t.start()
            w.destroyed.connect(lambda *_: destroyed_on.update(["gui" if threading.get_native_id() == gui else "other"]),
                                QtCore.Qt.ConnectionType.DirectConnection)
            w._cycle = w
            del w, t
        for _ in range(200):  # Java garbage, so that Java's collector runs (and JPype calls Python's)
            Array(100_000)
        jpype.java.lang.System.gc()
        end = time.monotonic() + 0.3
        while time.monotonic() < end:  # the GUI thread's loop, with the collector's timer
            app.processEvents()
            QtCore.QThread.msleep(2)
    gc.collect()
    app.processEvents()
    print(repr((dict(started_on), dict(destroyed_on), [m for m in messages if "thread" in m])))
""")


def _jvm_or_skip():
    jpype = pytest.importorskip("jpype")
    try:
        jpype.getDefaultJVMPath()
    except Exception as exc:  # noqa: BLE001 - JPype's own errors for "no Java found"
        pytest.skip(f"no Java: {exc}")


def test_with_java_running_no_collection_starts_off_the_gui_thread(tmp_path):
    _jvm_or_skip()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # a crash report the JVM writes stays in this test's folder (as the workflow's own setting would put it elsewhere)
    env["JAVA_TOOL_OPTIONS"] = f"{env.get('JAVA_TOOL_OPTIONS', '')} -XX:ErrorFile={tmp_path / 'hs_err_pid%p.log'}".strip()
    done = subprocess.run([sys.executable, "-c", SCRIPT, "collector"], capture_output=True, text=True, timeout=240, env=env,
                          cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    started_on, destroyed_on, thread_messages = eval(done.stdout.strip().splitlines()[-1])  # noqa: S307 - our own output
    assert started_on.get("gui") and not started_on.get("other")  # without the collector: most ran on a Java thread
    assert destroyed_on == {"gui": 12 * 25}
    assert thread_messages == []


if __name__ == "__main__":  # the reproduction by hand: "none" (no collector, as before the fix) or "collector"
    import tempfile

    folder = tempfile.mkdtemp(prefix="collector-")  # the JVM's crash report, if any, lands here, not in the checkout
    print("running in", folder)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    subprocess.run([sys.executable, "-X", "faulthandler", "-c", SCRIPT, sys.argv[1] if len(sys.argv) > 1 else "none"],
                   check=False, cwd=folder, env=env)
