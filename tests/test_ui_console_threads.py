"""The Immediate console leaves no threads behind: closing its window ends its kernel's IOPub thread and ZeroMQ context.

An in-process Jupyter kernel's IOPub thread signals itself through sockets on a ZeroMQ context of its own, whose two
native threads (I/O and reaper) are invisible to ``threading``. Stopping the kernel did not terminate that context, so
every window closed kept two threads for the life of the process; a Windows CI worker reached over 200 of them.
Counted two ways: open ``zmq.Context`` objects (each holds its two threads; any platform) and, on Linux, the threads
ZeroMQ names ``ZMQbg/...``.
"""

import gc
import os
import sys
import threading

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")
zmq = pytest.importorskip("zmq")

from pymetrix.commands import Session  # noqa: E402


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _open_contexts() -> int:
    gc.collect()
    return sum(isinstance(o, zmq.Context) and not o.closed for o in gc.get_objects())


def _zmq_threads() -> int | None:
    """ZeroMQ's own threads, on Linux (where they are named); None elsewhere."""
    if not sys.platform.startswith("linux"):
        return None
    count = 0
    for task in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{task}/comm") as f:
                count += f.read().startswith("ZMQbg")
        except OSError:  # a thread that ended meanwhile
            pass
    return count


def _counts() -> tuple[int, int | None, int]:
    return _open_contexts(), _zmq_threads(), threading.active_count()


def test_closing_windows_leaves_no_zmq_threads(app):
    from pymetrix.ui.qt import main_window

    before = _counts()
    for _ in range(3):
        w = main_window(Session())
        w.show()
        app.processEvents()
        assert _open_contexts() == before[0] + 1  # the console's own context, while it is open
        w.state["session"].dirty = False
        assert w.close()
        w.deleteLater()
        app.processEvents()
    assert _counts() == before  # three windows made and closed: no context, ZeroMQ thread or Python thread left


def test_a_console_shut_down_twice_or_left_open(app):
    from pymetrix.ui import console_qt

    before = _counts()
    console = console_qt.Immediate(Session(), lambda record: None)
    console.shutdown()
    console.shutdown()  # a second time (the window closing after the test session shut it down) is harmless
    assert _counts() == before
    left_open = [console_qt.Immediate(Session(), lambda record: None) for _ in range(2)]
    assert _open_contexts() == before[0] + 2
    assert console_qt.shutdown_all() == 2  # what conftest's _consoles_shut_down_here does after each module
    assert console_qt.shutdown_all() == 0
    assert _counts() == before
    for c in left_open:
        c.widget.deleteLater()
    console.widget.deleteLater()
    app.processEvents()
