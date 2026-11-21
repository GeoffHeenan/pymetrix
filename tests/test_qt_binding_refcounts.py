"""The Qt binding keeps Python's reference counts: a Qt call that returns nothing (a setter, update()) hands back None
without dropping a reference to it that it never took.

PySide6 6.12.0 did drop one on every such call. Where None is immortal (Python 3.12 and later) that is harmless; on
Python 3.11 None's count falls with every call until it reaches zero, and the process ends with "Fatal Python error:
none_dealloc: deallocating None" at the next garbage collection: the desktop app after enough use, and on CI the test
process that had run the most Qt calls (the full Linux Python 3.11 job lost its worker gw0 near the end of the run,
every run, once the window title painter and the statistics panel added calls on every repaint). pyproject.toml leaves
that release out on Python 3.11; this test names the fault at once, should a binding with it be installed again."""

import sys

import pytest

QtCore = pytest.importorskip("PySide6.QtCore")

CALLS = 1000


@pytest.mark.parametrize("call", ["QObject.setObjectName", "QSize.setWidth"])
def test_a_qt_call_that_returns_nothing_leaves_the_reference_count_of_none_alone(call):
    import PySide6

    obj, size = QtCore.QObject(), QtCore.QSize()
    run = {"QObject.setObjectName": lambda: obj.setObjectName("x"), "QSize.setWidth": lambda: size.setWidth(3)}[call]
    run()
    before = sys.getrefcount(None)
    for _ in range(CALLS):
        run()
    lost = before - sys.getrefcount(None)
    # one reference a call is the fault; a few either way are other threads' own use of None meanwhile
    assert lost < CALLS // 10, (f"PySide6 {PySide6.__version__} on Python {sys.version.split()[0]}: {call} dropped "
                                f"{lost} references to None in {CALLS} calls")
