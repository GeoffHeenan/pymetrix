import os

# One BLAS thread per test process unless the environment says otherwise (set before NumPy loads its BLAS). The tests
# solve many small systems, where BLAS's threads cost more than they save, and on a busy machine (or with several test
# processes) far more: statsmodels' VARMAX fit in test_varma.py took 8 s on one thread and 360 s on four with the
# machine loaded. pymetrix itself limits its own long calculations the same way (pymetrix.threads).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pymetrix import Dataset  # noqa: E402


def pytest_configure(config):
    """The app's settings (window layout, course folders, recent projects) go to a folder of the test run, not the
    user's own: QSettings("pymetrix", "desktop") takes the default format, set here to INI files in that folder.

    Also names this run for the workers' crash reports (pytest_sessionstart): the main process's id, which the
    pytest-xdist workers, started after this, inherit."""
    os.environ.setdefault("PYMETRIX_FAULT_RUN", str(os.getpid()))
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        return
    import tempfile

    folder = tempfile.mkdtemp(prefix="pymetrix-settings-", dir=str(config.rootpath / ".pytest_tmp")
                              if (config.rootpath / ".pytest_tmp").is_dir() else None)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, folder)


def pytest_sessionfinish(session, exitstatus):
    """Stop Java, then either leave Qt alone (CI) or delete the Qt windows the tests left behind (local runs).

    Why: if Python destroys the windows during its final clean-up, it does so in no particular order, which can crash the
    process after every test has passed (seen on CI and locally). Locally, the windows are deleted here while Qt is still
    fully alive; deleting a window deletes its children too, so the list is re-read after each one; popups are left to
    their owners (a completer deletes its own list). Python's cyclic garbage collector is paused meanwhile: a collection
    that another thread happens to run would otherwise finalize Qt wrappers while this loop deletes the objects under them.

    On CI (any platform, once Qt's widgets are loaded) nothing is deleted and the process ends at once instead, as soon
    as pytest has reported (see pytest_unconfigure). The deleting loop itself crashed there: on Linux with the oldest
    supported PySide6, and on macOS (Python 3.11-3.13, "Segmentation fault: 11" after "[100%]") with the main thread in
    this loop and another thread garbage-collecting. Ending at once runs no Qt or Python teardown, so it cannot crash
    there, and the exit status is the one pytest computed, so failures still fail the job."""
    import gc
    import sys

    session.config._pymetrix_exitstatus = int(exitstatus)
    _breadcrumb(f"sessionfinish, status {int(exitstatus)}")
    jdemetra = sys.modules.get("pymetrix_seasonal.jdemetra")
    if jdemetra is not None:
        _breadcrumb("stopping Java")
        jdemetra.shutdown()  # Java (TRAMO-SEATS, the seasonal plugin) first, as the app does when it quits: on Windows, Qt's teardown crashed with it running
    if _end_at_once():
        gc.disable()  # nothing needs collecting before the process ends; no collection can race what is left of the run
        _breadcrumb("sessionfinish done")
        return
    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets is not None else None
    if app is None:
        return
    import shiboken6
    from PySide6.QtCore import Qt

    popup, tooltip = Qt.WindowType.Popup, Qt.WindowType.ToolTip

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(1000):
            windows = [w for w in app.topLevelWidgets() if shiboken6.isValid(w) and w.parent() is None
                       and w.windowType() not in (popup, tooltip)]  # popups belong to their completer or menu
            if not windows:
                break
            windows[0].hide()
            shiboken6.delete(windows[0])
            app.processEvents()
    finally:
        if was_enabled:
            gc.enable()


_fault_report = None  # this worker's crash report file, open until the process ends (pytest_sessionstart)


def _fault_files(config):
    """Where each pytest-xdist worker of this run writes its crash report: $PYMETRIX_FAULT_DIR, else $RUNNER_TEMP
    (GitHub Actions), else the system's temporary folder; one file per worker, named by the run and the worker."""
    import tempfile

    folder = os.environ.get("PYMETRIX_FAULT_DIR") or os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    return os.path.join(folder, f"pymetrix-faults-{os.environ['PYMETRIX_FAULT_RUN']}")


def _breadcrumb(text: str) -> None:
    """A line in this worker's crash report: what it was doing, so a worker that ends with no signal (nothing for
    faulthandler to report) still shows how far it got."""
    if _fault_report is not None:
        try:
            _fault_report.write(text + "\n")
            _fault_report.flush()
        except (OSError, ValueError):
            pass


def pytest_sessionstart(session):
    """In each pytest-xdist worker, a crash report of its own: on a fatal signal (a segmentation fault, an abort)
    faulthandler writes every thread's Python stack there, after the worker's breadcrumbs (each test as it starts, and
    the steps of the end of the run). A worker that died said only "worker 'gw0' crashed": its stderr is not reliably
    in the log. The main process removes the file of a worker that finished cleanly and adds the exit status of one
    that did not (pytest_testnodedown); the workflow prints what is left when a job fails. Runs after pytest's own
    faulthandler set-up (pytest_configure), and replaces its file."""
    global _fault_report
    import faulthandler

    workerinput = getattr(session.config, "workerinput", None)
    if workerinput is None:
        return
    try:
        _fault_report = open(f"{_fault_files(session.config)}-{workerinput['workerid']}.txt", "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        return
    _breadcrumb(f"worker {workerinput['workerid']}, process {os.getpid()}")
    faulthandler.enable(file=_fault_report, all_threads=True)
    _keep_captured_stderr(session.config, f"{_fault_files(session.config)}-{workerinput['workerid']}-stderr.txt")


def _keep_captured_stderr(config, path: str) -> None:
    """Make pytest's capture of this worker's stderr (file level, the default) a named file, so what a test wrote to
    stderr last outlives a crash: above all Python's own "Fatal Python error" message and stack, which Python writes to
    stderr itself before it disables faulthandler and aborts, leaving the crash report above empty. pytest reads and
    empties the file after each test as before, so nothing else changes. Uses pytest's capture internals (the global
    capture's stderr FDCapture and its tmpfile): with a pytest whose internals differ, or with --capture=sys or no, it
    does nothing."""
    capman = config.pluginmanager.getplugin("capturemanager")
    capturing = getattr(capman, "_global_capturing", None)
    err = getattr(capturing, "err", None)
    if err is None or getattr(err, "targetfd", None) != 2 or not hasattr(err, "tmpfile"):
        return
    try:
        from _pytest.capture import EncodedFile

        named = EncodedFile(open(path, "w+b", buffering=0), encoding="utf-8", errors="replace", newline="",  # noqa: SIM115
                            write_through=True)
    except (ImportError, OSError):
        return
    running = getattr(err, "_state", None) == "started"
    if running:
        capman.suspend_global_capture(in_=False)
    try:
        old, err.tmpfile = err.tmpfile, named
        if getattr(err, "syscapture", None) is not None and getattr(err.syscapture, "tmpfile", None) is old:
            err.syscapture.tmpfile = named
        old.close()
    finally:
        if running:
            capman.resume_global_capture()


def pytest_runtest_logstart(nodeid, location):
    _record_qt_messages()
    _collect_on_the_gui_thread()
    _breadcrumb(f"start {nodeid}")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """After each test, the garbage collection that is due, here on the GUI (main) thread (``_collect_on_the_gui_thread``)."""
    if _collect_on_the_gui_thread():
        import sys

        sys.modules["pymetrix.ui.collector"].collect_due()


def _collect_on_the_gui_thread() -> bool:
    """Once this process has a QApplication, Python's cyclic collector runs on the GUI thread only, as in the app
    (``pymetrix.ui.collector``): JPype would otherwise run it on a Java thread, where a widget it frees is destroyed
    with its timers still registered on the GUI thread, which crashed Windows workers in a timer event later. Tests
    that make dialogs but no main window need it as much as those that do (``main_window`` installs it); the tests
    rarely run the event loop, so the collection that is due also runs after each test. True once installed."""
    import sys

    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets is not None else None
    if app is None:
        return False
    from pymetrix.ui import collector

    collector.install(app)
    return True


_qt_message_handler = []  # the handler, kept alive once installed


def _record_qt_messages() -> None:
    """Once this worker has loaded Qt, Qt's own messages go into its crash report too, with the thread that sent each.

    pytest keeps a test's stderr only until the test ends, so a warning Qt gave a few tests before a crash (an object
    destroyed on the wrong thread, "Timers cannot be stopped from another thread", say) was lost. Every message is
    still written to stderr as Qt would; only the offscreen platform's "This plugin does not support ..." notes, given
    by every window, stay out of the report. Installed at the first test start after PySide6.QtCore is imported, so
    it imports nothing itself."""
    if _qt_message_handler or _fault_report is None:
        return
    import sys

    core = sys.modules.get("PySide6.QtCore")
    if core is None:
        return
    import threading

    def handler(mode, context, message):
        if not message.startswith("This plugin does not support"):  # the record first: it must not depend on stderr
            _breadcrumb(f"Qt {getattr(mode, 'name', mode)} [{threading.current_thread().name}]: {message}")
        try:
            print(message, file=sys.stderr)
        except Exception:  # noqa: BLE001 - a Java thread mid-collection, pytest's capture closed: the record is kept
            pass

    core.qInstallMessageHandler(handler)
    _qt_message_handler.append(handler)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """In the main process, as a pytest-xdist worker ends: a clean end removes its crash report; any other adds the
    process's exit status (negative: the signal that ended it; zero or more: it exited without reporting)."""
    path = f"{_fault_files(node.config)}-{node.gateway.id}.txt"
    try:
        if error is None:
            for each in (path, path[:-4] + "-stderr.txt"):
                if os.path.exists(each):
                    os.remove(each)
            return
        status = None
        try:
            status = node.gateway._io.popen.wait(timeout=30)  # execnet's popen gateway: the worker's process
        except Exception:  # noqa: BLE001 - another kind of gateway, or still running: no status
            pass
        with open(path, "a", encoding="utf-8") as report:
            report.write(f"main process: the worker went down ({error}); its exit status: {status}\n")
    except OSError:
        pass


def _end_at_once() -> bool:
    """On CI with Qt's widgets loaded, end the process without tearing Qt (or Python) down."""
    import os
    import sys

    return bool(os.environ.get("CI")) and "PySide6.QtWidgets" in sys.modules


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """End the process once pytest has reported, with pytest's own exit status (see pytest_sessionfinish).

    Only when pytest_sessionfinish recorded that status; otherwise pytest exits normally, so no status is ever guessed.
    Python's buffers are flushed first; what they hand to the operating system is not lost when the process ends.
    On Linux and macOS os._exit calls _exit(2), which runs no atexit handlers and no C++ static destructors, so none
    of Qt's shutdown code. On Windows os._exit calls the C runtime's _exit, which still ends through ExitProcess:
    every DLL, Qt's included, gets DLL_PROCESS_DETACH and runs its static destructors. TerminateProcess on the
    process itself skips that, so it is used there; os._exit stays as the fallback should it ever return.

    With pytest-xdist (-n) each worker that loaded Qt ends here too. By then it has sent every result and its
    "finished" message: pytest-xdist writes them to the pipe, and flushes it, from pytest_sessionfinish, before this
    hook; the main process takes a worker that ends after "finished" as done, not crashed, and computes the run's exit
    status from the results. The main process runs no tests, never loads Qt's widgets, and exits normally."""
    import os
    import sys

    status = getattr(config, "_pymetrix_exitstatus", None)
    if status is None or not _end_at_once():
        return
    sys.stdout.flush()
    sys.stderr.flush()
    _breadcrumb(f"ending the process, status {status}")
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), status)
    os._exit(status)


@pytest.fixture
def first_run():
    """The main window as a first run opens it: no window arrangement saved by an earlier window of the same test
    process, and the app's settings put back as they were afterwards.

    Each test process (each pytest-xdist worker) keeps one settings folder (pytest_configure), and a main window saves
    its arrangement when it closes (geometry, the panels' sizes, the workspace); the next window restores it and, a
    saved arrangement being the user's, leaves the panels' sizes alone. So a test that checks what a window shows as
    it first opens depends on whichever tests ran before it in that process: after a window of 1400 by 1000, the
    Immediate panel of a 1093 by 614 window is restored 173 pixels tall rather than fitted to 96, and the workspace
    loses 77 pixels of its height."""
    pytest.importorskip("PySide6.QtCore")
    from PySide6.QtCore import QSettings

    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    kept = {key: settings.value(key) for key in settings.allKeys()}
    for key in ("geometry", "state", "workspace"):  # "workspace" with each workspace's own arrangement under it
        settings.remove(key)
    settings.sync()
    yield
    settings.clear()
    for key, value in kept.items():
        settings.setValue(key, value)
    settings.sync()


@pytest.fixture(autouse=True, scope="session")
def _private_trust_store(tmp_path_factory):
    """Keep project trust decisions made by tests out of the user's own ~/.pymetrix."""
    import os

    os.environ["PYMETRIX_HOME"] = str(tmp_path_factory.mktemp("pymetrix_home"))
    yield


@pytest.fixture(autouse=True, scope="module")
def _qt_garbage_collected_here():
    """After each test module that loaded Qt's widgets, its garbage is collected here, on the GUI (main) thread.

    Windows, dialogs and consoles a module left behind would otherwise wait for whichever thread next runs the
    collector: a background step's worker, or a Java thread (JPype runs a full collection there after Java's collector
    has run). Finalizing Qt objects off the GUI thread can crash the process (seen on CI with the main thread painting).
    Objects deleted with deleteLater go first, as the event loop would delete them."""
    yield
    import gc
    import sys

    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets is not None else None
    if app is None:
        return
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()


@pytest.fixture(autouse=True, scope="module")
def _consoles_shut_down_here():
    """After each test module, the Immediate consoles its tests left open (windows never closed) are shut down.

    Each console's in-process kernel runs an IOPub thread with a ZeroMQ context of two native threads; left open, they
    piled up for the life of a test worker (over 200 threads on Windows CI by the time a worker crashed). Torn down
    before _qt_garbage_collected_here's collection (defined after it); module-scoped, as some modules share a window."""
    yield
    import sys

    console_qt = sys.modules.get("pymetrix.ui.console_qt")
    if console_qt is not None:
        console_qt.shutdown_all()


@pytest.fixture(autouse=True)
def _code_notation():
    """Every test starts and ends in code notation, whatever a desktop setting or another test chose."""
    from pymetrix.notation import set_notation

    set_notation("code")
    yield
    set_notation("code")


@pytest.fixture
def ts() -> Dataset:
    """Quarterly time series, 2000Q1-2024Q4, with AR(1) errors."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + rng.normal(scale=0.5)
    y = 1.0 + 0.5 * x - 0.3 * z + e
    yd = 100 * np.exp(np.cumsum(rng.normal(0.005, 0.01, size=n)))
    c = 0.8 * yd * np.exp(rng.normal(0, 0.005, size=n))
    df = pd.DataFrame(
        {"y": y, "x": x, "z": z, "yd": yd, "c": c},
        index=pd.period_range("2000Q1", periods=n, freq="Q", name="time"),
    )
    return Dataset.from_frame(df)


@pytest.fixture
def panel() -> Dataset:
    """12 countries x 20 years with country effects."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(12):
        alpha = rng.normal(scale=2)
        for year in range(2000, 2020):
            x = rng.normal() + 0.3 * alpha
            rows.append({"country": f"C{i:02d}", "year": year, "x": x, "w": rng.normal(),
                         "y": alpha + 0.7 * x + rng.normal(scale=0.5), "region": "A" if i < 6 else "B"})
    return Dataset.from_frame(pd.DataFrame(rows), entity="country", time="year", freq="Y")
