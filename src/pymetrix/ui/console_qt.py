"""The Immediate window: a Jupyter console (qtconsole) running in this process.

Its namespace is the session's: dialog results are there to use, and objects
made at the console appear in the Objects panel. Every cell is recorded in the
history (see ``Session.begin_code`` and ``Session.end_code``), so the script
and projects reproduce it.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..commands import Session

BANNER = "PyMetrix Immediate window: Python with the session's objects. Everything run here is recorded in the history.\n"


def available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("qtconsole", "qtpy", "ipykernel"))


def palette_is_dark() -> bool:
    """Whether the application's palette is dark (its window colour darker than mid-grey): the OS's dark mode."""
    from PySide6.QtGui import QGuiApplication, QPalette

    app = QGuiApplication.instance()
    return app is not None and app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def _rescale_console(widget: Any, ratio: float) -> None:
    """The console's text at ``ratio`` of its size now (View → Text size, ``pymetrix.ui.text_size``)."""
    from PySide6.QtGui import QFont

    font = QFont(widget.font)  # qtconsole's own property: its document's default font
    font.setPointSizeF(font.pointSizeF() * ratio)
    widget.font = font


class Immediate:
    """The console widget, bound to one session at a time.

    ``on_change`` is called after each cell with the new history record, so the
    window can refresh its panels. ``source`` names where the next cell comes
    from ("immediate", or "editor" when the Code window runs code here).
    """

    def __init__(self, session: Session, on_change: Callable[[Any], None]) -> None:
        os.environ.setdefault("QT_API", "pyside6")
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")  # quiets a debugger notice at kernel start
        from qtconsole.inprocess import QtInProcessKernelManager
        from qtconsole.rich_jupyter_widget import RichJupyterWidget

        from traitlets.config import Config

        config = Config()
        # pymetrix records the history itself; IPython's own history database would add a
        # background thread that can crash Python when it exits with Qt still loaded
        config.HistoryManager.enabled = False
        config.InteractiveShell.enable_tip = False  # IPython's tips (and its Windows apology) in the banner
        config.InteractiveShell.banner1 = ""  # Python's version and copyright lines: the window's own line says enough
        config.InteractiveShell.banner2 = ""
        self.manager = QtInProcessKernelManager(config=config)
        self.manager.start_kernel(show_banner=False)
        self.shell = self.manager.kernel.shell
        self.client = self.manager.client()
        self.client.start_channels()
        self.widget = RichJupyterWidget()
        if palette_is_dark():  # qtconsole draws its own colours: white unless told, under a dark palette too
            self.widget.set_default_style("linux")
        self.widget.banner = BANNER
        # its text is in its document's font (from the app's as it was made): View → Text size changes it there too
        self.widget.text_size_changed = lambda ratio, w=self.widget: _rescale_console(w, ratio)
        self.widget.kernel_manager, self.widget.kernel_client = self.manager, self.client
        self.on_change = on_change
        self.source = "immediate"
        self.last_error: BaseException | None = None
        self._before: dict[str, int] | None = None
        # IPython's shell is one per process (a singleton), shared by every window's console: this window's hooks
        # come off it when the window closes or its console is deleted, or code run in a later window would run them
        # too, refreshing a closed window (and its plugins' panels, their widgets maybe deleted: a crash)
        self.shell.events.register("pre_run_cell", self._pre)
        self.shell.events.register("post_run_cell", self._post)
        shell, pre, post = self.shell, self._pre, self._post
        self.widget.destroyed.connect(lambda *_: _unregister(shell, pre, post))
        _LIVE[id(self)] = self
        self.bind(session)

    def bind(self, session: Session) -> None:
        """Use ``session``'s namespace from now on (a new session, or an opened project)."""
        self.session = session
        session.push()
        self.shell.user_ns.clear()
        self.shell.init_user_ns()
        self.shell.user_ns.update(session.namespace)
        session.namespace = self.shell.user_ns  # one namespace, shared with the session

    def _pre(self, info: Any) -> None:
        self._before = self.session.begin_code()

    def _post(self, result: Any) -> None:
        code = result.info.raw_cell
        source, self.source = self.source, "immediate"
        if self._before is None or not code.strip() or getattr(result.info, "silent", False):
            return
        error = result.error_before_exec or result.error_in_exec
        self.last_error = error
        record = self.session.end_code(self._before, code, ok=error is None,
                                       error=None if error is None else f"{type(error).__name__}: {error}", source=source)
        self._before = None
        self.on_change(record)

    def execute(self, code: str, source: str = "immediate", hidden: bool = False) -> None:
        """Run code as if typed (shown in the console, recorded in the history)."""
        self.source = source
        self.widget.execute(code.rstrip("\n"), hidden=hidden)

    def shutdown(self) -> None:
        """Stop the console's kernel and free what it holds: its hooks, channels, IOPub thread and ZeroMQ context.

        The in-process kernel's IOPub thread signals itself through sockets on a ZeroMQ context of its own (the
        kernel's dummy IOPub socket's). Stopping the kernel joins the thread and closes those sockets but never
        terminates the context, whose two native threads (its I/O thread and reaper) would then outlive the
        console: two more for every window closed, for as long as the process runs. Safe to call more than once."""
        _unregister(self.shell, self._pre, self._post)
        _LIVE.pop(id(self), None)
        self.client.stop_channels()
        kernel = self.manager.kernel
        self.manager.shutdown_kernel()
        if kernel is not None:
            _free_iopub(kernel.iopub_thread)


# consoles not shut down yet, held until they are (each one's IOPub thread runs till then whatever else is
# collected): the test session shuts down those its tests leave open (tests/conftest.py)
_LIVE: dict[int, Immediate] = {}


def _free_iopub(iopub: Any) -> None:
    """Terminate a stopped IOPub thread's ZeroMQ context."""
    if iopub.thread.is_alive():  # did not stop: its sockets may still be in use there, so leave them
        return
    context = iopub.socket.context
    if not context.closed:
        context.destroy(linger=0)  # closes any socket left on it, then ends its threads


def shutdown_all() -> int:
    """Shut down every console not shut down yet; the number shut down."""
    consoles = list(_LIVE.values())
    for console in consoles:
        console.shutdown()
    return len(consoles)


def _unregister(shell: Any, pre: Callable[[Any], None], post: Callable[[Any], None]) -> None:
    """Take a console's hooks off IPython's shell (each once; gone already is fine)."""
    for event, callback in (("pre_run_cell", pre), ("post_run_cell", post)):
        try:
            shell.events.unregister(event, callback)
        except ValueError:  # already taken off
            pass
