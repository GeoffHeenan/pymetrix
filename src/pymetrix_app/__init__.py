"""The PyMetrix desktop app's launcher: its name, version and copyright, and the opening screen.

``pymetrix-gui`` starts here, not in ``pymetrix.ui.qt``, so that the opening screen shows at once. Importing the
``pymetrix`` package loads pandas, scipy and statsmodels, which takes several seconds on a first start; this module
needs only Qt, so the window that says what is starting appears first and the rest loads behind it.

The copyright line is left blank (``COPYRIGHT_HOLDER``) until it is settled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = ["COPYRIGHT_HOLDER", "COPYRIGHT_YEAR", "ICON", "ICON_WINDOWS", "NAME", "about_lines", "identify", "main", "splash", "update",
           "version"]

NAME = "PyMetrix"
TAGLINE = "Applied macroeconometrics, from data to forecast report"
COPYRIGHT_YEAR = "2026"
COPYRIGHT_HOLDER = ""  # to be settled; the line reads "Copyright © 2026" until then
ICON = Path(__file__).with_name("icons") / "pymetrix.png"  # the window's and the Linux launcher's
ICON_WINDOWS = ICON.with_suffix(".ico")  # the Windows shortcut's, at every size Windows asks for
APP_ID = "PyMetrix.Desktop"  # Windows groups the taskbar by this, not by python.exe
INK, PAPER, MUTED = "#1b2a2f", "#f4efe6", "#9aa7a9"  # the icon's colours (scripts/make_icon.py)


def version() -> str:
    """pymetrix's version, without importing it: from the _version.py beside this package (the code that runs, from a
    wheel or a clone), else from the installed metadata. Metadata alone would be stale in an editable install whose
    source has moved on to a new version since it was installed."""
    source = Path(__file__).resolve().parents[1] / "pymetrix" / "_version.py"
    for line in source.read_text(encoding="utf-8").splitlines() if source.is_file() else ():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    try:
        from importlib.metadata import version as installed

        return installed("pymetrix")
    except Exception:  # noqa: BLE001 - neither (a broken install)
        return ""


def copyright_line() -> str:
    return " ".join(part for part in ("Copyright ©", COPYRIGHT_YEAR, COPYRIGHT_HOLDER) if part)


def about_lines() -> list[str]:
    """The name, version, tagline and copyright, one to a line, as the opening screen and Help → About show them."""
    return [NAME, f"Version {version()}", TAGLINE, copyright_line()]


def identify(app: Any) -> None:
    """Name the app to Qt and the system: its name, icon and, on Windows, its own taskbar entry. Without the app ID,
    Windows shows the app as Python (pythonw.exe), with Python's icon, grouped with any other Python program."""
    from PySide6.QtGui import QIcon

    app.setApplicationName(NAME)
    app.setApplicationDisplayName(NAME)
    app.setApplicationVersion(version())
    if ICON.is_file():
        app.setWindowIcon(QIcon(str(ICON)))
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except (AttributeError, OSError):  # an older Windows: the taskbar groups it with Python, as before
            pass


def splash(app: Any) -> Any:
    """The opening screen, shown at once and kept up until the main window replaces it (``finish``); its
    ``showMessage`` says what is loading."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    ratio = app.primaryScreen().devicePixelRatio() if app.primaryScreen() else 1.0
    width, height = 480, 260
    pixmap = QPixmap(int(width * ratio), int(height * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QColor(INK))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    if ICON.is_file():
        painter.drawPixmap(QRect(width - 32 - 92, 28, 92, 92), QPixmap(str(ICON)))
    painter.setPen(QColor(PAPER))
    name, number, tagline, rights = about_lines()
    for text, size, bold, top, line_height in ((name, 30, True, 46, 50), (number, 12, False, 100, 22),
                                               (tagline, 11, False, 128, 22), (rights, 9, False, 196, 18)):
        font = QFont()
        font.setPointSize(size)
        font.setBold(bold)
        painter.setFont(font)
        painter.drawText(QRect(32, top, width - 64 - (104 if top < 120 else 0), line_height), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
    painter.end()

    class Screen(QSplashScreen):
        def drawContents(self, painter: Any) -> None:  # the loading message, lined up with the text above it
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect().adjusted(32, 0, -32, -14), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                             self.message())

    screen = Screen(pixmap)
    screen.setObjectName("pymetrix-splash")
    screen.show()
    screen.showMessage("Starting…", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor(MUTED))
    app.processEvents()
    return screen


def checkout() -> Path | None:
    """The git checkout this PyMetrix runs from (an editable install, ``pip install -e``), or None if it was
    installed from a wheel."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() and (root / "src" / "pymetrix").is_dir() else None


def update(root: Path | None = None) -> tuple[bool, str]:
    """``--latest``: bring the checkout up to date with ``git pull --ff-only`` before pymetrix loads, so the app
    starts on the most recent build of its branch. Returns (updated, a line saying what happened). Nothing stops
    the app starting: offline, with local changes in the way, or installed from a wheel, it starts as it is."""
    import subprocess

    root = root or checkout()
    if root is None:
        return False, "Installed from a package, not a git checkout: --latest has nothing to update"

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
                              timeout=60).stdout.strip()

    try:
        before = git("rev-parse", "--short", "HEAD")
        git("pull", "--ff-only", "--quiet")
        after = git("rev-parse", "--short", "HEAD")
    except FileNotFoundError:
        return False, "git is not installed: starting the build as it is"
    except subprocess.TimeoutExpired:
        return False, "git pull took too long (offline?): starting the build as it is"
    except subprocess.CalledProcessError as error:
        reason = (error.stderr or "").strip().splitlines()
        return False, "Could not update" + (f" ({reason[-1]})" if reason else "") + ": starting the build as it is"
    if before == after:
        return False, f"Up to date at {after}"
    return True, f"Updated {before} → {after}"


def main(argv: list[str] | None = None) -> None:
    """``pymetrix-gui``: the opening screen at once, then the app (``pymetrix.ui.qt.main``) loaded behind it.
    With ``--latest``, the checkout is updated first (``update``), while the opening screen says so."""
    args = sys.argv[1:] if argv is None else list(argv)
    if "--shortcut" in args or "-h" in args or "--help" in args:  # no window for these
        from pymetrix.ui.qt import main as run

        run(args)
        return
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise SystemExit("the desktop app needs PySide6: pip install pymetrix[qt]") from None
    app = QApplication.instance() or QApplication(sys.argv[:1] + args)
    identify(app)
    screen = splash(app)
    if "--latest" in args:  # before pymetrix is imported, so the app runs the updated code
        import time

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor

        def say(text: str) -> None:
            screen.showMessage(text, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor(MUTED))
            app.processEvents()

        say("Checking for the latest build…")
        _, outcome = update()
        say(outcome)
        print(outcome, file=sys.stderr)
        time.sleep(1.5)  # long enough to read before the loading messages replace it
    from pymetrix.ui.qt import main as run

    run(args, app=app, splash=screen)


if __name__ == "__main__":
    main()
