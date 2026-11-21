"""The app's launcher (``pymetrix-gui``, ``pymetrix_app``): it shows the opening screen before pymetrix loads, says the
name, version and copyright, hands over to the main window, and the name is PyMetrix wherever the app shows it."""

import os
import subprocess
import sys

import pytest

import pymetrix
import pymetrix_app


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_launcher_loads_nothing_heavy():
    """Importing it reads the version without importing pymetrix, pandas or scipy, so the opening screen comes first."""
    probe = ("import sys, pymetrix_app; v = pymetrix_app.version(); "
             "print(v, [m for m in ('pymetrix', 'pandas', 'scipy', 'numpy') if m in sys.modules])")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [pymetrix.__version__, "[]"]


def test_name_version_and_a_blank_copyright():
    name, number, tagline, rights = pymetrix_app.about_lines()
    assert (name, number) == ("PyMetrix", f"Version {pymetrix.__version__}") and tagline
    assert pymetrix_app.COPYRIGHT_HOLDER == "" and rights == "Copyright © 2026"  # the holder, until it is settled


def test_the_opening_screen_gives_way_to_the_window(app, monkeypatch):
    from pymetrix.ui import qt

    screen = pymetrix_app.splash(app)
    assert screen.isVisible()
    made = []
    build = qt.main_window
    monkeypatch.setattr(qt, "main_window", lambda session: made.append(build(session)) or made[-1])
    monkeypatch.setattr(qt, "_run", lambda application: 0)  # no event loop: main returns once the window is up
    with pytest.raises(SystemExit):
        qt.main([], app=app, splash=screen)
    window = made[0]
    try:
        assert not screen.isVisible()  # finished by the main window
        assert window.windowTitle().startswith("PyMetrix — untitled") and app.applicationDisplayName() == "PyMetrix"
        assert app.applicationName() == "PyMetrix" and not app.windowIcon().isNull()  # its own icon, not Qt's or Python's
        assert pymetrix_app.ICON.is_file() and pymetrix_app.ICON_WINDOWS.is_file()
        assert any(a.text().replace("&", "") == "About PyMetrix" for a in window.help_menu.actions())
        box = qt.about_dialog(window)
        from pymetrix.ui.messages import TITLES_SHOWN

        # a Mac shows no title on a message box (Qt drops it there); the name heads the text on every platform
        assert box.windowTitle() == ("About PyMetrix" if TITLES_SHOWN else "")
        assert box.text().startswith("<h2>PyMetrix</h2>") and f"Version {pymetrix.__version__}" in box.text()
    finally:
        window.state["session"].dirty = False  # nothing to ask about, so the window closes (and stops its console)
        window.close()
    assert not window.isVisible()


def test_menus_end_dialogs_with_an_ellipsis():
    """Commands and plugins write "Data/Set sample..."; every menu shows "Set sample…", the same as the app's own."""
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import MENU_TOPS, menu_entries, shown

    assert shown("Set sample...") == "Set sample…" and shown("Plot") == "Plot"
    labels = []

    def walk(entries):
        for e in entries:
            labels.append(e.label)
            walk(e.entries)

    infos = [i for i in COMMANDS.values() if i.menu]
    for top in dict.fromkeys([*MENU_TOPS, *(i.menu.partition("/")[0] for i in infos)]):
        walk(menu_entries(top, infos))
    assert labels and not [label for label in labels if label.endswith("...")]
