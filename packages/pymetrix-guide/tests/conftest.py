"""The guide's tests: skipped where the plugin is not installed (the core-only run)."""

import importlib.util

import pytest

collect_ignore_glob = [] if importlib.util.find_spec("pymetrix_guide") is not None else ["test_*.py"]


def _look(name):
    """The app as on another machine, put back afterwards: "default"; "wide" (the font a third bigger: text as wide
    as with Windows' Segoe UI or CI's DejaVu Sans); "windows" (Qt's Windows style, push buttons as wide as Windows 11
    draws them). Layouts that fit with this machine's font and style but not Windows' failed only on Windows CI."""
    from contextlib import contextmanager

    @contextmanager
    def applied():
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        font, style, palette, sheet = app.font(), app.style().name(), app.palette(), app.styleSheet()
        try:
            if name == "wide":
                wide = app.font()
                wide.setPointSizeF(font.pointSizeF() * 1.35 if font.pointSizeF() > 0 else 13)
                app.setFont(wide)
            if name == "windows":
                app.setStyle("Windows")
                app.setStyleSheet("QPushButton { min-width: 96px; }")
            yield name
        finally:
            if name == "windows":
                app.setStyleSheet(sheet)
                app.setStyle(style)
                app.setPalette(palette)
            app.setFont(font)

    return applied()


@pytest.fixture(params=("default", "wide", "windows"))
def look(request):
    pytest.importorskip("PySide6.QtWidgets")
    with _look(request.param) as name:
        yield name
