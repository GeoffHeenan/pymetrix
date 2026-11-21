"""The assistant's tests: skipped where the plugin is not installed (the core-only run)."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("pymetrix_assistant") is not None else ["test_*.py"]


def pytest_configure(config):
    """The app's settings (Show the code, the local model's address) go to a folder of the test run, not the user's
    own, as pymetrix's tests do: QSettings' default format is set to INI files in that folder."""
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        return
    import tempfile

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tempfile.mkdtemp(prefix="pymetrix-settings-"))
