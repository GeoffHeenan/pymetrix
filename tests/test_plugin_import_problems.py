"""An installed plugin whose code cannot be imported: the message names the distribution, whether it is an editable
install and from where (and whether that folder is still there), and gives the pip command for the Python in use; a
missing dependency of the plugin is told apart from the plugin's own code being missing."""

import json
import os
import sys
from importlib.metadata import distributions

import pytest

from pymetrix import plugins


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugins, "_STATE", None)
    monkeypatch.setattr(plugins, "_PROBLEMS", {})
    monkeypatch.setattr(plugins, "_DETAILS", {})
    monkeypatch.setattr(plugins, "_PLUGINS", dict(plugins._PLUGINS))
    monkeypatch.setattr(plugins, "_loaded", set())
    yield
    plugins._STATE = None


def _fake_dist(site, name, module, *, direct_url=None, requires=(), code=None):
    """A distribution installed in ``site`` (on sys.path) whose entry points name ``module``; ``code`` is the module's
    source, or None for a module that is not there."""
    dist_info = site / f"{name.replace('-', '_')}-0.1.0.dev0.dist-info"
    dist_info.mkdir(parents=True)
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", "Version: 0.1.0.dev0", *(f"Requires-Dist: {r}" for r in requires)]
    (dist_info / "METADATA").write_text("\n".join(meta) + "\n", encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(f"[pymetrix.methods]\nghost = {module}:register\n\n"
                                                f"[pymetrix.plugins]\nghost = {module}:PLUGIN\n", encoding="utf-8")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    if code is not None:
        (site / f"{module}.py").write_text(code, encoding="utf-8")
    return next(d for d in distributions(path=[str(site)]) if d.metadata["Name"] == name)


def _load(monkeypatch, site, dist):
    monkeypatch.syspath_prepend(str(site))
    eps = list(dist.entry_points)
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [ep for ep in eps if ep.group == group])
    plugins.enable(dist.metadata["Name"])
    with pytest.warns(UserWarning):
        plugins.load("methods")
    return plugins.load_problems()


def _python():
    return f'"{sys.executable}"' if " " in sys.executable else sys.executable


def test_an_editable_install_whose_folder_is_gone(monkeypatch, tmp_path):
    source = tmp_path / "clone" / "packages" / "pymetrix-ghost"  # never created: moved or deleted
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost",
                      direct_url={"url": source.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(plugins, "FIRST_PARTY", (*plugins.FIRST_PARTY, "pymetrix-ghost"))
    problems = _load(monkeypatch, tmp_path / "site", dist)
    assert list(problems) == ["pymetrix-ghost"]  # one line per plugin, whichever group failed first
    short = problems["pymetrix-ghost"]
    assert "\n" not in short and "editable" in short and "moved or deleted" in short and "No module named" not in short
    detail = plugins.problem_detail("pymetrix-ghost")
    text = detail["detail"]
    assert text.startswith("pymetrix-ghost 0.1.0.dev0 is installed (editable, from ")
    assert str(source) in text and "moved or deleted" in text and not source.exists()
    assert detail["command"] == f"{_python()} -m pip install -e <clone>/packages/pymetrix-ghost"
    assert f"{_python()} -m pip uninstall pymetrix-ghost" in text and f"Python in use: {sys.executable}" in text


def test_an_editable_install_whose_folder_lacks_the_code(monkeypatch, tmp_path):
    source = tmp_path / "clone" / "packages" / "pymetrix-ghost"
    source.mkdir(parents=True)  # there, but checked out at a branch without the package
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost2",
                      direct_url={"url": source.as_uri(), "dir_info": {"editable": True}})
    _load(monkeypatch, tmp_path / "site", dist)
    detail = plugins.problem_detail("pymetrix-ghost")
    assert "the folder exists but has no pymetrix_ghost2" in detail["detail"]
    assert detail["command"] == f"{_python()} -m pip install -e {plugins._shell_quote(str(source))}"


def test_a_regular_install_whose_files_are_missing(monkeypatch, tmp_path):
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost3",
                      direct_url={"url": "https://example.org/x.whl", "archive_info": {}})
    problems = _load(monkeypatch, tmp_path / "site", dist)
    assert "files are missing" in problems["pymetrix-ghost"]
    detail = plugins.problem_detail("pymetrix-ghost")
    assert "editable" not in detail["detail"] and "its files are missing; reinstall with" in detail["detail"]
    assert detail["command"] == f"{_python()} -m pip install --force-reinstall pymetrix-ghost"


def test_a_missing_dependency_is_not_the_plugin_missing(monkeypatch, tmp_path):
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost4", requires=["no-such-dep-xq>=1.2", "numpy"],
                      code="import no_such_dep_xq\n", direct_url={"url": (tmp_path / "x").as_uri(), "dir_info": {"editable": True}})
    problems = _load(monkeypatch, tmp_path / "site", dist)
    assert problems["pymetrix-ghost"] == "could not be imported: needs no_such_dep_xq, which isn't installed"
    detail = plugins.problem_detail("pymetrix-ghost")
    assert detail["detail"].startswith("the ghost plugin needs no_such_dep_xq, which isn't installed: ")
    assert detail["command"] == f"{_python()} -m pip install no-such-dep-xq"  # the distribution, from Requires-Dist


def test_an_unknown_dependency_falls_back_to_the_module_name(monkeypatch, tmp_path):
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost5", code="import nothing_named_this_zz.sub\n")
    _load(monkeypatch, tmp_path / "site", dist)
    assert plugins.problem_detail("pymetrix-ghost")["command"] == f"{_python()} -m pip install nothing_named_this_zz"


def test_other_errors_keep_their_message(tmp_path):
    class EP:
        value, name, dist = "pymetrix_x:f", "x", None

    assert plugins.import_problem(EP(), ValueError("bad")) is None
    assert plugins.import_problem(EP(), ModuleNotFoundError("no name given")) is None


def test_windows_file_urls_become_paths():
    if os.name == "nt":
        assert plugins._url_path("file:///C:/Users/a%20b/pkg") == "C:\\Users\\a b\\pkg"
    else:
        assert plugins._url_path("file:///home/a%20b/pkg") == "/home/a b/pkg"


def test_help_plugins_and_check_plugin_explain_it(monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.plugins_qt import check_report, plugin_manager

    source = tmp_path / "gone"
    dist = _fake_dist(tmp_path / "site", "pymetrix-ghost", "pymetrix_ghost6",
                      direct_url={"url": source.as_uri(), "dir_info": {"editable": True}})
    _load(monkeypatch, tmp_path / "site", dist)
    monkeypatch.setattr(plugins, "load_all", lambda: None)  # only the fake plugin's entry points
    assert "moved or deleted" in check_report("pymetrix-ghost") and "pip install -e" in check_report("pymetrix-ghost")
    dialog = plugin_manager()
    names = [r["name"] for r in dialog.rows()]
    dialog.table.selectRow(names.index("pymetrix-ghost"))
    shown = dialog.details.toPlainText()
    assert "Why it could not load" in shown and str(source) in shown and "-m pip install -e" in shown
    assert dialog.copy_button.text() == "Copy command" and dialog.copy_button.isEnabled()
    dialog.copy_button.click()
    assert QtWidgets.QApplication.clipboard().text() == plugins.problem_detail("pymetrix-ghost")["command"]
    dialog.close()
