"""Plugin management: projects open when a plugin is missing or disabled (its steps kept, not run, and saved again
unchanged); plugins are enabled and disabled for the app and scripts alike; new plugins are described from their
metadata without running them, and wait for the app's answer."""

import json
import zipfile

import pytest

from pymetrix import Dataset, plugins
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project
from pymetrix.errors import PluginUnavailable


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugins, "_STATE", None)
    monkeypatch.setattr(plugins, "_SKIPPED", set())
    yield
    plugins._STATE = None


def _project_with_foreign_steps(tmp_path):
    """A saved project, edited to hold a step and a recorded call from a plugin that is not installed, and a step
    that uses what the first made."""
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(GenerateSeries("z", "y * 2"))
    path = save_project(s, tmp_path / "p.pymx")
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    commands = json.loads(files["commands.json"])
    manifest = json.loads(files["manifest.json"])
    foreign = {"id": "fancy:smooth", "version": 2, "command": "Smooth", "fields": {"series": "y", "name": "smooth"},
               "seconds": 0.1, "outputs": ["smooth"]}
    call = {"id": "pymetrix:call", "version": 1, "command": "Call",
            "fields": {"function": "fancy:score", "version": 1, "target": None, "arguments": {}, "output": "score", "source": ""},
            "seconds": 0.0, "outputs": ["score"]}
    after = dict(commands[0], fields={**commands[0]["fields"], "name": "w", "expression": "smooth + 1"}, outputs=["data"])
    commands += [foreign, call]
    manifest["plugins"] = {**manifest.get("plugins", {}), "pymetrix-fancy": "1.4"}
    files["commands.json"] = json.dumps(commands).encode()
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return path, foreign, after


def test_a_project_opens_with_a_missing_plugins_steps_kept(tmp_path):
    path, foreign, _ = _project_with_foreign_steps(tmp_path)
    s = load_project(path)
    kinds = [type(r.command).__name__ for r in s.history]
    assert kinds == ["Call", "MissingStep", "MissingStep"] or kinds[-2:] == ["MissingStep", "MissingStep"]
    smooth = s.history[-2]
    assert smooth.skipped and "pymetrix-fancy" in smooth.error and "pip install pymetrix-fancy" in smooth.error
    assert smooth.code.startswith("# not run: fancy:smooth needs pymetrix-fancy 1.4")
    assert "z" in s.objects["data"].names  # the steps that can run did
    assert "pymetrix-fancy 1.4" in s.missing_plugins
    again = save_project(s, tmp_path / "again.pymx")  # the steps go back unchanged, for whoever has the plugin
    with zipfile.ZipFile(again) as z:
        saved = json.loads(z.read("commands.json"))
        manifest = json.loads(z.read("manifest.json"))
    kept = next(e for e in saved if e.get("id") == "fancy:smooth")
    assert kept["fields"] == foreign["fields"] and kept["version"] == 2 and kept["outputs"] == ["smooth"]
    assert any(e.get("fields", {}).get("function") == "fancy:score" for e in saved)
    assert manifest["plugins"]["pymetrix-fancy"] == "1.4"
    assert "# not run: fancy:smooth" in s.script()


def test_a_step_using_a_disabled_plugins_method_is_kept(tmp_path):
    pytest.importorskip("linearmodels")
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y", method="LIML", endog="y", instruments="i", name="liml"))
    s.run(GenerateSeries("z", "y"))
    path = save_project(s, tmp_path / "liml.pymx")
    plugins.disable("pymetrix-linearmodels")
    try:
        opened = load_project(path)
        liml = next(r for r in opened.history if "liml" in r.outputs)
        assert liml.skipped and "disabled" in liml.error and "liml" not in opened.objects
        assert "z" in opened.objects["data"].names
        with pytest.raises(PluginUnavailable, match="disabled"):
            plugins.registry("methods").get("LIML")
        assert "LIML" not in plugins.registry("methods").labels()  # out of the dialogs' lists
    finally:
        plugins.enable("pymetrix-linearmodels")
    assert "LIML" in plugins.registry("methods").labels()


def test_the_state_is_shared_by_the_app_and_scripts(tmp_path):
    plugins.disable("pymetrix-sklearn")
    plugins._STATE = None  # another process (a script) reads the same file
    assert plugins.is_disabled("pymetrix-sklearn") and plugins.is_disabled("pymetrix_sklearn")
    table = plugins.plugin_table()
    assert dict((r[0], r[-1]) for r in table.rows)["pymetrix-sklearn"] == "disabled"
    plugins.enable("pymetrix-sklearn", "0.1")
    assert not plugins.is_disabled("pymetrix-sklearn")
    with pytest.raises(Exception, match="cannot be disabled"):
        plugins.disable("pymetrix")


class _Dist:
    def __init__(self, name, version, meta):
        self.name, self.version, self.metadata = name, version, meta


class _EP:
    def __init__(self, name, value, dist):
        self.name, self.value, self.dist = name, value, dist
        self.loaded = False

    def load(self):
        self.loaded = True
        raise AssertionError("the plugin's code ran")


def test_new_plugins_are_described_without_running_them(monkeypatch):
    dist = _Dist("pymetrix-shiny", "0.3", {"Author": "Ann Example", "License": "MIT", "Summary": "shiny things"})
    eps = {"pymetrix.methods": [_EP("glow", "pymetrix_shiny:Glow", dist)], "pymetrix.commands": [_EP("Glow", "pymetrix_shiny.ui:Glow", dist)]}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: eps.get(group, []))
    found = plugins.pending()
    assert [p.name for p in found] == ["pymetrix-shiny"]
    p = found[0]
    assert (p.version, p.author, p.license) == ("0.3", "Ann Example", "MIT") and set(p.groups) == {"methods", "commands"}
    plugins.set_mode("app")
    try:
        assert not plugins._admit(eps["pymetrix.methods"][0])  # the app waits for the answer: nothing is imported
        plugins.skip_for_now("pymetrix-shiny")
        assert plugins.pending() == []
        plugins.disable("pymetrix-shiny")
        assert not plugins._admit(eps["pymetrix.methods"][0])
        plugins.enable("pymetrix-shiny", "0.3")
        assert plugins._admit(eps["pymetrix.methods"][0])
    finally:
        plugins.set_mode("script")
    assert not any(ep.loaded for group in eps.values() for ep in group)


def test_scripts_load_a_new_plugin_with_a_warning(monkeypatch):
    dist = _Dist("pymetrix-brand-new", "1.0", {})
    ep = _EP("x", "pymetrix_brand_new:X", dist)
    with pytest.warns(UserWarning, match="new: loaded, but not yet enabled"):
        assert plugins._admit(ep)


@pytest.fixture
def app():
    import os

    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_manager_enables_and_disables(app):
    from pymetrix.ui.plugins_qt import plugin_manager

    dialog = plugin_manager()
    names = [r["name"] for r in dialog.rows()]
    assert "pymetrix" in names and "pymetrix-linearmodels" in names
    row = names.index("pymetrix-linearmodels")
    dialog.table.selectRow(row)
    assert dialog.disable_button.isEnabled() and "methods" in dialog.details.toPlainText()
    dialog.disable_button.click()
    assert plugins.is_disabled("pymetrix-linearmodels") and dialog.rows()[row]["status"] == "disabled"
    dialog.table.selectRow(row)
    dialog.enable_button.click()
    assert not plugins.is_disabled("pymetrix-linearmodels")
    dialog.table.selectRow(names.index("pymetrix"))
    assert not dialog.disable_button.isEnabled()  # pymetrix itself stays


def test_the_app_asks_about_new_plugins_from_metadata(app, monkeypatch):
    from pymetrix.ui.plugins_qt import ask_new_plugins, missing_plugins_note

    dist = _Dist("pymetrix-shiny", "0.3", {"Author": "Ann Example", "License": "MIT"})
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [_EP("glow", "pymetrix_shiny:Glow", dist)] if group == "pymetrix.methods" else [])
    asked = []
    assert ask_new_plugins(answer=lambda p: asked.append((p.name, p.groups)) or "enable") == ["pymetrix-shiny"]
    assert asked == [("pymetrix-shiny", ("methods",))] and plugins.is_known("pymetrix-shiny")
    assert "pymetrix-fancy 1.4: needs it" in missing_plugins_note({"pymetrix-fancy 1.4": "needs it"})
