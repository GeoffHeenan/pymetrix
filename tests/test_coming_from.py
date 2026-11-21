"""The "Coming from EViews" and "Coming from MATLAB/IRIS and Dynare" help topics and the palette's synonyms for
familiar commands: every menu path and every pymetrix name they cite exists, every synonym finds its command or
topic, and synonyms never push pymetrix's own matches down."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import importlib
import os
import re
from pathlib import Path

import pytest

import pymetrix
from pymetrix import help as hp
from pymetrix.coming_from import TRANSLATIONS, palette_synonyms, translation_table
from pymetrix.ui.palette import PaletteEntry, rank, synonym_entries

TOPICS = ("coming-from-eviews", "coming-from-iris")


def _receivers() -> dict[str, list[type]]:
    """The objects the tables' Python column calls methods on, by the names the topics give them."""
    from pymetrix.commands import Session
    from pymetrix.models.linear import LinearSolution
    from pymetrix.models.templates import Template

    return {"data": [pymetrix.Dataset], "fit": [pymetrix.FittedEquation], "var": [pymetrix.FittedVAR],
            "model": [pymetrix.Model], "nk": [Template, pymetrix.DynareModel], "sol": [LinearSolution],
            "est": [pymetrix.FittedModel], "mod": [pymetrix.DynareModel], "rep": [pymetrix.Report],
            "session": [Session]}


# first-party plugins the tables name by module (``pymetrix_seasonal.X13()``): checked against the installed plugin,
# or, where it is not installed (the no-extras run), against the names its source in packages/ exports
PLUGINS = {"pymetrix_seasonal": Path(__file__).resolve().parents[1] / "packages" / "pymetrix-seasonal" / "src"
           / "pymetrix_seasonal" / "__init__.py"}


def _plugin_has(module: str, name: str) -> bool:
    try:
        return hasattr(importlib.import_module(module), name)
    except ImportError:
        tree = ast.parse(PLUGINS[module].read_text(encoding="utf-8"))
        exported = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                        and any(getattr(target, "id", "") == "__all__" for target in node.targets))
        return name in exported


def _has(cls: type, name: str) -> bool:
    fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    return hasattr(cls, name) or name in fields


def _code_spans(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def cited_problems(code: str, receivers: dict[str, list[type]]) -> list[str]:
    """What in a code span does not exist: a called name that is neither pymetrix's nor Python's, an attribute of a
    pymetrix name that it lacks, or a method of a known receiver (data, fit, sol...) that its class lacks."""
    bare = re.sub(r'"[^"]*"', '""', code)  # strings hold formulas and paths, not names
    problems = []
    for name in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", bare):
        if not (hasattr(pymetrix, name) or hasattr(builtins, name) or name in receivers):
            problems.append(f"{name}() is not pymetrix.{name}")
    for owner, attr in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\.([A-Za-z_]\w*)", bare):
        if owner in receivers:
            if not any(_has(cls, attr) for cls in receivers[owner]):
                problems.append(f"{owner}.{attr}: no such attribute of {[c.__name__ for c in receivers[owner]]}")
        elif owner in PLUGINS:
            if not _plugin_has(owner, attr):
                problems.append(f"{owner}.{attr} does not exist")
        elif hasattr(pymetrix, owner) and not _has(getattr(pymetrix, owner), attr):
            problems.append(f"pymetrix.{owner}.{attr} does not exist")
    return problems


def test_the_checker_finds_a_wrong_name():
    receivers = _receivers()
    assert cited_problems('data.with_series(dy="d(y)") and Equation("c ~ y").fit(data, method=IV(endog=["y"]))', receivers) == []
    assert cited_problems('genr(x) and data.genr("x") and Dataset.readfile("x")', receivers) == [
        "genr() is not pymetrix.genr", "data.genr: no such attribute of ['Dataset']", "pymetrix.Dataset.readfile does not exist"]
    assert cited_problems('data.seasonal_adjust("c", pymetrix_seasonal.X13())', receivers) == []
    assert cited_problems("pymetrix_seasonal.X14()", receivers) == ["pymetrix_seasonal.X14 does not exist"]


def test_the_seasonal_rows_name_the_plugin_and_work_without_it(monkeypatch):
    """X-13 and TRAMO-SEATS are the pymetrix-seasonal plugin's: their rows say so and call it by module, the dialog they
    point to is the core's, and the checker reads the plugin's exports from its source where it is not installed."""
    rows = [t for t in TRANSLATIONS if t.python and ("X13" in t.python or "TramoSeats" in t.python)]
    assert {t.package for t in rows} == {"EViews", "IRIS"}
    for t in rows:
        assert "pymetrix_seasonal." in t.python and "pymetrix-seasonal plugin" in t.does, t
        assert t.menu == "Data → Transform → Seasonal adjustment…"
    assert "pymetrix-seasonal" in hp.topic("coming-from-eviews").text
    from pymetrix.commands import SeasonalAdjust

    assert SeasonalAdjust.__module__.startswith("pymetrix.")  # the menu item is the core's, plugin or not
    real = importlib.import_module

    def without(name, *args, **kwargs):
        if name == "pymetrix_seasonal":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without)
    assert _plugin_has("pymetrix_seasonal", "X13") and _plugin_has("pymetrix_seasonal", "TramoSeats")
    assert not _plugin_has("pymetrix_seasonal", "X14")


def test_every_pymetrix_name_the_topics_cite_exists():
    receivers = _receivers()
    problems = {}
    for key in TOPICS:
        for code in _code_spans(hp.topic(key).text):
            if found := cited_problems(code, receivers):
                problems[code] = found
    assert problems == {}
    for t in TRANSLATIONS:  # and each row's call is in its topic
        if t.python:
            topic = "coming-from-eviews" if t.package == "EViews" else "coming-from-iris"
            assert t.python.replace("|", "\\|") in hp.topic(topic).text, t


def test_the_topics_hold_their_tables_and_link_the_defaults():
    eviews, iris = (hp.topic(k) for k in TOPICS)
    assert translation_table("EViews") in eviews.text and translation_table("IRIS", "Dynare") in iris.text
    for t in (eviews, iris):
        assert "pymetrix:topic/defaults" in t.text and "defaults" in t.see_also
        for heading in ("## The objects", "## Commands and menus", "## Defaults that differ", "## Different by design"):
            assert heading in t.text
    for word in ("read_eviews", "read_eviews_xml", "read_eviews_sspace", "Write as", "FittedEquation", "smpl", "spool"):
        assert word in eviews.text
    for word in ("read_iris", "read_mod", "sstate", "stoch_simul", "estimated_params", "Kalman", "databank"):
        assert word in iris.text
    found = [title for _, title, _ in hp.search("genr")]
    assert "Coming from EViews" in found  # the help's own search finds the familiar names too
    home = hp.home_markdown()
    assert "pymetrix:topic/coming-from-eviews" in home and "pymetrix:topic/coming-from-iris" in home


def test_every_synonym_has_a_menu_path_or_a_topic():
    synonyms = palette_synonyms()
    words = {w for w, *_ in synonyms}
    for word in ("genr", "smpl", "ls", "wfopen", "freeze", "spool", "sstate", "dbload", "stoch_simul", "estimated_params"):
        assert word in words, word
    for word, package, menu, topic in synonyms:
        assert package in ("EViews", "IRIS", "Dynare") and (menu or hp.topic(topic))
        assert " " not in word  # one word, as typed


def _fake(log: list[str]) -> list[PaletteEntry]:
    titles = ["Data → Generate series…", "Data → Plot series…", "Estimate → Equation…",
              "Finance → Fixed income → Dynamic Nelson-Siegel…", "Model → Model editor…", "Model → Import → IRIS file…"]
    out = [PaletteEntry(t, "Menu", lambda t=t: log.append(t)) for t in titles]
    out.append(PaletteEntry("Solving models", "Help", lambda: log.append("help"), "Background"))
    return out


def test_synonyms_rank_after_pymetrix_names():
    log: list[str] = []
    entries = _fake(log)
    synonyms = [("genr", "EViews", "Data → Generate series…", "coming-from-eviews"),
                ("series", "EViews", "Data → Generate series…", "coming-from-eviews"),
                ("ls", "EViews", "Estimate → Equation…", "coming-from-eviews"),
                ("model", "IRIS", "Model → Import → IRIS file…", "coming-from-iris"),
                ("model", "EViews", "Model → Model editor…", "coming-from-eviews"),
                ("acf", "IRIS", "", "coming-from-iris"),
                ("smpl", "EViews", "Data → Sample → Set sample…", "coming-from-eviews")]  # not in these menus
    extra = synonym_entries(entries, synonyms, lambda key: log.append(key))
    titles = [e.title for e in extra]
    assert titles == ["Generate series… (EViews: genr)", "Generate series… (EViews: series)", "Equation… (EViews: ls)",
                      "IRIS file… (IRIS: model)", "Coming from MATLAB/IRIS and Dynare (IRIS: acf)"]  # model editor: its name
    everything = entries + extra
    assert rank(everything, "genr")[0].title == "Generate series… (EViews: genr)"
    assert rank(everything, "gen")[0].title == "Data → Generate series…"  # pymetrix's own name first
    by_series = [e.title for e in rank(everything, "series")]
    assert by_series[:2] == ["Data → Plot series…", "Data → Generate series…"] or \
        by_series[:2] == ["Data → Generate series…", "Data → Plot series…"]
    assert by_series[2] == "Generate series… (EViews: series)"
    by_ls = [e.title for e in rank(everything, "ls")]
    assert by_ls[0] == "Equation… (EViews: ls)" and "Finance → Fixed income → Dynamic Nelson-Siegel…" in by_ls  # ahead of letters inside a word
    by_model = [e.title for e in rank(everything, "model")]
    assert by_model.index("IRIS file… (IRIS: model)") > max(by_model.index(t) for t in ("Model → Model editor…", "Solving models"))
    assert rank(everything, "a") == [] or all(not e.synonym for e in rank(everything, "a"))  # one letter finds no synonym
    assert all(not e.synonym for e in rank(everything, ""))  # nothing typed: the menus as they are
    rank(everything, "genr")[0].run()
    rank(everything, "acf")[0].run()
    assert log == ["Data → Generate series…", "coming-from-iris"]


@pytest.fixture(scope="module")
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    w = main_window(Session())
    yield w
    w.state["session"].dirty = False
    w.close()


def _menu_paths(window) -> set[str]:
    """Every item of the menu bar, by its path as the palette shows it, disabled and hidden ones too."""
    out: set[str] = set()

    def walk(menu, path):
        for action in menu.actions():
            text = action.text().replace("&", "")
            if action.menu() is not None:
                walk(action.menu(), f"{path} → {text}")
            elif text and not action.isSeparator():
                out.add(f"{path} → {text}")

    for top in window.menuBar().actions():
        if top.menu() is not None:
            walk(top.menu(), top.text().replace("&", ""))
    return out


def test_every_menu_path_in_the_tables_is_a_menu_item(window):
    paths = _menu_paths(window)
    missing = sorted({t.menu for t in TRANSLATIONS if t.menu and t.menu not in paths})
    assert missing == []
    from test_docs_menu_paths import TOPS, bad_paths, menu_tree  # and the paths written in the topics' prose

    tree = menu_tree(window)
    assert {key: bad_paths(hp.topic(key).text, tree, TOPS) for key in TOPICS} == {key: [] for key in TOPICS}


def test_typing_a_familiar_name_finds_the_command(window):
    entries = window.palette_entries()
    normal = [e for e in entries if not e.synonym]
    for word, package, menu, topic in palette_synonyms():
        found = rank(entries, word)
        if menu:
            assert any(e.title == menu or (e.synonym == word.lower() and e.detail == menu) for e in found), (word, menu)
        else:
            assert any(e.synonym == word.lower() and e.kind == "Help" for e in found), (word, topic)
        # whatever pymetrix's own names match comes first
        first_synonym = next((k for k, e in enumerate(found) if e.synonym), len(found))
        named = [e for e in rank(normal, word) if all(any(w.startswith(t) for w in re.split(r"[\s→/()_.,:-]+",
                 e.title.lower().rsplit("→", 1)[-1])) for t in word.lower().split())]
        assert all(found.index(e) < first_synonym for e in named if e in found), word
    shown = {e.title for e in rank(entries, "genr")}
    assert "Generate series… (EViews: genr)" in shown
    assert rank(entries, "var")[0].title == "Estimate → VARs and local projections → VAR…"  # unchanged
    assert rank(entries, "seas adj")[0].title == "Data → Transform → Seasonal adjustment…"


def test_the_help_menu_opens_the_topics(window):
    from PySide6.QtWidgets import QWidget

    actions = {a.text().replace("&", ""): a for a in window.help_menu.actions()}
    for label, key in (("Coming from EViews", "coming-from-eviews"), ("Coming from MATLAB/IRIS", "coming-from-iris")):
        actions[label].trigger()
        helps = [w for w in window.findChildren(QWidget) if w.windowTitle() == "PyMetrix help" and hasattr(w, "view")
                 and w.isVisible()]
        assert helps and helps[0].view.current() == hp.link("topic", key)
        helps[0].close()
