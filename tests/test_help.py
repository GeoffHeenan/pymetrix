"""Help: pages, links, dialog context help, course material (notes and notebooks), and the Qt and Jupyter views."""

import json
import os
from pathlib import Path

import pytest

from pymetrix import help as hp
from pymetrix.commands import FitEquation, Session
from pymetrix.testing import check_course

EXAMPLE_COURSE = Path(__file__).parents[1] / "examples" / "course"


@pytest.fixture
def course_dir(tmp_path):
    folder = tmp_path / "econ101"
    (folder / "slides").mkdir(parents=True)
    (folder / "slides" / "week3.pdf").write_bytes(b"%PDF-1.4")
    (folder / "week3.md").write_text(
        "---\ntitle: Week 3. Volatility\ncourse: ECON 101\nlinks: [garch, variances/GJR-GARCH, dialog/FitEquation]\n"
        "references:\n  - Tsay (2010), chapter 3\n---\n\nSee the [slides](slides/week3.pdf) and the [notes on ML](pymetrix:topic/maximum-likelihood).\n",
        encoding="utf-8")
    (folder / "broken.md").write_text("---\nlinks: [no-such-topic, methods/Nope]\n---\n# Broken\n[gone](missing.pdf)\n", encoding="utf-8")
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# GARCH in practice\n", "Fit a GARCH(1,1)."]},
                    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": ["fit = Equation('r ~ 1', variance=GARCH()).fit(data, method=ML())"]}],
          "metadata": {"pymetrix": {"course": "ECON 101", "links": ["garch"]}}, "nbformat": 4, "nbformat_minor": 5}
    (folder / "garch.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    yield folder
    hp.unload_course(folder)


def test_course_notes_and_notebooks_link_into_the_help(course_dir):
    course = hp.load_course(course_dir)
    by_title = {n.title: n for n in course.notes}
    assert set(by_title) == {"Week 3. Volatility", "GARCH in practice", "Broken"}
    week3, nb = by_title["Week 3. Volatility"], by_title["GARCH in practice"]
    assert week3.course == "ECON 101" and week3.references == ("Tsay (2010), chapter 3",)
    assert nb.kind == "notebook" and "```python" in nb.text
    assert (course_dir / "slides" / "week3.pdf").as_uri() in week3.text  # relative links become file links
    assert sorted(course.problems) == ["broken.md: link 'methods/Nope' matches no help page",
                                       "broken.md: link 'no-such-topic' matches no help page",
                                       "broken.md: linked file missing.pdf not found"]
    garch_topic = hp.page_for_link(hp.link("topic", "garch"))
    assert "## Course material" in garch_topic and "Week 3. Volatility" in garch_topic and "example notebook" in garch_topic
    gjr = hp.page_for_link(hp.link("variances", "pymetrix:gjr_garch"))
    assert "Week 3. Volatility" in gjr and "GARCH in practice" not in gjr
    dialog = hp.dialog_markdown(FitEquation, {"method": "ML", "variance": "GJR-GARCH"})
    assert dialog.count("Week 3. Volatility") == 2  # on the dialog's page and on GJR-GARCH's
    assert any(t == "ECON 101" or t.endswith("ECON 101") for t, _ in hp.contents())
    assert [k for k, _, _ in hp.search("volatility week")][0] == "note"  # first, ahead of any volatility topics
    page = hp.page_for_link(week3.link)
    assert "*ECON 101*" in page and "See in the help" in page and "Open the notebook" not in page
    assert "Open the notebook" in hp.page_for_link(nb.link)
    with pytest.raises(AssertionError, match="matches no help page"):
        check_course(course_dir)


def test_example_course_passes_the_check():
    course = check_course(EXAMPLE_COURSE)
    assert {n.kind for n in course.notes} == {"note", "notebook"}
    hp.unload_course(EXAMPLE_COURSE)


def test_courses_from_the_environment(course_dir, monkeypatch):
    monkeypatch.setenv("PYMETRIX_COURSES", str(course_dir))
    monkeypatch.setattr(hp, "_ENV_LOADED", False)
    hp.unload_course(course_dir)
    hp.contents()
    assert any(c.path == course_dir.resolve() for c in hp.courses())


def test_pages_and_links():
    md = hp.dialog_markdown(FitEquation, {"method": "LIML", "variance": "constant"})
    assert md.startswith("# Equation") and "**Method:** [OLS](pymetrix:methods/pymetrix:ols)" in md
    assert "## LIML" in md and "pymetrix:reference/methods/linearmodels:liml" in md
    ref = hp.page_for_link("pymetrix:reference/methods/linearmodels:liml")
    assert "```text" in ref
    home = hp.page_for_link(hp.link("home"))
    assert home.index("▸ Commands]") < home.index("▸ Estimation methods]") < home.index("▸ Background]")
    assert hp.link_of(hp.topic("garch")) == "pymetrix:topic/garch"


def test_dialog_help_pane_and_browser(course_dir):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.help_qt import course_dialog, help_window
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, Session())
    dialog.show()
    assert not dialog.help_pane.isVisible()
    dialog.toggle_help()
    pane = dialog.help_pane
    assert pane.isVisible() and pane.current() == "context" and "Equation" in pane.text.toPlainText()
    assert "Huber" not in pane.text.toPlainText()
    dialog.editors["method"].widget.setCurrentText("Robust regression (M)")  # a core method: no plugin needed
    assert "Huber" in pane.text.toPlainText()  # follows the chosen method
    pane.go("pymetrix:topic/instrumental-variables")
    dialog.editors["method"].widget.setCurrentText("OLS")
    assert pane.current() == "pymetrix:topic/instrumental-variables"  # the reader's page is kept
    pane.back()
    assert pane.current() == "context" and "Huber" not in pane.text.toPlainText()  # and refreshed on return

    hp.load_course(course_dir)
    window = help_window()
    window.search.setText("week volatility")
    first = window.tree.topLevelItem(0)
    assert first.text(0).startswith("Week 3. Volatility")
    window.tree.itemClicked.emit(first, 0)
    assert "ECON 101" in window.view.text.toPlainText()

    from PySide6.QtCore import QSettings

    settings = QSettings(str(course_dir.parent / "settings.ini"), QSettings.Format.IniFormat)
    courses = course_dialog(settings)
    assert any("ECON 101" in courses.listing.item(i).text() for i in range(courses.listing.count()))
    assert "missing.pdf" in courses.report.toPlainText()
    courses.add_folder(str(course_dir))
    assert str(course_dir.resolve()) in settings.value("help/courses", type=list)  # remembered as the user's own
    app.processEvents()


def test_jupyter_help(course_dir):
    pytest.importorskip("ipywidgets")
    from pymetrix import OLS
    from pymetrix.ui.jupyter import form, show_help

    assert "**GARCH**" in show_help("garch").data and "pymetrix:" not in show_help("garch").data
    assert show_help(OLS).data.startswith("# OLS")
    f = form(FitEquation, Session())
    f.help_button.value = True
    assert f.help_text.startswith("# Equation") and "**OLS**" in f.help_text


def test_projects_carry_course_material(course_dir, tmp_path, monkeypatch):
    import zipfile

    from pymetrix import Dataset
    from pymetrix.commands import load_project, save_project

    monkeypatch.setenv("PYMETRIX_CACHE", str(tmp_path / "cache"))
    session = Session()
    session.objects["data"] = Dataset.read(str(EXAMPLE_COURSE.parent / "sample_macro.parquet"))
    session.run(FitEquation("c ~ y", name="fit"))
    session.attach_course(course_dir)
    assert session.dirty and session.courses == [str(course_dir.resolve())]
    (course_dir / ".ipynb_checkpoints").mkdir()
    (course_dir / ".ipynb_checkpoints" / "garch-checkpoint.ipynb").write_text("{}", encoding="utf-8")
    path = save_project(session, tmp_path / "assignment.pymx")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert json.loads(z.read("manifest.json"))["courses"] == [{"name": "ECON 101", "dir": "course/0"}]
    assert {"course/0/week3.md", "course/0/garch.ipynb", "course/0/slides/week3.pdf"} <= set(names)
    assert not any("checkpoint" in n for n in names)

    hp.unload_course(course_dir)  # a student's machine: the course is only in the project
    opened = load_project(path)
    unpacked = Path(opened.courses[0])
    assert unpacked.is_relative_to(tmp_path / "cache") and (unpacked / "slides" / "week3.pdf").exists()
    titles = {n.title for c in hp.courses() if c.path == unpacked for n in c.notes}
    assert titles == {"Week 3. Volatility", "GARCH in practice", "Broken"}
    assert "Week 3. Volatility" in hp.page_for_link(hp.link("topic", "garch"))
    assert not opened.dirty and opened["fit"].params.names == session["fit"].params.names
    assert Path(load_project(path).courses[0]) == unpacked  # unchanged content reuses the unpacked folder
    hp.unload_course(unpacked)


def test_unsafe_course_paths_are_refused(tmp_path, monkeypatch):
    import zipfile

    from pymetrix.commands import load_project

    monkeypatch.setenv("PYMETRIX_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "bad.pymx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps({"format_version": 1, "courses": [{"name": "x", "dir": "course/0"}]}))
        z.writestr("commands.json", "[]")
        z.writestr("course/0/../../../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        load_project(path)
    assert not (tmp_path / "evil.txt").exists()


def test_links_outside_the_course_folder_are_reported(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    (folder / "n.md").write_text("---\nlinks: [garch]\n---\n# N\n[data](../data.csv)\n", encoding="utf-8")
    course = hp.load_course(folder)
    assert course.problems == ("n.md: linked file ../data.csv is outside the course folder, so projects will not include it",)
    hp.unload_course(folder)


def test_course_dialog_includes_folders_in_the_project(course_dir, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QSettings

    from pymetrix.ui.help_qt import course_dialog, forget_project_courses

    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    session = Session()
    dialog = course_dialog(settings, session=session)
    dialog.add_folder(str(course_dir))
    dialog.listing.setCurrentRow(next(i for i in range(dialog.listing.count()) if str(course_dir.resolve()) in dialog.listing.item(i).text()))
    dialog.include(True)
    assert session.courses == [str(course_dir.resolve())] and session.dirty
    assert "in project, mine" in dialog.listing.currentItem().text()
    settings.setValue("help/courses", [])  # not the user's own: switching sessions takes it out of the help
    forget_project_courses(session, settings)
    assert all(c.path != course_dir.resolve() for c in hp.courses())


def test_the_defaults_page_compares_with_other_packages():
    from pymetrix.help import _TOPICS

    text = _TOPICS["defaults"].text
    for phrase in ("HC1", "Schwarz", "dfk", "14,400", "X-13ARIMA-SEATS"):
        assert phrase in text


def test_the_contents_list_one_page_per_line_under_headings_that_fold():
    home = hp.page_for_link(hp.link("home"))  # every section closed: its heading, a link that opens it
    assert "## [▸ Tests](pymetrix:home/tests) (" in home and "- [" not in home
    tests = hp.page_for_link("pymetrix:home/tests")
    section = tests.split("## [▾ Tests](pymetrix:home/none)")[1].split("\n## ")[0]
    lines = [line for line in section.strip().splitlines()[1:] if line.strip()]  # after the heading's count
    assert len(lines) > 10 and all(line.startswith("- [") and line.count("](pymetrix:") == 1 for line in lines)  # one per line
    assert "## [▸ Commands](pymetrix:home/commands,tests)" in tests  # opening another keeps this one open
    assert hp.page_for_link("pymetrix:home/nothing,tests") == tests  # an unknown section is ignored
    opened = hp.home_markdown("all")
    assert "▸" not in opened.split("Close all")[1] and opened.count("- [") > 100  # for readers that cannot click
    folded = hp.home_markdown(details=True)
    assert folded.count("<details><summary><b>") == len(hp.contents()) and "Open all" not in folded


def test_notebooks_fold_the_contents():
    pytest.importorskip("IPython")
    from pymetrix.ui.jupyter import show_help

    shown = show_help().data  # a notebook folds <details> itself; its links are names
    assert "<details><summary><b>Tests</b>" in shown and "- **Augmented Dickey-Fuller test (adf)**" in shown and "pymetrix:" not in shown


def test_the_help_window_folds_the_contents():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.help_qt import help_window

    window = help_window()
    view = window.view
    view.go("pymetrix:home/tests")  # a heading clicked
    assert view.current() == "pymetrix:home/tests" and "Augmented Dickey-Fuller" in view.text.toPlainText()
    view.back()
    assert "Augmented Dickey-Fuller" not in view.text.toPlainText()  # Back folds it again
    window.close()
