"""The Exercise panel and Help → Exercises… (offscreen): the steps render with their hints and code, Check shows each
check green or red with its message, the data load once the user agrees (as recorded steps), the dialog lists packs
from folders and starts one, and the main window docks the panel. Nothing here depends on fonts or the screen's size."""

import os
import shutil
from pathlib import Path

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from pymetrix import exercises as ex  # noqa: E402
from pymetrix.commands import Session  # noqa: E402
from pymetrix.extension import AppInterface  # noqa: E402
from pymetrix.ui import exercises_qt  # noqa: E402

PACK = Path(__file__).parent / "fixtures" / "exercise-pack"


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pane(session, answer=True):
    messages, asked = [], []

    def run(cmd):
        result = session.run(cmd)
        interface._changed()
        return result

    opened = []
    interface = AppInterface(lambda: session, lambda: None, lambda cls, initial=None: opened.append((cls, initial)), run,
                             messages.append)

    def confirm(question):
        asked.append(question)
        return answer

    pane = exercises_qt.create_pane(interface, confirm)
    pane.messages, pane.asked, pane.opened = messages, asked, opened
    return pane


def test_the_pane_shows_the_steps_hints_and_code(app):
    exercise = ex.load_pack(PACK).exercise(1)
    pane = _pane(Session(), answer=False)
    pane.start(exercise)
    assert [b.step.number for b in pane.boxes] == [1, 2]
    first = pane.boxes[0]
    assert "Step 1: Fit the wage equation" in first.findChild(QtWidgets.QLabel, "exercise-step-title").text()
    assert "Estimate → Equation…" in first.text.text()
    assert first.text.textFormat() == first.text.textFormat().MarkdownText
    assert not first.hint_label.isVisibleTo(pane) and first.hint_button.isEnabled()
    first.hint_button.click()
    assert first.hint_label.isVisibleTo(pane) and "log of the hourly wage" in first.hint_label.text()
    assert "experience squared" not in first.hint_label.text() and first.hint_button.text() == "Another hint"
    first.hint_button.click()
    assert "experience squared" in first.hint_label.text() and not first.hint_button.isEnabled()
    assert not first.code_row.isVisibleTo(pane)
    first.code_toggle.click()
    assert first.code_row.isVisibleTo(pane) and first.code_toggle.text() == "Hide the code"
    assert first.code.toPlainText() == exercise.steps[0].code
    assert first.code.fontMetrics().lineSpacing() < first.code.height()  # room for its line, whatever the font
    first.code_toggle.click()
    assert not first.code_row.isVisibleTo(pane)
    titles = [w.text() for w in pane.findChildren(QtWidgets.QLabel) if w.objectName() in ("exercise-title", "exercise-closing")]
    assert any("A wage equation" in t for t in titles) and any("What next" in t for t in titles)
    nile = ex.load_pack(PACK).exercise(2)
    pane.start(nile, load=False)
    assert len(pane.boxes) == 2 and not pane.boxes[1].hint_button.isEnabled()  # step 2 has no hint


def test_check_shows_each_result_green_or_red_with_its_message(app):
    session = Session()
    pane = _pane(session)
    exercise = ex.load_pack(PACK).exercise(1)
    pane.start(exercise)
    assert len(pane.asked) == 1 and "mroz" in pane.asked[0]
    assert "mroz" in session.objects and session.history[-1].ok  # loaded as a recorded step
    assert "(loaded)" in pane.data_label.text()
    step1, step2 = pane.boxes
    results = step1.run_checks()
    assert [r.passed for r in results] == [False, False, False] and step1.results.isVisibleTo(pane)
    assert exercises_qt.fail_color() in step1.results.text() and "no object named &#x27;fit&#x27; yet" in step1.results.text()
    session.run_code('fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)\nfit_white = fit')
    step1.check_button.click()
    assert all(r.passed for r in step1.last) and "All 3 checks pass." in step1.results.text()
    assert exercises_qt.pass_color() in step1.results.text() and exercises_qt.fail_color() not in step1.results.text()
    step2.check_button.click()
    by_id = {r.id: r for r in step2.last}
    assert not by_id["robust"].passed and by_id["robust-p"].passed
    text = step2.results.text()
    assert "2 of 3 checks pass" not in text and "1 of 3 checks pass." in text
    assert "Choose Standard errors: White (HC1)." in text and "✗ robust" in text and "✓ robust-p" in text
    assert pane.messages[-1] == "Exercise, step 2: 1 of 3 checks pass"


def test_the_data_load_only_when_the_user_agrees(app):
    session = Session()
    pane = _pane(session, answer=False)
    pane.start(ex.load_pack(PACK).exercise(2))
    assert "nile" not in session.objects and "not loaded" in pane.data_label.text()
    assert pane.messages[-1] == "Exercise: the data were not loaded"
    session.objects["nile"] = object()
    pane.confirm = lambda q: pane.asked.append(q) or True
    assert pane.load_data()
    assert "replaces the session's nile" in pane.asked[-1] and type(session.objects["nile"]).__name__ == "Dataset"
    session.undo()
    assert type(session.objects["nile"]).__name__ == "object"  # Undo takes it back


def test_the_dialog_lists_folders_and_starts_an_exercise(app, tmp_path, monkeypatch):
    from pymetrix import plugins

    monkeypatch.setattr(plugins, "_entry_points", lambda group: [])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    started = []
    dialog = exercises_qt.exercises_dialog(None, settings, started.append)
    assert dialog.packs == [] and "No exercise packs yet" in dialog.tree.topLevelItem(0).text(0)
    pack_dir = Path(shutil.copytree(PACK, tmp_path / "pack"))
    dialog.add_folder(str(pack_dir))
    assert settings.value(exercises_qt.FOLDERS_KEY, [], type=list) == [str(pack_dir.resolve())]
    top = dialog.tree.topLevelItem(0)
    assert "Test pack: two short exercises" in top.text(0) and top.childCount() == 2 and "[public]" in top.text(0)
    assert top.text(3) == "the test suite"  # the audience, in its column
    dialog.tree.setCurrentItem(top)
    assert not dialog.start_button.isEnabled() and dialog.remove_button.isEnabled()
    assert "robust standard errors" in dialog.details.text()
    dialog.tree.setCurrentItem(top.child(1))
    assert dialog.start_button.isEnabled() and "Data: nile" in dialog.details.text()
    dialog.start_button.click()
    assert [e.id for e in started] == ["02-unit-root-nile"]
    (pack_dir / "pack.toml").write_text("[pack]\n", encoding="utf-8")  # broken since: said, not raised
    again = exercises_qt.exercises_dialog(None, settings, started.append)
    assert again.packs == [] and "needs 'name'" in again.details.text()
    with pytest.raises(Exception, match="problem"):
        again.add_folder(str(pack_dir))


def test_the_main_window_has_help_exercises_and_docks_the_pane(app, first_run):
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        actions = {a.text().replace("&", ""): a for a in window.help_menu.actions()}
        assert "Exercises…" in actions
        names = list(actions)
        assert names.index("Exercises…") == names.index("How do I…") + 1
        pane = window.exercise_pane()
        pane.confirm = lambda q: False
        window.start_exercise(ex.load_pack(PACK).exercise(1))
        docks = [d for d in window.findChildren(QtWidgets.QDockWidget) if d.objectName() == "exercise"]
        assert len(docks) == 1 and docks[0].widget() is pane and docks[0].isVisibleTo(window)
        assert len(pane.boxes) == 2
        pane.boxes[0].check_button.click()  # against the window's own session
        assert pane.boxes[0].last and not any(r.passed for r in pane.boxes[0].last)
    finally:
        window.state["session"].dirty = False
        window.close()


def test_the_header_shows_the_classification_and_models_open_their_import_dialogs(app, tmp_path):
    from pymetrix.commands.builtin import ImportDynare

    copy = Path(shutil.copytree(PACK, tmp_path / "pack"))
    md = copy / "exercises" / "02-unit-root-nile" / "exercise.md"
    md.write_text(md.read_text(encoding="utf-8").replace("data: [nile]", "data: [nile]\nmodels: [rbc]"), encoding="utf-8")
    exercise = ex.load_pack(copy).exercise(2)
    pane = _pane(Session(), answer=False)
    pane.start(exercise)
    head = pane.findChild(QtWidgets.QLabel, "exercise-title").text()
    assert "Classification: public" in head and "beginner" in head
    assert list(pane.model_buttons) == ["rbc"]
    pane.model_buttons["rbc"].click()
    (cls, initial), = pane.opened
    assert cls is ImportDynare and initial.path == str(copy.resolve() / "models" / "rbc.mod") and initial.name == "rbc"


def test_import_uses_the_name_pack_toml_gives_the_model(app, tmp_path):
    from pymetrix.commands.builtin import ImportDynare

    copy = Path(shutil.copytree(PACK, tmp_path / "pack"))
    (copy / "pack.toml").write_text((copy / "pack.toml").read_text(encoding="utf-8") + '\n[models.rbc]\nname = "growth"\n',
                                    encoding="utf-8")
    md = copy / "exercises" / "02-unit-root-nile" / "exercise.md"
    md.write_text(md.read_text(encoding="utf-8").replace("data: [nile]", "data: [nile]\nmodels: [rbc]"), encoding="utf-8")
    pane = _pane(Session(), answer=False)
    pane.start(ex.load_pack(copy).exercise(2))
    assert pane.findChild(QtWidgets.QLabel, "exercise-model").text() == "Model: growth (rbc.mod)"
    assert "run it to add growth" in pane.model_buttons["rbc"].toolTip()
    pane.model_buttons["rbc"].click()
    (cls, initial), = pane.opened
    assert cls is ImportDynare and initial == ImportDynare(path=str(copy.resolve() / "models" / "rbc.mod"), name="growth")
