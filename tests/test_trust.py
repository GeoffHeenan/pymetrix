"""Project trust: typed code runs on opening only when the project is trusted."""

import os
import warnings
from pathlib import Path

import pytest

from pymetrix import Dataset, Equation
from pymetrix.commands import DefineEquation, FitEquation, RunCode, Session, load_project, save_project
from pymetrix.commands.project import ProjectTrustWarning, project_code
from pymetrix.trust import code_fingerprint, forget, is_trusted, trust_file

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture()
def shared_project(tmp_path):
    """A project from someone else: saved, then its code forgotten, as if received by email."""
    marker = tmp_path / "ran.txt"
    s = Session()
    s.objects["data"] = Dataset.read(str(DATA))
    code = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')\ndata = data.with_series(ly='100*log(y)')"
    s.run(RunCode(code, source="editor"))
    s.run(FitEquation("c ~ y", name="cons"))
    s.run(FitEquation("ly ~ c", name="needs_code"))  # uses a series the code made
    marker.unlink()
    path = save_project(s, tmp_path / "shared.pymx")
    info = project_code(path)
    assert info.trusted  # saved by its author, whose code ran
    forget(info.fingerprint)
    return path, marker, code


def test_untrusted_code_is_kept_but_not_run(shared_project, tmp_path):
    path, marker, code = shared_project
    info = project_code(path)
    assert info.steps == (code,) and info.needs_trust
    with pytest.warns(ProjectTrustWarning, match="1 step\\(s\\) of typed code were not run"):
        s = load_project(path)
    assert not marker.exists()  # the code did not run
    assert "cons" in s.objects  # dialog steps that do not need the code did
    assert "needs_code" not in s.objects
    skipped = [r for r in s.history if r.skipped]
    assert [type(r.command).__name__ for r in skipped] == ["RunCode", "FitEquation"]
    assert "not trusted" in skipped[0].error and "may need typed code" in skipped[1].error
    assert "# Not run when the project was opened" in s.script() and "# pathlib.Path(" in s.script()
    # saving again keeps the code, and does not make it trusted
    again = save_project(s, tmp_path / "again.pymx")
    assert project_code(again).steps == (code,) and not project_code(again).trusted


def test_trusting_runs_the_code_and_is_remembered(shared_project):
    path, marker, _ = shared_project
    s = load_project(path, trust=True)
    assert marker.read_text() == "ran" and "needs_code" in s.objects
    assert project_code(path).trusted
    marker.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProjectTrustWarning)  # no trust warning once trusted
        load_project(path)
    assert marker.exists()
    marker.unlink()
    with pytest.warns(ProjectTrustWarning):
        load_project(path, trust=False)  # never runs, even when trusted
    assert not marker.exists()


def test_changed_code_needs_trust_again(shared_project, tmp_path):
    import json
    import zipfile

    path, _, code = shared_project
    load_project(path, trust=True)
    changed = tmp_path / "changed.pymx"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(changed, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            if item.filename == "commands.json":
                commands = json.loads(data)
                commands[0]["fields"]["code"] = code + "\nimport os"
                data = json.dumps(commands).encode()
            dst.writestr(item, data)
    assert project_code(changed).needs_trust


def test_projects_without_code_and_ipython_lines_need_no_trust(tmp_path):
    s = Session()
    s.objects["data"] = Dataset.read(str(DATA))
    s.run(DefineEquation("c ~ y", name="eq"))
    s.run(RunCode("%timeit 1 + 1"))  # IPython syntax: recorded, never run
    path = save_project(s, tmp_path / "plain.pymx")
    assert project_code(path).steps == () and not project_code(path).needs_trust
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProjectTrustWarning)
        assert isinstance(load_project(path)["eq"], Equation)


def test_the_store_lives_in_pymetrix_home():
    assert str(trust_file()).startswith(os.environ["PYMETRIX_HOME"])
    fp = code_fingerprint(["x = 1"])
    assert not is_trusted(fp) and fp != code_fingerprint(["x = 2"])


def test_the_desktop_dialog_shows_the_code(shared_project):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import trust_dialog

    path, _, code = shared_project
    dialog = trust_dialog(None, "shared.pymx", project_code(path))
    assert code in dialog.code.toPlainText() and dialog.code.isReadOnly()
    assert dialog.buttons["without"].isDefault()
    dialog.buttons["without"].click()
    assert dialog.choice is False
    dialog = trust_dialog(None, "shared.pymx", project_code(path))
    dialog.buttons["run"].click()
    assert dialog.choice is True
