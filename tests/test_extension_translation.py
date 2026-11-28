"""Extension API 1.11: ``pymetrix.translation``, the translation framework, a stable module, with its main names in
``pymetrix.extension``. The worked example (examples/translation_toy.py, a toy language of two statements) shows the
interface serves a language other than EViews: its script and its steps give equal objects, every line has one row,
and the values are worked out by hand. A function's ``result`` (``Def.result``, for languages whose functions return
a value) is checked the same way; the panel shows a front end's translations by its ``PanelSpec``."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
TOY = """-- a toy program
years 2001 2005
let t = obsnum()  -- 1, 2, ...

let Y = 2 * t + 1
print y
plot y
let = 3
"""


@pytest.fixture(scope="module")
def toy():
    spec = importlib.util.spec_from_file_location("translation_toy", EXAMPLES / "translation_toy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["translation_toy"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("translation_toy", None)


def test_the_version_and_the_public_names():
    from pymetrix import extension as ext
    from pymetrix import translation
    from pymetrix.plugins import API

    assert tuple(map(int, API.split("."))) >= (1, 11)
    assert "pymetrix.translation" in ext.STABLE
    pairs = {"Translation": translation.Translation, "TranslationReport": translation.TranslationReport,
             "TranslationOutcome": translation.Outcome, "TranslationPanelSpec": translation.PanelSpec,
             "translation_panel": translation.translation_panel}
    for name, obj in pairs.items():
        assert getattr(ext, name) is obj, name
        assert name in ext.__all__
    for name in translation.__all__:  # the stable module's names all exist
        assert hasattr(translation, name), name


def test_the_eviews_plugin_uses_the_framework():
    """The EViews plugin's translation is core's, subclassed: its report and panel are core's, its modules import
    pymetrix.translation itself, never its submodules."""
    pytest.importorskip("pymetrix_eviews")
    from pymetrix import translation
    from pymetrix_eviews import ProgramTranslation, ReportRow, TranslationReport, translate_program

    assert issubclass(ProgramTranslation, translation.Translation) and issubclass(ReportRow, translation.ReportRow)
    assert TranslationReport is translation.TranslationReport
    t = translate_program("wfcreate a 2001 2003\nseries x = 1\ngraph g.line x\n")
    assert t.language == "EViews" and t.recorded_by == "eviews:program" and t.attachments == "pymetrix-eviews"
    assert t.report.summary().columns[3] == "EViews" and t.report[1].eviews == t.report[1].source == "series x = 1"
    from pymetrix.translation.testing import assert_rows_cover, assert_steps_equal_script, status_counts

    assert_rows_cover(t)
    assert status_counts(t) == {"translated": 2, "translated with a note": 0, "left out": 1, "error": 0}
    assert_steps_equal_script(t)
    import pymetrix_eviews

    root = Path(pymetrix_eviews.__file__).parent
    for path in [*root.glob("program/*.py"), root / "panel.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pymetrix.translation"):
                assert node.module == "pymetrix.translation", f"{path.name}: from {node.module}"


# -- the worked example: a toy language of two statements --------------------------------------------------------


def test_the_toy_front_end_script_report_and_steps(toy):
    from pymetrix.commands import Session
    from pymetrix.translation import ERROR, LEFT, NOTE, TRANSLATED
    from pymetrix.translation.testing import assert_rows_cover, assert_steps_equal_script, status_counts

    t = toy.translate_toy(TOY, "toy.txt")
    # the report: a row per line with text (the blank line 5 has none), statuses and reasons as the toy gives them
    assert [(r.line, r.status) for r in t.report] == [(1, TRANSLATED), (2, TRANSLATED), (3, TRANSLATED), (5, NOTE),
                                                      (6, LEFT), (7, ERROR), (8, ERROR)]
    assert t.report.line(1).python == "# a toy program" and t.report.line(1).reason == "a comment"
    assert t.report.line(5).reason == "Y is called y"
    assert t.report.line(6).reason == "display only; in pymetrix: print(data) in a script; the Data window in the app"
    assert "'plot' is not a statement" in t.report.line(7).reason and "let takes a name" in t.report.line(8).reason
    assert t.report.summary().columns[3] == "Toy"
    assert_rows_cover(t)
    assert status_counts(t) == {"translated": 3, "translated with a note": 1, "left out": 1, "error": 2}
    # the script: each line a comment above what it became, the comment lines kept, pymetrix's imports found
    script = t.script.split("\n")
    assert script[2] == "from pymetrix import Dataset"
    assert 'data = data.with_series(y="2 * t + 1")' in script
    at = t.script_line(5)
    assert script[at - 1] == "# 5: let Y = 2 * t + 1" and script[at] == "#   note: Y is called y"
    assert script[t.script_line(6) - 1] == "# 6: print y"
    # the script and the steps give equal objects; the values by hand: t = 1, ..., 5 and y = 2t + 1
    s = Session()
    namespace, out = assert_steps_equal_script(t, s)
    frame = s["data"].frame
    assert list(frame.index.astype(str)) == ["2001", "2002", "2003", "2004", "2005"]
    np.testing.assert_array_equal(frame["t"].to_numpy(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(frame["y"].to_numpy(), [3, 5, 7, 9, 11])
    # each step recorded, with the toy's source; the script and the report kept under its key
    steps = [r for r in s.history if getattr(r.command, "source", "") == "toy:program"]
    assert len(steps) == 3
    files = s.attachments["toy-translator"]
    assert files["toy.py"].decode() == t.script and files["toy.txt"].decode() == t.report.to_text()


def test_a_function_that_returns_a_value():
    """``Def.result``: a function's one value (an R function's last expression), returned in the script and given
    to the call's target in steps mode; ``returns`` and ``result`` together are refused."""
    from pymetrix.translation import (TRANSLATED, Code, Def, If, Invoke, Jump, ReportRow, Translation, TranslationReport,
                                      body_lines)
    from pymetrix.translation.testing import assert_steps_equal_script

    body = ((1, Code("y = 2 * x", ("y",)), ""), (2, If("x < 0", ((2, Jump("return"), ""),)), ""),
            (3, Code("y = y + 1", ("y",)), ""))
    f = Def("f", ("x",), local=True, body=body, result="y")  # its own variables, as an R function has
    ops = ((1, f, ""), (4, Code("a = 5", ("a",)), ""), (5, Invoke("f", ("a",), ("b",)), ""), (6, Invoke("f", ("-1",), ("c",)), ""))
    script = "\n".join(body_lines(ops, 0)) + "\n"
    assert script == ("def f(x):\n    y = 2 * x\n    if x < 0:\n        return y\n    y = y + 1\n    return y\n"
                      "a = 5\nb = f(a)\nc = f(-1)\n")
    t = Translation(script, TranslationReport((ReportRow(1, TRANSLATED, "b = f(5)"),)), ops=ops[1:], functions=(ops[0],))
    namespace, out = assert_steps_equal_script(t)
    assert out["b"] == namespace["b"] == 11 and out["c"] == namespace["c"] == -2  # 2 * 5 + 1; 2 * -1, returned early
    with pytest.raises(ValueError, match="not both"):
        Def("f", ("x",), returns=("x",), result="x")


def test_names_for_the_program_objects():
    from pymetrix.translation import allocate_name, reserved_names

    taken: dict[str, str] = {}
    reserved = reserved_names("data")
    assert {"Dataset", "print", "data"} <= reserved
    assert allocate_name("x", "x", "series x", taken, reserved) == ("x", "")
    assert allocate_name("x", "x", "series x", taken, reserved) == ("x", "")  # the same thing again: the same name
    assert allocate_name("X!", "x", "scalar x", taken, reserved) == ("x_2", "X! is called x_2 (x is taken)")
    assert allocate_name("print", "print", "series print", taken, reserved) == (
        "print_v", "print is called print_v (print is a Python or pymetrix name)")


def test_the_panel_shows_a_front_ends_translations(toy):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtGui import QColor

    from pymetrix.commands import Session
    from pymetrix.extension import AppInterface
    from pymetrix.translation import COLOURS, PanelSpec, translation_panel

    t = toy.translate_toy(TOY, "toy.txt")
    made: list = []
    s = Session()
    messages, dialogs = [], []
    spec = PanelSpec("Toy", lambda session: list(made), "ToyDialog", "File → Translate toy program…", "a toy program",
                     prefix="toy")
    interface = AppInterface(lambda: s, lambda: None, lambda cls, initial: dialogs.append(cls), s.run, messages.append)
    panel = translation_panel(spec)(interface)
    assert panel.objectName() == "toy-translation-panel"
    assert panel.message.text() == "No program translated yet: File → Translate toy program…."
    made.append((None, t))
    interface._changed()
    assert panel.current is t and panel.table.rowCount() == len(t.report) == 7
    assert [panel.table.horizontalHeaderItem(j).text() for j in range(5)] == ["line", "status", "Toy", "Python", "reason"]
    assert panel.table.item(4, 1).text() == "left out"
    assert panel.table.item(4, 1).background().color() == QColor(*COLOURS["left out"])
    panel.translate_button.click()
    assert dialogs == ["ToyDialog"]
    assert panel.run_steps() and "Ran toy.txt as steps" in messages[-1]
    np.testing.assert_array_equal(s["data"]["y"].to_numpy(), [3, 5, 7, 9, 11])
    panel.deleteLater()


def test_the_help_topic_and_its_example(toy):
    """The help topic's example is the worked example's: run as written, it gives what its comments say."""
    from pymetrix.commands import Session
    from pymetrix.help import topic

    text = topic("translation-framework").text
    assert "examples/translation_toy.py" in text and "pymetrix.translation.testing" in text
    first = next(line.strip() for line in text.split("\n") if line.strip().startswith("t = translate_toy("))
    namespace = {"translate_toy": toy.translate_toy}
    exec(first, namespace)  # noqa: S102 - the help's own example
    t = namespace["t"]
    assert 'data = Dataset.empty("Y", "2001", "2005")' in t.script
    assert t.report.counts() == {"translated": 3, "translated with a note": 0, "left out": 1, "error": 0}
    s = Session()
    t.run(s)
    assert len(s.history) == 3
