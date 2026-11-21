"""Counting what was tried: every fit of an outcome, kept or not, counted by specification; the report line and the
specification curve."""

import pytest
from pymetrix_labbook import lab_book, register, specification_curve, tried, verify
from pymetrix_labbook.tried import fits, outcome_counts

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session, save_project

register()


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def test_each_fit_records_its_outcome_sample_and_estimates(session):
    session.run(FitEquation("c ~ y", name="fit"))
    info = lab_book(session).entries[-1]["data"]["writes"]["fit"]
    assert info["outcome"] == "c" and info["sample"].count("-") >= 1
    value, se = info["estimates"]["y"]
    assert value == pytest.approx(session.objects["fit"].params.values[1], rel=1e-7) and se > 0


def test_every_fit_counts_kept_or_not(session, tmp_path):
    s = session
    s.run(FitEquation("c ~ y", name="a"))
    s.run(FitEquation("c ~ y + y[-1]", name="b"))
    s.undo()  # undone: still tried
    s.run(FitEquation("c ~ y[-1]", name="b"))
    s.run(FitEquation("c ~ y", name="again"))  # the same specification as a, renamed
    s.run(FitEquation("i ~ y", name="inv"))
    assert outcome_counts(s) == {"c": (3, 4), "i": (1, 1)}
    line = tried(s, "b")
    assert line.startswith("b is 1 of 3 specifications fitted for c (4 fits, lab book entries ")
    with pytest.raises(ValueError):
        tried(s, "nothing")
    result = verify(save_project(s, tmp_path / "study"))
    assert any(n.startswith("c: 3 specification(s) fitted (4 fit(s)") for n in result.notes), str(result)


def test_the_specification_curve(session):
    s = session
    for formula, name in (("c ~ y", "a"), ("c ~ y + y[-1]", "b"), ("c ~ y + i", "c2")):
        s.run(FitEquation(formula, name=name))
    fig = specification_curve(s, "c", "y", highlight="b")
    ax = fig.axes[0]
    assert "3 specification(s) of c" in ax.get_xlabel() and "b highlighted" in ax.get_title()
    values = sorted(f["estimates"]["y"][0] for f in fits(s, "c"))
    points = sorted(line.get_ydata()[0] for line in ax.lines if line.get_marker() == "o")
    assert points == pytest.approx(values)
    with pytest.raises(ValueError, match="no fit"):
        specification_curve(s, "c", "nonexistent")


def test_an_exported_report_says_what_was_tried(session, tmp_path):
    from pymetrix.commands import ExportReport

    s = session
    s.run(FitEquation("c ~ y", name="a"))
    s.run(FitEquation("c ~ y + y[-1]", name="b"))
    path = tmp_path / "report.md"
    s.run(ExportReport(objects="b, data", path=str(path)))
    assert "What was tried" not in path.read_text(encoding="utf-8")  # off unless asked: exports replay exactly
    s.run(ExportReport(objects="b, data", path=str(path), plugin_notes=True))
    assert s.history[-1].code.startswith("# the plugins' notes")
    text = path.read_text(encoding="utf-8")
    assert "What was tried" in text and "b is 1 of 2 specifications fitted for c (2 fits" in text
    assert text.index("What was tried") < text.index("Appendix")  # before the appendix
