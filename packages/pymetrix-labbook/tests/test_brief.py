"""The project brief, notes and objectives (research-audit layer 2): recorded in the lab book before the work, amended
never edited, and checked by verify."""

import pytest
from pymetrix_labbook import (
    brief,
    current_brief,
    lab_book,
    note,
    objective,
    register,
    skip_brief,
    verify,
)
from pymetrix_labbook.verify import brief_report

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session, save_project

register()


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def _brief(s, **extra):
    return brief(s, "Consumption and income", "How much of a rise in income is consumed?",
                 objectives={"O1": "Estimate the marginal propensity to consume"},
                 hypotheses={"H1": {"statement": "The MPC is below one", "supports": "a coefficient on y under 1"}},
                 primary_outcomes=["the coefficient on y"], data={"sources": "sample_macro"}, **extra)


def test_a_brief_before_the_work_and_steps_linked_to_it_verify_clean(session, tmp_path):
    s = session
    _brief(s)
    with objective("O1", "H1"):
        s.run(FitEquation("c ~ y", name="fit"))
    note(s, "Kept the full sample: no breaks found", about="O1")
    book = lab_book(s)
    events = [e["event"] for e in book.entries]
    assert events == ["start", "brief", "step", "note"]
    assert book.entries[2]["data"]["objectives"] == ["O1", "H1"]
    assert book.entries[1]["by"]["kind"] == "person" and book.entries[3]["data"]["about"] == "O1"
    assert current_brief(s)["hypotheses"]["H1"]["supports"] == "a coefficient on y under 1"
    result = verify(save_project(s, tmp_path / "study"))
    assert result.ok and not result.warnings, str(result)
    assert any("written at entry 1, before any step" in c for c in result.checked)
    assert any("every objective and hypothesis" in c for c in result.checked)


def test_amendments_need_a_reason_and_keep_the_first_brief(session):
    _brief(session)
    with pytest.raises(ValueError, match="reason"):
        _brief(session)
    brief(session, "Consumption and income", "How much of a rise in income is consumed?",
          objectives={"O1": "Estimate the MPC", "O2": "Test for a break"}, reason="a referee asked about breaks")
    briefs = [e for e in lab_book(session).entries if e["event"] == "brief"]
    assert len(briefs) == 2 and "reason" not in briefs[0]["data"] and briefs[1]["data"]["reason"].startswith("a referee")
    assert set(current_brief(session)["objectives"]) == {"O1", "O2"}


def test_verify_reports_what_the_brief_shows(session):
    s = session
    s.run(FitEquation("c ~ y", name="early"))  # work before the brief
    _brief(s)
    with objective("O1"):
        s.run(FitEquation("c ~ y[-1]", name="fit"))
    with objective("H2"):
        s.run(FitEquation("c ~ y + y[-1]", name="fit2"))
    brief(s, "Consumption and income", "How much of a rise in income is consumed?",
          objectives={"O1": "Estimate the MPC", "O3": "Never done"},
          hypotheses={"H1": "The MPC is below one", "H2": "Lagged income matters"}, reason="added H2 and O3")
    warnings, notes, _ = brief_report(lab_book(s))
    text = " | ".join(warnings)
    assert "after 1 step(s) of work" in text  # the brief came after the first fit
    assert "no analysis linked to them: O3, H1" in text
    assert "H2 was added to the brief" in text  # a hypothesis named only after a step on it
    assert any("1 step(s) are linked to no objective" in n for n in notes)
    assert any("amended" in n and "added H2 and O3" in n for n in notes)


def test_skipping_the_brief_is_recorded(session):
    skip_brief(session, "a teaching exercise")
    _, notes, _ = brief_report(lab_book(session))
    assert any("skipped" in n and "a teaching exercise" in n for n in notes)
    with pytest.raises(ValueError):
        skip_brief(session, "")


def test_objectives_nest_and_end_with_their_block(session):
    s = session
    with objective("O1"):
        with objective("H1"):
            s.run(FitEquation("c ~ y", name="fit"))
        s.run(FitEquation("c ~ y[-1]", name="fit2"))
    s.run(FitEquation("c ~ y + y[-1]", name="fit3"))
    steps = [e["data"].get("objectives") for e in lab_book(s).entries if e["event"] == "step"]
    assert steps == [["O1", "H1"], ["O1"], None]
