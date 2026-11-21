"""A private brief: only its hash in the lab book until it is revealed, then checked against the commitment and
counted as written when committed."""

import json

import pytest
from pymetrix_labbook import (
    commit_brief,
    current_brief,
    lab_book,
    objective,
    register,
    reveal_brief,
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


def test_commit_then_reveal(session, tmp_path):
    s = session
    path = tmp_path / "brief.json"
    digest = commit_brief(s, path, "Consumption and income", "How much of a rise in income is consumed?",
                          objectives={"O1": "Estimate the MPC"})
    commitment = lab_book(s).entries[-1]
    assert commitment["event"] == "brief_commitment" and commitment["data"] == {"hash": digest, "algorithm": "sha256"}
    assert "Consumption" not in json.dumps(lab_book(s).entries)  # nothing of the brief but its hash
    assert current_brief(s) is None
    with objective("O1"):
        s.run(FitEquation("c ~ y", name="fit"))
    _, notes, _ = brief_report(lab_book(s))
    assert any("not yet revealed" in n for n in notes)
    reveal_brief(s, path)
    assert current_brief(s)["title"] == "Consumption and income"
    with pytest.raises(ValueError, match="already revealed"):
        reveal_brief(s, path)
    result = verify(save_project(s, tmp_path / "study"))
    assert result.ok and not result.warnings, str(result)
    assert any(f"written at entry {commitment['seq']}, before any step" in c for c in result.checked)  # when committed
    assert any("matches its commitment" in c for c in result.checked)


def test_a_changed_file_is_refused_and_files_are_never_overwritten(session, tmp_path):
    path = tmp_path / "brief.json"
    commit_brief(session, path, "Title", "Question?")
    payload = json.loads(path.read_text())
    payload["brief"]["question"] = "A different question?"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="matches no commitment"):
        reveal_brief(session, changed)
    with pytest.raises(FileExistsError):
        commit_brief(session, path, "Title", "Question?", reason="again")
    with pytest.raises(ValueError, match="reason"):
        commit_brief(session, tmp_path / "second.json", "Title", "Question?")  # a second commitment amends


def test_a_salt_hides_a_guessable_brief(session, tmp_path):
    a = commit_brief(session, tmp_path / "a.json", "T", "Q?")
    other = Session()
    b = commit_brief(other, tmp_path / "b.json", "T", "Q?")
    assert a != b  # the same brief, committed twice, gives different hashes
