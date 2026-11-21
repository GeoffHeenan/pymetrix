"""The lab book: what it records, that it travels with the project, and what verify catches."""

import json
import zipfile

import pytest
from pymetrix_labbook import Chain, lab_book, register, verify
from pymetrix_labbook.cli import main
from pymetrix_labbook.verify import describe_actor

from pymetrix import Dataset
from pymetrix.commands import (
    FitEquation,
    GenerateSeries,
    Session,
    load_project,
    save_project,
)
from pymetrix.progress import Cancelled

register()  # as the entry point does when pymetrix loads its observers


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def _events(book):
    return [e["event"] for e in book.entries]


def _rewrite(path, out, blob):
    """A copy of the project with its lab book replaced by ``blob``: tampering, for the tests."""
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            data = blob if item.filename == "attachments/labbook/labbook.jsonl" else src.read(item.filename)
            dst.writestr(item, data)
    return out


def test_it_records_the_work_including_what_leaves_no_trace(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.run(GenerateSeries("z", "y * 2"))
    s.undo()
    with pytest.raises(Cancelled):
        s.run_code("from pymetrix.progress import Cancelled\nraise Cancelled('stopped')")
    book = lab_book(s)
    assert _events(book) == ["start", "step", "step", "undone", "cancelled"]
    assert book.entries[0]["data"]["existing_steps"] == 0
    step = book.entries[1]["data"]
    assert step["code"] == "fit = Equation('c ~ y').fit(data, method=OLS())" and step["call"] == "pymetrix:equation.fit"
    assert step["writes"]["fit"]["type"] == "FittedEquation" and step["writes"]["fit"]["nobs"] == 100
    assert len(step["writes"]["fit"]["fingerprint"]) == 16
    assert book.entries[3]["data"]["codes"] == [book.entries[2]["data"]["code"]]  # the step Undo took back
    assert not book.check()


def test_the_lab_book_travels_with_the_project_and_verifies(session, tmp_path):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    path = save_project(s, tmp_path / "study")
    opened = load_project(path)
    opened.run(FitEquation("c ~ y[-1]", name="fit2"))
    save_project(opened, path)
    book = Chain.from_bytes(zipfile.ZipFile(path).read("attachments/labbook/labbook.jsonl"))
    assert _events(book) == ["start", "step", "saved", "opened", "step", "saved"]  # replayed steps are not recorded again
    result = verify(path)
    assert result.ok, str(result)
    assert any("chain" in c for c in result.checked) and any("replay" in c for c in result.checked)
    assert any("coverage" in c for c in result.checked) and not result.warnings


def test_verify_catches_a_changed_entry(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "study")
    lines = zipfile.ZipFile(path).read("attachments/labbook/labbook.jsonl").decode().splitlines()
    entry = json.loads(lines[1])
    entry["data"]["code"] = "fit = Equation('c ~ y + i').fit(data, method=OLS())"  # a different model, after the fact
    lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    tampered = _rewrite(path, tmp_path / "tampered.pymx", ("\n".join(lines) + "\n").encode())
    result = verify(tampered, replay=False)
    assert not result.ok and any("entry 1" in e and "changed" in e for e in result.errors)


def test_verify_catches_removed_entries(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run(FitEquation("c ~ i", name="other"))
    path = save_project(session, tmp_path / "study")
    lines = zipfile.ZipFile(path).read("attachments/labbook/labbook.jsonl").decode().splitlines()
    del lines[2]  # the second fit, as if never tried
    result = verify(_rewrite(path, tmp_path / "cut.pymx", ("\n".join(lines) + "\n").encode()), replay=False)
    assert not result.ok and any("numbered" in e or "does not follow" in e for e in result.errors)


def test_verify_shows_work_saved_without_the_lab_book(session, tmp_path, monkeypatch):
    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "study")
    opened = load_project(path)
    import pymetrix.commands.events as events

    monkeypatch.setattr(events, "recorders", lambda: {})  # saved as if the plugin were not installed
    monkeypatch.setattr(events, "_observers", lambda: [])
    opened.run(FitEquation("c ~ i", name="unrecorded"))
    save_project(opened, path)
    monkeypatch.undo()
    result = verify(path)
    assert any("without the lab book active" in w for w in result.warnings)
    assert not result.ok and any("not those the lab book recorded" in e for e in result.errors)


def test_a_replay_that_gives_other_numbers_is_an_error(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "study")
    lines = zipfile.ZipFile(path).read("attachments/labbook/labbook.jsonl").decode().splitlines()
    chain = Chain()  # rebuilt with a different recorded fingerprint, so the chain itself is intact
    for line in lines:
        e = json.loads(line)
        if e["event"] == "step":
            e["data"]["writes"]["fit"]["fingerprint"] = "0" * 16
        chain.append(e["event"], e["data"], e["during"])
    result = verify(_rewrite(path, tmp_path / "other.pymx", chain.to_bytes()))
    assert not any("chain" in e for e in result.errors)  # the chain is intact: only the replay disagrees
    assert any("fit differs on replay" in e for e in result.errors)


def test_a_lab_book_started_on_existing_work_says_so(tmp_path, monkeypatch):
    import pymetrix.commands.events as events

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    monkeypatch.setattr(events, "_observers", lambda: [])  # as if the lab book were not installed yet
    s.run(FitEquation("c ~ y", name="before"))
    monkeypatch.undo()
    s.run(FitEquation("c ~ i", name="after"))
    book = lab_book(s)
    assert book.entries[0]["event"] == "start" and book.entries[0]["data"]["existing_steps"] == 1
    result = verify(save_project(s, tmp_path / "late"))
    assert any("already had 1 step" in n for n in result.notes)


def test_the_command_line(session, tmp_path, capsys):
    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "study")
    assert main(["verify", str(path)]) == 0 and "Verified: no errors." in capsys.readouterr().out
    assert main(["show", str(path)]) == 0
    out = capsys.readouterr().out
    assert "start" in out and "fit = Equation('c ~ y')" in out and "saved" in out
    plain = save_project(Session(), tmp_path / "plain")
    with zipfile.ZipFile(plain) as z:
        assert "attachments/labbook/labbook.jsonl" in z.namelist()  # installed: every saved project has one


def test_each_entry_says_who_acted_and_verify_counts_them(session, tmp_path, monkeypatch):
    from pymetrix.commands import events
    from pymetrix.commands.events import Actor, acting

    monkeypatch.setattr(events, "INTERFACE", "python")  # the app's tests, run earlier, set "app" for the process
    s = session
    s.run(FitEquation("c ~ y", name="fit"))  # the person at the machine, through Python
    with acting(s, Actor("agent", "claude-ai 0.1.0", "mcp")):
        s.run(FitEquation("c ~ y[-1]", name="fit2"))
    book = lab_book(s)
    person, agent = book.entries[1]["by"], book.entries[2]["by"]
    assert person["kind"] == "person" and person["via"] == "python"
    assert agent == {"kind": "agent", "name": "claude-ai 0.1.0", "via": "mcp"}
    assert s.actor is None  # acting ends with the block
    path = save_project(s, tmp_path / "study")
    result = verify(path)
    assert result.ok and result.actors == {describe_actor(person): 1, "agent claude-ai 0.1.0 (mcp)": 1}
    assert "Who acted:" in str(result) and "agent claude-ai 0.1.0 (mcp): 1 step(s)" in str(result)


def test_who_acted_cannot_be_changed_afterwards(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "study")
    blob = zipfile.ZipFile(path).read("attachments/labbook/labbook.jsonl")
    entries = [json.loads(line) for line in blob.decode().splitlines()]
    entries[1]["by"] = {"kind": "person", "name": "someone else", "via": "app"}  # claim an agent's or another's step
    forged = _rewrite(path, tmp_path / "forged.pymx", b"".join(json.dumps(e).encode() + b"\n" for e in entries))
    result = verify(forged, replay=False)
    assert not result.ok and any("entry 1" in e and "changed" in e for e in result.errors)


def test_an_assistant_over_mcp_is_recorded_as_an_agent(tmp_path):
    pytest.importorskip("mcp")
    import asyncio

    from mcp import Client

    from pymetrix.mcp_server import Policy, build_server

    server = build_server(policy=Policy(root=tmp_path))

    async def go():
        async with Client(server) as client:
            await client.call_tool("run_command", {"name": "LoadExample", "settings": {}})

    asyncio.run(go())
    by = lab_book(server.session_tools.session).entries[-1]["by"]
    assert by["kind"] == "agent" and by["via"] == "mcp" and by["name"]  # the client as it named itself


def test_a_step_an_assistant_proposed_says_so(session):
    with session.proposing(FitEquation, "Claude"):  # pymetrix extension API 1.9
        session.run(FitEquation("c ~ y", data="data", name="fit"))
    session.run(FitEquation("c ~ i", data="data", name="fit2"))
    steps = [e for e in lab_book(session).entries if e["event"] == "step"]
    assert [s["data"].get("proposed_by") for s in steps] == ["Claude", None]
