"""Ask (assistant stage 2): a question goes with the project in metadata only, the model's tools only look things up
and propose, and a proposal opens in its dialog, filled in, for the user to run, or is skipped. A fake client stands
in for Anthropic's: no network."""

import os
import time
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
from pymetrix_assistant import chat
from pymetrix_assistant.chat import Conversation, propose

from pymetrix.commands import FitEquation, LoadData, Session


def text(t):
    return NS(type="text", text=t)


def use(i, tool, **inp):
    return NS(type="tool_use", id=f"tu{i}", name=tool, input=inp)


def reply(*blocks, stop="end_turn", iterations=None):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(iterations=iterations))


class FakeClient:
    """Anthropic's ``beta.messages.create``, answering from a script and keeping each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **request):
        self.requests.append({**request, "messages": [dict(m) for m in request["messages"]]})
        return self.responses.pop(0)


@pytest.fixture
def session(tmp_path):
    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"c": 987654.3 + rng.normal(size=40), "y": 123456.7 + rng.normal(size=40)}, index=idx)
    folder = tmp_path / "secret_folder"
    folder.mkdir()
    frame.to_csv(folder / "data.csv")
    s = Session()
    s.run(LoadData(str(folder / "data.csv"), time="time", freq="Q", name="data"))
    s.run(FitEquation("c ~ y", data="data", name="fit"))
    return s


def facts(session):
    from pymetrix.disclosure import describe_session, to_text

    return to_text(describe_session(session))


def test_a_question_looks_up_then_proposes_and_nothing_runs(session):
    client = FakeClient(
        reply(use(1, "list_commands", menu="Estimate"), stop="tool_use"),
        reply(use(2, "describe_command", name="FitEquation"), stop="tool_use"),
        reply(text("Try this."), use(3, "propose_step", command="FitEquation",
                                      settings={"formula": "c ~ y + c[-1]", "data": "data", "name": "fit2"},
                                      why="Add lagged consumption."), stop="tool_use"),
        reply(text("Open it to run it.")),
    )
    steps = len(session.history)
    answer = Conversation(client).ask("Improve the consumption function", facts(session))
    assert len(session.history) == steps and "fit2" not in session.objects  # nothing ran
    [p] = answer.proposals
    assert p.command_cls is FitEquation and p.command.formula == "c ~ y + c[-1]" and p.why == "Add lagged consumption."
    assert "Try this." in answer.text and "Open it to run it." in answer.text
    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    results = {r["tool_use_id"]: r for m in client.requests[-1]["messages"] if m["role"] == "user"
               for r in m["content"] if isinstance(r, dict) and r.get("type") == "tool_result"}
    assert "FitEquation" in results["tu1"]["content"] and "formula" in results["tu2"]["content"]
    assert "Shown to the user" in results["tu3"]["content"] and not results["tu3"].get("is_error")


def test_what_is_sent_carries_no_values_paths_or_formulas(session):
    client = FakeClient(reply(text("ok")))
    Conversation(client).ask("What next?", facts(session))
    sent = repr(client.requests[0]["messages"])  # the system prompt and tools are pymetrix's own, the same for all
    for secret in ("987654", "123456", "secret_folder", "c ~ y"):
        assert secret not in sent, secret
    assert '"name":"data"' in sent and "<project metadata_only>" in sent


def test_a_wrong_proposal_is_told_back_to_the_model():
    client = FakeClient(
        reply(use(1, "propose_step", command="FitEquaton", settings={}, why="x"), stop="tool_use"),
        reply(use(2, "propose_step", command="FitEquation", settings={"nonsense": 1}, why="x"), stop="tool_use"),
        reply(use(3, "propose_step", command="RunCode", settings={"code": "1"}, why="x"), stop="tool_use"),
        reply(text("Sorry.")),
    )
    answer = Conversation(client).ask("?", "{}")
    assert answer.proposals == []
    results = [r for m in client.requests[-1]["messages"] if m["role"] == "user" for r in m["content"]
               if isinstance(r, dict) and r.get("type") == "tool_result"]
    assert all(r["is_error"] for r in results)
    assert "did you mean FitEquation" in results[0]["content"] and "no setting 'nonsense'" in results[1]["content"]
    assert "not proposed" in results[2]["content"]


def test_a_refusal_takes_the_question_back():
    conversation = Conversation(FakeClient(reply(text("hi")), reply(stop="refusal")))
    conversation.ask("first", "{}")
    kept = list(conversation.messages)
    answer = conversation.ask("second", "{}")
    assert "declined" in answer.text and conversation.messages == kept


def test_run_code_is_not_offered():
    assert "RunCode" not in chat.list_commands()
    with pytest.raises(chat.ToolError):
        propose("RunCode", {"code": "import os"}, "")


def test_the_key_is_kept_in_the_keychain(monkeypatch):
    kept = {}
    import keyring

    monkeypatch.setattr(keyring, "set_password", lambda s, u, k: kept.__setitem__((s, u), k))
    monkeypatch.setattr(keyring, "get_password", lambda s, u: kept.get((s, u)))
    assert chat.api_key() is None
    chat.save_api_key("  sk-test  ")
    assert kept == {("pymetrix-assistant", "anthropic"): "sk-test"} and chat.api_key() == "sk-test"


# --- the panel -------------------------------------------------------------------------------------------------------

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel_with(qapp, session):
    from pymetrix_assistant.app import create_panel

    from pymetrix.extension import AppInterface

    opened, messages, ok_pressed = [], [], []

    def open_dialog(cls, initial):  # the user presses OK when the test says so, after changing the name
        import dataclasses

        opened.append((cls, initial))
        if ok_pressed:
            session.run(dataclasses.replace(initial, name=initial.name + "_mine"))

    app = AppInterface(lambda: session, lambda: None, open_dialog, session.run, messages.append)

    def make(client, press_ok=False):
        if press_ok:
            ok_pressed.append(True)
        return create_panel(app, client_factory=lambda: client), opened

    return make


def _wait(qapp, tab, seconds=20):
    end = time.monotonic() + seconds
    while (tab.worker() is not None) and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert tab.worker() is None, "timed out"


def test_the_mode_is_the_projects_and_off_until_chosen(panel_with, session, monkeypatch):
    from pymetrix_assistant.app import project_mode

    client = FakeClient()
    panel, _ = panel_with(client)
    tab = panel.ask
    assert project_mode(session) == "off" and tab.mode.currentData() == "off"
    import pymetrix_assistant.app as panel_module

    monkeypatch.setattr(panel_module, "ask_first_use", lambda parent: None)  # the user cancels
    tab.question.setPlainText("What next?")
    tab.ask_button.click()
    assert client.requests == [] and tab.worker() is None  # declined: nothing sent
    tab.mode.setCurrentIndex(tab.mode.findData("metadata_only"))
    assert session.attachments["pymetrix-assistant"]["mode"] == b"metadata_only" and session.dirty


def test_a_proposal_opens_in_its_dialog_or_is_skipped(qapp, panel_with, session, monkeypatch):
    from pymetrix_assistant.app import set_project_mode

    set_project_mode(session, "metadata_only")
    client = FakeClient(
        reply(use(1, "propose_step", command="FitEquation", settings={"formula": "c ~ y", "data": "data"}, why="A start."),
              stop="tool_use"),
        reply(text("Here is a first step.")),
    )
    panel, opened = panel_with(client)
    tab = panel.ask
    tab.question.setPlainText("Fit something")
    tab.ask_button.click()
    _wait(qapp, tab)
    qapp.processEvents()
    assert "Here is a first step." in tab.transcript.toPlainText()
    cards = [w for w in tab.proposals.findChildren(QtWidgets.QFrame, "assistant-proposal")]
    assert len(cards) == 1
    steps = len(session.history)
    cards[0].open_it.click()
    [(cls, initial)] = opened
    assert cls is FitEquation and initial.formula == "c ~ y" and len(session.history) == steps  # the user runs it
    cards[0].skip.click()
    qapp.processEvents()
    assert not cards[0].isVisible()


def test_the_record_and_the_history_mark(qapp, panel_with, session, monkeypatch, tmp_path):
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import set_project_mode

    from pymetrix.commands import load_project, save_project

    set_project_mode(session, "metadata_only")
    client = FakeClient(
        reply(use(1, "propose_step", command="FitEquation", settings={"formula": "c ~ y", "data": "data", "name": "fit2"},
                  why="First."), stop="tool_use"),
        reply(use(2, "propose_step", command="FitEquation", settings={"formula": "y ~ c", "data": "data", "name": "fit3"},
                  why="Second."), stop="tool_use"),
        reply(text("Two ideas.")),
    )
    panel, _ = panel_with(client, press_ok=True)
    tab = panel.ask
    tab.question.setPlainText("Two fits please")
    tab.ask_button.click()
    _wait(qapp, tab)
    qapp.processEvents()
    first, second = tab.proposals.findChildren(QtWidgets.QFrame, "assistant-proposal")
    first.open_it.click()  # opened, and the user pressed OK
    assert session.history[-1].proposed_by == "Claude" and "fit2_mine" in session.objects
    second.skip.click()
    kinds = [(e["kind"], e.get("proposal"), e.get("step")) for e in audit.read(session)]
    assert kinds == [("asked", None, None), ("opened", 1, len(session.history)), ("skipped", 2, None)]
    asked = audit.read(session)[0]
    assert asked["text"] == "Two fits please" and asked["mode"] == "metadata_only" and asked["model"] == "claude-opus-5-5"
    assert '"name":"data"' in asked["sent"] and "fit2" not in asked["sent"]  # the project as it was when asked
    assert [p["command"] for p in asked["proposals"]] == ["FitEquation", "FitEquation"]
    readable = audit.as_text(audit.read(session))
    assert audit.read(session)[1]["changed"] == ["name"]
    assert "Question 1" in readable and "after changing name" in readable and "Proposal 1.2 (FitEquation)" in readable
    reopened = load_project(save_project(session, tmp_path / "p.pymx"))  # kept with the project
    assert audit.read(reopened) == audit.read(session) and reopened.history[-1].proposed_by == "Claude"


def test_a_question_that_fails_is_recorded(qapp, panel_with, session):
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import set_project_mode

    class Broken(FakeClient):
        def create(self, **request):
            raise ConnectionError("offline")

    set_project_mode(session, "metadata_only")
    panel, _ = panel_with(Broken())
    tab = panel.ask
    tab.question.setPlainText("Anything?")
    tab.ask_button.click()
    _wait(qapp, tab)
    qapp.processEvents()
    [entry] = audit.read(session)
    assert entry["kind"] == "failed" and entry["question"] == 1 and "offline" in entry["error"]


def test_full_context_shows_each_send_first(qapp, panel_with, session, monkeypatch):
    import pymetrix_assistant.app as panel_module
    from pymetrix_assistant import audit

    shown, answers = [], []

    def confirm(parent, title, note, text):
        shown.append(text)
        return answers.pop(0)

    monkeypatch.setattr(panel_module, "confirm_send", confirm)
    monkeypatch.setattr(panel_module, "confirm_full_context", lambda parent: True)
    client = FakeClient(
        reply(use(1, "show_object", name="fit"), stop="tool_use"),
        reply(use(2, "show_object", name="data"), stop="tool_use"),
        reply(text("The slope is about one.")),
    )
    panel, _ = panel_with(client)
    tab = panel.ask
    tab.mode.setCurrentIndex(tab.mode.findData("full_context"))
    assert session.attachments["pymetrix-assistant"]["mode"] == b"full_context"

    answers[:] = [False]  # the question itself: not sent
    tab.question.setPlainText("Is the fit good?")
    tab.ask_button.click()
    assert client.requests == [] and tab.question.toPlainText() == "Is the fit good?" and "c ~ y" in shown[0]

    answers[:] = [True, True, False]  # the question, fit's table, then data's table declined
    tab.ask_button.click()
    _wait(qapp, tab)
    qapp.processEvents()
    first = client.requests[0]
    assert "show_object" in {t["name"] for t in first["tools"]} and "<project full_context>" in repr(first["messages"])
    assert "c ~ y" in repr(first["messages"])  # the steps' code, as approved
    results = [r for m in client.requests[-1]["messages"] if m["role"] == "user" for r in m["content"]
               if isinstance(r, dict) and r.get("type") == "tool_result"]
    assert "coef" in results[0]["content"] and not results[0].get("is_error")
    assert results[1]["is_error"] and "declined" in results[1]["content"]
    kinds = [(e["kind"], e.get("object"), e.get("sent") if e["kind"] == "shown" else None) for e in audit.read(session)]
    assert kinds == [("shown", "fit", True), ("shown", "data", False), ("asked", None, None)]
    assert "coef" in audit.read(session)[0]["text"] and audit.read(session)[1]["text"] == ""  # what went, and nothing
    assert audit.read(session)[-1]["mode"] == "full_context"


def test_metadata_only_never_offers_show_object(session):
    client = FakeClient(reply(use(1, "show_object", name="fit"), stop="tool_use"), reply(text("ok")))
    Conversation(client).ask("?", facts(session))
    assert "show_object" not in {t["name"] for t in client.requests[0]["tools"]}
    result = client.requests[-1]["messages"][-1]["content"][0]
    assert result["is_error"] and "metadata only" in result["content"]


@pytest.fixture
def course(tmp_path):
    """A course folder whose notes set ``assistant`` (loaded into the help, then unloaded)."""
    from pymetrix import help as pm_help

    loaded = []

    def make(rule):
        folder = tmp_path / f"course_{rule}"
        folder.mkdir()
        (folder / "week1.md").write_text("---\ntitle: Week 1\ncourse: ECON 101\n---\n# Week 1\n", encoding="utf-8")
        (folder / "exercise.md").write_text(f"---\ntitle: Exercise\nassistant: {rule}\n---\n# Exercise\n", encoding="utf-8")
        loaded.append(pm_help.load_course(folder))
        return loaded[-1]

    yield make
    for c in loaded:
        pm_help.unload_course(c.path)


def test_a_course_sets_its_rule_and_the_strictest_applies(course, tmp_path):
    from pymetrix_assistant.app import course_rule

    from pymetrix import help as pm_help

    assert course_rule() == ("", "")
    assert course("explain").assistant == "explain" and course_rule() == ("explain", "ECON 101")
    course("off")
    assert course_rule()[0] == "off"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "a.md").write_text("---\nassistant: maybe\n---\n# A\n", encoding="utf-8")
    loaded = pm_help.load_course(bad)
    try:
        assert loaded.assistant == "" and any("assistant 'maybe'" in p for p in loaded.problems)
    finally:
        pm_help.unload_course(bad)


def test_explaining_only_offers_no_proposals(session):
    from pymetrix_assistant.chat import system_for

    client = FakeClient(reply(use(1, "propose_step", command="FitEquation", settings={}, why="x"), stop="tool_use"),
                        reply(text("Think about which variables belong.")))
    answer = Conversation(client, explain=True).ask("Do exercise 3 for me", facts(session))
    assert answer.proposals == [] and "propose_step" not in {t["name"] for t in client.requests[0]["tools"]}
    assert "explain only" in client.requests[0]["system"] and "propose steps with propose_step" not in client.requests[0]["system"]
    assert client.requests[-1]["messages"][-1]["content"][0]["is_error"]
    assert system_for(explain=True) != system_for()  # the instructions really change


def test_the_panel_follows_the_course(qapp, panel_with, session, course, monkeypatch):
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import set_project_mode

    set_project_mode(session, "metadata_only")
    course("explain")
    client = FakeClient(reply(text("Start from the help page on OLS.")))
    panel, _ = panel_with(client)
    tab = panel.ask
    note = tab.findChild(QtWidgets.QLabel, "assistant-course")
    assert "explain only" in note.text() and "ECON 101" in note.text()
    tab.question.setPlainText("Solve it")
    tab.ask_button.click()
    _wait(qapp, tab)
    qapp.processEvents()
    assert "propose_step" not in {t["name"] for t in client.requests[0]["tools"]}
    assert audit.read(session)[-1]["explain"] is True
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    panel.allow.setChecked(True)  # connections: not while the course is loaded
    assert not panel.allow.isChecked() and panel.server() is None and "ECON 101" in shown[0]


def test_a_course_can_turn_the_assistant_off(qapp, panel_with, session, course):
    from pymetrix_assistant.app import set_project_mode

    set_project_mode(session, "metadata_only")
    course("off")
    client = FakeClient()
    panel, _ = panel_with(client)
    tab = panel.ask
    assert not tab.ask_button.isEnabled() and not tab.question.isEnabled()
    tab.question.setPlainText("Anything")
    tab.ask_button.click()
    assert client.requests == []


def test_a_long_proposal_is_shown_whole_in_a_short_panel(qapp, panel_with, session):
    """A card's words wrap onto as many lines as they need, and the cards scroll, however short the panel: a wrapped
    label squeezed below its height cut the reason off (seen with a local model's longer reasons)."""
    from pymetrix_assistant.app import set_project_mode

    set_project_mode(session, "metadata_only")
    why = ("Regress consumption on income with a lag of consumption, because the residuals of the static fit are "
           "autocorrelated and a partial adjustment model is the usual next step; compare the long-run coefficient "
           "with the static one before going further, and check the Breusch-Godfrey test again afterwards.")
    client = FakeClient(*[reply(use(i, "propose_step", command="FitEquation",
                                    settings={"formula": "c ~ y + c[-1]", "data": "data", "name": f"fit{i}"}, why=why),
                                stop="tool_use") for i in range(3)], reply(text("Three ways to go on.")))
    panel, _ = panel_with(client)
    panel.resize(360, 520)
    panel.show()
    tab = panel.ask
    tab.question.setPlainText("What next?")
    tab.ask_button.click()
    _wait(qapp, tab)
    for _ in range(5):
        qapp.processEvents()
    cards = tab.proposals.findChildren(QtWidgets.QFrame, "assistant-proposal")
    assert len(cards) == 3
    for card in cards:
        label = card.findChild(QtWidgets.QLabel, "assistant-proposal-why")
        needed = label.heightForWidth(label.width())
        assert label.height() >= needed, (label.height(), needed)  # every line of the reason, not cut off
        assert why[-30:] in label.text()
    panel.close()


def test_continue_where_i_left_off(qapp, panel_with, session, tmp_path):
    """A reopened project offers its last conversation: back on screen, and given to the model with the next
    question; the record numbers each conversation, and New conversation starts another."""
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import create_panel, set_project_mode

    from pymetrix.commands import load_project, save_project
    from pymetrix.extension import AppInterface

    set_project_mode(session, "metadata_only")
    first = FakeClient(reply(text("Start with OLS.")), reply(text("Then test the residuals.")))
    panel, _ = panel_with(first)
    tab = panel.ask
    assert not tab.resume.isVisibleTo(tab)  # nothing to continue yet
    for q in ("Where do I start?", "And then?"):
        tab.question.setPlainText(q)
        tab.ask_button.click()
        _wait(qapp, tab)
        qapp.processEvents()
    assert [e["conversation"] for e in audit.read(session)] == [1, 1]

    reopened = load_project(save_project(session, tmp_path / "p.pymx"))
    second = FakeClient(reply(text("Now forecast.")))
    app = AppInterface(lambda: reopened, lambda: None, lambda *a: None, reopened.run, lambda t: None)
    panel2 = create_panel(app, client_factory=lambda: second)
    tab2 = panel2.ask
    assert tab2.resume.isVisibleTo(tab2) and "2 questions" in tab2.resume.text()
    tab2.resume.click()
    assert "Start with OLS." in tab2.transcript.toPlainText() and not tab2.resume.isVisibleTo(tab2)
    tab2.question.setPlainText("What next?")
    tab2.ask_button.click()
    _wait(qapp, tab2)
    qapp.processEvents()
    sent = second.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user", "assistant", "user"]
    assert "Where do I start?" in repr(sent[0]) and "Then test the residuals." in repr(sent[3])
    assert audit.read(reopened)[-1]["conversation"] == 1  # the same conversation, continued

    tab2.new_conversation.click()
    third = FakeClient(reply(text("Fresh start.")))
    second.responses.extend(third.responses)
    tab2.question.setPlainText("Hello")
    tab2.ask_button.click()
    _wait(qapp, tab2)
    qapp.processEvents()
    assert audit.read(reopened)[-1]["conversation"] == 2
    assert len(second.requests[-1]["messages"]) == 1  # nothing carried over


def test_a_conversation_goes_back_only_where_it_came_from(session):
    """A local model's answers from full context are not offered for Anthropic in metadata only, nor the reverse."""
    from pymetrix_assistant import audit

    audit.append(session, "asked", question=1, provider="local", mode="local_model", full=True, text="q",
                 reply="The coefficient is 0.75", conversation=1)
    assert audit.resumable(audit.read(session), "anthropic", "metadata_only", False) == (None, [])
    assert audit.resumable(audit.read(session), "local", "local_model", False) == (None, [])
    number, entries = audit.resumable(audit.read(session), "local", "local_model", True)
    assert number == 1 and [e["reply"] for e in entries] == ["The coefficient is 0.75"]
