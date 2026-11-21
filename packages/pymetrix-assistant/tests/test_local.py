"""Ask a model on this computer (local_model): only a loopback address, straight to it (no proxy, no redirect), the
same tools and proposals as with Claude, and the panel's choice of server and model. A small HTTP server on
127.0.0.1 plays an OpenAI-compatible one."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pymetrix_assistant.local import (
    LocalConversation,
    LocalModelError,
    check_local,
    list_models,
)

from pymetrix.commands import FitEquation, LoadExample, Session


@pytest.mark.parametrize("url", ["http://localhost:11434/v1", "http://127.0.0.1:1234/v1/", "http://[::1]:8080/v1",
                                 "https://127.0.0.2/v1"])
def test_this_computer_is_accepted(url):
    assert check_local(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["http://example.com/v1", "http://192.168.1.10:11434/v1", "http://localhost.evil.com/v1",
                                 "ftp://localhost/v1", "localhost:11434", "", "http://user:pw@localhost/v1",
                                 "http://10.0.0.1/v1"])
def test_anywhere_else_is_refused(url):
    with pytest.raises(LocalModelError):
        check_local(url)
    with pytest.raises(LocalModelError):
        LocalConversation(url, "m")


class FakeServer:
    """An OpenAI-compatible server on 127.0.0.1: answers /models, and /chat/completions from a script."""

    def __init__(self, answers, redirect=False):
        self.answers, self.requests = list(answers), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body, code=200):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", "http://example.com/v1/models")
                    self.end_headers()
                    return
                self._send({"data": [{"id": "qwen3:8b"}, {"id": "llama3.1:8b"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                self._send(outer.answers.pop(0))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def message(content="", calls=(), finish="stop"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                             for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else finish}]}


@pytest.fixture
def server(request):
    made = []

    def make(answers, **kw):
        made.append(FakeServer(answers, **kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


def test_a_local_model_looks_up_and_proposes(server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")  # ignored: the request goes straight to this computer
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:9")
    srv = server([
        message(calls=[("describe_command", {"name": "FitEquation"})]),
        message("Try this.", calls=[("propose_step", {"command": "FitEquation", "settings": {"formula": "c ~ y", "data": "data"},
                                                      "why": "A start."})]),
        message("Open it to run it."),
    ])
    assert list_models(srv.url) == ["llama3.1:8b", "qwen3:8b"]
    conversation = LocalConversation(srv.url, "qwen3:8b")
    answer = conversation.ask("Fit something", '{"objects":[]}')
    [p] = answer.proposals
    assert p.command_cls is FitEquation and p.command.formula == "c ~ y"
    assert answer.text == "Try this.\n\nOpen it to run it."
    first = srv.requests[0]
    assert first["model"] == "qwen3:8b" and first["messages"][0]["role"] == "system"
    assert {t["function"]["name"] for t in first["tools"]} == {"list_commands", "describe_command", "propose_step"}
    assert "<project metadata_only>" in first["messages"][1]["content"]
    tool_messages = [m for m in srv.requests[-1]["messages"] if m["role"] == "tool"]
    assert "formula" in tool_messages[0]["content"] and "Shown to the user" in tool_messages[1]["content"]


def test_a_redirect_elsewhere_is_refused(server):
    srv = server([], redirect=True)
    with pytest.raises(LocalModelError, match="redirect"):
        list_models(srv.url)


def test_no_server_says_what_to_do():
    with pytest.raises(LocalModelError, match="start Ollama or LM Studio"):
        LocalConversation("http://127.0.0.1:9/v1", "m").ask("?", "{}")


def test_bad_arguments_are_told_back_and_a_failure_takes_the_question_back(server):
    srv = server([
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "propose_step", "arguments": "{not json"}}]},
            "finish_reason": "tool_calls"}]},
        message("Sorry."),
    ])
    conversation = LocalConversation(srv.url, "m")
    answer = conversation.ask("?", "{}")
    assert answer.proposals == [] and srv.requests[-1]["messages"][-1]["content"].startswith("Error:")
    kept = list(conversation.messages)
    srv.answers.append({"nonsense": True})
    with pytest.raises(LocalModelError):
        conversation.ask("again", "{}")
    assert conversation.messages == kept


QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_the_panel_asks_the_model_on_this_computer(server):
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import create_panel, set_project_mode

    from pymetrix.extension import AppInterface

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = Session()
    session.run(LoadExample())
    set_project_mode(session, "local_model")
    srv = server([message("Here.", calls=[("propose_step", {"command": "FitEquation",
                                                              "settings": {"formula": "c ~ y", "data": "data"}, "why": "x"})]),
                  message("Done.")])
    opened = []
    app = AppInterface(lambda: session, lambda: None, lambda cls, initial: opened.append(cls), session.run, lambda t: None)
    panel = create_panel(app, client_factory=lambda: pytest.fail("Anthropic is not asked in local mode"))
    tab = panel.ask
    tab.local_url.setText(srv.url)
    tab.find_models.click()
    assert [tab.local_model.itemText(i) for i in range(tab.local_model.count())] == ["llama3.1:8b", "qwen3:8b"]
    tab.local_model.setCurrentText("qwen3:8b")
    tab.question.setPlainText("Fit something")
    tab.ask_button.click()
    end = time.monotonic() + 20
    while tab.worker() is not None and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert "qwen3:8b" in tab.transcript.toPlainText() and "Done." in tab.transcript.toPlainText()
    [entry] = audit.read(session)
    assert entry["provider"] == "local" and entry["model"] == "qwen3:8b" and entry["url"] == srv.url
    [card] = tab.proposals.findChildren(QtWidgets.QFrame, "assistant-proposal")
    card.open_it.click()
    assert opened == [FitEquation]


def test_a_local_model_explains_only_when_a_course_asks(server):
    srv = server([message("Look at the residuals first.")])
    LocalConversation(srv.url, "m", explain=True).ask("Do it for me", "{}")
    sent = srv.requests[0]
    assert "propose_step" not in {t["function"]["name"] for t in sent["tools"]}
    assert "explain only" in sent["messages"][0]["content"]


def test_full_context_to_the_local_model(server, monkeypatch):
    """Ticked, a model on this computer is sent the steps' code and may read an object's tables, without being asked
    each time (nothing leaves the computer); the record keeps what was sent. Unticked, metadata only, as before."""
    import pymetrix_assistant.app as panel_module
    from pymetrix_assistant import audit
    from pymetrix_assistant.app import (
        create_panel,
        local_full_context,
        set_project_mode,
    )

    from pymetrix.extension import AppInterface

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(panel_module, "confirm_send", lambda *a: pytest.fail("not asked: it stays on this computer"))
    session = Session()
    session.run(LoadExample())
    session.run(FitEquation("c ~ y", data="data", name="fit"))
    set_project_mode(session, "local_model")
    srv = server([message(calls=[("show_object", {"name": "fit"})]), message("The slope is close to one.")])
    app = AppInterface(lambda: session, lambda: None, lambda *a: None, session.run, lambda t: None)
    panel = create_panel(app)
    tab = panel.ask
    assert tab.local_full.isVisibleTo(tab) and not tab.local_full.isChecked() and not local_full_context(session)
    tab.local_full.setChecked(True)
    assert local_full_context(session)
    tab.local_url.setText(srv.url)
    tab.local_model.setCurrentText("qwen3:8b")
    tab.question.setPlainText("Is the fit good?")
    tab.ask_button.click()
    end = time.monotonic() + 20
    while tab.worker() is not None and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    first = srv.requests[0]
    assert "show_object" in {t["function"]["name"] for t in first["tools"]}
    assert "c ~ y" in first["messages"][1]["content"] and "<project full_context>" in first["messages"][1]["content"]
    tool = next(m for m in srv.requests[-1]["messages"] if m["role"] == "tool")
    assert "coef" in tool["content"]
    shown, asked = audit.read(session)
    assert shown["kind"] == "shown" and shown["sent"] and "coef" in shown["text"]
    assert asked["full"] is True and asked["provider"] == "local"


def test_metadata_only_stays_the_local_default(server):
    srv = server([message("ok")])
    LocalConversation(srv.url, "m").ask("?", "{}")
    assert "show_object" not in {t["function"]["name"] for t in srv.requests[0]["tools"]}
