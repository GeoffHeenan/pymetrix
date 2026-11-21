"""The Assistant panel's server: pymetrix's MCP tools over the app's own session, on this computer only, behind a
secret address, with each tool call run on the main thread and each step run as the app runs it."""

import asyncio
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix_assistant import register, serve
from pymetrix_assistant.server import MainThread

from pymetrix.commands import Session
from pymetrix.extension import AppInterface

register()


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until_done(qapp, thread, seconds=60):
    """Keep the main thread's events moving (it answers the tool calls) until ``thread`` ends."""
    end = time.monotonic() + seconds
    while thread.is_alive() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert not thread.is_alive(), "timed out"


def test_main_thread_runs_calls_from_other_threads_there(qapp):
    main = MainThread()
    seen = {}

    def ask():
        seen["result"] = main(lambda: threading.current_thread() is threading.main_thread())
        try:
            main(lambda: 1 / 0)
        except ZeroDivisionError:
            seen["error"] = True

    t = threading.Thread(target=ask)
    t.start()
    _until_done(qapp, t)
    assert seen == {"result": True, "error": True}
    assert main(lambda: 42) == 42  # on the main thread: at once


@pytest.fixture
def app_session():
    s = Session()
    ran = []

    def run(command):
        ran.append(threading.current_thread() is threading.main_thread())
        return s.run(command)

    messages = []
    return s, AppInterface(lambda: s, lambda: None, lambda cls, initial: None, run, messages.append), ran


def test_an_assistant_works_in_the_apps_session(qapp, app_session):
    from mcp import Client

    s, app, ran = app_session
    told = []
    server = serve(app, on_step=told.append)
    try:
        assert server.url.startswith("http://127.0.0.1:") and len(server.url.rsplit("/", 1)[1]) >= 16
        out = {}

        def assistant():
            async def go():
                async with Client(server.url) as client:
                    names = {t.name for t in (await client.list_tools()).tools}
                    loaded = await client.call_tool("run_command", {"name": "LoadExample", "settings": {}})
                    objects = await client.call_tool("list_objects", {})
                    return names, loaded, objects

            out["result"] = asyncio.run(go())

        t = threading.Thread(target=assistant)
        t.start()
        _until_done(qapp, t)
        names, loaded, objects = out["result"]
        assert {"run_command", "list_objects", "show_object"} <= names and not loaded.is_error
        assert "data" in s.objects and s.history[-1].code.startswith("data = Dataset.example(")  # in the app's session
        assert ran == [True]  # the step ran through the app's runner, on the main thread
        assert told == [s.history[-1].code] and server.steps == told
        assert "`data`" in "".join(c.text for c in objects.content if c.type == "text")
    finally:
        server.stop()
    assert not server.running


def test_only_the_secret_address_answers(qapp, app_session):
    """Any other address is answered "404 Not Found", and nothing else: not the secret, nor that an MCP server is there.
    Each refusal reaches the client as an answer: the server used to send it before reading the request's body, which
    urllib sends after the headers, and on Windows the reset of a socket closed with data unread lost the answer now
    and then (WinError 10053 in the client). So it is asked many times, and a raw request's connection must end
    cleanly after the 404, not be reset."""
    _, app, _ = app_session
    server = serve(app)
    try:
        base, secret = server.url.rsplit("/", 1)
        wrong = "x" * len(secret)
        port_root = base.rsplit("/", 1)[0]  # http://127.0.0.1:<port>
        for address in [base, f"{base}/{wrong}", f"{port_root}/", f"{port_root}/mcp/{secret}/x"] + [base] * 50:
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(urllib.request.Request(address, data=b"{}", method="POST"), timeout=10)
            assert err.value.code == 404, address
            answer = err.value.read() + str(err.value.headers).encode()
            assert secret.encode() not in answer and b"mcp" not in answer.lower(), answer
        for body in [b"{}"] * 20 + [b"x" * 300_000] * 5:  # headers, then the body: is the connection closed or reset?
            with socket.create_connection(("127.0.0.1", server.port), timeout=10) as c:
                c.sendall(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
                          % len(body))
                c.sendall(body)
                got = b""
                while chunk := c.recv(65536):  # a reset raises ConnectionResetError here
                    got += chunk
            assert got.startswith(b"HTTP/1.1 404 ") and got.endswith(b"\r\n\r\nNot Found"), got
    finally:
        server.stop()



def test_the_server_starts_in_an_app_with_no_console(qapp, app_session, monkeypatch):
    """Started from the desktop shortcut (pythonw), the app has no stdout or stderr: uvicorn's own logging set-up
    failed there ("Unable to configure formatter 'default'")."""
    _, app, _ = app_session
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)
    server = serve(app)
    try:
        assert server.running
    finally:
        server.stop()

def test_the_panel_starts_and_stops_the_server(qapp, app_session):
    from pymetrix_assistant.app import create_panel

    _, app, _ = app_session
    panel = create_panel(app)
    panel.allow.setChecked(True)
    server = panel.server()
    assert server is not None and server.running
    assert panel.findChild(QtWidgets.QLineEdit, "assistant-address").text() == server.url
    assert server.claude_code_command() == f"claude mcp add --transport http pymetrix {server.url}"
    panel.allow.setChecked(False)
    assert panel.server() is None and not server.running
