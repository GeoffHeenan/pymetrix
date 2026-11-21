"""Serving the app's session over MCP: the server on this computer only, its address, and each tool call run on the
app's main thread (the session and the window live there)."""

from __future__ import annotations

import asyncio
import secrets
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["AppServer", "MainThread", "serve"]

HOST = "127.0.0.1"  # this computer only


class MainThread:
    """Runs a function on the thread that made it (the app's main thread) and waits for its result. Called on that
    thread, it runs the function at once. Made on the main thread, after the app's QApplication."""

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Relay(QObject):
            asked = Signal(object)

        self._thread = threading.current_thread()
        self._relay = _Relay()
        self._relay.asked.connect(self._answer)  # a queued connection when emitted from another thread

    @staticmethod
    def _answer(job: Any) -> None:
        fn, done, box = job
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - raised again on the asking thread
            box["error"] = exc
        finally:
            done.set()

    def __call__(self, fn: Callable[[], Any]) -> Any:
        if threading.current_thread() is self._thread:
            return fn()
        done, box = threading.Event(), {}
        self._relay.asked.emit((fn, done, box))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")


_REFUSED = b"Not Found"
_READ_AT_MOST = 1 << 20  # of a refused request's body; past that the connection is closed anyway


def _only_at(path: str, web: Any) -> Any:
    """``web`` (an ASGI app) at ``path`` only: a request for any other address is answered 404 Not Found, the same
    words as before and nothing else, once its body has been read. The 404 used to go before the body was read, and a
    socket closed with data unread is reset rather than closed; on Windows the reset discards the answer the client
    has not read yet, and urllib, which sends a request's body after its headers, saw WinError 10053 or 10054 instead
    of the 404 (in a few requests in a hundred)."""

    async def guarded(scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["path"] == path:
            await web(scope, receive, send)
            return
        read = 0
        while read <= _READ_AT_MOST:
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                return
            read += len(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 404,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                (b"content-length", str(len(_REFUSED)).encode()), (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": _REFUSED})

    return guarded


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


@dataclass
class AppServer:
    """A running server: its address, and ``stop()``."""

    url: str
    port: int
    steps: list[str] = field(default_factory=list)  # the steps the assistant ran, as recorded
    _server: Any = field(default=None, repr=False)
    _thread: Any = field(default=None, repr=False)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def claude_code_command(self) -> str:
        """The command that adds this server to Claude Code."""
        return f"claude mcp add --transport http pymetrix {self.url}"


def serve(app: Any, port: int = 0, root: Path | None = None, call: Callable[[Callable[[], Any]], Any] | None = None,
          on_step: Callable[[str], None] | None = None) -> AppServer:
    """Serve pymetrix's MCP tools over the app's session (``app``: an ``AppInterface``) on this computer, at
    ``http://127.0.0.1:<port>/mcp/<secret>``. Steps run through ``app.run``; every tool call runs through ``call``
    (default: on the thread calling ``serve``, the app's main thread). ``root``: the folder files may be read from
    (default: the current folder). ``on_step(code)`` is told each step the assistant ran."""
    import uvicorn

    from pymetrix.extension import MCPPolicy as Policy
    from pymetrix.extension import build_mcp_server

    call = call or MainThread()
    found: AppServer | None = None

    def run(command: Any) -> Any:
        before = len(app.session.history)
        result = app.run(command)
        for record in app.session.history[before:]:
            if found is not None:
                found.steps.append(record.code)
            if on_step is not None:
                on_step(record.code)
        return result

    server = build_mcp_server(session_of=lambda: app.session, run=run, call=call,
                              policy=Policy(root=Path(root) if root else Path.cwd()))
    port = port or _free_port()
    path = f"/mcp/{secrets.token_urlsafe(16)}"  # a secret in the address: other programs on the computer cannot guess it
    web = _only_at(path, server.streamable_http_app(streamable_http_path=path, host=HOST))
    # log_config=None: uvicorn's own logging set-up asks whether stdout is a terminal, and an app started without a
    # console (pythonw, the desktop shortcut) has no stdout, so it fails; the app's logging is used instead
    config = uvicorn.Config(web, host=HOST, port=port, log_level="warning", lifespan="on", log_config=None)
    runner = uvicorn.Server(config)
    runner.install_signal_handlers = lambda: None  # not the main thread: the app keeps its own handlers
    thread = threading.Thread(target=lambda: asyncio.run(runner.serve()), name="pymetrix assistant", daemon=True)
    found = AppServer(f"http://{HOST}:{port}{path}", port, _server=runner, _thread=thread)
    thread.start()
    for _ in range(200):  # until it listens (up to 10 s)
        if runner.started:
            break
        threading.Event().wait(0.05)
    if not runner.started:
        found.stop()
        raise RuntimeError(f"the assistant's server did not start on port {port}")
    return found
