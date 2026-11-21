"""Ask a model on this computer (the ``local_model`` mode): an OpenAI-compatible server such as Ollama or LM Studio,
for those without an Anthropic account, or who want nothing to leave the machine.

    conversation = LocalConversation("http://localhost:11434/v1", "qwen3:8b")
    reply = conversation.ask("Fit a consumption function", to_text(describe_session(session)))

The same tools, proposals and record as Ask with Claude (``chat``): the model looks up commands and proposes steps,
and runs nothing. The address must be this computer's (``localhost``, ``127.0.0.1``, ``::1``), checked before every
request; requests go straight to it, never through a proxy, and a redirect elsewhere is refused. The model must
support tool calls (most current open models do: Qwen 3, Llama 3.1 and later, Mistral).
"""

from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .chat import MAX_ROUNDS, Reply, run_tool, system_for, tools_for

__all__ = ["DEFAULT_URL", "LocalConversation", "LocalModelError", "check_local", "list_models"]

DEFAULT_URL = "http://localhost:11434/v1"  # Ollama's; LM Studio's is http://localhost:1234/v1
TIMEOUT = 300  # seconds for an answer: a local model on a laptop can be slow
MAX_TOKENS = 8_000


class LocalModelError(RuntimeError):
    """The local model's server is not this computer's, cannot be reached, or answered in a way not understood."""


def check_local(url: str) -> str:
    """``url`` without a trailing slash, if it is an http(s) address on this computer; else ``LocalModelError``."""
    text = str(url or "").strip().rstrip("/")
    parts = urllib.parse.urlsplit(text)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        raise LocalModelError(f"{text or 'the address'}: give the server's address, such as {DEFAULT_URL}")
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback or parts.username or parts.password:
        raise LocalModelError(f"{text}: a local model must be on this computer (localhost, 127.0.0.1 or ::1)")
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise LocalModelError("the local model's server redirected the request elsewhere: refused")


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())  # no proxy: straight to it


def _request(url: str, body: dict[str, Any] | None = None, timeout: float = TIMEOUT) -> dict[str, Any]:
    """GET (no ``body``) or POST JSON to ``url`` on this computer; the JSON answer."""
    base = urllib.parse.urlsplit(url)
    check_local(f"{base.scheme}://{base.netloc}")
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LocalModelError(f"the local model's server answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LocalModelError(f"no model server at {base.scheme}://{base.netloc}: start Ollama or LM Studio "
                              f"({exc.reason})") from exc
    except (TimeoutError, OSError) as exc:
        raise LocalModelError(f"the local model did not answer: {exc}") from exc
    except ValueError as exc:
        raise LocalModelError("the local model's server did not answer in JSON") from exc


def list_models(url: str, request: Callable[..., dict[str, Any]] | None = None) -> list[str]:
    """The models the server at ``url`` offers (``GET /models``)."""
    answer = (request or _request)(check_local(url) + "/models", None, 10)
    return sorted(str(m.get("id")) for m in answer.get("data", []) if isinstance(m, dict) and m.get("id"))


def _functions(explain: bool, full: bool = False) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools_for(full, explain)]


class LocalConversation:
    """One conversation with a model on this computer; ``ask`` as ``chat.Conversation``'s. ``request(url, body,
    timeout)`` sends a request (tests give a fake one)."""

    provider = "local"
    fallback_used = False

    def __init__(self, url: str, model: str, request: Callable[..., dict[str, Any]] | None = None,
                 explain: bool = False, full: bool = False, show: Callable[[str, str], str] | None = None) -> None:
        self.explain = explain  # a course asks for explaining only: no proposals
        self.full = full  # full context, on this computer: the steps' code, and show_object through ``show``
        self.show = show
        self.url = check_local(url)
        if not str(model or "").strip():
            raise LocalModelError("choose the local model to ask (Find models lists those the server has)")
        self.model = str(model).strip()
        self._request = request or _request
        self.messages: list[dict[str, Any]] = []

    def seed(self, turns: list[tuple[str, str]]) -> None:
        """Start from earlier questions and answers (a conversation continued from the project's record)."""
        for question, answer in turns:
            self.messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    def ask(self, question: str, facts: str) -> Reply:
        saved = list(self.messages)
        self.messages.append({"role": "user", "content": f"<project {'full_context' if self.full else 'metadata_only'}>\n{facts}\n</project>\n\n{question}"})
        proposals: list[Any] = []
        texts: list[str] = []
        try:
            for _ in range(MAX_ROUNDS):
                answer = self._request(self.url + "/chat/completions", {
                    "model": self.model, "messages": [{"role": "system", "content": system_for(self.full, self.explain)},
                                                    *self.messages],
                    "tools": _functions(self.explain, self.full), "max_tokens": MAX_TOKENS, "stream": False}, TIMEOUT)
                try:
                    choice = answer["choices"][0]
                    message = choice["message"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise LocalModelError("the local model's answer was not understood") from exc
                calls = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
                content = message.get("content") or ""
                kept: dict[str, Any] = {"role": "assistant", "content": content}
                if calls:
                    kept["tool_calls"] = calls
                self.messages.append(kept)
                if content.strip():
                    texts.append(content.strip())
                if choice.get("finish_reason") == "length":
                    texts.append("(The answer was cut short.)")
                    if calls:
                        self.messages[:] = saved
                    break
                if not calls:
                    break
                for i, call in enumerate(calls):
                    fn = call.get("function") or {}
                    result, error = run_tool(str(fn.get("name") or ""), fn.get("arguments"), proposals,
                                             self.show if self.full else None, self.explain)
                    self.messages.append({"role": "tool", "tool_call_id": str(call.get("id") or f"call_{i}"),
                                          "content": ("Error: " + result) if error else result})
            else:
                texts.append("(Stopped after several rounds of looking things up.)")
        except Exception:
            self.messages[:] = saved  # the question is taken back
            raise
        return Reply("\n\n".join(texts).strip(), proposals)
