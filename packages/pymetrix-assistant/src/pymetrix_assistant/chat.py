"""Ask Claude inside the app (assistant stage 2): a question goes to Anthropic's API with the project described in
metadata only (``pymetrix.disclosure``), and the answer comes back as text and **proposals**: steps the user opens in
their dialogs, filled in, and runs with OK, or skips. The model runs nothing and sees no result: its tools only list
pymetrix's commands, describe their settings, and propose a step.

    conversation = Conversation(client_from_keychain())
    reply = conversation.ask("Fit a consumption function", to_text(describe_session(session)))
    reply.text, reply.proposals      # Proposal(command_cls, command, why)

``client`` is anything with Anthropic's ``beta.messages.create``; tests give a fake one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any

__all__ = ["KEYRING_SERVICE", "KEYRING_USER", "MODEL", "Conversation", "Proposal", "Reply", "api_key",
           "client_from_keychain", "forget_api_key", "run_tool", "save_api_key", "show_text"]

MODEL = "claude-opus-5-5"
BETAS = ["server-side-fallback-2026-07-01"]  # a declined request is re-run on Anthropic's recommended fallback model
MAX_TOKENS = 16_000
MAX_ROUNDS = 8  # tool rounds per question
KEYRING_SERVICE = "pymetrix-assistant"
KEYRING_USER = "anthropic"
REFUSED = frozenset({"RunCode"})  # never proposed: Python is the user's to type

SYSTEM = """You help an economist work in PyMetrix, an econometrics and forecasting app. You cannot run anything \
yourself. You propose steps with propose_step: each opens the step's dialog in the app, filled in, and the user runs \
it by pressing OK, changes it, or skips it.

You are told the project in metadata only: the names, kinds, sizes and periods of its objects, the settings of its \
results that are plain identifiers, and the steps taken, by dialog. You are not told data values, estimates, \
formulas, labels, notes or file paths, and you do not see what a step produces. When you need one of those, ask the \
user.

Find the command for a step with list_commands and read its settings with describe_command before proposing it; \
write settings as the dialog takes them (formulas such as "c ~ y + c[-1]", lists as "ly, lc"). Propose one step at a \
time unless the user asks for several. Keep replies short, plain and specific to the project."""

SYSTEM_FULL = SYSTEM.replace(
    "You are told the project in metadata only: the names, kinds, sizes and periods of its objects, the settings of \
its results that are plain identifiers, and the steps taken, by dialog. You are not told data values, estimates, \
formulas, labels, notes or file paths, and you do not see what a step produces. When you need one of those, ask the \
user.",
    "You are told the project's objects (names, kinds, sizes, periods) and its steps with their code. To read an \
object's results (estimates, tests, data), call show_object: the user sees what would be sent and may decline, so \
ask only for what you need, and work from what you are shown rather than guessing numbers.")

TOOLS: list[dict[str, Any]] = [
    {"name": "list_commands",
     "description": "PyMetrix's commands (the app's dialogs), grouped by menu, one line each. menu: File, Data, "
                    "Estimate, Forecast, Test, Model or Other; empty for all.",
     "input_schema": {"type": "object", "properties": {"menu": {"type": "string"}}, "additionalProperties": False}},
    {"name": "describe_command",
     "description": "A command's settings (its dialog's fields): kind, default, whether required, help, and the JSON "
                    "Schema. Read it before propose_step.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                      "additionalProperties": False}},
    {"name": "propose_step",
     "description": "Propose one step: the app shows it with your reason, and the user opens it in its dialog filled "
                    "in with these settings and runs it, or skips it. Nothing runs without the user. Settings not "
                    "given keep the dialog's defaults.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string", "description": "The command's name, from list_commands"},
                                     "settings": {"type": "object", "description": "Setting name -> value"},
                                     "why": {"type": "string", "description": "One sentence for the user"}},
                      "required": ["command", "settings", "why"], "additionalProperties": False}},
]


SHOW_TOOL: dict[str, Any] = {
    "name": "show_object",
    "description": "One tab of an object in the project, as a table (an estimate's coefficients, a test's results, a "
                   "dataset's first and last rows). The user is shown it and decides whether to send it. section: the "
                   "tab's name; empty for the first tab (the answer lists the tabs).",
    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "section": {"type": "string"}},
                     "required": ["name"], "additionalProperties": False}}
TOOLS_FULL = [*TOOLS, SHOW_TOOL]

EXPLAIN = """

A course is loaded that asks you to explain only. The user is a student and these may be exercises: do not propose steps, do not give the settings, formula or code to type, and do not do the exercise for them. Explain the methods, what a dialog does and which choices matter and why, what results mean, and how they could find the answer themselves (which menu, which help page). Asking them questions that lead them on is good."""


def system_for(full: bool = False, explain: bool = False) -> str:
    """The model's instructions for the project's mode, and for a course that asks for explaining only."""
    text = SYSTEM_FULL if full else SYSTEM
    if explain:
        text = text.replace("You propose steps with propose_step: each opens the step's dialog in the app, filled in, "
                            "and the user runs it by pressing OK, changes it, or skips it.",
                            "You do not propose steps here.") + EXPLAIN
    return text


def tools_for(full: bool = False, explain: bool = False) -> list[dict[str, Any]]:
    """The model's tools: no propose_step while a course asks for explaining only; show_object in full_context."""
    tools = TOOLS_FULL if full else TOOLS
    return [t for t in tools if not (explain and t["name"] == "propose_step")]


# --- the key, in the system keychain -------------------------------------------------------------------------------

def api_key() -> str | None:
    """The Anthropic API key saved in the system keychain (Windows Credential Manager, macOS Keychain, the Secret
    Service), if any."""
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or None
    except Exception:  # noqa: BLE001 - no keychain here: no key
        return None


def save_api_key(key: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key.strip())


def forget_api_key() -> None:
    import keyring

    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except keyring.errors.PasswordDeleteError:
        pass


def client_from_keychain() -> Any:
    """Anthropic's client, with the keychain's key (else ``ANTHROPIC_API_KEY``, as Anthropic's tools read it)."""
    import anthropic

    key = api_key()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


# --- the tools -----------------------------------------------------------------------------------------------------

@dataclass
class Proposal:
    """A step proposed: its command class, the command filled in (the dialog's ``initial``) and why."""

    command_cls: type
    command: Any
    why: str

    @property
    def name(self) -> str:
        return self.command_cls.__name__

    @property
    def code(self) -> str:
        try:
            return self.command.to_code()
        except Exception:  # noqa: BLE001 - a command that cannot say its code yet is shown by name
            return self.name


@dataclass
class Reply:
    text: str
    proposals: list[Proposal] = field(default_factory=list)


class ToolError(Exception):
    """Told to the model as an error result, so it can correct itself."""


def _commands() -> dict[str, Any]:
    from pymetrix.commands import COMMANDS

    return COMMANDS


def _info(name: str) -> Any:
    import difflib

    commands = _commands()
    if name in commands:
        return commands[name]
    close = difflib.get_close_matches(name, list(commands), n=3)
    raise ToolError(f"no command {name!r}" + (f"; did you mean {', '.join(close)}?" if close else "; see list_commands"))


def list_commands(menu: str = "") -> str:
    lines: dict[str, list[str]] = {}
    for name, info in sorted(_commands().items()):
        if name in REFUSED:
            continue
        top = (info.menu or "Other/").split("/")[0]
        if menu and top.lower() != menu.lower():
            continue
        doc = " ".join((info.cls.__doc__ or "").strip().split("\n\n")[0].split())
        lines.setdefault(top, []).append(f"- {name}: {doc}")
    if not lines:
        raise ToolError(f"no menu {menu!r}; the menus are File, Data, Estimate, Forecast, Test, Model and Other")
    return "\n\n".join(f"## {top}\n" + "\n".join(items) for top, items in lines.items())


def describe_command(name: str) -> str:
    from pymetrix.commands import form_spec
    from pymetrix.schema import json_schema

    info = _info(name)
    rows = []
    for s in form_spec(info.cls):
        choices = f" Choices: {', '.join(map(str, s.choices))}." if s.choices else ""
        default = "" if s.default is None else f" Default {s.default!r}."
        rows.append(f"- {s.name} ({s.kind}{', required' if s.required else ''}):{default} {s.help}{choices}")
    schema = json.dumps(json_schema(info.cls), separators=(",", ":"))
    return "\n".join([f"# {name}", (info.cls.__doc__ or "").strip(), *rows, f"JSON Schema: {schema}"])


def _as_field(value: Any, kind: Any) -> Any:
    """A JSON value as the dialog field takes it: the forms take most settings as text ("2", "ly, lc")."""
    if kind is str:
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    elif kind is bool and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    elif kind in (int, float) and isinstance(value, str):
        try:
            return kind(value)
        except ValueError:
            return value
    return value


def propose(command: str, settings: dict[str, Any] | None, why: str) -> Proposal:
    """The step filled in, or ``ToolError`` saying what to change."""
    from pymetrix.commands import form_spec

    if command in REFUSED:
        raise ToolError(f"{command} is not proposed: running Python is the user's to type")
    info = _info(command)
    settings = dict(settings or {})
    known = {f.name for f in fields(info.cls) if f.init}
    unknown = sorted(set(settings) - known)
    if unknown:
        raise ToolError(f"{command} has no setting {', '.join(map(repr, unknown))}; its settings are {', '.join(sorted(known))}")
    types = {s.name: s.type for s in form_spec(info.cls)}
    try:
        filled = info.cls(**{k: _as_field(v, types.get(k)) for k, v in settings.items()})
    except TypeError as exc:
        raise ToolError(f"{command}: {exc}; see describe_command({command!r})") from exc
    except Exception as exc:
        raise ToolError(f"{command}: {exc}") from exc
    return Proposal(info.cls, filled, str(why or "").strip())


def run_tool(name: str, args: Any, proposals: list[Proposal], show: Any = None, explain: bool = False) -> tuple[str, bool]:
    """One of the model's tools (``TOOLS``), whichever model asked: its answer, and whether it is an error the model
    should correct. A proposal is added to ``proposals``. ``show(name, section)`` answers ``show_object`` in
    full_context (it asks the user first, and raises ``ToolError`` when they decline); without it the tool is refused.
    With ``explain`` (a course asks for explaining only), ``propose_step`` is refused."""
    try:
        if isinstance(args, str):  # some models send the arguments as JSON text
            args = json.loads(args or "{}")
        args = dict(args or {})
        if name == "list_commands":
            return list_commands(str(args.get("menu") or "")), False
        if name == "describe_command":
            return describe_command(str(args.get("name") or "")), False
        if name == "show_object":
            if show is None:
                raise ToolError("show_object is not available: this project sends metadata only")
            return show(str(args.get("name") or ""), str(args.get("section") or "")), False
        if name == "propose_step":
            if explain:
                raise ToolError("propose_step is not available: a course asks for explaining only")
            settings = args.get("settings")
            if isinstance(settings, str):
                settings = json.loads(settings or "{}")
            p = propose(str(args.get("command") or ""), settings, str(args.get("why") or ""))
            proposals.append(p)
            return f"Shown to the user as a proposal ({p.name}); they will run it, change it or skip it.", False
        raise ToolError(f"no tool {name!r}")
    except (ToolError, ValueError, TypeError) as exc:
        return str(exc), True


# --- the conversation ----------------------------------------------------------------------------------------------

class Conversation:
    """One conversation with the model about the project. Each question goes with the project as the model may be
    told it now (``to_text(describe_session(session))``, made on the app's main thread); ``ask`` may run on another
    thread: its tools read only pymetrix's command catalogue."""

    provider = "anthropic"

    def __init__(self, client: Any, model: str = MODEL, full: bool = False, show: Any = None,
                 explain: bool = False) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.fallback_used = False
        self.full = full  # full_context: the show_object tool, through ``show`` (which asks the user first)
        self.show = show
        self.explain = explain  # a course asks for explaining only: no proposals

    def _create(self) -> Any:
        return self.client.beta.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, system=system_for(self.full, self.explain),
            tools=tools_for(self.full, self.explain), messages=self.messages,
            thinking={"type": "adaptive"}, betas=BETAS, fallbacks="default")

    def _tool(self, block: Any, proposals: list[Proposal]) -> dict[str, Any]:
        content, error = run_tool(block.name, getattr(block, "input", None), proposals, self.show if self.full else None,
                                  self.explain)
        out: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id, "content": content}
        if error:
            out["is_error"] = True
        return out

    def seed(self, turns: list[tuple[str, str]]) -> None:
        """Start from earlier questions and answers (a conversation continued from the project's record)."""
        for question, answer in turns:
            self.messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def ask(self, question: str, facts: str) -> Reply:
        """Ask ``question``; ``facts`` is the project now, in metadata only. The model's text and its proposals."""
        saved = list(self.messages)
        turn = [{"type": "text", "text": f"<project {'full_context' if self.full else 'metadata_only'}>\n{facts}\n</project>"},
                {"type": "text", "text": question}]
        if self.messages and self.messages[-1]["role"] == "user":  # the last question stopped mid-way: one user turn
            self.messages[-1] = {"role": "user", "content": [*self.messages[-1]["content"], *turn]}
        else:
            self.messages.append({"role": "user", "content": turn})
        proposals: list[Proposal] = []
        texts: list[str] = []
        for _ in range(MAX_ROUNDS):
            response = self._create()
            iterations = getattr(getattr(response, "usage", None), "iterations", None) or []
            if any(getattr(i, "type", None) == "fallback_message" for i in iterations):
                self.fallback_used = True
            if response.stop_reason == "refusal":
                self.messages[:] = saved  # the question is taken back: the conversation stays as it was
                return Reply("The model declined to answer that. Try asking another way.", proposals)
            self.messages.append({"role": "assistant", "content": response.content})
            texts += [b.text for b in response.content if getattr(b, "type", None) == "text" and b.text.strip()]
            uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason == "max_tokens":
                texts.append("(The answer was cut short.)")
                if uses:  # an unanswered tool call cannot stay in the conversation
                    self.messages[:] = saved
                break
            if response.stop_reason != "tool_use" or not uses:
                break
            self.messages.append({"role": "user", "content": [self._tool(b, proposals) for b in uses]})
        else:
            texts.append("(Stopped after several rounds of looking things up.)")
        return Reply("\n\n".join(texts).strip(), proposals)


def show_text(obj_name: str, obj: Any, section: str = "", max_rows: int = 40, keep: int = 15) -> str:
    """One tab of ``obj`` as markdown, for ``show_object``: long tables keep their first and last ``keep`` rows; a
    chart is named, not sent."""
    from dataclasses import replace

    from pymetrix.output import Table
    from pymetrix.views import view_sections

    try:
        sections = view_sections(obj)
    except Exception as exc:
        raise ToolError(f"{obj_name} has nothing to show ({type(obj).__name__})") from exc
    if not sections:
        raise ToolError(f"{obj_name} has nothing to show ({type(obj).__name__})")
    titles = [t for t, _ in sections]
    chosen = next(((t, v) for t, v in sections if t.lower() == section.lower()), None) if section else sections[0]
    if chosen is None:
        raise ToolError(f"{obj_name} has no tab {section!r}; its tabs are {', '.join(titles)}")
    title, value = chosen
    head = f"{obj_name} ({type(obj).__name__}); tabs: {', '.join(titles)}. Showing {title}."
    if not isinstance(value, Table):
        return f"{head}\n\nThis tab is a chart: it is not sent. Ask for a table tab."
    rows, note = list(value.rows), ""
    if len(rows) > max_rows:
        kept = list(range(keep)) + list(range(len(rows) - keep, len(rows)))
        value = replace(value, rows=[rows[i] for i in kept],
                        index=[value.index[i] for i in kept] if value.index is not None else None)
        note = f"\n\n(The first and last {keep} of {len(rows)} rows.)"
    return f"{head}\n\n{value.to_markdown()}{note}"
