"""An MCP server for pymetrix: an AI assistant runs pymetrix commands in a session the user can replay and keep.

The assistant (Claude Desktop, Claude Code, VS Code and other MCP clients) discovers the commands, runs them by name
with the settings their dialogs take, and reads the results as pymetrix's own tables and charts. Every step is
recorded in the session, so ``export_script`` gives the Python that reproduces it and ``save_project`` a project that
replays. See docs/notes/mcp-server.md.

Closed by default: no Python code, no file writes and no network data unless the server is started with
``--allow-code``, ``--allow-writes`` or ``--allow-network``, and every file it reads or writes must be inside its root
folder. Run it with ``pymetrix-mcp`` (``pip install pymetrix[mcp]``); a client configuration such as Claude Desktop's::

    {"mcpServers": {"pymetrix": {"command": "pymetrix-mcp", "args": ["--root", "C:/work/forecasts"]}}}
"""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .adapters import require

__all__ = ["Policy", "build_server", "main"]

#: Commands that run Python, write files, or read files (their path settings); see ``Policy``.
CODE_COMMANDS = frozenset({"RunCode"})
WRITE_PATHS = {"ExportReport": "path", "SaveRound": "folder"}
READ_PATHS = {"LoadData": "path", "OpenResult": "path", "ImportDynare": "path", "ImportIris": "path", "CompareRounds": "folder"}
NETWORK_SCHEMES = ("fred://", "worldbank://", "sdmx://", "http://", "https://", "ftp://")
MAX_ROWS = 40  # longer tables are shown as their first and last ROWS_KEPT rows
ROWS_KEPT = 15
MAX_HELP = 12_000  # characters of a help page

INSTRUCTIONS = """pymetrix is an econometrics and forecasting workbench. This server holds one pymetrix session: datasets, \
estimates, forecasts and tests persist between calls, and every step is recorded.

Work as an economist would in the PyMetrix app:
1. Load data: run_command("LoadExample", {}) loads a simulated quarterly economy as "data" (c, i, g, y); \
LoadData reads a file in the server's folder.
2. Find the command for a step with list_commands, then describe_command for its settings: they are exactly the \
settings of the app's dialog, written as the dialog takes them (formulas such as "c ~ y + c[-1]", lists as "ly, lc").
3. run_command runs it; show_object shows a result's tables and charts, one tab at a time.
4. search_help and read_help explain methods and tests.
5. export_script gives the Python that reproduces the whole session: show it to the user, who can rerun and keep it.

Prefer pymetrix's commands to describing what you would do: the user can audit and replay every command."""


@dataclass(frozen=True)
class Policy:
    """What the server may do beyond estimation: run Python code, write files, fetch network data, and where files
    may be read or written (``root`` and below)."""

    root: Path = field(default_factory=Path.cwd)
    allow_code: bool = False
    allow_writes: bool = False
    allow_network: bool = False

    def refusal(self, command: str) -> str | None:
        """Why ``command`` is disabled, or None when it may run."""
        if command in CODE_COMMANDS and not self.allow_code:
            return "running Python is off; start the server with --allow-code to allow it"
        if command in WRITE_PATHS and not self.allow_writes:
            return "writing files is off; start the server with --allow-writes to allow it"
        return None

    def path(self, value: str, writing: bool = False) -> str:
        """``value`` resolved inside the root folder (absolute), or a network address when those are allowed."""
        text = str(value).strip()
        if text.lower().startswith(NETWORK_SCHEMES):
            if writing or not self.allow_network:
                raise PermissionError(f"{text}: network data is off; start the server with --allow-network to allow it")
            return text
        root = self.root.resolve()
        target = (root / text).resolve()
        if target != root and root not in target.parents:
            raise PermissionError(f"{text}: outside the server's folder {root}; use a path inside it")
        return str(target)


def _commands() -> dict[str, Any]:
    from .commands import COMMANDS

    return COMMANDS


def _first_line(doc: str | None) -> str:
    return " ".join((doc or "").strip().split("\n\n")[0].split())


def _table_markdown(table: Any) -> str:
    rows = list(table.rows)
    note = ""
    if len(rows) > MAX_ROWS:
        keep = list(range(ROWS_KEPT)) + list(range(len(rows) - ROWS_KEPT, len(rows)))
        index = [table.index[i] for i in keep] if table.index is not None else None
        table = replace(table, rows=[rows[i] for i in keep], index=index)
        note = f"\n\n(Showing the first and last {ROWS_KEPT} of {len(rows)} rows.)"
    return table.to_markdown() + note


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


def _describe(name: str, obj: Any) -> str:
    text = repr(obj)
    kind = type(obj).__name__
    return f"- `{name}`: {kind}" + (f", {text}" if len(text) <= 120 and not text.startswith(f"{kind}(") else "")


class _Tools:
    """The server's tools, over one session and one policy (kept apart from the MCP plumbing, for testing).

    Every tool runs with standard output sent to standard error: over the stdio transport, standard output carries
    the protocol, and a stray print from a command would break it."""

    def __init__(self, session: Any, policy: Policy, session_of: Any = None, run: Any = None, call: Any = None) -> None:
        self._session = session
        self._session_of = session_of  # the current session, when it can change (the app opens another project)
        self._run = run  # runs a step; None: session.run
        self._call = call  # runs each tool call (the app: on its main thread); None: here
        self.policy = policy

    @property
    def session(self) -> Any:
        return self._session_of() if self._session_of is not None else self._session

    def _run_step(self, cmd: Any) -> None:
        if self._run is None:
            self.session.run(cmd)
        else:
            self._run(cmd)

    def __getattribute__(self, name: str) -> Any:
        attr = object.__getattribute__(self, name)
        if name.startswith("_") or not callable(attr):
            return attr
        call = object.__getattribute__(self, "_call")

        def quiet(*args: Any, **kwargs: Any) -> Any:
            import contextlib
            import sys

            def go() -> Any:
                with contextlib.redirect_stdout(sys.stderr):
                    return attr(*args, **kwargs)

            return go() if call is None else call(go)

        return quiet

    def _error(self, message: str) -> Exception:
        return require("mcp.server.mcpserver.exceptions").ToolError(message)

    def _info(self, name: str) -> Any:
        commands = _commands()
        if name not in commands:
            lowered = {n.lower(): n for n in commands}
            close = [lowered[m] for m in difflib.get_close_matches(name.lower(), list(lowered), n=3)]
            raise self._error(f"no command {name!r}" + (f"; did you mean {', '.join(close)}?" if close else "; see list_commands"))
        return commands[name]

    def list_commands(self, menu: str = "") -> str:
        groups: dict[str, list[str]] = {}
        for name, info in sorted(_commands().items()):
            top = (info.menu or "Other/").split("/")[0]
            if menu and top.lower() != menu.lower():
                continue
            refusal = self.policy.refusal(name)
            line = f"- `{name}` ({(info.menu or '').rpartition('/')[2].rstrip('. ') or 'no menu'}): {_first_line(info.cls.__doc__)}"
            groups.setdefault(top, []).append(line + (f" **Disabled:** {refusal}." if refusal else ""))
        if not groups:
            raise self._error(f"no menu {menu!r}; the menus are File, Data, Estimate, Forecast, Test, Model and Other")
        return "\n\n".join(f"## {top}\n\n" + "\n".join(lines) for top, lines in groups.items())

    def describe_command(self, name: str) -> str:
        from .commands import form_spec

        info = self._info(name)
        rows = ["| setting | kind | default | required | help |", "|---|---|---|---|---|"]
        for s in form_spec(info.cls):
            choices = f" Choices: {', '.join(map(str, s.choices))}." if s.choices else ""
            default = "" if s.default is None else repr(s.default)
            rows.append(f"| `{s.name}` | {s.kind} | {default} | {'yes' if s.required else ''} | {s.help}{choices} |")
        refusal = self.policy.refusal(name)
        from .schema import json_schema

        schema = json.dumps(json_schema(info.cls), indent=1)
        return "\n\n".join([f"# {name}" + (f" ({info.menu})" if info.menu else ""), (info.cls.__doc__ or "").strip(), "\n".join(rows),
                            *([f"**Disabled:** {refusal}."] if refusal else []),
                            f"Command `{info.id}`, schema version {info.cls.schema_version}. JSON Schema of its settings:\n\n```json\n{schema}\n```"])

    def run_command(self, name: str, settings: dict[str, Any] | None = None, client: str = "") -> str:
        from .commands import form_spec
        from .commands.events import Actor, acting

        info = self._info(name)
        refusal = self.policy.refusal(name)
        if refusal:
            raise self._error(f"{name} is disabled: {refusal}")
        settings = dict(settings or {})
        known = {f.name: f for f in fields(info.cls) if f.init}
        unknown = sorted(set(settings) - set(known))
        if unknown:
            raise self._error(f"{name} has no setting {', '.join(map(repr, unknown))}; its settings are {', '.join(known)}")
        try:
            for command, key in ((READ_PATHS, False), (WRITE_PATHS, True)):
                if name in command and settings.get(command[name]):
                    settings[command[name]] = self.policy.path(settings[command[name]], writing=key)
        except PermissionError as exc:
            raise self._error(str(exc)) from exc
        types = {s.name: s.type for s in form_spec(info.cls)}
        settings = {key: _as_field(value, types.get(key)) for key, value in settings.items()}
        try:
            cmd = info.cls(**settings)
            with acting(self.session, Actor("agent", client, "mcp")):  # observers (a lab book) see an agent's step
                self._run_step(cmd)
        except TypeError as exc:
            raise self._error(f"{name}: {exc}; see describe_command({name!r})") from exc
        except Exception as exc:  # pymetrix's own errors say what to change
            raise self._error(f"{name} failed: {exc}") from exc
        record = self.session.history[-1]
        parts = [f"Ran {name}. Recorded as:\n\n```python\n{record.code}\n```"]
        written = [n for n in record.writes if n in self.session.objects]
        if written:
            parts.append("Created or changed:\n" + "\n".join(_describe(n, self.session.objects[n]) for n in written))
            summary = self._summary(self.session.objects[written[0]])
            if summary:
                parts.append(f"{written[0]}, first tab:\n\n{summary}")
        if record.cautions:
            parts.append("Cautions: " + "; ".join(record.cautions))
        return "\n\n".join(parts)

    def _summary(self, obj: Any) -> str:
        from .output import Table
        from .views import view_sections

        try:
            sections = view_sections(obj)
        except Exception:  # an object with no window
            return ""
        title, value = next(((t, v) for t, v in sections if isinstance(v, Table)), ("", None))
        return f"**{title}**\n\n{_table_markdown(value)}" if value is not None else ""

    def list_objects(self) -> str:
        if not self.session.objects:
            return "The session is empty. Load data first, e.g. run_command(\"LoadExample\", {})."
        return "\n".join(_describe(n, o) for n, o in self.session.objects.items())

    def show_object(self, name: str, section: str = "") -> list[Any]:
        from .output import Table
        from .views import view_sections

        if name not in self.session.objects:
            raise self._error(f"no object {name!r}; the session holds {', '.join(self.session.objects) or 'nothing yet'}")
        sections = view_sections(self.session.objects[name])
        if not sections:
            return [_describe(name, self.session.objects[name])]
        titles = [t for t, _ in sections]
        chosen = next(((t, v) for t, v in sections if t.lower() == section.lower()), None) if section else sections[0]
        if chosen is None:
            raise self._error(f"{name} has no tab {section!r}; its tabs are {', '.join(titles)}")
        title, value = chosen
        header = f"`{name}` tabs: {', '.join(titles)}. Showing **{title}**."
        if isinstance(value, Table):
            return [f"{header}\n\n{_table_markdown(value)}"]
        image = require("mcp.server.mcpserver").Image(data=value.png(dpi=90), format="png")
        return [header, image]

    def export_script(self) -> str:
        return self.session.script()

    def save_project(self, path: str) -> str:
        from .commands import save_project

        if not self.policy.allow_writes:
            raise self._error("writing files is off; start the server with --allow-writes to allow it")
        try:
            target = self.policy.path(path, writing=True)
        except PermissionError as exc:
            raise self._error(str(exc)) from exc
        saved = save_project(self.session, target)
        return f"Saved the session as {saved}; open it in the PyMetrix app (File → Open project) to replay it."

    def search_help(self, text: str) -> str:
        from . import help as hp

        found = hp.search(text)[:15]
        if not found:
            return f"Nothing in the help matches {text!r}."
        return "\n".join(f"- {title} ({kind}): `{hp.link_of(obj)}`" for kind, title, obj in found)

    def read_help(self, link: str) -> str:
        from . import help as hp

        try:
            page = hp.page_for_link(link)
        except Exception as exc:
            raise self._error(f"no help page {link!r}: {exc}; find links with search_help") from exc
        return page if len(page) <= MAX_HELP else page[:MAX_HELP] + "\n\n(The page continues; this is its first part.)"


def _client_name(ctx: Any) -> str:
    """The MCP client's name and version as it gave them when connecting ("claude-ai 0.1.0"), or blank."""
    try:
        info = ctx.session.client_params.client_info
        return " ".join(part for part in (info.name, info.version) if part)
    except Exception:  # noqa: BLE001 - no context, or a client that did not say: unknown
        return ""


def build_server(session: Any = None, policy: Policy | None = None, *, session_of: Any = None, run: Any = None,
                 call: Any = None) -> Any:
    """The MCP server over ``session`` (a new ``Session`` by default), allowed what ``policy`` allows.

    For serving a session that lives elsewhere, such as the desktop app's (extension API 1.8, ``build_mcp_server``):
    ``session_of()`` gives the current session each time (the app opens other projects); ``run(command)`` runs a step
    (the app's own runner: in the background, undoable, shown in History); ``call(fn)`` runs each tool call and
    returns its result (the app: on its main thread, since the session and the window live there)."""
    from ._version import __version__
    from .commands import Session

    server_module = require("mcp.server.mcpserver")
    Context = server_module.Context  # the SDK's class, for run_command's signature
    tools = _Tools(session if session is not None or session_of is not None else Session(), policy or Policy(),
                   session_of=session_of, run=run, call=call)
    server = server_module.MCPServer("pymetrix", title="pymetrix", version=__version__, instructions=INSTRUCTIONS)
    server.session_tools = tools  # for tests and embedding

    @server.tool(structured_output=False)
    def list_commands(menu: str = "") -> str:
        """The pymetrix commands (the app's dialogs), by menu, with what each does. ``menu`` (File, Data, Estimate,
        Forecast, Test, Model, Other) lists one menu only."""
        return tools.list_commands(menu)

    @server.tool(structured_output=False)
    def describe_command(name: str) -> str:
        """A command's settings (as its dialog has them): each one's kind, default, whether it is required, and help."""
        return tools.describe_command(name)

    def run_command(name: str, settings: dict[str, Any] | None = None, ctx: Context | None = None) -> str:
        """Run a pymetrix command in the session, e.g. name "FitEquation" with settings {"formula": "c ~ y + c[-1]",
        "cov": "hac"}. Returns the Python it recorded and a summary of what it made."""
        return tools.run_command(name, settings, _client_name(ctx))

    # the SDK finds the context parameter by its type, which postponed annotations would leave as a string
    run_command.__annotations__.update(settings=dict[str, Any] | None, ctx=Context | None)
    server.tool(structured_output=False)(run_command)

    @server.tool(structured_output=False)
    def list_objects() -> str:
        """The objects in the session (datasets, fits, forecasts, backtests, tests) and their types."""
        return tools.list_objects()

    @server.tool(structured_output=False)
    def show_object(name: str, section: str = "") -> list[Any]:
        """One tab of an object's window, as the app shows it: a table (Markdown) or a chart (PNG). Without ``section``,
        its first tab; the reply lists all its tabs."""
        return tools.show_object(name, section)

    @server.tool(structured_output=False)
    def export_script() -> str:
        """The whole session as a Python script that reproduces it, step by step."""
        return tools.export_script()

    @server.tool(structured_output=False)
    def save_project(path: str) -> str:
        """Save the session as a pymetrix project (.pymx) in the server's folder, to reopen and replay in the app.
        Needs the server started with --allow-writes."""
        return tools.save_project(path)

    @server.tool(structured_output=False)
    def search_help(text: str) -> str:
        """Search the pymetrix help (dialogs, methods, tests, topics, course notes); returns titles and links."""
        return tools.search_help(text)

    @server.tool(structured_output=False)
    def read_help(link: str) -> str:
        """A help page by its link from search_help, e.g. "pymetrix:commands/pymetrix:runbacktest"."""
        return tools.read_help(link)

    @server.resource("pymetrix://help", name="help", title="pymetrix help", mime_type="text/markdown")
    def help_home() -> str:
        from . import help as hp

        return hp.home_markdown("all")  # every section open: a reader that cannot click

    @server.resource("pymetrix://session/script", name="script", title="The session as Python", mime_type="text/x-python")
    def session_script() -> str:
        return tools.export_script()

    return server


def main(argv: list[str] | None = None) -> None:
    """``pymetrix-mcp``: serve one pymetrix session over MCP's standard input and output."""
    parser = argparse.ArgumentParser(prog="pymetrix-mcp", description="An MCP server for pymetrix (docs/notes/mcp-server.md).")
    parser.add_argument("--root", default=".", help="the folder the server may read and write files in (default: this one)")
    parser.add_argument("--allow-code", action="store_true", help="allow the RunCode command (runs Python)")
    parser.add_argument("--allow-writes", action="store_true", help="allow saving projects, reports and forecast rounds")
    parser.add_argument("--allow-network", action="store_true", help="allow loading data from fred://, worldbank://, sdmx:// and URLs")
    args = parser.parse_args(argv)
    policy = Policy(Path(args.root).resolve(), args.allow_code, args.allow_writes, args.allow_network)
    build_server(policy=policy).run("stdio")


if __name__ == "__main__":
    main()
