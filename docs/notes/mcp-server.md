# An MCP server for pymetrix

Design note, 18 October 2026. **Status: stage 1 built** (`pymetrix.mcp_server`, `pip install pymetrix[mcp]`).

## Why

The [note on AI assistance](ai-assistance-in-the-ui.md) adopts one principle: an AI proposes pymetrix *commands*,
which run through the same engine, history and project replay as the dialogs. The Model Context Protocol (MCP) is
now the standard way to give an assistant tools: Claude Desktop, Claude Code, VS Code and others are MCP clients.
An MCP server is therefore the obvious first form of that principle. It needs no AI inside pymetrix, and it works
with whichever assistant the user already has.

What pymetrix adds over an assistant writing ad-hoc Python:

- **The steps are recorded.** Every step is a recorded command, so the session exports as a script and saves as a
  project that replays. The user can check, rerun and keep exactly what the assistant did.
- **Results are summarised.** Results come back as pymetrix's own summary tables and charts, so the assistant
  reads what an economist would read.
- **Data stays local.** The server runs on the user's machine and returns only what its tools return.

## Who it is for

- **Economists who don't program:** the main audience. They use pymetrix from Claude Desktop, in their own words,
  and get a replayable project at the end.
- **Programmers:** they can write pymetrix Python already. For them the server adds structured, recorded steps.
- **Students:** usable, with the obvious caution that an assistant can do the exercises too. A later course mode
  could explain without running.

## Design

**One session per server.** The server holds one pymetrix `Session`, the same object the desktop app uses, for as
long as the client keeps it running. Objects persist between calls: data, fits, backtests.

**A few general tools, not one per dialog.** There are about 85 commands. Few clients handle that many tools well,
and their schemas would duplicate the command registry. Instead the assistant discovers commands and then runs
them by name:

| Tool | What it does |
|---|---|
| `list_commands(menu)` | The commands, by menu, with a line on each; commands the policy disables are marked |
| `describe_command(name)` | A command's settings: name, kind, default, choices and help, from `form_spec` |
| `run_command(name, settings)` | Builds the command from its settings, runs it in the session, and returns the Python it recorded and a summary of the result |
| `list_objects()` | What the session holds, with each object's type and a short description |
| `show_object(name, section)` | A result's window: the list of its tabs, and one tab as a Markdown table or a PNG chart |
| `export_script()` | The session as a Python script that reproduces it |
| `save_project(path)` | A `.pymx` project; only when writing is allowed |
| `search_help(text)`, `read_help(link)` | The in-app help: dialogs, methods, tests, topics and course notes |

- **Resources:** the help home page, any help page by link, and the session script. Clients can attach these.
- **No prompts in stage 1.** Ready-made workflows (from the course labs) are stage 2.

**Settings are what the dialogs take.** `run_command("FitEquation", {"formula": "c ~ y + c[-1]", "cov": "hac"})` is
exactly the dialog's form. Validation, errors, history, cautions about out-of-date inputs, and the generated code
are all the session's own. Nothing is reimplemented.

**Output is bounded.**
- Tables longer than 40 rows are cut to the first and last 15 rows, with a note saying so.
- Charts are PNG images at a modest resolution.
- `show_object` returns one tab at a time.

## Safety policy

Stage 1 is closed by default. Each setting below is off unless the user starts the server with its flag:

| Capability | Default | Flag | Why |
|---|---|---|---|
| Running Python (`RunCode`) | Off | `--allow-code` | Arbitrary code execution |
| Writing files (`ExportReport`, `SaveRound`, `save_project`) | Off | `--allow-writes` | Changes the user's files |
| Network data (`LoadData` with `fred://`, `worldbank://`, `sdmx://` or any URL) | Off | `--allow-network` | Sends requests from the user's machine |
| Reading files (`LoadData`, `OpenResult`, the Dynare and IRIS imports, `CompareRounds`) | Only inside the root folder | `--root` | Keeps the assistant inside one folder |

Even when allowed, every path must resolve inside the root folder (by default the folder the server starts in).
This holds after `..` and symbolic links are resolved.

The MCP client asks the user before each tool call. Stage 1 relies on that approval, as the AI note intends:
nothing runs that the user did not approve.

## Not in stage 1

- **Stage 2:**
  - prompts built from the course labs and the report templates;
  - opening a saved project into the session (`open_project`, with the project-trust rules that typed code
    already has);
  - a redaction setting that returns summaries but never series values.
- **Stage 3:** connecting to the running desktop app, so its history and windows update as the assistant works.
  The app would need a local connection for that. It is the most useful form, and the hardest.

## Implementation notes

- **Package:** `pymetrix.mcp_server`, needing the MCP Python SDK 2.x (the extra `[mcp]`). The command is
  `pymetrix-mcp`, which talks over standard input and output, MCP's local transport.
- **Testability:** `build_server(session, policy)` returns the SDK's `MCPServer`, so tests drive it in-process
  with the SDK's own `Client`.
- **Client setup:** a user adds the server to their client's configuration, for example in Claude Desktop:
  ```json
  {"mcpServers": {"pymetrix": {"command": "pymetrix-mcp", "args": ["--root", "C:/work/forecasts"]}}}
  ```
