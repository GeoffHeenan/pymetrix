# An AI assistant as a plugin

Design note, 25 October 2026 · stage 1 built (`packages/pymetrix-assistant`, November 2026); stage 2 started (its core summary, `pymetrix.disclosure`) · builds on [`docs/notes/mcp-server.md`](../notes/mcp-server.md)
(stage 1 built), [`docs/notes/ai-assistance-in-the-ui.md`](../notes/ai-assistance-in-the-ui.md) (the direction)
and [plugins.md](plugins.md) (the extension API)

## The question

pymetrix has an MCP server: an assistant the user already has (Claude Desktop, VS Code...) can run pymetrix
commands in a session of its own. Could AI assistance inside the app be a plugin, an add-in installed separately,
rather than part of pymetrix?

Yes, and it should be. Most of what it needs exists already. What is missing is a few extension points in the app.

## What exists

- **The tools.** `pymetrix.mcp_server` gives an assistant a small, general set of tools over a pymetrix `Session`:
  - `list_commands`, `describe_command`, `run_command`;
  - `list_objects`, `show_object` (one tab at a time, tables as Markdown, charts as PNG);
  - `export_script`, `save_project`;
  - `search_help`, `read_help`.

  `build_server(session=..., policy=...)` already accepts the session to act on.
- **The safety policy.** `mcp_server.Policy` refuses running Python, writing files and network data unless each is
  allowed, and keeps every path inside a root folder (after `..` and links are resolved).
- **Recorded steps.** Almost every dialog builds a recorded call ([commands.md](commands.md)). What an assistant runs
  is therefore the same step a user would get from the dialog: in the history, in the script, replayed by the
  project.
- **The principle** (`ai-assistance-in-the-ui.md`): an AI proposes pymetrix commands, never freeform code. It sees
  only what the project's confidentiality mode allows (`off`, `local_help`, `metadata_only`, `full_context`,
  `local_model`). Nothing is sent silently.

## Why a plugin, not core

- **Opt-in by installation.** An institution whose rules forbid sending anything to a model does not install it, and
  can see that it is not there (*Help → Plugins*). A setting that is merely off is weaker assurance.
- **Its dependencies stay out of core**: model providers' SDKs, keys, network access. pymetrix adds no heavy
  dependency for users who do not want an assistant.
- **Choice.** A cloud model, or a local one through an OpenAI-compatible endpoint. An institution can ship its own
  assistant plugin, wired to its approved provider, without changing pymetrix.

## The proposal: `pymetrix-assistant`, in two stages

### Stage 1: connect the open app to an outside assistant

*Built:* `packages/pymetrix-assistant`, the Assistant panel (View → Assistant, **Allow connections**). The server is
streamable HTTP on 127.0.0.1 at an address with a secret in it; each tool call runs on the app's main thread and
each step through the app's runner. Core hook: extension API 1.8, `build_mcp_server(session_of=, run=, call=)` and
`MCPPolicy`. A panel rather than an Assistant menu, as plugins add panels, not menu items.

The plugin serves the existing MCP tools over a local connection, against the **app's own session**, not a separate
one. Claude Desktop, VS Code or any MCP client then acts in the window the user has open: each step appears in the
History panel as it runs, and the user can inspect, undo or rerun it there.

- **Reuse:** `build_server(session=<the app's session>, policy=...)`, unchanged; no model client inside pymetrix.
- **Approval:** the MCP client asks the user before each tool call, as it does for the standalone server.
- **Off until asked:** *Assistant → Allow connections* starts the local server, shows the address to give the
  client, and a status-bar note while one is connected. It stops when the app closes.
- **Cost:** small. The main new piece is running steps through the app's own step runner (below).

### Stage 2: an assistant panel in the app

A docked panel where the user asks in words. The plugin holds the model client, calls the same tools in-process,
and shows what it proposes.

- **Proposals are dialogs.** Each step the model proposes opens in its own dialog, filled in, so the user sees
  exactly the form they would have filled. They change anything they like, then press *Run*, or *Skip*. Nothing runs
  without that click. The step recorded is the dialog's call, the same as if the user had opened the dialog
  themselves.
- **Context follows the confidentiality mode:**
  - `metadata_only` sends object summaries, command schemas and help snippets, never raw data, formulas, file paths or
    report text;
  - `full_context` asks before each send and shows what is sent;
  - `local_model` sends to a local endpoint only.

  The mode is the project's, saved with it.
- **Audit.** Each conversation turn that led to a step is kept with that step's record: what was asked, what was
  proposed, what was changed before *Run*. The project then shows which steps an assistant proposed.
- **Providers** are the plugin's own: a small adapter per provider (Anthropic, an OpenAI-compatible local server).
  Keys come from the shared credentials helper if pymetrix adopts one ([plugins.md](plugins.md), decision 2).

## What core needs

These are small, and useful beyond an assistant.

1. **A `panels` extension point.** Plugins can add dialogs (`commands`) and result tabs (`views`), but not a docked
   panel. A `pymetrix.panels` entry point would name a function that receives the main window's interface
   (below) and returns a widget to dock, with its title and default place.
2. **A small window interface for plugins,** stable as the app changes:
   - the session;
   - the current object and selection;
   - `open_dialog(command)` to open a command in its dialog, pre-filled (today's `run_dialog(cls, initial=...)`,
     made public);
   - `run(command)` through the app's step runner;
   - a notification when the history changes.
3. **Every step through the app's runner.** The app now runs long steps in the background, with Cancel and Undo
   (`run_blocking`, the busy state). An assistant must go through the same path, never call `session.run` directly.
   This keeps one step at a time, keeps Cancel working, and puts the step in Undo.
4. **Summaries from core.** What `metadata_only` sends should be built by pymetrix, not by each assistant, so it is
   the same whichever assistant is installed and can be tested once. `views.summary_text` and the MCP server's object
   descriptions are the start. They need a sanitised form that leaves out values, formulas and paths, with a test
   that it does.

   *Built:* `pymetrix.disclosure` (`describe_session`, `describe_object`, `to_text`, `MODES`). It is an allow-list,
   not a filter: names, kinds, counts, sizes, period labels and settings that are identifiers, whole numbers or
   true/false; never data values, decimals, formulas, paths, free text or a step's code (a step is its dialog's name
   and what it made). A name that is not an identifier is counted, not quoted, so text arriving as a series name
   cannot carry a sentence into a prompt. `tests/test_disclosure.py` fills a project with marked values, a formula,
   a file path, labels, report text and typed code, checks the project holds them, and that none is in what is sent.

## Risks

- **Prompt injection.** Text from outside reaches the model and can try to steer it:
  - series labels and notes from a data provider;
  - a project's notes;
  - a plugin's help pages.

  The defences are the ones already chosen: the user approves every step; running code, writing files and the
  network stay off unless allowed; paths stay inside the root. The plugin must keep them on, not only the standalone
  server.
- **Leaking what the mode forbids.** A summary that includes a formula or a value would break `metadata_only`. Hence
  item 4: one tested sanitiser in core, not one per assistant.
- **Teaching.** An assistant can do a student's exercises. The note's course mode, which explains without running,
  still applies: a course could switch the panel to explaining only.
- **Over-trust.** Results an assistant proposed look the same as any other. The audit record (stage 2) and a mark in
  the History panel ("proposed by the assistant") keep that visible.

## Decisions taken (5 November 2026)

- **Provider first:** Anthropic. A reviewer without an Anthropic account uses a model on their own computer (an
  OpenAI-compatible server such as Ollama or LM Studio, the `local_model` mode) or stage 1 with their own assistant.
- **Default mode:** `off`; the panel asks which mode to use the first time.
- **The API key:** in the system keychain (Windows Credential Manager, macOS Keychain, the Secret Service on Linux),
  never in the project or a settings file.
- **Order:** the core summary first (done), then the panel with Run/Skip proposals in `metadata_only` (done), then
  the audit record and History mark (done), then `local_model` (done), `full_context` (done) and course mode (done).

*Built (step 2):* the Assistant panel's **Ask** tab (`pymetrix_assistant.chat`). Each question goes to Anthropic's API
(the model's own client, `anthropic`, server-side fallbacks on) with the project as `to_text(describe_session(...))`,
made afresh on the main thread. The model's tools only look up and propose: `list_commands`, `describe_command`
and `propose_step`, which checks the settings against the command and returns an error the model can correct.
It never sees a result. Each proposal is a card: **Open…** opens its dialog filled in (`AppInterface.open_dialog`)
for the user to run with OK; **Skip** drops it. `RunCode` is never proposed. The mode is kept in the project
(`session.attachments["pymetrix-assistant"]`), `off` until chosen; **What is sent…** shows the exact text. The key is
kept in the keychain through `keyring`, else `ANTHROPIC_API_KEY` is used.

*Built (step 3):* the record and the mark.
- **The mark** (core, extension API 1.9): the panel opens a proposal inside `session.proposing(CommandClass,
  "Claude")`, and the step the user runs from that dialog is recorded with `CommandRecord.proposed_by`. Only the
  proposed step is marked, not a recalculation after it. The mark is saved with the project and reopened. History
  shows "proposed" after the dialog's name, the script has a comment line above the step, and the lab book's step
  entry has `proposed_by`.
- **The record** (`pymetrix_assistant.audit`, `attachments["pymetrix-assistant"]["audit.jsonl"]`): each question
  with its model, mode, the exact text sent, the answer and the proposals, then what became of each proposal:
  - opened and run as step N, with the names of the settings the user changed first;
  - opened and closed without running;
  - skipped;
  - or the question failed, and why.

  **Record…** shows it.

*Built (local model):* `pymetrix_assistant.local`, the `local_model` mode. It asks an OpenAI-compatible server
(Ollama, LM Studio) through plain HTTP: no new dependency.
- The address must be loopback (`localhost`, `127.0.0.1`, `::1`). It is checked before every request; the request
  bypasses any proxy, and a redirect is refused.
- The model gets the same metadata, tools and proposals as with Claude. The record names the provider, server and
  model.
- The server address and model are this computer's settings, not the project's. **Find models** lists the
  server's models.
- The first use offers Anthropic or This computer.

*Built (full context):* the `full_context` mode, to Anthropic. Choosing it explains what more is sent and asks first.
- **With each question:** `describe_session(session, "full_context")` adds each step's code (formulas, settings, file
  paths, typed code). The panel shows the exact text, and nothing goes unless the user presses **Send**.
- **On request:** the model gains a `show_object(name, section)` tool, one tab of an object as a table. Long tables
  keep their first and last rows, and charts are not sent. Each request is shown to the user, on the main thread,
  and needs **Send**. A refusal is told to the model as such.
- **The record** keeps each request: the table sent, or that it was not.
- **Not offered:** `show_object` is not offered in metadata-only modes, and is refused if called.

*Built (course mode):* a course's notes can set `assistant: explain` or `assistant: off` in their front matter
(`pymetrix.help.Course.assistant`; the strictest note and the strictest loaded course apply).
- **`explain`:** the model is told the user is a student and not to do the exercise. `propose_step` is withdrawn from
  its tools, and refused if called. It explains methods, dialogs and results, and points to where the answer is found.
- **`off`:** the Ask tab is disabled.
- **Either rule:** **Allow connections** refuses to start, since a connected assistant runs steps. The panel says
  which course set the rule, and the record marks each explaining-only question.

## Decisions needed

1. **Stage 1 first?** *Recommended:* yes. It is the smaller step, needs no model client, and is useful on its own.
2. **Where it lives:** a separate package (`pymetrix-assistant`), or a bundled plugin that is off by default.
   *Recommended:* separate, so that "not installed" is the assurance.
3. **The default mode** for a new project when the plugin is installed: `off` or `metadata_only`. *Recommended:*
   `off`, with the first use asking which mode to set.
4. **The core items (1 to 4 above).** The panel extension point and the window interface belong with the plugin
   machinery, and should be designed with the plugin manager being planned separately (enable, disable, asking
   before a new plugin first runs).
