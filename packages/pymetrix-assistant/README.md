# pymetrix-assistant

An AI assistant in the PyMetrix desktop app, as a plugin, with two tabs in **View → Assistant**:

- **Ask** asks Claude about the project, through Anthropic's API, and answers with steps for you to run (stage 2).
- **Connect** serves the open window's session to an assistant you already use (Claude Code, VS Code, other MCP
  clients), on this computer only (stage 1).

See pymetrix's `docs/design/assistant-plugin.md`.

```bash
pip install -e packages/pymetrix-assistant        # from the repository (not on PyPI yet)
```

## Ask

1. Press **API key…** and paste your Anthropic API key (from console.anthropic.com). It is kept in this computer's
   keychain, never in the project. (`ANTHROPIC_API_KEY` is used when the keychain has none.)
2. Ask, for example "fit a consumption function on data". The first time, the panel asks whether this project may
   send its **metadata only**: the names, kinds, sizes and periods of its objects and the names of the steps taken.
   Data values, estimates, formulas, labels, notes, report text, file paths and typed code are never sent.
   **What is sent…** shows the exact text. The choice is saved with the project; **Send: Off** stops it.
3. The answer comes with proposals. **Open…** opens the step's dialog, filled in: change it if you like and press OK
   to run it, as you would any step. **Skip** drops it. The model runs nothing itself and never sees results, and it
   never proposes Python.
4. A step you ran from a proposal says "proposed" in History, and a comment above it in the exported script names
   who proposed it. **Record…** shows the record kept in the project: every question, the exact text sent, the
   answer and what became of each proposal.

**Continue where I left off** appears when a reopened project has an earlier conversation: it puts that
conversation back on screen and gives it to the model with your next question. It is offered only when questions go
to the same place, with the same mode and full-context choice, as before. **New conversation** starts afresh.

Anthropic's server-side fallback is on: if the model declines a request, Anthropic's recommended fallback model
answers instead, and the status bar says so.

### Full context

**Send: Full context, to Anthropic** lets Claude read more than metadata, so it can interpret results ("is this
residual test a problem?") rather than only propose steps.
- **With each question:** the code of the project's steps is sent too.
- **On request:** Claude can ask to see one of an object's tables.
- **You approve each send:** it is shown to you first, and goes only if you press **Send**. Charts are never sent.
- **Recorded:** the record keeps what was sent.

### In a course

A course can ask for explaining only (`assistant: explain` in a note's front matter). While it is loaded, the
assistant explains methods and results and points to where the answer is found, but proposes no steps, and
**Allow connections** stays off. `assistant: off` turns the assistant off. The panel says which course set the rule.

### A model on your own computer

Without an Anthropic account, or to keep everything on the machine, choose **Send: Metadata only, to a model on this
computer**.

1. Install [Ollama](https://ollama.com) and pull a model that supports tool calls, for example
   `ollama pull qwen3:8b`. LM Studio works too: start its server, at `http://localhost:1234/v1`.
2. In the panel, check the address (Ollama's `http://localhost:11434/v1` is the default) and press **Find models**.
3. Choose the model and ask.

**Full context (it stays on this computer)**, a tick box under the address, also sends the model the code of the
project's steps and lets it read an object's tables (estimates, tests, data) when it asks. Since nothing leaves the
computer, it is not asked about each time; the record keeps what was sent. It is the project's choice, off until
ticked.

Only an address on this computer is accepted. Requests go straight to it, never through a proxy, and a redirect
elsewhere is refused. Smaller models are slower and less reliable at choosing settings than Claude, but each
proposal is checked against its dialog before you see it.

## Connect

In the app: **View → Assistant**, **Connect** tab, tick **Allow connections**, and give the assistant the address shown. For Claude
Code, **Copy Claude Code command** copies:

```bash
claude mcp add --transport http pymetrix http://127.0.0.1:<port>/mcp/<secret>
```

## What a connected assistant can do

The tools are pymetrix's MCP tools (`pymetrix-mcp`): list and describe the commands, run one, list and show the
objects, export the script, search and read the help. Against the window's own session:

- **Its steps run as yours do:** one at a time, in the background with progress and Cancel, in History, undoable.
  The panel lists the steps it has run, and a lab book records them as an agent's.
- **It is closed by default:** no Python, no file writes, no network data (pymetrix's MCP policy). Files are read
  only from the folder the app started in.
- **On this computer only,** at an address with a secret in it: other programs cannot guess it. Unticking
  **Allow connections**, or closing the app, stops it.

Here the assistant is your own, and it asks you before each tool call.

## How it works

The server (uvicorn, streamable HTTP) runs on its own thread. Each tool call is passed to the app's main thread,
where the session and the window live, and waits for its result; a step goes through the app's step runner
(`AppInterface.run`). The core hook is pymetrix's extension API 1.8: `build_mcp_server(session_of=, run=, call=)`.

License: BSD-3-Clause.
