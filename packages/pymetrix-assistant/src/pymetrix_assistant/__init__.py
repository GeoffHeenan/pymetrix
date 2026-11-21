"""An AI assistant in the pymetrix desktop app (``pymetrix-assistant``), stages 1 and 2 of pymetrix's
``docs/design/assistant-plugin.md``.

The **Assistant** panel (View → Assistant) has two tabs. **Ask** (stage 2) asks Claude, through Anthropic's API,
about the project: it is told the project in metadata only (``pymetrix.disclosure``), off until the project chooses,
and answers with proposals, each opened in its dialog filled in for the user to run with OK, or skipped; it runs
nothing itself (``pymetrix_assistant.chat``). The API key is kept in the system keychain.

**Connect** (stage 1) has **Allow connections**: it starts pymetrix's MCP tools (the same as
``pymetrix-mcp``'s) against the session in the window, on this computer only, at an address with a secret in it,
which the panel shows for the assistant's configuration (Claude Code, VS Code, other MCP clients). The assistant's
steps run as the app runs a step: one at a time, in the background with progress and Cancel, in History, undoable,
and recorded by a lab book as an agent's. The policy is pymetrix's: no Python, no file writes, no network data.
Unticking it, or closing the app, stops the server.

There, the assistant is the user's own, and it asks the user before each tool call.
"""

from __future__ import annotations

from pymetrix.extension import PluginInfo

from ._version import __version__

PLUGIN = PluginInfo(
    name="pymetrix-assistant",
    version=__version__,
    api="1.9",  # build_mcp_server (1.8); Session.proposing (1.9)
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="An AI assistant in the desktop app: ask Claude, or serve the open session to your own assistant over MCP",
    requires={"mcp": ">=2.3", "uvicorn": "", "anthropic": ">=2.6", "keyring": ">=24"},
    install='pip install "pymetrix-assistant"',
)

from .server import AppServer, serve

_registered = False


def register() -> None:
    """The entry point: register the Assistant panel (once). Qt is imported only when the app builds the panel."""
    global _registered
    if _registered:
        return
    from pymetrix.extension import AppPanel, plugin_context
    from pymetrix.extension import register as add

    def create(app):
        from .app import create_panel

        return create_panel(app)

    with plugin_context(PLUGIN):
        add("panels", AppPanel("Assistant", create, area="right"), name="assistant")
    _registered = True


__all__ = ["PLUGIN", "AppServer", "__version__", "register", "serve"]
