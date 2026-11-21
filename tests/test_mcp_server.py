"""The MCP server (docs/notes/mcp-server.md), driven by the MCP SDK's own client as an assistant would: commands run in
one session and replay from the exported script; results come back as tables and images; the policy holds."""

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")
pytest.importorskip("statsmodels")

from mcp import Client  # noqa: E402

from pymetrix.mcp_server import Policy, build_server  # noqa: E402


def call(server, *steps):
    """Run (tool, arguments) steps through a client connected in-process; return each result."""

    async def go():
        async with Client(server) as client:
            return [await client.call_tool(name, args) for name, args in steps]

    return asyncio.run(go())


def text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


@pytest.fixture
def server(tmp_path):
    return build_server(policy=Policy(root=tmp_path))


def test_the_tools_and_resources_on_offer(server):
    async def go():
        async with Client(server) as client:
            return await client.list_tools(), await client.list_resources()

    tools, resources = asyncio.run(go())
    assert {t.name for t in tools.tools} == {"list_commands", "describe_command", "run_command", "list_objects", "show_object",
                                            "export_script", "save_project", "search_help", "read_help"}
    assert {str(r.uri) for r in resources.resources} == {"pymetrix://help", "pymetrix://session/script"}
    run = next(t for t in tools.tools if t.name == "run_command")
    assert set(run.input_schema["properties"]) == {"name", "settings"} and "FitEquation" in run.description


def test_an_assistant_session_replays_from_its_script(server):
    data_step = ("run_command", {"name": "LoadExample", "settings": {}})
    fit_step = ("run_command", {"name": "FitEquation", "settings": {"formula": "c ~ y + c[-1]", "cov": "hac"}})
    var_step = ("run_command", {"name": "FitVAR", "settings": {"variables": ["c", "i"], "lags": 2}})  # JSON values, not dialog text
    loaded, fitted, var, objects, script = call(server, data_step, fit_step, var_step, ("list_objects", {}), ("export_script", {}))
    assert not any(r.is_error for r in (loaded, fitted, var))
    assert "fit = Equation('c ~ y + c[-1]').fit(data, method=OLS(cov='hac'))" in text(fitted)
    assert "**Summary**" in text(fitted) and "HAC standard errors" in text(fitted)
    assert "`fit`: FittedEquation" in text(objects) and "`var`: FittedVAR" in text(objects)
    session = server.session_tools.session
    ns: dict = {}
    exec(text(script), ns)  # the user's copy of what the assistant did
    assert np.allclose(ns["fit"].params.values, session["fit"].params.values)
    assert np.allclose(ns["var"].coefs, session["var"].coefs)


def test_results_come_back_as_the_app_shows_them(server):
    _, summary, graph, missing = call(server, ("run_command", {"name": "LoadExample", "settings": {}}),
                                      ("show_object", {"name": "data", "section": "data"}),
                                      ("show_object", {"name": "data", "section": "Graph"}),
                                      ("show_object", {"name": "data", "section": "Nope"}))
    assert "tabs: Data, Graph, Variables" in text(summary)
    assert "Showing the first and last 15 of 100 rows" in text(summary)  # long tables are cut
    image = next(c for c in graph.content if c.type == "image")
    assert image.mime_type == "image/png" and len(image.data) > 1000
    assert missing.is_error and "its tabs are Data, Graph, Variables" in text(missing)


def test_discovery_and_help(server):
    listing, estimate, described, found, page = call(server, ("list_commands", {}), ("list_commands", {"menu": "estimate"}),
                                                     ("describe_command", {"name": "RunBacktest"}),
                                                     ("search_help", {"text": "backtest"}),
                                                     ("read_help", {"link": "pymetrix:commands/pymetrix:runbacktest"}))
    assert "## Forecast" in text(listing) and "`RunCode`" in text(listing) and "**Disabled:** running Python is off" in text(listing)
    assert text(estimate).startswith("## Estimate") and "## Forecast" not in text(estimate)
    assert "| `horizon` |" in text(described) and "| `window` |" in text(described)
    assert "pymetrix:commands/pymetrix:runbacktest" in text(found)
    assert text(page).startswith("# Backtest")


def test_mistakes_get_messages_an_assistant_can_act_on(server):
    unknown, misspelt, bad_setting, failed = call(
        server, ("run_command", {"name": "FitEquaton", "settings": {}}),
        ("describe_command", {"name": "fitvar"}),
        ("run_command", {"name": "LoadExample", "settings": {"datset": "sample_macro"}}),
        ("run_command", {"name": "FitEquation", "settings": {"formula": "c ~ y"}}))  # no data loaded yet
    assert unknown.is_error and "did you mean FitEquation" in text(unknown)
    assert misspelt.is_error and "FitVAR" in text(misspelt)
    assert bad_setting.is_error and "no setting 'datset'" in text(bad_setting) and "dataset" in text(bad_setting)
    assert failed.is_error and "FitEquation failed" in text(failed)


def test_the_policy_closes_code_writes_network_and_other_folders(server, tmp_path):
    outside = tmp_path.parent / "elsewhere.csv"
    code, report, project, network, escape = call(
        server, ("run_command", {"name": "RunCode", "settings": {"code": "import os"}}),
        ("run_command", {"name": "ExportReport", "settings": {"path": "report.html"}}),
        ("save_project", {"path": "work.pymx"}),
        ("run_command", {"name": "LoadData", "settings": {"path": "fred://GDPC1"}}),
        ("run_command", {"name": "LoadData", "settings": {"path": str(outside)}}))
    assert code.is_error and "--allow-code" in text(code)
    assert report.is_error and project.is_error and "--allow-writes" in text(project)
    assert network.is_error and "--allow-network" in text(network)
    assert escape.is_error and "outside the server's folder" in text(escape)
    assert not list(tmp_path.iterdir())  # nothing written


def test_allowed_writes_stay_in_the_folder_and_the_project_reopens(tmp_path):
    from pymetrix.commands import load_project

    (tmp_path / "macro.csv").write_text("time,x\n2000Q1,1\n2000Q2,2\n2000Q3,4\n", encoding="utf-8")
    server = build_server(policy=Policy(root=tmp_path, allow_writes=True))
    loaded, saved, escape = call(server, ("run_command", {"name": "LoadData", "settings": {"path": "macro.csv", "time": "time", "freq": "Q"}}),
                                 ("save_project", {"path": "work.pymx"}),
                                 ("save_project", {"path": "../work.pymx"}))
    assert not loaded.is_error and not saved.is_error and (tmp_path / "work.pymx").exists()
    assert escape.is_error and not (tmp_path.parent / "work.pymx").exists()
    reopened = load_project(tmp_path / "work.pymx")
    assert list(reopened["data"]["x"]) == [1, 2, 4]


def test_the_command_line(tmp_path, monkeypatch):
    from pymetrix import mcp_server

    seen = {}

    class Server:
        def run(self, transport):
            seen["transport"] = transport

    def build(policy):
        seen["policy"] = policy
        return Server()

    monkeypatch.setattr(mcp_server, "build_server", build)
    mcp_server.main(["--root", str(tmp_path), "--allow-writes"])
    assert seen["transport"] == "stdio"
    assert seen["policy"] == Policy(Path(tmp_path).resolve(), allow_writes=True)


def test_a_client_can_launch_it_over_stdio(tmp_path):
    """The real transport: the client starts ``python -m pymetrix.mcp_server`` and talks to it over stdin and stdout."""
    import os
    import sys

    from mcp import StdioServerParameters

    src = str(Path(__file__).parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    params = StdioServerParameters(command=sys.executable, args=["-m", "pymetrix.mcp_server", "--root", str(tmp_path)], env=env)

    async def go():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            loaded = await client.call_tool("run_command", {"name": "LoadExample", "settings": {}})
            return names, loaded

    names, loaded = asyncio.run(go())
    assert "run_command" in names and not loaded.is_error and "data = Dataset.example('sample_macro')" in text(loaded)


def test_printing_cannot_break_the_stdio_stream(server, capsys):
    """Over stdio, standard output carries the protocol: whatever a step prints goes to standard error."""
    import pymetrix.mcp_server as module

    tools = server.session_tools

    class Noisy(module._Tools):
        def list_objects(self):
            print("progress: 50%")
            return "ok"

    noisy = Noisy(tools.session, tools.policy)
    assert noisy.list_objects() == "ok"
    out, err = capsys.readouterr()
    assert out == "" and "progress: 50%" in err
