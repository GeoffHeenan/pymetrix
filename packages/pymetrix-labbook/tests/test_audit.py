"""The auditor's read-only tools over a project file: summary, entries, brief, deviations, specifications and the
verify report, as functions and as an MCP server."""

import asyncio
import hashlib

import pytest
from pymetrix_labbook import brief, lock_plan, objective, register
from pymetrix_labbook.audit import AuditTools, build_audit_server

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session, save_project

register()


@pytest.fixture
def project(tmp_path):
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y[-1]", name="peek"))  # before the brief and the plan
    brief(s, "Consumption and income", "How much of a rise in income is consumed?", objectives={"O1": "Estimate the MPC"})
    lock_plan(s, [FitEquation("c ~ y", name="fit")], data=["data"])
    with objective("O1"):
        s.run(FitEquation("c ~ y", name="fit"))
    s.run(FitEquation("c ~ y + y[-1]", name="extra"))
    return save_project(s, tmp_path / "study")


def test_the_tools_answer_from_the_lab_book(project):
    tools = AuditTools(project)
    summary = tools.summary()
    assert "Brief: Consumption and income (written at entry 2" in summary and "Plan: 1 specification(s) locked" in summary
    assert "c: 3 specification(s), 3 fit(s)" in summary
    steps = tools.entries(kind="step")
    assert steps.count("\n") == 2 and '"plan": "exploratory"' in steps and '"objectives": ["O1"]' in steps
    assert "No entries match." == tools.entries(kind="step", outcome="nothing")
    assert "(1 more" in tools.entries(kind="step", limit=2)
    assert "written at entry 2" in tools.brief()
    deviations = tools.deviations()
    assert "warning: the brief was written at entry 2, after 1 step(s) of work" in deviations
    assert "warning: the planned data data were read at entry 1" in deviations
    detail = tools.specifications("c")
    assert detail.count("- entry") == 3 and "y " in detail
    assert "Verified: no errors." in tools.verify_report()


def test_the_tools_never_write_the_project(project):
    before = hashlib.sha256(project.read_bytes()).hexdigest()
    tools = AuditTools(project)
    for call in (tools.summary, tools.entries, tools.brief, tools.deviations, tools.specifications, tools.verify_report):
        call()
    tools.verify_report(replay=True)
    assert hashlib.sha256(project.read_bytes()).hexdigest() == before


def test_a_project_without_a_lab_book_is_refused(project, tmp_path):
    import json
    import zipfile

    plain = tmp_path / "plain.pymx"  # the same project with its lab book taken out
    with zipfile.ZipFile(project) as src, zipfile.ZipFile(plain, "w") as dst:
        for item in src.infolist():
            if item.filename.startswith("attachments/labbook/"):
                continue
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest.get("attachments", {}).pop("labbook", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    with pytest.raises(ValueError, match="no lab book"):
        AuditTools(plain)


def test_the_mcp_server_offers_the_tools(project):
    pytest.importorskip("mcp")
    from mcp import Client

    server = build_audit_server(project)

    async def go():
        async with Client(server) as client:
            tools = await client.list_tools()
            result = await client.call_tool("summary", {})
            return tools, result

    tools, result = asyncio.run(go())
    assert {t.name for t in tools.tools} == {"summary", "entries", "brief", "deviations", "specifications", "verify_report"}
    assert "Consumption and income" in "\n".join(c.text for c in result.content if c.type == "text")
