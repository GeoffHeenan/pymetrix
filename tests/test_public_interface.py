"""The public interface for programs that write and read projects (docs/guide.md, "Reading and writing projects from
programs"): its names are exported, documented as public and provisional before 1.0 (not "stable"), and the guide's
examples run as written and say what they give."""

import re
from pathlib import Path

import numpy as np

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide.md"
HEADING = "## Reading and writing projects from programs"

EXPORTED = {
    "pymetrix.calls": ["Call", "make_call", "call_of", "call_record"],
    "pymetrix.codec": ["encode", "decode", "Ref", "SerialisationError"],
    "pymetrix.commands": ["Session", "CommandRecord", "save_project", "load_project"],
    "pymetrix.commands.session": ["Session", "CommandRecord", "CommandError"],
    "pymetrix.commands.registry": ["to_json", "from_json"],
    "pymetrix.saved": ["save_result", "load_result", "SavedResult", "result_bytes", "result_from_bytes"],
    "pymetrix.build": ["build_info"],
}


def _section() -> str:
    text = GUIDE.read_text(encoding="utf-8")
    start = text.index(HEADING)
    end = text.index("\n## ", start + len(HEADING))
    return text[start:end]


def test_the_names_are_exported():
    import importlib

    for module, names in EXPORTED.items():
        exported = importlib.import_module(module).__all__
        assert set(names) <= set(exported), (module, sorted(set(names) - set(exported)))
    from pymetrix.calls import Call
    from pymetrix.commands import Session

    assert callable(Call.invoke) and isinstance(Session().attachments, dict)


def test_documented_as_public_and_provisional_not_stable():
    section = _section()
    assert "public interface, provisional before 1.0" in section
    assert not re.search(r"\b(is|are|as) stable\b|stable (interface|api)|stability (promise|guarantee)d?\b", section.lower())
    for name in ("Call", "make_call", "encode", "decode", "Session.attachments", "invoke", "load_project", "save_project"):
        assert f"`{name}" in section or f"{name}(" in section, name


def test_the_guides_examples_run_as_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)```", _section(), flags=re.S)
    assert len(blocks) == 3
    ns: dict = {}
    for block in blocks:
        exec(compile(block, "guide", "exec"), ns)  # noqa: S102 - the guide's own examples
    assert ns["calls"][0].to_code() == "fit = Equation('c ~ y + c[-1]').fit(data, method=OLS(cov='hac'))"
    from pymetrix import OLS

    assert ns["decode"](ns["calls"][0].arguments["method"]) == OLS(cov="hac")
    assert [step.get("deferred", False) for step in ns["steps"]] == [False, True]
    assert ns["log"][-1]["pymetrix"] and "commit" in ns["log"][-1]
    session = ns["s"]
    assert session.attachments["mytool"]["run.json"] == b'{"pipeline": "nightly"}'
    assert set(session.pending()) == {"later"}
    again = session.history[0].call.invoke(session.objects)
    assert np.array_equal(again.params.values, session["fit"].params.values)
    assert np.allclose(again.params.se, session["fit"].params.se)
