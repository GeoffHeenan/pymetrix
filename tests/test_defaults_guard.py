"""Defaults do not change silently. Scripts and saved steps leave fields at their defaults out, so a changed default would
change what an old script or project computes. Every built-in command's and method's defaults are frozen in
tests/fixtures/defaults.json; a change fails here until it is made deliberately:

- for a command, raise its ``schema_version`` and add a migration that writes the old default into steps saved before
  (see pymetrix.commands.registry), as the ``replace`` option did;
- for a method, test, filter or variance, say so in CHANGELOG.md, since scripts written before will now compute
  something else;
- then refresh the snapshot: ``python tests/test_defaults_guard.py``.
"""

import dataclasses
import json
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "defaults.json"
GROUPS = ("methods", "tests", "filters", "variances", "seasonal")


def _default(f: dataclasses.Field) -> object:
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        value = f.default_factory()  # type: ignore[misc]
    else:
        return "<required>"
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def current() -> dict[str, dict[str, object]]:
    """{class: {field: default}} for the built-in commands and the core plugin's registered classes."""
    from pymetrix.commands.registry import COMMAND_IDS
    from pymetrix.plugins import registry

    out: dict[str, dict[str, object]] = {}
    for cid, info in sorted(COMMAND_IDS.items()):
        if info.plugin == "pymetrix":
            out[f"command {cid} (schema {info.cls.schema_version})"] = {f.name: _default(f) for f in dataclasses.fields(info.cls)}
    for group in GROUPS:
        for entry in registry(group).entries():
            cls = entry.obj if isinstance(entry.obj, type) else type(entry.obj)
            if entry.plugin.name == "pymetrix" and dataclasses.is_dataclass(cls):
                out[f"{group} {entry.name}"] = {f.name: _default(f) for f in dataclasses.fields(cls)}
    return out


def test_no_default_changed_silently():
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    now = current()
    changed = []
    for key, defaults in frozen.items():
        if key not in now:
            continue  # a command whose schema rose is listed under its new key: checked by test_project_format
        for field, value in defaults.items():
            if field in now[key] and now[key][field] != value:
                changed.append(f"{key}: {field} {value!r} -> {now[key][field]!r}")
    assert not changed, ("defaults changed (see this file's docstring for what to do):\n" + "\n".join(changed))


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(current(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE} ({len(current())} classes)")
    sys.exit(0)
