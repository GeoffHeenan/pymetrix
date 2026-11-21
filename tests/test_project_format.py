"""Project files stay readable: stable command IDs, schema versions, migrations, and old files (review R2)."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from pymetrix import Dataset, Equation
from pymetrix.commands import FitEquation, load_project, save_project
from pymetrix.commands.registry import COMMAND_IDS, COMMANDS, MIGRATIONS, Command, command, migration, upgrade

FIXTURES = Path(__file__).parent / "fixtures"


def test_built_in_command_ids_never_change():
    """Projects store these IDs: renaming a class must not change its ID (give command_id=)."""
    frozen = {k: v for k, v in json.loads((FIXTURES / "command_ids.json").read_text()).items() if not k.startswith("_")}
    current = {cid: info.cls.schema_version for cid, info in COMMAND_IDS.items() if info.plugin == "pymetrix"}
    assert not set(frozen) - set(current), f"IDs that disappeared (old projects would not open): {sorted(set(frozen) - set(current))}"
    assert not set(current) - set(frozen), f"new commands: add their IDs to tests/fixtures/command_ids.json: {sorted(set(current) - set(frozen))}"
    for cid, version in frozen.items():
        assert current[cid] == version, f"{cid}: schema {current[cid]}, frozen at {version}; add a migration and update the fixture"
        for v in range(1, version):
            assert (cid, v) in MIGRATIONS, f"{cid}: no migration from schema {v}"


def test_projects_saved_in_format_1_still_open_and_are_saved_in_format_2(tmp_path):
    old = FIXTURES / "projects" / "format1.pymx"
    with zipfile.ZipFile(old) as z:
        assert json.loads(z.read("manifest.json"))["format_version"] == 1
        steps = json.loads(z.read("commands.json"))
    assert "DefineGroup" in [s["command"] for s in steps]  # names from before the renames
    s = load_project(old, trust=True)
    assert {"cons", "inv", "identity", "solution", "spending", "var", "pivot", "ratio"} <= set(s.objects)
    data = s["data"]
    assert "growth_c" in data.names
    fresh = Equation("c ~ y + c[-1]").fit(data)  # the project's own copy of the data, as it was when saved
    assert np.allclose(s["cons"].params.values, fresh.params.values)
    resaved = save_project(s, tmp_path / "resaved.pymx")
    with zipfile.ZipFile(resaved) as z:
        manifest, steps = json.loads(z.read("manifest.json")), json.loads(z.read("commands.json"))
    assert manifest["format_version"] == 2 and manifest["plugins"] == {"pymetrix": manifest["package_version"]}
    assert {k: steps[0][k] for k in ("id", "version", "command", "fields")} == {
        "id": "pymetrix:select_series", "version": 1, "command": "SelectSeries",
        "fields": {"series": "c, i", "patterns": "", "data": "data", "name": "spending"}}
    assert steps[0]["outputs"] == ["spending"] and "seconds" in steps[0]  # added for opening without redoing work
    again = load_project(resaved, trust=True)
    assert np.allclose(again["solution"]["y"], s["solution"]["y"], equal_nan=True)


@pytest.fixture()
def versioned_command():
    """A command at schema 3 whose field was renamed at 2 and whose meaning changed at 3, with migrations."""

    @dataclass(frozen=True)
    class Scale(Command):
        schema_version: ClassVar[int] = 3
        factor_pct: float = 100.0  # schema 1: "multiplier"; schema 2: "factor" (a ratio); schema 3: percent
        name: str = "scaled"

        def run(self, session):
            session.objects[self.name] = self.factor_pct
            return self.factor_pct

        def to_code(self):
            return f"{self.name} = {self.factor_pct!r}"

    command(None, id="testing:scale")(Scale)

    @migration("testing:scale", 1)
    def _(fields):
        fields["factor"] = fields.pop("multiplier")
        return fields

    @migration("testing:scale", 2)
    def _(fields):
        fields["factor_pct"] = fields.pop("factor") * 100
        return fields

    yield Scale
    COMMAND_IDS.pop("testing:scale")
    COMMANDS.pop("Scale")
    MIGRATIONS.pop(("testing:scale", 1))
    MIGRATIONS.pop(("testing:scale", 2))


def _project_with(tmp_path, steps, manifest_extra=None):
    path = tmp_path / "p.pymx"
    manifest = {"format_version": 2, "package_version": "test", "datasets": [], "placed": [], **(manifest_extra or {})}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("commands.json", json.dumps(steps))
    return path


def test_migrations_bring_old_steps_up_to_date(versioned_command, tmp_path):
    assert upgrade("testing:scale", 1, {"multiplier": 1.5, "name": "x"}) == {"factor_pct": 150.0, "name": "x"}
    assert upgrade("testing:scale", 3, {"factor_pct": 5.0}) == {"factor_pct": 5.0}
    path = _project_with(tmp_path, [{"id": "testing:scale", "version": 1, "command": "Scale", "fields": {"multiplier": 0.5, "name": "x"}}])
    assert load_project(path)["x"] == 50.0


def test_clear_errors_for_steps_that_cannot_be_read(versioned_command, tmp_path):
    newer = _project_with(tmp_path, [{"id": "testing:scale", "version": 4, "fields": {}}])
    with pytest.raises(ValueError, match=r"saved by a newer pymetrix \(step schema 4; installed: 3\); upgrade pymetrix"):
        load_project(newer)
    unknown = _project_with(tmp_path, [{"id": "testing:scale", "version": 3, "fields": {"factor_pct": 1.0, "colour": "red"}}])
    with pytest.raises(ValueError, match="has no field colour"):
        load_project(unknown)
    missing = _project_with(tmp_path, [{"id": "acme:forecast_fan", "version": 1, "fields": {}}], {"plugins": {"pymetrix-acme": "2.1"}})
    opened = load_project(missing)  # a missing plugin's step is kept, not run, with what to install: the project opens
    step = opened.history[-1]
    assert step.skipped and "acme:forecast_fan comes from pymetrix-acme, which is not installed: pip install pymetrix-acme" in step.error
    assert opened.missing_plugins == {"pymetrix-acme 2.1": step.error.removeprefix("not run: ")}
    future = _project_with(tmp_path, [], {"format_version": 99, "package_version": "9.0"})
    with pytest.raises(ValueError, match="saved by a newer version of pymetrix .9.0."):
        load_project(future)


def test_ids_are_checked_when_commands_register():
    @dataclass(frozen=True)
    class FitEquation2(Command):
        def run(self, session):
            return None

        def to_code(self):
            return ""

    with pytest.raises(ValueError, match="already used by pymetrix.commands.builtin.FitEquation"):
        command(None, id="pymetrix:fit_equation")(FitEquation2)
    with pytest.raises(ValueError, match="must look like 'plugin:name'"):
        command(None, id="Not An ID")(FitEquation2)
    assert FitEquation.command_id == "pymetrix:fit_equation" and FitEquation.schema_version == 1
