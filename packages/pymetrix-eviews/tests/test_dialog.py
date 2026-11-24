"""File → Import EViews workfile…: the step equals the Dataset.read calls it stands for, its script replays (with the
plugin, and from the Parquet copies without it), its ID is fixed, and its help topic is there."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import pymetrix_eviews as pe
from pymetrix import Dataset
from pymetrix.commands import Session, load_project, save_project
from pymetrix.commands.registry import COMMAND_IDS, COMMANDS

FIXTURES = Path(__file__).parent / "fixtures"


def _dialog(**fields):
    return COMMANDS["ImportEViews"].cls(**fields)


def _bundle(tmp_path):
    shutil.copytree(FIXTURES / "macro", tmp_path / "macro")
    return str(tmp_path / "macro" / "macro.evbundle")


def _equal(a, b):
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.axes == b.axes
    assert {k: (v.label, v.units, dict(v.extra)) for k, v in a.variables.items()} == \
        {k: (v.label, v.units, dict(v.extra)) for k, v in b.variables.items()}
    assert a.meta.extra["eviews"] == b.meta.extra["eviews"]


def test_the_command_id_is_fixed():
    frozen = {k: v for k, v in json.loads((FIXTURES / "command_ids.json").read_text()).items() if not k.startswith("_")}
    ours = {cid: info.cls.schema_version for cid, info in COMMAND_IDS.items() if info.plugin == "pymetrix-eviews"}
    assert ours == frozen
    info = COMMAND_IDS["eviews:import_eviews"]
    assert info.cls is COMMANDS["ImportEViews"].cls and info.menu == "File/Import EViews workfile..."


def test_reading_every_page_equals_dataset_read_per_page(tmp_path):
    path = _bundle(tmp_path)
    s = Session()
    s.run(_dialog(path=path))
    record = s.history[-1]
    assert record.ok, record.error
    assert set(record.writes) == {"quarterly", "annual", "panel", "cross"}
    assert record.code == "\n".join(f"{n} = Dataset.read({path!r}, page={p!r})" for n, p in
                                    (("quarterly", "quarterly"), ("annual", "Annual"), ("panel", "panel"), ("cross", "cross")))
    for name, page in (("quarterly", "quarterly"), ("annual", "Annual"), ("panel", "panel"), ("cross", "cross")):
        _equal(s[name], Dataset.read(path, page=page))
    ns: dict = {}
    exec(s.script(), ns)  # the script replays
    for name in ("quarterly", "annual", "panel", "cross"):
        _equal(ns[name], s[name])


def test_the_step_is_what_typing_its_code_records(tmp_path):
    path = _bundle(tmp_path)
    s = Session()
    s.run(_dialog(path=path, pages="quarterly"))
    typed = Session()
    typed.run_code(s.history[-1].code)
    call = typed.history[-1].command
    assert type(call).__name__ == "Call" and call.function == "pymetrix:dataset.read" and call.to_code() == s.history[-1].code
    _equal(typed["quarterly"], s["quarterly"])


def test_a_project_with_the_step_reopens(tmp_path):
    path = _bundle(tmp_path)
    s = Session()
    s.run(_dialog(path=path, pages="quarterly, cross"))
    again = load_project(save_project(s, tmp_path / "p.pymx"))
    assert type(again.history[-1].command).__name__ == "ImportEViews"
    _equal(again["quarterly"], s["quarterly"])
    _equal(again["cross"], s["cross"])


def test_save_the_data_as_parquet_replays_without_the_plugin(tmp_path, monkeypatch):
    from pymetrix.io import registry

    path = _bundle(tmp_path)
    s = Session()
    s.run(_dialog(path=path, pages="quarterly,Annual", parquet=True))
    copies = [str(tmp_path / "macro" / "quarterly.parquet"), str(tmp_path / "macro" / "annual.parquet")]
    assert s.history[-1].code == f"quarterly = Dataset.read({copies[0]!r})\nannual = Dataset.read({copies[1]!r})"
    _equal(s["quarterly"], Dataset.read(path, page="quarterly"))
    _equal(s["annual"], Dataset.read(path, page="Annual"))
    monkeypatch.delitem(registry._READERS, ".evbundle")  # no plugin: the script reads only the Parquet copies
    monkeypatch.setattr(registry.plugins, "load", lambda group: None)
    with pytest.raises(Exception, match="no reader for '.evbundle'"):
        Dataset.read(path, page="quarterly")
    ns: dict = {}
    exec(s.script(), ns)
    _equal(ns["quarterly"], s["quarterly"])
    _equal(ns["annual"], s["annual"])


def test_dated_pages_as_undated(tmp_path):
    shutil.copytree(FIXTURES / "semiannual", tmp_path / "semi")
    path = str(tmp_path / "semi" / "semi.evbundle")
    s = Session()
    assert s.run is not None
    errors = _dialog(path=path).validate(s)
    assert errors == ["page halves (S): semi-annual: pandas has no half-year period (a known gap); tick Read dated pages as undated"]
    s.run(_dialog(path=path, undated=True))
    assert s.history[-1].code == f"halves = Dataset.read({path!r}, page='halves', undated=True)"
    _equal(s["halves"], Dataset.read(path, page="halves", undated=True))


def test_what_the_dialog_refuses_before_running(tmp_path):
    s = Session()
    assert _dialog().validate(s) == ["choose the bundle (.evbundle) the export program wrote"]
    assert _dialog(path=str(FIXTURES / "newer" / "newer.evbundle")).validate(s)[0].endswith(
        "update pymetrix-eviews (pip install -U pymetrix-eviews)")
    path = _bundle(tmp_path)
    assert _dialog(path=path, pages="quarterly, monthly").validate(s) == [
        "the bundle has no page 'monthly'; its pages: quarterly, Annual, panel, cross"]
    assert _dialog(path=path, pages="quarterly, QUARTERLY").validate(s) == [
        "two pages would be called quarterly: read them one at a time"]
    assert _dialog(what="get the export program (experimental)").validate(s) == ["choose where to save the export program"]


def test_get_the_export_program(tmp_path):
    s = Session()
    out = tmp_path / "pymetrix_export.prg"
    s.run(_dialog(what="get the export program (experimental)", save_to=str(out)))
    assert out.read_bytes() == pe.export_program().replace("\n", "\r\n").encode("utf-8")
    assert s.history[-1].code == f"from pymetrix_eviews import save_export_program\nsave_export_program({str(out)!r})"
    out.unlink()
    exec(s.script(), {})
    assert out.is_file()


def test_the_form_says_the_program_is_experimental_and_links_its_help():
    from pymetrix.forms import form_spec
    from pymetrix.help import topic

    cls = COMMANDS["ImportEViews"].cls
    fields = {f.name: f for f in form_spec(cls)}
    assert list(fields) == ["what", "path", "pages", "parquet", "undated", "save_to"]
    assert fields["what"].choices == ("workfile bundle", "get the export program (experimental)")
    assert "experimental" in fields["save_to"].help and "untested in EViews" in fields["what"].help
    assert fields["parquet"].label == "Save the data as Parquet"
    doc = cls.__doc__
    assert "EXPERIMENTAL" in doc and "CSV or Excel" in doc and "supported way in" in doc
    help_text = topic("eviews-import").text
    assert cls.topics == ("eviews-import",)
    assert "has NOT been run in EViews" in help_text and "Supported: save from EViews as CSV or Excel" in help_text


def test_importing_the_plugin_first_leaves_pymetrix_own_dialogs_as_they_are():
    """The plugin imported before anything else of pymetrix (as a script's first line, or the .evbundle reader's
    entry point) must not register pymetrix's own dialogs, imported on the way, as its own."""
    import subprocess
    import sys

    code = ("import pymetrix_eviews\nfrom pymetrix.commands.registry import COMMAND_IDS\n"
            "print(sorted(c for c, i in COMMAND_IDS.items() if i.plugin == 'pymetrix-eviews'))\n"
            "print(COMMAND_IDS['pymetrix:fit_equation'].plugin)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300).stdout.split("\n")
    assert out[:2] == ["['eviews:import_eviews', 'eviews:translate_program']", "pymetrix"]
