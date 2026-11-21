"""Report notes (extension API 1.6): a plugin's line about each exported result, under its heading, before the
appendix; a note that fails is left out with a warning and never stops the report."""

import pytest

from pymetrix import Dataset, plugins
from pymetrix.commands import ExportReport, FitEquation, Session
from pymetrix.extension import PluginInfo, ReportNote, plugin_context


@pytest.fixture
def notes():
    saved = dict(plugins.registry("report_notes")._entries)
    yield
    plugins.registry("report_notes")._entries = saved


def test_notes_go_under_their_heading_and_a_failing_one_is_left_out(notes, tmp_path):
    def broken(session, name):
        raise RuntimeError("no")

    with plugin_context(PluginInfo(name="pymetrix-notes", version="0.1")):
        plugins.register("report_notes", ReportNote("size", "Sizes", lambda s, n: f"{n} has {len(s[n].params.names)} terms"),
                         name="size")
        plugins.register("report_notes", ReportNote("broken", "Broken", broken), name="broken")
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y", name="fit"))
    path = tmp_path / "report.md"
    with pytest.warns(UserWarning, match="report note .*broken.* failed on fit"):
        s.run(ExportReport(objects="fit", path=str(path), plugin_notes=True))
    text = path.read_text(encoding="utf-8")
    assert "Sizes" in text and "fit has 2 terms" in text and "Broken" not in text
