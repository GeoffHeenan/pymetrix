"""Single results saved to a file (pymetrix.saved): every kind of result comes back with the same views, numbers and
summary, figures redraw from their series, and the file holds only JSON and text. The objects are the 54 kinds of
tests/_gallery.py and the demo projects, as in test_result_views.py."""

import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

import _gallery
from pymetrix import SavedResult, load_result, save_result
from pymetrix.errors import SpecificationError
from pymetrix.figures import Figure
from pymetrix.output import Table
from pymetrix.saved import result_bytes, result_from_bytes
from pymetrix.views import summary_text, view_sections


@pytest.fixture(scope="module")
def objects():
    return {k: o for k, o in _gallery.every_object().items() if type(o).__module__.startswith("pymetrix")}


def _same_cell(a, b):
    if isinstance(a, (float, np.floating)) and not isinstance(a, bool):
        return isinstance(b, float) and (a == b or (math.isnan(a) and math.isnan(b)))
    if isinstance(a, (bool, np.bool_)):
        return b == bool(a)
    if isinstance(a, (int, np.integer)):
        return b == int(a)
    return b == (None if a is None else str(a))  # text, and what JSON writes as text: dates, periods


def _same_view(original, loaded):
    if isinstance(original, Table):
        assert isinstance(loaded, Table) and str(loaded) == str(original)  # shown the same
        assert (loaded.title, list(loaded.columns), list(loaded.notes)) == (original.title, list(original.columns), list(original.notes))
        bad = [(i, j) for i, (r, s) in enumerate(zip(original.rows, loaded.rows)) for j, (a, b) in enumerate(zip(r, s)) if not _same_cell(a, b)]
        assert not bad and len(loaded.rows) == len(original.rows), bad[:5]  # every number exactly
    else:
        assert isinstance(original, Figure) and isinstance(loaded, Figure)
        pd.testing.assert_frame_equal(loaded.data(), original.data(), check_dtype=False, check_index_type=False,
                                      check_column_type=False)
        assert [p.title for p in loaded.panels] == [p.title for p in original.panels]
        assert [[type(x).__name__ for x in p.layers] for p in loaded.panels] == [[type(x).__name__ for x in p.layers] for p in original.panels]


def test_every_kind_of_result_comes_back_with_the_same_views(objects, tmp_path):
    failed = []
    for kind, obj in objects.items():
        try:
            path = save_result(obj, tmp_path / kind)
            saved = load_result(path)
            again_saved = isinstance(obj, SavedResult)  # saved again: its own views, without the Saved tab, and its kind
            original = list(obj.views) if again_saved else view_sections(obj)
            assert saved.kind == (obj.kind if again_saved else kind) and saved.name == kind and path.suffix == ".pymxr"
            assert [t for t, _ in view_sections(saved)] == [t for t, _ in original] + ["Saved"]
            for (_, a), (_, b) in zip(original, saved.views):
                _same_view(a, b)
            assert saved.summary() == summary_text(obj)
            with zipfile.ZipFile(path) as z:  # plain files only: nothing to unpickle or run
                assert all(n.endswith((".json", ".txt")) for n in z.namelist()), z.namelist()
                assert json.loads(z.read("manifest.json"))["not_saved"] == []
            again = result_from_bytes(result_bytes(saved))  # saved again, as read
            assert [t for t, _ in again.views] == [t for t, _ in saved.views] and again.text == saved.text and again.kind == saved.kind
        except Exception as exc:  # noqa: BLE001 - the test lists every failure
            failed.append(f"{kind}: {type(exc).__name__}: {exc}")
    assert not failed, failed


def test_saved_figures_redraw_and_saved_results_show_in_a_notebook(objects, tmp_path):
    fit = objects["FittedEquation"]
    saved = load_result(save_result(fit, tmp_path / "cons"))
    figure = dict(saved.views)["Graph"]
    assert figure.png()[1:4] == b"PNG" and figure.with_options(theme="policy").png()[1:4] == b"PNG"
    assert saved.to_frame("coefficients").equals(fit.to_frame("Coefficients"))
    assert "<table" in saved._repr_html_() and "FittedEquation" in repr(saved)
    assert saved.provenance["sample"] and saved.provenance["type"] == "Provenance"  # how it was made, as data


def test_saved_files_are_checked_when_read(tmp_path):
    bad = tmp_path / "bad.pymxr"
    bad.write_bytes(b"not a zip file")
    with pytest.raises(SpecificationError, match="not a saved pymetrix result"):
        load_result(bad)
    newer = tmp_path / "newer.pymxr"
    with zipfile.ZipFile(newer, "w") as z:
        z.writestr("manifest.json", json.dumps({"format_version": 99, "package_version": "9.0", "views": []}))
    with pytest.raises(SpecificationError, match="newer version of pymetrix"):
        load_result(newer)
    with pytest.raises(SpecificationError, match="no views to save"):
        save_result(SavedResult("Nothing", "nothing", "", "", ()), tmp_path / "empty")


def test_open_result_is_a_step_and_projects_keep_the_file(objects, tmp_path):
    from pymetrix.commands import OpenResult, Session
    from pymetrix.commands.project import load_project, save_project

    path = save_result(objects["FittedEquation"], tmp_path / "cons")
    session = Session()
    session.run(OpenResult(str(path), "theirs"))
    assert isinstance(session["theirs"], SavedResult) and session.history[-1].code == f"theirs = load_result({str(path)!r})"
    project = save_project(session, tmp_path / "project")
    path.unlink()  # the project opens without the file, and without being trusted (it holds no code)
    reopened = load_project(project, trust=False)
    assert [t for t, _ in reopened["theirs"].views] == [t for t, _ in session["theirs"].views]
    assert reopened["theirs"].text == session["theirs"].text and reopened.history[-1].ok
    assert save_project(reopened, tmp_path / "again").is_file() and load_project(tmp_path / "again.pymx", trust=False)["theirs"].kind == "FittedEquation"
    namespace = {}
    exec("from pymetrix import *", namespace)  # the exported script's header: its load_result line runs as written
    assert namespace["load_result"] is load_result and namespace["save_result"] is save_result
