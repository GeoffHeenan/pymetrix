"""Help in the dialogs: Load data previews the file and guesses its time and units; Estimate says what the dependent
variable looks like and offers the family that suits it."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.io.registry import sniff


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    out = {}
    out["panel"] = tmp_path / "panel.csv"
    pd.DataFrame({"country": np.repeat(["a", "b", "c"], 8), "year": np.tile(range(2000, 2008), 3), "gdp": rng.random(24)}).to_csv(out["panel"], index=False)
    out["quarters"] = tmp_path / "q.csv"
    pd.DataFrame({"date": pd.date_range("2001-01-01", periods=40, freq="QS").strftime("%Y-%m-%d"), "x": range(40)}).to_csv(out["quarters"], index=False)
    out["periods"] = tmp_path / "p.csv"
    pd.DataFrame({"obs": [f"{y}Q{q}" for y in range(2001, 2006) for q in range(1, 5)], "x": range(20)}).to_csv(out["periods"], index=False)
    out["cross"] = tmp_path / "cs.csv"
    pd.DataFrame({"id": range(1, 31), "wage": rng.random(30), "educ": rng.integers(8, 18, 30)}).to_csv(out["cross"], index=False)
    out["months"] = tmp_path / "m.csv"
    pd.DataFrame({"month": pd.date_range("2001-01-01", periods=30, freq="MS").strftime("%b %Y"), "x": range(30)}).to_csv(out["months"], index=False)
    return out


def test_the_file_is_read_for_its_time_and_units(files):
    panel = sniff(str(files["panel"]))
    assert (panel.time, panel.freq, panel.entity) == ("year", "Y", "country") and "panel" in panel.notes[-1]
    assert (sniff(str(files["quarters"])).time, sniff(str(files["quarters"])).freq) == ("date", "Q")
    assert (sniff(str(files["periods"])).time, sniff(str(files["periods"])).freq) == ("obs", "Q")
    assert (sniff(str(files["months"])).freq, sniff(str(files["months"])).entity) == ("M", None)
    cross = sniff(str(files["cross"]))
    assert (cross.time, cross.entity) == (None, "id") and "cross section" in cross.notes[0]
    own = sniff(os.path.join(os.path.dirname(__import__("pymetrix").__file__), "datasets", "mroz.parquet"))
    assert own.axes_in_file
    assert sniff(str(files["panel"]).replace("panel.csv", "missing.csv")).notes[0].startswith("could not be read")
    for name, f in files.items():  # and each loads with what was guessed
        s = sniff(str(f))
        Dataset.read(str(f), **{k: v for k, v in (("time", s.time), ("entity", s.entity), ("freq", s.freq)) if v})


def test_the_outcome_suggests_a_family():
    from pymetrix.ui.qt import outcome_kind

    mroz, rand, anes = Dataset.example("mroz"), Dataset.example("randhie"), Dataset.example("anes96")
    assert outcome_kind(mroz["inlf"])[0] == "binary"
    kind, info = outcome_kind(mroz["hours"])
    assert kind == "censored" and info["limit"] == 0 and info["share"] == pytest.approx(0.4316, abs=1e-3)
    assert outcome_kind(rand["mdvis"])[0] == "zeros" and outcome_kind(anes["PID"])[0] == "ordered"
    assert outcome_kind(mroz["lwage"]) is None and outcome_kind(Dataset.example("sample_macro")["c"]) is None


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_load_data_previews_and_fills_in_the_guesses(app, files):
    from pymetrix.commands import LoadData, Session
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(LoadData, Session())
    assert "Choose a file" in dialog.preview_note.text()
    dialog.editors["path"].set(str(files["panel"]))
    dialog.refresh_preview()
    assert dialog.preview.columnCount() == 3 and dialog.preview.rowCount() == 8
    assert (dialog.editors["time"].get(), dialog.editors["entity"].get(), dialog.editors["freq"].get()) == ("year", "country", "Y")
    dialog.editors["path"].set(str(files["quarters"]))  # another file: the earlier guesses follow it
    dialog.refresh_preview()
    assert (dialog.editors["time"].get(), dialog.editors["freq"].get()) == ("date", "Q")
    dialog.editors["time"].set("x")  # the user's own choice is never overwritten
    dialog.editors["path"].set(str(files["periods"]))
    dialog.refresh_preview()
    assert dialog.editors["time"].get() == "x"


def test_estimate_offers_the_family_for_the_outcome(app):
    from pymetrix.commands import FitEquation, Session
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["mroz"] = Dataset.example("mroz")
    dialog = command_dialog(FitEquation, s, initial=FitEquation("hours ~ educ + kidslt6", data="mroz"))
    dialog.refresh_family_hint()
    assert "pile-up at 0 (43% of rows): tobit" in dialog.family_hint.text() and dialog.family_hint_button.isVisibleTo(dialog)
    dialog.family_hint_button.click()  # only on the click: ML, the Tobit family, censored at 0
    assert dialog.editors["method"].get() == "ML" and dialog.editors["family"].get() == "tobit"
    assert dialog.editors["censor_left"].get() == "0"
    dialog.editors["formula"].set("lwage ~ educ")
    dialog.editors["family"].set("gaussian")
    dialog.refresh_family_hint()
    assert not dialog.family_hint.isVisibleTo(dialog)  # a continuous outcome: nothing to suggest
