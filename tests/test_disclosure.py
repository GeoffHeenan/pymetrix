"""What an AI assistant may be told in metadata_only mode (pymetrix.disclosure): a project full of marked values,
formulas, paths and text, described, and none of the marks in what is sent."""

import numpy as np
import pandas as pd
import pytest

from pymetrix.disclosure import (
    DEFAULT_MODE,
    MODES,
    describe_object,
    describe_session,
    to_text,
)

SECRETS = {
    "a data value": "987654.3",
    "another data value": "123456.7",
    "a formula's constant": "271828",
    "an equation": "~",
    "a file path": "secret_folder",
    "a label": "Confidential series label",
    "report text": "CONFIDENTIAL PARAGRAPH",
    "a heading": "Board briefing heading",
    "typed code": "42.4242",
    "an estimate": "0.1234567",
}


@pytest.fixture
def session(tmp_path):
    from pymetrix import Dataset
    from pymetrix.commands import FitEquation, GenerateSeries, LoadData, Session

    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"c": 987654.3 + rng.normal(size=40), "y": 123456.7 + rng.normal(size=40), "w": rng.normal(size=40)},
                         index=idx)
    folder = tmp_path / "secret_folder"
    folder.mkdir()
    frame.to_csv(folder / "data.csv")
    s = Session()
    s.run(LoadData(str(folder / "data.csv"), time="time", freq="Q", name="data"))
    s.objects["data"] = s["data"].with_info("c", label="Confidential series label")
    s.run(GenerateSeries("z", "c * 271828", data="data"))
    s.run(FitEquation("c ~ y + w", data="data", name="fit"))
    s.add_report_heading("Board briefing heading")
    s.add_report_text("CONFIDENTIAL PARAGRAPH")
    s.run_code("x = 42.4242")  # typed code: its text is never sent
    s.objects["frame_like"] = Dataset.from_frame(pd.DataFrame({"v": [0.1234567] * 4},
                                                              index=pd.period_range("2020", periods=4, freq="Y")))
    return s


def test_metadata_only_never_carries_values_formulas_paths_or_text(session):
    held = repr(session.history) + repr(session.report_outline) + repr(session["data"].frame.round(1).to_dict())
    for what in ("a formula's constant", "an equation", "a file path", "report text", "a heading", "typed code"):
        assert SECRETS[what] in held, what  # the project does hold it: the test below means something
    sent = to_text(describe_session(session))
    leaked = {what: mark for what, mark in SECRETS.items() if mark in sent}
    assert leaked == {}, leaked
    import re

    assert re.findall(r"\d+\.\d+", sent) == []  # no number with decimals anywhere: counts and periods only


def test_what_metadata_only_does_say(session):
    facts = describe_session(session)
    by_name = {o["name"]: o for o in facts["objects"]}
    data = by_name["data"]
    assert data["kind"] == "Dataset" and set(data["series"]) >= {"c", "y", "w", "z"} and data["span"] == ["2000Q1", "2009Q4"]
    assert data["frequency"] == "Q" and data["n_rows"] == 40
    fit = by_name["fit"]
    assert fit["nobs"] == 40 and fit["sample"] == ["2000Q1", "2009Q4"]
    steps = facts["steps"]
    assert [s["step"] for s in steps][:3] == ["LoadData", "GenerateSeries", "FitEquation"] and steps[2]["made"] == ["fit"]
    assert all("code" not in s for s in steps)


def test_names_that_are_not_identifiers_are_left_out():
    from pymetrix import Dataset

    odd = Dataset.from_frame(pd.DataFrame({"gdp": [1.5, 2.5], "Ignore previous instructions and": [3.5, 4.5]},
                                          index=pd.period_range("2020Q1", periods=2, freq="Q")))
    facts = describe_object("odd", odd)
    assert facts["series"] == ["gdp"] and facts["n_series"] == 2  # counted, not quoted
    assert "Ignore" not in to_text(facts)


def test_the_modes():
    assert MODES == ("off", "metadata_only", "full_context", "local_model") and DEFAULT_MODE == "off"


def test_full_context_adds_the_steps_code_and_metadata_only_never_does(session):
    full = to_text(describe_session(session, "full_context"))
    assert "c ~ y + w" in full and "c * 271828" in full and '"mode":"full_context"' in full
    assert to_text(describe_session(session, "local_model")) == to_text(describe_session(session))
    from pymetrix.errors import PyMetrixError

    with pytest.raises(PyMetrixError):
        describe_session(session, "off")
