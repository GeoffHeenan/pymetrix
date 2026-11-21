"""The pre-registered analysis plan: locked before the data are read, each later step labelled planned or exploratory,
and verify's deviation report."""

import pytest
from pymetrix_labbook import current_plan, lab_book, lock_plan, register, verify
from pymetrix_labbook.plan import normalise, plan_report

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session, save_project

register()


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def test_specifications_match_whatever_the_result_is_named():
    assert normalise("fit = Equation('c ~ y').fit(data)") == normalise("other  =  Equation('c ~ y').fit(data)")
    assert normalise("fit = Equation('c ~ y').fit(data)") != normalise("fit = Equation('c ~ y[-1]').fit(data)")
    assert normalise("x == 1") == "x==1"  # a comparison is not an assignment


def test_a_plan_locked_first_and_followed_verifies_clean(session, tmp_path):
    s = session
    entry = lock_plan(s, [FitEquation("c ~ y", name="fit")], data=["data"], sample="whole sample", hypotheses=["H1"])
    assert entry["event"] == "plan" and len(entry["data"]["plan_hash"]) == 64
    s.run(FitEquation("c ~ y", name="main"))  # named differently: still the planned specification
    labels = [e["data"].get("plan") for e in lab_book(s).entries if e["event"] == "step"]
    assert labels == ["planned"]
    result = verify(save_project(s, tmp_path / "study"))
    assert result.ok and not [w for w in result.warnings if "plan" in w], str(result)
    assert any("locked at entry 1, before its data were read" in c for c in result.checked)
    assert any("every planned specification was run (1)" in c for c in result.checked)


def test_the_deviation_report(session):
    s = session
    s.run(FitEquation("c ~ y[-1]", name="peek"))  # the data read before the plan
    lock_plan(s, ["fit = Equation('c ~ y').fit(data, method=OLS())", "fit2 = Equation('c ~ y + y[-1]').fit(data, method=OLS())"],
              data=["data"])
    s.run(FitEquation("c ~ y", name="fit"))
    s.run(FitEquation("c ~ y[-2]", name="extra"))  # not in the plan
    with pytest.raises(ValueError, match="reason"):
        lock_plan(s, ["fit = Equation('c ~ y').fit(data, method=OLS())"])
    lock_plan(s, ["fit = Equation('c ~ y').fit(data, method=OLS())", "fit3 = Equation('c ~ y[-2]').fit(data, method=OLS())"],
              data=["data"], reason="the lag of two fits better")
    warnings, notes, _ = plan_report(lab_book(s))
    text = " | ".join(warnings)
    assert "the planned data data were read at entry 1, before the plan was locked at entry 2" in text
    assert "the plan was changed at entry 5, after 2 step(s) since it was locked: the lag of two fits better" in text
    assert not any("not run" in w for w in warnings)  # the changed plan's specifications were both run
    assert any("1 step(s) since the plan was locked are exploratory" in n for n in notes)
    assert current_plan(s)["reason"] == "the lag of two fits better"
    labels = [e["data"].get("plan") for e in lab_book(s).entries if e["event"] == "step"]
    assert labels == [None, "planned", "exploratory"]


def test_planned_specifications_not_run_are_reported(session):
    lock_plan(session, ["fit = Equation('c ~ y').fit(data, method=OLS())"])
    warnings, _, _ = plan_report(lab_book(session))
    assert any("1 planned specification(s) not run" in w for w in warnings)
    with pytest.raises(ValueError, match="at least one"):
        lock_plan(session, [], reason="x")
