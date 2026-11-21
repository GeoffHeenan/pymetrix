"""A saved result's provenance carries the call that made it (function, arguments as recorded, span and row condition as
given, the datasets it used by name): Call(**provenance["call"]).invoke(objects) gives the saved numbers again."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pymetrix import Equation, OLS
from pymetrix.calls import Call, call_of, make_call
from pymetrix.codec import Ref
from pymetrix.commands import FitEquation, FitVAR, LoadExample, Session, load_project, save_project
from pymetrix.errors import SpecificationError
from pymetrix.saved import load_result, result_bytes, result_from_bytes, save_result


@pytest.fixture
def session():
    s = Session()
    s.run(LoadExample())
    return s


def _again(saved, objects):
    call = Call(**saved.provenance["call"])
    return call.invoke({name: objects[name] for name in saved.provenance["call_uses"]})


def _same_table(a, b):
    """Two coefficient tables with the same numbers (a saved table's row labels come back as text)."""
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def _same_views(a, b):
    """The saved views of two results hold the same numbers (tables cell by cell; figures series by series). The
    Provenance tab's "created" row is when each result was made, to the minute, so it differs when the two are made
    either side of a minute's turn; it is left out."""
    assert [t for t, _ in a.views] == [t for t, _ in b.views]
    from pymetrix.output import Table

    for (_, x), (_, y) in zip(a.views, b.views):
        if isinstance(x, Table):
            assert x.columns == y.columns and len(x.rows) == len(y.rows)
            for rx, ry in zip(x.rows, y.rows):
                if rx and str(rx[0]) == "created" and str(ry[0]) == "created":
                    continue
                assert [str(v) for v in rx] == [str(v) for v in ry]


def test_a_dialogs_fit_with_span_and_condition(session, tmp_path):
    session.run(FitEquation("c ~ y + c[-1]", name="fit", start="1990Q1", end="2005Q4", where="y > 0"))
    saved = load_result(save_result(session["fit"], tmp_path / "fit"))
    call = saved.provenance["call"]
    assert call["function"] == "pymetrix:equation.fit" and call["output"] == "fit"
    assert call["arguments"]["span"] == {"$tuple": ["1990Q1", "2005Q4"]} and call["arguments"]["where"] == "y > 0"  # as given
    assert call["arguments"]["data"] == {"$ref": "data"} and saved.provenance["call_uses"] == ["data"]  # the dataset's name
    again = _again(saved, session.objects)
    _same_table(again.coefficients(), saved.coefficients())  # the saved numbers, exactly
    assert again.stats["nobs"] == saved.stats["nobs"] == session["fit"].stats["nobs"]


def test_a_var_and_typed_code_recorded_as_calls(session, tmp_path):
    session.run(FitVAR(variables="c, y", lags="2", name="var"))
    session.run_code("hac = Equation('c ~ y').fit(data, method=OLS(cov='hac', hac_lags=3))")
    for name in ("var", "hac"):
        saved = load_result(save_result(session[name], tmp_path / name))
        again = _again(saved, session.objects)
        _same_table(again.coefficients(), saved.coefficients())
        _same_views(saved, result_from_bytes(result_bytes(again, name)))
    se = load_result(tmp_path / "hac.pymxr").coefficients()["std_error"].to_numpy()
    assert np.array_equal(se, session["hac"].coefficients()["std_error"].to_numpy())  # HAC, as recorded


def test_a_forecast_from_a_named_result(session, tmp_path):
    session.run_code("fit = Equation('c ~ y + c[-1]').fit(data)")
    session.run_code("fc = fit.forecast(data, ('2006Q1', '2007Q4'))")
    saved = load_result(save_result(session["fc"], tmp_path / "fc"))
    assert saved.provenance["call_uses"] == ["data", "fit"]
    again = _again(saved, session.objects)
    pd.testing.assert_frame_equal(again.frame, session["fc"].frame)
    _same_views(saved, result_from_bytes(result_bytes(again, "fc")))


def test_results_not_made_by_a_recorded_call_are_saved_as_before(session, tmp_path):
    session.run_code("made = [Equation('c ~ y').fit(data)][0]")  # typed code kept as code: no call
    assert call_of(session["made"]) is None
    with zipfile.ZipFile(save_result(session["made"], tmp_path / "made")) as z:
        provenance = json.loads(z.read("provenance.json"))
    assert "call" not in provenance and "call_uses" not in provenance
    outside = Equation("c ~ y").fit(session["data"])  # outside a session: nothing recorded
    assert call_of(outside) is None and "call" not in load_result(save_result(outside, tmp_path / "out")).provenance


def test_a_call_given_for_a_result_made_outside_a_session(session, tmp_path):
    data = session["data"]
    fit = Equation("c ~ y").fit(data, span=("1985Q1", "2000Q4"), method=OLS(cov="white"))
    call = make_call("pymetrix:equation.fit", Equation("c ~ y"), data=Ref("data"), span=("1985Q1", "2000Q4"),
                     method=OLS(cov="white"))
    saved = load_result(save_result(fit, tmp_path / "fit", call=call))
    again = Call(**saved.provenance["call"]).invoke({"data": data})
    pd.testing.assert_frame_equal(again.coefficients(), fit.coefficients())
    with pytest.raises(SpecificationError):
        save_result(fit, tmp_path / "bad", call="Equation('c ~ y').fit(data)")


def test_saved_again_and_from_a_reopened_project(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit", start="1990Q1"))
    first = load_result(save_result(session["fit"], tmp_path / "a"))
    second = load_result(save_result(first, tmp_path / "b"))  # a saved result saved again keeps its call
    assert second.provenance["call"] == first.provenance["call"]
    reopened = load_project(save_project(session, tmp_path / "p.pymx"))
    third = load_result(save_result(reopened["fit"], tmp_path / "c"))
    assert third.provenance["call"] == first.provenance["call"]
    _same_table(_again(third, reopened.objects).coefficients(), first.coefficients())


def test_robust_and_restricted_fits_keep_their_calls(session, tmp_path):
    """The newer methods' settings are in the recorded call: robust regression (dialog) and restricted least squares
    (OLS by dialog, IV by typed code) come back with the same coefficients and standard errors."""
    session.run(FitEquation("c ~ y + c[-1]", method="Robust regression (M)", method_settings="norm='bisquare', tuning=4.0",
                            name="rob"))
    session.run(FitEquation("c ~ y + c[-1]", method="OLS", method_settings="restrictions=['y + c[-1] = 1']", name="rls"))
    session.run_code("riv = Equation('c ~ y + i').fit(data, method=IV(endog=['y'], instruments=['g', 'c[-1]'], "
                     "restrictions=['y + i = 1']))")
    for name, kind in (("rob", "RobustLS"), ("rls", "OLS"), ("riv", "IV")):
        saved = load_result(save_result(session[name], tmp_path / name))
        assert kind in json.dumps(saved.provenance["call"]["arguments"])
        again = _again(saved, session.objects)
        _same_table(again.coefficients(), saved.coefficients())
        np.testing.assert_array_equal(again.coefficients()["std_error"].to_numpy(),
                                      session[name].coefficients()["std_error"].to_numpy())
    assert "restrictions" in json.dumps(load_result(tmp_path / "rls.pymxr").provenance["call"]["arguments"])


def test_a_solution_with_the_solver_options_keeps_its_call(tmp_path):
    """Solve model with the convergence test, line search, kept Jacobian, fallback and ordering by blocks: the saved
    call solves again to the same numbers."""
    from pymetrix import Dataset, Model
    from pymetrix.commands import SolveModel

    model = Model([Equation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", name="cons"),
                   Equation("i = 0.2*y - 0.5*r + 0.4*i[-1]", name="inv"),
                   Equation("y = c + i + g", name="gdp"),
                   Equation("w = 0.9*w[-1] + 0.1*y", name="wage")])
    index = pd.period_range("2019Q1", "2024Q4", freq="Q", name="time")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({v: 1.0 + 0.1 * rng.standard_normal(len(index)) for v in ("c", "i", "y", "w", "g", "r")}, index=index)
    frame.loc["2022Q1":, ["c", "i", "y", "w"]] = np.nan
    s = Session()
    s.objects["m"], s.objects["data"] = model, Dataset.from_frame(frame)
    s.run(SolveModel("m", "2022Q1", "2023Q4", solver_settings="convergence='step', line_search=True, refresh='period'",
                     ordering="blocks", fallback=True, name="solved"))
    saved = load_result(save_result(s["solved"], tmp_path / "solved"))
    text = json.dumps(saved.provenance["call"]["arguments"])
    assert "Fallback" in text and '"blocks"' in text and "line_search" in text
    again = _again(saved, s.objects)
    np.testing.assert_array_equal(again.frame.to_numpy(float), s["solved"].frame.to_numpy(float))
    pd.testing.assert_frame_equal(again.convergence, s["solved"].convergence)
