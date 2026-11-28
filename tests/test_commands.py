import numpy as np
import pytest

from pymetrix import Model
from pymetrix.commands import COMMANDS, CommandError, FitEquation, RunTest, Session, form_spec
from pymetrix.ui.render import to_html


def test_forms_come_from_field_types():
    specs = {s.name: s for s in form_spec(FitEquation)}
    assert specs["formula"].kind == "formula" and specs["formula"].required
    assert specs["cov"].kind == "choice" and specs["cov"].choices == ("classic", "white", "hc0", "hc2", "hc3", "hac", "cluster")
    assert specs["data"].kind == "object" and specs["data"].source == "Dataset"
    assert "Estimate/Equation..." in {i.menu for i in COMMANDS.values()}


def test_session_records_a_replayable_script(ts):
    session = Session()
    session.objects["data"] = ts
    fit = session.run(FitEquation("y ~ x + z", cov="white", start="2005Q1", end="2020Q4"))
    session.run(RunTest(source="fit", test="jarque_bera"))
    script = session.script()
    assert "Equation('y ~ x + z').fit(data, span=('2005Q1', '2020Q4'), method=OLS(cov='white'))" in script

    namespace = {"data": ts}
    exec(script, namespace)
    assert np.allclose(namespace["fit"].params.values, fit.params.values)
    assert np.isclose(namespace["test"].statistic, session["test"].statistic)


def test_failed_commands_stay_out_of_the_script(ts):
    session = Session()
    session.objects["data"] = ts
    try:
        session.run(FitEquation("y ~ nothing"))
    except Exception:
        pass
    assert session.history[-1].ok is False
    assert "nothing" not in session.script()


def test_test_choices_follow_the_target(ts):
    session = Session()
    session.objects["data"] = ts
    session.run(FitEquation("y ~ x"))
    specs = {s.name: s for s in form_spec(RunTest)}
    assert session.choices(specs["source"], {}) == ["data", "fit"]
    assert "breusch_godfrey" in session.choices(specs["test"], {"source": "fit"})
    assert {"adf", "kpss", "dfgls"} <= set(session.choices(specs["test"], {"source": "data"}))

    pytest.importorskip("statsmodels")  # the ADF test is computed by statsmodels
    result = session.run(RunTest(source="data", test="adf", options="yd, lags=2"))
    assert result.details["lags used"] == 2
    assert "test = data.test('adf', 'yd', lags=2)" in session.script()
    with pytest.raises(CommandError, match="does not apply"):
        session.run(RunTest(source="data", test="jarque_bera"))


def test_render(ts):
    session = Session()
    session.objects["data"] = ts
    fit = session.run(FitEquation("y ~ x"))
    assert "<table" in to_html(fit)
    assert "Dataset" in to_html(ts)
    solution = Model([fit]).solve(ts, span=("2010Q1", "2010Q4"))
    html = to_html(solution)
    assert "<Solution 2010Q1 to 2010Q4" in html.replace("&lt;", "<")
    assert "2009Q4" not in html and "2010Q4" in html


def test_a_start_after_the_end_is_refused(ts):
    from pymetrix.commands import CommandError, FitVAR, Session

    s = Session()
    s.objects["data"] = ts
    with pytest.raises(CommandError, match=r"the start \(2010Q1\) is after the end \(2005Q4\)"):
        s.run(FitVAR("y, x", start="2010Q1", end="2005Q4"))
    assert s.period_choices({"data": "data"})[:2] == ["2000Q1", "2000Q2"]
    s.run(FitVAR("y, x", start="2005Q1", end="2010Q4"))  # in order: fine


def test_default_names_do_not_shadow_python_or_pymetrix():
    """A generated script starts `from pymetrix import *`: `backtest = backtest(...)` would break the next call."""
    import builtins
    import dataclasses

    import pymetrix
    from pymetrix.commands.registry import COMMAND_IDS

    taken = set(pymetrix.__all__) | set(dir(builtins))
    clashes = [(info.cls.__name__, f.default) for info in COMMAND_IDS.values() for f in dataclasses.fields(info.cls)
               if f.name == "name" and isinstance(f.default, str) and f.default in taken]
    assert not clashes


def test_every_call_id_a_dialog_names_is_recorded():  # the Proxy SVAR dialog once named an id that did not exist
    import re
    from pathlib import Path

    import pymetrix.commands.builtin  # noqa: F401 - registers the dialogs' calls
    from pymetrix.calls import API_CALLS

    src = Path(__file__).resolve().parents[1] / "src" / "pymetrix"
    named = {(p.name, i) for p in src.rglob("*.py") for i in re.findall(r'make_call\(\s*"(pymetrix:[^"]+)"', p.read_text(encoding="utf-8"))}
    assert not [(f, i) for f, i in named if i not in API_CALLS]


def test_a_test_without_its_required_settings_says_which(tmp_path):
    from pymetrix import Dataset

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    with pytest.raises(Exception, match=r"the adf test needs variable: give variable=\.\.\."):
        session.run(RunTest(source="data", test="adf"))  # was Python's "ADF.__init__() missing 1 required positional argument"
    assert session.run(RunTest(source="data", test="adf", options="variable=c")) is not None
