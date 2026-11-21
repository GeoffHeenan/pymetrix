"""No API argument or dialog field is named ``target``: each was renamed, and the old name still works (with a
DeprecationWarning for keywords, silently for saved settings, calls and dialog steps)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import BMA, VAR, Dataset, Equation, MaxShare, backtest
from pymetrix.codec import decode, encode
from pymetrix.commands.registry import from_json


def _data(T=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    y = 0.5 * x + rng.normal(size=T)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "z": rng.normal(size=T)},
                                           index=pd.period_range("1990Q1", periods=T, freq="Q", name="time")))


def _old(match):
    return pytest.warns(DeprecationWarning, match=f"'target' of {match} is now called")


def test_settings_take_the_old_keyword_and_read_old_saved_fields():
    with _old(r"BMA\(\)"):
        assert BMA(target="y", candidates=("x",)).outcome == "y"
    with _old(r"MaxShare\(\)"):
        assert MaxShare(target="tfp").variable == "tfp"
    for value, new in ((BMA("y", ("x", "z")), "outcome"), (MaxShare("tfp", horizon=12), "variable")):
        saved = encode(value)
        saved["target"] = saved.pop(new)  # as a project saved before the rename has it
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert decode(saved) == value


def test_functions_take_the_old_keyword():
    from pymetrix.adapters.statsmodels import pymetrix_name
    from pymetrix.help import help_page
    from pymetrix.proxy import proxy_impact
    from pymetrix.stats import available_tests, run_test

    data = _data()
    fit = Equation("y ~ x").fit(data)
    with _old(r"run_test\(\)"):
        old = run_test(target=fit, test="jarque_bera")
    assert old.statistic == run_test(fit, "jarque_bera").statistic
    with _old(r"available_tests\(\)"):
        assert available_tests(target=fit) == available_tests(fit)
    with _old(r"help_page\(\)"):
        assert help_page(target=Equation).title == help_page(Equation).title
    with _old(r"pymetrix_name\(\)"):
        assert pymetrix_name("ar.L2", target="y") == "y[-2]" == pymetrix_name("ar.L2", "y")
    rng = np.random.default_rng(1)
    U, z = rng.normal(size=(200, 2)), rng.normal(size=200)
    with _old(r"proxy_impact\(\)"):
        old = proxy_impact(U, z, target=0, sigma=np.eye(2))
    assert old == pytest.approx(proxy_impact(U, z, 0, np.eye(2)))


def test_writing_a_dataset_takes_path(tmp_path):
    data = _data(T=8)
    data.write(str(tmp_path / "new.csv"))
    with _old(r"Dataset\.write\(\)"):
        data.write(target=str(tmp_path / "old.csv"))
    assert (tmp_path / "new.csv").read_text() == (tmp_path / "old.csv").read_text()


def test_backtest_takes_variable_and_replays_old_calls():
    from pymetrix.calls import Call
    from pymetrix.commands import Session

    data = _data()
    fit = Equation("y ~ x").fit(data)
    new = backtest(fit, data, ("2010Q1", "2012Q4"), horizon=2, variable="out")
    with _old(r"backtest\(\)"):
        old = backtest(fit, data, ("2010Q1", "2012Q4"), horizon=2, target="out")
    pd.testing.assert_frame_equal(old.errors, new.errors)
    assert set(new.errors["variable"]) == {"out"}

    s = Session()
    s.objects.update(data=data, fit=fit)
    s.run_code('bt = backtest(fit, data, ("2010Q1", "2012Q4"), horizon=2, variable="out")')
    call = s.history[-1].command
    assert call.version == 2 and call.arguments["variable"] == "out"
    saved = dict(call.arguments)  # the same call saved at version 1, with target
    saved["target"] = saved.pop("variable")
    s.run(Call("pymetrix:backtest", 1, None, saved, output="older"))
    pd.testing.assert_frame_equal(s.objects["older"].errors, new.errors)


@pytest.mark.parametrize("command_id, fields, new, value", [
    ("pymetrix:correlogram", {"target": "data", "variable": "y"}, "source", "data"),
    ("pymetrix:run_test", {"target": "fit", "test": "jarque_bera"}, "source", "fit"),
    ("pymetrix:fit_bma", {"target": "y", "candidates": "x, z"}, "outcome", "y"),
])
def test_dialog_steps_saved_with_target_still_open(command_id, fields, new, value):
    step = from_json({"kind": "command", "id": command_id, "version": 1, "fields": fields})
    assert getattr(step, new) == value and not hasattr(step, "target")


def test_the_dialogs_run_under_their_new_field_names():
    from pymetrix.commands import Correlogram, RunTest, Session
    from pymetrix.commands.statespace import FitBMA

    data = _data()
    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("y ~ x").fit(data)
    s.objects["var"] = VAR(["y", "x"], lags=1).fit(data)
    s.run(Correlogram(source="data", variable="y", lags=4))
    s.run(RunTest(source="fit", test="jarque_bera"))
    s.run(FitBMA(outcome="y", candidates="x, z"))
    assert all(r.ok for r in s.history[-3:])
    assert "data.correlogram('y', lags=4)" in s.history[-3].code and "fit.test('jarque_bera')" in s.history[-2].code


def test_results_name_their_dependent_variable_and_keep_target_as_an_old_name():
    import pickle

    from pymetrix.ardl import FittedARDL
    from pymetrix.proxy import ProxySVAR

    data = _data()
    fit = Equation("y ~ x").fit(data)
    assert fit.dependent == "y" == Equation("y ~ x").dependent
    with pytest.warns(DeprecationWarning, match=r"FittedEquation\.target is deprecated .* use \.dependent"):
        assert fit.target == "y"
    with pytest.warns(DeprecationWarning, match=r"Equation\.target is deprecated"):
        assert Equation("y ~ x").target == "y"

    psvar = VAR(["y", "x"], lags=1).fit(data).proxy_svar(data, "z", draws=0)
    assert psvar.normalise_on == "y"
    with pytest.warns(DeprecationWarning, match=r"ProxySVAR\.target is deprecated .* use \.normalise_on"):
        assert psvar.target == "y"
    # a result pickled (as projects cache slow steps) before the rename has its old field name
    for obj, new in ((psvar, "normalise_on"), (object.__new__(FittedARDL), "dependent")):
        state = dict(obj.__dict__)
        state["target"] = state.pop(new, "y")
        restored = object.__new__(type(obj))
        restored.__setstate__(state)
        assert getattr(restored, new) == "y" and "target" not in restored.__dict__
    assert pickle.loads(pickle.dumps(psvar)).normalise_on == "y"


def test_records_rename_target_and_read_what_was_saved_with_it():
    from pymetrix import templates
    from pymetrix.calls import Call, _inner_call
    from dataclasses import replace

    from pymetrix.figures import StyleCall

    # a recorded call: its receiver, from a step and from a nested call saved before the rename
    step = from_json({"kind": "command", "id": "pymetrix:call", "version": 1,
                      "fields": {"function": "pymetrix:equation.fit", "version": 1, "target": {"$ref": "eq"},
                                 "arguments": {"data": {"$ref": "data"}}, "output": "fit"}})
    from pymetrix.codec import Ref

    assert step.receiver == Ref("eq")
    with pytest.warns(DeprecationWarning, match=r"Call\.target is deprecated .* use \.receiver"):
        assert step.target == Ref("eq")
    nested = {"function": "pymetrix:equation.fit", "version": 1, "target": {"$ref": "eq"}, "arguments": {}}
    assert _inner_call(nested).receiver == {"$ref": "eq"}
    with _old(r"Call\(\)"):
        assert Call("pymetrix:equation.fit", 1, target={"$ref": "eq"}).receiver == {"$ref": "eq"}

    # a style call: its kind, from a dict saved before the rename (a result file, a house style)
    old = {"target": "set", "where": [0, "left"], "props": {"ylabel": "%"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = replace(_data(T=8).plot("y"), style=(old,))
    assert figure.style[0] == StyleCall("set", (0, "left"), {"ylabel": "%"}) and figure.style[0].to_dict()["kind"] == "set"
    with pytest.warns(DeprecationWarning, match=r"StyleCall\.target is deprecated .* use \.kind"):
        assert figure.style[0].target == "set"

    # a model's equations: the variable each determines
    first = templates.new_keynesian().model.model_equations[0]
    assert first.dependent == "y"
    with pytest.warns(DeprecationWarning, match=r"ModelEquation\.target is deprecated .* use \.dependent"):
        assert first.target == "y"


def test_require_dependent_was_require_target():
    from pymetrix.expr import ExpressionError

    assert Equation("y ~ x").require_dependent() == "y"
    with pytest.warns(DeprecationWarning, match=r"require_target\(\) is deprecated .* use \.require_dependent\(\)"):
        assert Equation("y ~ x").require_target() == "y"
    with pytest.raises(ExpressionError, match="exactly one current-period variable"):
        Equation("y + z ~ x").require_dependent()


def _named_target(fn):
    import inspect

    try:
        return "target" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def test_no_public_function_or_dialog_field_is_named_target():
    """The standing rule: no recorded call, public function or method, or dialog field takes an argument named
    ``target`` (the old names live on only as deprecated aliases, outside the signatures)."""
    import dataclasses
    import inspect

    import pymetrix
    import pymetrix.commands.builtin  # noqa: F401 - the built-in dialogs
    import pymetrix.models
    import pymetrix.models.bayes
    from pymetrix._api import API_CALLS
    from pymetrix.commands import COMMANDS
    from pymetrix.plugins import load_all

    load_all()  # installed plugins' calls and dialogs too
    found = [f"call {cid}" for cid, spec in API_CALLS.items() if _named_target(spec.function)]
    found += [f"dialog {name}.target" for name, info in COMMANDS.items()
              if any(f.name == "target" for f in dataclasses.fields(info.cls))]
    for module in (pymetrix, pymetrix.models, pymetrix.models.bayes):
        for name in getattr(module, "__all__", None) or [n for n in vars(module) if not n.startswith("_")]:
            obj = getattr(module, name, None)
            if callable(obj) and _named_target(obj):
                found.append(f"{module.__name__}.{name}")
            if inspect.isclass(obj) and obj.__module__.startswith("pymetrix"):
                found += [f"{module.__name__}.{name}.{m}" for m, f in vars(obj).items()
                          if not m.startswith("_") and callable(f) and _named_target(f)]
    assert len(API_CALLS) > 200 and len(COMMANDS) > 150
    assert not found, f"arguments named target (rename them, keeping the old name with renamed_arguments): {found}"
