"""The choice of random generator (pymetrix.set_random_generator): PCG64, Mersenne Twister, Philox or SFC64.

Checked: every name gives exactly NumPy's own Generator over that bit generator (raw draws compared with NumPy called
directly, for integer seeds, seed lists and spawned SeedSequence streams), and the default is exactly
``np.random.default_rng``; no module makes a Generator except through ``pymetrix._random`` (a scan of the source);
with the Mersenne Twister chosen, seeded features (stochastic simulation, a VAR bootstrap, a short Metropolis-Hastings
chain, SMC, a copula simulation) never call ``default_rng`` and do build MT19937s, give other draws than PCG64, and the
same draws again for the same seed; chains in other processes draw as in this one (the setting travels to them);
results record the generator in their provenance (also through a save and a load, and old saves read as PCG64);
a project saves the generator with each step and replays with it whatever the current setting (check_project), its
script sets it, and a generator this version lacks is warned about; unknown names are refused; the context manager
restores the previous setting. Default results bit-identical to before the setting existed were checked when it was
added, on the same features."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import VAR, Dataset, Equation, Model, templates
from pymetrix import _random
from pymetrix.errors import SpecificationError
from pymetrix.estimation.priors import Normal
from pymetrix.metadata import Provenance, Role

BITS = {"pcg64": np.random.PCG64, "mt19937": np.random.MT19937, "philox": np.random.Philox, "sfc64": np.random.SFC64}


@pytest.fixture(autouse=True)
def _default_generator():
    before = pymetrix.get_random_generator()
    pymetrix.set_random_generator("pcg64")
    yield
    pymetrix.set_random_generator(before)


def _draws(g: np.random.Generator) -> np.ndarray:
    return np.concatenate([g.random(5), g.standard_normal(5), g.integers(0, 1000, 5).astype(float)])


@pytest.mark.parametrize("name", list(BITS))
def test_each_name_is_numpys_generator_over_that_bit_generator(name):
    with pymetrix.random_generator(name):
        for seed in (0, 12345, [7, 3]):
            g = _random.generator(seed)
            assert isinstance(g.bit_generator, BITS[name])
            np.testing.assert_array_equal(_draws(g), _draws(np.random.Generator(BITS[name](seed))))
        streams = _random.seed_sequence(42).spawn(3)
        direct = np.random.SeedSequence(42).spawn(3)
        for s, d in zip(streams, direct):
            np.testing.assert_array_equal(_draws(_random.generator(s)), _draws(np.random.Generator(BITS[name](d))))
    # the default is exactly NumPy's default
    np.testing.assert_array_equal(_draws(_random.generator(9)), _draws(np.random.default_rng(9)))
    g = np.random.default_rng(1)
    assert _random.generator(g) is g


def test_names_and_the_context_manager():
    assert pymetrix.get_random_generator() == "pcg64"
    with pymetrix.random_generator("MT19937") as using:
        assert using == pymetrix.get_random_generator() == "mt19937"
        with pymetrix.random_generator("sfc64"):
            assert pymetrix.get_random_generator() == "sfc64"
        assert pymetrix.get_random_generator() == "mt19937"
    assert pymetrix.get_random_generator() == "pcg64"
    for bad in ("mt", "randomstate", "", None, 19937):
        with pytest.raises(SpecificationError, match="random generator is one of"):
            pymetrix.set_random_generator(bad)
    assert pymetrix.get_random_generator() == "pcg64"
    assert pymetrix.set_random_generator("philox") == "pcg64"  # returns the previous one


def test_no_module_makes_a_generator_except_through_the_helper():
    roots = [Path(pymetrix.__file__).parent]
    finance = Path(__file__).resolve().parents[1] / "packages" / "pymetrix-finance" / "src"
    if finance.exists():
        roots.append(finance)
    found = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "_random.py":
                continue
            example = False  # inside a command's example: ClassVar[str] = """...""" (code shown to users, run as it is)
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.match(r'\s*(example: ClassVar\[str\]|_\w*EXAMPLE\w*) = """', line):
                    example = True
                elif example and '"""' in line:
                    example = False
                if re.search(r"np\.random\.(default_rng|SeedSequence|PCG64|MT19937|Philox|SFC64)\(", line) and not example \
                        and "simulated returns from a known GARCH" not in line:  # an example in a help text
                    found.append(f"{path.name}:{i}")
    assert not found, found


def _ar1_fit():
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=120, freq="Q", name="time")))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Normal(0.5, 0.2)})


def _features():
    nk = templates.new_keynesian()
    r = np.random.default_rng(3).normal(size=(200, 2)).cumsum(0) * 0.1
    vd = Dataset.from_frame(pd.DataFrame(r, columns=["a", "b"], index=pd.period_range("1970Q1", periods=200, freq="Q", name="time")))
    var = VAR(["a", "b"], lags=2).fit(vd)
    fit = _ar1_fit()
    out = {
        "simulation": lambda: nk.solve().simulate(60, nk.shock_sd, seed=11).to_numpy(),
        "bootstrap": lambda: var.irf(periods=6, draws=30, seed=3).to_frame().to_numpy(float),
        "mcmc": lambda: fit.mcmc(draws=60, chains=2, seed=5, workers=1)["rho"],
        "smc": lambda: fit.smc(particles=100, seed=1, workers=1)["rho"],
    }
    try:
        from pymetrix_finance.copulas import FittedCopula
    except ImportError:
        return out
    fc = FittedCopula.__new__(FittedCopula)
    for k, v in (("family", "clayton"), ("params", {"theta": 2.0}), ("series", ("a", "b"))):
        object.__setattr__(fc, k, v)
    out["copula"] = lambda: fc.simulate(300, seed=7)
    return out


def test_seeded_features_draw_with_the_mersenne_twister_when_chosen(monkeypatch):
    features = _features()
    default = {k: np.asarray(f(), float) for k, f in features.items()}
    built = []

    def recording(seed):
        built.append(seed)
        return np.random.MT19937(seed)

    def forbidden(*args, **kwargs):
        raise AssertionError("np.random.default_rng was called with the Mersenne Twister chosen")

    monkeypatch.setitem(_random.GENERATORS, "mt19937", recording)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pymetrix.random_generator("mt19937"):
        for name, f in features.items():
            built.clear()
            first, again = np.asarray(f(), float), np.asarray(f(), float)
            assert built, f"{name}: no MT19937 was built"
            np.testing.assert_array_equal(first, again, err_msg=name)  # the same seed, the same draws
            assert first.shape == default[name].shape
            assert not np.allclose(np.nan_to_num(first), np.nan_to_num(default[name])), f"{name}: the same draws as PCG64"


def test_chains_in_other_processes_use_the_chosen_generator():
    fit = _ar1_fit()
    with pymetrix.random_generator("mt19937"):
        here = fit.mcmc(draws=40, burn=20, chains=2, seed=4, workers=1)
        there = fit.mcmc(draws=40, burn=20, chains=2, seed=4, workers=2)
    np.testing.assert_array_equal(here.draws, there.draws)
    assert not np.array_equal(here.draws, fit.mcmc(draws=40, burn=20, chains=2, seed=4, workers=1).draws)


def test_provenance_records_the_generator_with_the_seed():
    from pymetrix.codec import decode, encode

    fit = _ar1_fit()
    with pymetrix.random_generator("philox"):
        post = fit.smc(particles=100, seed=1, workers=1)
    assert post.provenance.seed == 1 and post.provenance.random_generator == "philox"
    assert fit.smc(particles=100, seed=1, workers=1).provenance.random_generator == "pcg64"
    plain = Provenance({}, {}, {}, None, 10, 0)  # no seed: no generator
    assert plain.random_generator is None
    seeded = Provenance({}, {}, {}, None, 10, 0, seed=3, random_generator="sfc64")
    saved = json.loads(json.dumps(encode(seeded)))
    assert saved["random_generator"] == "sfc64" and decode(saved).random_generator == "sfc64"
    del saved["random_generator"]  # saved before the field existed: drawn with PCG64, the only generator then
    with pymetrix.random_generator("mt19937"):
        assert decode(saved).random_generator == "pcg64"
        unseeded = json.loads(json.dumps(encode(plain)))
        del unseeded["random_generator"]
        assert decode(unseeded).random_generator is None
    from pymetrix.views import provenance_table

    assert "random generator" in str(provenance_table(post.provenance)) and "philox" in str(provenance_table(post.provenance))
    assert "random generator" not in str(provenance_table(plain))


def test_projects_replay_each_step_with_its_generator(tmp_path):
    from pymetrix.commands import Session, load_project, save_project
    from pymetrix.commands.builtin import FitEquation
    from pymetrix.testing import check_project

    rng = np.random.default_rng(4)
    x = rng.normal(size=120)
    data = Dataset.from_frame(pd.DataFrame({"y": 1 + 0.5 * x + rng.normal(size=120), "x": x},
                                           index=pd.period_range("2000Q1", periods=120, freq="Q", name="time")))
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("y ~ x", name="ols"))
    step = FitEquation("y ~ x", method="Bayes", method_settings="draws=300, seed=3", name="bayes")
    with pymetrix.random_generator("mt19937"):
        check_project(s, step, tmp_path)  # saved and reopened (with MT19937 still chosen): the same result
    mt = s["bayes"].params.values
    assert s["bayes"].provenance.seed == 3 and s["bayes"].provenance.random_generator == "mt19937"
    script = s.script()
    assert "set_random_generator('mt19937')" in script
    assert script.index("ols = ") < script.index("set_random_generator('mt19937')") < script.index("bayes = ")
    path = tmp_path / "conformance.pymx"
    with zipfile.ZipFile(path) as z:
        steps = json.loads(z.read("commands.json"))
    assert [e.get("random_generator") for e in steps] == [None, "mt19937"]  # the default is not written
    with pymetrix.random_generator("sfc64"):  # whatever is chosen when it is opened
        reopened = load_project(path)
        assert pymetrix.get_random_generator() == "sfc64"  # and the setting is left as it was
    np.testing.assert_array_equal(reopened["bayes"].params.values, mt)
    assert reopened["bayes"].provenance.random_generator == "mt19937"
    assert [r.random_generator for r in reopened.history] == ["pcg64", "mt19937"]
    assert reopened.script() == script
    s.run(step)  # PCG64 again: other draws
    assert not np.array_equal(s["bayes"].params.values, mt)
    # a generator this version does not have: warned about, replayed with PCG64
    steps[1]["random_generator"] = "xoshiro"
    other = tmp_path / "other.pymx"
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(other, "w") as out:
        for item in z.namelist():
            out.writestr(item, json.dumps(steps) if item == "commands.json" else z.read(item))
    with pytest.warns(UserWarning, match="random generator 'xoshiro'"):
        replayed = load_project(other)
    assert replayed.history[-1].random_generator == "pcg64"
    np.testing.assert_array_equal(replayed["bayes"].params.values, s["bayes"].params.values)
    save_project(replayed, tmp_path / "again.pymx")


def test_the_desktop_setting():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings

    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    saved = settings.value("random_generator") if settings.contains("random_generator") else None
    try:
        window = main_window(Session())
        actions = window.generator_actions
        assert list(actions) == list(BITS) and actions[pymetrix.get_random_generator()].isChecked()
        actions["mt19937"].setChecked(True)
        assert pymetrix.get_random_generator() == "mt19937" and not actions["pcg64"].isChecked()
        assert settings.value("random_generator") == "mt19937"
        window.close()
        pymetrix.set_random_generator("pcg64")
        again = main_window(Session())  # a new window takes the saved choice
        assert pymetrix.get_random_generator() == "mt19937" and again.generator_actions["mt19937"].isChecked()
        again.close()
    finally:
        if saved is None:
            settings.remove("random_generator")
        else:
            settings.setValue("random_generator", saved)
