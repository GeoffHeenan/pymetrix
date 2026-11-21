"""Extension API v1: every built-in component and reference plugin passes the conformance kit,
and a third-party plugin loads through entry points exactly as an installed package would."""

import sys
import types
from importlib.metadata import EntryPoint

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import Dataset, Equation, registry
from pymetrix import plugins as plugin_api
from pymetrix.commands import COMMANDS, FitEquation, Filter, Session
from pymetrix.help import help_page, search
from pymetrix.plugins import PluginInfo, plugin_table
from pymetrix.testing import (
    DECLARATIVE_GROUPS,
    check_determinism,
    check_entry,
    check_form,
    check_plugin,
    check_project,
    check_result,
    check_roundtrip,
    crosscheck,
)

ENTRIES = [e for g in DECLARATIVE_GROUPS for e in registry(g).entries()]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e.group}/{e.qualified}")
def test_builtin_extensions_conform(entry):
    check_entry(entry)


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_commands_conform(name):
    check_form(COMMANDS[name].cls)


@pytest.mark.parametrize("module", ["linearmodels", "arch"])
def test_reference_plugins_meet_the_contrib_standard(module):
    plugin = __import__(f"pymetrix.ext.{module}", fromlist=["PLUGIN"]).PLUGIN
    entries = check_plugin(plugin, contrib=True)
    assert all(e.qualified.startswith(f"{module}:") for e in entries)
    assert plugin.license == "BSD-3-Clause"


@pytest.fixture(scope="module")
def ivdata():
    rng = np.random.default_rng(3)
    n = 800
    z1, z2, w = rng.normal(size=(3, n))
    u = rng.normal(size=n)
    x = 0.6 * z1 + 0.4 * z2 + 0.3 * w + 0.8 * u + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"id": range(n), "y": 1 + 2 * x + 0.5 * w + u, "x": x, "w": w, "z1": z1, "z2": z2})
    return df, Dataset.from_frame(df, entity="id")


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(1)
    r = rng.standard_t(6, size=600) * 0.8
    y = np.cumsum(rng.normal(0.5, 1, size=120))
    return (Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2000-01-01", periods=len(r), freq="D", name="time"))),
            Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1990Q1", periods=len(y), freq="Q", name="time"))))


def test_plugin_results_record_backend_and_plugin(ivdata, series):
    lm_iv = pytest.importorskip("linearmodels.iv")
    df, data = ivdata
    liml = pymetrix.LIML(endog=["x"], instruments=["z1", "z2"])
    fit = Equation("y ~ x + w").fit(data, method=liml)
    check_result(fit, backend="linearmodels", plugin="pymetrix-linearmodels")
    ref = lm_iv.IVLIML(df["y"], df[["w"]].assign(const=1.0), df["x"], df[["z1", "z2"]]).fit(cov_type="unadjusted")
    crosscheck(fit.params["x"], ref.params["x"], what="LIML slope")
    pytest.importorskip("arch")
    returns, _ = series
    eg = Equation("r ~ 1", variance=pymetrix.EGARCH()).fit(returns, method=pymetrix.ML())
    check_result(eg, backend="arch", plugin="pymetrix-arch")
    check_result(Equation("y ~ y[-1]").fit(series[1]))  # core: no backend required


def test_project_round_trip_and_determinism(ivdata, series, tmp_path):
    pytest.importorskip("linearmodels")
    _, data = ivdata
    session = Session()
    session.objects["data"] = data
    check_project(session, FitEquation("y ~ x + w", method="LIML", endog="x", instruments="z1, z2",
                                       method_settings="fuller=1.0", name="liml"), tmp_path)
    q = Session()
    q.objects["data"] = series[1]
    check_project(q, Filter("y", method="Hamilton", settings="h=8"), tmp_path)
    check_determinism(lambda: series[1].filter("y", pymetrix.HP(lamb=1600)))


def test_roundtrip_of_configured_objects():
    for obj in (pymetrix.OLS(cov=pymetrix.Cluster("id")), pymetrix.GARCH(asym=1, dist="t"), pymetrix.HP(one_sided=True),
                pymetrix.VECM(["a", "b"], rank=1), pymetrix.Newton(tol=1e-8)):
        check_roundtrip(obj)


def test_settings_typed_as_text_keep_their_types():
    entry = registry("variances").get("APARCH")
    from pymetrix.forms import build, option_string, settings_form

    specs = settings_form(entry)
    text = option_string(specs, {"delta": "1.5", "arch": 2})
    assert text == "arch=2, delta=1.5"
    assert build(entry, text).delta == 1.5
    gjr = registry("variances").get("GJR-GARCH")
    assert build(gjr, "").asym == 1 and {s.name: s.default for s in settings_form(gjr)}["asym"] == 1


# third-party plugins through entry points


@pytest.fixture
def isolated(monkeypatch):
    """Registries restored after the test, and entry points under the test's control."""
    saved = ({g: dict(r._entries) for g, r in plugin_api._REGISTRIES.items()}, dict(plugin_api._PLUGINS),
             dict(plugin_api._PROBLEMS), set(plugin_api._loaded))
    found: dict[str, list] = {}
    monkeypatch.setattr(plugin_api, "_entry_points", lambda group: found.get(group, []))

    def provide(group: str, name: str, value: str) -> None:
        found.setdefault(f"pymetrix.{group}", []).append(EntryPoint(name, value, f"pymetrix.{group}"))
        plugin_api._loaded.discard(group)

    yield provide
    for g, entries in saved[0].items():
        plugin_api._REGISTRIES[g]._entries = entries
    plugin_api._PLUGINS.clear(), plugin_api._PLUGINS.update(saved[1])
    plugin_api._PROBLEMS.clear(), plugin_api._PROBLEMS.update(saved[2])
    plugin_api._loaded.clear(), plugin_api._loaded.update(saved[3])


def test_third_party_plugin_loads_and_conforms(isolated, series):
    isolated("filters", "demean", "sample_plugin:Demean")
    isolated("tests", "mean_zero", "sample_plugin:MeanZero")
    import sample_plugin

    entry = registry("filters").get("Demean")
    assert entry.qualified == "sample:demean" and entry.plugin is sample_plugin.PLUGIN
    check_plugin(sample_plugin.PLUGIN, contrib=True)

    _, data = series
    session = Session()
    session.objects["data"] = data
    session.run(Filter("y", method="Demean", settings="scale=0.5", name="d"))
    session.run(__import__("pymetrix.commands", fromlist=["RunTest"]).RunTest(source="data", test="mean_zero", options="variable='y'"))
    script = session.script()
    assert "from sample_plugin import Demean" in script and "d = data.filter('y', Demean(scale=0.5))" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["d"]["y_trend"], 0.5 * data["y"].mean())
    table = plugin_table().to_frame()
    assert table.set_index("plugin").loc["pymetrix-sample", "status"] == "available"


def test_unsupported_api_and_missing_libraries_are_reported(isolated):
    future = types.ModuleType("future_plugin")
    future.PLUGIN = PluginInfo("pymetrix-future", "9.0", api="2", license="MIT")
    missing = types.ModuleType("missing_plugin")
    missing.PLUGIN = PluginInfo("pymetrix-missing", "0.1", requires={"no_such_library_xyz": ">=1"}, license="MIT",
                                install="pip install pymetrix-missing")
    exec("from dataclasses import dataclass\n"
         "@dataclass(frozen=True)\nclass Smooth:\n    '''Needs a library that is not installed.'''\n"
         "    label = 'Smooth'\n    def describe(self, f): return 'Smooth'\n    def trend(self, y, f): return y\n", missing.__dict__)
    future.Smooth = missing.Smooth
    sys.modules.update(future_plugin=future, missing_plugin=missing)
    try:
        isolated("filters", "future", "future_plugin:Smooth")
        isolated("filters", "smooth", "missing_plugin:Smooth")
        reg = registry("filters")
        assert "Smooth" not in reg.labels() and "Smooth" in reg.labels(available_only=False)
        assert reg.get("Smooth").install_hint == "pip install pymetrix-missing"
        status = plugin_table().to_frame().set_index("plugin")["status"]
        assert "extension API 2" in status["pymetrix-future"]
        assert status["pymetrix-missing"] == "needs no_such_library_xyz: pip install pymetrix-missing"
    finally:
        del sys.modules["future_plugin"], sys.modules["missing_plugin"]


def test_same_label_from_two_plugins_is_qualified(isolated):
    other = PluginInfo("pymetrix-other", "1.0", license="MIT")
    plugin_api.register("filters", pymetrix.HP, name="hp", plugin=other)
    labels = registry("filters").labels()
    assert "HP (pymetrix:hp)" in labels and "HP (other:hp)" in labels
    with pytest.raises(KeyError, match="ambiguous"):
        registry("filters").get("HP")
    assert registry("filters").get("other:hp").plugin is other


def test_help_pages_and_search():
    page = help_page(registry("methods").get("LIML"))
    assert page.wraps == "linearmodels.iv.IVLIML" and page.topics == ("instrumental-variables",)
    assert "Fuller" in page.to_markdown() and "pymetrix-linearmodels" in page.to_markdown()
    if registry("methods").get("LIML").available:
        assert "LIML" in page.reference() or "liml" in page.reference().lower()
    hits = search("instrument")
    assert hits[0][0] in ("topic", "methods") and any(t == "Instrumental variables" for _, t, _ in hits)
    assert help_page(pymetrix.HP).title == "HP" and help_page(pymetrix.HP(one_sided=True)).summary.startswith("Hodrick-Prescott")
