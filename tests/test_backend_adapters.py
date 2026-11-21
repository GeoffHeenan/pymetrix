"""Backend adapters (plan item 5): an explicit mapping spec per wrapped estimator, backend_options passed through
and recorded, unsupported settings refused, library warnings and messages kept with the result."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import Cluster, Dataset, Equation
from pymetrix.adapters import arch as arch_adapter
from pymetrix.adapters import linearmodels as lm_adapter
from pymetrix.adapters import sklearn as sk_adapter
from pymetrix.adapters import statsmodels as sm_adapter
from pymetrix.adapters.base import Options
from pymetrix.codec import decode, encode
from pymetrix.errors import PyMetrixError, SpecificationError
from pymetrix.plugins import GROUPS, registry


def _wrapped_classes():
    found = {}
    for group in GROUPS:
        for entry in registry(group).entries():
            cls = entry.obj
            if isinstance(cls, type) and "adapter_spec" in dir(cls) and dataclasses.is_dataclass(cls):
                found[cls.__name__] = cls
    return [found[k] for k in sorted(found)]


WRAPPED = _wrapped_classes()


def test_the_wrapped_estimators_are_found():
    names = {c.__name__ for c in WRAPPED}
    assert {"LIML", "IVGMM", "RandomEffects", "Between", "FirstDifference", "FamaMacBeth", "EGARCH", "TARCH", "FIGARCH", "APARCH",
            "Quantile", "ExponentialSmoothing", "VECM", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting",
            "NeuralNetwork", "NearestNeighbours"} <= names
    assert all(hasattr(c, "adapter_spec") for c in UNREGISTERED)


UNREGISTERED = [pymetrix.MarkovSwitching, pymetrix.ARDL]  # model specifications used from code, not listed in a registry


@pytest.mark.parametrize("cls", [*WRAPPED, *UNREGISTERED], ids=lambda c: c.__name__)
def test_every_setting_is_mapped_used_by_pymetrix_or_refused(cls):
    spec = cls.adapter_spec
    settings = {f.name for f in dataclasses.fields(cls) if f.init}
    assert "backend_options" in settings
    accounted = spec.fields() | {"backend_options"}
    unmapped = settings - accounted
    # settings shared with the estimation method (ML's cov and max_iter) are described in the spec's notes
    assert not unmapped or all(f"`{n}`" in spec.notes for n in unmapped), f"{cls.__name__}: {sorted(unmapped)} not in its adapter spec"
    assert spec.fields() <= settings, f"{cls.__name__}: the spec names settings it does not have: {sorted(spec.fields() - settings)}"
    text = spec.describe()
    assert spec.wraps in text and "backend_options" in text


def test_options_are_read_only_hashable_and_typed_as_text():
    o = Options("maxiter=500, tol=1e-8")
    assert dict(o) == {"maxiter": 500, "tol": 1e-8} and o == {"maxiter": 500, "tol": 1e-8}
    assert hash(o) == hash(Options({"tol": 1e-8, "maxiter": 500}))
    assert Options("{'fit.maxiter': 5}") == {"fit.maxiter": 5}
    with pytest.raises(SpecificationError):
        Options("500")
    with pytest.raises(SpecificationError):
        Options({"not a name": 1})
    with pytest.raises(TypeError):
        o["maxiter"] = 1  # type: ignore[index]


def test_settings_with_backend_options_round_trip():
    q = pymetrix.Quantile(q=0.25, backend_options={"max_iter": 100})
    value = json.loads(json.dumps(encode(q)))
    assert value["backend_options"] == {"max_iter": 100}
    assert decode(value) == q and isinstance(decode(value).backend_options, Options)


@pytest.fixture(scope="module")
def ts():
    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.standard_t(5, size=n)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "z": rng.normal(size=n)}, index=pd.period_range("1980Q1", periods=n, freq="Q")))


def test_backend_options_reach_the_library_and_are_recorded(ts):
    fit = Equation("y ~ x").fit(ts, method=pymetrix.Quantile(q=0.5, backend_options={"max_iter": 77, "p_tol": 1e-7}))
    prov = fit.provenance
    assert prov.backend["name"] == "statsmodels" and prov.backend["version"] == __import__("statsmodels").__version__
    assert prov.options["stages"]["fit"]["max_iter"] == 77  # overrides the spec's fixed 5000
    assert prov.options["stages"]["fit"]["p_tol"] == 1e-7
    assert prov.options["backend_options"] == {"max_iter": 77, "p_tol": 1e-7}
    assert prov.method["backend"] == f"statsmodels {prov.backend['version']}"
    default = Equation("y ~ x").fit(ts, method=pymetrix.Quantile())
    assert default.provenance.options["stages"]["fit"]["max_iter"] == 5000  # the fixed argument, recorded


def test_a_backend_option_that_pymetrix_sets_is_refused_naming_the_setting(ts):
    with pytest.raises(SpecificationError, match=r"q \(set it with q=...\)"):
        Equation("y ~ x").fit(ts, method=pymetrix.Quantile(backend_options={"q": 0.9}))
    with pytest.raises(SpecificationError, match="go to model or fit"):
        Equation("y ~ x").fit(ts, method=pymetrix.Quantile(backend_options={"solve.maxiter": 3}))


def test_model_stage_options(ts):
    ets = pymetrix.ExponentialSmoothing("y", backend_options={"model.initialization_method": "heuristic"})
    fit = ets.fit(ts)
    assert fit.provenance.options["stages"]["model"]["initialization_method"] == "heuristic"
    assert fit.provenance.options["stages"]["fit"]["disp"] is False


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    rows = []
    for e in range(30):
        a = rng.normal()
        for t in range(12):
            x = rng.normal()
            rows.append({"id": e, "year": 2000 + t, "x": x, "y": a + 0.7 * x + rng.normal(scale=0.5)})
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="year", freq="Y")


def test_unsupported_settings_are_refused_not_ignored(panel):
    pytest.importorskip("linearmodels")
    with pytest.raises(SpecificationError, match="clustered covariance is not available"):
        Equation("y ~ x").fit(panel, method=pymetrix.FamaMacBeth(cov=Cluster("id")))
    fit = Equation("y ~ x").fit(panel, method=pymetrix.FamaMacBeth(cov="white"))
    assert fit.provenance.backend["name"] == "linearmodels"


def test_linearmodels_records_both_stages(panel):
    pytest.importorskip("linearmodels")
    fit = Equation("y ~ x").fit(panel, method=pymetrix.RandomEffects(backend_options={"fit.extra_df": 0}))
    stages = fit.provenance.options["stages"]
    assert stages["model"]["weights"] is None and stages["model"].keys() >= {"weights"}
    assert stages["fit"]["extra_df"] == 0 and stages["fit"]["cov_type"] == "unadjusted"
    assert fit.provenance.method["plugin"].startswith("pymetrix-linearmodels")


def test_arch_warnings_are_kept_not_silenced(ts):
    pytest.importorskip("arch")
    eq = Equation("y ~ 1", variance=pymetrix.EGARCH(backend_options={"tol": 1e-6}))
    with pytest.warns(Warning, match="scale"):
        fit = eq.fit(ts.with_series(y="10000 * y"))  # badly scaled: arch warns, and used to be silenced
    assert any("scale" in w for w in fit.provenance.warnings), "arch's warnings belong to the result"
    assert fit.provenance.options["stages"]["fit"]["tol"] == 1e-6


def test_a_backend_option_that_pymetrix_sets_in_arch_is_refused(ts):
    pytest.importorskip("arch")
    eq = Equation("y ~ 1", variance=pymetrix.EGARCH(backend_options={"options": {"maxiter": 3}}))
    with pytest.raises(SpecificationError, match="options are set by pymetrix"):
        eq.fit(ts)


def test_messages_and_warnings_inside_a_run_are_recorded():
    import warnings

    run = sm_adapter.ADAPTER.run(sm_adapter.QUANTREG)
    with run:
        run.call("fit", lambda **kw: warnings.warn("did not converge", RuntimeWarning), q=0.5)
        run.message("optimiser: maximum iterations reached")
    record = run.record
    assert record.warnings == ("RuntimeWarning: did not converge",)
    assert record.messages == ("optimiser: maximum iterations reached",)
    assert record.options["fit"] == {"max_iter": 5000, "q": 0.5}


def test_sklearn_options_reach_every_cross_validation_fit(ts):
    pytest.importorskip("sklearn")
    fit = Equation("y ~ x + z").fit(ts, method=pymetrix.RandomForest(trees=20, backend_options={"max_depth": 3}))
    assert fit.provenance.options["stages"]["model"]["max_depth"] == 3
    assert all(t.max_depth == 3 for t in fit.model.estimators_)
    lasso = Equation("y ~ x + z").fit(ts, method=pymetrix.Lasso(alpha=0.1))
    assert lasso.provenance.options["stages"]["model"]["max_iter"] == 100_000  # fixed by the spec, recorded


def test_private_library_attributes_are_read_in_one_place():
    class Bare:
        model = object()

    with pytest.raises(PyMetrixError, match="DynamicFactorMQ"):
        sm_adapter.dynamic_factor_index(Bare())
    with pytest.raises(PyMetrixError, match="orig_endog"):
        sm_adapter.original_endog(object())
    with pytest.raises(PyMetrixError, match="Zivot-Andrews"):
        arch_adapter.zivot_andrews_all_stats(object())


@pytest.mark.parametrize("module", [lm_adapter, arch_adapter, sm_adapter, sk_adapter], ids=lambda m: m.__name__.rpartition(".")[2])
def test_every_spec_describes_itself(module):
    for name, spec in module.SPECS.items():
        assert spec.backend == module.ADAPTER.name, name
        assert spec.describe().startswith(f"**{spec.wraps}**")


def test_stage_options_typed_as_text():
    assert Options("fit.iter_limit=5, model.check_rank=False, tol=1e-8") == {"fit.iter_limit": 5, "model.check_rank": False, "tol": 1e-8}


def test_unit_root_tests_go_through_the_statsmodels_adapter():
    from pymetrix.stats import run_test

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=80).cumsum(), "x": rng.normal(size=80).cumsum()},
                         index=pd.period_range("2000Q1", periods=80, freq="Q"))
    data = Dataset.from_frame(frame)
    adf = run_test(data, "adf", variable="y", lags=2)
    assert adf.provenance.backend["wraps"] == "statsmodels.tsa.stattools.adfuller" and adf.backend_call is None
    assert adf.provenance.options["stages"]["test"] == {"maxlag": 2, "regression": "c", "autolag": None}
    kpss = run_test(data, "kpss", variable="y")
    assert any("InterpolationWarning" in w for w in kpss.provenance.warnings) and "note" in kpss.details
    eg = run_test(data, "engle_granger", variables="y,x", max_lags=3, backend_options={"method": "aeg"})
    assert eg.provenance.options["stages"]["test"]["maxlag"] == 3 and eg.provenance.options["backend_options"] == {"method": "aeg"}
    with pytest.raises(SpecificationError, match="set it with trend"):
        run_test(data, "adf", variable="y", backend_options={"regression": "n"})
    with pytest.raises(SpecificationError, match="set it with max_lags"):
        run_test(data, "engle_granger", variables="y,x", backend_options={"maxlag": 3})


def test_arch_unit_root_tests_go_through_the_arch_adapter():
    pytest.importorskip("arch")
    from pymetrix.stats import run_test

    rng = np.random.default_rng(0)
    walk = rng.normal(size=120).cumsum()
    data = Dataset.from_frame(pd.DataFrame({"y": walk, "z": walk + rng.normal(size=120)},
                                           index=pd.period_range("1990Q1", periods=120, freq="Q")))
    for name, settings in [("phillips_perron", {"variable": "y"}), ("zivot_andrews", {"variable": "y"}), ("dfgls", {"variable": "y"}),
                           ("variance_ratio", {"variable": "y"}), ("phillips_ouliaris", {"variables": "y,z"})]:
        prov = run_test(data, name, **settings).provenance
        assert prov.backend["name"] == "arch" and prov.backend["wraps"].startswith("arch.unitroot"), name
        assert prov.options["stages"]["test"]["trend"] == "c", name
    with pytest.raises(SpecificationError, match="set it with lags"):
        run_test(data, "dfgls", variable="y", backend_options={"lags": 2})
