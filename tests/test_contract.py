"""The results contract: every result estimated from data says where its numbers come from (provenance with a
fingerprint of the data used), every error pymetrix raises belongs to one hierarchy, and data can be checked first."""

import inspect
import pkgutil
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import (ARDL, PCA, VAR, VECM, Dataset, DataError, Equation, LocalProjection, NotFoundError, PyMetrixError,
                      SpecificationError, TemporalDisaggregation, templates)
from pymetrix.estimation.priors import Gamma

SRC = Path(pymetrix.__file__).parent


def _data(ts):
    return ts.with_series(ly="log(yd)", lc="log(c)")


def _results(ts):
    """One result of each kind estimated from data."""
    data = _data(ts)
    makers = {
        "equation": lambda: Equation("y ~ x + z").fit(data),
        "var": lambda: VAR(["x", "z"], lags=2).fit(data),
        "vecm": lambda: VECM(["lc", "ly"], lags=1).fit(data),
        "ardl": lambda: ARDL("lc ~ ly", lags=1, order=1).fit(data),
        "pca": lambda: PCA(["x", "z", "y"]).fit(data),
        "local projection": lambda: LocalProjection("y", "x", horizons=3, lags=1).fit(data),
        "test on data": lambda: data.test("adf", "y"),
        "test on a fit": lambda: Equation("y ~ x + z").fit(data).test("breusch_godfrey"),
    }
    out = {}
    for kind, make in makers.items():
        try:
            out[kind] = make()
        except ImportError as exc:  # the core-only install has no statsmodels
            if "statsmodels" not in str(exc):
                raise
    from pymetrix import Bayes

    out["bvar"] = VAR(["x", "z"], lags=1).fit(data, method=Bayes())
    from pymetrix import backtest

    out["backtest"] = backtest(Equation("y ~ x").fit(data), data, ("2015Q1", "2018Q4"), horizon=2)
    annual = data.to_freq("Y", how="sum").keep("c")
    out["disaggregation"] = TemporalDisaggregation("c ~ yd").fit(annual, data)
    from pymetrix import reconcile

    out["reconciliation"] = reconcile(data.with_series(cyd="c + yd"), ["c", "yd"], "cyd", annual)
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(120, nk.shock_sd, seed=1)
    truth.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    fit = nk.estimate(nk.measurement.apply(Dataset.from_frame(truth)), estimate=["kappa"], priors={"kappa": Gamma(0.3, 0.1)})
    out["model estimate"] = fit
    out["posterior"] = fit.mcmc(draws=20, chains=1, seed=3)
    out["dsge-var"] = nk.dsge_var(fit.data, lags=1, estimate=["kappa"], priors={"kappa": Gamma(0.3, 0.1)})
    from pymetrix import StateSpace

    out["state space"] = StateSpace(["s"], ["s = rho*s[-1] + e"], ["y = s"], {"e": 1.0}, ["rho"]).fit(data)
    return out


def test_every_result_from_data_has_provenance_with_a_fingerprint(ts):
    for kind, result in _results(ts).items():
        prov = result.provenance
        assert prov is not None, kind
        assert prov.data_fingerprint, kind
        assert prov.package_version == pymetrix.__version__, kind
        assert prov.sample is not None or kind == "test on data", kind
    posterior = _results(ts)["posterior"].provenance
    assert posterior.method["sampler"] == "random-walk Metropolis-Hastings" and posterior.seed == 3
    if "test on data" in _results(ts):
        assert _results(ts)["test on data"].provenance.method["test"]


def test_the_fingerprint_follows_the_data(ts):
    data = _data(ts)
    a = VAR(["x", "z"], lags=2).fit(data).provenance.data_fingerprint
    b = VAR(["x", "z"], lags=2).fit(data).provenance.data_fingerprint
    changed = data.with_series(x=data["x"] * 1.01)
    c = VAR(["x", "z"], lags=2).fit(changed).provenance.data_fingerprint
    other_series = data.with_series(y=data["y"] * 2)
    d = VAR(["x", "z"], lags=2).fit(other_series).provenance.data_fingerprint
    assert a == b and a != c and a == d  # the data used, not the rest of the dataset


def test_every_pymetrix_error_is_a_pymetrix_error():
    classes = []
    for info in pkgutil.walk_packages(pymetrix.__path__, "pymetrix."):
        if ".ui" in info.name or ".ext" in info.name:
            continue
        try:
            module = __import__(info.name, fromlist=["_"])
        except ImportError:
            continue
        classes += [c for _, c in inspect.getmembers(module, inspect.isclass)
                    if issubclass(c, Exception) and c.__module__.startswith("pymetrix") and not issubclass(c, Warning)]
    assert classes and all(issubclass(c, PyMetrixError) for c in set(classes)), [c for c in set(classes) if not issubclass(c, PyMetrixError)]
    # and nothing raises the built-in errors directly any more
    plain = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py") if "__pycache__" not in p.parts
             for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if re.search(r"\braise (ValueError|KeyError)\(", line)]
    assert plain == [], plain


def test_errors_are_still_the_built_in_kinds(ts):
    with pytest.raises(NotFoundError) as missing:
        ts["nope"]
    assert isinstance(missing.value, KeyError) and str(missing.value) == "no series 'nope' in the dataset"
    with pytest.raises(ValueError):  # code that caught ValueError still does
        Equation("y ~ nope").fit(ts)
    with pytest.raises(PyMetrixError):
        templates.new_keynesian(phi_pi=0.5).solve()
    assert issubclass(SpecificationError, ValueError) and issubclass(DataError, ValueError)
    with pytest.raises(pymetrix.PlannedError):
        from pymetrix._planned import planned

        planned("something", 9)


def test_validation_finds_problems_and_passes_clean_data(ts, panel):
    assert ts.validate().ok and not ts.validate().warnings
    idx = pd.period_range("2000Q1", periods=24, freq="Q", name="time")
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"y": rng.normal(size=24), "c": 1.0, "z": rng.normal(size=24), "label": "x"}, index=idx)
    f.iloc[5, 0] = np.nan
    f.iloc[10, 2] = np.inf
    f.iloc[20:, 0] = np.nan
    f.iloc[3, 2] = 40.0
    f.iloc[7, 3] = "y"
    report = Dataset.from_frame(f.drop(idx[15])).validate()
    found = {(x.level, x.series, x.check) for x in report.findings}
    assert {("error", "", "missing periods"), ("error", "z", "infinite values"), ("warning", "c", "constant"),
            ("warning", "y", "missing values"), ("warning", "z", "outliers"), ("note", "", "ragged edge")} <= found
    assert ("warning", "z", "missing values") not in found  # an infinite value is an error, not a gap
    assert not report.ok and "2 errors" in report.summary().notes[0]
    with pytest.raises(DataError, match="the data have errors"):
        Dataset.from_frame(f.drop(idx[15])).validate(strict=True)
    gappy = panel.frame.drop(panel.frame.index[[3, 50]])
    p_report = Dataset(gappy, panel.axes, panel.variables).validate()
    assert sum(x.check == "missing periods" for x in p_report.findings) == 2
    text = Dataset.from_frame(pd.DataFrame({"v": ["1", "2", "n/a", "4"]}, index=idx[:4]))
    assert any(x.check == "not numbers" for x in text.validate().findings)


def test_the_check_data_dialog(ts):
    from pymetrix.commands import CheckData, Session
    from pymetrix.views import view_sections

    s = Session()
    s.objects["data"] = ts
    s.run(CheckData())
    assert "checks = data.validate()" in s.script() and [t for t, _ in view_sections(s["checks"])] == ["Checks"]


def test_iterative_estimators_record_convergence(ts):
    from pymetrix import ARMA, ML, Logit

    data = ts.with_series(b="step(y - 1)")
    assert Equation("b ~ x", family=Logit()).fit(data, method=ML()).provenance.converged is True
    assert Equation("y ~ x", errors=ARMA(1, 0)).fit(data).provenance.converged is True
    assert Equation("y ~ x + z").fit(data).provenance.converged is None  # closed form: nothing to converge
