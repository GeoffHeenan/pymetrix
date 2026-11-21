"""The toolkit for custom estimators and solvers, the worked examples, NumPy/pandas in the
session namespace, and matrix views."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, Cluster, Dataset, Equation, Model, toolkit
from pymetrix.plugins import registry
from pymetrix.testing import check_entry

EXAMPLES = Path(__file__).parents[1] / "examples"
DATA = EXAMPLES / "sample_macro.parquet"


@pytest.fixture
def data():
    return Dataset.read(str(DATA))


@pytest.fixture
def examples(monkeypatch):
    """The worked examples, imported as a user's modules would be; their registrations are undone afterwards."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    saved = {g: dict(r._entries) for g, r in __import__("pymetrix.plugins", fromlist=["_REGISTRIES"])._REGISTRIES.items()}
    modules = {}
    for name in ("custom_estimator", "custom_solver"):
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    yield modules
    for g, entries in saved.items():
        __import__("pymetrix.plugins", fromlist=["_REGISTRIES"])._REGISTRIES[g]._entries = entries
    for name in modules:
        sys.modules.pop(name, None)


def test_design_and_least_squares_reproduce_ols(data):
    eq = Equation("c ~ y + c[-1]")
    d = toolkit.design(eq, data, span=("2001Q1", None))
    fit = eq.fit(data, span=("2001Q1", None), method=OLS(cov="hac"))
    beta, u = toolkit.least_squares(d.y, d.X)
    assert d.names == list(fit.params.names) and d.df_resid == d.n - 3
    assert np.allclose(beta, fit.params.values)
    assert np.allclose(toolkit.covariance("hac", d.X, u, d.df_resid), fit.params.cov)
    ours = toolkit.result(eq, data, d, "my method", beta, toolkit.covariance("hac", d.X, u, d.df_resid), u)
    assert np.isclose(ours.stats["r2"], fit.stats["r2"]) and ours.provenance.sample == fit.provenance.sample
    assert list(d.frame().columns) == ["c", "const", "y", "c[-1]"]


def test_design_handles_instruments_clusters_and_fixed_effects():
    rng = np.random.default_rng(1)
    rows = [{"id": k, "t": p, "x": rng.normal(), "z": rng.normal()} for k in range(20) for p in pd.period_range("2000", periods=5, freq="Y")]
    df = pd.DataFrame(rows)
    df["y"] = 2 * df["x"] + df["id"] * 0.5 + rng.normal(size=len(df))
    panel = Dataset.from_frame(df, time="t", entity="id")
    d = toolkit.design("y ~ x + fe(id)", panel, instruments=["z"])
    assert d.Z.shape == (100, 1) and d.absorbed == 20 and d.names == ["x"]
    fit = Equation("y ~ x + fe(id)").fit(panel, method=OLS(cov=Cluster("id")))
    beta, u = toolkit.least_squares(d.y, d.X)
    assert np.allclose(toolkit.covariance(Cluster("id"), d.X, u, d.df_resid, groups=d.groups(Cluster("id"))), fit.params.cov)


def test_arrays_derivatives_and_sandwich(data):
    index, M = toolkit.arrays(data, ["c", "dlog(y)"], span=("2010Q1", "2012Q4"))
    assert M.shape == (12, 2) and str(index[0]) == "2010Q1"
    _, kept = toolkit.arrays(data, ["dlog(y)"], missing="keep")
    assert np.isnan(kept[0, 0])
    f = lambda x: (x[0] - 1) ** 2 + 3 * x[0] * x[1] + x[1] ** 2  # noqa: E731
    assert np.allclose(toolkit.gradient(f, np.array([1.0, 2.0])), [6.0, 7.0], atol=1e-6)
    assert np.allclose(toolkit.hessian(f, np.array([1.0, 2.0])), [[2, 3], [3, 2]], atol=1e-4)
    H, scores = np.array([[2.0, 0.5], [0.5, 1.0]]), np.random.default_rng(0).normal(size=(50, 2))
    assert np.allclose(toolkit.sandwich(H, scores, "classic"), np.linalg.inv(H))
    assert np.allclose(toolkit.sandwich(H, scores), np.linalg.inv(H) @ scores.T @ scores @ np.linalg.inv(H))
    assert toolkit.to_series([1, 2], index[:2], "x").name == "x"


def test_custom_estimator_works_like_a_builtin(examples, data):
    Ridge = examples["custom_estimator"].MyRidge
    eq = Equation("c ~ y + y[-1] + i")
    assert np.allclose(eq.fit(data, method=Ridge(alpha=0)).params.values, eq.fit(data).params.values)
    shrunk = eq.fit(data, method=Ridge(alpha=50))
    assert shrunk.method == Ridge(alpha=50) and shrunk.extra["alpha"] == 50
    entry = registry("methods").get("My ridge")
    check_entry(entry)  # the conformance kit: form, help, round trip
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y + i", method="My ridge", method_settings="alpha=5.0", name="ridge"))
    script = s.script()
    assert "from custom_estimator import MyRidge" in script and "method=MyRidge(alpha=5.0)" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["ridge"].params.values, s["ridge"].params.values)
    fits = data.series("c", "i").fit("{s} ~ y", data, method=Ridge(alpha=1))  # loops over selections too
    assert list(fits) == ["c", "i"]


def test_custom_solver_works_like_a_builtin(examples, data):
    DampedNewton = examples["custom_solver"].DampedNewton
    cons = Equation("c ~ y + c[-1]").fit(data)
    model = Model([cons, Equation("y = c + i + g")])
    ours = model.solve(data, span=("2020Q1", "2024Q4"), solver=DampedNewton())
    assert np.allclose(ours.solved()["y"], model.solve(data, span=("2020Q1", "2024Q4")).solved()["y"])
    check_entry(registry("solvers").get("Damped Newton"))
    from pymetrix.commands import FitEquation, Session, SolveModel

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    s.run(SolveModel("cons", "2020Q1", "2020Q4", solver="Damped Newton"))
    assert "solver=DampedNewton()" in s.script() and "from custom_solver import DampedNewton" in s.script()


def test_numpy_pandas_and_scipy_are_ready_in_the_session(data, tmp_path):
    from pymetrix.commands import Session, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run_code("M = np.cov(data.frame[['c', 'y']].to_numpy().T)\nevals = np.linalg.eigvalsh(M)\nz = sp.stats.norm.ppf(0.975)\ntab = pd.DataFrame(M)")
    assert isinstance(s["M"], np.ndarray) and isinstance(s["tab"], pd.DataFrame) and np.isclose(s["z"], 1.959964)
    assert "np" not in s.objects and "pd" not in s.objects  # modules are not workspace objects
    script = s.script()
    assert script.startswith("import numpy as np\nimport pandas as pd\nimport scipy as sp\nfrom pymetrix import *")
    reopened = load_project(save_project(s, tmp_path / "matrices.pymx"))
    assert np.allclose(reopened["M"], s["M"])
    plain = Session()
    plain.objects["data"] = data
    plain.run_code("x = data['c'].mean()")
    assert plain.script().startswith("from pymetrix import *")  # imports only when used


def test_matrices_open_as_tables():
    from pymetrix.ui.render import sections

    title, table = sections(np.arange(6.0).reshape(2, 3))[0]
    assert table.title == "Matrix: 2 × 3" and len(table.rows) == 2
    assert sections(np.arange(3.0))[0][1].title == "Vector: 3"
    assert sections(pd.DataFrame({"a": [1, 2]}))[0][1].title == "Table: 2 × 1"
    assert sections(pd.Series([1.0, 2.0], name="s"))[0][1].title == "Series: 2 values"
    assert sections(np.float64(2.5))[0][0] == "Value"
