"""Systems of equations: SUR and 3SLS, cross-checked against linearmodels; restrictions; nonlinear equations."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from pymetrix import SUR, Dataset, Equation, Model, System, SystemOLS, ThreeSLS, TwoSLS
from pymetrix.expr.nodes import ExpressionError

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(1)
    n = 200
    x1, x2, x3, z1, z2 = rng.normal(size=(5, n))
    e = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=n)
    y2 = 1 + 0.5 * x2 + 0.8 * z1 - 0.4 * z2 + e[:, 1]
    y1 = 2 + 1.5 * x1 - 0.7 * y2 + e[:, 0]
    frame = pd.DataFrame(dict(t=pd.period_range("1970Q1", periods=n, freq="Q"), y1=y1, y2=y2, x1=x1, x2=x2, x3=x3, z1=z1, z2=z2))
    return frame, Dataset.from_frame(frame, time="t")


def _X(frame, *cols):
    return pd.concat([pd.Series(1.0, index=frame.index, name="const"), frame[list(cols)]], axis=1)


def test_sur_matches_linearmodels(sim):
    lm = pytest.importorskip("linearmodels.system")
    frame, data = sim
    system = System([Equation("y1 ~ x1 + x3"), Equation("y2 ~ x2 + z1")])
    spec = {"y1": {"dependent": frame.y1, "exog": _X(frame, "x1", "x3")}, "y2": {"dependent": frame.y2, "exog": _X(frame, "x2", "z1")}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cov, lm_cov in (("classic", "unadjusted"), ("white", "robust")):
            ours = system.fit(data, method=SUR(cov=cov))
            theirs = lm.SUR(spec).fit(method="gls", cov_type=lm_cov)
            assert np.allclose(ours.params.values, theirs.params.values, atol=1e-10)
            assert np.allclose(ours.params.se, theirs.std_errors.values, atol=1e-10)
        iterated = system.fit(data, method=SUR(iterate=True, tol=1e-12))
        theirs = lm.SUR(spec).fit(method="gls", iterate=True, iter_limit=1000, tol=1e-12)
    assert np.allclose(iterated.params.values, theirs.params.values, atol=1e-7)
    # at convergence the covariance is the GLS formula with the converged residual covariance
    X = block_diag(_X(frame, "x1", "x3").to_numpy(), _X(frame, "x2", "z1").to_numpy())
    W = np.kron(np.linalg.inv(iterated.sigma.to_numpy()), np.eye(len(frame)))
    assert np.allclose(iterated.params.se, np.sqrt(np.diag(np.linalg.inv(X.T @ W @ X))), atol=1e-10)
    assert list(ours.params.names) == ["y1.const", "y1.x1", "y1.x3", "y2.const", "y2.x2", "y2.z1"]


def test_three_stage_least_squares_matches_linearmodels(sim):
    lm = pytest.importorskip("linearmodels.system")
    frame, data = sim
    system = System([Equation("y1 ~ x1 + y2"), Equation("y2 ~ x2 + z1 + z2")], instruments=["x3"])
    assert [str(z) for z in system.instrument_list()] == ["x1", "x2", "z1", "z2", "x3"]  # y2 is a left-hand side: endogenous
    ours = system.fit(data, method=ThreeSLS())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = lm.IV3SLS({
            "y1": {"dependent": frame.y1, "exog": _X(frame, "x1"), "endog": frame[["y2"]], "instruments": frame[["x2", "z1", "z2", "x3"]]},
            "y2": {"dependent": frame.y2, "exog": _X(frame, "x2", "z1", "z2"), "instruments": frame[["x1", "x3"]]},
        }).fit(cov_type="unadjusted")
    assert np.allclose(ours.params.values, theirs.params.values, atol=1e-10)
    assert np.allclose(ours.params.se, theirs.std_errors.values, atol=1e-10)
    assert ours.params["y1.y2"] == pytest.approx(-0.7, abs=0.15)
    assert ours.stats["j_df"] == 5 and 0 <= ours.stats["j_pvalue"] <= 1
    assert "Hansen-Sargan" in str(ours) and "Instruments: const, x1, x2, z1, z2, x3" in str(ours)


def test_simpler_methods_and_their_relations(sim):
    _, data = sim
    same = System([Equation("y1 ~ x1 + x2"), Equation("y2 ~ x1 + x2")])
    sur = same.fit(data, method=SUR())
    for name, eq in (("y1", "y1 ~ x1 + x2"), ("y2", "y2 ~ x1 + x2")):  # identical regressors: SUR is OLS
        assert np.allclose(sur[name].params.values, Equation(eq).fit(data).params.values)
    system = System([Equation("y1 ~ x1 + x3"), Equation("y2 ~ x2 + z1")])
    ols = system.fit(data, method=SystemOLS())
    assert np.allclose(ols["y1"].params.values, Equation("y1 ~ x1 + x3").fit(data).params.values)
    iv = System([Equation("y1 ~ x1 + y2"), Equation("y2 ~ x2 + z1 + z2")]).fit(data, method=TwoSLS())
    from pymetrix import IV

    single = Equation("y1 ~ x1 + y2").fit(data, method=IV(endog=("y2",), instruments=("x2", "z1", "z2")))
    assert np.allclose(iv["y1"].params.values, single.params.values)
    assert sur.test_diagonal().p_value < 0.05  # the errors are correlated (0.7)


def test_cross_equation_restrictions_two_ways(sim):
    _, data = sim
    restricted = System([Equation("y1 ~ x1 + x3"), Equation("y2 ~ x2 + x1")], restrictions="y1.x1 = y2.x1").fit(data, method=SUR())
    assert restricted.params["y1.x1"] == pytest.approx(restricted.params["y2.x1"], abs=1e-10)
    assert restricted.stats["restrictions"] == 1
    shared = System({"y1": Equation("y1 ~ a1 + b*x1 + c1*x3", params=["a1", "b", "c1"]),
                     "y2": Equation("y2 ~ a2 + d*x2 + b*x1", params=["a2", "d", "b"])}).fit(data, method=SUR())
    assert shared.params["b"] == pytest.approx(restricted.params["y1.x1"], abs=1e-8)
    se_r = dict(zip(restricted.params.names, restricted.params.se))
    se_s = dict(zip(shared.params.names, shared.params.se))
    assert se_s["b"] == pytest.approx(se_r["y1.x1"], rel=1e-6)
    free = System([Equation("y1 ~ x1 + x3"), Equation("y2 ~ x2 + x1")]).fit(data, method=SUR())
    test = free.wald("y1.x1 = y2.x1")
    assert test.null == "y1.x1 = y2.x1" and test.df == 1
    with pytest.raises(ExpressionError, match="unknown coefficient"):
        System([Equation("y1 ~ x1")], restrictions="y1.x9 = 0").fit(data)


def test_nonlinear_system(sim):
    rng = np.random.default_rng(5)
    frame, _ = sim
    n = len(frame)
    x = rng.uniform(0, 2, n)
    e = rng.multivariate_normal([0, 0], [[0.04, 0.02], [0.02, 0.04]], size=n)
    f = frame.assign(x=x, q1=1.0 + 2.0 * np.exp(0.5 * x) + e[:, 0], q2=3.0 - 1.0 * x**0.5 + e[:, 1])
    data = Dataset.from_frame(f, time="t")
    fit = System([Equation("q1 ~ a + b*exp(c*x)", params=["a", "b", "c"]), Equation("q2 ~ g - h*x^0.5", params=["g", "h"])]).fit(
        data, method=SUR(iterate=True))
    assert fit.params["c"] == pytest.approx(0.5, abs=0.1) and fit.params["h"] == pytest.approx(1.0, abs=0.1)


def test_simultaneous_model_with_an_identity():
    data = Dataset.read(str(DATA))
    system = System([Equation("c ~ y + c[-1]"), Equation("i ~ y[-1] + i[-1]"), Equation("y = c + i + g")])
    assert system.endogenous_variables == ("c", "i", "y")
    assert [str(z) for z in system.instrument_list()] == ["c[-1]", "y[-1]", "i[-1]", "g"]
    fit = system.fit(data, method=ThreeSLS())
    assert fit.names == ("c", "i") and len(fit.equations) == 2
    model = fit.model()  # the estimated equations with the identity, ready to solve
    assert set(model.endogenous) == {"c", "i", "y"}
    solution = model.solve(data, ("2020Q1", "2021Q4"))
    assert np.all(np.isfinite(solution.frame.loc[pd.Period("2020Q1"):pd.Period("2021Q4"), "y"]))
    assert set(Model([fit]).endogenous) == {"c", "i", "y"}


def test_errors(sim):
    _, data = sim
    with pytest.raises(ValueError, match="too few instruments"):
        System([Equation("y1 ~ x1 + y2 + x2 + x3 + z1 + z2"), Equation("y2 ~ x1")]).fit(data, method=ThreeSLS())
    with pytest.raises(ExpressionError, match="at least one equation"):
        System([Equation("y1 = x1 + x2")])
    with pytest.raises(ExpressionError, match="more than one equation"):
        System([Equation("y1 ~ x1"), Equation("y1 ~ x2")])
    with pytest.raises(KeyError, match="no equation"):
        System([Equation("y1 ~ x1")]).fit(data)["y9"]


def test_dialog_records_and_replays(sim, tmp_path):
    from pymetrix.commands import DefineEquation, FitSystem, Session, load_project, save_project

    _, data = sim
    s = Session()
    s.objects["data"] = data
    s.run(DefineEquation("y1 ~ x1 + y2", name="e1"))
    s.run(DefineEquation("y2 ~ x2 + z1 + z2", name="e2"))
    cmd = FitSystem("e1, e2", method="3sls", method_settings="iterate=True", instruments="x3", restrictions="", start="1975Q1")
    assert cmd.validate(s) == []
    s.run(cmd)
    assert "sys = System([e1, e2], instruments=['x3']).fit(data, method=ThreeSLS(iterate=True), span=('1975Q1', None))" in s.script()
    assert s["sys"].provenance.sample[0] == "1975Q1"
    assert FitSystem("e1, e9").validate(s) == ["no equation named 'e9'"]
    reopened = load_project(save_project(s, tmp_path / "sys.pymx"))
    assert np.allclose(reopened["sys"].params.values, s["sys"].params.values)


def test_views(sim):
    pytest.importorskip("matplotlib")
    from pymetrix.ui.render import view_sections

    _, data = sim
    fit = System([Equation("y1 ~ x1 + x3"), Equation("y2 ~ x2 + z1")]).fit(data)
    assert [t for t, _ in view_sections(fit)] == ["Summary", "Graph", "Equations", "Residual covariance", "Residual correlation", "Residuals"]
    assert len(fit.plot().panels) == 2
