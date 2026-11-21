"""Reading IRIS model files: checked against the same models read from Dynare files and built by hand; reporting
equations against the same quantities worked out by hand from given paths."""

import numpy as np
import pytest
import pandas as pd

from pymetrix import read_iris, read_mod
from pymetrix.models import MeasurementMap, ModelError
from pymetrix.models.iris import _expand_for

from test_dynare import NK, RBC

IRIS_NK = """
%{
  The three-equation New Keynesian model of test_dynare, written for IRIS.
%}
!transition_variables
    "Output gap" y, "Inflation" pi, i, r
!transition_shocks
    ey, epi, ei
!parameters
    sig = 1, bet = 0.99, kap = 0.1, rho = 0.8, phi
    std_ey, std_epi = 0.3, std_ei = 0.2
!measurement_variables
    "Observed output gap" obs_y, obs_pi
!measurement_shocks
    me_pi
!transition_equations
    "IS curve" y = y{+1} - sig*(i - pi{+1}) + ey;
    'Phillips' pi = bet*pi{+1} + kap*y + epi;   % a comment
    i = rho*i{-1} + (1-rho)*(phi*pi + 0.5*y) + ei;
    r = i - pi{+1} !! r = i - pi;
!measurement_equations
    obs_y = y;
    obs_pi = 2 + 4*pi + me_pi;
"""

IRIS_RBC = """
!transition_variables c, k, y
!log_variables !all_but
!transition_shocks e
!parameters alpha = 0.33, beta = 0.99, delta = 0.025
!transition_equations
    1/c = beta/c{+1}*(1 + alpha*k^(alpha-1) - delta);
    y = exp(e)*k{-1}^alpha;
    k + c = y + (1-delta)*k{-1};
"""


def test_a_new_keynesian_model_solves_as_the_same_model_read_from_dynare():
    nk = read_iris(IRIS_NK, params={"phi": 1.5, "std_ey": 0.5, "std_me_pi": 0.1})
    assert nk.model.endogenous == ("y", "pi", "i", "r")
    assert nk.shocks == ("ey", "epi", "ei") and nk.shock_sd == {"ey": 0.5, "epi": 0.3, "ei": 0.2}
    assert nk.labels == {"y": "Output gap", "pi": "Inflation", "obs_y": "Observed output gap"}
    assert "r = i - pi[+1]" in repr(nk.model.equations[3])  # the dynamic part, before !!
    dyn = read_mod(NK)
    assert np.allclose(nk.solve().irf(8).values, dyn.solve().irf(8).values, atol=1e-12)


def test_measurement_equations_and_errors_give_the_state_space_form_built_by_hand():
    nk = read_iris(IRIS_NK, params={"phi": 1.5, "std_ey": 0.5, "std_me_pi": 0.1})
    assert list(nk.measurement.equations) == ["obs_y = y", "obs_pi = 2 + 4*pi + 0"]
    assert dict(nk.measurement.errors) == {"obs_pi": 0.1}
    system = nk.system()
    by_hand = read_mod(NK).solve().system(MeasurementMap(["obs_y = y", "obs_pi = 2 + 4*pi"], errors={"obs_pi": 0.1}),
                                          shock_sd={"ey": 0.5, "epi": 0.3, "ei": 0.2})
    assert system.observables == by_hand.observables == ("obs_y", "obs_pi")
    for m in ("T", "R", "Q", "Z", "d", "H"):
        assert np.allclose(getattr(system, m), getattr(by_hand, m), atol=1e-12), m
    assert np.diag(system.H) == pytest.approx([0.0, 0.01])


def test_log_variables_give_responses_in_log_deviations():
    rbc = read_iris(IRIS_RBC, guess={"k": 30, "c": 2, "y": 3})
    dyn = read_mod(RBC)
    levels = dyn.steady_state()
    logs = rbc.steady_state()
    for v in ("c", "k", "y"):
        assert logs[v] == pytest.approx(np.log(levels[v]), rel=1e-10)
    ours, theirs = rbc.solve().irf(10), dyn.solve().irf(10)
    order = list(ours.responses)
    scale = np.array([levels[v] for v in order])[None, :, None]
    assert np.allclose(ours.values, theirs.values / scale, atol=1e-10)
    assert "log variables" in rbc.description


def test_for_loops_expand_as_the_iris_preprocessor_does():
    assert " ".join(_expand_for("!for ?c = us, ea !do y_?c_gap = rho_?c*y_?c_gap{-1}; !end").split()) == \
        "y_us_gap = rho_us*y_us_gap{-1}; y_ea_gap = rho_ea*y_ea_gap{-1};"
    assert " ".join(_expand_for("!for us, ea !do !for ?k = a, b !do x_?_?k; !end !end").split()) == "x_us_a; x_us_b; x_ea_a; x_ea_b;"
    two = read_iris("""
    !transition_variables !for us, ea !do y_? !end
    !transition_shocks !for us, ea !do e_? !end
    !parameters !for us, ea !do rho_? !end
    !transition_equations
    !for ?c = us, ea !do
        y_?c = rho_?c*y_?c{-1} + e_?c;
    !end
    """, params={"rho_us": 0.5, "rho_ea": 0.9})
    assert two.model.endogenous == ("y_us", "y_ea") and two.shocks == ("e_us", "e_ea")
    rules = two.solve().rules().to_frame().set_index("")
    assert rules.loc["y_us", "y_us[-1]"] == pytest.approx(0.5) and rules.loc["y_ea", "y_ea[-1]"] == pytest.approx(0.9)


def test_diff_and_difflog():
    t = read_iris("""
    !transition_variables x, g, h
    !transition_shocks e
    !transition_equations
        x = 0.5*x{-1} + e;
        g = diff(x);
        h = diff(x{-1});
    """)
    rules = t.solve().rules().to_frame().set_index("")
    assert rules.loc["g", "e"] == pytest.approx(1.0) and rules.loc["g", "x[-1]"] == pytest.approx(-0.5)
    assert rules.loc["h", "e"] == pytest.approx(0.0, abs=1e-12)
    from pymetrix.models.iris import _to_pymetrix
    assert _to_pymetrix("difflog(z{-1})", set()) == "(log(z[-1]) - log(z[-2]))"


def test_substitutions_and_parameter_expressions():
    t = read_iris("""
    !transition_variables x, z
    !transition_shocks e
    !parameters a = 0.25, b = 2*a, std_e = 1/2
    !substitutions
        pers = b*x{-1};
        both = $pers$ + e;
    !transition_equations
        x = $both$;
        z = x - $pers$;
    """)
    assert t.model.params == {"b": 0.5} and t.shock_sd == {"e": 0.5}
    rules = t.solve().rules().to_frame().set_index("")
    assert rules.loc["x", "x[-1]"] == pytest.approx(0.5) and rules.loc["z", "x[-1]"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ModelError, match=r"no substitution for \$nope\$"):
        read_iris("!transition_variables x\n!transition_equations x = $nope$;")


def test_files_the_dialog_and_errors(tmp_path):
    path = tmp_path / "nk.model"
    path.write_text(IRIS_NK, encoding="utf-8")
    assert read_iris(path, params={"phi": 1.5}).name == "nk"
    assert read_iris(str(path), params={"phi": 1.5}).shock_sd["ey"] == 1.0  # no std_ey value: 1

    from pymetrix.commands import Session
    from pymetrix.commands.builtin import ImportIris

    s = Session()
    s.run(ImportIris(str(path), params="phi=1.5"))
    assert f"iris = read_iris({str(path)!r}, params={{'phi': 1.5}})" in s.script()
    assert s["iris"].solve() is not None

    with pytest.raises(ModelError, match="no values for the parameters"):
        read_iris(IRIS_NK)
    with pytest.raises(ModelError, match="not transition variables"):
        read_iris("!transition_variables x\n!transition_shocks e\n!transition_equations x = 0.5*x{-1} + &q + e;")
    with pytest.raises(ModelError, match="2 transition equations for 1"):
        read_iris("!transition_variables x\n!transition_equations x = 0.5*x{-1}; x = 1;")
    with pytest.raises(ModelError, match="dtrends"):
        read_iris("!transition_variables x\n!transition_equations x = 0.5*x{-1};\n!dtrends x += 1;")
    with pytest.raises(ModelError, match="unknown section"):
        read_iris("!transition_variables x\n!transition_equations x = 0.5*x{-1};\n!nonsense")
    with pytest.raises(ModelError, match="!for"):
        read_iris("!for a, b !do y_? ")


def test_steady_state_references():
    t = read_iris("""
    !transition_variables x, g, y
    !log_variables y
    !transition_shocks e, ey
    !parameters c = 2, ybar = 3, rho = 0.8
    !measurement_variables obs
    !transition_equations
        x = c + 0.5*(x{-1} - &x) + e;
        y = ybar^(1-rho) * y{-1}^rho * exp(ey);
        g = y/&y;
    !measurement_equations
        obs = 100*(x - &x);
    """)
    assert t.model.parameters["ss_x"].value == pytest.approx(2.0) and t.model.parameters["ss_y"].value == pytest.approx(3.0)
    assert t.model.parameters["ss_x"].status == "fixed"  # estimation leaves it alone
    ss = t.steady_state()
    assert ss["x"] == pytest.approx(2.0) and ss["g"] == pytest.approx(1.0) and np.exp(ss["y"]) == pytest.approx(3.0)
    by_hand = read_iris("""
    !transition_variables x, g, y
    !log_variables y
    !transition_shocks e, ey
    !parameters c = 2, ybar = 3, rho = 0.8
    !transition_equations
        x = c + 0.5*(x{-1} - 2) + e;
        y = ybar^(1-rho) * y{-1}^rho * exp(ey);
        g = y/3;
    """)
    assert np.allclose(t.solve().irf(6).values, by_hand.solve().irf(6).values, atol=1e-12)
    assert "ss_x" in list(t.measurement.equations)[0] and t.measurement.params["ss_x"] == pytest.approx(2.0)


def test_if_blocks_choose_by_parameter_values():
    text = """
    !transition_variables x
    !transition_shocks e
    !parameters rho = 0.5
    !transition_equations
    !if version == 1
        x = rho*x{-1} + e;
    !elseif version == 2
        !if smooth && ~fast
            x = 0.9*x{-1} + e;
        !else
            x = 0.1*x{-1} + e;
        !end
    !else
        x = e;
    !end
    """

    def slope(**params):
        rules = read_iris(text, params=params).solve().rules().to_frame().set_index("")
        return rules.loc["x", "x[-1]"] if "x[-1]" in rules.columns else 0.0

    assert slope(version=1) == pytest.approx(0.5)
    assert slope(version=2, smooth=1, fast=0) == pytest.approx(0.9)
    assert slope(version=2, smooth=1, fast=1) == pytest.approx(0.1)
    assert slope(version=3) == 0.0
    with pytest.raises(ModelError, match="cannot evaluate the !if condition"):
        read_iris(text)  # version is not given


IRIS_REPORTING = """
!transition_variables
    "Output gap" y, yp, z
!log_variables z
!transition_shocks
    ey, ep
!parameters
    "Persistence of the gap" rho = 0.5, g = 0.3
    "Size of gap shocks" std_ey = 0.2
    "Size of the error" std_m = 0.1
!measurement_variables
    obs_y
!measurement_shocks
    m
!transition_equations
    y = rho*y{-1} + ey;
    yp = g + 0.9*yp{-1} + ep;
    z = 2*exp(y);
!measurement_equations
    obs_y = y + m;
!reporting_equations
    "Gap, percent" gap = 100*y;
    lvl = yp + g*y;
    chg = lvl - (yp{-1} + g*y{-1});
    "z in levels" zl = 10*z;
"""


def test_reporting_equations_and_descriptions():
    """!reporting_equations become the measurement map's reported series (checked against the same quantities worked out
    by hand from given paths, a log variable in levels and an earlier reported series put in); parameter descriptions
    become the model's parameter metadata, those of std_<shock> the labels of the shock sizes and measurement errors."""
    from pymetrix import Dataset

    t = read_iris(IRIS_REPORTING)
    m = t.measurement
    assert m.observables == ("obs_y",) and [r.observable for r in m.reports] == ["gap", "lvl", "chg", "zl"]
    assert m.labels == {"gap": "Gap, percent", "zl": "z in levels"} and m.errors == {"obs_y": 0.1}
    assert t.model.spec("rho").label == "Persistence of the gap" and t.model.spec("g").label == ""
    assert t.model.spec("sd_ey").label == "Size of gap shocks" and t.model.spec("sd_obs_y").label == "Size of the error"
    assert t.model.spec("sd_ep") is None and t.labels["rho"] == "Persistence of the gap"  # labels kept as before
    y, yp, z = np.array([0.5, -1.0, 2.0]), np.array([1.0, 1.5, 1.2]), np.array([0.1, 0.2, -0.3])  # z: the log
    paths = Dataset.from_frame(pd.DataFrame({"time": ["2020Q1", "2020Q2", "2020Q3"], "y": y, "yp": yp, "z": z}), time="time", freq="Q")
    got = m.reported_values(paths)
    np.testing.assert_allclose(got["gap"], 100 * y, atol=1e-12)
    np.testing.assert_allclose(got["lvl"], yp + 0.3 * y, atol=1e-12)
    np.testing.assert_allclose(got["chg"].to_numpy()[1:], np.diff(yp + 0.3 * y), atol=1e-12)
    assert np.isnan(got["chg"].iloc[0])
    np.testing.assert_allclose(got["zl"], 10 * np.exp(z), atol=1e-12)
    # the model is the same as without the reporting equations
    plain = read_iris(IRIS_REPORTING.split("!reporting_equations")[0])
    np.testing.assert_allclose(t.solve().P, plain.solve().P, atol=1e-14)
    assert not plain.measurement.reports and plain.model.spec("rho").label == "Persistence of the gap"


def test_reporting_equations_alone_and_their_errors():
    """Reporting equations without measurement equations: a map that only reports, which gives no state-space form.
    Problems are named."""
    text = IRIS_REPORTING.replace("!measurement_variables\n    obs_y\n!measurement_shocks\n    m\n", "")
    text = text.replace("!measurement_equations\n    obs_y = y + m;\n", "")
    t = read_iris(text)
    assert t.measurement.observables == () and len(t.measurement.reports) == 4 and "reported gap" in repr(t.measurement)
    with pytest.raises(ModelError, match="observes no series"):
        t.system()
    with pytest.raises(ModelError, match="observes no series"):
        t.solve().system(t.measurement)
    for bad, message in [("q = 100*w;", "w, which are not variables"), ("y = 2*yp;", "y is already declared"),
                         ("gap = 1 + y;", "gap is already reported"), ("q = chg{-1};", "uses chg at another period"),
                         ("q = y{+1};", "cannot depend|present and the past")]:
        with pytest.raises(ModelError, match=message):
            read_iris(IRIS_REPORTING.replace('"z in levels" zl = 10*z;', bad))


QPM_TOOLKIT_NK = """
!modelname
"Three-equation NK, toolkit dialect"

!Transition-Variables
y "Output gap (% of potential)"
pi "Inflation", i, r
!shocks:transition
ey, epi, ei
!parameters
// IS and Phillips curves
sig = 1; bet = 0.99; kap = 0.1;
rho = 0.8                       # policy smoothing
phi, std_ey, std_epi, std_ei, std_me_pi
!assignments
phi = 1.5; std_ey = 0.5; std_epi = 0.3; std_ei = 0.2;
!variables-measurement
obs_y "Observed output gap", obs_pi
!measurement-shocks
me_pi
!parameter-values
std_me_pi = 0.1;
!dynamic-equations
is_curve: y = y{+1} - sig*(i - pi{+1}) + ey;
phillips: pi = bet*pi{+1} + kap*y + epi;   // a comment
i = rho*i{-1} + (1-rho)*(phi*pi + 0.5*y) + ei;
real_rate: r = i - pi{+1};
!equations-measurement
meas_y: obs_y = y;
meas_pi: obs_pi = 2 + 4*pi + me_pi;
"""


def test_the_qpm_toolkit_dialect_reads_as_the_same_model():
    iris = read_iris(IRIS_NK, params={"phi": 1.5, "std_ey": 0.5, "std_me_pi": 0.1})
    qpm = read_iris(QPM_TOOLKIT_NK)
    assert qpm.model.endogenous == iris.model.endogenous and qpm.shock_sd == iris.shock_sd
    assert qpm.labels == {"y": "Output gap (% of potential)", "pi": "Inflation", "obs_y": "Observed output gap"}
    assert "Three-equation NK, toolkit dialect" in qpm.description
    a, b = iris.system(), qpm.system()
    for m in ("T", "R", "Q", "Z", "d", "H"):
        assert np.allclose(getattr(a, m), getattr(b, m), atol=1e-12), m
    with pytest.raises(ModelError, match="not in !parameters"):
        read_iris(QPM_TOOLKIT_NK.replace("std_me_pi = 0.1;", "std_me_pi = 0.1; nope = 1;"))
    with pytest.raises(ModelError, match="unknown section !bogus_block"):
        read_iris(QPM_TOOLKIT_NK.replace("!assignments", "!Bogus-Block"))
