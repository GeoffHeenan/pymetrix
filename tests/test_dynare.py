"""Reading Dynare .mod files."""

import numpy as np
import pytest

from pymetrix import Equation, Model, read_mod
from pymetrix.models import ModelError

NK = """
// a three-equation New Keynesian model
var y pi i r $r$ (long_name='real rate');
varexo ey epi ei;
parameters sig bet kap rho phi;
sig = 1; bet = 0.99; kap = 0.1;
rho = 0.8; phi = 1 + 0.5;   /* an expression */
model;
[name='IS'] y = y(+1) - sig*(i - pi(+1)) + ey;
[name='Phillips'] pi = bet*pi(+1) + kap*y + epi;
i = rho*i(-1) + (1-rho)*(phi*pi + 0.5*y) + ei;
r = i - pi(+1);
end;
shocks; var ey; stderr 0.5; var epi = 0.09; var ei; stderr 0.2; end;
steady; check;
stoch_simul(order=1, irf=20) y pi i;
"""

RBC = """
var c k y;
varexo e;
parameters alpha beta delta;
alpha = 0.33; beta = 0.99; delta = 0.025;
model;
# mpk = alpha*k^(alpha-1);
1/c = beta/c(+1)*(1 + mpk - delta);
y = exp(e)*k(-1)^alpha;
k + c = y + (1-delta)*k(-1);
end;
initval; k = 30; c = 2; y = 3; end;
"""


def test_a_new_keynesian_model_solves_as_written_by_hand():
    dyn = read_mod(NK)
    assert dyn.endogenous == ("y", "pi", "i", "r") and dyn.exogenous == ("ey", "epi", "ei")
    assert dyn.params == {"sig": 1.0, "bet": 0.99, "kap": 0.1, "rho": 0.8, "phi": 1.5}
    assert dyn.shock_sd == {"ey": 0.5, "epi": pytest.approx(0.3), "ei": 0.2}
    assert dyn.commands == ("steady", "check", "stoch_simul(order=1, irf=20) y pi i")
    assert dyn.equations[0] == "y ~ y[+1] - sig*(i - pi[+1]) + ey" and dyn.equations[3] == "r = i - pi[+1]"
    by_hand = Model([Equation("y ~ y[+1] - s*(i - pi[+1]) + ey", params=["s"]), Equation("pi ~ b*pi[+1] + k*y + epi", params=["b", "k"]),
                     Equation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", params=["r", "phi"]), Equation("rr = i - pi[+1]")],
                    params=dict(s=1.0, b=0.99, k=0.1, r=0.8, phi=1.5)).linearize(shocks=["ey", "epi", "ei"]).solve()
    assert np.allclose(dyn.solve().irf(8).values[:, :3], by_hand.irf(8).values[:, :3], atol=1e-12)


def test_an_rbc_model_with_local_variables_and_a_rewritten_equation():
    dyn = read_mod(RBC)
    assert "mpk" not in " ".join(dyn.equations) and "alpha*k^(alpha-1)" in dyn.equations[0]
    assert dyn.equations[2].startswith("k ~ k + (k + c)")  # two variables on the left: rewritten to solve for k
    ss = dyn.steady_state()
    k = ((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1))
    assert ss["k"] == pytest.approx(k, rel=1e-8) and ss["y"] == pytest.approx(k**0.33, rel=1e-8)
    assert ss["c"] == pytest.approx(k**0.33 - 0.025 * k, rel=1e-8)
    rules = dyn.solve().rules().to_frame().set_index("")
    assert 0.9 < rules.loc["k", "k[-1]"] < 1 and rules.loc["y", "e"] == pytest.approx(k**0.33, rel=1e-6)


def test_files_and_errors(tmp_path):
    path = tmp_path / "nk.mod"
    path.write_text(NK, encoding="utf-8")
    dyn = read_mod(str(path))
    assert dyn.source.endswith("nk.mod")
    with pytest.raises(ModelError, match="no values for the parameters"):
        read_mod("var x; varexo e; parameters a; model; x = a*x(-1) + e; end;")
    with pytest.raises(ModelError, match="2 equations for 1 variables"):
        read_mod("var x; varexo e; model; x = x(-1) + e; x(+1) = x; end;")
    with pytest.raises(ModelError, match="unknown macro directive @#bogus"):  # the macro processor: test_dynare_macros.py
        read_mod("@#bogus N = 2\nvar x;")


def test_the_dialog_imports_a_file(tmp_path):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import ImportDynare
    from pymetrix.ui.render import view_sections

    path = tmp_path / "rbc.mod"
    path.write_text(RBC, encoding="utf-8")
    s = Session()
    s.run(ImportDynare(str(path)))
    assert f"dynare = read_mod({str(path)!r})" in s.script()
    assert [t for t, _ in view_sections(s["dynare"])] == ["Equations", "Parameters"]
