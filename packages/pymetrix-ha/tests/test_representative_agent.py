"""The representative-agent limit: a model with no het block is an ordinary DSGE model, so two independent methods
must agree. sequence-jacobian's RBC example (its first notebook's model), solved through ``HAModel`` by sequence-space
Jacobians, against the same model written as a Dynare .mod file and solved by pymetrix's own QZ first-order solution
(itself checked against Dynare 6.0's recorded output by the dynare-reference workflow)."""

import numpy as np
import pytest

from pymetrix import read_mod
from pymetrix_ha import AR1, HAModel

RHO, SD = 0.9, 0.01


@pytest.fixture(scope="module")
def sequence_space():
    from sequence_jacobian.examples import rbc

    model = HAModel([rbc.household, rbc.firm, rbc.mkt_clearing],
                    calibration={"eis": 1.0, "frisch": 1.0, "delta": 0.025, "alpha": 0.11, "L": 1.0},
                    steady_state={"unknowns": {"vphi": 0.92, "beta": 1 / 1.01, "K": 2.0, "Z": 1.0},
                                  "targets": {"goods_mkt": 0.0, "r": 0.01, "euler": 0.0, "Y": 1.0}, "solver": "hybr"},
                    equilibrium={"unknowns": ["K", "L"], "targets": ["goods_mkt", "euler"]},
                    shocks={"Z": AR1(RHO, SD)}, horizon=300, name="RBC")
    ss = model.steady_state()
    return ss, model.linearize(ss).irf(40)


def _mod(v):
    return f"""
var C K L r w Y I Z;
varexo e;
parameters alpha delta eis frisch vphi beta rho Zbar;
alpha = {v['alpha']!r}; delta = {v['delta']!r}; eis = {v['eis']!r}; frisch = {v['frisch']!r};
vphi = {v['vphi']!r}; beta = {v['beta']!r}; rho = {RHO!r}; Zbar = {v['Z']!r};
model;
r = alpha*Z*(K(-1)/L)^(alpha-1) - delta;
w = (1-alpha)*Z*(K(-1)/L)^alpha;
Y = Z*K(-1)^alpha*L^(1-alpha);
L = (w/vphi/C^(1/eis))^frisch;
I = K - (1-delta)*K(-1);
K = (1-delta)*K(-1) + Y - C;
C = (beta*(1+r(+1))*C(+1)^(-1/eis))^(-eis);
Z = (1-rho)*Zbar + rho*Z(-1) + e;
end;
initval; C = {v['C']!r}; K = {v['K']!r}; L = 1; r = 0.01; w = {v['w']!r}; Y = 1; I = {v['I']!r}; Z = {v['Z']!r}; end;
"""


def test_the_rbc_limit_matches_pymetrix_first_order_solution(sequence_space):
    ss, irf = sequence_space
    dyn = read_mod(_mod(ss.values))
    steady = dyn.steady_state()
    for k in ("C", "K", "Y", "Z"):
        assert steady[k] == pytest.approx(ss[k], rel=1e-10), k
    qz = dyn.solve().irf(40, shock_sd={"e": SD})
    for name in ("K", "C", "L", "r", "w", "Y", "I", "Z"):
        np.testing.assert_allclose(irf[name, "Z"].to_numpy(), qz[name, "e"].to_numpy(), rtol=0, atol=1e-8, err_msg=name)
