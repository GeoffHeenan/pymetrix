"""Third-order perturbation, checked four ways:

- against Dynare 6.0's own stoch_simul(order=3) (tests/fixtures/dynare-rbc-order3.json and
  dynare-rbc-order3-habit.json, recorded from tests/fixtures/dynare/rbc_order3.mod and rbc_order3_habit.mod): every
  decision-rule term, simult_ paths with and without pruning, and the pruned process's mean;
- against the exact solution of the growth model with log utility and full depreciation (in levels every term is a
  derivative of alpha beta exp(e) k^alpha; in logs the rule is linear, so every term beyond the first order is zero);
- for consistency: the second-order terms equal second_order's, the corrections for risk vanish without risk and grow
  with the shocks' variance, and the Sylvester solver against a dense Kronecker solve;
- by simulation: the variance of the pruned process from many simulated paths against Dynare's theoretical variance.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pymetrix import Equation, Model, read_mod, second_order, templates, third_order
from pymetrix.metadata import Role
from pymetrix.models.third_order import _kron_sylvester

FIXTURES = Path(__file__).parent / "fixtures"
ALPHA, BETA = 0.33, 0.96
TERMS = [("ghx", "x"), ("ghu", "u"), ("ghxx", "xx"), ("ghxu", "xu"), ("ghuu", "uu"), ("ghs2", ""), ("ghxxx", "xxx"),
         ("ghxxu", "xxu"), ("ghxuu", "xuu"), ("ghuuu", "uuu"), ("ghxss", "x"), ("ghuss", "u")]


def _dynare_name(name: str, endo: list[str]) -> str:
    """Dynare's AUX_ENDO_LAG_3_2 (the variable with zero-based index 3, two periods back) is pymetrix's z_lag2."""
    if name.startswith("AUX_ENDO_LAG_"):
        index, lag = name[len("AUX_ENDO_LAG_"):].split("_")
        return f"{endo[int(index)]}_lag{lag}"
    return name


def _compare(s, ref):
    names = [_dynare_name(v, ref["endo_names"]) for v in ref["endo_names"]]
    rows = [names[i - 1] for i in ref["order_var"]]  # Dynare's rows are in its decision-rule order
    states = [names[i - 1] for i in ref["state_var"]]
    assert set(states) == set(s.states) and tuple(ref["exo_names"]) == s.shocks
    ri, si = [s.variables.index(v) for v in rows], [s.states.index(v) for v in states]
    n, ns, k = len(rows), len(si), len(s.shocks)
    for name, kinds in TERMS:
        ours = getattr(s, name)[ri].reshape((n,) + tuple(ns if c == "x" else k for c in kinds))
        for axis, c in enumerate(kinds, start=1):
            if c == "x":
                ours = np.take(ours, si, axis=axis)
        theirs = np.array(ref[name], dtype=float).reshape(n, -1)
        scale = np.abs(theirs).max()
        assert np.allclose(ours.reshape(n, -1), theirs, rtol=1e-9, atol=1e-11 * max(scale, 1e-6)), name
    assert np.allclose(s.ghsss, 0.0, atol=1e-15)  # symmetric shocks: no third-order constant


@pytest.fixture(scope="module")
def rbc3():
    ref = json.loads((FIXTURES / "dynare-rbc-order3.json").read_text(encoding="utf-8"))
    return read_mod(FIXTURES / "dynare" / "rbc_order3.mod").solve(order=3), ref


def test_every_term_against_dynare(rbc3):
    s, ref = rbc3
    assert ref["dynare_version"] == "6.0" and ref["order"] == 3
    assert np.allclose(s.sigma, np.array(ref["Sigma_e"]), rtol=1e-12)  # correlated shocks
    _compare(s, ref)


def test_every_term_against_dynare_with_auxiliary_states():
    """Habits and an AR(4): seven states, three of them auxiliary variables for the longer lags."""
    ref = json.loads((FIXTURES / "dynare-rbc-order3-habit.json").read_text(encoding="utf-8"))
    s = read_mod(FIXTURES / "dynare" / "rbc_order3_habit.mod").solve(order=3)
    assert len(s.states) == 7
    _compare(s, ref)


def test_paths_and_mean_against_dynare(rbc3):
    s, ref = rbc3
    sim = ref["simulation"]
    shocks = np.array(sim["shocks"])
    for kind, pruning in (("plain", False), ("pruned", True)):
        ours = s.simulate(len(shocks), shocks=shocks, pruning=pruning)[ref["endo_names"]].to_numpy()
        assert np.allclose(ours, np.array(sim[kind])[1:], rtol=1e-12, atol=1e-12), kind  # Dynare's first row is the start
    plain = s.simulate(len(shocks), shocks=shocks, pruning=False)
    pruned = s.simulate(len(shocks), shocks=shocks, pruning=True)
    assert not np.allclose(plain.to_numpy(), pruned.to_numpy(), rtol=1e-12, atol=0)
    assert np.allclose(s.mean()[ref["endo_names"]].to_numpy(), ref["pruned_mean"], rtol=1e-12)
    second = read_mod(FIXTURES / "dynare" / "rbc_order3.mod").solve(order=2).mean()
    assert np.allclose(s.mean().to_numpy(), second[list(s.endogenous)].to_numpy(), rtol=1e-12)  # the third order adds nothing


def test_simulated_variance_against_dynare(rbc3):
    """Many independent pruned paths, 400 periods each from the steady state: their spread at the end against
    Dynare's theoretical variance of the pruned process (4,000 paths: the sampling error of a variance is about 2%)."""
    s, ref = rbc3
    U = s._draws((400, 4000), np.random.default_rng(7))
    x, _ = s._run(U, pruning=True)
    keep = [s.variables.index(v) for v in ref["endo_names"]]
    end = x[-1][:, keep]
    V, Vd = np.cov(end.T), np.array(ref["pruned_var"])
    assert np.allclose(np.diag(V), np.diag(Vd), rtol=0.07)
    corr = V / np.sqrt(np.outer(np.diag(V), np.diag(V)))
    corr_d = Vd / np.sqrt(np.outer(np.diag(Vd), np.diag(Vd)))
    assert np.allclose(corr, corr_d, atol=0.05)
    se = np.sqrt(np.diag(Vd) / len(end))
    assert np.all(np.abs(end.mean(axis=0) + np.array(ref["steady_state"]) - np.array(ref["pruned_mean"])) < 4 * se)


@pytest.fixture(scope="module")
def exact():
    kb = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    yb = kb**ALPHA
    levels = _levels_model()
    logs = Model([Equation("exp(-lc) ~ beta*alpha*exp(ly[+1] - lk - lc[+1])", params=["alpha", "beta"]),
                  Equation("ly ~ e + alpha*lk[-1]", params=["alpha"]),
                  Equation("exp(lk) = exp(ly) - exp(lc)")], params={"alpha": ALPHA, "beta": BETA}, roles={"e": Role.SHOCK})
    s = third_order(levels, ["e"], {"e": 0.02}, guess={"k": kb, "y": yb, "c": (1 - ALPHA * BETA) * yb})
    sl = third_order(logs, ["e"], {"e": 0.02}, guess={"lk": np.log(kb), "ly": np.log(yb), "lc": np.log((1 - ALPHA * BETA) * yb)})
    return s, sl, kb


def test_levels_equal_the_exact_rule(exact):
    """k' = alpha beta exp(e) k^alpha: each third-order term is a third derivative of it at the steady state."""
    s, _, kb = exact
    k = s.variables.index("k")
    ab = ALPHA * BETA
    assert s.ghxxx[k, 0] == pytest.approx(ab * ALPHA * (ALPHA - 1) * (ALPHA - 2) * kb ** (ALPHA - 3), rel=1e-11)
    assert s.ghxxu[k, 0] == pytest.approx(ab * ALPHA * (ALPHA - 1) * kb ** (ALPHA - 2), rel=1e-11)
    assert s.ghxuu[k, 0] == pytest.approx(ab * ALPHA * kb ** (ALPHA - 1), rel=1e-11)
    assert s.ghuuu[k, 0] == pytest.approx(kb, rel=1e-11)
    y = s.variables.index("y")
    assert s.ghxxx[y, 0] == pytest.approx(ALPHA * (ALPHA - 1) * (ALPHA - 2) * kb ** (ALPHA - 3), rel=1e-11)
    # the policy does not depend on uncertainty: no correction for risk at any order
    for name in ("ghs2", "ghxss", "ghuss", "ghsss"):
        assert np.allclose(getattr(s, name), 0.0, atol=1e-13), name
    exact_path, kt = [], kb
    shocks = (0.05, 0.0, -0.03, 0.0)
    for e in shocks:
        kt = ab * np.exp(e) * kt**ALPHA
        exact_path.append(kt)
    third = s.simulate(4, shocks=np.array(shocks)[:, None], pruning=False)["k"].to_numpy()
    second = second_order(_levels_model(), ["e"], {"e": 0.02}, guess={"k": kb, "y": kb**ALPHA, "c": (1 - ab) * kb**ALPHA})
    second = second.simulate(4, shocks=np.array(shocks)[:, None])["k"].to_numpy()
    assert abs(third[0] - exact_path[0]) < abs(second[0] - exact_path[0]) / 10  # far closer than the second order


def _levels_model():
    return Model([Equation("1/c ~ beta*alpha*y[+1]/(k*c[+1])", params=["alpha", "beta"]),
                  Equation("y ~ exp(e)*k[-1]^alpha", params=["alpha"]),
                  Equation("k = y - c")], params={"alpha": ALPHA, "beta": BETA}, roles={"e": Role.SHOCK})


def test_logs_are_exactly_linear(exact):
    _, sl, _ = exact
    lk, ly = sl.variables.index("lk"), sl.variables.index("ly")
    assert sl.ghx[lk, 0] == pytest.approx(ALPHA, rel=1e-12) and sl.ghu[ly, 0] == pytest.approx(1.0, rel=1e-12)
    for name in ("ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss", "ghsss"):
        assert np.allclose(getattr(sl, name), 0.0, atol=1e-12), name


def test_second_order_terms_and_risk():
    rbc = templates.rbc()
    s3, s2 = rbc.solve(order=3), rbc.solve(order=2)
    for name in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"):
        assert np.allclose(getattr(s3, name), getattr(s2, name), rtol=1e-10, atol=1e-14), name
    m = rbc.model
    calm = third_order(m, rbc.shocks, {s: 0.0 for s in rbc.shocks}, steady=rbc.steady_state())
    for name in ("ghs2", "ghxss", "ghuss", "ghsss"):
        assert np.allclose(getattr(calm, name), 0.0, atol=1e-15), name
    assert np.allclose(calm.ghxxx, s3.ghxxx, rtol=1e-12)  # the cubic terms do not depend on risk
    sd = {s: 2 * float(np.sqrt(s3.sigma[i, i])) for i, s in enumerate(rbc.shocks)}
    double = third_order(m, rbc.shocks, sd, steady=rbc.steady_state())
    assert np.abs(s3.ghxss).max() > 0
    assert np.allclose(double.ghxss, 4 * s3.ghxss, rtol=1e-10) and np.allclose(double.ghuss, 4 * s3.ghuss, rtol=1e-10)


def test_the_kronecker_sylvester_solver():
    rng = np.random.default_rng(0)
    D = rng.standard_normal((4, 4))
    T = np.triu(rng.standard_normal((3, 3))) * 0.6
    for power in (1, 2, 3):
        F = rng.standard_normal((4, 3**power))
        K = T
        for _ in range(power - 1):
            K = np.kron(K, T)
        dense = np.linalg.solve(np.eye(4 * 3**power) + np.kron(K.T, D), F.reshape(-1, order="F")).reshape(F.shape, order="F")
        assert np.allclose(_kron_sylvester(D, T, F, power), dense, rtol=1e-10, atol=1e-12)


def test_responses_moments_and_rules():
    rbc = templates.rbc()
    s = rbc.solve(order=3)
    small = s.irf(12, shock_sd={"e": 1e-4}).values[:, :, 0] / 1e-4
    linear = rbc.solve().irf(12, shock_sd={"e": 1.0}).values[:, :, 0]
    assert np.allclose(small, linear, rtol=1e-3, atol=1e-6)  # small shocks: the first-order responses
    up = s.irf(12, shock_sd={"e": 0.05}).values[:, :, 0]
    down = s.irf(12, shock_sd={"e": -0.05}).values[:, :, 0]
    assert not np.allclose(up, -down)
    girf = s.irf(12, replications=300, seed=1)
    assert np.allclose(girf.values, s.irf(12).values, rtol=0.05, atol=1e-5)  # with other shocks: close here
    high = s.irf(12, state={"k": rbc.steady_state()["k"] * 1.1}).values
    assert not np.allclose(high, s.irf(12).values, rtol=1e-6)  # responses depend on where they start
    with pytest.raises(ValueError, match="not states"):
        s.irf(4, state={"c": 1.0})
    moments = s.moments(periods=20_000, burn=500)
    assert list(moments.index) == list(s.endogenous)
    assert abs(moments.rows[s.endogenous.index("k")][1] - s.mean()["k"]) < 0.05 * s.mean()["k"]
    rules = s.rules()
    assert rules.columns[0] == "constant" and any(c.count("*") == 2 for c in rules.columns)  # cubic terms
    unpruned = s.simulate(500, seed=2, pruning=False)
    assert np.isfinite(unpruned.to_numpy()).all()
    assert "ThirdOrderSolution" in repr(s)


def test_the_dialog():
    from pymetrix.commands import LoadTemplate, Session, SolveSecondOrder
    from pymetrix.views import view_sections

    s = Session()
    s.run(LoadTemplate("rbc", name="rbc"))
    s.run(SolveSecondOrder("rbc", name="third", order=3))
    assert "third = rbc.solve(order=3)" in s.script()
    assert type(s["third"]).__name__ == "ThirdOrderSolution"
    assert [t for t, _ in view_sections(s["third"])][:3] == ["Decision rules", "Means", "Moments"]
