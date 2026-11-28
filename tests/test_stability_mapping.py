"""Prior stability mapping (global sensitivity analysis, Ratto 2008; Dynare's dynare_sensitivity).

Checked against: Dynare 6.0's own stability mapping (its draws, classes, share and Smirnov statistic); the share of
determinate draws where the determinacy region is known in closed form (the New Keynesian
model's Taylor principle, phi_pi > 1 with phi_y = 0, and phi_pi + (1 - beta) phi_y / kappa > 1 with both), within
four binomial standard errors; an AR(1) explosive beyond a = 1 and a model without a steady state for b <= 0; each
draw's class against the model's own Blanchard-Kahn check (``LinearModel.solve``'s refusal); the Smirnov statistic and
p-value against SciPy's ``ks_2samp`` called directly; the same seed, the same draws."""

import math
import sys

import numpy as np
import pytest
from scipy import stats

from pymetrix import Equation, Model, read_mod
from pymetrix.estimation.priors import Normal, Uniform
from pymetrix.metadata import Role
from pymetrix.models import ModelError, templates
from pymetrix.models.sensitivity import OUTCOMES, classify, stability_mapping

NK = """
var y pi i;
varexo ey epi ei;
parameters sigma beta kappa phi_pi phi_y;
sigma = 1; beta = 0.99; kappa = 0.1; phi_pi = 1.5; phi_y = 0;
model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + ey;
pi = beta*pi(+1) + kappa*y + epi;
i = phi_pi*pi + phi_y*y + ei;
end;
shocks; var ey; stderr 0.5; var epi; stderr 0.3; var ei; stderr 0.2; end;
estimated_params;
phi_pi, uniform_pdf, , , 0, 3;
kappa, gamma_pdf, 0.1, 0.05;
stderr ey, inv_gamma_pdf, 0.5, inf;
end;
"""


def _within(share, p, n, k=4.0):
    return abs(share - p) < k * math.sqrt(p * (1 - p) / n)


def test_taylor_principle_share_in_closed_form():
    n = 2000
    st = stability_mapping(templates.new_keynesian(rho=0.0, phi_y=0.0), draws=n, seed=1, priors={"phi_pi": Uniform(0, 3)})
    assert st.names == ("phi_pi",) and len(st.draws) == n
    assert _within(float(st.shares["determinate"]), 2 / 3, n)  # P(phi_pi > 1) = 2/3
    assert float(st.shares["indeterminate"]) == pytest.approx(1 - float(st.shares["determinate"]))
    det = st.draws.loc[st.outcome == "determinate", "phi_pi"]
    assert det.min() > 1 and st.draws.loc[st.outcome != "determinate", "phi_pi"].max() < 1  # the region exactly
    assert st.smirnov.loc["phi_pi", "statistic"] == pytest.approx(1.0) and st.smirnov.loc["phi_pi", "drives"]


def test_two_parameters_region_and_the_smirnov_test_against_scipy():
    n = 2000  # determinate iff phi_pi + (1 - beta) phi_y / kappa > 1: P(phi_pi > 1 - 0.1 phi_y) = (2 - 0.95) / 2
    st = stability_mapping(templates.new_keynesian(rho=0.0), draws=n, seed=2,
                           priors={"phi_pi": Uniform(0, 2), "phi_y": Uniform(0, 1)})
    assert _within(float(st.shares["determinate"]), 0.525, n)
    d = st.draws
    boundary = d["phi_pi"] + 0.1 * d["phi_y"] - 1
    clear = boundary.abs() > 1e-6
    assert ((boundary[clear] > 0) == (d["outcome"][clear] == "determinate")).all()
    good = d["outcome"] == "determinate"
    for name in ("phi_pi", "phi_y"):
        ref = stats.ks_2samp(d.loc[good, name], d.loc[~good, name])
        assert st.smirnov.loc[name, "statistic"] == pytest.approx(ref.statistic, abs=1e-15)
        assert st.smirnov.loc[name, "p-value"] == pytest.approx(ref.pvalue, rel=1e-12)
    assert st.smirnov.loc["phi_pi", "drives"]


def test_dynares_own_draws_classified_as_dynare_does():
    """Dynare 6.0's dynare_sensitivity(stab=1, Nsam=2000) on the same model: its 2,000 quasi-random draws, each classified
    here as Dynare did, its share (52.3%) and its Smirnov statistic for phi_pi (0.975, printed to three decimals)."""
    import json
    from pathlib import Path

    rec = json.loads((Path(__file__).parent / "fixtures" / "dynare-stability-nk.json").read_text(encoding="utf-8"))
    dyn = read_mod(rec["model"])
    model, shocks = dyn.model, dyn.exogenous
    ours = [classify(model, dict(zip(rec["params"], x)), shocks) for x in rec["draws"]]
    assert ours == rec["classes"]
    det = np.array([c == "determinate" for c in ours])
    assert f"{det.mean():.1%}" == rec["printed"]["unique"]
    phi_pi = np.array([x[0] for x in rec["draws"]])
    d = stats.ks_2samp(phi_pi[det], phi_pi[~det]).statistic
    assert round(d, 3) == rec["printed"]["smirnov_unique_phi_pi"][0]
    assert _within(det.mean(), 0.525, len(det))  # and the closed form: phi_pi + 0.1 phi_y > 1


def test_classes_against_the_blanchard_kahn_check():
    nk = templates.new_keynesian()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        values = {"phi_pi": float(rng.uniform(0, 3)), "phi_y": float(rng.uniform(0, 1)), "rho": float(rng.uniform(0, 0.95))}
        got = classify(nk.model, values, nk.shocks)
        m = nk.model.with_parameters(values)
        try:
            m.linearize(m.steady_state(), shocks=list(nk.shocks)).solve()
            expected = "determinate"
        except ModelError as exc:
            expected = "indeterminate" if "not unique" in str(exc) else "explosive" if "no stable solution" in str(exc) else "?"
        assert got == expected, values
        seen.add(got)
    assert seen == {"determinate", "indeterminate"}


def test_explosive_and_no_steady_state():
    ar = Model([Equation("x ~ a*x[-1] + e", params=["a"])], params={"a": 0.5}, roles={"e": Role.SHOCK})
    st = stability_mapping(ar, draws=1000, seed=3, priors={"a": Uniform(0, 2)})
    assert _within(float(st.shares["determinate"]), 0.5, 1000) and float(st.shares["explosive"]) == pytest.approx(
        1 - float(st.shares["determinate"]))
    assert (st.draws.loc[st.outcome == "explosive", "a"] > 1).all()
    lg = Model([Equation("y ~ rho*y[-1] + (1 - rho)*log(b) + e", params=["rho", "b"])], params={"rho": 0.5, "b": 2.0},
               roles={"e": Role.SHOCK})
    with np.errstate(all="ignore"):
        st = stability_mapping(lg, draws=1000, seed=4, priors={"b": Uniform(-1, 3)})
    assert _within(float(st.shares["no steady state"]), 0.25, 1000)
    assert (st.draws.loc[st.outcome == "no steady state", "b"] <= 0).all()
    assert set(st.smirnov_by_class()) == {"no steady state"}


def test_a_dynare_file_draws_its_structural_priors_only():
    dyn = read_mod(NK)
    st = stability_mapping(dyn, draws=1000, seed=5)
    assert st.names == ("phi_pi", "kappa")  # stderr ey is not drawn: it cannot change the solution's existence
    assert _within(float(st.shares["determinate"]), 2 / 3, 1000)
    again = stability_mapping(dyn, draws=1000, seed=5)
    assert again.draws.equals(st.draws)
    assert not stability_mapping(dyn, draws=1000, seed=6).draws.equals(st.draws)
    st2 = stability_mapping(dyn, draws=50, seed=1, priors={"phi_pi": Normal(1.5, 0.1)}, params=["phi_pi"])
    assert st2.names == ("phi_pi",) and float(st2.shares["determinate"]) > 0.9


def test_views_and_refusals():
    from pymetrix.views import sections

    st = stability_mapping(read_mod(NK), draws=200, seed=1)
    titles = [t for t, _ in sections(st)]
    assert titles[:3] == ["Summary", "Smirnov tests", "Graph"] and "Against indeterminate" in titles and titles[-1] == "Draws"
    fig = st.plot()
    assert len(fig.panels) == 2 and [layer.label for layer in fig.panels[0].layers] == ["determinate", "not determinate"]
    assert list(st.counts.index) == list(OUTCOMES) and int(st.counts.sum()) == 200
    with pytest.raises(ModelError, match="no parameter of the model has a prior"):
        stability_mapping(templates.new_keynesian(), draws=10)
    with pytest.raises(ModelError, match="not parameters of the model"):
        stability_mapping(read_mod(NK), draws=10, params=["sd_ey"])
    with pytest.raises(ModelError, match="no prior for"):
        stability_mapping(read_mod(NK), draws=10, params=["sigma"])


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.sensitivity import MapStability

    s = Session()
    s.objects["dyn"] = read_mod(NK)
    cmd = MapStability("dyn", draws=100, seed=1)
    assert cmd.validate(s) == [] and MapStability("nothing").validate(s)
    s.run(cmd)
    out = s.objects["stability"]
    assert len(out.draws) == 100 and "stability_mapping(" in cmd.to_code()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
