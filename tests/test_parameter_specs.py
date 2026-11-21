"""Parameter metadata on models (``Model(specs=[Param(...)])``): what estimation is told, what is reported, what is
read and written.

Checked:
- estimating with specs gives exactly the same fit as estimating with the same priors, bounds, starting values and
  parameter list given to ``estimate`` (the specs are the same information in another place); ``fixed`` leaves a
  parameter out, and naming it is refused; a log transform bounds a parameter at zero;
- a model's spec over an equation's: each field the model's gives wins, the rest come from the equation's;
- reported: the parameter table and the fitted summary (prior, bounds, description), symbols in algebraic notation;
- read: Dynare's ``$\\rho$`` and ``long_name`` and ``estimated_params`` as specs; the model text's settings, and their
  problems where they are; written back in pymetrix's syntax (the same text) and Dynare's (read again: the same specs).
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, ModelSource, Param, read_mod, templates
from pymetrix.estimation.priors import Beta, Gamma, InvGamma1, Uniform
from pymetrix.errors import SpecificationError
from pymetrix.models import Model, ModelError
from pymetrix.models.writers import algebraic_equations
from pymetrix.notation import from_latex


@pytest.fixture(scope="module")
def nk_data():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(300, nk.shock_sd, seed=11)
    truth.index = pd.period_range("1950Q1", periods=300, freq="Q", name="time")
    return nk, nk.measurement.apply(Dataset.from_frame(truth))


def _with_specs(nk, specs):
    m = nk.model
    return replace(nk, model=Model(m.equations, roles=m.roles, params=m.params, specs=specs))


def test_specs_are_what_estimate_is_told(nk_data):
    nk, data = nk_data
    specs = [Param("rho", prior=Beta(0.7, 0.1), start=0.75, label="policy smoothing"), Param("sd_ey", prior=InvGamma1(0.5, 2.0), start=0.4),
             Param("sd_epi", bounds=(0.05, 2.0)), Param("kappa", fixed=True), Param("beta", fixed=True), Param("sigma", fixed=True),
             Param("phi_pi", fixed=True), Param("phi_y", fixed=True), Param("sd_ei", fixed=True)]
    fit = _with_specs(nk, specs).estimate(data)
    assert fit.params.names == ("rho", "sd_ey", "sd_epi")  # the fixed ones are left out
    # the same information given to estimate: the same fit, to the last digit
    start = templates.new_keynesian(rho=0.75)
    by_hand = replace(start, shock_sd={**start.shock_sd, "ey": 0.4}).estimate(
        data, estimate=["rho", "sd_ey", "sd_epi"], priors={"rho": Beta(0.7, 0.1), "sd_ey": InvGamma1(0.5, 2.0)},
        bounds={"sd_epi": (0.05, 2.0)})
    assert np.allclose(fit.params.values, by_hand.params.values, rtol=1e-10) and fit.stats["logpost"] == pytest.approx(by_hand.stats["logpost"], abs=1e-9)
    assert fit.bounds["sd_epi"] == (0.05, 2.0)
    summary = fit.summary()
    assert summary.rows[0][2] == "Beta(0.7, 0.1)" and summary.rows[0][4] == "policy smoothing"
    with pytest.raises(ModelError, match="kappa are fixed"):
        _with_specs(nk, specs).estimate(data, estimate=["rho", "kappa"])
    logged = _with_specs(nk, [Param("phi_y", transform="log")]).estimate(data, estimate=["phi_y"])  # no bounds of its own
    assert logged.bounds["phi_y"] == (0.0, np.inf)
    kept = _with_specs(nk, [Param("rho", transform="log")]).estimate(data, estimate=["rho"])  # its equation's bounds stand
    assert kept.bounds["rho"] == (0.0, 1.0)


def test_a_models_spec_over_an_equations():
    eq = Equation("x ~ a*x[-1] + e", params=[Param("a", symbol="α_1", label="persistence", bounds=(0, 1))])
    m = Model([eq], params={"a": 0.5}, specs=[Param("a", prior=Beta(0.5, 0.2), label="AR coefficient")])
    spec = m.spec("a")
    assert spec.label == "AR coefficient" and spec.symbol == "α_1" and spec.bounds == (0, 1) and spec.prior == Beta(0.5, 0.2)
    table = m.parameters.table()
    assert table.rows[0][5] == "Beta(0.5, 0.2)" and table.rows[0][6] == "AR coefficient" and table.rows[0][4] == "[0, 1]"
    assert algebraic_equations(m)[0].text == "xₜ = α_1 xₜ₋₁ + eₜ"  # a symbol given is shown as given
    with pytest.raises(SpecificationError, match="not parameters of the model"):
        Model([eq], params={"a": 0.5}, specs=[Param("b")])
    with pytest.raises(SpecificationError, match="more than one spec"):
        Model([eq], params={"a": 0.5}, specs=[Param("a"), Param("a")])
    with pytest.raises(SpecificationError, match="transform"):
        Param("a", transform="exp")
    assert Param("s", transform="logit").limits == (0.0, 1.0) and Param("s", transform="log", bounds=(1, 2)).limits == (1, 2)


DYNARE = """
var y $y$ (long_name='output');
varexo e $\\varepsilon$ (long_name='demand shock');
parameters rho $\\rho_{y}$ (long_name='persistence') c;
rho = 0.5; c = 1;
model;
y = c*0 + rho*y(-1) + e;
end;
shocks; var e; stderr 0.2; end;
estimated_params;
rho, beta_pdf, 0.6, 0.1;
c, 1.5, 0, 10;
stderr e, inv_gamma_pdf, 0.1, inf;
end;
"""


def test_dynare_names_descriptions_and_priors_become_specs():
    assert from_latex("\\phi_{\\pi}") == "φ_π" and from_latex("$\\varrho$") == "ϱ" and from_latex("\\mathit{ey}") == "ey"
    dyn = read_mod(DYNARE)
    rho, c = dyn.model.spec("rho"), dyn.model.spec("c")
    assert (rho.symbol, rho.label, rho.prior) == ("ρ_y", "persistence", Beta(0.6, 0.1))
    assert c.bounds == (0.0, 10.0) and c.start == 1.5 and dyn.model.spec("sd_e").prior.mean == 0.1
    assert dyn.labels == {"y": "output", "e": "demand shock"}
    assert algebraic_equations(dyn)[0].text.startswith("yₜ = ") and "ρ_y yₜ₋₁" in algebraic_equations(dyn)[0].text
    written = ModelSource(DYNARE, "dynare").export("dynare")
    assert "rho $\\rho_y$ (long_name='persistence')" in written.text and "stderr e, inv_gamma1_pdf, 0.1, inf;" in written.text
    again = read_mod(written.text)
    for name in ("rho", "c", "sd_e"):
        a, b = dyn.model.spec(name), again.model.spec(name)
        assert (a.prior, a.bounds, a.label, a.symbol) == (b.prior, b.bounds, b.label, b.symbol), name


TEXT = '''parameters
    rho = 0.8  bounds 0 1  prior beta(0.7, 0.1)  symbol ϱ    "policy smoothing"
    kappa = 0.1  fixed
    c = 2  prior uniform(0, 10)  start 2.5
shocks
    e = 1  prior gamma(0.5, 0.2)    "demand"
equations
    y = rho*y[-1] + kappa*y[-2] + c*0 + e
'''


def test_the_model_text():
    m = ModelSource(TEXT).build().model
    assert m.spec("rho") == Param("rho", label="policy smoothing", symbol="ϱ", bounds=(0.0, 1.0), prior=Beta(0.7, 0.1))
    assert m.spec("kappa").fixed and m.spec("c").start == 2.5 and m.spec("c").prior == Uniform(0, 10)
    assert m.spec("sd_e").prior == Gamma(0.5, 0.2)
    check = ModelSource(TEXT).check()
    rho = next(s for s in check.symbols if s.name == "rho")
    assert rho.estimation == "Beta(0.7, 0.1); bounds 0 to 1; symbol ϱ" and rho.description == "policy smoothing"
    assert ModelSource(TEXT).export("pymetrix").text.split("\n", 1)[1] == TEXT  # written back as it was
    bad = ModelSource(TEXT.replace("prior beta(0.7, 0.1)", "prior betta(0.7, 0.1)")).check()
    assert bad.errors[0].line == 2 and bad.errors[0].column == 34 and "did you mean 'beta'" in bad.errors[0].message
    odd = ModelSource(TEXT.replace("fixed", "fixd")).check()
    assert odd.errors[0].line == 3 and "'fixd' does not read here" in odd.errors[0].message
    iris = ModelSource(TEXT).export("iris")
    assert '"policy smoothing" rho = 0.8' in iris.text and any("priors, bounds and fixed" in n for n in iris.notes)
