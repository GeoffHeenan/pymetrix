"""Estimating models read from Dynare files: estimated_params, varobs, estimation options, loglinear, .m data.

The fs2000 comparison with Dynare's own output is in test_dynare_fs2000.py."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pymetrix import Dataset, Equation, Model, read_mod
from pymetrix.estimation.priors import Beta, Gamma, InvGamma, InvGamma1, InvGamma2, Normal, Uniform
from pymetrix.models import MeasurementMap, ModelError
from pymetrix.models.dynare import read_m_data

AR = """
var y;
varexo e;
parameters rho a b c del;
rho = 0.5; a = 1; b = 2; c = 3; del = 0.1;
model;
log(y) = rho*log(y(-1)) + e;
end;
initval; y = 1; end;
shocks; var e; stderr 0.2; end;
estimated_params;
rho, beta_pdf, 0.6, 0.1;
a, 1.5, 0, 10;
b, 2.5, , , normal_pdf, 2, 0.5;
c, uniform_pdf, , , 0, 10;
del, gamma_pdf, 0.1, 0.05;
stderr e, inv_gamma_pdf, 0.1, inf;
end;
varobs y;
estimation(datafile=ardata, first_obs=3, nobs=100, mh_replic=0);
"""


def test_estimated_params_in_all_three_forms():
    dyn = read_mod(AR)
    assert dyn.estimated == ("rho", "a", "b", "c", "del", "sd_e")
    assert dyn.priors == {"rho": Beta(0.6, 0.1), "b": Normal(2.0, 0.5), "c": Uniform(0.0, 10.0), "del": Gamma(0.1, 0.05),
                          "sd_e": InvGamma1(0.1, math.inf)}
    assert dyn.start == {"a": 1.5, "b": 2.5} and dyn.bounds == {"a": (0.0, 10.0)}
    assert dyn.varobs == ("y",) and dyn.options == {"datafile": "ardata", "first_obs": 3, "nobs": 100, "mh_replic": 0}
    assert not dyn.loglinear
    shapes = read_mod(AR.replace("del, gamma_pdf", "del, 2").replace("c, uniform_pdf, , , 0, 10", "c, uniform_pdf, 5, 1"))
    assert "del" not in shapes.priors and shapes.priors["c"] == Uniform(5 - math.sqrt(3), 5 + math.sqrt(3))
    two = read_mod(AR.replace("inv_gamma_pdf", "inv_gamma2_pdf").replace("0.1, inf", "0.1, 0.05"))
    assert two.priors["sd_e"] == InvGamma2(0.1, 0.05) and InvGamma2 is InvGamma
    capitals = read_mod(AR.replace("beta_pdf", "BETA_PDF").replace("normal_pdf", "Normal_Pdf"))  # Dynare ignores case
    assert capitals.priors == dyn.priors
    with pytest.raises(ModelError, match="not a prior shape"):  # never drop a prior silently
        read_mod(AR.replace("b, 2.5, , , normal_pdf, 2, 0.5", "b, 2.5, , , lognormal_pdf, 2, 0.5"))


def test_the_inverse_gamma_of_type_1_is_dynares():
    for mean, sd in [(0.035449, math.inf), (0.1, 0.05), (0.01, 0.001), (1.0, 1.0), (5.0, 1.0), (1.0, 0.3)]:  # means of 1 or more too
        prior = InvGamma1(mean, sd)
        density = lambda x, p=prior: math.exp(p.logpdf(x))  # noqa: E731
        assert integrate.quad(density, 0, math.inf, limit=500)[0] == pytest.approx(1.0, abs=1e-6)
        assert integrate.quad(lambda x: x * density(x), 0, math.inf, limit=500)[0] == pytest.approx(mean, rel=1e-6)
        if math.isfinite(sd):
            second = integrate.quad(lambda x: x * x * density(x), 0, math.inf, limit=500)[0]
            assert math.sqrt(second - mean**2) == pytest.approx(sd, rel=1e-5)
    nu, s = InvGamma1(0.035449, math.inf).nu_s
    assert nu == 2 and s == pytest.approx(2 * 0.035449**2 / math.pi)
    # sigma^2 is inverse gamma(nu/2, s/2): draws of sigma have the prior's mean
    nu, s = InvGamma1(0.1, 0.05).nu_s
    draws = np.sqrt(1 / np.random.default_rng(0).gamma(nu / 2, 2 / s, size=200000))
    assert draws.mean() == pytest.approx(0.1, rel=0.01)
    assert InvGamma1(0.1, 0.05).logpdf(-1.0) == -math.inf and Beta(0.5, 0.1).support == (0.0, 1.0)


def test_m_data_files(tmp_path):
    path = tmp_path / "d.m"
    path.write_text("% a comment\nraw=[1 10 2\n2 20 4\n4 40 8];\nratio = raw(:,2)./raw(:,3); %per head\n"
                    "growth=diff(log(raw(:,1)));\nrow = [1, 2, 3];\nscaled = 100*growth.^2;\n", encoding="utf-8")
    v = read_m_data(path)
    assert v["raw"].shape == (3, 3) and np.allclose(v["ratio"], [5, 5, 5]) and np.allclose(v["growth"], [math.log(2)] * 2)
    assert np.allclose(v["row"], [1, 2, 3]) and np.allclose(v["scaled"], 100 * math.log(2) ** 2)
    path.write_text("x = [1 2 3]';\n", encoding="utf-8")
    with pytest.raises(ModelError, match="transposes"):
        read_m_data(path)
    path.write_text("for i = 1:3\n", encoding="utf-8")
    with pytest.raises(ModelError, match="cannot read this MATLAB"):
        read_m_data(path)


def _ar_data(tmp_path, n=120, logged=False):
    rng = np.random.default_rng(3)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + 0.2 * rng.normal()
    values = x if logged else np.exp(x)
    (tmp_path / "ardata.m").write_text("y = [" + "\n".join(f"{v:.12g}" for v in values) + "];\n", encoding="utf-8")
    return x


def test_loglinear_models_are_solved_in_logs_and_estimated_like_the_model_written_in_logs(tmp_path):
    x = _ar_data(tmp_path)
    text = AR.replace("estimation(datafile=ardata, first_obs=3, nobs=100, mh_replic=0)", "estimation(datafile=ardata, loglinear)")
    path = tmp_path / "ar.mod"
    path.write_text(text, encoding="utf-8")
    dyn = read_mod(path)
    assert dyn.loglinear and dyn.guess == {"y": 0.0} and "exp(y[-1])" in dyn.equations[0]
    data = dyn.data()
    assert np.allclose(data.frame["y"].to_numpy(), x)  # the reader takes logs of the data
    assert data.frame.index[0] == pd.Period(year=1, quarter=1, freq="Q")  # Dynare's default first period; pandas 3.1 prints it 0001Q1
    fit = dyn.estimate(estimate=["rho", "sd_e"])
    by_hand = Model([Equation("y ~ rho*y[-1] + e", params=["rho"])], params={"rho": 0.5}).estimate(
        Dataset.from_frame(pd.DataFrame({"y": x}, index=data.frame.index)), MeasurementMap.direct(["y"]), shocks=["e"],
        shock_sd={"e": 0.2}, estimate=["rho", "sd_e"], priors={"rho": Beta(0.6, 0.1), "sd_e": InvGamma1(0.1, math.inf)})
    assert fit.stats["logpost"] == pytest.approx(by_hand.stats["logpost"], abs=1e-6)
    assert np.allclose(fit.params.values, by_hand.params.values, atol=1e-4)
    assert fit.bounds["rho"] == (0.0, 1.0) and fit.bounds["sd_e"] == (0.0, math.inf)  # from the priors' supports


def test_the_sample_options_and_logdata(tmp_path):
    x = _ar_data(tmp_path, logged=True)
    path = tmp_path / "ar.mod"
    path.write_text(AR.replace("mh_replic=0", "loglinear, logdata"), encoding="utf-8")
    data = read_mod(path).data(start="1990Q1")
    assert len(data.frame) == 100 and np.allclose(data.frame["y"].to_numpy(), x[2:102])
    assert str(data.frame.index[0]) == "1990Q1"


def test_estimation_errors(tmp_path):
    with pytest.raises(ModelError, match="corr"):
        read_mod(AR.replace("stderr e, inv_gamma_pdf, 0.1, inf;", "corr e, e, normal_pdf, 0, 0.1;"))
    with pytest.raises(ModelError, match="generalised prior bounds"):
        read_mod(AR.replace("rho, beta_pdf, 0.6, 0.1;", "rho, beta_pdf, 0.6, 0.1, -1, 1;"))
    with pytest.raises(ModelError, match="not a parameter"):
        read_mod(AR.replace("rho, beta_pdf", "zeta, beta_pdf"))
    with pytest.raises(ModelError, match="stderr must name a shock"):
        read_mod(AR.replace("stderr e,", "stderr rho,"))  # stderr y is y's measurement error (y is in varobs)
    with pytest.raises(ModelError, match="varobs names"):
        read_mod(AR.replace("varobs y;", "varobs z;"))
    with pytest.raises(ModelError, match="was not found"):
        read_mod(AR).data()
    with pytest.raises(ModelError, match="no varobs"):
        read_mod(AR.replace("varobs y;", "")).estimate()


def test_a_dynare_file_imported_in_the_app_is_estimated_with_its_own_data_and_priors(tmp_path):
    """Model > Import Dynare file, then Model > Estimate model with the data left empty: the file's datafile, priors and
    options, as Dynare's estimation command; Metropolis-Hastings with its advanced settings; the script says the same."""
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateModel, ImportDynare, RunMCMC

    _ar_data(tmp_path)
    path = tmp_path / "ar.mod"
    path.write_text(AR, encoding="utf-8")
    s = Session()
    s.run(ImportDynare(str(path), name="ar"))
    assert "ar" in s.of_type("Template, DynareModel")  # offered by the Estimate model dialog
    s.run(EstimateModel("ar", data="", estimate="rho, sd_e", name="fit"))
    direct = read_mod(path).estimate(estimate=["rho", "sd_e"])
    assert s["fit"].stats["logpost"] == pytest.approx(direct.stats["logpost"], abs=1e-9)
    s.run(RunMCMC("fit", draws=50, chains=1, burn=20, workers=1, name="post"))
    assert s["post"].draws.shape == (1, 50, 2) and s["post"].burn == 20
    script = s.script()
    assert "fit = ar.estimate(estimate=['rho', 'sd_e'])" in script
    assert "post = fit.mcmc(draws=50, chains=1, burn=20, seed=1, workers=1)" in script
    with pytest.raises(Exception, match="choose the data"):
        from pymetrix import templates

        s.objects["nk"] = templates.new_keynesian()
        s.run(EstimateModel("nk", data="", name="bad"))
