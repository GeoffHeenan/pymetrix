"""Model.estimate(cov=...): "classic" (the default) is the inverse Hessian, as before; "none" skips it, giving the same mode
with no standard errors. Template.estimate and DynareModel.estimate pass it through."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, read_mod, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError


@pytest.fixture(scope="module")
def ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=120, freq="Q", name="time")))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model, data


def test_none_gives_the_same_mode_without_standard_errors(ar1):
    model, data = ar1
    classic = model.estimate(data, ["x"], estimate=["rho", "sd_e"], shock_sd={"e": 0.5})
    default_is_classic = model.estimate(data, ["x"], estimate=["rho", "sd_e"], shock_sd={"e": 0.5}, cov="classic")
    none = model.estimate(data, ["x"], estimate=["rho", "sd_e"], shock_sd={"e": 0.5}, cov="none")
    np.testing.assert_array_equal(classic.params.values, default_is_classic.params.values)
    np.testing.assert_array_equal(classic.params.cov, default_is_classic.params.cov)
    np.testing.assert_array_equal(none.params.values, classic.params.values)
    assert np.all(np.isfinite(classic.params.se)) and np.all(np.isnan(none.params.se))
    assert none.stats["loglike"] == classic.stats["loglike"]
    assert "Standard errors not computed" in str(none.summary()) and "Standard errors not computed" not in str(classic.summary())
    assert "covariance" not in classic.provenance.method and none.provenance.method["covariance"] == "none"
    with pytest.raises(ModelError, match="'classic'.*'none'"):
        model.estimate(data, ["x"], estimate=["rho"], cov="hac")


def test_template_passes_it_through():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=4)
    truth.index = pd.period_range("1990Q1", periods=80, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    fit = nk.estimate(data, estimate=["rho"], cov="none")
    assert np.all(np.isnan(fit.params.se)) and fit.provenance.method["covariance"] == "none"


def test_dynare_model_passes_it_through(tmp_path):
    rng = np.random.default_rng(3)
    x = np.zeros(100)
    for t in range(1, 100):
        x[t] = 0.7 * x[t - 1] + 0.2 * rng.normal()
    (tmp_path / "ardata.m").write_text("y = [" + "\n".join(f"{v:.12g}" for v in x) + "];\n", encoding="utf-8")
    path = tmp_path / "ar.mod"
    path.write_text("""
var y; varexo e; parameters rho; rho = 0.5;
model(linear); y = rho*y(-1) + e; end;
shocks; var e; stderr 0.2; end;
estimated_params; rho, 0.5, -0.99, 0.99; stderr e, 0.2, 0.01, 5; end;
varobs y;
estimation(datafile=ardata, mh_replic=0);
""", encoding="utf-8")
    dyn = read_mod(path)
    classic, none = dyn.estimate(), dyn.estimate(cov="none")
    np.testing.assert_array_equal(none.params.values, classic.params.values)
    assert np.all(np.isfinite(classic.params.se)) and np.all(np.isnan(none.params.se))
