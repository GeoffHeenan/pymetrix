"""Measurement errors in Dynare files: ``stderr`` on a varobs variable, in the shocks block (a fixed size) or in
estimated_params (estimated, as sd_<variable>), is the observed series' measurement error, as in Dynare. Checked: the
same likelihood and estimates as the model written by hand with a MeasurementMap carrying those errors."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, read_mod
from pymetrix.estimation.priors import InvGamma1, Normal
from pymetrix.metadata import Role
from pymetrix.models import MeasurementMap, ModelError

BASE = """
var y x;
varexo e u;
parameters rho;
rho = 0.6;
model(linear);
x = rho*x(-1) + e;
y = x + u;
end;
shocks;
var e; stderr 0.5;
var u; stderr 0.3;
{shocks}
end;
estimated_params;
rho, normal_pdf, 0.6, 0.2;
stderr e, inv_gamma_pdf, 0.5, 2;
{estimated}
end;
varobs y;
estimation(datafile=data, mh_replic=0);
"""


def _write(tmp_path, shocks="", estimated=""):
    rng = np.random.default_rng(5)
    n = 150
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + 0.5 * rng.normal()
    y = x + 0.3 * rng.normal(size=n) + 0.4 * rng.normal(size=n)
    (tmp_path / "data.m").write_text("y = [" + "\n".join(f"{v:.12g}" for v in y) + "];\n", encoding="utf-8")
    path = tmp_path / "me.mod"
    path.write_text(BASE.format(shocks=shocks, estimated=estimated), encoding="utf-8")
    return read_mod(path)


def _by_hand(dyn, errors):
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"]), Equation("y = x + u")], params={"rho": 0.6},
                  roles={"e": Role.SHOCK, "u": Role.SHOCK})
    return model, MeasurementMap.direct(["y"], errors)


def fit_at(model, measurement, data):
    """The log posterior of the by-hand model at rho = 0.6, sd_e = 0.5."""
    system = model.linearize(shocks=["e", "u"]).solve().system(measurement, {"e": 0.5, "u": 0.3})
    return system.loglike_observed(*system.observations(data)) + Normal(0.6, 0.2).logpdf(0.6) + InvGamma1(0.5, 2).logpdf(0.5)


def test_fixed_measurement_error(tmp_path):
    dyn = _write(tmp_path, shocks="var y; stderr 0.4;")
    assert dyn.measurement_sd == {"y": 0.4} and "y" not in dyn.shock_sd
    model, measurement = _by_hand(dyn, {"y": 0.4})
    fit = dyn.estimate()
    ref = model.estimate(dyn.data(), measurement, shocks=["e", "u"], shock_sd={"e": 0.5, "u": 0.3}, estimate=["rho", "sd_e"],
                         priors={"rho": Normal(0.6, 0.2), "sd_e": InvGamma1(0.5, 2)})
    assert fit.stats["logpost"] == pytest.approx(ref.stats["logpost"], abs=1e-8)
    np.testing.assert_allclose(fit.params.values, ref.params.values, atol=1e-6)
    assert fit.measurement.errors["y"] == 0.4
    # and the log posterior at given values, with and without the error
    without = _write(tmp_path)
    assert without.measurement_sd == {}
    at = {"rho": 0.6, "sd_e": 0.5}
    assert dyn.log_posterior(at) == pytest.approx(fit_at(model, measurement, dyn.data()), abs=1e-8)
    assert dyn.log_posterior(at) != pytest.approx(without.log_posterior(at))


def test_variance_form_and_estimated_measurement_error(tmp_path):
    assert _write(tmp_path, shocks="var y = 0.16;").measurement_sd == {"y": pytest.approx(0.4)}
    dyn = _write(tmp_path, estimated="stderr y, inv_gamma_pdf, 0.3, 2;")
    assert dyn.estimated == ("rho", "sd_e", "sd_y") and "sd_y" in dyn.priors
    fit = dyn.estimate()
    model, measurement = _by_hand(dyn, {})
    ref = model.estimate(dyn.data(), measurement, shocks=["e", "u"], shock_sd={"e": 0.5, "u": 0.3}, estimate=["rho", "sd_e", "sd_y"],
                         priors={"rho": Normal(0.6, 0.2), "sd_e": InvGamma1(0.5, 2), "sd_y": InvGamma1(0.3, 2)})
    assert fit.stats["logpost"] == pytest.approx(ref.stats["logpost"], abs=1e-8)
    np.testing.assert_allclose(fit.params.values, ref.params.values, atol=1e-6)
    assert fit.measurement.errors["y"] == pytest.approx(fit.params.as_dict()["sd_y"])


def test_measurement_error_needs_varobs(tmp_path):
    with pytest.raises(ModelError, match=r"measurement errors \(stderr\) on \['x'\], which are not in varobs"):
        _write(tmp_path, shocks="var x; stderr 0.4;")
    with pytest.raises(ModelError, match="not in varobs"):
        _write(tmp_path, estimated="stderr x, inv_gamma_pdf, 0.3, 2;")
    with pytest.raises(ModelError, match="stderr must name a shock"):
        _write(tmp_path, estimated="stderr nothing, inv_gamma_pdf, 0.3, 2;")


def test_files_without_measurement_errors_are_read_as_before(tmp_path):
    dyn = _write(tmp_path)
    assert dyn.measurement_sd == {} and dyn.shock_sd == {"e": 0.5, "u": 0.3}
    assert isinstance(dyn.data(), Dataset) and list(pd.Index(dyn.estimated)) == ["rho", "sd_e"]
