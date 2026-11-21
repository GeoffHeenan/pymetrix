"""The settings the dialogs' Advanced tabs gained: each reaches its API call, and runs or replays as that call would."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import Session
from pymetrix.commands.arima import FitSARIMA
from pymetrix.commands.builtin import (CheckIdentification, DefineEquation, EstimateModel, FitCointegratingRegression, FitEquation,
                                       FitTVPVAR, Forecast, IdentifyStructuralVAR, ImpulseResponse, LoadTemplate, ProxySVARCommand,
                                       RunMCMC, VarianceDecomposition)
from pymetrix.commands.contributions import DataContributions
from pymetrix.commands.disaggregation import Disaggregate
from pymetrix.commands.figures import FigureOptions
from pymetrix.commands.forecasting import CompareManyForecasts
from pymetrix.commands.nonparametric import FitKernelDensity
from pymetrix.commands.occbin import OccasionallyBinding
from pymetrix.commands.panel import PanelStatistics
from pymetrix.commands.projections import FitGrowthAtRisk, FitLocalProjection
from pymetrix.commands.report import ExportReport
from pymetrix.commands.samples import CheckData, NewSample
from pymetrix.commands.statespace import FitStateSpace
from pymetrix.commands.treatment import FitEventStudy, FitStaggeredDiD, FitSyntheticDiD, PropensityWeighting
from pymetrix.errors import SpecificationError


def _session(**objects):
    session = Session()
    session.objects.update(objects)
    return session


@pytest.mark.parametrize("cmd, text", [
    (FitCointegratingRegression("y", "x", trend="ctt", bandwidth=4.0), "trend='ctt', bandwidth=4.0"),
    (FitCointegratingRegression("y", "x", method="dols", leads=2, lags=1), "leads=2, lags=1"),
    (FitTVPVAR("a, b", kq=0.02, thin=5), "kq=0.02, thin=5"),
    (ProxySVARCommand("v", "z", block=4, alpha=0.1, seed=3), "block=4, alpha=0.1, seed=3"),
    (FitKernelDensity("y", points=512, cut=1.0), "points=512, cut=1.0"),
    (OccasionallyBinding("m", "e[1]=1", horizon=200, max_iter=50), "horizon=200, max_iter=50"),
    (FitGrowthAtRisk("g", "x", quantiles="0.1, 0.5, 0.9"), "quantiles=[0.1, 0.5, 0.9]"),
    (CheckData(strict=True, outlier=4.0), "strict=True, outlier=4.0"),
    (FitEventStudy("y", "d", reference=-2), "reference=-2"),
    (FitStaggeredDiD("y", "d", first=-3, last=4, seed=7), "window=(-3, 4), seed=7"),
    (PropensityWeighting("y", "d", "x", seed=5), "seed=5"),
    (FitSyntheticDiD("y", "d", se="placebo", placebos=50, seed=2), "se='placebo', placebos=50, seed=2"),
    (FitSARIMA("y", backend_options="fit.maxiter=80, model.enforce_stationarity=False"),
     "backend_options={'fit.maxiter': 80, 'model.enforce_stationarity': False}"),
    (FitLocalProjection("y", "s", alpha=0.32), "alpha=0.32"),
    (FitLocalProjection("y", "s", smooth=True, penalty=10.0, order=1, folds=4), "penalty=10.0, order=1, folds=4"),
    (IdentifyStructuralVAR(A="1 0; * 1", starts=3, seed=9), "starts=3, seed=9"),
    (LoadTemplate(shock_sd="ey=0.5"), "shock_sd={'ey': 0.5}"),
    (NewSample(windows="DEU=1991Q1:; *=:2019Q4"), "windows={'DEU': ('1991Q1', None), '*': (None, '2019Q4')}"),
    (DefineEquation("x ~ rho*x[-1]", params="rho", label="AR", starts="rho=0.5", bounds="rho=(0, 1)"),
     "params=[Param(name='rho', bounds=[0.0, 1.0], start=0.5)], name='AR'"),
])
def test_each_setting_reaches_its_call(cmd, text):
    assert text in cmd.to_call().to_code()


def test_defaults_add_nothing_to_the_code():
    for cmd in (FitCointegratingRegression("y", "x"), FitTVPVAR("a, b"), FitKernelDensity("y"), CheckData(), FitSyntheticDiD("y", "d"),
                FitStaggeredDiD("y", "d"), FitGrowthAtRisk("g", "x"), FitSARIMA("y"), FitLocalProjection("y", "s")):
        code = cmd.to_call().to_code()
        for word in ("bandwidth", "kq", "points", "strict", "placebos", "window", "quantiles", "backend_options", "alpha"):
            assert f"{word}=" not in code, (cmd, code)


def test_settings_change_the_results(ts):
    session = _session(data=ts)
    wide = session.run(FitKernelDensity("y", points=100, cut=1.0, name="a"))
    assert len(wide.grid) == 100 and wide.grid.min() > ts.frame["y"].min() - 2 * wide.bandwidth
    fit = session.run(FitEquation("y ~ x", where="x > 0"))
    assert fit.stats["nobs"] == int((ts.frame["x"] > 0).sum())
    assert "where='x > 0'" in session.script()
    assert len(session.run(CheckData(outlier=1.5, name="c")).findings) > len(session.run(CheckData(name="d")).findings)


def test_validation_of_the_new_text_settings():
    assert FitStaggeredDiD("y", "d", first=-3).validate(_session())
    with pytest.raises(SpecificationError, match="quantiles"):
        FitGrowthAtRisk("g", "x", quantiles="0.1, low").to_call()
    with pytest.raises(SpecificationError, match="name=value"):
        FitSARIMA("y", backend_options="80").to_call()
    with pytest.raises(SpecificationError, match="rho's range"):
        Disaggregate("gdp ~ ip", "annual", rho_bounds="0.5, 0.1")._spec()
    with pytest.raises(SpecificationError, match="entity=first:last"):
        NewSample(windows="DEU 1991Q1").to_call()
    with pytest.raises(SpecificationError, match="parameters named under Params"):
        DefineEquation("x ~ rho*x[-1]", params="rho", starts="beta=1").to_call()
    assert FitLocalProjection("y", "s", smooth=True, state="rec").validate(_session(data=None))
    assert EstimateModel("m", bounds="kappa=5").validate(_session())
    assert CheckIdentification("m", at="kappa").validate(_session())


def test_disaggregation_with_rho_fixed():
    rng = np.random.default_rng(0)
    q = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    ip = np.cumsum(rng.normal(1, 1, 80)) + 50
    high = Dataset.from_frame(pd.DataFrame({"ip": ip}, index=q))
    a = pd.period_range("2000", periods=20, freq="Y", name="time")
    low = Dataset.from_frame(pd.DataFrame({"gdp": ip.reshape(20, 4).sum(1) * 2 + rng.normal(0, 1, 20)}, index=a))
    session = _session(annual=low, data=high)
    fit = session.run(Disaggregate("gdp ~ ip", "annual", rho=0.5, add_to_high=False))
    assert fit.rho == 0.5
    free = session.run(Disaggregate("gdp ~ ip", "annual", rho_bounds="-0.9, 0.9", add_to_high=False, name="free"))
    assert -0.9 <= free.rho <= 0.9
    np.testing.assert_allclose(fit.series.to_numpy().reshape(20, 4).sum(1), low.frame["gdp"].to_numpy())


def test_sign_restriction_settings_reach_the_rotations(ts):
    session = _session(data=ts)
    from pymetrix.commands.builtin import FitVAR

    session.run(FitVAR("y, x", lags="1", name="var"))
    signs = "s1: y+, x+; s2: y+, x-"
    irf = ImpulseResponse("var", periods=4, scheme="SignRestrictions", signs=signs, draws=0, rotations=50, sign_seed=4)
    code = irf.to_code()
    assert "rotations=50" in code and "seed=4" in code
    a = session.run(irf)
    b = session.run(ImpulseResponse("var", periods=4, scheme="SignRestrictions", signs=signs, draws=0, rotations=50, sign_seed=4, name="b"))
    c = session.run(ImpulseResponse("var", periods=4, scheme="SignRestrictions", signs=signs, draws=0, rotations=50, sign_seed=5, name="c"))
    assert np.allclose(a.values, b.values) and not np.allclose(a.values, c.values)
    fevd = VarianceDecomposition("var", scheme="SignRestrictions", signs=signs, rotations=50, narrative_signs="s1: 2010Q1+")
    assert "narrative_signs={'s1': {'2010Q1': '+'}}" in fevd.to_code()
    assert "narrative_major={'s1': {'2010Q1': 'x'}}" in VarianceDecomposition(
        "var", scheme="SignRestrictions", signs=signs, narrative_major="s1: 2010Q1 x").to_code()


def test_forecast_draws_are_refused_where_nothing_is_simulated(ts):
    session = _session(data=ts)
    session.run(FitEquation("y ~ x"))
    problems = Forecast("fit", "2020Q1", "2020Q4", draws=100).validate(session)
    assert problems and "take no draws" in problems[0]
    assert not Forecast("fit", "2020Q1", "2020Q4").validate(session)
    assert "draws=100, seed=3" in Forecast("fit", "2020Q1", "2020Q4", draws=100, seed=3).to_code()


def test_model_settings_reach_their_calls():
    cmd = EstimateModel("nk", starts=4, seed=11, workers=2, initial="diffuse", presample=4, max_iter=500, bounds="kappa=(0.01, 1)", growth=True)
    advanced = cmd._advanced()
    assert advanced == {"starts": 4, "seed": 11, "workers": 2, "start": "diffuse", "presample": 4, "max_iter": 500, "growth": True,
                        "bounds": {"kappa": (0.01, 1.0)}}
    assert EstimateModel("nk")._advanced() == {}
    assert "seed=1" in EstimateModel("nk", starts=2).to_code()  # as before: several starts are seeded
    assert RunMCMC("fit", scale=0.3, adapt=False, jump=0.0).to_code().endswith("scale=0.3, adapt=False, jump=0.0)")
    assert CheckIdentification("fit", at="kappa=0.2").to_code() == "identification = fit.identification(lags=4, at={'kappa': 0.2})"
    assert DataContributions("fit", data="newer").to_code() == "contributions = fit.contributions(data=newer)"


def test_template_shock_sizes_and_report_settings(tmp_path, ts):
    session = _session(data=ts)
    template = session.run(LoadTemplate(shock_sd="ey=0.25"))
    assert template.shock_sd["ey"] == 0.25
    session.run(FitEquation("y ~ x"))
    path = tmp_path / "r.md"
    session.run(ExportReport("fit", str(path), date="2001-02-03", figure_width=4.0))
    assert "2001-02-03" in path.read_text()
    assert "date='2001-02-03', figure_size=(4.0, None)" in ExportReport("fit", "r.md", date="2001-02-03", figure_width=4.0).to_code()
    assert "fit" in session.of_type("Reportable") and "data" in session.of_type("Reportable")


def test_figure_layout_options(ts):
    from pymetrix.commands.figures import PlotSeries

    session = _session(data=ts)
    session.run(PlotSeries("y, x", name="fig"))
    figure = session.run(FigureOptions("fig", start="2005Q1", theme="publication", columns=2, sharey=True, name="fig2"))
    assert (figure.span, figure.theme, figure.columns, figure.sharey) == (("2005Q1", None), "publication", 2, True)
    again = FigureOptions.for_figure("fig2", figure, is_figure=True)
    assert (again.start, again.theme, again.columns, again.sharey) == ("2005Q1", "publication", 2, True)
    assert "span=('2005Q1', None)" in session.script()


def test_panel_axis_and_compare_settings(panel):
    assert "by='id'" in PanelStatistics(show="variation", by="id").to_call().to_code()
    assert "by=" not in PanelStatistics(by="id").to_call().to_code()  # coverage takes no axis
    code = CompareManyForecasts("a, b", "y", reps=200, block=4, seed=1).to_code()
    assert code.endswith("0.1, 1, reps=200, block=4, seed=1)")


def test_state_space_settings():
    spec = ("trend", "trend = trend[-1] + e", "y = trend + u", "e = sd_e, u = sd_u")
    code = FitStateSpace(*spec, params="sd_e, sd_u", cov="none", max_iter=300).to_call().to_code()
    assert "cov='none', max_iter=300" in code
    assert "max_iter" not in FitStateSpace(*spec, params="sd_e, sd_u").to_call().to_code()
