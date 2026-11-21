"""Limited dependent variables, checked independently: Tobit and Heckman against Wooldridge's published Mroz
results; the negative binomial, zero-inflated Poisson, ordered and multinomial models against statsmodels' own
implementations (pymetrix writes its own likelihoods); marginal effects against finite differences done by hand
and, for the probit, the textbook formula."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pymetrix import ML, Dataset, Equation, Heckman  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402
from pymetrix.estimation import Multinomial, NegativeBinomial, OrderedLogit, OrderedProbit, Probit, Tobit, ZeroInflatedPoisson  # noqa: E402


@pytest.fixture(scope="module")
def mroz():
    return Dataset.example("mroz")  # bundled: Mroz (1987), as in Wooldridge


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(11)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.integers(0, 2, n).astype(float)
    mu = np.exp(0.4 + 0.5 * x1 - 0.3 * x2)
    nb = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * mu))  # NB2 with alpha 0.6
    zero = rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.8 * x2)))
    zip_y = np.where(zero, 0, rng.poisson(mu))
    latent = 0.8 * x1 - 0.5 * x2 + 0.4 * d + rng.logistic(size=n)
    ordered = np.digitize(latent, [-1.0, 0.5, 1.5]).astype(float)  # four outcomes, 0 to 3
    u = rng.gumbel(size=(n, 3))
    choice = np.argmax(np.column_stack([u[:, 0], 0.5 + 1.0 * x1 + u[:, 1], -0.3 - 0.7 * x2 + u[:, 2]]), axis=1).astype(float)
    frame = pd.DataFrame({"obs": range(n), "x1": x1, "x2": x2, "d": d, "nb": nb.astype(float), "zip": zip_y.astype(float),
                          "ord": ordered, "choice": choice})
    return Dataset.from_frame(frame, entity="obs")


def _fit(eq, data, method=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eq.fit(data, method=method or ML())


def _sm(model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=2000, **kw)


def test_tobit_matches_wooldridge(mroz):
    # Wooldridge, Introductory Econometrics, Table 17.2: hours of married women, censored at zero
    fit = _fit(Equation("hours ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6", family=Tobit(left=0)), mroz)
    p = fit.params.as_dict()
    expected = {"const": 965.31, "nwifeinc": -8.81, "educ": 80.65, "exper": 131.56, "expersq": -1.86, "age": -54.41,
                "kidslt6": -894.02, "kidsge6": -16.22, "sigma": 1122.02}
    for name, value in expected.items():
        assert p[name] == pytest.approx(value, abs=0.01 + 1e-5 * abs(value)), name
    assert fit.stats["loglike"] == pytest.approx(-3819.09, abs=0.01)
    se = dict(zip(fit.params.names, fit.params.se))
    assert se["educ"] == pytest.approx(21.58, abs=0.05) and se["kidslt6"] == pytest.approx(111.88, abs=0.1)  # Table 17.2's
    assert np.all(fit.fitted >= 0)  # E[y] of a variable censored at zero


def test_heckman_two_step_matches_wooldridge(mroz):
    # Wooldridge, Example 17.5: log wage, selected by labour-force participation
    fit = _fit(Equation("lwage ~ educ + exper + expersq"), mroz,
               Heckman(selection="inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"))
    p, se = fit.params.as_dict(), dict(zip(fit.params.names, fit.params.se))
    assert p["educ"] == pytest.approx(0.1091, abs=5e-4) and se["educ"] == pytest.approx(0.0156, abs=5e-4)
    assert p["exper"] == pytest.approx(0.0439, abs=5e-4) and p["expersq"] == pytest.approx(-0.00086, abs=2e-5)
    assert p["const"] == pytest.approx(-0.5781, abs=1e-3) and p["lambda"] == pytest.approx(0.0323, abs=1e-3)
    assert se["lambda"] == pytest.approx(0.1336, abs=2e-3)
    assert fit.stats["nobs_selected"] == 428 and fit.stats["nobs"] == 753
    ml = _fit(Equation("lwage ~ educ + exper + expersq"), mroz,
              Heckman(selection="inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6", estimator="ml"))
    assert ml.params["educ"] == pytest.approx(p["educ"], abs=0.01)  # close to the two-step estimate
    assert abs(ml.params["rho"]) < 0.5 and ml.stats["lambda"] == pytest.approx(ml.params["rho"] * ml.params["sigma"])


def test_heckman_ml_recovers_a_simulated_selection():
    rng = np.random.default_rng(5)
    n = 4000
    z, x = rng.normal(size=n), rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)  # selection and outcome errors, rho = 0.6
    s = (0.3 + 1.0 * z + 0.5 * x + e[:, 0] > 0).astype(float)
    y = np.where(s == 1, 1.0 + 2.0 * x + 1.5 * e[:, 1], np.nan)  # sigma 1.5
    data = Dataset.from_frame(pd.DataFrame({"i": range(n), "y": y, "s": s, "z": z, "x": x}), entity="i")
    fit = _fit(Equation("y ~ x"), data, Heckman(selection="s ~ z + x", estimator="ml"))
    assert fit.params["x"] == pytest.approx(2.0, abs=0.08) and fit.params["rho"] == pytest.approx(0.6, abs=0.08)
    assert fit.params["sigma"] == pytest.approx(1.5, abs=0.06)
    ols = Equation("y ~ x").fit(data)
    assert abs(ols.params["const"] - 1.0) > abs(fit.params["const"] - 1.0)  # least squares on the selected is biased


def test_negative_binomial_matches_statsmodels(sim):
    import statsmodels.api as sm

    fit = _fit(Equation("nb ~ x1 + x2", family=NegativeBinomial()), sim)
    f = sim.frame.reset_index(drop=True)
    ref = _sm(sm.NegativeBinomial(f["nb"], sm.add_constant(f[["x1", "x2"]]), loglike_method="nb2"))
    assert fit.params.values == pytest.approx(ref.params.to_numpy(), rel=1e-4)
    assert fit.params.se == pytest.approx(ref.bse.to_numpy(), rel=2e-3)
    assert fit.stats["loglike"] == pytest.approx(ref.llf, rel=1e-8)
    assert fit.params["alpha"] == pytest.approx(0.6, abs=0.1)


def test_zero_inflated_poisson_matches_statsmodels(sim):
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedPoisson as SMZIP

    fit = _fit(Equation("zip ~ x1 + x2", family=ZeroInflatedPoisson(inflate="x2")), sim)
    f = sim.frame.reset_index(drop=True)
    ref = _sm(SMZIP(f["zip"], sm.add_constant(f[["x1", "x2"]]), exog_infl=sm.add_constant(f[["x2"]]), inflation="logit"))
    ours = fit.params.as_dict()
    theirs = dict(zip(ref.params.index, ref.params.to_numpy()))
    assert ours["x1"] == pytest.approx(theirs["x1"], rel=1e-4) and ours["inflate: x2"] == pytest.approx(theirs["inflate_x2"], rel=1e-3)
    assert fit.stats["loglike"] == pytest.approx(ref.llf, rel=1e-8)


@pytest.mark.parametrize("family, distr", [(OrderedLogit(), "logit"), (OrderedProbit(), "probit")])
def test_ordered_models_match_statsmodels(sim, family, distr):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    fit = _fit(Equation("ord ~ x1 + x2 + d", family=family), sim)
    f = sim.frame.reset_index(drop=True)
    ref = _sm(OrderedModel(f["ord"], f[["x1", "x2", "d"]], distr=distr), method="bfgs")
    beta = fit.params.values[:3]
    assert beta == pytest.approx(ref.params.to_numpy()[:3], rel=1e-4)
    thresholds = ref.model.transform_threshold_params(ref.params.to_numpy())[1:-1]
    assert fit.params.values[3:] == pytest.approx(thresholds, rel=1e-4)  # the cut points
    assert fit.stats["loglike"] == pytest.approx(ref.llf, rel=1e-8)
    assert "const" not in fit.params.names and list(fit.params.names[3:]) == ["cut1", "cut2", "cut3"]


def test_multinomial_logit_matches_statsmodels(sim):
    import statsmodels.api as sm

    fit = _fit(Equation("choice ~ x1 + x2", family=Multinomial()), sim)
    f = sim.frame.reset_index(drop=True)
    ref = _sm(sm.MNLogit(f["choice"], sm.add_constant(f[["x1", "x2"]])))
    assert fit.params.values == pytest.approx(ref.params.to_numpy().T.ravel(), rel=1e-4)
    assert fit.params.se == pytest.approx(ref.bse.to_numpy().T.ravel(), rel=2e-3)
    assert fit.params.names[0] == "1: const"
    probs = fit.predict(sim)
    assert probs.shape == (1500, 3) and np.allclose(probs.sum(axis=1), 1.0)


def test_marginal_effects_by_hand_and_the_probit_formula(sim, mroz):
    probit = _fit(Equation("inlf ~ nwifeinc + educ + kidslt6", family=Probit()), mroz)
    me = probit.marginal_effects()
    from scipy.stats import norm

    X = probit.design.to_numpy(float)
    b = probit.params.values
    ame_educ = np.mean(norm.pdf(X @ b) * b[2])  # dP/dx = phi(x b) b: the textbook formula
    rows = dict(zip(me.index, (r[0] for r in me.rows)))
    assert rows["educ"] == pytest.approx(ame_educ, rel=1e-6)
    assert rows["kidslt6"] == pytest.approx(ame := np.mean(norm.pdf(X @ b) * b[3]), rel=1e-5) and ame < 0
    at_mean = probit.marginal_effects(at="mean")
    xbar = X.mean(axis=0)
    assert dict(zip(at_mean.index, (r[0] for r in at_mean.rows)))["educ"] == pytest.approx(norm.pdf(xbar @ b) * b[2], rel=1e-6)
    nb = _fit(Equation("nb ~ x1 + d", family=NegativeBinomial()), sim)
    effects = dict(zip(nb.marginal_effects().index, (r[0] for r in nb.marginal_effects().rows)))
    mu = np.exp(nb.design.to_numpy(float) @ nb.params.values[:3])
    assert effects["x1"] == pytest.approx(np.mean(mu) * nb.params["x1"], rel=1e-5)  # dE[y]/dx = mu b
    X1, X0 = nb.design.to_numpy(float).copy(), nb.design.to_numpy(float).copy()
    X1[:, 2], X0[:, 2] = 1.0, 0.0
    b_nb = nb.params.values[:3]
    assert effects["d"] == pytest.approx(np.mean(np.exp(X1 @ b_nb) - np.exp(X0 @ b_nb)), rel=1e-8)  # a 0/1 regressor: 0 to 1
    ordered = _fit(Equation("ord ~ x1 + x2", family=OrderedLogit()), sim)
    me_o = ordered.marginal_effects()
    by = dict(zip(me_o.index, (r[0] for r in me_o.rows)))
    assert sum(by[f"x1: P({j})"] for j in range(4)) == pytest.approx(0.0, abs=1e-8)  # probabilities still add to one
    with pytest.raises(SpecificationError, match="linear model"):
        Equation("nb ~ x1").fit(sim).marginal_effects()


def test_refusals_and_prediction(sim, mroz):
    with pytest.raises(SpecificationError, match="non-negative count"):
        _fit(Equation("x1 ~ x2", family=NegativeBinomial()), sim)
    with pytest.raises(SpecificationError, match="outside the limits"):
        _fit(Equation("x1 ~ x2", family=Tobit(left=0)), sim)
    with pytest.raises(SpecificationError, match="selection equation"):
        _fit(Equation("lwage ~ educ"), mroz, Heckman())
    tobit = _fit(Equation("hours ~ educ + kidslt6", family=Tobit(left=0)), mroz)
    assert tobit.predict(mroz).to_numpy() == pytest.approx(tobit.fitted.to_numpy(), rel=1e-10)
    assert (tobit.predict(mroz, kind="index") < 0).any()  # the latent index goes below the limit
    zipfit = _fit(Equation("zip ~ x1", family=ZeroInflatedPoisson()), sim)
    assert zipfit.params.names[-1] == "inflate: const"


def test_the_dialog(mroz):
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["mroz"] = mroz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.run(FitEquation("hours ~ educ + kidslt6", data="mroz", method="ML", family="tobit", censor_left="0", name="tob"))
        s.run(FitEquation("lwage ~ educ + exper", data="mroz", method="Heckman", method_settings="selection='inlf ~ educ + kidslt6'",
                          name="heck"))
    assert "family=Tobit()" in s.history[0].code  # censored at zero, the default
    assert "Heckman(selection='inlf ~ educ + kidslt6')" in s.history[1].code
    assert s.objects["heck"].stats["nobs_selected"] == 428


def test_the_window_shows_marginal_effects_and_a_fit_saves(mroz, tmp_path):
    from pymetrix import load_result, save_result
    from pymetrix.views import sections

    tobit = _fit(Equation("hours ~ educ + kidslt6", family=Tobit(left=0)), mroz)
    titles = [title for title, _ in sections(tobit)]
    assert "Marginal effects" in titles and "Summary" in titles
    back = load_result(save_result(tobit, tmp_path / "tobit"))  # a saved result keeps its views
    saved = back.to_frame("Marginal effects")
    assert saved["effect"].to_numpy() == pytest.approx(tobit.marginal_effects().to_frame()["effect"].to_numpy(), rel=1e-10)
