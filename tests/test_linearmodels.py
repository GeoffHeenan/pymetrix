"""linearmodels-backed IV and panel estimators, the Hausman test and the method registry."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402

lm_iv = pytest.importorskip("linearmodels.iv")
lm_panel = pytest.importorskip("linearmodels.panel")

from pymetrix import (  # noqa: E402
    IV,
    IVGMM,
    LIML,
    Between,
    Cluster,
    Dataset,
    Equation,
    FamaMacBeth,
    FirstDifference,
    RandomEffects,
    compare,
)
from pymetrix.commands import FitEquation, Session  # noqa: E402
from pymetrix.estimation import available_methods  # noqa: E402
from pymetrix.estimation.registry import method_settings  # noqa: E402


@pytest.fixture(scope="module")
def ivdata():
    rng = np.random.default_rng(3)
    n = 1500
    z1, z2, w = rng.normal(size=(3, n))
    u = rng.normal(size=n)
    x = 0.6 * z1 + 0.4 * z2 + 0.3 * w + 0.8 * u + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"id": range(n), "y": 1 + 2 * x + 0.5 * w + u, "x": x, "w": w, "z1": z1, "z2": z2})
    return df, Dataset.from_frame(df, entity="id")


@pytest.fixture(scope="module")
def paneldata():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(30):
        a = rng.normal(scale=2)
        for t in range(2000, 2015):
            x = rng.normal() + 0.3 * a  # correlated with the entity effect
            rows.append({"country": f"C{i:02d}", "year": t, "x": x, "w": rng.normal(), "y": a + 0.7 * x + rng.normal(scale=0.5)})
    df = pd.DataFrame(rows)
    pdf = df.assign(year=pd.to_datetime(df.year.astype(str))).set_index(["country", "year"])
    return pdf, Dataset.from_frame(df, entity="country", time="year", freq="Y")


def test_liml_and_gmm_match_linearmodels(ivdata):
    df, data = ivdata
    exog = df[["w"]].assign(const=1.0)[["const", "w"]]
    names = ["const", "x", "w"]
    ours = Equation("y ~ x + w").fit(data, method=LIML(["x"], ["z1", "z2"], fuller=1))
    ref = lm_iv.IVLIML(df.y, exog, df[["x"]], df[["z1", "z2"]], fuller=1).fit(cov_type="unadjusted", debiased=True)
    assert np.allclose(ours.params.values, ref.params[names]) and np.allclose(ours.params.se, ref.std_errors[names])
    gmm = Equation("y ~ x + w").fit(data, method=IVGMM(["x"], ["z1", "z2"]))
    ref_g = lm_iv.IVGMM(df.y, exog, df[["x"]], df[["z1", "z2"]]).fit(cov_type="robust", debiased=True)
    assert np.allclose(gmm.params.values, ref_g.params[names]) and np.isclose(gmm.stats["hansen_j"], ref_g.j_stat.stat)
    j = gmm.test("hansen_j")
    assert j.df == 1 and not j.reject()
    assert "First-stage F (x)" in str(gmm.summary()) and "[linearmodels" in str(gmm.summary())


def test_k_class_with_kappa_one_is_our_2sls(ivdata):
    _, data = ivdata
    k1 = Equation("y ~ x + w").fit(data, method=LIML(["x"], ["z1", "z2"], kappa=1.0))
    tsls = Equation("y ~ x + w").fit(data, method=IV(["x"], ["z1", "z2"]))
    assert np.allclose(k1.params.values, tsls.params.values) and np.allclose(k1.params.se, tsls.params.se)


@pytest.mark.parametrize("method, cls, drop_const", [
    (RandomEffects(), "RandomEffects", False),
    (Between(), "BetweenOLS", False),
    (FirstDifference(), "FirstDifferenceOLS", True),
    (FamaMacBeth(), "FamaMacBeth", False),
])
def test_panel_estimators_match_linearmodels(paneldata, method, cls, drop_const):
    pdf, data = paneldata
    X = pdf[["x", "w"]].assign(const=1.0)[["const", "x", "w"]]
    if drop_const:
        X = X[["x", "w"]]
    ref = getattr(lm_panel, cls)(pdf.y, X).fit(debiased=True)
    ours = Equation("y ~ x + w").fit(data, method=method)
    names = list(ours.params.names)
    assert np.allclose(ours.params.values, ref.params[names]) and np.allclose(ours.params.se, ref.std_errors[names])


def test_between_is_ols_on_entity_means_and_clustering(paneldata):
    pdf, data = paneldata
    means = pdf.groupby(level=0)[["y", "x", "w"]].mean()
    between = Equation("y ~ x + w").fit(data, method=Between())
    assert np.allclose(between.params.values, sm.OLS(means.y, sm.add_constant(means[["x", "w"]])).fit().params)
    X = pdf[["x", "w"]].assign(const=1.0)[["const", "x", "w"]]
    clustered = Equation("y ~ x + w").fit(data, method=RandomEffects(cov=Cluster("country")))
    ref = lm_panel.RandomEffects(pdf.y, X).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    assert np.allclose(clustered.params.se, ref.std_errors)


def test_hausman(paneldata):
    _, data = paneldata
    fe = Equation("y ~ x + w + fe(country)").fit(data)
    re = Equation("y ~ x + w").fit(data, method=RandomEffects())
    h = compare(fe, re, test="hausman", sigma="separate")  # the textbook formula
    d = fe.params.values - re.params.values[1:]
    V = fe.params.cov - re.params.cov[1:, 1:]
    assert np.isclose(h.statistic, d @ np.linalg.solve(V, d)) and h.df == 2
    scaled = compare(fe, re, test="hausman")  # Stata's sigmamore: the efficient fit's error variance for both
    Vs = fe.params.cov * (re.stats["se_regression"] / fe.stats["se_regression"]) ** 2 - re.params.cov[1:, 1:]
    assert np.isclose(scaled.statistic, d @ np.linalg.solve(Vs, d))
    # short panels with effects strongly correlated with x: random effects is inconsistent
    rng = np.random.default_rng(7)
    rows = []
    for i in range(200):
        a = rng.normal(scale=2)
        for t in range(2010, 2015):
            x = rng.normal() + a
            rows.append({"id": i, "year": t, "x": x, "w": rng.normal(), "y": a + 0.7 * x + rng.normal(scale=0.5)})
    short = Dataset.from_frame(pd.DataFrame(rows), entity="id", time="year", freq="Y")
    h2 = compare(Equation("y ~ x + w + fe(id)").fit(short), Equation("y ~ x + w").fit(short, method=RandomEffects()), test="hausman")
    assert h2.reject()


def test_method_registry_and_dialog_settings(ivdata):
    labels = available_methods()
    assert {"OLS", "IV", "ML", "LIML", "IV-GMM", "Random effects", "Fama-MacBeth"} <= set(labels)
    assert [s.name for s in method_settings("LIML")] == ["fuller", "kappa", "debiased", "backend_options"]
    assert all(s.advanced for s in method_settings("ML"))
    _, data = ivdata
    session = Session()
    session.objects["data"] = data
    fit = session.run(FitEquation("y ~ x + w", method="LIML", endog="x", instruments="z1, z2", method_settings="fuller=1.0", name="liml"))
    session.run(FitEquation("y ~ x + w", cov="cluster", cluster="id", name="ols"))
    script = session.script()
    assert "method=LIML(endog=['x'], instruments=['z1', 'z2'], fuller=1.0)" in script
    assert "method=OLS(cov=Cluster('id'))" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["liml"].params.values, fit.params.values)


def test_method_settings_sub_form(ivdata):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    _, data = ivdata
    session = Session()
    session.objects["data"] = data
    dialog = command_dialog(FitEquation, session)
    dialog.editors["method"].widget.setCurrentText("IV-GMM")
    settings = dialog.editors["method_settings"]
    assert list(settings.sub) == ["weight_type", "iter_limit", "tol", "debiased", "backend_options"]
    settings.sub["iter_limit"].widget.setValue(50)
    assert settings.get() == "iter_limit=50"
    settings.sub["backend_options"].widget.setText("fit.iter_limit=5")  # typed as text, like other mappings
    from pymetrix.forms import build
    from pymetrix.plugins import registry

    assert build(registry("methods").get("IV-GMM"), settings.get(), endog=["x"], instruments=["z1"]).backend_options == {"fit.iter_limit": 5}
    settings.sub["backend_options"].widget.setText("")
    assert isinstance(settings.sub["tol"].widget, QtWidgets.QLineEdit)  # small tolerances are typed, not spun
    dialog.editors["method"].widget.setCurrentText("ML")
    assert list(dialog.editors["method_settings"].sub) == ["optimizer", "max_iter", "tol", "start"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dialog.row_visible("family")
