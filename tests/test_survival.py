"""Survival analysis: Kaplan-Meier and the log-rank test against statsmodels called directly, a hand calculation on
six subjects, and the published results for Freireich's leukaemia trial (Kleinbaum and Klein, Survival Analysis: A
Self-Learning Text, 3rd ed., chapters 2 and 3); Cox's model against statsmodels' PHReg, the same published results,
and lifelines 0.30.3's recorded estimates, robust errors and proportional-hazards tests."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SpecificationError, cox_ph, kaplan_meier

# Freireich et al. (1963): weeks in remission, 6-MP (censored marked 0) and placebo (every one relapsed)
MP = [(6, 1), (6, 1), (6, 1), (6, 0), (7, 1), (9, 0), (10, 1), (10, 0), (11, 0), (13, 1), (16, 1), (17, 0), (19, 0), (20, 0), (22, 1), (23, 1),
      (25, 0), (32, 0), (32, 0), (34, 0), (35, 0)]
PLACEBO = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]


@pytest.fixture(scope="module")
def trial():
    rows = [{"weeks": t, "relapse": c, "placebo": 0} for t, c in MP] + [{"weeks": t, "relapse": 1, "placebo": 1} for t in PLACEBO]
    frame = pd.DataFrame(rows)
    frame["patient"] = np.arange(len(frame))
    return Dataset.from_frame(frame, entity=["patient"])


def _spells(n=300, seed=11):
    """Continuous durations (no ties): Weibull hazards, proportional in age, treated and score, with censoring."""
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 1, n)
    treated = (rng.random(n) < 0.5).astype(float)
    score = rng.uniform(-1, 1, n)
    scale = np.exp(-(0.5 * age - 0.7 * treated + 0.3 * score) / 1.5)
    t = scale * rng.weibull(1.5, n) * 10
    c = rng.uniform(2, 25, n)
    return pd.DataFrame({"id": np.arange(n), "duration": np.round(np.minimum(t, c), 6), "event": (t <= c).astype(float),
                         "age": age, "treated": treated, "score": score})


def test_kaplan_meier_by_hand():
    """Six subjects: events at 2, 3 (two), 5 and 8; censored at 3 and 6. At risk 6, 5, 2, 1."""
    frame = pd.DataFrame({"id": range(7), "t": [2, 3, 3, 3, 5, 6, 8], "d": [1, 1, 1, 0, 1, 0, 1]})
    km = kaplan_meier(Dataset.from_frame(frame, entity=["id"]), "t", "d")
    table = km.survival_table("all")
    S = np.cumprod([1 - 1 / 7, 1 - 2 / 6, 1 - 1 / 3, 1 - 1 / 1])
    assert np.allclose(table["survival"], S, atol=1e-15) and list(table["at risk"]) == [7, 6, 3, 1]
    greenwood = np.sqrt(S[:3] ** 2 * np.cumsum([1 / (7 * 6), 2 / (6 * 4), 1 / (3 * 2)]))
    assert np.allclose(table["std err"].to_numpy()[:3], greenwood, atol=1e-15)
    # log-log interval (Kalbfleisch-Prentice) by its formula
    z = 1.959963984540054
    w = z * greenwood / (S[:3] * np.abs(np.log(S[:3])))
    assert np.allclose(table["lower"].to_numpy()[:3], S[:3] ** np.exp(w)) and np.allclose(table["upper"].to_numpy()[:3], S[:3] ** np.exp(-w))
    assert km.survival_at([0, 2.5, 5, 7]).to_numpy()[:, 0].tolist() == pytest.approx([1.0, S[0], S[2], S[2]])
    assert km.groups.loc["all", "median"] == 5.0


def test_freireich_published_and_statsmodels(trial):
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff

    km = kaplan_meier(trial, "weeks", "relapse", group="placebo")
    mp = km.survival_table(0)
    # Kleinbaum and Klein, chapter 2: the 6-MP group's estimates, and the log-rank statistic 16.793
    published = {6: 0.8571, 7: 0.8067, 10: 0.7529, 13: 0.6902, 16: 0.6275, 22: 0.5378, 23: 0.4482}
    assert np.allclose(mp["survival"].to_numpy(), list(published.values()), atol=5e-5) and list(mp.index) == list(published)
    assert km.test.statistic == pytest.approx(16.793, abs=5e-4) and km.test.df == 1
    f = trial.frame
    ref = SurvfuncRight(f["weeks"][f["placebo"] == 0], f["relapse"][f["placebo"] == 0])
    assert np.allclose(mp["survival"], ref.surv_prob, atol=1e-15) and np.allclose(mp["std err"], ref.surv_prob_se, atol=1e-15)
    stat, p = survdiff(f["weeks"], f["relapse"], f["placebo"])
    assert km.test.statistic == pytest.approx(stat, abs=1e-12) and km.test.p_value == pytest.approx(p, abs=1e-15)
    # lifelines 0.30.3, KaplanMeierFitter's exponential Greenwood (log-log) interval for the 6-MP group
    lifelines_ci = [[0.6197179552659838, 0.9515517476469969], [0.5631465646222333, 0.9228090192028802], [0.5031995107999564, 0.8893618351692651],
                    [0.4316102224861839, 0.8490659633494505], [0.36751085598750194, 0.8049121894718411], [0.26777893676713493, 0.7467907176353943],
                    [0.18805200595923233, 0.6801426284952412]]
    assert np.allclose(mp[["lower", "upper"]].to_numpy(), lifelines_ci, atol=1e-10)
    for test, weight in (("wilcoxon", "gb"), ("tarone-ware", "tw")):
        other = kaplan_meier(trial, "weeks", "relapse", group="placebo", test=test)
        assert other.test.statistic == pytest.approx(survdiff(f["weeks"], f["relapse"], f["placebo"], weight_type=weight)[0], abs=1e-12)
    assert km.groups.loc[1, "median"] == 8.0 and km.groups.loc[0, "median"] == 23.0 and np.isnan(km.groups.loc[0, "median upper"])


def test_cox_published_and_statsmodels(trial):
    from statsmodels.duration.hazard_regression import PHReg

    f = trial.frame
    # Kleinbaum and Klein, chapter 3: Breslow 1.509 (0.410), hazard ratio 4.523; Efron (R's coxph) 1.572 (0.412)
    for ties, (b, se, hr) in {"breslow": (1.509, 0.410, 4.523), "efron": (1.572, 0.412, 4.817)}.items():
        fit = cox_ph(trial, "weeks", "relapse", ["placebo"], ties=ties)
        assert fit.values[0] == pytest.approx(b, abs=5e-4) and fit.se[0] == pytest.approx(se, abs=5e-4)
        assert fit.hazard_ratios()["hazard ratio"].iloc[0] == pytest.approx(hr, abs=5e-4)
        ref = PHReg(f["weeks"].to_numpy(), f[["placebo"]].to_numpy(), status=f["relapse"].to_numpy(), ties=ties).fit()
        assert np.allclose(fit.values, ref.params, atol=1e-12) and np.allclose(fit.se, ref.bse, atol=1e-12)
        assert fit.loglike == pytest.approx(ref.llf, abs=1e-10)
        H = ref.baseline_cumulative_hazard[0]
        assert np.allclose(fit.baseline["cumulative hazard"].to_numpy(), H[1]) and np.allclose(fit.baseline["survival"].to_numpy(), H[2])


LIFELINES = {  # CoxPHFitter (Efron) on _spells(): age, treated, score; then the PH tests by transform
    "params": [0.427466250022, -0.678508590282, 0.263275783041],
    "se": [0.082957013764, 0.156219003395, 0.142239245646],
    "loglike": -848.0725185039701,
    "robust": [0.082226040284, 0.154698562339, 0.150489828075],
    "km": [0.144654830536, 2.703157068698, 1.542767813299],
    "rank": [0.143497319616, 2.494802335233, 1.636310666861],
    "identity": [0.129005294987, 3.173042027024, 1.47920952181],
    "log": [0.32562105013, 3.332479207227, 2.925001590191],
}


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_against_lifelines(ties):
    """No tied event times, so Breslow's and Efron's partial likelihoods are the same (lifelines has only Efron's)."""
    data = Dataset.from_frame(_spells(), entity=["id"])
    fit = cox_ph(data, "duration", "event", ["age", "treated", "score"], ties=ties)
    assert np.allclose(fit.values, LIFELINES["params"], atol=1e-7) and np.allclose(fit.se, LIFELINES["se"], rtol=1e-6)
    assert fit.loglike == pytest.approx(LIFELINES["loglike"], abs=1e-7)
    for transform in ("km", "rank", "identity", "log"):
        stats = fit.ph_test(transform).to_frame()["chi-squared"].to_numpy()
        assert np.allclose(stats[:3], LIFELINES[transform], rtol=1e-6), transform
    robust = cox_ph(data, "duration", "event", ["age", "treated", "score"], ties=ties, cov="robust")
    assert np.allclose(robust.values, fit.values) and np.allclose(robust.se, LIFELINES["robust"], rtol=1e-6)


def test_global_ph_test_by_hand():
    """The global statistic of Grambsch and Therneau (as R's cox.zph computed it before version 3), coded with loops."""
    data = Dataset.from_frame(_spells(), entity=["id"])
    fit = cox_ph(data, "duration", "event", ["age", "treated", "score"])
    table = fit.ph_test("rank").to_frame()
    r = fit.schoenfeld.sort_values("time")[["age", "treated", "score"]].to_numpy()
    d = len(r)
    g = np.arange(1.0, d + 1) - (d + 1) / 2
    u = np.zeros(3)
    for k in range(d):
        u += g[k] * r[k]
    V = fit.naive_cov
    stat = d * (u @ V @ u) / sum(x * x for x in g)
    assert table.loc["global", "chi-squared"] == pytest.approx(stat, rel=1e-12) and table.loc["global", "df"] == 3


def test_entry_strata_and_clusters_against_statsmodels():
    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff

    f = _spells()
    rng = np.random.default_rng(3)
    f["entry"] = np.minimum(rng.uniform(0, 3, len(f)), f["duration"] * 0.9)
    f["site"] = rng.integers(0, 3, len(f))
    f["firm"] = rng.integers(0, 40, len(f))
    data = Dataset.from_frame(f, entity=["id"])
    km = kaplan_meier(data, "duration", "event", group="treated", entry="entry")
    ref = SurvfuncRight(f["duration"][f["treated"] == 1], f["event"][f["treated"] == 1], entry=f["entry"][f["treated"] == 1])
    assert np.allclose(km.survival_table(1.0)["survival"], ref.surv_prob, atol=1e-15)
    stat, _ = survdiff(f["duration"], f["event"], f["treated"], entry=f["entry"])
    assert km.test.statistic == pytest.approx(stat, abs=1e-12)
    stratified = kaplan_meier(data, "duration", "event", group="treated", strata="site")
    stat, _ = survdiff(f["duration"], f["event"], f["treated"], strata=f["site"])
    assert stratified.test.statistic == pytest.approx(stat, abs=1e-12) and "stratified" in stratified.test.name
    with pytest.raises(SpecificationError, match="strata or entry"):  # statsmodels 0.15's survdiff mixes them up
        kaplan_meier(data, "duration", "event", group="treated", strata="site", entry="entry")
    fit = cox_ph(data, "duration", "event", ["age", "treated"], entry="entry", strata="site", ties="efron", cov="cluster", cluster="firm")
    model = PHReg(f["duration"].to_numpy(), f[["age", "treated"]].to_numpy(), status=f["event"].to_numpy(), entry=f["entry"].to_numpy(),
                  strata=pd.factorize(f["site"])[0], ties="efron")
    res = model.fit(groups=pd.factorize(f["firm"])[0])
    assert np.allclose(fit.values, res.params, atol=1e-12) and np.allclose(fit.se, res.bse, atol=1e-12)
    assert fit.baseline.index.get_level_values("stratum").nunique() == 3


def test_errors(trial):
    with pytest.raises(SpecificationError, match="1 \\(the event happened\\) or 0"):
        kaplan_meier(trial, "weeks", "weeks")
    with pytest.raises(SpecificationError, match="95% only"):
        kaplan_meier(trial, "weeks", "relapse", bands=True, alpha=0.1)
    with pytest.raises(SpecificationError, match="needs cluster"):
        cox_ph(trial, "weeks", "relapse", ["placebo"], cov="cluster")
    with pytest.raises(SpecificationError, match="constant"):
        cox_ph(trial.with_series(one="placebo * 0 + 1"), "weeks", "relapse", ["placebo", "one"])
    bands = kaplan_meier(trial, "weeks", "relapse", group="placebo", bands=True)
    both = bands.table[["band lower", "lower"]].dropna()
    assert len(both) > 15 and (both["band lower"] <= both["lower"] + 1e-12).all()  # simultaneous bands are wider


def test_the_dialogs_record_the_calls_and_the_script_replays(trial):
    from pymetrix.commands import COMMANDS, Session

    s = Session()
    s.objects["trial"] = trial
    s.run(COMMANDS["FitKaplanMeier"].cls("weeks", "relapse", data="trial", group="placebo"))
    assert s.history[-1].code == "km = kaplan_meier(trial, 'weeks', 'relapse', group='placebo')"
    s.run(COMMANDS["FitCoxPH"].cls("weeks", "relapse", "placebo", data="trial", ties="efron"))
    assert s.history[-1].code == "cox = cox_ph(trial, 'weeks', 'relapse', ['placebo'], ties='efron')"
    assert COMMANDS["FitCoxPH"].cls("weeks", "relapse", "placebo", cov="cluster").validate(s)
    namespace = {"trial": trial}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces both
    assert np.array_equal(namespace["cox"].values, s.objects["cox"].values)
    assert namespace["km"].test.statistic == s.objects["km"].test.statistic
    assert s.objects["cox"].provenance.backend["name"] == "statsmodels"
