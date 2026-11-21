"""Markov-switching regressions (statsmodels): agreement with statsmodels, and recovery of simulated regimes."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pymetrix import Dataset, MarkovSwitching  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def simulated():
    """Mean +1 (stay probability 0.95) and -1 (stay probability 0.9), noise s.d. 0.6."""
    rng = np.random.default_rng(2)
    T = 300
    s = np.zeros(T, int)
    for t in range(1, T):
        stay = 0.95 if s[t - 1] == 0 else 0.9
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    y = np.where(s == 0, 1.0, -1.0) + rng.normal(0, 0.6, T)
    x = rng.normal(size=T)
    frame = pd.DataFrame({"g": y, "x": x, "gx": y + np.where(s == 0, 0.5, 2.0) * x},
                         index=pd.period_range("1950Q1", periods=T, freq="Q", name="time"))
    return Dataset.from_frame(frame), s


def _high(fit):
    """The regime with the larger intercept."""
    return int(np.argmax([fit.result.params[f"const[{i}]"] for i in range(fit.spec.regimes)]))


def test_recovers_the_regimes(simulated):
    data, s = simulated
    fit = MarkovSwitching("g ~ 1").fit(data)
    hi = _high(fit)
    assert fit.result.params[f"const[{hi}]"] == pytest.approx(1, abs=0.15)
    assert fit.result.params[f"const[{1 - hi}]"] == pytest.approx(-1, abs=0.15)
    assert fit.transition.iloc[hi, hi] == pytest.approx(0.95, abs=0.05)
    assert np.allclose(fit.transition.sum(axis=1), 1) and np.allclose(fit.durations, 1 / (1 - np.diag(fit.transition)))
    p = fit.probabilities()
    assert np.mean((p.iloc[:, hi] > 0.5).to_numpy() == (s == 0)) > 0.95
    assert np.allclose(p.sum(axis=1), 1) and str(p.index[0]) == "1950Q1"
    assert np.allclose(fit.probabilities("filtered").sum(axis=1), 1)
    assert "Expected durations: " in str(fit.summary())


def test_matches_statsmodels(simulated):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    data, _ = simulated
    fit = MarkovSwitching("gx ~ x", switching_variance=True).fit(data)
    f = data.frame.set_axis(data.frame.index.to_timestamp())
    ref = MarkovRegression(f["gx"], 2, exog=f[["x"]], switching_variance=True).fit(search_reps=20, rng=0, disp=False)
    assert fit.result.llf == pytest.approx(ref.llf, rel=1e-6)
    ours = dict(zip(fit.params.names, fit.params.values))
    # which regime is called 0 is arbitrary (it differs across statsmodels versions): compare up to relabelling
    swapped = {"p[0->0]": 1 - ours["p[1->0]"], "p[1->0]": 1 - ours["p[0->0]"],
               **{f"{k}[{i}]": ours[f"{k}[{1 - i}]"] for k in ("const", "x", "sigma2") for i in (0, 1)}}
    theirs = np.asarray(ref.params, dtype=float)
    assert any(np.allclose([o[n] for n in fit.params.names], theirs, atol=1e-4) for o in (ours, swapped))
    hi = _high(fit)
    assert "x[0]" in fit.params.names and "x[0]" in str(fit.summary())
    slopes = {fit.params[f"x[{i}]"] for i in range(2)}
    assert min(slopes) == pytest.approx(0.5, abs=0.2) and max(slopes) == pytest.approx(2, abs=0.2)
    assert fit.result.params[f"const[{hi}]"] > 0


def test_switching_autoregression_and_seeding(simulated):
    data, _ = simulated
    a = MarkovSwitching("g ~ 1", ar=1, search=5).fit(data)
    b = MarkovSwitching("g ~ 1", ar=1, search=5).fit(data)
    assert np.allclose(a.params.values, b.params.values)  # the random search is seeded
    assert len(a.probabilities()) == 299 and "1 switching AR lags" in str(a.summary())
    state = np.random.get_state()[1].copy()
    MarkovSwitching("g ~ 1", search=2).fit(data)
    assert np.array_equal(np.random.get_state()[1], state)  # NumPy's global generator is untouched
    with pytest.raises(ValueError, match="2 or more"):
        MarkovSwitching("g ~ 1", regimes=1).fit(data)


def test_dialog_view_and_project(simulated, tmp_path):
    from pymetrix.commands import FitMarkovSwitching, Session, load_project, save_project
    from pymetrix.ui.render import view_sections

    data, _ = simulated
    s = Session()
    s.objects["data"] = data
    s.run(FitMarkovSwitching("g ~ 1", search=5, start="1960Q1"))
    assert "ms = MarkovSwitching(formula='g ~ 1', search=5).fit(data, span=('1960Q1', None))" in s.script()
    assert str(s["ms"].probabilities().index[0]) == "1960Q1"
    titles = [t for t, _ in view_sections(s["ms"])]
    assert titles[0] == "Summary" and {"Transition probabilities", "Regime probabilities"} <= set(titles)
    reopened = load_project(save_project(s, tmp_path / "ms.pymx"))
    assert np.allclose(reopened["ms"].params.values, s["ms"].params.values)
