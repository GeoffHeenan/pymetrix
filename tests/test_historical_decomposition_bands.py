"""Bands on historical decompositions.

- A Bayesian VAR's posterior draws: for every draw, its contributions and its baseline add up to the data (1e-10);
  the point decomposition (at the posterior mean) is the one without bands, unchanged; the bands hold the median.
- The per-draw contributions (by the VAR's recursion) equal the moving-average sums of the point decomposition at the
  same coefficients and impact matrix.
- Sign and zero-sign identified sets (at the estimates, and over posterior draws): the same identity on every draw.
- A point-identified VAR without posterior draws refuses bands; views, the bands' table and figure; the dialog."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Bayes, Cholesky, Dataset, Minnesota, SignRestrictions
from pymetrix.commands import FitVAR, HistoricalDecomposition, Session
from pymetrix.errors import SpecificationError
from pymetrix.var import _contributions

NAMES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    T = 300
    A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.3, 0.6]])
    P = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.2, 0.3, 0.6]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = np.array([0.1, 0.2, -0.1]) + A1 @ y[t - 1] + P @ rng.standard_normal(3)
    idx = pd.period_range("1950Q1", periods=T, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame(y, columns=NAMES, index=idx))


@pytest.fixture(scope="module")
def bvar(data):
    return VAR(NAMES, lags=2).fit(data, method=Bayes(prior=Minnesota(), draws=200))


def _adds_up(fit, draws):
    contributions, baselines, _ = fit._hd_samples(draws, 0)
    assert len(contributions) == draws
    Y = fit.arrays.Y
    assert np.max(np.abs(contributions.sum(axis=3) + baselines - Y[None])) < 1e-10
    return contributions


def test_posterior_bands(bvar):
    point = bvar.historical_decomposition()
    banded = bvar.historical_decomposition(draws=200, alpha=0.1)
    assert point.lower is None and not point.has_bands
    for v in NAMES:
        assert banded.contributions[v].equals(point.contributions[v])  # the point decomposition is unchanged
        lo, med, hi = banded.lower[v].to_numpy(), banded.median[v].to_numpy(), banded.upper[v].to_numpy()
        assert np.all(lo <= med) and np.all(med <= hi) and np.any(hi - lo > 0)
    assert banded.coverage == pytest.approx(0.9)
    contributions = _adds_up(bvar, 200)
    assert np.allclose(np.median(contributions, axis=0)[:, 0, :], banded.median["a"].to_numpy())
    total = point.baseline + pd.DataFrame({v: point.contributions[v].sum(axis=1) for v in NAMES})
    assert np.allclose(total.to_numpy(), bvar.arrays.Y, atol=1e-10)


def test_the_recursion_matches_the_moving_average_sums(bvar):
    hd = bvar.historical_decomposition()
    d = len(bvar.arrays.d_names)
    U, _ = bvar._hd_parts(bvar.coefs, bvar.estimate.B[:d])
    c = _contributions(bvar.coefs, bvar.impact(), U)
    for i, v in enumerate(NAMES):
        assert np.allclose(c[:, i, :], hd.contributions[v].to_numpy(), atol=1e-10)


def test_identified_sets(data, bvar):
    var = VAR(NAMES, lags=2).fit(data)
    signs = SignRestrictions({"s1": {"a": "+", "b": "+"}, "s2": {"b": "+", "c": "-"}}, rotations=100)
    hd = var.identify(signs).historical_decomposition(draws=80)
    assert "accepted rotations" in hd.band_source
    _adds_up(var.identify(signs), 80)
    zero_sign = SignRestrictions({"s1": {"b": "+", "c": "+"}}, zeros={"s1": ["a"]}, rotations=100)
    for fit in (var.identify(zero_sign), bvar.identify(zero_sign)):
        contributions = _adds_up(fit, 60)
        assert np.max(np.abs(contributions[:, 0, 0, 0])) < 1e-10  # no contribution to a on impact in the first period
        hd = fit.historical_decomposition(draws=60)
        assert np.allclose(hd.upper["a"]["s1"].iloc[0], 0, atol=1e-10)
    with pytest.raises(SpecificationError, match="point-identified"):
        var.historical_decomposition(draws=10)
    _adds_up(bvar.identify(Cholesky(["c", "b", "a"])), 30)


def test_views_and_figure(bvar):
    from pymetrix.views import view_sections as views

    hd = bvar.historical_decomposition(draws=50)
    table = hd.bands_table("b")
    assert "a lower" in table.columns and "a upper" in table.columns
    fig = hd.plot_bands(["a"])
    assert len(fig.panels) == 3
    names = [name for name, _ in views(hd)]
    assert "Bands: a" in names
    assert not any(n.startswith("Bands") for n, _ in views(bvar.historical_decomposition()))
    with pytest.raises(SpecificationError, match="no bands"):
        bvar.historical_decomposition().bands_table("a")


def test_the_dialog(data):
    session = Session()
    session.objects["data"] = data
    session.run(FitVAR("a, b, c", lags="2", name="var"))
    session.run(HistoricalDecomposition("var", scheme="SignRestrictions", signs="s1: a+, b+", zeros="s1: c", rotations=40,
                                        draws=40, coverage=0.68))
    script = session.script()
    assert "historical_decomposition(draws=40, alpha=0.32)" in script
    assert session["hd"].has_bands
    ns = {"data": session["data"]}
    exec(script, ns)  # noqa: S102
    assert np.allclose(ns["hd"].upper["a"].to_numpy(), session["hd"].upper["a"].to_numpy())
