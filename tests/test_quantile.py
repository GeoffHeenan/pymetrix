"""Quantile regression, a wrapper of statsmodels' QuantReg."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402

from pymetrix import Dataset, Equation, Quantile, quantile_process  # noqa: E402


@pytest.fixture(scope="module")
def data():
    """y = 1 + 2x + (0.5 + x) e: the spread grows with x, so the slope rises with the quantile."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, 500)
    y = 1 + 2 * x + (0.5 + x) * rng.normal(size=500)
    return Dataset.from_frame(pd.DataFrame({"i": range(500), "x": x, "y": y}), entity="i")


def test_matches_statsmodels(data):
    for q, cov in ((0.5, "robust"), (0.9, "iid")):
        ours = Equation("y ~ x").fit(data, method=Quantile(q=q, vcov=cov))
        X = np.column_stack([np.ones(500), data["x"].to_numpy()])
        ref = QuantReg(data["y"].to_numpy(), X).fit(q=q, vcov=cov)
        assert np.allclose(ours.params.values, ref.params) and np.allclose(ours.params.se, ref.bse)
        assert ours.stats["pseudo_r2"] == pytest.approx(ref.prsquared) and ours.stats["q"] == q


def test_median_of_a_constant_is_the_median(data):
    fit = Equation("y ~ 1").fit(data, method=Quantile(q=0.5))
    middle = np.sort(data["y"].to_numpy())[249:251]  # with an even count, any value between the middle two is a median
    assert middle[0] - 1e-6 <= fit.params["const"] <= middle[1] + 1e-6


def test_the_process_shows_slopes_rising_with_the_quantile(data):
    pytest.importorskip("matplotlib")
    proc = quantile_process(Equation("y ~ x"), data, quantiles=(0.1, 0.5, 0.9))
    slopes = [f.params["x"] for f in proc.fits.values()]
    assert slopes[0] < slopes[1] < slopes[2] and slopes[0] < 1 < 3 < slopes[2]
    table = proc.table().to_frame()
    assert list(table["quantile"])[-1] == "OLS"
    fig = proc.plot()
    assert [p.title for p in fig.panels] == ["const", "x"]


def test_requirements(data):
    with pytest.raises(ValueError, match="between 0 and 1"):
        Quantile(q=1.2)
    with pytest.raises(ValueError, match="weights"):
        Equation("y ~ x").fit(data, method=Quantile(), weights="x")


def test_the_estimate_dialog_offers_it(data):
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("y ~ x", method="Quantile regression", method_settings="q=0.25"))
    assert "method=Quantile(q=0.25)" in s.script() and "Quantile 0.25" in str(s["fit"].summary())
