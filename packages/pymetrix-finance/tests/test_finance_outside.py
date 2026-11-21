"""Finance results against outside references: Black-Litterman against He and Litterman's (1999) published example;
the CDS bootstrap against QuantLib's PiecewiseFlatHazardRate; the copula densities against statsmodels' copulas."""

import numpy as np
import pytest

from pymetrix_finance import black_litterman, cds_curve

# He, G. and Litterman, R. (1999), The intuition behind Black-Litterman model portfolios, Goldman Sachs: seven
# countries' equity markets (Australia, Canada, France, Germany, Japan, UK, USA), Table 1 and the first view
CORR = np.array([[1, .488, .478, .515, .439, .512, .491], [.488, 1, .664, .655, .310, .608, .779],
                 [.478, .664, 1, .861, .355, .783, .668], [.515, .655, .861, 1, .354, .777, .653],
                 [.439, .310, .355, .354, 1, .405, .306], [.512, .608, .783, .777, .405, 1, .652],
                 [.491, .779, .668, .653, .306, .652, 1]])
SD = np.array([16.0, 20.3, 24.8, 27.1, 21.0, 20.0, 18.7]) / 100
WEIGHTS = np.array([1.6, 2.2, 5.2, 5.5, 11.6, 12.4, 61.5]) / 100
EQUILIBRIUM = [3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6]  # percent, as published (one decimal)
WITH_VIEW = [4.3, 7.6, 9.3, 11.0, 4.5, 7.0, 8.1]  # Germany outperforms France and the UK (cap-weighted) by 5%


def test_black_litterman_against_he_and_litterman():
    S = CORR * np.outer(SD, SD)
    pi, _ = black_litterman(S, WEIGHTS, delta=2.5)
    assert pi * 100 == pytest.approx(EQUILIBRIUM, abs=0.05)
    view = np.array([[0, 0, -0.295, 1, 0, -0.705, 0]])
    mu, _ = black_litterman(S, WEIGHTS, view, [0.05], delta=2.5, tau=0.05)  # Omega = diag(P tau S P'), as there
    assert mu * 100 == pytest.approx(WITH_VIEW, abs=0.05)


def test_the_cds_bootstrap_against_quantlib():
    ql = pytest.importorskip("QuantLib")
    mats, spreads, R, r = [1, 3, 5, 7, 10], [0.008, 0.011, 0.014, 0.015, 0.016], 0.4, 0.03
    ours = cds_curve(mats, spreads, recovery=R, rate=r)
    today = ql.Date(20, 3, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    disc = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)), dc, ql.Continuous))
    helpers = [ql.SpreadCdsHelper(ql.QuoteHandle(ql.SimpleQuote(s)), ql.Period(m, ql.Years), 0, ql.NullCalendar(), ql.Quarterly,
                                  ql.Unadjusted, ql.DateGeneration.TwentiethIMM, dc, R, disc) for m, s in zip(mats, spreads)]
    curve = ql.PiecewiseFlatHazardRate(today, helpers, dc)
    for m in mats:
        d = today + ql.Period(m, ql.Years)
        # survival to each maturity: QuantLib pays on IMM dates with day counts, pymetrix on a quarterly grid
        assert ours.survival([dc.yearFraction(today, d)])[0] == pytest.approx(curve.survivalProbability(d), abs=5e-4)


def test_copula_densities_against_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    import importlib

    cp = importlib.import_module("pymetrix_finance.copulas")
    u = np.random.default_rng(0).uniform(0.02, 0.98, (50, 2))
    R = np.array([[1, 0.4], [0.4, 1]])
    assert cp.clayton_logpdf(u, 2.0) == pytest.approx(sm.ClaytonCopula(theta=2.0).logpdf(u), abs=1e-12)
    assert cp.gumbel_logpdf(u, 1.8) == pytest.approx(sm.GumbelCopula(theta=1.8).logpdf(u), abs=1e-12)
    assert cp.t_logpdf(u, R, 5.0) == pytest.approx(sm.StudentTCopula(corr=R, df=5).logpdf(u), abs=1e-12)
    assert cp.gaussian_logpdf(u, R) == pytest.approx(sm.GaussianCopula(corr=R).logpdf(u), abs=1e-12)
