"""One object of each kind that has a window in the desktop app, built from the public API on example data: what
tests/test_result_windows.py opens tab by tab. Each builder is independent, so one failing to build is reported on
its own."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def _two_returns():
    """Two correlated daily returns, 400 days: enough for a multivariate GARCH."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    r = rng.multivariate_normal([0, 0], [[1e-4, 4e-5], [4e-5, 1e-4]], size=400)
    return Dataset.from_frame(pd.DataFrame(r, columns=["x", "y"], index=pd.period_range("2001-01-01", periods=400, freq="D", name="time")))


def _three_returns():
    """Three correlated daily returns, 400 days: a vine's smallest interesting case."""
    from pymetrix import Dataset

    rng = np.random.default_rng(1)
    cov = [[1e-4, 4e-5, 3e-5], [4e-5, 1e-4, 5e-5], [3e-5, 5e-5, 1e-4]]
    r = rng.multivariate_normal([0, 0, 0], cov, size=400)
    return Dataset.from_frame(pd.DataFrame(r, columns=["x", "y", "z"], index=pd.period_range("2001-01-01", periods=400, freq="D", name="time")))


def _quotes():
    """Calls at five strikes and two expiries, priced with a smile."""
    from pymetrix import Dataset
    from pymetrix_finance.options import black_scholes

    K, T = np.tile([80.0, 90, 100, 110, 120], 2), np.repeat([0.25, 1.0], 5)
    price = black_scholes(100, K, T, 0.02, 0.2 + 0.1 * (K / 100 - 1) ** 2)
    return Dataset.from_frame(pd.DataFrame({"quote": [f"q{i}" for i in range(10)], "price": price, "strike": K, "expiry": T}), entity="quote")


def _yields():
    """Nelson-Siegel yields at six maturities, monthly, with a little noise."""
    from pymetrix import Dataset
    from pymetrix_finance.yields import loadings

    rng = np.random.default_rng(0)
    F = np.column_stack([5 + np.cumsum(rng.normal(0, 0.1, 120)), -1 + np.cumsum(rng.normal(0, 0.1, 120)), rng.normal(0, 1, 120)])
    Y = F @ loadings([0.25, 1, 2, 5, 10, 30], 0.6).T + rng.normal(0, 0.01, (120, 6))
    cols = ["m3", "y1", "y2", "y5", "y10", "y30"]
    return Dataset.from_frame(pd.DataFrame(Y, columns=cols, index=pd.period_range("2010-01", periods=120, freq="M", name="time")))


def _intraday():
    """Five-minute returns, 78 a day for 80 trading days, with a persistent daily variance."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    lv = np.log(1e-4) + np.cumsum(rng.normal(0, 0.1, 80)) * 0.5
    r = rng.normal(0, 1, (80, 78)) * np.sqrt(np.exp(lv)[:, None] / 78)
    days = pd.bdate_range("2020-01-02", periods=80)
    idx = pd.PeriodIndex([pd.Period(d + pd.Timedelta(minutes=570 + 5 * j), "5min") for d in days for j in range(78)], name="time")
    return Dataset.from_frame(pd.DataFrame({"r": r.ravel()}, index=idx))


def _merton():
    """A firm whose equity is a call on simulated assets."""
    from pymetrix import Dataset, merton
    from pymetrix_finance.options import black_scholes

    rng = np.random.default_rng(0)
    V = 100 * np.exp(np.cumsum(rng.normal(0, 0.3 / np.sqrt(252), 120)))
    E = black_scholes(V, 70.0, 1.0, 0.02, 0.3)
    idx = pd.PeriodIndex(pd.bdate_range("2020-01-01", periods=120), freq="D", name="time")
    return merton(Dataset.from_frame(pd.DataFrame({"mcap": E, "debt": 70.0}, index=idx)), "mcap", "debt", rate=0.02, periods_per_year=252)


def _gar_data():
    """Growth whose downside widens with financial conditions, quarterly."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = 2 - 0.5 * np.roll(x, 1) + (1 + 0.4 * np.abs(np.roll(x, 1))) * rng.normal(size=100)
    return Dataset.from_frame(pd.DataFrame({"g": y, "nfci": x}, index=pd.period_range("1990Q1", periods=100, freq="Q", name="time")))


def _ma_sim(theta, shocks):
    return shocks[1:] + theta[0] * shocks[:-1]


def _simulation():
    import numpy as np

    from pymetrix import indirect_inference

    e = np.random.default_rng(0).normal(size=201)
    return indirect_inference(e[1:] + 0.5 * e[:-1], _ma_sim, lambda z: __import__("pymetrix").simulation_estimators.ar_coefficients(z, 2)[1:3], [0.2], paths=5)


def _partially_linear():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, partially_linear

    rng = np.random.default_rng(0)
    z = rng.uniform(-2, 2, 120)
    x = 0.5 * z + rng.normal(size=120)
    frame = pd.DataFrame({"id": range(120), "y": 1.5 * x + np.sin(z) + rng.normal(size=120) * 0.5, "x": x, "z": z})
    return partially_linear(Dataset.from_frame(frame, entity="id"), "y", ["x"], ["z"], bandwidth=0.4)


def _semiparametric_garch():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, semiparametric_garch

    r = np.random.default_rng(0).standard_t(6, 300) * 0.8
    return semiparametric_garch(Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2000-01-03", periods=300, freq="D"))), "r", rounds=1)


def _inar():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, inar

    rng = np.random.default_rng(0)
    y = [3]
    for _ in range(119):
        y.append(rng.binomial(y[-1], 0.5) + rng.poisson(1.5))
    return inar(Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("2000-01", periods=120, freq="M"))), "y")


def _staggered_block():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    rows = [(i, 2000 + t, rng.normal() + 0.1 * t + (1.0 if i >= 10 and t >= 6 else 0.0), float(i >= 10 and t >= 6))
            for i in range(14) for t in range(9)]
    return Dataset.from_frame(pd.DataFrame(rows, columns=["unit", "year", "y", "d"]), entity="unit", time="year", freq="Y")


def _favar():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, favar

    rng = np.random.default_rng(0)
    f = np.cumsum(rng.normal(size=(120, 2)), axis=0) * 0.2
    cols = {f"x{i}": f @ rng.normal(size=2) + rng.normal(size=120) * 0.5 for i in range(12)}
    cols["r"] = 0.5 * f[:, 0] + rng.normal(size=120) * 0.3
    data = Dataset.from_frame(pd.DataFrame(cols, index=pd.period_range("1990-01", periods=120, freq="M")))
    return favar(data, [f"x{i}" for i in range(12)], ["r"], factors=2, lags=1)


def _spatial():
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, knn_weights, spatial_regression

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, (60, 2))
    data = Dataset.from_frame(pd.DataFrame({"id": range(60), "y": rng.normal(size=60), "x": rng.normal(size=60), "lon": xy[:, 0], "lat": xy[:, 1]}),
                              entity="id")
    return spatial_regression(data, "y", ["x"], knn_weights(data, ["lon", "lat"], k=4))


def _proxy():
    """A proxy SVAR on a simulated VAR whose first shock a noisy proxy measures."""
    from pymetrix import VAR, Dataset

    rng = np.random.default_rng(0)
    eps = rng.normal(size=(300, 2))
    y = np.zeros((300, 2))
    for t in range(1, 300):
        y[t] = 0.5 * y[t - 1] + np.array([[1.0, 0.3], [0.5, 1.0]]) @ eps[t]
    frame = pd.DataFrame({"r": y[:, 0], "x": y[:, 1], "z": eps[:, 0] + rng.normal(size=300)},
                         index=pd.period_range("1990-01", periods=300, freq="M", name="time"))
    data = Dataset.from_frame(frame)
    return VAR(["r", "x"], lags=1).fit(data).proxy_svar(data, "z", periods=6, draws=50)


def _staggered():
    """A balanced panel with adoption in three cohorts and effects that grow after it."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    cohort = rng.choice([3, 5, 7, np.inf], 60)
    rows = [{"unit": i, "year": 2000 + t, "d": float(t >= cohort[i]),
             "y": i % 5 + 0.2 * t + (1 + 0.5 * (t - cohort[i]) if t >= cohort[i] else 0.0) + rng.normal(0, 0.5)}
            for i in range(60) for t in range(10)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="unit", time="year", freq="Y")


def _macro():
    from pymetrix import Dataset

    return Dataset.example("sample_macro")


def _nk():
    from pymetrix import templates

    return templates.new_keynesian()


def _solution():
    nk = _nk()
    return nk.model.linearize(nk.model.steady_state(), shocks=nk.shocks).solve()


def _dsge_var():
    """A DSGE-VAR of the New Keynesian template on its own simulated data, lambda chosen on a small grid."""
    from pymetrix import Dataset
    from pymetrix.estimation.priors import Beta

    nk = _nk()
    sim = nk.solve().simulate(100, nk.shock_sd, seed=2)
    sim.index = pd.period_range("1990Q1", periods=100, freq="Q", name="time")
    return nk.dsge_var(nk.measurement.apply(Dataset.from_frame(sim)), lags=1, lambda_="grid", lambdas=[0.5, 1, 2],
                       estimate=["rho"], priors={"rho": Beta(0.7, 0.1)})


def _particle_filter():
    """The bootstrap particle filter on the New Keynesian template's second-order solution, with measurement errors."""
    from pymetrix import Dataset

    nk = _nk()
    m = nk.measurement.with_errors({o: 1.0 for o in nk.measurement.observables})
    path = nk.solve().simulate(40, nk.shock_sd, seed=3)
    path.index = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    return nk.solve(order=2).particle_filter(m.apply(Dataset.from_frame(path)), m, particles=200, seed=1)


_EVIEWS = __import__("pathlib").Path(__file__).parent / "fixtures" / "eviews"
_SSPACE = """sspace ss
ss.append @signal y = trend + cycle + [var = c(3)^2]
ss.append @state trend = trend(-1) + [var = c(1)^2]
ss.append @state cycle = c(4)*cycle(-1) + ecyc
ss.append @ename ecyc
ss.append @evar var(ecyc) = c(2)^2
c(1) = 0.3
c(2) = 0.8
c(3) = 0.5
c(4) = 0.7
"""


def _frbus():
    """The FRB/US excerpt in tests/fixtures/eviews and its data, the add factor series at zero."""
    from pymetrix import Dataset
    from pymetrix.models.eviews import read_eviews_xml

    read = read_eviews_xml(_EVIEWS / "frbus_excerpt.xml")
    frame = pd.read_csv(_EVIEWS / "frbus_longbase_excerpt.csv", comment="#")
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q", name="time")
    missing = [v for v in read.model.exogenous if v.endswith("_aerr") and v not in frame.columns]
    return read, Dataset.from_frame(pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1))


def _optimal_control():
    from pymetrix import optimal_control

    read, data = _frbus()
    return optimal_control(read.model, data, ("2026Q2", "2027Q4"), ["rff"], {"zdivgr": 1.0, "delrff": 1.0},
                           desired={"zdivgr": 0.05}, control=("2026Q2", "2027Q2"))


def _var():
    from pymetrix import VAR

    data = _macro()
    return VAR(["c", "y", "i"], lags=2).fit(data)


def _monthly_quarterly():
    from pymetrix import Dataset

    rng = np.random.default_rng(3)
    months = pd.period_range("2010-01", periods=120, freq="M", name="time")
    quarters = pd.period_range("2010Q1", periods=40, freq="Q", name="time")
    m = pd.DataFrame({"pmi": rng.normal(50, 2, 120), "gr": rng.normal(0.2, 0.5, 120)}, index=months)
    q = pd.DataFrame({"gy": rng.normal(0.5, 0.6, 40)}, index=quarters)
    return Dataset.from_frame(m), Dataset.from_frame(q)


_FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _backward_solution():
    from test_models import _economy

    model, data = _economy()
    return model.solve(data, span=("2005Q1", "2010Q4"))


def _scenarios():
    from pymetrix import Scenario, compare_solutions
    from test_models import _economy

    model, data = _economy()
    span = ("2005Q1", "2010Q4")
    return compare_solutions(base=model.solve(data, span=span), shocked=model.solve(data, span=span, scenario=Scenario(shift={"g": 1.0})))


def _news():
    from pymetrix import DynamicFactor

    monthly, quarterly = _monthly_quarterly()
    q_early = quarterly.select(span=("2010Q1", "2018Q4"))  # the quarter after is nowcast from monthly figures
    first = DynamicFactor(monthly=["gr", "pmi"], quarterly=["gy"]).fit(q_early, monthly.select(span=("2010-01", "2019-01")))
    return first.update(q_early, monthly.select(span=("2010-01", "2019-03"))).news(first, "gy")


def _session_with_loops():
    from pymetrix.commands import FitEach, SelectSeries, Session

    s = Session()
    s.objects["data"] = _macro()
    s.run(SelectSeries(series="c, i", patterns="g*", name="spend"))
    s.run(FitEach("{s} ~ {s}[-1]", selection="spend"))
    return s



def _occbin_filter():
    """The OccBin filter on data simulated with the floor binding."""
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, occbin, occbin_filter, templates
    from pymetrix.models import MeasurementMap

    zlb = templates.new_keynesian(lower_bound=0.0)
    rng = np.random.default_rng(1)
    shocks = {s: {t + 1: float(v) for t, v in enumerate(rng.normal(0, zlb.shock_sd[s], 40))} for s in zlb.shocks}
    shocks["ey"].update({20: -10.0, 21: -10.0})
    frame = occbin(zlb, shocks, periods=40).frame[["y", "pi", "i"]]
    frame.index = pd.period_range("2010Q1", periods=40, freq="Q", name="time")
    return occbin_filter(zlb, Dataset.from_frame(frame), MeasurementMap.direct(["y", "pi", "i"]))

def _micro_regions():
    """300 regions with exposure shares to 12 industries, a shift-share shock and a treatment it moves."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    S = rng.dirichlet(np.full(12, 0.5), size=300)
    g = rng.normal(size=12)
    u = rng.normal(size=300)
    x = S @ g + 0.5 * u + rng.normal(size=300)
    frame = pd.DataFrame(S, columns=[f"s{k}" for k in range(12)]).assign(x=x, y=2 * x + u, w=rng.normal(size=300), id=range(300))
    return Dataset.from_frame(frame, entity="id"), dict(zip([f"s{k}" for k in range(12)], g))


def _factor_panel():
    """40 units over 20 years with one common factor that also drives the regressor."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    F, L = rng.normal(size=20), rng.normal(size=40)
    x = np.outer(L, F) + rng.normal(size=(40, 20))
    y = x + np.outer(L, F) + rng.normal(size=(40, 20))
    frame = pd.DataFrame({"unit": np.repeat([f"u{i:02d}" for i in range(40)], 20), "year": np.tile(np.arange(2000, 2020), 40), "y": y.ravel(), "x": x.ravel()})
    return Dataset.from_frame(frame, entity="unit", time="year", freq="Y")


def _choices():
    """200 trips choosing among three modes, by cost."""
    from pymetrix import Dataset

    rng = np.random.default_rng(0)
    cost = rng.normal(5, 2, size=(200, 3))
    u = -0.5 * cost + np.array([0.0, 0.3, -0.2]) + rng.gumbel(size=(200, 3))
    chosen = (u == u.max(axis=1, keepdims=True)).astype(float)
    frame = pd.DataFrame({"trip": np.repeat(np.arange(200), 3), "mode": np.tile(["bus", "car", "train"], 200), "cost": cost.ravel(), "chosen": chosen.ravel()})
    return Dataset.from_frame(frame, entity=["trip", "mode"])


BUILDERS: dict[str, Callable[[], Any]] = {
    "Equation": lambda: __import__("pymetrix").Equation("c ~ y + c[-1]"),
    "Series": lambda: _macro()["c"],
    "Table": lambda: _macro().summary() if hasattr(_macro(), "summary") else __import__("pymetrix").Table(["a"], [[1.0]]),
    "Figure": lambda: _macro().plot(["c", "y"]),
    "Sample": lambda: __import__("pymetrix").Sample(("2005Q1", "2020Q4")),
    "TestResult": lambda: __import__("pymetrix").stats.run_test(_macro(), "adf", variable="c"),
    "FittedARDL": lambda: __import__("pymetrix").ARDL("c ~ y + i", lags=2, order=2).fit(_macro()),
    "FittedVAR": _var,
    "IRF": lambda: _var().irf(12),
    "FEVD": lambda: _var().fevd(12),
    "FittedVECM": lambda: __import__("pymetrix").VECM(["c", "y"], lags=2, rank=1).fit(_macro()),
    "FittedLocalProjection": lambda: __import__("pymetrix").LocalProjection("y", shock="i", horizons=8, lags=2).fit(_macro()),
    "FittedMarkovSwitching": lambda: __import__("pymetrix").MarkovSwitching("g ~ 1").fit(_macro()),
    "FittedSystem": lambda: __import__("pymetrix").System([__import__("pymetrix").Equation("c ~ y"), __import__("pymetrix").Equation("i ~ y")]).fit(_macro(), method=__import__("pymetrix").SUR()),
    "FittedStateSpace": lambda: __import__("pymetrix").UnobservedComponents("y", trend="local linear trend", cycle=0).fit(_macro()),
    "FittedPCA": lambda: __import__("pymetrix").PCA(["c", "y", "i", "g"]).fit(_macro()),
    "FittedFactorModel": lambda: __import__("pymetrix").FactorModel(["educ", "motheduc", "fatheduc", "huseduc", "age", "exper"], 2,
                                                                     rotation="promax").fit(__import__("pymetrix").Dataset.example("mroz")),
    "RollingFit": lambda: __import__("pymetrix").Equation("c ~ y").fit_rolling(_macro(), 40),
    "Backtest": lambda: __import__("pymetrix").backtest(__import__("pymetrix").Equation("c ~ y").fit(_macro()), _macro(), ("2018Q1", "2021Q4"), horizon=2),
    "QuantileProcess": lambda: __import__("pymetrix").quantile_process(__import__("pymetrix").Equation("c ~ y"), _macro()),
    "FittedMIDAS": lambda: (lambda mq: __import__("pymetrix").MIDAS("gy ~ pmi", lags=3).fit(mq[1], mq[0]))(_monthly_quarterly()),
    "FittedBridge": lambda: (lambda mq: __import__("pymetrix").Bridge("gy ~ gr + pmi").fit(mq[1], mq[0]))(_monthly_quarterly()),
    "FittedDynamicFactor": lambda: (lambda mq: __import__("pymetrix").DynamicFactor(monthly=["gr", "pmi"], quarterly=["gy"]).fit(mq[1], mq[0]))(_monthly_quarterly()),
    "FittedDisaggregation": lambda: __import__("pymetrix").TemporalDisaggregation("c ~ y").fit(_macro().to_freq("Y", how="sum").keep("c"), _macro()),
    "Reconciliation": lambda: __import__("pymetrix").reconcile(_macro().with_series(cg="c + g"), ["c", "g"], "cg", _macro().to_freq("Y", how="sum").keep("c")),
    "LinearSolution": _solution,
    "FittedDSGEVAR": _dsge_var,
    "SteadyState": lambda: _nk().model.steady_state(),
    "Moments": lambda: _solution().moments(_nk().shock_sd) if hasattr(_solution(), "moments") else None,
    "ParameterSet": lambda: _nk().model.compile().parameters,
    "ModelSystem": lambda: _nk().model.compile(),
    "ModelCheck": lambda: _nk().model.check(),
    "MeasurementMap": lambda: _nk().measurement,
    "StateSpaceSystem": lambda: _solution().system(_nk().measurement, _nk().shock_sd),
    "ScenarioSet": _scenarios,
    "Calibration": lambda: __import__("pymetrix").templates.rbc().model.calibrate({"k/y": 10}, ["beta"]),
    "Solution": _backward_solution,
    "DynareModel": lambda: __import__("pymetrix").read_mod(str(_FIXTURES / "dynare" / "rbc_order2.mod")),
    "News": _news,
    "Validation": lambda: _macro().validate(),
    "ndarray": lambda: np.arange(12.0).reshape(3, 4),
    "Selection": lambda: _session_with_loops()["spend"],
    "Results": lambda: next(o for o in _session_with_loops().objects.values() if type(o).__name__ == "Results"),
    "FittedExponentialSmoothing": lambda: __import__("pymetrix").ExponentialSmoothing("c", trend="add", damped=True).fit(_macro()),
    "DiD": lambda: __import__("pymetrix").did(__import__("pymetrix").Dataset.example("policy_panel"), "y", "d"),
    "EventStudy": lambda: __import__("pymetrix").event_study(__import__("pymetrix").Dataset.example("policy_panel"), "y", "d", window=(-3, 3)),
    "RDD": lambda: __import__("pymetrix").rdd(__import__("pymetrix").Dataset.example("rd_sample"), "y", "score", bandwidth=0.5),
    "TreatmentEffect": lambda: __import__("pymetrix").treatment_effect(__import__("pymetrix").Dataset.example("policy_panel"), "y", "treated", ["x"], reps=20),
    "SyntheticControl": lambda: __import__("pymetrix").synthetic_control(__import__("pymetrix").Dataset.example("synth_panel"), "y", 1, 2005),
    "FittedMGARCH": lambda: __import__("pymetrix").mgarch(_two_returns(), ["x", "y"], model="ccc"),
    "Performance": lambda: __import__("pymetrix").performance(_two_returns(), ["x", "y"], benchmark="y"),
    "FactorModel": lambda: __import__("pymetrix").factor_model(_two_returns(), ["x"], ["y"]),
    "Portfolio": lambda: __import__("pymetrix").optimize_portfolio(_two_returns(), ["x", "y"], objective="max_sharpe"),
    "OptionValuation": lambda: __import__("pymetrix").option_price(100, 95, 0.5, 0.03, 0.2),
    "VolatilitySmile": lambda: __import__("pymetrix").implied_volatility(_quotes(), "price", "strike", "expiry", spot=100, rate=0.02),
    "YieldCurve": lambda: __import__("pymetrix").yield_curve(_yields(), ["m3", "y1", "y2", "y5", "y10", "y30"], [0.25, 1, 2, 5, 10, 30]),
    "RiskContributions": lambda: __import__("pymetrix").risk_contributions(_two_returns(), ["x", "y"], [0.6, 0.4]),
    "BlackLittermanPosterior": lambda: __import__("pymetrix").black_litterman(_two_returns().cov(["x", "y"]), [0.6, 0.4], [[1, -1]], [0.0002]),
    "AbnormalReturns": lambda: __import__("pymetrix").abnormal_returns(_two_returns(), {"x": ["2001-09-11", "2001-12-03"]}, market="y",
                                                                      estimation=(-200, -11)),
    "FittedHAR": lambda: __import__("pymetrix").har(__import__("pymetrix").realized_measures(_intraday(), "r"), "rv"),
    "StrategyBacktest": lambda: __import__("pymetrix").backtest_strategy(_two_returns(), ["x", "y"], window=100, rebalance=20),
    "FittedDNS": lambda: __import__("pymetrix").dynamic_nelson_siegel(_yields(), ["m3", "y1", "y2", "y5", "y10", "y30"], [0.25, 1, 2, 5, 10, 30],
                                                                   maxiter=50),
    "FittedCopula": lambda: __import__("pymetrix").copula(_two_returns(), ["x", "y"], family="t"),
    "FittedVine": lambda: __import__("pymetrix").vine_copula(_three_returns(), ["x", "y", "z"]),
    "FittedTVCopula": lambda: __import__("pymetrix").tv_copula(_two_returns(), ["x", "y"]),
    "FittedDCCCopula": lambda: __import__("pymetrix").dcc_copula(_three_returns(), ["x", "y", "z"]),
    "CDSCurve": lambda: __import__("pymetrix").cds_curve([1, 3, 5], [0.008, 0.011, 0.014]),
    "MertonModel": lambda: _merton(),
    "FittedVolSurface": lambda: __import__("pymetrix").vol_surface(_quotes(), "price", "strike", "expiry", spot=100, rate=0.02, model="sabr"),
    "RiskForecast": lambda: __import__("pymetrix").value_at_risk(__import__("pymetrix").Dataset.example("sample_macro").with_series(gy="dlog(y)"), "gy", method="normal", window=40),
    "GrowthAtRisk": lambda: __import__("pymetrix").growth_at_risk(_gar_data(), "g", ["nfci"], horizons=(1,)),
    "FittedSmoothTransitionAR": lambda: __import__("pymetrix").SmoothTransitionAR("g", lags=1).fit(_gar_data()),
    "ProxySVAR": lambda: _proxy(),
    "SimulationEstimate": lambda: _simulation(),
    "MinimumDistance": lambda: __import__("pymetrix").minimum_distance([0.5, 0.26, 0.12], [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], lambda th: [th[0], th[0] ** 2, th[0] ** 3], [0.4]),
    "HeteroskedasticitySVAR": lambda: __import__("pymetrix").VAR(["g", "nfci"], lags=1).fit(_gar_data()).heteroskedasticity_svar(
        [str(_gar_data().frame.index[60])]),
    "StructuralEstimate": lambda: __import__("pymetrix").VAR(["g", "nfci"], lags=1).fit(_gar_data()).identify(
        __import__("pymetrix").AB(A=[[1, 0], ["*", 1]], B=[["*", 0], [0, "*"]])).structural(),
    "KernelDensity": lambda: __import__("pymetrix").kde(_gar_data(), "g"),
    "KernelRegression": lambda: __import__("pymetrix").kernel_regression(_gar_data(), "g", ["nfci"], bandwidth=0.5),
    "PartiallyLinear": lambda: _partially_linear(),
    "SemiparametricGARCH": lambda: _semiparametric_garch(),
    "SeriesModel": lambda: _inar(),
    "CointegratingRegression": lambda: __import__("pymetrix").cointegrating_regression(__import__("pymetrix").Dataset.example("sample_macro"), "c", ["y"]),
    "PanelMeanGroup": lambda: __import__("pymetrix").mean_group(__import__("pymetrix").Dataset.example("grunfeld"), "invest", ["value", "capital"]),
    "PooledMeanGroup": lambda: __import__("pymetrix").pooled_mean_group(__import__("pymetrix").Dataset.example("grunfeld"), "invest", ["value"]),
    "SyntheticDiD": lambda: __import__("pymetrix").synthetic_did(_staggered_block(), "y", "d"),
    "FittedThresholdVAR": lambda: __import__("pymetrix").threshold_var(_gar_data(), ["g", "nfci"], lags=1),
    "FittedVARMA": lambda: __import__("pymetrix").varma(_gar_data(), ["g", "nfci"], p=1, q=1),
    "FittedBVARSV": lambda: __import__("pymetrix").BVARSV(["g", "nfci"], lags=1).fit(_gar_data(), draws=20, burn=10),
    "FittedFAVAR": lambda: _favar(),
    "FittedMSVAR": lambda: __import__("pymetrix").MSVAR(["g"], lags=1).fit(_gar_data(), starts=2),
    "Spectrum": lambda: __import__("pymetrix").spectrum(_gar_data(), "g"),
    "CrossSpectrum": lambda: __import__("pymetrix").cross_spectrum(_gar_data(), "g", "nfci"),
    "FittedARFIMA": lambda: __import__("pymetrix").arfima(_gar_data(), "nfci"),
    "FittedQuantileVAR": lambda: __import__("pymetrix").QuantileVAR(["g", "nfci"], lags=1, quantile=0.1).fit(_gar_data()),
    "SpatialModel": lambda: _spatial(),
    "FittedTVPVAR": lambda: __import__("pymetrix").TVPVAR(["g", "nfci"], lags=1).fit(_gar_data(), draws=20, burn=10, training=20),
    "StaggeredDiD": lambda: __import__("pymetrix").staggered_did(_staggered(), "y", "d", draws=20),
    "BaconDecomposition": lambda: __import__("pymetrix").bacon_decomposition(_staggered(), "y", "d"),
    "FittedSmoothLocalProjection": lambda: __import__("pymetrix").LocalProjection("g", "nfci", horizons=6, lags=1).fit_smooth(_gar_data()),
    "FittedPanelVAR": lambda: __import__("pymetrix").VAR(["lwage", "hours"], lags=1, by="nr").fit(__import__("pymetrix").Dataset.example("wage_panel")),
    "PanelDecomposition": lambda: __import__("pymetrix").VAR(["lwage", "hours"], lags=1, by="nr").fit(
        __import__("pymetrix").Dataset.example("wage_panel")).historical_decomposition(draws=20),
    "StructuralBreaks": lambda: __import__("pymetrix").Equation("dlog(c) ~ dlog(y)").fit(_macro()).breaks(max_breaks=2),
    "Matrix": lambda: __import__("pymetrix").Matrix.cov(__import__("pymetrix").Equation("c ~ y + i").fit(_macro())),
    "SVD": lambda: __import__("pymetrix").Matrix.cov(__import__("pymetrix").Equation("c ~ y + i").fit(_macro())).svd(),
    "QR": lambda: __import__("pymetrix").Matrix.cov(__import__("pymetrix").Equation("c ~ y + i").fit(_macro())).qr(),
    "LU": lambda: __import__("pymetrix").Matrix.cov(__import__("pymetrix").Equation("c ~ y + i").fit(_macro())).lu(),
    "Eig": lambda: __import__("pymetrix").Matrix([[0.5, -0.8], [0.9, 0.3]], ["a", "b"], ["a", "b"]).eig(),
    "Schur": lambda: __import__("pymetrix").Matrix([[0.5, -0.8], [0.9, 0.3]], ["a", "b"], ["a", "b"]).schur(),
    "QZ": lambda: __import__("pymetrix").Matrix([[0.5, -0.8], [0.9, 0.3]]).qz([[1.0, 0.2], [0.0, 1.0]]),
    "FittedSARIMA": lambda: __import__("pymetrix").SARIMA("c", (1, 1, 0), (0, 0, 1)).fit(_macro()),
    "FittedTheta": lambda: __import__("pymetrix").Theta("c").fit(_macro()),
    "FittedSTLForecast": lambda: __import__("pymetrix").STLForecast("c").fit(_macro()),
    "FittedBMA": lambda: __import__("pymetrix").BMA("dlog(c)", ["dlog(y)", "dlog(i)", "dlog(c)[-1]"]).fit(_macro()),
    "FittedThresholdAR": lambda: __import__("pymetrix").ThresholdAR("g", lags=1).fit(_macro()),
    **({"FittedLearner": lambda: __import__("pymetrix").Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]").fit(
        _macro(), method=__import__("pymetrix").GradientBoosting(trees=50))} if __import__("importlib.util").util.find_spec("sklearn") else {}),
    "OccBinFilter": _occbin_filter,
    "ParticleFilter": _particle_filter,
    "Translation": lambda: __import__("pymetrix").export_model(_nk(), "dynare"),
    "EViewsXmlModel": lambda: _frbus()[0],
    "StateSpace": lambda: __import__("pymetrix.models.eviews", fromlist=["x"]).read_eviews_sspace(_SSPACE),
    "OptimalControl": _optimal_control,
    "dict": lambda: {"price": 101.2, "duration": 4.3, "convexity": 21.0},  # a plain result of a recorded call (bond measures)
    "tuple": lambda: (np.array([0.05, 0.07]), np.array([[0.04, 0.01], [0.01, 0.09]])),  # a plain tuple of arrays (a mean and a covariance)
    "FittedDoubleML": lambda: __import__("pymetrix").double_ml(_micro_regions()[0], "y", "x", ["w", "s0", "s1"], folds=3, repeats=2),
    "FittedShiftShare": lambda: __import__("pymetrix").shift_share(_micro_regions()[0], "y", "x", [f"s{k}" for k in range(12)], _micro_regions()[1]),
    "FittedInteractiveFE": lambda: __import__("pymetrix").interactive_fe(_factor_panel(), "y", ["x"], factors="auto", max_factors=3),
    "FittedConditionalLogit": lambda: __import__("pymetrix").choice_logit(_choices(), "chosen", ["cost"], alternative_constants=True),
    "ConditionalLogitEffects": lambda: __import__("pymetrix").choice_logit(_choices(), "chosen", ["cost"], alternative_constants=True).marginal_effects(),
    "SavedResult": lambda: __import__("pymetrix.saved", fromlist=["x"]).result_from_bytes(
        __import__("pymetrix.saved", fromlist=["x"]).result_bytes(__import__("pymetrix").Equation("c ~ y").fit(_macro()), "cons")),
}

# the finance plugin's results, only where the plugin is installed (the core-only run has none of them)
FINANCE = ("RiskForecast", "FittedMGARCH", "Performance", "FactorModel", "Portfolio", "OptionValuation", "VolatilitySmile",
           "YieldCurve", "RiskContributions", "AbnormalReturns", "FittedHAR", "StrategyBacktest", "FittedDNS", "FittedCopula",
           "FittedVine", "FittedTVCopula", "FittedDCCCopula", "MertonModel", "CDSCurve", "FittedVolSurface", "BlackLittermanPosterior")
if __import__("importlib.util").util.find_spec("pymetrix_finance") is None:
    for _name in FINANCE:
        BUILDERS.pop(_name, None)


def build(name: str) -> Any:
    return BUILDERS[name]()


@functools.cache
def every_object() -> dict[str, Any]:
    """One of each gallery kind, then each demo project's objects of other kinds, by kind: built once per test process.

    test_result_views.py and test_saved_results.py check the same objects and only read them (views, text, saving),
    so they share one set; building it takes some 30 s. Callers take a copy of the dict, never change an object."""
    from pymetrix import demo

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from small examples
        found = {name: build(name) for name in BUILDERS}
        for project in ("consumption", "forecast_round", "nk_estimation", "policy_and_nonlinear"):
            for obj in demo(project).objects.values():
                found.setdefault(type(obj).__name__, obj)
    return found
