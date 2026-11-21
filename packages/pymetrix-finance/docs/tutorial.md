# Finance with pymetrix: a tutorial

This page walks through the finance plugin (`pymetrix-finance`, bundled with pymetrix) on its example datasets:
returns and performance, factor models, portfolios and their risk, value at risk, volatility from intraday data,
event studies, yield curves and options. Each step is also a dialog in the app's **Finance** menu, and the app
writes it as code in the **Script** window. The [user guide](../../../docs/guide.md) has the detail on each function, and the
plugin's help topics (in Help → Help contents) the background and references.

The code blocks run in order, as one script; a test runs them as written, so they stay correct. The same page is a
notebook, `docs/finance-tutorial.ipynb`. Two datasets are
real: the Fama-French returns (Kenneth French's data library) and US Treasury yields (the Federal Reserve's H.15, from
FRED). The rest are simulated, with the true values in each dataset's description, so you can see the methods recover
them. The Fama-French returns are the one download: the library's files are Eugene Fama's and Kenneth French's
copyright, so pymetrix does not ship them, and the first `Dataset.example("french")` fetches them from the library
(an internet connection is needed once) and keeps them in `~/.pymetrix/cache/datasets/`. The library revises its
history now and then, so your numbers may differ slightly from the ones described here. Returns are fractions per
period (0.01 for 1%) throughout.

## 1. Returns and performance

```python
import numpy as np
from pymetrix import Dataset, performance

ff = Dataset.example("french")                       # monthly, 1949-2017: factors, industries, sorted portfolios (downloaded once)
print(ff.meta.description)

perf = performance(ff, ["mkt_rf", "smb", "hml", "mom"], kind="simple")
perf.table()                                         # annual return and volatility, Sharpe (with its standard error), drawdowns...
```

`perf.plot()` draws the growth of 1 invested and the drawdowns. Momentum has the highest Sharpe ratio here, and by
far the fattest left tail: compare its skewness and kurtosis with the market's. With prices rather than returns,
`returns(data, ["spx"])` adds log or simple returns first.

## 2. A factor model

```python
from pymetrix import factor_model

industries = ["nodur", "durbl", "manuf", "enrgy", "hlth", "money"]
fm = factor_model(ff, industries, ["mkt_rf", "smb", "hml"], rf="rf")
fm.summary()                                         # alphas, betas with Newey-West t statistics, R2, the GRS test
p_grs = fm.test[1]
```

Each industry's excess return is regressed on the three Fama-French factors. The Gibbons-Ross-Shanken test asks
whether all six alphas are zero together; here it rejects, largely on health care's positive alpha.
`fm.rolling_betas(60)` shows how the exposures moved over five-year windows.

## 3. A portfolio and where its risk comes from

```python
from pymetrix import optimize_portfolio, risk_contributions

rf = float(ff["rf"].mean())
port = optimize_portfolio(ff, industries, objective="max_sharpe", rf=rf)    # long only, by default
port.weights_table()
frontier = port.frontier(20)                        # the efficient frontier within the same bounds

rc = risk_contributions(ff, industries, port.weights, level=0.99)
rc.table()                                           # marginal and component VaR and ES; the components add up
assert abs(rc.frame()["component VaR"].sum() - rc.var) < 1e-12
```

Non-durables carry the largest weight and the largest share of the portfolio's risk. `objective="min_variance"`
or `"target_return"` (with `target_return=`) give the other classic portfolios, and `cov=` takes a covariance matrix or a
fitted multivariate GARCH.

### Would it have worked?

```python
from pymetrix import backtest_strategy

bt = backtest_strategy(ff, industries, ["equal_weight", "min_variance", "max_sharpe", "risk_parity"],
                       window=60, rebalance=12, cost=0.002, rf="rf")
bt.performance()                                     # out of sample, net of 20 basis points per unit of turnover
```

Each year the weights are re-estimated from the five years before. Out of sample, minimum variance has the best
Sharpe ratio and maximum Sharpe the worst, with the most turnover: estimated means are too noisy to optimise on.

## 4. Value at risk, and checking it

```python
from pymetrix import value_at_risk

risk = value_at_risk(ff, "mkt_rf", level=0.99, method="garch", window=120)
risk.backtests()                                     # Kupiec, Christoffersen, dynamic quantile, McNeil-Frey, Basel zone

fat = value_at_risk(ff, "mkt_rf", level=0.99, method="garch", window=120, dist="t", asymmetric=True)
cf = value_at_risk(ff, "mkt_rf", level=0.99, method="cornish_fisher", window=120)
counts = {"normal GARCH": risk.violations, "t-GJR GARCH": fat.violations, "Cornish-Fisher": cf.violations}
```

Each month's VaR uses only the months before it; about 7 of the 699 months should exceed a 99% VaR. A normal
GARCH(1,1) is too thin-tailed for monthly market returns: its violations are nearly three times that, and the
backtests say so. Student-t errors with the GJR asymmetry help, but not enough, as the market's monthly crashes are
fatter-tailed still; Cornish-Fisher, which adapts to the window's skewness and kurtosis, passes the Kupiec test.

## 5. Volatility from intraday returns

```python
from pymetrix import har, realized_measures

intraday = Dataset.example("intraday_returns")      # 250 days of 78 five-minute returns, with ten jumps
daily = realized_measures(intraday, "r")            # one row a day: rv, bv, the jump test, rvol
jump_days = daily.frame.index[daily.frame["jump"] > 0]
print(len(jump_days), "jump days found;", intraday.meta.description.split("ten days: ")[1])

fit = har(daily, "rv", log=True)                    # tomorrow's realised variance from its daily, weekly, monthly averages
fit.summary()
tomorrow = fit.value_at_risk(0.99)                  # a one-day VaR from the HAR's variance forecast
```

Bipower variation measures each day's variance without its jumps, realised variance with them; the
Barndorff-Nielsen-Shephard test compares the two and finds the planted jump days.

## 6. An event study

```python
from pymetrix import abnormal_returns

er = Dataset.example("event_returns")               # 20 firms and the market; each firm has one event worth +2%
listed = er.meta.description.split("on the day: ")[1]
events = {e.split()[0]: e.split()[1] for e in listed.split("; ")}

ev = abnormal_returns(er, events, market="mkt", car_window=(0, 0))
ev.tests()                                           # cross-sectional t, Patell, BMP and rank tests
mean_car = float(ev.car().mean())                    # close to the true 0.02
```

Normal returns come from each firm's market model over trading days -250 to -11. `ev.plot()` shows the cumulative
average abnormal return around the event, and `car_window=(-1, 1)` tests a three-day window.

## 7. Yield curves

```python
from pymetrix import yield_curve

treasury = Dataset.example("treasury_yields")       # US Treasury constant-maturity yields, monthly, from FRED
names = ["m3", "m6", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]
years = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
yc = yield_curve(treasury, names, years)
yc.table()                                           # the fit at each maturity (a few basis points), and the decay
ahead = yc.forecast(12)                              # a year ahead from an AR(1) for each factor (Diebold-Li)
sv = yield_curve(treasury, names, years, model="svensson")   # a second hump fits the short end better

simulated = yield_curve(Dataset.example("yield_curves"), names, years)   # made with decay 0.7308
assert abs(simulated.lam - 0.7308) < 0.05
```

Yields are in percent. The 20-year series is missing from 1987 to 1993 and the 30-year from 2002 to 2006, when the
Treasury did not issue them; each month is fitted on the maturities it has. On the simulated curves the decay comes
out close to the true 0.7308 and the AR(1) coefficients near the true 0.99, 0.95 and 0.8. `yc.factors` is an
ordinary table: put it into a VAR for the factors' joint dynamics.

## 8. Options

```python
from pymetrix import implied_volatility, option_price

put = option_price(spot=100, strike=95, expiry=0.5, rate=0.03, vol=0.2, dividend=0.01, kind="put", american=True)
put.table()                                          # the price and Greeks, from a binomial tree

chain = Dataset.example("option_chain")             # out-of-the-money puts and calls at five expiries
smile = implied_volatility(chain, "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, kind="cp")
error = np.nanmax(np.abs(smile.frame["implied vol"].to_numpy() - chain["true_vol"].to_numpy()))
assert error < 1e-3                                  # the volatilities the prices were made with, to the prices' rounding
```

`smile.plot()` draws the skew, one line per expiry: puts below the spot are dearer in volatility than calls above
it, as in equity index markets.

```python
from pymetrix import vol_surface

surf = vol_surface(chain, "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, kind="cp", model="svi")
surf.arbitrage()                                    # butterfly and calendar checks, expiry by expiry
heston = vol_surface(chain, "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, kind="cp", model="heston")
```

SVI fits each expiry's smile to within a few hundredths of a volatility point and passes the arbitrage checks over the
quotes. Heston, with one set of parameters for every expiry, cannot: its smiles flatten as expiry grows and these do
not, and its summary says its parameters have reached their bounds. `surf.plot()` shows market and model smiles.
