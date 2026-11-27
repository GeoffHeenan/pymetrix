---
summary: re-estimate a model at each date in the past, forecast ahead from there, measure the errors by horizon, and test whether another model forecast better.
audience: forecasters in an institution
order: 12
---

# Check a forecast's record (backtest)

Find out how well a model would have forecast in the past: re-estimate it on the data up to each date (each origin),
forecast up to a year ahead from there, and compare the forecasts with what happened. The errors by horizon are the
model's record; a test then says whether another model's record is better by more than chance.

## Before you start

- The models to judge, estimated. This page uses the VAR in output and investment growth of [Estimate a VAR, then
  impulse responses and a forecast](estimate-a-var.md), with one lag, and a simpler benchmark: an AR(1) for output
  growth alone, `dly ~ dly[-1]`. Both on the example `sample_macro`.
- The origins: the dates each forecast is made from. Here every quarter from 2015Q1 to 2022Q4, 32 origins, each
  forecasting up to 4 quarters ahead; the last forecasts, made in 2022Q4, reach 2023Q4, inside the data.
- Enough origins for the comparison to mean something: a few dozen at least.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data. Data → Derived series…, with
   Data: data, New series name: dly and Formula: `100*dlog(y)`; again with New series name: dli and Formula:
   `100*dlog(i)`.
2. Estimate → VARs and local projections → VAR…, with Series: dly, dli, Lags: 1 and Save as: var. Estimate →
   Equation…, with Formula: `dly ~ dly[-1]` and Save as: ar.
3. Forecast → Backtest…, with Estimated model: var, Start: 2015Q1, End: 2022Q4, Longest horizon: 4 and Save as:
   bt_var. The model is re-estimated at each origin from Start to End on the data up to it, and forecast 1 to 4
   quarters ahead. Again with Estimated model: ar and Save as: bt_ar.
4. In the backtest's window: Accuracy (the RMSE, MAE and mean error for each variable and horizon), Graph, Tests,
   and Forecasts (each forecast, by its origin and target).
5. Forecast → Compare backtests…, with First backtest: bt_var, Second backtest: bt_ar, Variable forecast: dly,
   Horizon: 1, Test: Diebold-Mariano, Loss: Squared and Save as: dm.

## In code

```python
from pymetrix import VAR, Dataset, Equation, backtest

data = Dataset.example("sample_macro").derive(dly="100*dlog(y)", dli="100*dlog(i)")
var = VAR(["dly", "dli"], lags=1).fit(data)
ar = Equation("dly ~ dly[-1]").fit(data)
bt_var = backtest(var, data, ("2015Q1", "2022Q4"), horizon=4, label="var")
bt_ar = backtest(ar, data, ("2015Q1", "2022Q4"), horizon=4, label="ar")
bt_var.accuracy()                         # dly: RMSE 0.833 one quarter ahead, mean error -0.25
bt_ar.accuracy()                          # dly: RMSE 0.829
dm = bt_var.compare(bt_ar, "dly", 1)      # Diebold-Mariano (HLN): statistic 0.37, p-value 0.71
bt_var.efficiency("dly", 1)               # Mincer-Zarnowitz: p-value 0.19
```

## Check the result

- The counts: 32 forecasts at each horizon (the n column), one from each origin.
- The errors are the outcome less the forecast (`bt_ar.errors`): a mean error of -0.25 says both models forecast
  output growth too high on average over these years. A mean error far from zero is a bias, and the Mincer-Zarnowitz
  test (`efficiency`) asks whether the forecasts could be improved by shifting and scaling them; at 0.19 it does not
  reject.
- The AR(1)'s one-quarter-ahead forecasts by hand: at each origin, least squares of `dly` on its lag over the data to
  that origin, then the forecast from the last value. Their root mean squared error is 0.829.
- The comparison: a statistic of 0.37 and a p-value of 0.71. The VAR's squared errors are a little larger, by far
  less than chance would give: investment growth does not help forecast output growth here.

## Pitfalls

- **The data are today's, not as they were.** A backtest on the latest release uses revised data the forecaster did
  not have, and so flatters the model. Give Vintages (under Advanced settings): for each origin, the dataset as it
  was then (`origin=dataset`), for a real-time record.
- **The model is re-estimated, not its specification chosen again.** The lags, the variables and any judgement are
  those of the model given; if they were chosen by looking at the whole sample, the record is better than a real
  forecaster's.
- **Few origins, little power.** With 32 origins, and overlapping errors at longer horizons, the tests reject only
  large differences. Not rejecting does not make two models equally good.
- **Nested models.** The AR(1) is nested in the VAR; Diebold-Mariano is then undersized, and Clark-West (Test, in
  Compare backtests…) is the test to use.
- **A rolling window adapts to change, at a cost.** Window (0 = expanding) under Advanced settings re-estimates on
  the last periods only: better after a break, noisier otherwise.

## See also

- [The forecast workflow](pymetrix:topic/forecast-workflow), in the help.
- [Run a forecast round: baseline, judgement, scenarios, report](run-a-forecast-round.md): the forecasts made in
  real time, kept in a store, are a record too (`ForecastStore.accuracy`).
- [The user guide's Forecast rounds and backtests](../../../docs/guide.md#forecast-rounds-and-backtests) and
  [Evaluating forecasts](../../../docs/guide.md#evaluating-forecasts): density scores, PIT, interval tests,
  combinations and superior predictive ability.
