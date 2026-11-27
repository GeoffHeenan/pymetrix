---
summary: estimate a vector autoregression, choose its lags, read its impulse responses with bands, and forecast from it.
audience: EViews users, and IRIS or Dynare users who want a reduced-form VAR beside their model
order: 3
---

# Estimate a VAR, then impulse responses and a forecast

Estimate a vector autoregression in output and investment growth, with the lag length chosen by an information
criterion; trace how each variable responds to a shock in the other, with bootstrap bands; and forecast both two
years ahead. The VAR is the reduced form: its responses depend on how the shocks are identified, here by the
Cholesky ordering.

## Before you start

- A dataset in the workspace. This page uses the example `sample_macro` (quarterly, 2000Q1 to 2024Q4) and two new
  series: `dly`, output growth, and `dli`, investment growth, each `100*dlog(...)`: per cent a quarter.
- Stationary series. A VAR in growth rates, as here, or in levels with a cointegration test first (Test → Johansen
  cointegration…, then Estimate → VARs and local projections → VECM…).
- The ordering of the variables. With Cholesky identification the first variable responds on impact only to its own
  shock, the last to every shock. Output first, then investment, says investment can react within the quarter to
  output, and not the other way round.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data.
2. Data → Derived series…, with Data: data, New series name: dly and Formula: `100*dlog(y)`. Again with New series
   name: dli and Formula: `100*dlog(i)`.
3. Estimate → VARs and local projections → VAR…, with Series: dly, dli (ticked in that order: the order is the
   Cholesky ordering), Lags: bic, Deterministic terms: Constant and Save as: var. Lags takes a number, or a criterion
   (aic, bic, hq, fpe) to choose it up to Maximum lags (8, under Advanced settings). BIC chooses one lag.
4. In the VAR's window: Summary (the coefficients, equation by equation, with the information criteria), Roots
   (every root inside the unit circle: the VAR is stable), Residuals and Residual correlogram.
5. Estimate → VARs and local projections → Impulse responses…, with Estimated VAR: var, Horizon: 12,
   Identification: Cholesky and Save as: irf. The Graph view has a panel for each response, `dli <- dly` for
   investment's response to an output shock, with 95% bootstrap bands (Draws or replications: 500, under Advanced
   settings).
6. Forecast → Forecast…, with Estimated model: var, Forecast start: 2025Q1, Forecast end: 2026Q4 and Save as: fc.
   A VAR's forecasts are dynamic: each period uses the forecasts before it. The Graph view shows them after the
   data, with 95% intervals.

## In code

```python
from pymetrix import Dataset, VAR

data = Dataset.example("sample_macro").derive(dly="100*dlog(y)", dli="100*dlog(i)")
var = VAR(["dly", "dli"], lags="bic").fit(data)       # VAR(1): BIC's choice; 98 observations, 2000Q3 to 2024Q4
print(var.summary())
var.is_stable                                          # True: the largest root is 0.33
irf = var.irf(periods=12)                              # Cholesky, in the order dly, dli
irf.to_frame()["dli <- dly"].iloc[0]                   # 0.64: investment's response on impact to an output shock
fc = var.forecast(data, ("2025Q1", "2026Q4"))
fc.to_frame()["dly"].iloc[0]                           # 0.63 per cent in 2025Q1, then towards 0.87
irf.plot()
fc.plot()
```

## Check the result

- The Summary's Sample line: 2000Q3 to 2024Q4, 98 observations. The growth rates start in 2000Q2, and one lag takes
  a quarter more.
- The coefficients against statsmodels' or EViews's VAR on the same series and lags: they are least squares,
  equation by equation, and should agree to the last digit. EViews's `var v.ls 1 1 dly dli` is this VAR.
- The impulse responses on impact: they are the Cholesky factor of the residual covariance (with its degrees of
  freedom correction), so `dly <- dly` is output's residual standard deviation, 1.07, and `dly <- dli` is zero by the
  ordering.
- The forecast's path: a stable VAR's forecasts go to its mean, here 0.87 per cent a quarter for output growth, close
  to the sample's 0.88.

## Pitfalls

- **The responses depend on the ordering.** Reorder the variables (Ordering, in the Impulse responses… dialog) and
  the Cholesky responses change, unless the residuals are uncorrelated. A conclusion that changes with the ordering
  rests on the ordering, not the data; use sign restrictions or Generalised (Pesaran-Shin) responses to see.
- **Criteria choose different lags.** AIC tends to choose more than BIC. Check the residuals (Residual correlogram)
  with the lags chosen; too few lags leave serial correlation, which biases every response.
- **The forecast intervals are too narrow in small samples.** A least-squares VAR's intervals allow for the shocks,
  not for the uncertainty about the coefficients. Forecast → Backtest… shows how far off the forecasts were in the
  past.
- **One unit or one standard deviation.** Orthogonalised responses (the default) are to a one-standard-deviation
  shock; untick Orthogonalised shocks (under Advanced settings) for a one-unit shock in a residual.
- **A VAR in levels of trending series.** Its responses need not die out, and bands from the bootstrap can be poor.
  Difference the series, or test for cointegration and estimate a VECM.

## See also

- [Vector autoregressions](pymetrix:topic/var) and [Structural VARs](pymetrix:topic/structural-var), in the help.
- [Check a forecast's record (backtest)](check-a-forecasts-record.md): this VAR's forecasts made at each date in
  the past and compared with what happened.
- [Run a forecast round: baseline, judgement, scenarios, report](run-a-forecast-round.md): its forecast as a
  baseline.
- [The user guide's Sign restrictions](../../../docs/guide.md#sign-restrictions) and
  [Bayesian VARs](../../../docs/guide.md#bayesian-vars): other identifications, and shrinkage.
