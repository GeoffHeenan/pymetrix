---
summary: remove the seasonal pattern from a quarterly or monthly series with STL, read the seasonal factors, and keep the adjustment up to date.
audience: newcomers, and EViews users of X-13 and TRAMO-SEATS
order: 8
---

# Seasonally adjust a series

Take a quarterly series with a seasonal pattern, remove the pattern, and read what was removed: the seasonal factor
of each quarter. The adjusted series is kept up to date, so when the data are revised the adjustment is redone.
STL, in pymetrix itself, is used here; X-13ARIMA-SEATS and TRAMO-SEATS come with a plugin and take the same steps.

## Before you start

- A seasonal series. This page uses six years of quarterly sales, a made-up series that rises about 0.8% a
  quarter, is low in the first quarter and high in the fourth, as `sales.csv`:

  ```text sales.csv
  quarter,sales
  2018Q1,94.0
  2018Q2,101.0
  2018Q3,99.4
  2018Q4,110.0
  2019Q1,96.8
  2019Q2,103.4
  2019Q3,102.8
  2019Q4,115.1
  2020Q1,99.9
  2020Q2,107.0
  2020Q3,106.4
  2020Q4,118.1
  2021Q1,103.5
  2021Q2,110.3
  2021Q3,109.5
  2021Q4,122.2
  2022Q1,105.9
  2022Q2,114.2
  2022Q3,111.8
  2022Q4,124.7
  2023Q1,109.0
  2023Q2,118.0
  2023Q3,115.9
  2023Q4,129.9
  ```

  It was made from seasonal factors of 0.94, 1.00, 0.98 and 1.08 times a trend, with a little noise.
- Multiplicative or additive. A positive series whose seasonal swings grow with its level, as most do, is adjusted
  multiplicatively (a factor near 1); one that can be zero or negative, or whose swings stay the same size,
  additively (a component near 0). Model the log: Automatic chooses by the sign of the data.
- Three years at least, and more for a pattern that changes: STL estimates each quarter's factor from that quarter in
  the years around it.

## In the app

1. File → Load data…, with Data file: sales.csv, Time column: quarter, Frequency: Quarterly and Save as: data.
2. Data → Transform → Seasonal adjustment…, with Series: sales, Data: data and Method: STL. Under Method settings,
   Seasonal: 7 (the years each quarter's factor is smoothed over) and Robust ticked (a few outliers do not move the
   factors). Keep up to date stays ticked; leave Save as empty to add the new series to `data`.
3. Two series are added to `data`: `sales_sa`, the adjusted series, and `sales_seasonal`, the seasonal factor. The
   dataset's Variables tab says how each was made, "sales seasonally adjusted (STL(seasonal=7, robust))".
4. Data → Plot series…, with `sales` and `sales_sa`: the adjusted series rises smoothly, without the fourth-quarter
   peaks.

## In code

```python
from pymetrix import STL, Dataset, Equation

data = Dataset.read("sales.csv", time="quarter", freq="Q")
data = data.seasonal_adjust("sales")                    # STL, multiplicative: adds sales_sa and sales_seasonal
factors = data["sales_seasonal"]
factors.groupby(factors.index.quarter).mean()           # 0.939, 1.003, 0.981 and 1.085 by quarter
data["sales_sa"]["2023Q4"]                              # 119.2: 129.9 divided by 2023Q4's factor
steadier = data.seasonal_adjust("sales", method=STL(seasonal=11))   # factors smoothed over more years
left = Equation("dlog(sales_sa) ~ seasonal()").fit(data)
left.test("wald", "seasonal[Q2] = 0, seasonal[Q3] = 0, seasonal[Q4] = 0").p_value  # 0.47: no pattern left
```

## Check the result

- The factors by quarter: 0.939, 1.003, 0.981 and 1.085 on average, close to the pattern the series was made
  from (0.94, 1.00, 0.98 and 1.08). On real data
  look for a pattern that makes sense (retail high before Christmas) and moves slowly.
- `sales` divided by `sales_seasonal` is `sales_sa`, period by period; for an additive adjustment, `sales` less the
  component.
- The adjusted series has no seasonal pattern left: its quarter-on-quarter growth is about the same in every
  quarter. Estimate → Equation… with Formula: `dlog(sales_sa) ~ seasonal()`, then a Wald test of the seasonal
  dummies, does not reject (p-value 0.47); on `sales` itself it rejects at any level.
- Against another package: statsmodels' `STL(log(sales), period=4, seasonal=7, robust=True)` gives the same
  factors (exponentiated); X-13ARIMA-SEATS's differ a little, as it uses other filters.

## Pitfalls

- **The last years are revised.** A factor at the end of the sample is estimated from fewer years after it, so new
  data change the latest adjusted values. That is why Keep up to date recomputes the adjustment when the series
  changes; untick it to keep the adjustment as first made.
- **A break in the level is not seasonality.** A one-off shift, a strike or a lockdown, pulls the factors of its
  quarter; Robust limits that. X-13ARIMA-SEATS and TRAMO-SEATS (the `pymetrix-seasonal` plugin) find and model such
  outliers, and calendar effects such as Easter and trading days.
- **Adjust the components or the total, not both.** The adjusted components of a total need not add up to the
  adjusted total; adjust each and reconcile them (Data → Transform → Reconcile components…), or adjust the total
  alone.
- **Annual and monthly data.** Annual data have no seasons; monthly data take the same steps, with twelve factors.
- **Estimating with adjusted data.** Seasonal adjustment smooths the series and changes its dynamics; with
  unadjusted data, put `seasonal()` in the equation instead.

## See also

- [Seasonal adjustment](pymetrix:topic/seasonal-adjustment) and [Derived series](pymetrix:topic/derived-series), in
  the help.
- [The user guide's Seasonal adjustment](../../../docs/guide.md#seasonal-adjustment): X-13ARIMA-SEATS and
  TRAMO-SEATS settings, calendar regressors and seasonal dummies.
- [Reconcile national accounts components to a total](reconcile-components-to-a-total.md): when adjusted
  components must add up.
