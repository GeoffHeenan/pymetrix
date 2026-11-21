---
title: Does the Nile's flow have a unit root?
level: beginner
minutes: 10
data: [nile]
---

The annual flow of the Nile at Aswan, 1871 to 1970: the augmented Dickey-Fuller test of a unit root.

## Step 1: Test around a constant level

Choose Test → Run test…, with Object to test: nile and Test: adf; under Test settings, Variable: volume; and
Save as: adf. The lags are chosen by AIC.

```python
adf = nile.test("adf", "volume")
```

> Hint: the null hypothesis is a unit root; a statistic below the critical value rejects it.

## Step 2: Allow for a trend

Run the test again with Trend: ct (a constant and a trend) under Test settings, and Save as: adf_trend.

```python
adf_trend = nile.test("adf", "volume", trend="ct")
```
