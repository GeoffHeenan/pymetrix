---
summary: test for unit roots and cointegration in 15 to 30 annual observations, with critical values simulated for the sample.
audience: applied macroeconomists with annual data, and EViews users used to MacKinnon's values
order: 3
---

# Test for cointegration in a short annual sample

Test whether two series that each have a unit root move together in the long run (Engle-Granger), when the sample
is short: 25 annual observations here. MacKinnon's critical values, which most packages report, are asymptotic, and
in samples this short, with a lag search, they are too lenient. pymetrix can simulate the critical values and the
p-value for the sample at hand instead.

## Before you start

- An annual dataset. This page converts the example `sample_macro` (quarterly, 2000Q1 to 2024Q4) to annual: 25
  years, 2000 to 2024.
- The series in logs, as new series: `lc` (log consumption) and `ly` (log output). The test's settings take series
  names, not formulas.
- An idea of the deterministic terms. Both series trend up, so the levels regression here has a constant and a
  trend.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data.
2. Data → Transform → Convert frequency…, with Data: data, New frequency: Annual and Save as: annual. Combine
   periods by: By measure (each series' metadata) sums the quarters of these flows.
3. Data → Derived series…, with Data: annual, New series name: lc and Formula: `log(c)`. Again with New series name:
   ly and Formula: `log(y)`.
4. Test → Run test…, with Object to test: annual and Test: adf. Under Test settings: Series: ly, Deterministic
   terms: Constant and trend, Critical values: Simulated for this sample; under Advanced settings, Maximum lags: 1.
   Save as: adf_ly. The same for lc.
5. Test → Run test… again, with Test: engle_granger. Under Test settings: Series: lc, ly, Deterministic terms:
   Constant and trend, Critical values: Simulated for this sample; under Advanced settings, Maximum lags: 1. Save
   as: eg.
6. To see what MacKinnon's values say, run it once more with Critical values: MacKinnon (asymptotic) and Save as:
   eg_mackinnon.

Draws (2,000) and Seed (0) are under Advanced settings too: the same seed gives the same critical values.

## In code

```python
from pymetrix import Dataset

annual = Dataset.example("sample_macro").to_freq("Y")         # 25 years, 2000 to 2024
annual = annual.derive(lc="log(c)", ly="log(y)")

adf_ly = annual.test("adf", "ly", trend="ct", max_lags=1, critical_values="simulated")
adf_lc = annual.test("adf", "lc", trend="ct", max_lags=1, critical_values="simulated")
adf_ly.p_value                                                 # 0.87: a unit root is not rejected

eg = annual.test("engle_granger", "lc, ly", trend="ct", max_lags=1, critical_values="simulated")
eg_mackinnon = annual.test("engle_granger", "lc, ly", trend="ct", max_lags=1)
eg.statistic, eg.p_value                                       # -5.40 and 0.007
eg.critical_values["5%"], eg_mackinnon.critical_values["5%"]  # -4.48 simulated, -4.20 MacKinnon's
```

## Check the result

- The table's note: "simulated for this sample: 2000 draws of 2 independent Gaussian random walks of 25
  observations, trend ct". The length should be your sample's; the draws are tested as the data are, with the same
  trend terms and lag rule.
- The critical values. Simulated for 25 years they are further out than MacKinnon's: -4.48 against -4.20 at 5% for
  Engle-Granger, -3.96 against -3.61 for ADF on `ly`. A statistic between the two is a rejection by MacKinnon's
  values only.
- The p-value and its Monte Carlo standard error (p-value MCSE): 0.007 and 0.002 here. Near a significance level,
  raise Draws (10,000 take under a second); the standard error falls as one over the square root of the draws.
- The other normalisation, `"ly, lc"` (-5.29, p-value 0.009 here): Engle-Granger can disagree with itself when the
  order changes.
- `eg.provenance` keeps the seed, the draws and the random generator.

## Pitfalls

- **The lag search runs away in short samples.** Left to itself the residuals' ADF regression searches up to 9 lags
  in 25 observations, leaving few degrees of freedom; with 20 it leaves none and the statistic is zero (the result
  says so). Set Maximum lags to 1 or 2 for annual data.
- **Trend terms change the critical values.** Constant and trend moves them further out than Constant alone; choose
  the terms from what the series do under the alternative, not from which rejects.
- **A break looks like no cointegration.** A level shift in the relation makes Engle-Granger fail to reject. Put a
  shift dummy in Exogenous series (under Advanced settings, with simulated critical values only), or use Test →
  Gregory-Hansen cointegration test….
- **Engle-Granger finds one relation at most.** With three or more series it cannot count the relations: use Test →
  Johansen cointegration…, whose critical values are asymptotic too.
- **Simulated is not exact.** The draws are Gaussian random walks; the p-value is a Monte Carlo estimate with the
  standard error shown.

## See also

- [Unit roots and stationarity](pymetrix:topic/unit-roots) and [Cointegration and VECMs](pymetrix:topic/cointegration),
  in the help.
- [The user guide's Critical values for short samples](../../../docs/guide.md#critical-values-for-short-samples):
  the simulation, and `exog=` for impulse and break dummies.
- [Estimate an equation and test its residuals](estimate-and-test-an-equation.md): the error-correction equation
  for these series.
- Next, estimate the relation: Estimate → VARs and local projections → Cointegrating regression… (FMOLS, DOLS), or
  Estimate → Time series → ARDL and bounds test….
