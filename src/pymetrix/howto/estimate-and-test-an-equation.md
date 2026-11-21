---
summary: fit one equation by least squares, read the tests of its residuals, and what to do when one fails.
audience: newcomers, and EViews users looking for the residual views
order: 1
---

# Estimate an equation and test its residuals

Estimate one equation by least squares, then check its residuals: no serial correlation left, a constant variance,
no gross departure from normality, no missing nonlinearity. When a test fails, change the equation, or keep it and
make its standard errors robust to the failure.

## Before you start

- A dataset in the workspace. This page uses the example `sample_macro`: a simulated quarterly economy, 2000Q1 to
  2024Q4, with consumption `c`, investment `i`, government spending `g` and output `y`.
- The equation, written with `~`: `dlog(c) ~ dlog(y) + dlog(c)[-1]`. A constant is included unless you write `- 1`;
  `x[-1]` is a lag and `dlog(x)` the change in its log.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data.
2. Estimate → Equation…, with Formula: `dlog(c) ~ dlog(y) + dlog(c)[-1]`, Method: OLS and Save as: fit.
3. In the result's window, choose Diagnostics in the list of views on the left. It has Breusch-Godfrey (serial
   correlation, 4 lags), Breusch-Pagan (heteroskedasticity), Jarque-Bera (normality) and RESET (omitted
   nonlinearity), each with its statistic, p-value and null hypothesis. The Correlogram view gives the residuals'
   autocorrelations by lag; the Q-Q plot their departure from the normal.
4. For a test with other settings, or one the Diagnostics view does not run (ARCH, White, Chow, CUSUM): Test → Run
   test…, or Next ▸ → Test… in the result's window. Object to test: fit; Test: breusch_godfrey; under Test settings,
   Lags: 4; Save as: bg.
5. Breusch-Godfrey rejects (p-value 0.0001): the residuals are serially correlated, and the Correlogram's first
   autocorrelation is -0.40. Change the equation: Estimate → Equation… again, with Formula:
   `dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]` and Save as: ecm. The lagged levels make it an error-correction
   equation. In its Diagnostics view Breusch-Godfrey no longer rejects (0.78), nor do Jarque-Bera (0.053) and RESET;
   Breusch-Pagan does (0.0007): the errors' variance is not constant.
6. Keep the coefficients and make the standard errors robust to it: estimate the same equation with Standard
   errors: White (HC1), under Sample and covariance, and Save as: ecm_white. (HAC (Newey-West) allows for serial
   correlation too.)

View → Script shows each step as the code below.

## In code

```python
from pymetrix import OLS, Dataset, Equation

data = Dataset.example("sample_macro")
fit = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]").fit(data)
fit.diagnostics()                                      # Breusch-Godfrey, Breusch-Pagan, Jarque-Bera, RESET
bg = fit.test("breusch_godfrey", lags=4)
bg.statistic, bg.p_value                               # 23.46 and 0.0001: serial correlation remains
fit.correlogram(lags=8)                                # the first autocorrelation is -0.40
```

Change the equation, then the standard errors:

```python
ecm = Equation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]").fit(data)
ecm.diagnostics()                                      # Breusch-Godfrey 0.78, Jarque-Bera 0.053; Breusch-Pagan 0.0007
ecm.test("breusch_pagan").reject(0.05)                 # True: heteroskedastic
ecm_white = Equation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]").fit(data, method=OLS(cov="white"))
```

## Check the result

- The Summary's Sample line: 2000Q3 to 2024Q4, 98 observations for `fit` (the difference of a lag takes two
  quarters), 2000Q2 to 2024Q4, 99 for `ecm`. More missing than that means gaps in the data.
- Each test's null hypothesis, in the last column of the Diagnostics view. A small p-value rejects it: for
  Breusch-Godfrey, Breusch-Pagan and Jarque-Bera that is the bad news.
- `ecm` and `ecm_white` have the same coefficients (-0.75 on the lagged log of consumption, 0.72 on that of output:
  close to equal and opposite, a long-run ratio). The White standard errors are larger: 0.137 against 0.097 for the
  first. The Summary's Method line says which standard errors it shows.
- `bg.reject(0.05)` is True; for `ecm`'s Breusch-Godfrey it is False.

## Pitfalls

- **The Diagnostics view runs each test at its defaults.** Breusch-Godfrey takes 4 lags whatever the frequency: for
  annual data use 1 or 2, for monthly 12, with Test → Run test….
- **Robust standard errors fix the standard errors, not the equation.** With a lagged dependent variable among the
  regressors, as in `fit`, serially correlated errors make least squares inconsistent: the coefficients are wrong,
  however the standard errors are computed (HAC included). Change the dynamics first.
- **Durbin-Watson is in the Summary, but not valid with a lagged dependent variable.** Use Breusch-Godfrey.
- **Jarque-Bera rejects for a few large residuals.** Look at the Residuals and Q-Q plot views before acting; with
  enough observations non-normal errors leave the estimates and their standard errors valid.
- **Not rejecting is not proof.** With few observations the tests have little power.
- **Numbers differ from another package's.** The information criteria are totals, not divided by the number of
  observations as in EViews; [Defaults, and other packages' defaults](pymetrix:topic/defaults) lists the others.

## See also

- [Least squares](pymetrix:topic/least-squares) and
  [Specification and diagnostic tests](pymetrix:topic/diagnostic-tests), in the help.
- [Test for cointegration in a short annual sample](cointegration-in-a-short-sample.md): whether the levels in the
  error-correction equation belong together.
- [The user guide's Working with results](../../../docs/guide.md#working-with-results): every number a fit and a
  test hold, in code.
- [Coming from EViews](pymetrix:topic/coming-from-eviews): where EViews' views and commands are.
