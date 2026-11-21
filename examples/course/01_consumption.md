---
title: Lecture 1. Estimating a consumption function
course: Example course (pymetrix)
links: [least-squares, instrumental-variables, methods/OLS, methods/IV, commands/Equation, commands/Run test]
references:
  - Davidson, J., Hendry, D., Srba, F. and Yeo, S. (1978). Econometric modelling of the aggregate time-series relationship between consumers' expenditure and income. Economic Journal 88.
---

# Lecture 1. Estimating a consumption function

These notes show how an instructor's material appears in the pymetrix help.
Every page this note links to (least squares, OLS, the Estimate equation and
Run test dialogs) lists it under **Course material**.

## The model

A simple Keynesian consumption function relates consumption to income:

    c ~ y

Load the example data (**File → Load example data**, `sample_macro`; in code,
`Dataset.example("sample_macro")`). In the **Estimate → Equation** dialog, type the
formula and choose **OLS**. Then use **Test → Run test** for
`breusch_godfrey` to check the residuals for serial correlation.

## Exercise

1. Add lagged consumption, `c ~ y + c[-1]`. What happens to the coefficient on income?
2. Re-estimate with `cov = hac`. Which standard errors change, and why?
3. Income includes consumption (`y = c + i + g`), so a shock to consumption moves income
   too, and OLS gives the income coefficient credit for it. Re-estimate by **IV**, with
   income endogenous and government spending and last quarter's income as instruments.
   The example economy was built with a marginal propensity to consume of 0.6 and 0.2 on
   lagged consumption: which estimates are closer, and why?

The worked solution is in the [example notebook](01_consumption.ipynb).

## Finding this in the help

Press **F1** in the Estimate dialog. Its help page lists this lecture under **Course
material**, beside the pages on least squares and the OLS method, with links to the papers
behind them. **Help → Help contents** searches every page and every note. Once you have
estimated something, **File → Export script** shows the same steps as Python, which is the
code in the example notebook.
