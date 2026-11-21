---
title: A wage equation with robust standard errors
level: beginner
minutes: 15
data: [mroz]
---

# A wage equation with robust standard errors

Married women's wages in 1975 (Mroz 1987): the return to a year of education, with standard errors that allow
the errors' variance to differ between women.

## Step 1: Fit the wage equation

Choose Estimate → Equation…, with Data: mroz, Formula: `lwage ~ educ + exper + expersq`, Method: OLS and
Save as: fit. Only the 428 women in the labour force have a wage; the others are left out.

```python
fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)
```

> Hint: `lwage` is the log of the hourly wage, so the coefficient on `educ` is close to the proportional return
> to a year of education.

> Hint: `expersq` is experience squared: wages rise with experience, at a falling rate.

## Step 2: Make the standard errors robust

Estimate the same equation again with Standard errors: White (HC1), under Sample and covariance, and
Save as: fit_white. The coefficients are the same; the standard errors change.

```python
fit_white = Equation("lwage ~ educ + exper + expersq").fit(mroz, method=OLS(cov="white"))
```

> Hint: the Summary's Method line says which standard errors it shows.

## What next

Compare the two standard errors on `educ`: 0.0141 and 0.0132. Here robustness changes little.
