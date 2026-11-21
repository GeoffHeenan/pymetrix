# Canonical definitions

Plan item 4 · built 24 October 2026 · `pymetrix.estimation.criteria`

pymetrix computes every fit statistic with one function per definition, in `pymetrix.estimation.criteria`, so
a number called `aic` means the same on every result. Where a result follows another convention on purpose,
`criteria.definitions(result)` says which, and the Statistics view notes it. Where a library's value differs and
is kept for comparison, it is stored as `<key>_backend`.

## Likelihood and information criteria

| Key | Definition id | Formula | Same as |
| --- | --- | --- | --- |
| `loglike` (least squares) | `loglike.gaussian` | −n/2 · (ln 2π + ln(SSR/n) + 1) | statsmodels OLS `llf`, Stata `e(ll)`, R `logLik(lm)` |
| `loglike` (ML) | `loglike.exact` | the model's log likelihood at the estimates | statsmodels `llf`, arch `loglikelihood` |
| `k` | `k.estimated` | parameters reported in `params`: for least squares the coefficients; the error variance is concentrated out | statsmodels `df_model + k_constant`, Stata `e(rank)` |
| `aic` | `aic.total` | −2 · loglike + 2k | statsmodels, Stata `estat ic` |
| `bic` | `bic.total` | −2 · loglike + k · ln n | statsmodels, Stata |
| `hq`, `hq_total` | `hq.total` | −2 · loglike + 2k · ln ln n | |
| (none) | `aic.per_observation` | (−2 · loglike + 2k) / n | EViews, which also divides BIC and HQ by n |

**How other packages differ:**
- R's `AIC(lm)` counts the error variance (k + 1).
- EViews reports every criterion per observation.
- statsmodels' ARDL and ETS count the error variance too. pymetrix's `aic` and `bic` on those results now follow
  `aic.total` and `bic.total`, and statsmodels' values are kept as `aic_backend` (= `aic` + 2;
  `aic.k_plus_variance`) and `bic_backend` (= `bic` + ln n; `bic.k_plus_variance`). This was checked on the example data: ARDL −880.461 against statsmodels' −878.461; ETS
  534.929 against 536.929.
- arch's and Markov switching's values already equal `aic.total` (arch counts the variance parameters among its
  `params`).
- **Quantile regression and shrinkage** report a Gaussian quasi-likelihood of their residuals (`aic.gaussian_quasi`).
  It is useful only to compare fits of the same estimator.

## VARs

A VAR's `aic`, `bic`, `hq` and `fpe` are Lütkepohl's, as statsmodels' VAR, EViews and Stata's `varsoc` report
them for lag selection (decision D3):

- `aic` (`aic.lutkepohl`) = ln|Σ_ml| + 2 · K · m / T, with K variables and m regressors in each equation;
- `bic` and `hq` (`bic.lutkepohl`, `hq.lutkepohl`) replace 2 by ln T and 2 ln ln T;
- `fpe` (`fpe.lutkepohl`) = ((T + m) / (T − m))^K · |Σ_ml|.

`aic_total`, `bic_total` and `hq_total` are pymetrix's totals with k = K · m (the coefficients, not the
covariance), comparable with single equations. For a fixed T they rank lag orders identically:
aic_total / T = aic + K(1 + ln 2π).

## Goodness of fit

| Key | Definition id | Formula |
| --- | --- | --- |
| `r2` | `r2.centred` | 1 − SSR / Σ(y − ȳ)², weighted when weights are given; with a constant |
| `r2` | `r2.uncentred` | 1 − SSR / Σy²; without a constant |
| `r2_within` | `r2.within` | for fixed effects: the R² of the demeaned equation |
| `pseudo_r2` | `r2.pseudo_mcfadden` | 1 − loglike / loglike_null |
| `r2_adj` | `r2.adjusted` | 1 − (n − c)/df_resid · (1 − R²), c = 1 with a constant, 0 without |
| `df_resid` | `df_resid` | n − k, less the fixed effects absorbed |
| `se_regression` | `se_regression` | √(SSR / df_resid) |
| `durbin_watson` | `durbin_watson` | Σ(u_t − u_{t−1})² / Σu_t² |

## Covariance estimators (`cov=`)

| Setting | Estimator | Same as (checked in `tests/test_definitions.py`) |
| --- | --- | --- |
| `classic` | s² (X'X)⁻¹, s² = SSR / df_resid | statsmodels `OLS().fit()` |
| `white` | HC1: n/df_resid · (X'X)⁻¹ X'ΩX (X'X)⁻¹ | statsmodels `cov_type="HC1"`, Stata `robust`, EViews White |
| `hac` | Newey-West, Bartlett kernel, with n/df_resid; lags `hac_lags`, or ⌊4(n/100)^(2/9)⌋ (recorded in `stats["hac_lags"]`) | statsmodels `cov_type="HAC", use_correction=True` |
| `Cluster(g)` | G/(G−1) · (n−1)/(n−k) · sandwich over clusters | statsmodels `cov_type="cluster"`, Stata `vce(cluster g)` |
| `Cluster(g, h)` | Cameron-Gelbach-Miller: V_g + V_h − V_gh | statsmodels two-way cluster |

**Tests.** Wald tests use the χ² form by default; F forms are labelled where they are reported.

## Using the definitions

```python
from pymetrix.estimation.criteria import DEFINITIONS, definitions
definitions(fit)             # {"aic": "aic.total", "loglike": "loglike.gaussian", ...}
DEFINITIONS["aic.lutkepohl"].formula
```
