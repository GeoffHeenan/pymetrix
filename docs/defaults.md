# Defaults, and other packages' defaults

<!-- Written by scripts/make_defaults_page.py from the in-app help page `defaults` (src/pymetrix/help.py): edit that, then rerun the script. -->

The same page is in the app: **Help → Help contents**, *Defaults, and other packages' defaults*.

Results that differ from another package's usually differ in a default, not in the
method. These are PyMetrix's defaults where packages are known to differ; set the
option shown to match. Where a comparison is not listed, check the other package's
documentation before comparing numbers.

Robust (White) standard errors: HC1, scaled by n/(n-k), as Stata's vce(robust).
Cluster-robust: the G/(G-1) x (n-1)/(n-k) small-sample factor, as Stata.

Newey-West (HAC): Bartlett kernel with floor(4(n/100)^(2/9)) lags, scaled by
n/(n-k); give hac_lags to fix the lag. Stata's newey has no default lag.

ADF unit-root test: lags chosen by AIC up to 12(n/100)^(1/4). EViews chooses by the
Schwarz criterion (lags="bic" matches it); Stata's dfuller uses no lags unless told.

VAR residual covariance: degrees-of-freedom adjusted (divided by T - k), as
statsmodels; impulse responses and their bands use it. Stata's var divides by T
unless dfk is given, so its responses are slightly smaller.

SUR and 3SLS: residual covariance divided by n, as Stata's sureg and reg3 without
dfk; df_correction=True divides by sqrt((n - k_i)(n - k_j)).

HP filter: lambda = 1600 (f/4)^2, that is 100, 1600 and 14,400 for annual,
quarterly and monthly data, as EViews; power=4 gives Ravn and Uhlig's rule.

Seasonal adjustment: STL by default (no external program); X13() uses the Census
Bureau's X-13ARIMA-SEATS, which statistical offices and EViews use; TramoSeats()
runs TRAMO-SEATS in Eurostat's JDemetra+ 3 (Java), the ESS's reference software.
Both are in the pymetrix-seasonal plugin.

Bayesian VAR: Minnesota prior with tightness 0.2 and lag decay 2, in conjugate
form, as in Giannone, Lenza and Primiceri; other packages' Minnesota priors differ
in their scaling, so compare hyperparameters with care.

Coefficient tests: t distributions for single-equation least squares; normal (z)
for systems, GMM and maximum likelihood. Intervals and bands are 95 per cent unless
stated (fan charts 30, 60 and 90).

## See also, in the app's help

- Least squares
- Unit roots and stationarity
- Vector autoregressions
- Systems of equations: SUR and 3SLS
- Trend-cycle filters
- Seasonal adjustment and calendar effects
- Bayesian VARs
