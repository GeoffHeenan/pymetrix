# pymetrix

**Applied macroeconometrics in Python, from data to forecast report.** Estimate equations and systems, test
them, solve models, run scenarios, manage forecast rounds and export reports, from Python scripts, Jupyter
notebooks or a desktop app with guided forms. Every action becomes Python you can read and rerun.

> **Status: early pre-release.** The source is at github.com/GeoffHeenan/pymetrix (BSD-3-Clause); hosted
> documentation will be published with the first 0.1 release. Interfaces may still change.

## Install

This is a pre-release, so pip installs it only when asked. It needs Python 3.11 or later.

```bash
pip install --pre pymetrix            # the core: estimation, tests, models, forecasting, projects
pip install --pre "pymetrix[all]"     # with the desktop app, panel and volatility models, and report writers
```

Then `pymetrix-gui` opens the desktop app.

## What it does

- **Data:** time series and panels with metadata, vintages and transformations; seasonal adjustment; filters.
- **Estimation:** least squares and IV, GMM, limited dependent variables, ARIMA and exponential smoothing,
  VARs and VECMs, Bayesian VARs, state-space and Markov-switching models, panel estimators, treatment effects.
- **Testing:** unit roots and cointegration, diagnostics, structural breaks, forecast comparison.
- **Models:** solve and estimate DSGE models, import Dynare `.mod` files, scenarios and optimal policy.
- **Forecasting:** forecast rounds with judgement, backtests in real time, combinations.
- **Reproducible projects:** every dialog records the call it runs; a project saves the data and the steps and
  rebuilds the results when opened, and each result records its sample, method and data vintage.

Wherever a mature library exists (statsmodels, linearmodels, arch, SciPy), pymetrix uses it, and its own code
is checked against libraries, published tables and hand formulas.

## Licence

BSD-3-Clause. The bundled example datasets from elsewhere are listed with their sources and licences in
`THIRD-PARTY-NOTICES.md`, included in the package.
