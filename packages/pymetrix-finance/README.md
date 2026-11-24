# pymetrix-finance

Financial econometrics for [pymetrix](https://github.com/GeoffHeenan/pymetrix), as a plugin: value at risk and expected shortfall with their
backtests, multivariate GARCH (CCC, DCC, asymmetric DCC, BEKK), copulas (Gaussian, t, Archimedean, BB1, BB6, BB7,
BB8, Tawn, rotated; fitted or rank margins; family comparison and goodness of fit by parametric or multiplier
bootstrap; R-, C- and D-vines with a Rosenblatt-transform goodness-of-fit test; Patton's time-varying copulas and DCC
copulas of several series), risk contributions, returns and performance,
factor models and the GRS test, portfolios and strategy backtests, event studies, realised volatility and HAR, option
pricing, implied volatility and volatility surfaces (SVI, SABR, Heston), yield curves and the dynamic Nelson-Siegel
model, bonds, and credit risk (Merton, CDS).

```
pip install pymetrix[finance]        # or: pip install pymetrix-finance
```

```python
from pymetrix import Dataset
from pymetrix_finance import performance, value_at_risk     # also importable from pymetrix once installed

ff = Dataset.example("french")    # downloaded from Kenneth French's data library on first use, then cached
value_at_risk(ff, "mkt_rf", level=0.99, method="garch", dist="t", asymmetric=True).backtests()
```

In the app, everything is under the **Finance** menu and workspace. The [tutorial](https://github.com/GeoffHeenan/pymetrix/blob/main/packages/pymetrix-finance/docs/tutorial.md) (also a
[notebook](https://github.com/GeoffHeenan/pymetrix/blob/main/packages/pymetrix-finance/docs/tutorial.ipynb)) walks through it on the example datasets, and its tests check what it says. The Fama-French returns (`french`)
are not shipped, as Kenneth French's data library gives no licence to redistribute them: the first
`Dataset.example("french")` downloads them from the library and keeps them in `~/.pymetrix/cache/datasets/`.

The plugin needs nothing beyond pymetrix's own dependencies. Its checks are listed in pymetrix's
[validation page](https://github.com/GeoffHeenan/pymetrix/blob/main/docs/validation.md#finance-the-bundled-plugin); arch and QuantLib are used there as test
oracles only. License: BSD-3-Clause.

## Development

From the repository root: `pip install -e packages/pymetrix-finance`, then `python -m pytest`; the repository's test
run includes this package's tests, which skip when it is not installed.
