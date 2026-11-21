# Third-party notices: pymetrix-finance

The plugin's own code, documentation and simulated data are under the BSD-3-Clause licence in `LICENSE`. One of its
bundled datasets comes from elsewhere, and one example is downloaded when first used; this notice ships with the
plugin's sdist and wheel. Each dataset also records its source in its own metadata
(`Dataset.example(name).meta.license`). pymetrix's own notices are in `THIRD-PARTY-NOTICES.md` in the `pymetrix`
package.

## US Treasury yields (Federal Reserve H.15, via FRED)

`src/pymetrix_finance/datasets/treasury_yields.parquet` (`Dataset.example("treasury_yields")`).

Constant-maturity Treasury yields, monthly, 1982-2026, from the Board of Governors of the Federal Reserve System's
H.15 release, retrieved through FRED (https://fred.stlouisfed.org/), Federal Reserve Bank of St. Louis. The H.15
series are the Board's and in the public domain under its website terms (https://www.federalreserve.gov/disclaimer.htm:
"may be copied and distributed without permission"; the Board is cited as the source); FRED is credited as the source
it was retrieved from.

## Fama-French factors and portfolios (Kenneth R. French Data Library): downloaded, not shipped

`Dataset.example("french")`: monthly factor and portfolio returns, January 1949 to March 2017, from the Kenneth R.
French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The library's files
are marked "Copyright Eugene F. Fama and Kenneth R. French" and the library states no licence to redistribute them,
so the plugin does not ship them: the first load downloads five of the library's CSV files and keeps the dataset in
the user's cache (`~/.pymetrix/cache/datasets/french.parquet`, `src/pymetrix_finance/datasets/french.py`). The data
remain Eugene F. Fama's and Kenneth R. French's.

## Simulated datasets (BSD-3-Clause)

`event_returns`, `intraday_returns`, `option_chain` and `yield_curves` are simulated by pymetrix and are under the
plugin's own licence.

## Reference values in the tests

`tests/test_finance_outside.py` transcribes a few published numbers, attributed where they are used, to check the
Black-Litterman results against: the correlations, volatilities, market weights and returns (equilibrium and with
the first view) of Table 1 of He, G. and Litterman, R. (1999), *The intuition behind Black-Litterman model
portfolios* (Goldman Sachs).
