# Third-party notices

pymetrix's own code, documentation and simulated data are under the BSD-3-Clause licence in [`LICENSE`](LICENSE).
This page lists the material from elsewhere that the repository and its packages bundle, or download when asked,
with where it comes from and the terms it is under. Each bundled dataset also records its source in its own
metadata (`Dataset.example(name).meta.license`).
The finance and seasonal plugins ship their own `THIRD-PARTY-NOTICES.md` with the parts of this page that concern
them, in their sdists and wheels.

## Bundled datasets

### From Wooldridge's textbook data, via linearmodels (NCSA licence)

`src/pymetrix/datasets/`: `mroz`, `card`, `jobtraining` and `wage_panel`.

From the data sets of J. M. Wooldridge, *Econometric Analysis of Cross Section and Panel Data*, as redistributed by
the [linearmodels](https://github.com/bashtage/linearmodels) package (version 7.1 when built), under its University
of Illinois/NCSA Open Source Licence. The original studies are cited in each dataset's description: Mroz (1987);
Card (1995); Holzer, Block, Cheatham and Knott (1993); Vella and Verbeek (1998). linearmodels' licence asks that
redistributions reproduce its notice:

> Copyright (c) 2017 Kevin Sheppard. All rights reserved.
>
> Developed by: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk, kevin.k.sheppard@gmail.com)
> <http://www.kevinsheppard.com>
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal with the Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>   disclaimers.
> - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
>   following disclaimers in the documentation and/or other materials provided with the distribution.
> - Neither the name of Kevin Sheppard, nor the names of other contributors may be used to endorse or promote
>   products derived from this Software without specific prior written permission.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
> THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
> CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
> WITH THE SOFTWARE.

### From statsmodels (public domain)

`src/pymetrix/datasets/`: `anes96`, `grunfeld`, `nile` and `randhie`.

From the datasets of [statsmodels](https://www.statsmodels.org/) (BSD-3-Clause), each of which statsmodels marks as
public domain: the American National Election Study, 1996 (ANES); Grunfeld's investment data, as recreated by
Kleiber and Zeileis; the annual flow of the Nile at Aswan, 1871-1970 (Cobb 1978); and the RAND Health Insurance
Experiment.

### US Treasury yields (Federal Reserve H.15, via FRED)

`packages/pymetrix-finance/src/pymetrix_finance/datasets/treasury_yields.parquet`.

Constant-maturity Treasury yields, monthly, 1982-2026, from the Board of Governors of the Federal Reserve System's
H.15 release, retrieved through [FRED](https://fred.stlouisfed.org/), Federal Reserve Bank of St. Louis. The H.15
series are the Board's and in the public domain under its [website terms](https://www.federalreserve.gov/disclaimer.htm)
("may be copied and distributed without permission"; the Board is cited as the source); FRED is credited as the
source it was retrieved from.

### Simulated datasets (BSD-3-Clause)

Every other bundled dataset is simulated by pymetrix and is under pymetrix's own licence: `sample_macro`,
`sample_monthly`, `policy_panel`, `rd_sample` and `synth_panel` in `src/pymetrix/datasets/` (and the copies of the
first two in `examples/`), and `event_returns`, `intraday_returns`, `option_chain` and `yield_curves` in the finance
plugin.

## Test fixtures

These are in `tests/` only; they are not part of the `pymetrix` sdist or wheel.

- **FRB/US excerpts** (`tests/fixtures/eviews/frbus_excerpt.xml`, `frbus_longbase_excerpt.csv`): six variables
  trimmed from `model.xml` and `LONGBASE.TXT` in the Federal Reserve Board's public
  [FRB/US model packages](https://www.federalreserve.gov/econres/us-models-package.htm). Source: Board of Governors
  of the Federal Reserve System. Terms: the Board's [website terms](https://www.federalreserve.gov/disclaimer.htm)
  state that, unless otherwise indicated, information on its website "is in the public domain and may be copied and
  distributed without permission", and ask users to cite the Board as the source; the packages carry no other
  licence statement (checked 20 November 2026). The Board's disclaimer for the FRB/US model also applies: the model
  is provided "as is", and users may not present it, used or modified, as official government material. The
  excerpts, trimmed by pymetrix, are attributed in their headers and are not official government material. The full
  packages are not shipped: the [FRB/US tutorial](docs/tutorials/frbus.md) asks readers to download them. The
  figures in `docs/tutorials/frbus/` are pymetrix's charts of its own simulations of the Fed's model.
- **Reference values from statsmodels' test suite** (`tests/fixtures/reference/`): certified and published values
  transcribed from statsmodels' test results (BSD-3-Clause): the NIST StRD Longley data and certified values (NIST,
  public domain), gretl's exact ML ARMA estimates and JMulTi's VAR estimates on statsmodels' data.
- **Recorded data-connector responses** (`tests/fixtures/connectors/`, including `search/`): short responses
  recorded from each provider's public interface so that the connector tests need no network, each attributed here
  to its source. Federal Reserve Bank of St. Louis, FRED (`fred_*`; the series GDPC1, from the US Bureau of Economic
  Analysis, and UNRATE, from the US Bureau of Labor Statistics, are US government works in the public domain); the
  World Bank (`wb_*`; World Bank Open Data, CC BY 4.0); the International Monetary Fund (`imf_*`; "Source:
  International Monetary Fund", reused under its terms for data, which ask for that attribution); the European
  Central Bank (`ecb_*`; "Source: ECB statistics", free reuse with the source acknowledged); Eurostat (`estat_*`;
  "Source: Eurostat", reuse with acknowledgement under Commission Decision 2011/833/EU); the Bank for International
  Settlements (`bis_*`; "Source: BIS", whose terms permit reproducing its statistics with the BIS cited). The
  `sdmx_*` files are pymetrix's own error and structure cases.
- **Dynare reference output** (`tests/fixtures/dynare-*.json`, `tests/fixtures/sw2007-points.json`): numbers only, produced by running Dynare (GPL) on
  pymetrix's own `.mod` files (`tests/fixtures/dynare/`), on Dynare's fs2000 example and on Smets and Wouters (2007)
  from Johannes Pfeifer's DSGE_mod collection. No Dynare code, model file or data is in the repository: the
  `dynare-reference` workflow fetches them when it runs.

Not in the repository: J.-L. Brillet's *Structural Econometric Modelling: Methodology and Tools with Applications
under EViews* is its author's copyright, so the tests that read the book's EViews programs
(`tests/test_eviews_syntax.py`) read the reader's own copies from the folder named by `PYMETRIX_BRILLET`, and skip
without it.

## Downloaded at run time, not shipped

- **Fama-French factors and portfolios** (`pymetrix-finance`, `Dataset.example("french")`): monthly factor and
  portfolio returns, January 1949 to March 2017, from the
  [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The
  library's files are marked "Copyright Eugene F. Fama and Kenneth R. French" and the library states no licence to
  redistribute them (checked 20 November 2026), so no package ships them: the first load downloads five of the
  library's CSV files and keeps the dataset in the user's cache (`~/.pymetrix/cache/datasets/french.parquet`). The
  data remain Eugene F. Fama's and Kenneth R. French's; the dataset's `meta.license` says so.
- **JDemetra+** (`pymetrix-seasonal`): `pymetrix_seasonal.jdemetra.install()` downloads JDemetra+'s jars
  (`eu.europa.ec.joinup.sat:jdplus-*`, version 3.9.0) from Maven Central, under the
  [European Union Public Licence (EUPL) 1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12),
  together with `protobuf-java` (Google, BSD-3-Clause). Each jar is checked against a recorded SHA-1. None of them is
  in the repository or in any pymetrix package.
- **Data connectors** (FRED, the World Bank, SDMX providers) fetch data only when asked; the data are under their
  providers' terms.
