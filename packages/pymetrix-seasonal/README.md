# pymetrix-seasonal

Seasonal adjustment for pymetrix by the two programs statistical offices use. They are in a plugin because each
needs software outside Python.

- **X-13ARIMA-SEATS** (`X13`): the US Census Bureau's `x13as` program, run on a specification pymetrix writes. It
  supports X-11 or SEATS, an automatic or given ARIMA model, outliers, trading-day and Easter effects, and any other
  specification text.
- **TRAMO-SEATS** (`TramoSeats`): as Eurostat's JDemetra+ 3 computes it, in Java. It uses one of JDemetra+'s
  specifications (RSA0 to RSAfull), changed by your settings, with known outliers and your own regressors.

STL and classical decomposition are in pymetrix itself.

## Install

From the repository:

```bash
pip install -e "packages/pymetrix-seasonal[jdemetra]"     # [jdemetra] adds JPype, for TRAMO-SEATS
```

- **X-13:** put the Census Bureau's `x13as` program on the PATH, or set `X13PATH` to its folder
  (https://www.census.gov/data/software/x13as.html).
- **TRAMO-SEATS:** install Java 21 or later (for example Eclipse Temurin). Then fetch JDemetra+'s libraries once
  (about 7 MB, from Maven Central, each checked against its recorded SHA-1):

  ```bash
  python -c "import pymetrix_seasonal.jdemetra as j; j.install()"
  ```

  Offline, copy the jars to a folder named by `JDPLUS_PATH`.

## Use

In the app, both appear under **Data → Transform → Seasonal adjustment…**, beside STL and classical decomposition.
In Python:

```python
from pymetrix import Dataset
from pymetrix_seasonal import X13, TramoSeats      # from pymetrix import X13 works too

data = Dataset.example("sample_monthly")
data = data.seasonal_adjust("retail", X13(trading=True, easter=8))
data = data.seasonal_adjust("ip", TramoSeats(spec="rsa4"))
data.variables["retail_sa"].extra                  # the program's version, its model and the outliers it found
```

Adjusted series are derived series: they are computed again when the data are revised. A project that used X-13 or
TRAMO-SEATS opens without this plugin. That step is kept but not run, and the plugin is named.

## How it is checked

- **X-13:** the adjusted series equal x13as's own saved tables (D11, S11) for each option.
- **TRAMO-SEATS:** checked against Eurostat's R interface to JDemetra+ (rjd3tramoseats) by the
  `jdemetra-reference` workflow.
- **Skipped tests:** without x13as, or without Java and the jars, those tests are skipped.

License: BSD-3-Clause. JDemetra+'s libraries are EUPL 1.2 and are not shipped here.
