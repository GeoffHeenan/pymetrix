# Third-party notices: pymetrix-seasonal

The plugin's own code and documentation are under the BSD-3-Clause licence in `LICENSE`. It bundles no third-party
code or data. It runs two outside programs, neither of which is in its sdist or wheel:

- **JDemetra+** (TRAMO-SEATS): `pymetrix_seasonal.jdemetra.install()` downloads JDemetra+'s jars
  (`eu.europa.ec.joinup.sat:jdplus-*`, version 3.9.0) from Maven Central when asked, under the European Union Public
  Licence (EUPL) 1.2 (https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12), together with
  `protobuf-java` (Google, BSD-3-Clause). Each jar is checked against a recorded SHA-1.
- **X-13ARIMA-SEATS**: the US Census Bureau's `x13as` program (https://www.census.gov/data/software/x13as.html),
  which the user installs; the plugin only runs it.

The test script `tests/fixtures/reference/make_rjd3tramoseats.R` reads five of R's own `datasets` when it is run; no
data from them is in the repository or the packages.
