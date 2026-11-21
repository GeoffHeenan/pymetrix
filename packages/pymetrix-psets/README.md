# pymetrix-psets

Reads series from a [PSETS](https://github.com/GeoffHeenan/Econ_timeseries_DB) database into pymetrix, through
PSETS's read-only plug-in API (`psets` 1.x). PSETS (Pretty Simple Economic Time Series) is a lightweight
time-series store for policy institutions; it records each figure with the publisher's release date, so pymetrix
can read the data as they stood on any date.

```python
from pymetrix import Dataset

data = Dataset.read("psets://forecasts/NGDP_R.Q,NGDP.Q")                  # alias "forecasts", two codes
data = Dataset.read("psets:///PCPI_LOC.M?start=2010-01-01")               # the user's default database
data = Dataset.read("psets://forecasts/PCPI_LOC.M?freq=Q&agg=avg")        # converted downward by PSETS
data = Dataset.read("psets://forecasts/NGDP_R.Q", vintage="2020-03-29")   # as released by that date
```

The read is recorded in the project like any other (`Dataset.read(...)`), so a project re-reads PSETS when it is
replayed.

| The address | |
| --- | --- |
| `psets://<alias>/<code>,<code>` | a registry alias, so the project works on any machine with that alias; empty for the user's default database |
| `?start=&end=` | ISO dates, against PSETS's end-of-period convention |
| `?freq=&agg=` | PSETS's own downward conversion (`agg` is required: `avg`, `sum`, `eop`...) |
| `vintage=` (keyword) | PSETS's `as_of`: the publisher's release date, never the load date |

Each series keeps:
- **Metadata:** its PSETS metadata, as pymetrix's `Variable`: label, units, adjustment, source, and the release it
  came from.
- **A reference back:** `external_ref = psets://<alias>/<code>@<vintage>`.
- **In `extra`:**
  - frequency, coverage, dataset, sector, topic and geography;
  - what a canonical code is built from;
  - the open publisher issues PSETS knows about.

  Open issues are also raised as a `PsetsIssueWarning` on reading, where they would stop someone.

**Backtests:** `psets_vintages(source, origins)` builds the `vintages=` mapping for `backtest`, the data as released
by the end of each origin's period, so a backtest uses only what was known at each forecast origin.

**Browsing:** `PsetsCatalogue` answers the questions pymetrix's proposed *Get data* dialog would ask (`search`,
`facets`, `describe`, `vintages`). pymetrix has no catalogue hook yet, so nothing registers it.

## Installing

```bash
pip install -e packages/pymetrix-psets                       # from the pymetrix repository (or: pip install "pymetrix[psets]")
pip install "PSETS @ git+https://github.com/GeoffHeenan/Econ_timeseries_DB#subdirectory=python"
```

- **Why PSETS isn't a dependency:** PSETS isn't on PyPI, and its repository is private, so installing it from git
  needs access to that repository. Until it is installed, **Help → Plugins** lists this plugin as unavailable, with
  the command above.
- **What PSETS brings:** its base install includes the packages its Excel add-in uses (xlwings, openpyxl). Importing
  `psets` loads none of them.

The plugin checks PSETS's API major version (`psets.__version__`) when it opens a database, and refuses a version
it wasn't written for rather than misreading it.

## Tests

- `tests/test_stand_in.py` runs the plugin against a stand-in for `psets` 1.0, with the same calls and records, so
  it is tested where PSETS isn't installed, as in pymetrix's CI.
- `tests/test_with_psets.py` runs it against PSETS itself. It uses a real database built with PSETS's own
  migrations and opened by a registry alias, and checks:
  - revised figures;
  - a gap the publisher left;
  - reading the data as released on an earlier date;
  - PSETS's quarterly conversion;
  - an unknown code.

  It is skipped where PSETS isn't installed.

License: BSD-3-Clause.
