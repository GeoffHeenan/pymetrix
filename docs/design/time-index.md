# One time-series abstraction

Plan item 6 · built 24 October 2026 · `pymetrix.adapters.base.to_axes`, `pymetrix.testing.check_axes`

## The rule

Every user-facing series and table on a result uses pymetrix's axes:
- **time-indexed values** use a `PeriodIndex`, or a `MultiIndex` of entity and period for panels;
- **other rows** carry their own labels: parameter names, regimes, lags, horizons, or long tables whose columns
  say what each row is, as tidy tables do.

Nothing user-facing carries a library's `DatetimeIndex`, bare integer positions where periods belong, or a
library's object; the last lives in `backend_result` (see [results-record.md](results-record.md)).

Parameter names are pymetrix expressions:

| statsmodels | pymetrix |
| --- | --- |
| `c.L1` | `c[-1]` |
| `y.L0` | `y` |
| `D.y.L1` | `d(y)[-1]` |
| `ar.L1[0]` | `g[-1][0]`: the target's lag, in regime 0 |

## What changed

- **One converter.** `to_axes(values, index)` turns a library's dates into periods and aligns them by label;
  anything else is placed by position after checking the length, and a mismatch raises rather than misaligning.
  It is used for the dynamic factor model's factors (statsmodels' monthly dates) and `interop.from_statsmodels`.
- **`from_statsmodels`** converts statsmodels' dates to the dataset's periods. For results made from plain arrays
  it takes `index=` (the rows estimated on); without it, it warns that the residuals are on integer positions.
- **Names.**
  - ARDL parameters and its error-correction form use pymetrix expressions.
  - Markov switching uses the target's lags for `ar.Lk`; it already used the regressors' names for `x1, x2, ...`.
  - The old names still work in `params[...]`, with a `DeprecationWarning` (`PointEstimate.aliases`).
- **The check.** `testing.check_axes(result)` walks a result's fields and public series and fails on library
  dates, on unnamed positions over numbers, and on library objects outside `backend_result`. It runs on every
  result in the test gallery and on wrapped results (`tests/test_axes_and_records.py`).

## Compatibility

- **ARDL.** `params["lc.L1"]` warns and still works; `summary()`, `ecm()` and `long_run()` show the new names.
- **Markov switching.** AR terms are renamed in the same way.
- **The dynamic factor model.** `factors` is indexed by monthly periods, not dates.
