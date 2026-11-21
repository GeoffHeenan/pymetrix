# Effective state in every result

Plan item 3 · built 24 October 2026 · modules `pymetrix._capture`, `pymetrix.metadata.Provenance`, the estimators

## The problem

Provenance already held the sample rule, the method with its defaults, warnings, convergence and a data
fingerprint. But:
- several results reported the wrong sample, because the library took rows pymetrix did not know about:
  - VECM counted its presample;
  - ARDL, Markov switching and ETS hard-coded `dropped=0`;
  - linearmodels' first differences reported 80 observations where it used 70;
- a sample with gaps looked the same as one without (only the first and last period were kept);
- VAR, VECM, systems and state space did not keep warnings;
- native estimators did not say which code did the numbers;
- tests on data counted every sample row, observed or not, and silently joined gaps.

## The decisions

1. **One function for the rows used.** `_capture.effective_sample(data, used, span=, where=, variables=,
   presample=)` gives:
   - `sample` (first and last period), `nobs` and `dropped`;
   - `rows`: runs of consecutive periods (`[["2000Q2", "2002Q2"], ["2003Q1", "2008Q4"]]`; by entity for panels;
     `None` without a time axis);
   - `presample`;
   - `dropped_rows`: `{"by_reason": {...}, "rows": [[row, reason], ...], "more": n}`, listing the first 50.

   The reason for each requested row that was not used is one of:
   - `presample`: the library took it for lags or initial values;
   - `missing`: a variable of the result is missing there;
   - `lag`: its variables are observed, but a lag, lead or difference reaches outside the data.
2. **Used everywhere.**
   - `Equation.fit` uses the residuals' rows. If a library dropped some of pymetrix's rows, those are the
     presample.
   - VAR, System and state space call it directly. State space uses the periods with an observation, since the
     filter runs through the rest.
   - VECM takes `lags + 1` presample rows from statsmodels' own `nobs`.
   - ARDL and Markov switching pass the rows the library held back. `provenance_for` (used by ETS, MIDAS and
     others) computes the rest.
3. **Every result names its backend.** `provenance.backend` is `{"name", "version", "wraps"}`:
   - the library, through its adapter (item 5);
   - pymetrix itself for native estimators;
   - for state space, whichever engine ran the filter;
   - for tests on data, the library named in the test's `wraps`. ADF, KPSS, Engle-Granger and Fisher ADF now
     declare it.
4. **Warnings everywhere.** `_capture.records_warnings` wraps `VAR.fit`, `VECM.fit`, `System.fit` and
   `StateSpace.fit`, as `Equation.fit` already did. Warnings are recorded once and still shown.
5. **Messages beside warnings.** `provenance.messages` holds what a library returned rather than warned:
   - arch's convergence flag;
   - IV-GMM hitting its iteration limit;
   - a state-space optimiser that did not converge;
   - a test that joined a gappy series.
6. **The Provenance view** shows the dropped rows by reason, any gaps, the presample, the backend, the backend
   options and the messages.

## Compatibility

- **New fields, with defaults.** `Provenance` gains `rows`, `presample` and `dropped_rows` (with item 5's
  `backend`, `options` and `messages`). All have defaults, so saved results load.
- **Some numbers change, because they become right:**
  - `nobs` and `sample` for VECM, first differences and state space with missing periods;
  - `dropped` for ARDL, Markov switching, ETS, VAR and tests on data;
  - `nobs` for tests on series with missing values.

  Data fingerprints are unchanged.

7. **Backtests (decision D7).** A backtest's provenance gathers what its re-estimates record:
   - `spec["model"]` and `method["refit"]`: the model and method re-estimated at each origin;
   - `backend` and `options`: the library and its options, once when every origin used the same, else by origin;
   - `warnings` and `messages`: each one once, with the origins where it arose; `converged` is false if any
     re-estimate did not converge;
   - `nobs`: the forecasts scored; `dropped` and `dropped_rows`: those with no outcome yet;
   - `vintage_request`: what the outcomes were asked as, and each origin's data when there are vintages.
   - `sample`: the first and last origins; `spec["re-estimates"]`: how many, and the estimation samples of the first and
     the last (`views.sample_text` writes them for the Provenance tab and a report's appendix). Rolling and recursive
     fits record the same over their windows' ends.

   A combination names each member's backend and options, and keeps their warnings and messages.

## Not yet

- **Results computed from a model alone** (impulse responses, decompositions, sweeps) still have no provenance,
  as documented in `known-gaps.md`.

## Tests

`tests/test_effective_state.py` covers:
- reasons for dropped rows, and gaps;
- presample for ARDL, VECM and Markov switching against the library's own counts;
- first differences and between;
- ETS trimming;
- VAR and system rows;
- state space with a missing period;
- tests on gappy data;
- warnings from a VAR;
- data without a time axis;
- the cap on listed rows.
