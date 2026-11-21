# Results are their own record

Plan item 7 · built 24 October 2026 · `pymetrix.results`, the result classes, `commands/project.py`

## The rule

A result keeps every number it reports in pymetrix fields, read from the library once, when the result is made.
The library's own result object is optional:
- it lives in `backend_result`, which is compared by nothing and never saved;
- pickling, and so a project's saved results, drop it;
- `codec` refuses results altogether, since they are passed by name.

## What changed

- **ARDL and Markov switching** used to be views onto statsmodels' objects. They now have their own fields:
  - **ARDL**: `params`, `inference` (coef, std err, t, p), `resid`, `fitted`, `stats`, `long_run_coefficients`,
    `ecm_params` and `ecm_inference`, `order`, and the bounds test at the default case;
  - **Markov switching**: `params`, `inference`, `smoothed` and `filtered` probabilities, `transition`,
    `durations`, `endog`, and `stats`, which it did not have before.

  Their summaries, plots and the default bounds test work from those fields, so a saved result keeps working.
  Only a bounds test for another case needs the statsmodels model, and asks for the result to be estimated again
  when it has been dropped.
- **`FittedEquation.backend_result`** holds linearmodels' or arch's result. `extra["linearmodels"]` and
  `extra["arch"]` still work, with a `DeprecationWarning`. arch forecasts now read the parameters from the fit,
  and choose the arch path from `provenance.backend`, so they work after saving too. A wrapper written before this
  change, putting the object in `extra`, has it moved to `backend_result`.
- **Deprecated names.** `FittedARDL.model/.result` and `FittedMarkovSwitching.model/.result` warn and read
  `backend_model`/`backend_result`.
- **Results whose verbs need the library** set `needs_backend = True`, and expose `backend_result` beside their
  existing field:
  - VECM forecasts and impulse responses (`results`);
  - exponential smoothing forecasts (`result`);
  - the dynamic factor model's nowcasts and news (`result`);
  - scikit-learn predictions (`model`).

  They keep the object when pickled for another process. Projects never save them: they are computed again on
  opening. So no project file contains a library object.

## The record form

`result.to_record()` gives the result as pymetrix records it, as JSON-ready data with no library objects:
- its kind;
- the tidy `coefficients()`, `statistics()` (with their definitions) and `observations()` (on the data's
  periods);
- its provenance.

`save_result` writes it into the `.pymxr` file as `record.json`. `load_result` gives back a `SavedResult` with the
same `coefficients()`, `statistics()`, `observations()` and `stats`, so a saved estimate can be read as numbers,
not only as its views. Files saved before this change open as before. Asked for these tables, they say they were
saved without them.

## Compatibility

- **Old attribute names** warn for at least one minor release, as the versioning rules require.
- **Projects** saved by earlier versions are unaffected: their result caches were tied to the package version
  and are recomputed anyway.
