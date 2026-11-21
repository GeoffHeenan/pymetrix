# Adding Features To pymetrix

Draft developer guide, October 2026.

This guide describes how a new upstream feature, native method, test, filter, model, solver, data reader or workflow should be added to pymetrix.

It complements [CONTRIBUTING.md](../CONTRIBUTING.md), which sets the rules every change must meet, and [concepts.md](concepts.md), the objects and verbs a feature should fit; this guide is how to build a feature so that it meets them. `examples/custom_estimator.py` and `examples/custom_solver.py` are worked examples, and `pymetrix.testing` checks a plugin against the conventions below.

The goal is that one well-structured implementation appears consistently in:

- the Python engine;
- desktop forms;
- Jupyter forms;
- generated Python;
- project save/reopen;
- reports;
- help pages;
- validation records;
- AI assistant context bundles (planned: see the [note on AI in the UI](notes/ai-assistance-in-the-ui.md)).

If a feature needs custom work in every frontend, the abstraction is not strong enough yet.

## Core Rule

Every feature should enter pymetrix through the same pattern:

> settings dataclass -> registry entry -> core result -> neutral views -> generated code -> project round-trip -> validation -> documentation.

## 1. Triage

Before implementing, decide whether the feature belongs in pymetrix.

Ask:

- Is it useful for applied macroeconometrics, teaching, forecasting, policy analysis or related professional workflows?
- Does a mature upstream library already implement it?
- Does it fit an existing pymetrix object, such as `Dataset`, `Equation`, `Model`, `VAR`, `StateSpace`, `System`, `TestResult`, `Table` or `Figure`?
- Is it a method, test, filter, solver, model specification, data reader, result view or command?
- Can it return an existing result type?
- Does it require a new public class?
- Does it affect project file compatibility?
- Does it introduce new confidentiality, trust or licensing issues?

If the answer requires changing the public API, file formats, command schema, plugin API or governance rules, write a design note before implementation.

## 2. Prefer Wrapping Mature Libraries

Where a mature, maintained library implements the feature well, pymetrix should wrap it rather than rewriting it.

The wrapper should:

- use pymetrix vocabulary and data structures;
- expose relevant upstream options;
- document any fixed choices;
- record the backend and version in provenance;
- return core pymetrix result types;
- cross-check against the upstream library called directly.

[wrappers.md](wrappers.md) gives the general form of a wrapper, part by part, with a tested worked example (`examples/custom_wrapper.py`: an estimation method and a forecasting model, both from statsmodels).

Native code is appropriate when the feature depends on pymetrix's expression system, model layer, project system, provenance, plugin integration or UI-generated command workflow.

## 3. Settings Dataclass

Most features should start as a frozen dataclass.

Example:

```python
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pymetrix.forms import Widget

@dataclass(frozen=True)
class NewMethod:
    """Short first sentence used in help pages."""

    label: ClassVar[str] = "New method"
    kind: ClassVar[str] = "ls"
    wraps: ClassVar[str] = "statsmodels.some.NewThing"
    docs: ClassVar[str] = "https://example.org/upstream-docs"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)

    option: Annotated[int, Widget("number", min=0, help="One-line help in economists' terms.")] = 4
    trend: Annotated[Literal["n", "c", "ct"], Widget("choice", help="Deterministic terms.")] = "c"
```

Fixed choices keep pymetrix's short codes (`n`, `c`, `ct` for trends; `classic`, `white`, `hac` for covariances), which is what scripts and saved projects contain. Forms show them as words through `CHOICE_LABELS` in `pymetrix.forms`; a field can also be given a clearer form label with `Widget(label=...)` or `LABELS`.

Rules:

- Use `@dataclass(frozen=True)`.
- Use field types that can be serialized and shown in forms.
- Give every visible field one-line help.
- Use pymetrix vocabulary where possible: `cov`, `lags`, `trend`, `span`, `where`, `dist`, `seed`, `alpha`, `method`.
- Use `Literal` for fixed choices, with pymetrix's codes as the values.
- Put rare options behind `advanced=True`.
- Avoid passing raw upstream objects through public settings.
- Keep defaults explicit and documented.

## 4. Registry Entry

Register the feature in the appropriate registry:

```python
from pymetrix.plugins import register

register("methods", NewMethod)        # or decorate the class: @register("methods")
```

Registry groups (`pymetrix.plugins.GROUPS`):

- `methods`
- `systems`
- `models`
- `variances`
- `tests`
- `filters`
- `seasonal`
- `solvers`
- `io`
- `views`
- `commands`
- `courses`

A plugin package registers through an entry point in the matching `pymetrix.<group>` group.

Once registered, the feature should become discoverable by forms, notebooks, help, project replay and AI context generation.

## 5. Engine Implementation

Implement the smallest engine wrapper that converts pymetrix inputs to the backend and converts backend output back to pymetrix results.

For estimation methods:

- use pymetrix sample/design builders;
- respect `span`, `where`, `weights` and `missing`;
- handle lags within entity;
- do not mutate input datasets;
- return `FittedEquation` or another core result type;
- record provenance.

For tests:

- return `TestResult`;
- declare `name`, `null` and `applies_to`;
- include critical values/details where relevant.

For filters:

- return series through `Dataset` methods;
- work within panel entities where applicable;
- document endpoint and missing-value behavior.

For data readers:

- return `Dataset`;
- preserve metadata and vintages where possible;
- avoid making institutional assumptions in the core.

For model solvers:

- expose deterministic settings;
- record shocks, parameters, sample, solver settings and convergence diagnostics.

## 6. Commands

Use an existing command when possible.

Examples:

- new estimation method -> existing equation-estimation command;
- new test -> existing run-test command;
- new filter -> existing filter command;
- new data reader -> existing load-data command;
- new covariance option -> existing fit command.

Add a new command only when the workflow is genuinely new.

Every command must have:

- stable command ID (`@command(menu=..., id="plugin:snake_name")`; the core's IDs are frozen in `tests/fixtures/command_ids.json`);
- schema version (`schema_version`);
- typed fields;
- validation (`validate`, returning messages);
- `run`;
- `to_code`;
- a migration (`@migration(id, from_version)`) when a field is renamed, removed or changes meaning. A new optional field with a default needs none.

The session records which objects each step read and wrote (`CommandRecord.reads` and `writes`), so commands do not declare them.

## 7. Generated Python

Every guided action must produce readable Python.

Generated code should be:

- import-complete;
- minimal;
- stable;
- equivalent to the command;
- suitable for scripts, notebooks and review.

Example:

```python
fit = Equation("y ~ x + y[-1]").fit(data, method=NewMethod(option=8))
```

If a feature cannot generate readable code, it is not ready for guided UI exposure.

## 8. Result Type And Views

Prefer existing result types:

- `FittedEquation`
- `FittedSystem`
- `FittedVAR`
- `FittedStateSpace`
- `Dataset`
- `Table`
- `Figure`
- `TestResult`

If a new result type is necessary, define neutral view sections:

- Summary
- Graph
- Parameters or Coefficients
- Diagnostics
- Details
- Provenance

Register views once (`register_view` in `pymetrix.views`) so that desktop, Jupyter, reports and AI summaries can all consume the same representation.

The result view contract is independent of PySide, Jupyter or any specific renderer: views are lists of titled `Table` and `Figure` sections, defined in `pymetrix.views`, which never imports the UI (a test checks this). Frontends only render them.

## 9. Project Persistence

Every feature exposed through the UI must round-trip through a project.

`pymetrix.testing.check_project(session, command, folder)` does all of this. Test:

1. run command;
2. save project;
3. reopen project;
4. compare result fingerprints (`pymetrix.testing.fingerprint`: every table the result's window shows, as text);
5. compare generated script if appropriate.

Project records should use stable IDs and schema versions. If fields change, add a migration.

Projects from newer pymetrix versions should fail clearly. Projects missing a plugin should say which plugin is required.

## 9a. Data Sources And Provenance

A reader or connector that brings data in fills, for each series' `Variable`, the minimum pymetrix keeps
itself:

- `source`: the provider or file, in words ("FRED, Federal Reserve Bank of St. Louis: GDPC1", "file rates.csv");
- `source_key` (also `source_id`): the series' ID there ("GDPC1");
- `vintage`: when the provider released the data, if it says; never the load date;
- `retrieved`: when pymetrix fetched or imported the values (ISO 8601);
- units, and whether the series is derived (its recipe), as for any series.

A plugin for a system with its own, richer record of sources and vintages (a statistical data management
system such as PSETS) does not copy that record into pymetrix. It sets `external_ref` to its own reference
for the series (e.g. `psets://national-accounts/gdp/2026-09`), which pymetrix keeps in datasets, Parquet
files and projects, shows in the data dictionary and does not interpret; the plugin can resolve it. A
series whose values are replaced loses its source, vintage, retrieval time and external reference, since
they described the old values.

## 10. Validation

A feature is not "working" until validation exists.

Minimum validation layers:

- Form builds and defaults instantiate.
- Settings round-trip through code and JSON.
- Generated Python rebuilds equivalent settings.
- Project save/reopen gives the same result.
- Result views render.
- Provenance records method, sample, backend and package versions.
- Documentation example runs in CI.

For numerical features, also add:

- direct cross-check against upstream library called independently;
- edge-case tests for missing values, lags, sample spans and panels where relevant;
- tolerance chosen and documented;
- deterministic seed test if random;
- known differences documented.

## 11. Validation Matrix

Each substantial feature should have a row in [validation.md](validation.md), in the table for its area, with the columns used there: **Method**, **Checked against** (saying whether pymetrix wraps that reference or computes the result itself) and **Agreement** (the tolerance). Native code that has no outside reference yet goes in the page's "Not yet independently checked" list.

Fuller columns (test file, coverage, known differences, status) may be added as the page grows:

| Feature | Type | Backend/reference | Test file | Tolerance | Coverage | Known differences | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| New method | Estimator | statsmodels direct call | `tests/test_new_method.py` | `1e-8` | time series, missing ends | upstream uses different small-sample correction by default | working |

## 12. Defaults And Differences

If defaults differ from EViews, Stata, MATLAB/IRIS, statsmodels, linearmodels, arch or another reference package, document them in the help page `defaults` (in `pymetrix/help.py`; a copy under `docs/` is planned).

Include:

- pymetrix default;
- reference default;
- option to match the reference where possible;
- reason for the choice;
- effect on results.

## 13. Help And Documentation

Each feature should have:

- first-line docstring;
- field help for every visible setting;
- help topic link;
- upstream reference link if wrapped;
- short example;
- validation note;
- default differences if relevant.

The help should use economists' terms, not only library-internal names.

## 14. Reports

If the feature returns a result likely to appear in a report, ensure that report export can include:

- summary table;
- main graph if applicable;
- relevant diagnostics;
- provenance.

Reports should not need feature-specific custom code unless the result type is genuinely new.

## 15. AI Assistant Context

*Planned: pymetrix has no AI assistant yet. The sources it will draw on already exist: the command registry and `form_spec` field specs, help topics, registered views and the validation and defaults pages. Keeping features discoverable through them is what this section asks.*

The AI assistant should discover new features from existing schemas, not bespoke prompts.

A new feature should appear in AI context through:

- registry entries;
- command schemas;
- field specs and help text;
- help pages;
- result view summaries;
- validation and defaults docs.

The assistant should return structured proposals that use the same command/session layer as the UI.

Example:

```json
{
  "type": "command_proposal",
  "command": "FitEquation",
  "fields": {
    "formula": "y ~ x + y[-1]",
    "method": "New method",
    "options": "option=8"
  },
  "code": "fit = Equation('y ~ x + y[-1]').fit(data, method=NewMethod(option=8))",
  "explanation": "Uses the new upstream method with option set to 8."
}
```

The AI layer must not have a private route to run the feature. It proposes; the user approves; the command engine runs.

## 16. Confidentiality

For features involving AI, cloud services, external data or institutional systems, decide what may leave the machine.

Protect:

- raw data;
- transformed data;
- model equations;
- parameters;
- forecasts;
- scenario assumptions;
- file paths;
- server names;
- report drafts;
- code comments containing confidential information.

Prefer allowlists over blocklists. Show the context that will be sent before sending it.

## 17. Upstream Version Handling

If a feature depends on a new upstream version:

- declare minimum version in `pyproject.toml`, and pin it in `.github/oldest-supported.txt` (CI tests every dependency at its floor; `tests/test_packaging.py` keeps the two in step);
- import it with `pymetrix.adapters.require`, which tells users which `pymetrix[extra]` installs it;
- skip tests gracefully when unavailable;
- show plugin/backend availability clearly;
- record backend version in provenance;
- document any behavior changes across upstream versions.

## 18. Checklist

Before calling a feature working, first the result-completeness questions (see the Results section
of [concepts.md](concepts.md)):

- [ ] Every number the feature shows is a named field of its result, or comes from one; nothing lives
      only in a summary's notes, a chart or a report.
- [ ] Choices made automatically (lag lengths, bandwidths, selected orders) are recorded in `stats` or
      on the result.
- [ ] The result has `provenance`: sample actually used, method and backend, data fingerprint, seed if
      random, warnings captured (`pymetrix._capture.capture`), `converged` if iterative.
- [ ] Tests read the result's fields directly, not only its printed summary.

Then:

- [ ] Feature belongs in pymetrix.
- [ ] Command ID added to `tests/fixtures/command_ids.json` if a new command.
- [ ] Design note written if public API, file format or command schema changes.
- [ ] Settings dataclass is frozen.
- [ ] Fields are serializable and form-compatible.
- [ ] Field help is present.
- [ ] Registry entry added.
- [ ] Engine wrapper uses pymetrix data/sample/design conventions.
- [ ] Existing command reused where possible.
- [ ] Generated Python is readable and tested.
- [ ] Result type is core or has registered neutral views.
- [ ] Project save/reopen test added.
- [ ] Direct upstream cross-check added where possible.
- [ ] Edge cases tested.
- [ ] Provenance records backend/version/sample/settings.
- [ ] Documentation example added.
- [ ] Defaults differences documented.
- [ ] Validation matrix updated.
- [ ] AI context discovers the feature through schemas/help, not bespoke prompt text (once the assistant exists).
- [ ] Confidentiality implications considered.

## Template: Wrapper Skeleton

```python
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pymetrix.forms import Widget
from pymetrix.plugins import register


@dataclass(frozen=True)
class NewMethod:
    """One-sentence description for help pages."""

    label: ClassVar[str] = "New method"
    kind: ClassVar[str] = "ls"
    wraps: ClassVar[str] = "upstream.module.NewMethod"
    docs: ClassVar[str] = "https://upstream.example/docs"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)

    option: Annotated[int, Widget("number", min=0, help="One-line help.")] = 4
    cov: Annotated[Literal["classic", "white", "hac"], Widget("choice", help="Covariance estimator.")] = "classic"

    def estimate(self, eq, data, *, span=None, where=None, weights=None, missing="drop"):
        # 1. Build sample/design with pymetrix helpers.
        # 2. Call upstream library.
        # 3. Convert result to core pymetrix result.
        # 4. Record provenance.
        raise NotImplementedError


register("methods", NewMethod)
```

## Maintainer Rule

Do not merge a feature merely because the upstream library has it.

Merge it when it fits the pymetrix workflow, has a stable user-facing shape, validates against a reference, round-trips through projects, and appears consistently in scripts, forms, notebooks, reports and AI context.
