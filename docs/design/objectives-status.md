# Status against the design objectives

Design note, 30 October 2026 · pymetrix 0.1.0a8 at `main` 3266e7e, with the lab-book plugin as built · an
assessment: nothing here is built by this note

## The question

How far does pymetrix meet its two central objectives?

1. **Three ways in, one record.** Most econometrics and macro-modelling workflows can be done headless (scripts,
   notebooks, an agent over MCP) or in the desktop app. Either way they leave a record that replays.
2. **An auditable lab book.** The record of the work can be checked by a reviewer, whether the work was done by a
   person or by an agent connected over MCP.

Sources:
- the feature list ([features.md](../features.md)) and the known gaps ([known-gaps.md](../known-gaps.md));
- the command and function reference ([reference](../reference/reference.md));
- the lab-book design ([research-audit.md](research-audit.md)) and its plugin (`packages/pymetrix-labbook`);
- the MCP server (`pymetrix.mcp_server`).

## Short answer

| Objective | Status |
|---|---|
| Headless and app for most workflows | Met. 96 of 97 listed features work; 151 of 228 public calls have a dialog; the rest are script follow-ups on a result. |
| Replicable | Met, and the strongest part: recorded calls, provenance on every result estimated from data, versioned projects, replay on open, trust for typed code. |
| Macro modelling as complete as econometrics | Partly. The standard workflow is covered; advanced Dynare and IRIS features are not yet. |
| Lab book for a human operator | Met at a first-version level: append-only, hash-chained, checked by replay. |
| Lab book for an agent over MCP | Partly. Agent steps are recorded and checked like any other, but not marked as an agent's, with no record of the instructions it acted on. |
| Plan before the data, count of what was tried, audit tools | Not built (layers 2, 3 and 5 of the lab-book design). |

## 1. Headless, replicable and app

### What meets it

- **One mechanism.**
  - Almost every dialog builds a recorded API call, the same code a user would type ([commands.md](commands.md)).
  - The Script window shows the code.
  - A project saves the data and the steps, and rebuilds the results when opened.
  - Scripts, notebooks, the app and the MCP server run the same commands, so a workflow started in one carries on
    in another.
- **Coverage.**
  - 96 of the 97 features in [features.md](../features.md) are marked working; third-order perturbation is the one
    not started.
  - Of 228 public calls in the reference, 151 have a dialog and 77 are script-only. The script-only calls are mostly
    follow-ups on a result (`FittedVAR.girf`, `vecm.fluctuation_test`) or specialist estimators (`MomentGMM`).
  - The everyday workflows have a dialog and code alike: data, estimation, tests, VARs and VECMs, panels, treatment
    effects, forecasting rounds, models, reports.
- **Replicability** ([provenance.md](provenance.md)).
  - Every result estimated from data, and every test on data, records:
    - data fingerprints and vintages, and the sample rule;
    - the rows dropped and why;
    - the library, its function and arguments;
    - seeds, warnings and convergence.
  - Projects are versioned and migrated when opened.
  - Typed code replays only once trusted.
- **Numbers that can be checked.** About 2,700 tests, many against independent implementations (statsmodels,
  linearmodels, arch, Dynare, R's `rjd3tramoseats`, published tables), listed in [validation.md](../validation.md).
- **Help that teaches.**
  - Every page explains its settings and the econometrics behind them.
  - The main pages carry a worked example that the test suite runs, and a table of how to read the results.
  - References link through doi.org where a DOI has been checked against Crossref.

### Where it falls short

- **Macro modelling is the youngest part** ([known-gaps.md](../known-gaps.md), Methods). Not yet:
  - estimation with occasionally binding constraints;
  - samplers beyond random-walk Metropolis-Hastings (SMC, slice);
  - Dynare's macro processor and correlated shocks;
  - links between entities in models solved per country;
  - second-order solutions of growing models;
  - third-order perturbation.

  The standard workflow is covered: models from equations, Dynare and IRIS files, solving, estimation, forecasts
  and scenarios.
- **Results from a model alone carry no data provenance.** Impulse responses, decompositions, OccBin paths, sweeps
  and optimal policy depend on the model, not on data. Forecast rounds record their own judgement but not the
  model's provenance.
- **The desktop app has had little hands-on use** beyond the developer's. The main demo path is well used; other
  dialogs have had less polish.
- **Newly public.** There has been no outside code review yet, and the API may change before 0.1.

## 2. An auditable lab book

### For a human operator: met at a first-version level

`pymetrix-labbook` (stages 1 and 2 of [research-audit.md](research-audit.md)):

- **Records every session with no set-up:** scripts, notebooks, the app and the MCP server.
- **What it keeps:** an append-only, hash-chained log inside the project. It includes the steps a project's history
  drops (undone, cancelled, failed, removed), so the paths not taken are visible.
- **No data values:** a lab book can go to a reviewer without the data.
- **What `pymetrix-labbook verify` checks:**
  - the chain;
  - coverage: no save made without the lab book, from the save log;
  - that the project's steps are those recorded;
  - a replay that reproduces each result, estimates to 10 significant digits and datasets exactly.
- **What it says it cannot check:** work done before the lab book was installed, untrusted code, missing plugins,
  results not yet computed.

### For an agent over MCP: partly met

An agent works through the MCP server's six tools: list, describe and run commands, show results, export the
script, and save the project. The server is closed by default: no Python, no file writes and no network unless
allowed, and only inside its root folder. Each step is recorded, chained and replayed exactly as a human's is.
What is missing:

1. **Who acted.** An entry holds the time, the event, the code, what was read and written, and any error. It does
   not say whether a step came from the app, a script, a notebook or an MCP client, nor which client or model. A
   reviewer cannot tell an agent's steps from a person's.
2. **Why.** The instructions an agent acted on are not kept, only its actions. That matches the lab book's choice
   to hold no data, but an audit of agent work usually wants the request, at least optionally.
3. **The plan and the count** (layers 2 and 3 of the design):
   - no project brief or analysis plan fixed before the data are opened;
   - no count of the specifications tried.

   These are what make the record useful against specification searching, whoever does the analysis.
4. **Audit tools** (layer 5). There are no read-only MCP tools to query a lab book, and the server cannot open an
   existing project, so an independent AI auditor has no way in yet.
5. **Outside time-stamping** (stage 6). The lab book proves what happened while it was installed, not when,
   against someone who controls the machine.

## Next steps, in order

1. **Record who acted.** Each lab-book entry gains:
   - the channel: app, script, notebook or MCP;
   - over MCP, the client's name and version from the MCP handshake;
   - an optional field for the agent's request or reason.

   Small, and it closes the main gap for agents. Core already knows the channel; the session event would carry it
   to the plugin.
2. **Layer 2:** the project brief and analysis plan, fixed before the data, with a deviation report in `verify`.
3. **Layer 3:** counting what was tried, with specification curves in reports.
4. **Layer 5:** read-only MCP audit tools and `open_project` on the server, as a separate auditor plugin, so the
   auditor is independent of the audited.
5. **Macro modelling:** OccBin estimation, then Dynare's macro processor and correlated shocks.
