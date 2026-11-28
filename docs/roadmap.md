# pymetrix Roadmap

Draft roadmap for discussion, October 2026.

This roadmap is founder-led until the project has active maintainers beyond the project lead. It is intended to make early review more concrete, not to lock the project into a fixed public plan.

> **Where things stand (28 November 2026): Stage 0 done; the repository is public and open source
> (BSD-3-Clause); the November development build, 0.1.0a21, is on PyPI as a pre-release (`pip install --pre
> pymetrix`, with six first-party plugins at 0.1.0a1, the EViews plugin among them: workfiles through a bundle,
> programs translated) and ready for the first demos, which have not happened yet.** The three demos, built from a wheel into a
> clean environment, run end to end as notebooks and in the app. The demo path, governance and contribution drafts,
> the validation page, the CI smoke test, the one-page pitch, the review brief and a note on
> what is new since 0.1.0a8 exist, and the modelling side is checked against Dynare
> 6.0 (fs2000, Smets-Wouters). Feature work has not been frozen: since 0.1.0a8 it has added DSGE estimation (NUTS,
> the particle filter, DSGE-VAR, OccBin and regimes), structural VAR identification, conditional forecasting,
> applied microeconometrics, finance and the FRB/US example (see the [changelog](../CHANGELOG.md)); guided case
> studies are under review as a plugin. Decisions already taken are recorded in
> [GOVERNANCE.md](../GOVERNANCE.md#decided); this page marks them below.

## Project Aim

pymetrix is an integrated Python workbench for applied macroeconometrics: data, estimation, testing, model solving, forecasting, reports and reproducible projects in one workflow, usable from scripts, notebooks and guided forms.

The project should be judged first by whether it supports a credible end-to-end economist workflow, not by whether it has the longest list of estimators.

## Guiding Principles

- Workflow first: demonstrate a complete path from data to report.
- Python-native: every guided action should have a clear Python equivalent.
- Integrate, do not rewrite: wrap mature libraries where they fit.
- Validate native code: cross-check against independent references where possible.
- Keep the engine headless: scripts, notebooks, forms and future UIs should use the same command/session layer.
- Be explicit about trust: project replay, plugins and AI-assisted contributions need clear rules.
- Avoid scope capture: pymetrix should interoperate with EViews, IRIS, Dynare, QuantEcon and institutional tools, but should not become a clone of any one of them.

## Stage 0: First Demos

**Goal:** prepare a credible demo for a small group of trusted reviewers.

**Target audience:** 2-3 econometricians and 2-3 current or former IMF / policy-modelling colleagues.

**Status target:** convincing, not perfect.

### Required Demo Path

The demo should show one coherent workflow:

1. load an installed example dataset;
2. transform and inspect series;
3. estimate an equation;
4. run diagnostics;
5. fit or solve a small macro model;
6. create a forecast or scenario;
7. export a report;
8. show the generated Python;
9. save and reopen the project;
10. show provenance and reproducibility.

The demo should not depend on live web APIs. Data connectors can be shown separately as optional capability.

### Deliverables

- [x] `examples/course/` contains the three demos, all tested in CI:
  - teaching: Lecture 1 and the Lab 2 exercise, with its solution in `examples/solutions/`;
  - Python: [Lab 4](../examples/course/04_engine.ipynb);
  - policy: [Lab 3](../examples/course/03_forecast_round.ipynb).

  The teaching and policy demos also run from the menus.
- [x] A short one-page pitch explains pymetrix as an integrated Python workbench: [pitch.md](pitch.md) (draft), following the [positioning note](notes/positioning.md).
- [x] A review brief asks focused questions about credibility, workflow, missing pieces and risks: (draft, kept outside the repository), built on the review questions under Stage 1.
- [x] [`GOVERNANCE.md`](../GOVERNANCE.md) is in draft form.
- [x] [`CONTRIBUTING.md`](../CONTRIBUTING.md) is in draft form, including DCO and AI-assisted contribution rules.
- [x] A [validation page](validation.md) lists major numerical cross-checks.
- [x] The demo workflow is covered by a smoke test in CI (`tests/test_course_notebooks.py` runs Lab 3 with assertions).

### Definition Of Done

The project lead can run the demo from a clean environment, explain the generated Python, reopen the saved project, and export a report without manual repair.

## Stage 1: Early Review

**Goal:** use expert feedback to decide what must change before 0.1.

**Time horizon:** roughly 4-6 weeks after the first demos.

### Review Questions

Reviewers should be asked:

- What would stop you trusting this for real analytical work?
- What would make this credible in a training course?
- What is missing for a central-bank or IMF-style macro workflow?
- Which defaults or results would you want independently validated?
- Where does the design feel too much like EViews, IRIS, Dynare, QuantEcon or OGP, and where not enough?
- What should be explicitly out of scope?
- Would you recommend this to a colleague? If not, why?

### Review Outputs

Feedback should be triaged into:

- must fix before 0.1;
- should fix soon after 0.1;
- useful later;
- out of scope;
- needs design decision.

### Deliverables

- A feedback summary, with names omitted where appropriate.
- A revised roadmap for 0.1.
- A 0.1 release checklist.
- A decision on package name, PyPI name and name protection.
- A decision on DCO-only versus contributor agreement.
- A decision on whether any reviewer is invited to contribute code, examples, validation cases or course material.

### Definition Of Done

The project has a short, prioritized 0.1 issue list and no unresolved blocker around licence, name, contribution terms, or project trust.

## Stage 2: 0.1 Release

**Goal:** release 0.1 with a coherent, testable promise (the repository is already public).

**Core promise:** install pymetrix, run the demo, inspect the generated Python, and reproduce the report.

### Required Capabilities

- [x] Clean installation instructions (README; the built wheel is tested in a fresh environment in CI).
- [x] CI on supported Python versions and platforms (3.11 to 3.13 on Windows, macOS and Linux, and every dependency at its oldest supported version).
- [x] A tested demo workflow.
- [x] Project save/reopen with trust controls.
- [x] A validation matrix for numerical methods ([validation.md](validation.md)).
- [ ] Clear documentation for data, equations, models, forecasts, reports and projects (the [user guide](guide.md) and in-app help exist; a hosted site does not).
- [x] Contribution rules, governance and AI-assisted contribution policy; a code of conduct beyond the short one in CONTRIBUTING.md is not yet adopted.
- [x] Issue templates for bug reports, numerical discrepancies, feature requests and validation cases (`.github/ISSUE_TEMPLATE`, beside the review-feedback form and the pull-request template; security reports go to the Security tab).

### Documentation Needed

- [x] `README.md`: concise project pitch and quick start.
- [x] `docs/roadmap.md`: current roadmap.
- [x] `GOVERNANCE.md` (repository root, GitHub's convention): founder-led governance and future maintainer path.
- [x] `CONTRIBUTING.md` (root): contribution process, DCO, tests and AI rules.
- [x] `docs/validation.md`: cross-checks and numerical references.
- [x] Defaults that differ from EViews, Stata, statsmodels, linearmodels or other tools: the in-app help page `defaults` (**Help → Help contents**), and [docs/defaults.md](defaults.md), written from it by `scripts/make_defaults_page.py` (a test keeps the two the same).
- [x] `SECURITY.md` (root): project trust model, plugin trust and reporting security issues (a first version).
- [x] `examples/course/03_forecast_round`: the main demo workflow.

### Definition Of Done

A new user can:

1. install pymetrix;
2. run the demo notebook or project;
3. reproduce the report;
4. see the generated Python;
5. understand what is working, what is experimental and what is planned;
6. file a useful issue.

## Stage 3: Post-0.1 Hardening

**Goal:** turn a credible demo system into a reliable early workbench.

### Priorities

- Improve error messages and validation in the main workflow.
- Add more independent validation cases.
- Expand documentation around model solving and forecasting.
- Strengthen plugin conformance tests.
- Improve performance benchmarks for expressions, panels, bootstraps and model solving.
- Add import/export paths that reviewers identify as high value.
- Heterogeneous-agent (HANK) models by the sequence-space Jacobian, wrapping sequence-jacobian in a plugin: proposed in [design/heterogeneous-agents.md](design/heterogeneous-agents.md).
- R scripts translated into pymetrix, on a translation framework shared with the EViews plugin: proposed in [design/r-translator.md](design/r-translator.md).
- Decide whether a VS Code front end is worth prototyping.
- Build a small public example gallery.

### Possible 0.2 Themes

The 0.2 theme should be chosen after the early review. Candidate themes:

- teaching workbench;
- policy forecasting workflow;
- macro model interoperability;
- data connectors and vintages;
- professional reporting;
- VS Code workbench.

## Non-Goals For The Demo Phase

The demo phase should not try to:

- match every EViews command;
- clone IRIS or Dynare;
- replace QuantEcon;
- support every estimator in the roadmap;
- make the GUI perfect;
- build a VS Code extension;
- optimize performance beyond visible demo bottlenecks;
- accept broad code contributions before contribution terms are settled.

## Open Decisions

1. ~~What exact workflow should be the main demo?~~ **Decided:** three demos on one shared dataset, each matched to a reviewer track: **teaching** (Lecture 1 and Lab 2, with a separate instructor solution), **Python** (Lab 4, pymetrix as a library), and **policy** (Lab 3, the forecast round with a structural scenario). Each reviewer starts with the demo for their track. None depends on AI.
2. Which reviewers should see the first demo, and in what order? *Proposed: the econometricians without competing products first, then those tied to EViews, QuantEcon and IRIS.*
3. ~~Should the repo be shared read-only immediately after the demo, or only after feedback is incorporated?~~ **Decided:** the repository is public, open source under BSD-3-Clause, before the demos.
4. ~~Is DCO-only enough for early contributions?~~ **Decided:** DCO only; any employer's agreement is handled for that contributor ([GOVERNANCE.md](../GOVERNANCE.md#decided)).
5. ~~Should the project name be reserved or changed before 0.1?~~ **Decided:** kept, and protected on PyPI by the 0.1 release; `pymetrix` was free on PyPI on 11 October 2026.
6. ~~What is the first interoperability target?~~ **Decided:** Dynare (built), then IRIS model files, then EViews data through EViews' own interface, then QuantEcon.
7. ~~Should AI-use disclosure live in `CONTRIBUTING.md`, `GOVERNANCE.md`, or both?~~ **Decided:** both; the policy in GOVERNANCE.md, the practice in CONTRIBUTING.md and the pull-request template, the reasoning in the [AI-assisted development note](notes/ai-assisted-development.md).
8. What is the minimum documentation set for 0.1? *Proposed: the "Documentation Needed" list above.*

## Near-Term Checklist

- [x] Choose the main demo narrative (three demos, matched to reviewer tracks).
- [ ] Freeze feature work not needed for the demo.
- [x] Make the demo run from a clean environment.
- [x] Add a smoke test for the demo workflow.
- [x] Draft the review brief.
- [x] Finalize draft governance and contribution notes.
- [x] Draft validation and defaults pages (defaults as an in-app help page).
- [x] Prepare a one-page pitch for reviewers.
- [ ] Schedule the first demos.
