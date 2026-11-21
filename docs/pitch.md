# pymetrix: applied macroeconometrics in Python, from data to forecast report

*A one-page introduction for reviewers and potential collaborators. Draft, November 2026.*

This page says where pymetrix is going. What exists today, and how mature each part is, are in the
[status note](design/objectives-status.md) and the [feature list](features.md). The two are kept apart on
purpose: the vision is larger than what one person has been able to test.

## The problem

Economists in central banks, finance ministries and universities do their applied work across several tools:
- a GUI package handles estimation and tests;
- MATLAB toolboxes handle models;
- spreadsheets carry the forecast round and its judgement;
- Word holds the report.

Python offers the openness to bring this together, but anyone doing so assembles notebooks, libraries and
scripts by hand. Each team rebuilds the same plumbing, and much of the work is hard to reproduce, hand over or
audit.

## The vision

pymetrix aims to be **the open workbench for applied macroeconometrics**: data, estimation, testing, model
solving, forecasting and reports in one workflow, where every step can be replayed and checked, whoever or
whatever took it.

- **One workflow, three ways in.**
  - The same engine runs a Python library, notebook forms and a desktop app with guided dialogs.
  - An AI agent can use it too, over the Model Context Protocol (MCP).
  - Every action, in any of them, becomes readable Python, so work moves freely between a script, a dialog and an
    agent.
- **Replicable by design.**
  - A project saves its data and steps, and rebuilds every result when opened.
  - Results know their sample, method, library and data vintage.
  - Reports end with an appendix of where every number came from.
- **Auditable, for people and for agents.**
  - A research lab book, append-only and hash-chained, records the paths not taken as well as the one reported.
  - It is checked by replaying the work.
  - The aim is a record that a journal, funder or reviewer can verify:
    - who did each step, a person or an agent;
    - what plan was fixed before the data were seen;
    - how many specifications were tried before the one shown.
- **Numbers you can check.**
  - Where a mature library exists (statsmodels, linearmodels, arch, SciPy), pymetrix uses it.
  - Its own code is checked against those libraries, other programs (gretl, Dynare, R), published tables and hand
    formulas.
  - Each feature says how mature it is.
- **Python first, guided when you want it.**
  - Analysts script.
  - Trainees use forms and read the code they generate, and the help explains the econometrics with worked
    examples that are tested.
  - Courses can ship their own material.
- **Open to extension.** Plugins add estimators, data sources, institutional models, finance tools and course
  packs, and appear in the forms, the help and the scripts automatically.

It works alongside the tools people already use rather than cloning any of them: it reads Dynare and IRIS model
files, and its help says where its defaults differ from EViews and Stata, and which option matches.

## Where it stands

- **Version 0.1.0a19:** a development build, founder-led, public and open source (BSD-3-Clause).
- **Broad, and unevenly tested.**
  - Most econometrics and macro-modelling workflows are there, in code and in the app.
  - About 4,200 tests run on Windows, macOS and Linux. The [validation page](validation.md) lists over 350
    independent checks, including Dynare 6.0 on fs2000 and Smets and Wouters (2007), and R's copula packages.
  - Each feature carries a maturity tier: *stable*, *beta* or *experimental*.
  - Since the October build the depth has grown most in DSGE estimation (the No-U-Turn sampler, the particle filter,
    DSGE-VAR, OccBin and regimes), structural VARs and conditional forecasting, applied microeconometrics and
    finance (see the [changelog](../CHANGELOG.md)). These newest parts are *beta* or *experimental*, and the
    app has had little hands-on use beyond its main path.
- **Built quickly, with substantial AI assistance,** under human design and responsibility. That is why every
  native method carries an independent check, why the lab book matters, and why outside testing now matters most.
- **Licence:** BSD-3-Clause.

## What it is not

It is not an EViews clone, a MATLAB replacement, a GUI for statsmodels, or a substitute for IRIS, Dynare or a
model suite. It does not try to have every estimator. It aims to make the everyday macroeconometric workflow open,
inspectable, reproducible and auditable.

## What we are asking

- **As a reviewer:** an hour or two of expert attention. Run the demo, push on what you know best, and tell us what
  would stop you trusting it, teaching with it, or recommending it. The
  [install guide](install.md) says how to get it running.
- **As a collaborator:** take on one part of the testing that one person cannot do alone. Replicate a workflow you
  know, pilot a course lab, or own a plugin. [How to help](how-to-help.md) lists the open slots.
