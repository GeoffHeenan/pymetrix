# How to help

pymetrix has grown faster than one person can test it. This page lists the work that would make it trustworthy,
in slots a collaborator can take on alone, mostly in a few hours. None of them needs you to write pymetrix code.

Before you start:
- **Getting it:** the [install guide](install.md) says how to install pymetrix; send feedback by email, a call, or
  the **Review feedback** issue form.
- **Contributions:** until the 0.1 release, broad code contributions wait until the contribution
  terms are settled ([CONTRIBUTING](../CONTRIBUTING.md)). Replications, testing notes, course pilots and design
  feedback are welcome now.
- **Credit:** a replication you contribute goes into the [validation page](validation.md), and you are named in it
  and in the changelog, unless you prefer not to be.

## 1. Replicate a workflow you know

This is the most valuable help.
- **Where to look:** each feature in the [feature list](features.md) has a maturity tier: *stable*, *beta* or
  *experimental*.
- **What moves a feature up a tier:** a replication against another program by someone who knows the method.

To take a slot:
1. Pick a feature, ideally beta or experimental, that you can run in another program (Stata, EViews, R, Dynare,
   MATLAB, gretl).
2. Run the same data and settings in both. Every help page has a worked example to start from, and pymetrix ships
   example datasets.
3. Send us:
   - both sets of numbers;
   - the other program and its version;
   - the settings you used.

   Differences are as useful as agreement.

Open slots where a replication would help most:

| Area | Replicate against | Why |
|---|---|---|
| Model solving and estimation (first order, OccBin, Metropolis-Hastings) | Dynare, IRIS | the youngest part of the package |
| Panel unit roots and cointegration (Pedroni, Kao, Westerlund, CIPS) | Stata `xtcointtest`, `xtwest`, `xtunitroot`; R `plm` | checked by hand and simulation, not against another package |
| Bayesian VARs (Minnesota, normal-Wishart, stochastic volatility) | R `BVAR`, the ECB's BEAR toolbox | no outside comparison yet |
| Treatment effects (staggered DiD, synthetic DiD, regression discontinuity) | R `did`, `synthdid`, `rdrobust`; Stata | recent methods with many variants |
| Nowcasting and MIDAS (dynamic factor models, news) | R `midasr`, `nowcasting`; New York Fed code | few independent checks |
| Seasonal adjustment (X-13, TRAMO-SEATS) | JDemetra+, X-13 directly | revision and diagnostics output not compared |

## 2. Use the app for a real task

The desktop app has had little hands-on use beyond its main path.
- **What to do:** take a task from your own work, do it in the app rather than in code, and note:
  - where it slowed you down, confused you or got something wrong;
  - what you expected to find and could not.
- **Most useful:** a short list with screenshots. The **Help → About PyMetrix** box gives the version to quote.

## 3. Teach with it

Courses are the cheapest way to get the app and the help used in earnest.
- **The material:** the [example course](../examples/course/) and the
  [forecasting course pack](../examples/forecasting/README.md) have labs, solutions and notebooks that run in the
  test suite.
- **What to do:** pilot one lab with a class or a few students and tell us where they got stuck.
- **Your own course:** you can ship your own material as a course plugin.

## 4. Own a plugin

Plugins let a component grow, and be tested, at its own pace ([plugins](design/plugins.md)). Openings:
- **Seasonal adjustment with external engines:** X-13 and TRAMO-SEATS, the first split proposed in
  [splitting the core](design/plugin-split.md).
- **A data connector for your institution's statistical store**, built on the I/O plugin group.
- **The finance plugin** (`pymetrix-finance`): risk, portfolios, term structure and options. It needs a maintainer
  from finance.

Start from `pymetrix.plugin_template` and the plugins in `packages/` (finance, the lab book, PSETS).

## 5. Shape the research lab book

The lab book aims to give journals and funders a record they can check: what was planned, what was tried, and who
did it, a person or an AI agent. The [design note](design/research-audit.md) ends with open decisions:
- whether the lab book is installed by default;
- the format of the plan and the brief, and how they map onto registries such as OSF and the AEA RCT Registry;
- where an independent AI auditor should live.

Views from editors, research-integrity staff and replication researchers would settle them.

## 6. Review the design

The [design notes](design/) explain the larger choices, each with its open questions:
- the command record and provenance;
- the plugin interface;
- whether to split the core;
- the lab book.

A critical read by someone who has built statistical software is welcome at any time.
