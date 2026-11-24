# pymetrix

**Applied macroeconometrics in Python, from data to forecast report.** Estimate equations and
systems, test them, solve models, run scenarios, manage forecast rounds and export reports, from
Python scripts, Jupyter notebooks or a desktop app with guided forms. Every action becomes
Python you can read and rerun.

pymetrix is for economists in policy institutions, teaching and research who want Python's
openness without assembling their work from notebooks, spreadsheets, GUI packages and one-off
scripts. It covers much of what EViews and MATLAB/IRIS are used for, and it is open source
(BSD-3-Clause).

> **Status: development build (0.1.0a19).** One developer so far; public and open source (BSD-3-Clause), and
> reviews, bug reports and contributions are welcome ahead of 0.1 ([CONTRIBUTING.md](CONTRIBUTING.md)). The core
> works and is tested on Windows and Linux; see [known gaps](docs/known-gaps.md) for what is missing or rough, and
> the [changelog](CHANGELOG.md).

**What's new since 0.1.0a8:** DSGE estimation with the No-U-Turn sampler, the particle filter, DSGE-VAR and
OccBin with regimes; zero and sign restrictions and historical decompositions with bands for VARs; conditional
forecasts with ranges and structural scenarios; applied microeconometrics (double machine learning, shift-share IV,
interactive fixed effects, conditional logits); copulas and more in the finance plugin; the FRB/US example; and in
the app, Redo, a series browser and figures annotated by dragging. The [changelog](CHANGELOG.md) has
every detail.

## Why pymetrix

- **One workflow.** Data and transformations, estimation and diagnostics, model solving and
  scenarios, forecast rounds with judgement, backtests and reports live together and pass
  results to each other.
- **Reproducible by design.** Every dialog records the Python it runs. A project saves the data
  and the steps and rebuilds the results when opened, and results know their sample, method and
  data vintage. Typed code in a project runs only once you trust it.
- **Numbers you can check.** Wherever a mature library exists (statsmodels, linearmodels, arch,
  SciPy), pymetrix uses it; its own code is checked against libraries, published tables, other
  packages and hand formulas. [Validation](docs/validation.md) lists the checks, and the help
  page on defaults says where they differ from EViews and Stata.
- **Python first, guided when you want it.** Write scripts and notebooks for control; use forms
  for guidance and teaching. Plugins add estimators, tests, data sources and course material,
  and they appear in the forms, the help and the scripts automatically.
- **Coming from EViews.** The `pymetrix-eviews` plugin (experimental, `packages/pymetrix-eviews`) reads workfiles
  through a bundle its export program writes, and translates EViews programs into readable pymetrix scripts with a
  report on every line: data work, estimation, tests, models and control flow (loops, ifs, subroutines).

## Install

Not on PyPI yet. Reviewers: [installing pymetrix](docs/install.md) covers the wheel attached to each
release, installing from this repository, and Codespaces. From a clone, with Python 3.11 or later:

```bash
pip install -e ".[all]"
```

`[all]` adds linearmodels, arch, scikit-learn, the desktop app (PySide6), Jupyter forms and report writers; the core
needs NumPy, pandas, SciPy, statsmodels, matplotlib and pyarrow.

**Try it in your browser, with nothing to install:** open this repository in GitHub Codespaces
([open a codespace](https://codespaces.new/GeoffHeenan/pymetrix), or **Code → Codespaces → Create codespace**). The codespace installs pymetrix and opens the desktop app
with the Lab 3 demo on a desktop in your browser: in the **Ports** tab, open **pymetrix desktop**
(port 6080), then connect with the password `vscode`. Notebooks run there too. It uses your
GitHub account's free monthly Codespaces allowance; see [.devcontainer/README.md](.devcontainer/README.md).

## Quick start

```python
from pymetrix import OLS, Dataset, Equation, Model, Scenario

data = Dataset.example("sample_macro")            # a simulated quarterly economy, installed with pymetrix
cons = Equation("c ~ y + c[-1]").fit(data, method=OLS(cov="hac"))
print(cons.summary())
print(cons.test("breusch_godfrey", lags=4))

model = Model([cons, Equation("i ~ y[-1]").fit(data), Equation("y = c + i + g")])
base = model.solve(data, ("2020Q1", "2024Q4"))
more_g = model.solve(data, ("2020Q1", "2024Q4"), scenario=Scenario(shift={"g": 5.0}))
print(((more_g["y"] - base["y"]) / 5).loc["2020Q1":"2020Q4"])   # the spending multiplier, building up
```

The desktop app is `pymetrix-gui`: start with **Load example data** (or your own file), then the
**Estimate**, **Test**, **Model** and **Forecast** menus. **File → Export script** shows what you did
as Python. `pymetrix-gui project.pymx` opens a project, and `pymetrix-gui --demo forecast_round`
a demo (also **File → Open demo project…**); `--maximized` fills the screen.

## Try the demos

Three demos, on one simulated economy installed with pymetrix:

- **Teaching:** [Lecture 1](examples/course/01_consumption.md), a consumption function linked
  into the help (OLS against IV: simultaneity bias, and the true parameters recovered), and the
  [Lab 2 exercise](examples/course/02_lab.ipynb): estimate, test, model and a fiscal scenario. The instructor's solution is in
  [examples/solutions](examples/solutions/02_lab_solution.ipynb).
- **Python:** [Lab 4](examples/course/04_engine.ipynb), pymetrix as a library:
  - estimators side by side, and a cross-check against statsmodels;
  - a new estimator added as a plugin;
  - dialog steps as plain Python;
  - a New Keynesian model estimated from data by maximum likelihood, and what is weakly identified.
- **Modelling:** [building and estimating models](docs/modelling-quickstart.md) for Dynare and
  IRIS users, and two demo projects in the app (`pymetrix-gui --demo nk_estimation`,
  `--demo policy_and_nonlinear`): Bayesian estimation, identification, the lower bound, optimal
  policy and second order.
- **Policy:** [Lab 3](examples/course/03_forecast_round.ipynb), a quarterly forecast round:
  - VAR forecast, judgement and revisions;
  - a backtest against a Bayesian VAR;
  - a structural fiscal scenario;
  - a report with provenance, also from the `forecast_round` template in the policy theme;
  - the project reopened;
  - a nowcast of the quarter in progress from monthly indicators (bridge equation and dynamic
    factor model), with the news as the quarter's figures arrive.
- **Forecasting course:** [six labs](examples/forecasting/README.md) following the topics of Elliott and
  Timmermann's *Economic Forecasting*:
  - loss functions, forecast comparison, combination, densities, many predictors, and breaks;
  - each on simulated data with known answers, and checked against an independent calculation;
  - with a chapter-by-chapter coverage map.

The teaching and policy demos also run from the desktop app's menus. Every notebook runs in the
test suite, so the demos are tested on every change, and none of them uses AI.

## Documentation

- [Start here: building and estimating models](docs/modelling-quickstart.md): for Dynare and IRIS
  users, equations to an estimated, forecasting model in one runnable script
- [User guide](docs/guide.md): every area by topic, with short examples
- [How do I…](docs/how-do-i/README.md): task pages, each one task start to finish, in the app and in code (also
  Help → How do I… in the app)
- [Features](docs/features.md): what works and what is planned
- [Validation](docs/validation.md): what the numbers are checked against
- [Known gaps](docs/known-gaps.md): what is missing or rough
- [Concepts](docs/concepts.md): the objects and verbs pymetrix is built from, and what is planned
- [Design](docs/design.md): architecture, decisions and the build log
- [Adding features](docs/adding-features.md): how a method, test or command is added so it works
  in scripts, forms, notebooks, projects and reports alike
- [Wrapping another library](docs/wrappers.md): the general form of a wrapper, with a worked example
- [Writing a case study](docs/case-studies.md): for lecturers, a lab the app's Guide panel takes trainees through
- [An MCP server](docs/notes/mcp-server.md): pymetrix for AI assistants (`pip install pymetrix[mcp]`), every step
  recorded and replayable
- In the app, **Help** (F1) has a page for every dialog, method and test, and links to the
  wrapped libraries' documentation

## The project

pymetrix is founder-led and open to review before its 0.1 release:

- [Pitch](docs/pitch.md): where pymetrix is going, on one page, for reviewers and collaborators;
- [Status](docs/design/objectives-status.md): what exists today against the design objectives, and the gaps;
- [How to help](docs/how-to-help.md): replications, testing, teaching and plugins that need an owner;
- [Roadmap](docs/roadmap.md): the stages from the first demos to 0.1 and after;
- [Governance](GOVERNANCE.md): roles, decisions, intellectual property and AI-assisted work;
- [Contributing](CONTRIBUTING.md): what a change needs, and the DCO sign-off;
- [Security](SECURITY.md): project trust, plugins, and reporting problems.

Background notes:
- [Positioning](docs/notes/positioning.md): where pymetrix sits among open-source workbenches;
- [AI-assisted development](docs/notes/ai-assisted-development.md): the reasoning behind the AI rules;
- [AI in the UI](docs/notes/ai-assistance-in-the-ui.md): the planned design for an assistant.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[dev,jupyter]"
pytest
```

## Licence

BSD-3-Clause (see `LICENSE`). Bundled datasets and test fixtures from elsewhere, and what pymetrix downloads
when asked, are listed with their sources and licences in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
