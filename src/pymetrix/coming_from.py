"""Where a familiar EViews, IRIS or Dynare command lives in pymetrix.

One table, ``TRANSLATIONS``, feeds both the help topics "Coming from EViews" and "Coming from MATLAB/IRIS and
Dynare" (a row per command: the menu path and the Python call) and the command palette's synonyms (typing ``genr``
in Find anything offers "Generate series… (EViews: genr)"), so the two cannot drift apart. Every menu path and
every ``pymetrix`` name used here is checked by ``tests/test_coming_from.py``. X-13ARIMA-SEATS and TRAMO-SEATS are
the ``pymetrix-seasonal`` plugin's (``pymetrix_seasonal.X13``, ``pymetrix_seasonal.TramoSeats``): their rows name the
plugin, and the Seasonal adjustment dialog they point to is the core's, there with or without it (STL and classical
decomposition without it).
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

__all__ = ["PACKAGES", "TRANSLATIONS", "Translation", "palette_synonyms", "translation_table"]


@dataclass(frozen=True)
class Translation:
    """One familiar command and where it is in pymetrix."""

    package: str  # "EViews", "IRIS" or "Dynare"
    commands: tuple[str, ...]  # as its users type them: ("genr", "series"); one-word ones are the palette's synonyms
    does: str  # what it does, in a few words
    menu: str = ""  # the app's menu path, as the menu bar shows it ("Data → Generate series…"); "" when code only
    python: str = ""  # the pymetrix call


# In the Python column: data is a Dataset, fit an estimated equation, var an estimated VAR, model a Model, nk a model
# read from a file or a template, sol its solution, est the estimated model, mod a Dynare model, rep a Report.
TRANSLATIONS: tuple[Translation, ...] = (
    # ---------------------------------------------------------------------------------------------- EViews
    Translation("EViews", ("wfopen", "import", "read"), "open a data file", "File → Load data…",
                'data = Dataset.read("macro.xlsx", time="date", freq="Q")'),
    Translation("EViews", ("dbopen", "fetch"), "fetch from a database", "File → Load from connector…",
                'Dataset.read("fred://GDPC1")'),
    Translation("EViews", ("wfcreate", "pagecreate"), "a new, empty workfile page", "File → New dataset…",
                'data = Dataset.empty("Q", "1990Q1", "2030Q4")'),
    Translation("EViews", ("wfsave",), "save the workfile", "File → Save project", ""),
    Translation("EViews", ("genr", "series"), "a new series, computed once", "Data → Generate series…",
                'data = data.with_series(dy="d(y)")'),
    Translation("EViews", ("frml",), "a series that follows its formula", "Data → Derived series…",
                'data = data.derive(dly="dlog(y)")'),
    Translation("EViews", ("smpl",), "the active sample", "Data → Sample → Set sample…",
                'data = data.with_sample(("2000Q1", "2019Q4"))'),
    Translation("EViews", ("smpl then y =",), "assign a series within a sample", "Data → Generate series…",
                'data = data.with_series(y="x / 2", within=("2000Q1", "2019Q4"))'),
    Translation("EViews", ("sample",), "a named sample", "Data → Sample → New sample…",
                'Sample(start="2000Q1", end="2019Q4")'),
    Translation("EViews", ("group",), "a group of series", "Data → Select series…", 'data.series("c", "i", "g")'),
    Translation("EViews", ("line", "graph", "plot"), "graphs of series", "Data → Plot series…",
                'data.plot(["c", "y"])'),
    Translation("EViews", ("stats",), "descriptive statistics", "Data → Statistics → Descriptive statistics…",
                "data.describe()"),
    Translation("EViews", ("cor", "cov"), "correlations", "Data → Statistics → Correlations…", "data.corr()"),
    Translation("EViews", ("copy", "frequency conversion"), "change frequency", "Data → Transform → Convert frequency…",
                'data.to_freq("Y")'),
    Translation("EViews", ("x13", "x12"), "X-13ARIMA-SEATS seasonal adjustment (the pymetrix-seasonal plugin)",
                "Data → Transform → Seasonal adjustment…", 'data.seasonal_adjust("c", pymetrix_seasonal.X13())'),
    Translation("EViews", ("tramo",), "TRAMO-SEATS seasonal adjustment (the pymetrix-seasonal plugin)",
                "Data → Transform → Seasonal adjustment…", 'data.seasonal_adjust("c", pymetrix_seasonal.TramoSeats())'),
    Translation("EViews", ("seas",), "seasonal adjustment by moving averages", "Data → Transform → Seasonal adjustment…",
                'data.seasonal_adjust("c", Classical())'),
    Translation("EViews", ("hpf",), "Hodrick-Prescott filter", "Data → Transform → Filter (trend and cycle)…",
                'data.filter("y", HP())'),
    Translation("EViews", ("ls",), "least squares", "Estimate → Equation…",
                'fit = Equation("c ~ y + c[-1]").fit(data)'),
    Translation("EViews", ("tsls",), "two-stage least squares", "Estimate → Equation…",
                'Equation("c ~ y + c[-1]").fit(data, method=IV(endog=["y"], instruments=["g", "g[-1]"]))'),
    Translation("EViews", ("arch",), "ARCH and GARCH", "Estimate → Equation…",
                'Equation("y ~ 1", variance=GARCH(1, 1)).fit(data)'),
    Translation("EViews", ("logit", "probit", "binary"), "binary choice", "Estimate → Equation…",
                'Equation("d ~ x", family=Logit()).fit(data)'),
    Translation("EViews", ("system", "sur", "3sls"), "a system of equations", "Estimate → System of equations…",
                'System([Equation("c ~ y"), Equation("i ~ y[-1]")]).fit(data, method=SUR())'),
    Translation("EViews", ("var",), "a VAR", "Estimate → VARs and local projections → VAR…",
                'var = VAR(["c", "i", "y"], lags=2).fit(data)'),
    Translation("EViews", ("impulse",), "impulse responses", "Estimate → VARs and local projections → Impulse responses…",
                "var.irf(20)"),
    Translation("EViews", ("decomp",), "variance decomposition",
                "Estimate → VARs and local projections → Variance decomposition…", "var.fevd(20)"),
    Translation("EViews", ("coint",), "Johansen cointegration test", "Test → Johansen cointegration…",
                'data.johansen(["c", "y"], lags=2)'),
    Translation("EViews", ("uroot",), "unit-root test", "Test → Run test…", 'data.test("adf", "y")'),
    Translation("EViews", ("auto",), "serial correlation LM test", "Test → Run test…",
                'fit.test("breusch_godfrey", lags=4)'),
    Translation("EViews", ("hettest", "white"), "heteroskedasticity tests", "Test → Run test…",
                'fit.test("white")'),
    Translation("EViews", ("archtest",), "ARCH LM test", "Test → Run test…", 'fit.test("arch", lags=4)'),
    Translation("EViews", ("hist", "jbera"), "normality of residuals", "Test → Run test…", 'fit.test("jarque_bera")'),
    Translation("EViews", ("reset",), "Ramsey's RESET", "Test → Run test…", 'fit.test("reset")'),
    Translation("EViews", ("chow",), "Chow breakpoint test", "Test → Run test…", 'fit.test("chow", at="2008Q1")'),
    Translation("EViews", ("wald",), "Wald test of coefficient restrictions", "Test → Run test…",
                'fit.wald("y = 0.5")'),
    Translation("EViews", ("correl",), "correlogram and Q statistics", "Test → Correlogram…", "fit.correlogram(12)"),
    Translation("EViews", ("forecast", "fit"), "forecast from an equation", "Forecast → Forecast…",
                'fit.forecast(data, ("2020Q1", "2021Q4"), dynamic=True)'),
    Translation("EViews", ("model", "append"), "a model object", "Model → Model editor…",
                'model = Model([fit, Equation("y = c + i + g")])'),
    Translation("EViews", ("solve",), "solve a model", "Model → Solution → Solve model…",
                'model.solve(data, ("2020Q1", "2024Q4"))'),
    Translation("EViews", ("addinit",), "add factors", "Model → Solution → Add factors…", "model.add_factors(data)"),
    Translation("EViews", ("scenario", "override"), "scenarios", "Model → Scenarios → Compare scenarios…",
                'model.solve(data, ("2020Q1", "2024Q4"), scenario=Scenario(shift={"g": 5.0}))'),
    Translation("EViews", ("stochastic",), "stochastic simulation", "Model → Solution → Solve model…",
                'model.solve(data, ("2020Q1", "2024Q4"), draws=1000)'),
    Translation("EViews", ("sspace",), "a state-space object", "Model → Import → EViews state-space object…",
                "read_eviews_sspace(text)"),
    Translation("EViews", ("freeze",), "keep a table or graph", "File → Save result…", 'save_result(fit, "consumption")'),
    Translation("EViews", ("spool", "report"), "collect output in a document", "View → Report",
                'Report("Forecast round").add(fit).save("round.docx")'),
    Translation("EViews", ("program", "run", "exec"), "a program (.prg)", "File → Open script…", ""),
    Translation("EViews", ("command window",), "type commands", "View → Immediate", ""),
    # ---------------------------------------------------------------------------------------------- IRIS
    Translation("IRIS", ("model",), "read a model file", "Model → Import → IRIS file…", 'nk = read_iris("nk.model")'),
    Translation("IRIS", ("sstate", "chksstate"), "the steady state", "Model → Check model…", "nk.steady_state()"),
    Translation("IRIS", ("solve",), "the first-order solution", "Model → Solution → Rational expectations…",
                "sol = nk.solve()"),
    Translation("IRIS", ("srf",), "shock responses", "Model → Solution → Rational expectations…", "sol.irf(20)"),
    Translation("IRIS", ("fevd",), "variance decomposition", "", "sol.fevd(20)"),
    Translation("IRIS", ("simulate", "plan", "exogenize", "endogenize"), "simulate, with a plan",
                "Model → Solution → Solve model…",
                'sol.solve(data, ("2025Q1", "2026Q4"), scenario=Scenario(shift={"ey": 1.0}))'),
    Translation("IRIS", ("filter",), "Kalman filter and smoother", "Model → Analysis → Kalman filter…",
                "sol.filter(data, nk.measurement, nk.shock_sd)"),
    Translation("IRIS", ("estimate",), "estimate parameters", "Model → Estimation → Estimate model…",
                'est = nk.estimate(data, estimate=["kappa", "rho"])'),
    Translation("IRIS", ("arwm",), "posterior simulation", "Model → Estimation → Metropolis-Hastings…",
                "est.mcmc(draws=10000)"),
    Translation("IRIS", ("jforecast",), "forecast", "Model → Analysis → Forecast from estimated model…",
                'est.forecast(("2025Q1", "2026Q4"))'),
    Translation("IRIS", ("contributions",), "shock decomposition", "Model → Analysis → Shock decomposition…",
                "est.decompose()"),
    Translation("IRIS", ("acf",), "theoretical moments", "", "est.moments()"),
    Translation("IRIS", ("sspace",), "state-space matrices", "Model → Analysis → State-space form…", "sol.system"),
    Translation("IRIS", ("dbload", "databank"), "read a databank", "File → Load data…",
                'data = Dataset.read("data.csv", time="date", freq="Q")'),
    Translation("IRIS", ("dbsave",), "save a databank", "", 'data.write("data.csv")'),
    Translation("IRIS", ("Series", "tseries"), "a time series", "Data → Spreadsheet editor…", 'data["y"]'),
    Translation("IRIS", ("hpf",), "Hodrick-Prescott filter", "Data → Transform → Filter (trend and cycle)…",
                'data.filter("y", HP())'),
    Translation("IRIS", ("x13",), "X-13ARIMA-SEATS seasonal adjustment (the pymetrix-seasonal plugin)",
                "Data → Transform → Seasonal adjustment…", 'data.seasonal_adjust("c", pymetrix_seasonal.X13())'),
    Translation("IRIS", ("rephrase", "reporting"), "reports", "View → Report",
                'Report("Forecast round").add(est).save("round.html")'),
    # ---------------------------------------------------------------------------------------------- Dynare
    Translation("Dynare", ("mod", "dynare"), "read a .mod file", "Model → Import → Dynare file…",
                'mod = read_mod("nk.mod")'),
    Translation("Dynare", ("steady", "check"), "steady state, Blanchard-Kahn", "Model → Check model…",
                "mod.steady_state()"),
    Translation("Dynare", ("stoch_simul",), "first-order solution and responses", "Model → Solution → Rational expectations…",
                "mod.solve().irf(20)"),
    Translation("Dynare", ("order=2",), "second-order solution", "Model → Solution → Second-order solution…",
                "mod.solve(order=2)"),
    Translation("Dynare", ("estimated_params", "estimation", "varobs"), "estimate the model",
                "Model → Estimation → Estimate model…", "est = mod.estimate()"),
    Translation("Dynare", ("mh_replic",), "Metropolis-Hastings", "Model → Estimation → Metropolis-Hastings…",
                "est.mcmc(draws=20000)"),
    Translation("Dynare", ("smc", "hssmc"), "sequential Monte Carlo", "Model → Estimation → Sequential Monte Carlo…",
                "est.smc()"),
    Translation("Dynare", ("dsge_var",), "DSGE-VAR", "Model → Estimation → Estimate DSGE-VAR…", "mod.dsge_var()"),
    Translation("Dynare", ("calib_smoother",), "Kalman smoother", "Model → Analysis → Kalman filter…",
                'mod.solve().filter(data, ["y", "pi", "i"], mod.shock_sd)'),
    Translation("Dynare", ("forecast",), "forecast", "Model → Analysis → Forecast from estimated model…",
                'est.forecast(("2025Q1", "2026Q4"))'),
    Translation("Dynare", ("shock_decomposition",), "shock decomposition", "Model → Analysis → Shock decomposition…",
                "est.decompose()"),
    Translation("Dynare", ("identification",), "identification", "Model → Analysis → Identification…",
                "est.identification()"),
    Translation("Dynare", ("dynare_sensitivity",), "prior stability mapping", "Model → Analysis → Stability mapping…",
                "stability_mapping(mod, draws=2000, seed=1)"),
    Translation("Dynare", ("osr",), "optimal simple rule", "Model → Policy → Optimal simple rule…",
                'osr(mod, ["phi_pi", "phi_y"], {"pi": 1.0, "y": 0.25})'),
    Translation("Dynare", ("ramsey_model", "discretionary_policy"), "optimal policy", "Model → Policy → Optimal policy…",
                'optimal_policy(mod, "i", {"pi": 1.0, "y": 0.25})'),
    Translation("Dynare", ("occbin_solver",), "occasionally binding constraints",
                "Model → Solution → Occasionally binding constraints…", 'occbin(mod, {"ey": {1: -3.0}})'),
    Translation("Dynare", ("perfect_foresight_solver", "simul"), "perfect foresight", "Model → Solution → Solve model…",
                'model.solve(data, ("2020Q1", "2024Q4"), solver=PerfectForesight())'),
)

PACKAGES = {"EViews": ("EViews",), "IRIS": ("IRIS", "Dynare")}  # the help topic each package's rows go in
_WORD = re.compile(r"^[A-Za-z0-9_]+$")  # a command typed as one word: a palette synonym


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def translation_table(*packages: str) -> str:
    """The rows of ``packages`` as a Markdown table: familiar command, where it is in the app, the Python call."""
    rows = ["| Familiar | In the app | In Python |", "|---|---|---|"]
    for t in TRANSLATIONS:
        if t.package not in packages:
            continue
        familiar = ", ".join(f"`{c}`" for c in t.commands) + f": {t.does}"
        if len(packages) > 1:
            familiar = f"{t.package} " + familiar
        rows.append(f"| {_cell(familiar)} | {_cell(t.menu) or '(code only)'} | "
                    f"{('`' + _cell(t.python) + '`') if t.python else ''} |")
    return "\n".join(rows)


def palette_synonyms() -> list[tuple[str, str, str, str]]:
    """The command palette's synonyms, from the tables: (familiar word, package, menu path, help topic id). A row
    with a menu path finds that menu item; a row without one finds its help topic."""
    out = []
    for t in TRANSLATIONS:
        topic = next(f"coming-from-{key.lower()}" for key, members in PACKAGES.items() if t.package in members)
        out.extend((c, t.package, t.menu, topic) for c in t.commands if _WORD.match(c))
    return out


def _eviews_text() -> str:
    from .help import link

    return _with_table(f"""
    pymetrix covers most of what EViews is used for in applied macroeconomics, from a data file to a forecast
    report. This page says where the familiar things are. Find anything (Ctrl+K) knows the EViews names too: type
    genr, smpl or ls and it offers the pymetrix command.

    ## The objects

    - **Workfile**: a project (File → Save project, a .pymx file), which keeps the data and every step that made
      the results, and rebuilds them when opened. A **page** is a Dataset: one per frequency or panel structure,
      as many as needed in a project.
    - **Series**: a column of a Dataset; `data["y"]` is a pandas Series indexed by period. Lags are written
      `y[-1]`, not y(-1); functions such as `d`, `dlog`, `log` and `@pch` (written `pch`) work as in EViews.
    - **Group**: a selection of series (Data → Select series…, `data.series("c", "i")`).
    - **Sample**: the dataset's active sample (Data → Sample → Set sample…); a named sample is a Sample.
    - **Equation object**: an Equation is the specification; estimating it gives a FittedEquation, whose window
      tabs are the views (estimation output, residuals, tests, forecast). Coefficients are named, not
      c(1): `fit.params["y"]`.
    - **System** and **VAR**: System gives a FittedSystem, VAR a FittedVAR.
    - **Model object**: a Model of estimated equations and identities; **scenarios** are Scenario objects
      (shift, set, fix, hold and free), and `scenarios(...)` solves several and compares them (a ScenarioSet).
    - **Spool**, **report** and frozen tables: every result is already an object; collect them in the report
      outline (Add to report, under any table or figure; View → Report) and export it (File → Export report…), or
      write a Report in code.
    - **Program** and the command window: Python scripts (File → New script) and the Immediate window
      (View → Immediate).

    ## Commands and menus

    In the Python column, data is a Dataset, fit an estimated equation, var an estimated VAR and model a Model.
    Each menu item records its Python in the History and Script windows, and each dialog field's tooltip names its
    argument.

    {{table}}

    ## Defaults that differ

    The ADF test chooses its lags by AIC, where EViews uses the Schwarz criterion (`lags="bic"` matches it); the HP
    filter's lambda follows the frequency, as in EViews. These, the robust and HAC standard errors and the VAR and
    SUR residual covariances are set out, with the option that matches, in
    [Defaults, and other packages' defaults]({link("topic", "defaults")}). One more: a series made under the active
    sample is computed over all the data (the active sample only chooses the rows that estimation, tests and graphs
    use); to assign only within a sample, as `genr` under `smpl` does, give it under **Within** in Generate series
    (`data.with_series(y="x / 2", within=("2000Q1", "2019Q4"))`).

    ## Different by design

    - **Every step is recorded.** Each dialog runs one line of Python, listed in the History and Script windows;
      File → Export script… saves them as a script that reproduces the work outside the app.
    - **Results are not changed in place.** Estimating again makes a new result; `data.with_series(...)` returns a
      new dataset. A derived series keeps its formula and follows its inputs (EViews' `frml`).
    - **Undo.** Edit → Undo (Ctrl+Z) takes back the last dialog step, up to 20; `session.undo()` in code.
    - **Projects rebuild.** A project saves the data and the steps; results are recomputed when it opens, and a
      result whose inputs changed is marked out of date.

    ## Bringing work across

    Seasonal adjustment: X-13ARIMA-SEATS and TRAMO-SEATS come with the seasonal plugin, pymetrix-seasonal
    (`pip install "pymetrix-seasonal[jdemetra]"`); installed, they are methods in the same Seasonal adjustment
    dialog, beside STL and classical decomposition, which need nothing more.

    Data: save the workfile's series as Excel or CSV and use File → Load data…; workfiles (.wf1) are not read.
    Models: `read_eviews` reads a model's text or a program (.prg) into a template, fitting its `equation eq.ls`
    specifications to data when given; `read_eviews_xml` reads an FRB/US-style XML model file and
    `read_eviews_sspace` a state-space object. In the app, Model → Model editor… reads EViews model text as typed,
    and its Write as button translates a model to pymetrix's syntax, Dynare's or IRIS's (Model → Export model…;
    `export_model` in code). `Model.from_file` reads any of these by its extension.
    """, "EViews")


def _iris_text() -> str:
    from .help import link

    return _with_table(f"""
    pymetrix reads IRIS and Dynare model files and works with them the way those toolboxes do: steady state,
    solution, simulation, the Kalman filter and smoother, estimation and posterior simulation, forecasts and
    decompositions. MATLAB is not needed. Find anything (Ctrl+K) knows the familiar names: type sstate,
    stoch_simul or estimated_params and it offers the pymetrix command.

    ## The objects

    - **IRIS `model` object**: `read_iris` reads a .model file into a template, with its measurement equations and
      shock sizes (the `std_` parameters); a Dynare .mod file read by `read_mod` gives a model that works the same
      way. Parameters set in code in IRIS are given with `params=`.
    - **`sstate` and `solve`**: `nk.steady_state()` and `sol = nk.solve()`; `sol.rules()` shows the decision rules.
    - **`simulate`** with a **Plan**: a Scenario on the solution, with shock paths and targets held by free shocks
      (IRIS's exogenize and endogenize).
    - **`filter`**: the Kalman filter and smoother, on the solution or through the estimated model.
    - **`estimate`**: maximum likelihood or the posterior mode, then Metropolis-Hastings or sequential Monte Carlo
      for the posterior.
    - **Databank**: a Dataset, one frequency, series as columns; dates are written `"2020Q1"`, not qq(2020,1).
    - **`Series`** (`tseries`): a column of a Dataset, a pandas Series indexed by period.
    - **Reports**: the report outline (View → Report) and Report in code, saved as Word, HTML, Excel or Markdown.

    ## Commands and menus

    In the Python column, nk is a model read from an IRIS file (or a template), sol its solution, est the
    estimated model, mod a model read from a Dynare file and data a Dataset. The Dynare rows give Dynare's command,
    block or option.

    {{table}}

    ## Defaults that differ

    Responses of log variables are log deviations, as in IRIS. Priors are given by mean and standard deviation, as
    in Dynare. Other differences from other packages are in
    [Defaults, and other packages' defaults]({link("topic", "defaults")}).

    ## Different by design

    - **Every step is recorded.** Each dialog runs one line of Python, listed in the History and Script windows;
      File → Export script… saves them as a script that runs in plain Python.
    - **Results are objects.** A solution, an estimate or a posterior is kept under a name, with the settings and
      data that made it.
    - **Undo.** Edit → Undo (Ctrl+Z) takes back the last dialog step; `session.undo()` in code.
    - **Projects rebuild.** A project saves the data and the steps and recomputes the results when opened.

    ## Bringing models across

    `read_iris` reads .model files (with `!for` loops, `!substitutions` and `!dtrends`), `read_mod` reads .mod
    files (the macro processor, `steady_state_model`, `shocks`, `estimated_params`, `varobs` and the estimation
    options); commands in the file are listed, not run. In the app: Model → Import → IRIS file… and
    Model → Import → Dynare file…. Model → Model editor… reads pymetrix's syntax, Dynare's, IRIS's and EViews's,
    and its Write as button translates a model into another (Model → Export model…; `export_model` in code), with
    notes on what does not carry across and a check that it solves the same. Data: a databank saved as CSV
    (`databank.toCSV`, `dbsave`) loads with File → Load data… once the rows of classes and units under the names
    are taken out (`skiprows=[1, 2]` in code); .mat databanks are not read.
    """, "IRIS", "Dynare")


def _with_table(text: str, *packages: str) -> str:
    """``text`` (indented, as written in the source) without its indent, its ``{table}`` the packages' table."""
    return textwrap.dedent(text).strip().replace("{table}", translation_table(*packages))


def add_topics() -> None:
    """The help topics "Coming from EViews" and "Coming from MATLAB/IRIS and Dynare"."""
    from .help import add_topic

    add_topic("coming-from-eviews", "Coming from EViews", _eviews_text(), (),
              ("defaults", "samples", "derived-series", "model-solution", "reports", "terms"))
    add_topic("coming-from-iris", "Coming from MATLAB/IRIS and Dynare", _iris_text(), (),
              ("defaults", "forward-looking-models", "state-space", "model-solution", "terms"))
