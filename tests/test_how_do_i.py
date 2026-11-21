"""The task pages ("How do I…", src/pymetrix/howto): each follows the template, its code runs as written and gives the
numbers it quotes, its steps in the app give the same results as its code (each step run as the dialog's command,
with the fields as the page names them), its links resolve, and the help, the Help menu and Ctrl+K reach it.

Independent checks: Breusch-Godfrey against statsmodels' on the same regression built by hand; Engle-Granger's
statistic against statsmodels' coint; the rational-expectations solution substituted back into the model's equations
(its expectations from the decision rules), and the IRIS file's solution against the Dynare file's."""

from __future__ import annotations

import os
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from pymetrix import help as hp
from pymetrix import howto

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "how-do-i" / "README.md"
PAGES = {p.path.stem: p for p in howto.pages()}
SECTIONS = ["Before you start", "In the app", "In code", "Check the result", "Pitfalls", "See also"]


def _flat(text: str) -> str:
    """The page's prose as one line, without Markdown's backticks and bold: for finding what it names."""
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", ""))


def _files(text: str) -> dict[str, str]:
    """The files the page shows in ```text <name> fences (indented in a list item or not), as written."""
    out = {}
    for m in re.finditer(r"^([ \t]*)```text ([^\s`]+)[ \t]*\n(.*?)^\1```", text, flags=re.M | re.S):
        out[m.group(2)] = textwrap.dedent(m.group(3))
    return out


def _code(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)


def _run(stem: str, folder: Path) -> dict:
    """The page's code blocks in order, as one script, in ``folder`` with the page's files written out."""
    page = PAGES[stem]
    for name, content in _files(page.text).items():
        (folder / name).write_text(content, encoding="utf-8")
    scope: dict = {"__name__": f"howto_{stem}"}
    here = os.getcwd()
    os.chdir(folder)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for i, block in enumerate(_code(page.text), 1):
                exec(compile(block, f"{page.path.name}, block {i}", "exec"), scope)  # noqa: S102 - our own page
    finally:
        os.chdir(here)
    scope["__folder__"] = folder
    return scope


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> dict:
    """Every page run once."""
    return {stem: _run(stem, tmp_path_factory.mktemp(stem)) for stem in PAGES}


# -- the template ------------------------------------------------------------------------------------------------------


def test_the_first_three_pages_are_there():
    assert list(PAGES) == ["estimate-and-test-an-equation", "bring-in-an-iris-or-dynare-model", "cointegration-in-a-short-sample"]


@pytest.mark.parametrize("stem", sorted(PAGES))
def test_each_page_follows_the_template(stem):
    page = PAGES[stem]
    meta = hp._front_matter(page.path.read_text(encoding="utf-8"))[0]
    assert {"summary", "audience", "order"} <= set(meta) and page.summary and page.audience
    lines = page.text.splitlines()
    assert lines[0].startswith("# ") and lines[0][2:] == page.title
    goal = page.text.split("\n\n")[1]
    assert not goal.startswith("#") and len(goal.split()) >= 15  # the goal, before the first section
    headings = re.findall(r"^## (.+)$", page.text, flags=re.M)
    assert headings == SECTIONS
    steps = page.text.split("## In the app")[1].split("\n## ")[0]
    assert re.search(r"^1\. ", steps, flags=re.M) and "→" in steps  # numbered steps with menu paths
    assert _code(page.text), "a page has code"
    pitfalls = page.text.split("## Pitfalls")[1].split("\n## ")[0]
    assert all(item.startswith("- **") for item in re.findall(r"^- .*", pitfalls, flags=re.M))


def test_the_plan_lists_every_page_and_the_template():
    plan = PLAN.read_text(encoding="utf-8")
    for page in PAGES.values():
        assert f"](../../src/pymetrix/howto/{page.path.name})" in plan and page.title in plan
    for section in SECTIONS:
        assert f"**{section}**" in plan
    rows = re.findall(r"^\| (\d+) \|", plan, flags=re.M)
    assert len(rows) >= 10 and rows == [str(k) for k in range(1, len(rows) + 1)]  # 10 to 15 tasks, in priority order


# -- the code runs and gives the numbers the pages quote -------------------------------------------------------------


def test_estimate_and_test_an_equation(ran):
    text = _flat(PAGES["estimate-and-test-an-equation"].text)
    s = ran["estimate-and-test-an-equation"]
    fit, bg, ecm, white = s["fit"], s["bg"], s["ecm"], s["ecm_white"]
    assert (round(bg.statistic, 2), round(bg.p_value, 4)) == (23.46, 0.0001) and bg.reject(0.05)
    assert "23.46 and 0.0001" in text and "p-value 0.0001" in text
    assert round(fit.correlogram(lags=8).to_frame()["AC"].iloc[0], 2) == -0.40 and "-0.40" in text
    table = ecm.diagnostics().to_frame().set_index("test")["p-value"]
    assert round(table["Breusch-Godfrey LM"], 2) == 0.78 and round(table["Jarque-Bera"], 3) == 0.053
    assert round(table["Breusch-Pagan (Koenker)"], 4) == 0.0007 and table["RESET"] > 0.05
    assert "Breusch-Godfrey 0.78, Jarque-Bera 0.053; Breusch-Pagan 0.0007" in text
    assert "no longer rejects (0.78), nor do Jarque-Bera (0.053) and RESET; Breusch-Pagan does (0.0007)" in text
    assert not ecm.test("breusch_godfrey", lags=4).reject(0.05) and ecm.test("breusch_pagan").reject(0.05)
    assert (len(fit.resid), len(ecm.resid)) == (98, 99) and "98 observations" in text and "99 for ecm" in text
    assert [str(p) for p in fit.provenance.sample] == ["2000Q3", "2024Q4"] and [str(p) for p in ecm.provenance.sample] == ["2000Q2", "2024Q4"]
    np.testing.assert_allclose(white.params.values, ecm.params.values, rtol=1e-12)  # White: the same coefficients
    assert (round(ecm.params["log(c[-1])"], 2), round(ecm.params["log(y[-1])"], 2)) == (-0.75, 0.72) and "-0.75" in text
    k = list(ecm.params.names).index("log(c[-1])")
    assert (round(white.params.se[k], 3), round(ecm.params.se[k], 3)) == (0.137, 0.097) and "0.137 against 0.097" in text
    assert "White (HC1) standard errors" in str(white.summary())


def test_breusch_godfrey_is_statsmodels_on_the_same_regression(ran):
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey

    frame = ran["estimate-and-test-an-equation"]["data"].frame
    dlc, dly = np.log(frame["c"]).diff(), np.log(frame["y"]).diff()
    rows = pd.concat([dlc, dly, dlc.shift()], axis=1).dropna()
    by_hand = sm.OLS(rows.iloc[:, 0], sm.add_constant(rows.iloc[:, 1:])).fit()
    stat, p = acorr_breusch_godfrey(by_hand, nlags=4)[:2]
    bg = ran["estimate-and-test-an-equation"]["bg"]
    assert bg.statistic == pytest.approx(stat, rel=1e-9) and bg.p_value == pytest.approx(p, rel=1e-9)


def test_bring_in_an_iris_or_dynare_model(ran):
    text = _flat(PAGES["bring-in-an-iris-or-dynare-model"].text)
    s = ran["bring-in-an-iris-or-dynare-model"]
    nk, sol, mod, irf = s["nk"], s["sol"], s["mod"], s["irf"]
    assert nk.check().ok and (sol.n_stable, sol.n_predetermined) == (1, 1)
    assert "Blanchard-Kahn: stable roots 1 (unit roots included), predetermined variables 1" in str(sol.roots())
    assert "Blanchard-Kahn: stable roots 1 (unit roots included), predetermined variables 1" in text
    assert "0.5406    stable" in str(sol.roots()) and "The stable root is 0.54" in text
    rules = sol.rules().to_frame().set_index("")
    assert round(rules.loc["y", "ei"], 2) == -1.97 and "-1.97" in text
    assert round(irf.to_frame()["y <- ei"].iloc[0], 3) == -0.394 and "-0.394" in text
    assert irf.to_frame()["y <- ei"].iloc[0] == pytest.approx(0.2 * rules.loc["y", "ei"], rel=1e-12)  # one s.d.: 0.2
    assert all(abs(v) < 1e-12 for v in nk.steady_state().values.values())
    assert mod.commands == ("steady", "check", "stoch_simul(order=1, irf=20) y pi i") and str(mod.commands) in text
    assert nk.shock_sd == mod.shock_sd == {"ey": 0.5, "epi": 0.3, "ei": 0.2}
    from pymetrix.models import ModelError

    with pytest.raises(ModelError, match=r"no values for the parameters \['phi'\]"):  # what the page says of phi
        from pymetrix import read_iris

        read_iris(str(s["__folder__"] / "nk.model"))
    with pytest.raises(ModelError, match="not unique"):  # phi=0.8: indeterminacy, as the page says
        read_iris(str(s["__folder__"] / "nk.model"), params={"phi": 0.8}).solve()
    assert "phi=0.8" in text and "The solution is not unique" in text


def test_the_solution_satisfies_the_models_equations(ran):
    """The decision rules substituted into the three equations, with the expectations they imply, leave nothing."""
    s = ran["bring-in-an-iris-or-dynare-model"]
    rules = s["sol"].rules().to_frame().set_index("")
    sig, bet, kap, rho, phi = 1, 0.99, 0.1, 0.8, 1.5
    rng = np.random.default_rng(0)
    for _ in range(5):
        lag, e = rng.normal(), dict(zip(("ey", "epi", "ei"), rng.normal(size=3)))
        x = {v: rules.loc[v, "i[-1]"] * lag + sum(rules.loc[v, k] * e[k] for k in e) for v in ("y", "pi", "i")}
        ahead = {v: rules.loc[v, "i[-1]"] * x["i"] for v in ("y", "pi")}  # E_t x_{t+1}: the shocks' mean is zero
        assert x["y"] == pytest.approx(ahead["y"] - sig * (x["i"] - ahead["pi"]) + e["ey"], abs=1e-12)
        assert x["pi"] == pytest.approx(bet * ahead["pi"] + kap * x["y"] + e["epi"], abs=1e-12)
        assert x["i"] == pytest.approx(rho * lag + (1 - rho) * (phi * x["pi"] + 0.5 * x["y"]) + e["ei"], abs=1e-12)
    np.testing.assert_allclose(s["dyn"].irf(20).values, s["sol"].irf(20).values, atol=1e-12)


def test_cointegration_in_a_short_sample(ran):
    text = _flat(PAGES["cointegration-in-a-short-sample"].text)
    s = ran["cointegration-in-a-short-sample"]
    annual, eg, mk, adf_ly, adf_lc = s["annual"], s["eg"], s["eg_mackinnon"], s["adf_ly"], s["adf_lc"]
    assert len(annual.frame) == 25 and str(annual.frame.index[0]) == "2000" and str(annual.frame.index[-1]) == "2024"
    assert (round(eg.statistic, 2), eg.p_value) == (-5.40, 0.007) and "-5.40 and 0.007" in text
    assert round(eg.critical_values["5%"], 2) == -4.48 and round(mk.critical_values["5%"], 2) == -4.20
    assert "-4.48 against -4.20" in text and eg.distribution == "simulated for this sample" and mk.distribution == "MacKinnon"
    assert round(adf_ly.p_value, 2) == 0.87 and adf_lc.p_value > 0.1 and "0.87" in text
    assert round(adf_ly.critical_values["5%"], 2) == -3.96 and "-3.96 against -3.61" in text
    assert round(annual.test("adf", "ly", trend="ct", max_lags=1).critical_values["5%"], 2) == -3.61
    assert round(eg.details["p-value MCSE"], 3) == 0.002 and "0.007 and 0.002" in text
    note = "simulated for this sample: 2000 draws of 2 independent Gaussian random walks of 25 observations, trend ct"
    assert note in eg.details["note"] and note in text
    other = annual.test("engle_granger", "ly, lc", trend="ct", max_lags=1, critical_values="simulated")
    assert (round(other.statistic, 2), round(other.p_value, 3)) == (-5.29, 0.009) and "-5.29, p-value 0.009" in text
    assert eg.provenance.seed == 0 and eg.provenance.method["simulation"]["draws"] == 2000
    import statsmodels.tsa.stattools as st  # the statistic is statsmodels' coint's

    frame = annual.frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = st.coint(frame["lc"].to_numpy(), frame[["ly"]].to_numpy(), trend="ct", maxlag=1, autolag="aic")
    assert eg.statistic == pytest.approx(ref[0], rel=1e-9) and mk.p_value == pytest.approx(ref[1], rel=1e-9)
    assert int(np.ceil(12 * (25 / 100) ** 0.25)) == 9 and "up to 9 lags" in text  # statsmodels' default largest lag


# -- the steps in the app: the dialogs' commands, with the fields as the page names them ------------------------------

# Each step: the command, the page's "Label: value" for each field (labels as the dialog shows them), the fields the
# page says are under Advanced settings, and the sub-form (Test settings) the same way. "{folder}" is the page's files.
STEPS = {
    "estimate-and-test-an-equation": [
        ("LoadExample", {"Example dataset": "sample_macro", "Save as": "data"}, {}),
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + dlog(c)[-1]", "Method": "OLS", "Save as": "fit"}, {}),
        ("RunTest", {"Object to test": "fit", "Test": "breusch_godfrey", "Save as": "bg"}, {"Lags": "4"}),
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", "Save as": "ecm"}, {}),
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", "Standard errors": "White (HC1)",
                         "Save as": "ecm_white"}, {}),
    ],
    "bring-in-an-iris-or-dynare-model": [
        ("ImportIris", {"IRIS file": "{folder}/nk.model", "Parameters": "phi=1.5", "Save as": "nk"}, {}),
        ("SolveRationalExpectations", {"Equations": "nk", "Save as": "sol"}, {}),
        ("ImportDynare", {"Dynare file": "{folder}/nk.mod", "Save as": "mod"}, {}),
        ("SolveRationalExpectations", {"Equations": "mod", "Save as": "dyn"}, {}),
    ],
    "cointegration-in-a-short-sample": [
        ("LoadExample", {"Example dataset": "sample_macro", "Save as": "data"}, {}),
        ("ConvertFrequency", {"Data": "data", "New frequency": "Annual", "Save as": "annual",
                              "Combine periods by": "By measure (each series' metadata)"}, {}),
        ("DeriveSeries", {"Data": "annual", "New series name": "lc", "Formula": "log(c)"}, {}),
        ("DeriveSeries", {"Data": "annual", "New series name": "ly", "Formula": "log(y)"}, {}),
        ("RunTest", {"Object to test": "annual", "Test": "adf", "Save as": "adf_ly"},
         {"Series": "ly", "Deterministic terms": "Constant and trend", "Critical values": "Simulated for this sample",
          "Maximum lags": "1"}),
        ("RunTest", {"Object to test": "annual", "Test": "engle_granger", "Save as": "eg"},
         {"Series": "lc, ly", "Deterministic terms": "Constant and trend", "Critical values": "Simulated for this sample",
          "Maximum lags": "1"}),
        ("RunTest", {"Object to test": "annual", "Test": "engle_granger", "Save as": "eg_mackinnon"},
         {"Series": "lc, ly", "Deterministic terms": "Constant and trend", "Critical values": "MacKinnon (asymptotic)",
          "Maximum lags": "1"}),
    ],
}
ADVANCED = {"Maximum lags"}  # what the pages say is under Advanced settings


def _value(spec, shown: str):
    """A field's value from what the dialog shows: a fixed choice by its words, a number field's number, else the text
    typed."""
    if spec.choices:
        found = [c for c in spec.choices if spec.display(c) == shown]
        assert found, (spec.label, shown, [spec.display(c) for c in spec.choices])
        return found[0]
    return int(shown) if spec.kind == "number" else shown


def _command(cls, fields: dict, sub: dict, session, folder: Path, text: str):
    from pymetrix.forms import form_spec
    from pymetrix.plugins import registry

    specs = {s.label: s for s in form_spec(cls)}
    values = {}
    for label, shown in fields.items():
        assert label in specs, (cls.__name__, label, sorted(specs))
        assert specs[label].advanced == (label in ADVANCED), (cls.__name__, label)
        named = Path(shown).name if shown.startswith("{folder}") else shown
        if label != "Save as" or shown != "dyn":  # the page leaves the second solution's name at its default
            assert f"{label}: {named}" in text, f"the page names {label}: {named}"
        values[specs[label].name] = _value(specs[label], shown.format(folder=folder))
    if sub:
        test = registry("tests").get(values["test"]).obj
        subspecs = {s.label: s for s in form_spec(test)}
        parts = []
        for label, shown in sub.items():
            assert label in subspecs, (test.__name__, label, sorted(subspecs))
            assert subspecs[label].advanced == (label in ADVANCED), (test.__name__, label)
            assert f"{label}: {shown}" in text, f"the page names {label}: {shown}"
            parts.append(f"{subspecs[label].name}={_value(subspecs[label], shown)!r}")
        values["options"] = ", ".join(parts)
    return cls(**values)


@pytest.mark.parametrize("stem", sorted(STEPS))
def test_the_steps_in_the_app_give_the_codes_results(stem, ran):
    from pymetrix.commands import COMMANDS, Session

    by_name = {i.cls.__name__: i.cls for i in COMMANDS.values()}
    text = _flat(PAGES[stem].text)
    code = ran[stem]
    folder = code["__folder__"]
    session = Session()
    for name, fields, sub in STEPS[stem]:
        session.run(_command(by_name[name], fields, sub, session, folder, text))
    from pymetrix import Dataset, DynareModel, FittedEquation
    from pymetrix.models.linear import LinearSolution
    from pymetrix.models.templates import Template
    from pymetrix.stats.base import TestResult

    compared = 0
    for key, obj in session.objects.items():
        mine = code.get(key)
        if mine is None:
            continue
        assert type(obj) is type(mine), key
        if isinstance(obj, FittedEquation):
            np.testing.assert_allclose(obj.params.values, mine.params.values, rtol=1e-12)
            np.testing.assert_allclose(obj.params.se, mine.params.se, rtol=1e-12)
        elif isinstance(obj, TestResult):
            assert (obj.statistic, obj.p_value, obj.critical_values) == (mine.statistic, mine.p_value, mine.critical_values), key
        elif isinstance(obj, LinearSolution):
            np.testing.assert_allclose(obj.irf(20).values, mine.irf(20).values, atol=1e-12)
        elif isinstance(obj, Dataset):
            pd.testing.assert_frame_equal(obj.frame[list(mine.frame.columns)], mine.frame, check_freq=False)
        elif isinstance(obj, (Template, DynareModel)):
            assert obj.shock_sd == mine.shock_sd and obj.solve().rules().to_frame().equals(mine.solve().rules().to_frame())
        else:
            raise AssertionError(f"{key}: {type(obj).__name__} not compared")
        compared += 1
    assert compared == len(session.objects) - (stem == "cointegration-in-a-short-sample")  # all but that page's data
    ns = dict(session.objects)
    exec(session.script(), ns)  # noqa: S102 - the script the steps recorded replays
    assert set(session.objects) <= set(ns)


def test_what_the_pages_say_of_the_windows_and_dialogs(ran):
    """The views and lists the pages tell readers to look at are the app's."""
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.forms import form_spec
    from pymetrix.views import sections

    s1, s2 = ran["estimate-and-test-an-equation"], ran["bring-in-an-iris-or-dynare-model"]
    views = [t for t, _ in sections(s1["fit"])]
    assert {"Summary", "Diagnostics", "Correlogram", "Q-Q plot", "Residuals"} <= set(views)
    assert list(s1["fit"].diagnostics().to_frame()["test"]) == ["Breusch-Godfrey LM", "Breusch-Pagan (Koenker)", "Jarque-Bera", "RESET"]
    assert [t for t, _ in sections(s2["nk"])] == ["Equations", "Parameters", "Measurement", "Shocks"]
    assert [t for t, _ in sections(s2["sol"])] == ["Decision rules", "Graph", "Steady state", "Roots"]
    assert "Commands" in [t for t, _ in sections(s2["mod"])]
    from pymetrix.ui.next_steps import NEXT_STEPS

    assert ("Test…", "RunTest", "source", "estimate") in NEXT_STEPS["FittedEquation"]  # Next ▸ → Test…
    # Check model… takes an imported model (the page once had a pitfall saying it did not)
    session = Session()
    session.objects.update(nk=s2["nk"], mod=s2["mod"])
    check = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "CheckModel")
    equations = next(spec for spec in form_spec(check) if spec.name == "equations")
    assert session.choices(equations, {}) == ["mod", "nk"]
    solve = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "SolveRationalExpectations")
    assert session.choices(next(spec for spec in form_spec(solve) if spec.name == "equations"), {}) == ["mod", "nk"]
    pytest.importorskip("PySide6")  # the model editor's syntax names are in its Qt module
    from pymetrix.ui.model_editor_qt import LABELS

    assert {"IRIS (.model)", "Dynare (.mod)"} <= set(LABELS.values())


# -- links, the help, the menu and Ctrl+K -----------------------------------------------------------------------------


def _anchor(heading: str) -> str:
    """GitHub's anchor for a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


@pytest.mark.parametrize("stem", sorted(PAGES))
def test_every_link_resolves(stem):
    page = PAGES[stem]
    for label, target in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", page.text):
        if target.startswith("pymetrix:topic/"):
            hp.topic(target.removeprefix("pymetrix:topic/"))  # raises if there is no such topic
            continue
        assert "://" not in target, target  # no web links: the repository is not public yet
        file, _, anchor = target.partition("#")
        # the links are relative to the page's place in the repository, also when the tests import an installed wheel
        path = (ROOT / "src" / "pymetrix" / "howto" / page.path.name).parent.joinpath(file).resolve()
        assert path.exists(), target
        if anchor:
            headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), flags=re.M)
            assert anchor in {_anchor(h) for h in headings}, target
    shown = hp.topic(page.id).text
    targets = re.findall(r"\]\(([^)\s]+)\)", shown)
    assert targets and all(t.startswith("pymetrix:") for t in targets)  # the help follows every link it keeps
    assert "../" not in shown and not shown.startswith("# ")  # repository links became their text; the title is the topic's


def test_the_help_lists_the_pages_first():
    hp._load_all()
    index = hp.topic(howto.INDEX)
    assert index.title == "How do I…"
    for page in PAGES.values():
        assert f"[{page.title}](pymetrix:topic/{page.id}): {page.summary}" in index.text
        assert hp.topic(page.id).title == page.title
    sections = hp.contents()
    title, pages = sections[0]
    assert title == "How do I…" and pages[0] == ("How do I…", "pymetrix:topic/how-do-i")
    assert [u for _, u in pages[1:]] == [f"pymetrix:topic/{p.id}" for p in PAGES.values()]
    background = dict(sections)["Background"]
    assert not [u for _, u in background if "howto-" in u or u.endswith("/how-do-i")]  # listed once
    assert "pymetrix:topic/how-do-i" in hp.home_markdown()
    found = [obj.id for kind, _, obj in hp.search("cointegration short annual") if kind == "topic"]
    assert found[0] == "howto-cointegration-in-a-short-sample"
    assert "## Pitfalls" in hp.page_for_link("pymetrix:topic/howto-estimate-and-test-an-equation")


@pytest.fixture(scope="module")
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    w = main_window(Session())
    yield w
    w.state["session"].dirty = False
    w.close()


def test_help_how_do_i_opens_the_index(window):
    from PySide6.QtWidgets import QWidget

    actions = {a.text().replace("&", ""): a for a in window.help_menu.actions()}
    actions["How do I…"].trigger()
    helps = [w for w in window.findChildren(QWidget) if w.windowTitle() == "PyMetrix help" and hasattr(w, "view") and w.isVisible()]
    assert helps and helps[0].view.current() == "pymetrix:topic/how-do-i"
    assert "Test for cointegration in a short annual sample" in helps[0].view.text.toPlainText()
    helps[0].close()


def test_ctrl_k_finds_each_page(window):
    from pymetrix.ui.palette import rank

    entries = window.palette_entries()
    for words, stem in (("residuals", "estimate-and-test-an-equation"), ("cointegration short", "cointegration-in-a-short-sample"),
                        ("dynare model", "bring-in-an-iris-or-dynare-model"), ("iris", "bring-in-an-iris-or-dynare-model")):
        found = [e for e in rank(entries, words) if e.kind == "Help" and e.title == PAGES[stem].title]
        assert found and found[0].detail == "How do I…", words
    titles = [e.title for e in rank(entries, "how do i")]
    assert "Help → How do I…" in titles and "How do I…" in titles
    assert rank(entries, "var")[0].title == "Estimate → VARs and local projections → VAR…"  # menu items still lead
