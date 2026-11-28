"""The task pages ("How do I…", src/pymetrix/howto): each follows the template, its code runs as written and gives the
numbers it quotes, its steps in the app give the same results as its code (each step run as the dialog's command,
with the fields as the page names them), its links resolve, and the help, the Help menu and Ctrl+K reach it.

Independent checks: Breusch-Godfrey against statsmodels' on the same regression built by hand; Engle-Granger's
statistic against statsmodels' coint; the rational-expectations solution substituted back into the model's equations
(its expectations from the decision rules), and the IRIS file's solution against the Dynare file's; the connectors'
series against the recorded responses read by hand (no page's test uses the network); the VAR against statsmodels' VAR
(coefficients, the Cholesky factor, the forecast); the conditional forecast against the least-norm shocks worked out
by hand; the seasonal factors against the pattern the series was made from and statsmodels' STL; the EViews model's
impact multiplier against 1 / (1 - c(2)) and its program's estimates against least squares by NumPy; the AR(1)
backtest against forecasts made by hand at each origin; the reconciliation against the same constrained least-squares
problem solved through its KKT system; the translated EViews program's equation against least squares by NumPy on
the CSV file."""

from __future__ import annotations

import contextlib
import importlib.util
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
from pymetrix.commands.project import ProjectTrustWarning

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


FIXTURES = ROOT / "tests" / "fixtures" / "connectors"
ROUTES = [  # (a pattern in the URL, the recorded response): the first that matches answers; as the connectors' tests
    ("ecb.europa.eu/service/dataflow/ECB/EXR/latest?references=all", "search/ecb_exr_structure.xml"),
    ("ecb.europa.eu/service/dataflow/ECB", "search/ecb_dataflows.xml"),
    ("/categoryscheme/", "search/sdmx_no_flow.xml"),
    ("data-api.ecb.europa.eu/service/data/", "ecb_exr.csv"),
    ("fredgraph.csv?id=GDPC1", "fred_gdpc1.csv"),
    ("fredgraph.csv?id=UNRATE", "fred_unrate.csv"),
    ("worldbank.org/v2/sources?", "search/wb_sources.json"),
    ("worldbank.org/v2/source/2/indicator", "search/wb_source2_indicators_p{page}.json"),
    ("worldbank.org/v2/topic?", "search/wb_topics.json"),
    ("worldbank.org/v2/country?", "search/wb_countries.json"),
    ("worldbank.org/v2/country/WSM;TON/indicator/NY.GDP.MKTP.CD", "wb_page{page}.json"),
]


def _recorded(url: str) -> bytes:
    """A connector's request answered from the recorded responses (tests/fixtures/connectors): never the network."""
    page = re.search(r"[?&]page=(\d+)", url)
    for pattern, name in ROUTES:
        if pattern in url:
            return (FIXTURES / name.format(page=page.group(1) if page else 1)).read_bytes()
    raise AssertionError(f"a task page asked for {url}, which has no recorded response")


@contextlib.contextmanager
def _offline(home: Path):
    """The connectors answered from the recorded responses, with a cache and settings of their own."""
    from pymetrix import connectors
    from pymetrix.ext import connectors as ext

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ext, "_get", _recorded)
        mp.setenv("PYMETRIX_HOME", str(home))
        mp.delenv("FRED_API_KEY", raising=False)
        connectors.set_api_key("fred", None)
        yield


def _has_plugin(page) -> bool:
    """The page's plugin is installed: importable, and loaded as a plugin (the core-only run has neither)."""
    if not page.plugin:
        return True
    from pymetrix.plugins import plugins

    return (importlib.util.find_spec(page.plugin.replace("-", "_")) is not None
            and page.plugin in {getattr(p, "name", "") for p in plugins()})


def _run(stem: str, folder: Path) -> dict:
    """The page's code blocks in order, as one script, in ``folder`` with the page's files written out."""
    page = PAGES[stem]
    for name, content in _files(page.text).items():
        (folder / name).write_text(content, encoding="utf-8")
    scope: dict = {"__name__": f"howto_{stem}"}
    here = os.getcwd()
    os.chdir(folder)
    try:
        with warnings.catch_warnings(), _offline(folder / "home"):
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", ProjectTrustWarning)  # the page opens a project without trusting it
            for i, block in enumerate(_code(page.text), 1):
                exec(compile(block, f"{page.path.name}, block {i}", "exec"), scope)  # noqa: S102 - our own page
    finally:
        os.chdir(here)
        import matplotlib.pyplot as plt

        plt.close("all")
    scope["__folder__"] = folder
    return scope


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> dict:
    """Every page run once; a page whose plugin is not installed (the core-only run) is not run."""
    return {stem: _run(stem, tmp_path_factory.mktemp(stem)) for stem, page in PAGES.items() if _has_plugin(page)}


def _ran(ran: dict, stem: str) -> dict:
    if stem not in ran:
        pytest.skip(f"{PAGES[stem].plugin} is not installed")
    return ran[stem]


# -- the template ------------------------------------------------------------------------------------------------------


ORDER = ["estimate-and-test-an-equation", "load-data-with-browse-series", "estimate-a-var", "bring-in-an-iris-or-dynare-model",
         "run-a-forecast-round", "save-and-share-a-project", "cointegration-in-a-short-sample", "seasonally-adjust-a-series",
         "bring-in-an-eviews-model", "run-a-script-in-the-app", "make-a-report", "check-a-forecasts-record",
         "solve-a-large-backward-looking-model", "reconcile-components-to-a-total", "translate-an-eviews-program"]


def test_the_pages_are_there_in_the_plans_order():
    assert list(PAGES) == ORDER
    assert [p.order for p in PAGES.values()] == list(range(1, len(ORDER) + 1))  # the plan's numbers
    plan = PLAN.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| ([^|]+?) \|.*\| (\w[\w ()]*) \|$", plan, flags=re.M)
    assert [title for _, title, _ in rows] == [p.title + (" (the `pymetrix-eviews` plugin)" if p.plugin else "")
                                              for p in PAGES.values()]
    assert {status for _, _, status in rows} == {"written"}
    assert [p.plugin for p in PAGES.values() if p.plugin] == ["pymetrix-eviews"]


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


def test_load_data_with_browse_series(ran):
    """The connectors' series are the recorded responses, read by hand; the script's address reads them again."""
    text = _flat(PAGES["load-data-with-browse-series"].text)
    s = ran["load-data-with-browse-series"]
    assert [f.code for f in s["flows"]] == ["ECB/EXR"] and s["flows"][0].title == "Exchange Rates"
    assert s["address"] == "sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A" and f"'{s['address']}'" in text
    assert [d.id for d in s["exr"].dimensions] == ["FREQ", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX"]
    fx, gdp, wb = s["fx"], s["gdp"], s["wb"]
    raw = pd.read_csv(FIXTURES / "ecb_exr.csv")
    for code, name in (("USD", "usd"), ("GBP", "gbp")):
        rows = raw[raw["CURRENCY"] == code]
        np.testing.assert_allclose(fx[name].to_numpy(), rows["OBS_VALUE"].to_numpy(), rtol=1e-12)
        assert [str(p) for p in fx.frame.index] == list(rows["TIME_PERIOD"])
    assert fx.frame.index.freqstr.startswith("M") and gdp.frame.index.freqstr.startswith("Q") and wb.kind == "panel"
    assert fx.variables["usd"].label == "US dollar/Euro ECB reference exchange rate" and fx.variables["usd"].label in text
    assert fx.variables["usd"].source == "ECB SDMX: EXR/M.USD+GBP.EUR.SP00.A" and fx.variables["usd"].source in text
    assert gdp.variables["gdpc1"].source == "FRED, Federal Reserve Bank of St. Louis: GDPC1" and gdp.variables["gdpc1"].source in text
    raw = pd.read_csv(FIXTURES / "fred_gdpc1.csv")
    np.testing.assert_allclose(gdp["gdpc1"].to_numpy(), raw["GDPC1"].to_numpy(), rtol=1e-12)
    assert sorted(wb.frame.index.get_level_values("country").unique()) == ["TON", "WSM"]
    assert wb.variables["ny_gdp_mktp_cd"].vintage.released is not None  # the World Bank says when it last updated
    assert s["connectors"].to_frame(s["found"])["code"].iloc[0] == "NY.GDP.MKTP.CD"
    from pymetrix import Dataset

    assert Dataset.read(str(s["__folder__"] / "fx.parquet")).frame.equals(fx.frame)  # the values as loaded, kept
    with pytest.raises(ValueError, match="different frequencies"), _offline(s["__folder__"] / "home"):  # the pitfall
        Dataset.read("fred://GDPC1,UNRATE")


def test_the_browser_as_the_page_uses_it(tmp_path):
    """File → Browse series…: the provider, the source, the search, the dataflow's dimensions and the codes chosen, the
    Key line, Name and Load, as the page says them; what it loads is the page's address, recorded as Dataset.read."""
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix import Dataset
    from pymetrix.commands import Session
    from pymetrix.ui.series_browser_qt import series_browser

    text = _flat(PAGES["load-data-with-browse-series"].text)
    with _offline(tmp_path):
        s = Session()
        b = series_browser(s)
        labels = {w.text() for w in b.dialog.findChildren(QtWidgets.QLabel)} | {
            w.text() for w in b.dialog.findChildren(QtWidgets.QPushButton)} | {w.title() for w in b.dialog.findChildren(QtWidgets.QGroupBox)}
        assert {"Provider", "Source", "Search", "Name", "Load", "Add to Load from connector…", "Series key",
                "FRED API key", "Countries (none: all)"} <= labels
        assert "Ask again (skip the cache)" in {w.text() for w in b.dialog.findChildren(QtWidgets.QCheckBox)}
        provider = b.provider.itemText(b.provider.findData("sdmx"))
        assert f"Provider: {provider}, Source: ECB" in text
        b.set_provider("sdmx")
        assert b.source_value() == "ECB"
        b.search("exchange rates")
        b.select_rows([0])
        assert list(b.pickers) == ["FREQ", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX"]
        for dim, codes in (("FREQ", ["M"]), ("CURRENCY", ["USD", "GBP"]), ("CURRENCY_DENOM", ["EUR"]), ("EXR_TYPE", ["SP00"]),
                           ("EXR_SUFFIX", ["A"])):
            b.choose(dim, codes)
            assert f"{dim}: {' and '.join(codes)}" in text
        assert b.key_preview.text() == "sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A" and b.key_preview.text() in text
        b.name.setText("fx")
        b.load()
        assert "fx = Dataset.read('sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A', vintage='latest')" in s.script()
        assert "fx = Dataset.read('sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A', vintage='latest')" in text
        assert s["fx"].frame.sort_index(axis=1).equals(Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A").frame.sort_index(axis=1))
        b.dialog.close()


def test_estimate_a_var(ran):
    """The VAR against statsmodels' on the same series: coefficients, the Cholesky factor of the residual covariance
    (the responses on impact) and the forecast."""
    text = _flat(PAGES["estimate-a-var"].text)
    s = ran["estimate-a-var"]
    var, irf, fc, data = s["var"], s["irf"], s["fc"], s["data"]
    assert var.lags == 1 and var.is_stable and round(max(abs(var.roots)), 2) == 0.33 and "largest root is 0.33" in text
    assert var.provenance.sample == ("2000Q3", "2024Q4") and len(var.resid) == 98 and "2000Q3 to 2024Q4, 98 observations" in text
    frame = irf.to_frame()
    assert round(frame["dli <- dly"].iloc[0], 2) == 0.64 and "0.64: investment's response" in text
    assert round(frame["dly <- dly"].iloc[0], 2) == 1.07 and frame["dly <- dli"].iloc[0] == 0 and "1.07" in text
    f = fc.to_frame()
    assert round(f["dly"].iloc[0], 2) == 0.63 and round(f["dly"].iloc[-1], 2) == 0.87 and "0.63 per cent in 2025Q1, then towards 0.87" in text
    assert round(data.frame["dly"].mean(), 2) == 0.88 and "sample's 0.88" in text
    sm = pytest.importorskip("statsmodels.tsa.api")
    rows = data.frame[["dly", "dli"]].dropna()
    ref = sm.VAR(rows.to_numpy()).fit(1)
    np.testing.assert_allclose([frame["dly <- dly"].iloc[0], frame["dli <- dly"].iloc[0], frame["dli <- dli"].iloc[0]],
                               np.linalg.cholesky(ref.sigma_u)[[0, 1, 1], [0, 0, 1]], rtol=1e-10)
    np.testing.assert_allclose(f[["dly", "dli"]].to_numpy(), ref.forecast(rows.to_numpy()[-1:], 8), rtol=1e-10)
    summary = str(var.summary())
    for row, values in (("const", ref.params[0]), ("dly[-1]", ref.params[1]), ("dli[-1]", ref.params[2])):
        line = next(x for x in summary.splitlines() if x.startswith(row))
        numbers = [float(v) for v in line.split()[1:]]
        np.testing.assert_allclose(numbers[0::2], values, atol=5e-5)  # the summary's 4 decimals
    assert "var v.ls 1 1 dly dli" in text


def test_run_a_forecast_round(ran):
    """The judgement's arithmetic, the store's files, and the scenario against the least-norm shocks worked out by
    hand (Waggoner and Zha: the most likely structural shocks that meet the conditions)."""
    text = _flat(PAGES["run-a-forecast-round"].text)
    s = ran["run-a-forecast-round"]
    base, scenario, var, fc, data = s["baseline"], s["scenario"], s["var"], s["fc"], s["data"]
    paths = base.to_frame()
    model = fc.to_frame()["dly"]
    assert paths["dly"].iloc[0] == pytest.approx(model.iloc[0] - 0.3) and paths["dly"].iloc[1] == pytest.approx(model.iloc[1] - 0.1)
    assert (round(model.iloc[0], 2), round(paths["dly"].iloc[0], 2)) == (0.63, 0.33) and "0.33: the model's 0.63, less 0.3" in text
    assert round(model.iloc[1], 2) == 0.92 and round(paths["dly"].iloc[1], 2) == 0.82 and "from 0.92 to 0.82" in text
    assert base.base["dly"].iloc[0] == pytest.approx(model.iloc[0]) and str(base.data_end) == "2024Q4" and "2024Q4" in text
    judged = base.judgement().to_frame()
    assert len(judged) == 2 and set(judged["reason"]) == {"a strike in January"}
    sc = scenario.to_frame()
    assert (sc["dli"].iloc[:4] == -2).all() and round(sc["dli"].iloc[4], 2) == 1.72 and "1.72 in 2026Q1" in text
    assert round(sc["dly"].iloc[0], 2) == -0.79 and "-0.79" in text
    rev = s["revisions"].to_frame()
    assert round(rev["dly revision"].iloc[0], 2) == -1.12 and "-1.12" in text
    folder = s["__folder__"] / "forecasts"
    assert sorted(p.name for p in folder.iterdir()) == ["2025Q1_baseline.json", "2025Q1_baseline.parquet",
                                                         "2025Q1_weak_investment.json", "2025Q1_weak_investment.parquet"]
    assert "2025Q1_baseline.parquet" in text and (s["__folder__"] / "round.html").stat().st_size > 0
    # the scenario by hand: x_t = c + A x_{t-1} + P e_t; the shocks e (8 of them) of least norm meeting dli = -2 in 2025
    B = np.asarray(var.params.values).reshape(2, 3)  # by equation: the constant, dly[-1], dli[-1]
    assert var.params.names[:3] == ("dly: const", "dly: dly[-1]", "dly: dli[-1]")
    c, A = B[:, 0], B[:, 1:]
    P = np.linalg.cholesky(np.asarray(var.sigma))
    last = data.frame[["dly", "dli"]].iloc[-1].to_numpy()
    free = [last]
    for _ in range(8):
        free.append(c + A @ free[-1])
    free = np.array(free[1:])
    M = np.zeros((8, 2, 8, 2))  # response of period t, variable i to shock j in period k
    for t in range(8):
        for k in range(t + 1):
            M[t, :, k, :] = np.linalg.matrix_power(A, t - k) @ P
    rows = M[:4, 1].reshape(4, 16)  # dli in 2025Q1 to 2025Q4
    e = np.linalg.pinv(rows) @ (-2 - free[:4, 1])
    path = free + np.einsum("tikj,kj->ti", M, e.reshape(8, 2))
    np.testing.assert_allclose(sc[["dly", "dli"]].to_numpy(), path, atol=1e-8)


def test_save_and_share_a_project(ran):
    text = _flat(PAGES["save-and-share-a-project"].text)
    s = ran["save-and-share-a-project"]
    mine, theirs, session = s["mine"], s["theirs"], s["session"]
    assert round(session["speed"], 2) == 0.75 and "0.75" in text
    assert set(mine.objects) == {"data", "ecm", "speed"} and mine["speed"] == session["speed"]
    np.testing.assert_array_equal(mine["ecm"].params.values, session["ecm"].params.values)
    assert set(theirs.objects) == {"data", "ecm"} and "speed" not in theirs.objects
    from pymetrix.commands.project import project_code

    assert project_code(s["__folder__"] / "consumption.pymx").steps == ("speed = -float(ecm.params['log(c[-1])'])",)
    script = (s["__folder__"] / "consumption.py").read_text(encoding="utf-8")
    line = "ecm = Equation('dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]').fit(data, method=OLS())"
    assert line in script and line in text


def test_seasonally_adjust_a_series(ran):
    """The factors against the pattern the series was made from, and against statsmodels' STL of its log."""
    text = _flat(PAGES["seasonally-adjust-a-series"].text)
    s = ran["seasonally-adjust-a-series"]
    data = s["data"]
    factors = data["sales_seasonal"]
    means = factors.groupby(factors.index.quarter).mean().round(3).tolist()
    assert means == [0.939, 1.003, 0.981, 1.085] and "0.939, 1.003, 0.981 and 1.085" in text
    np.testing.assert_allclose(means, [0.94, 1.00, 0.98, 1.08], atol=0.006)  # the pattern the series was made from
    np.testing.assert_allclose(data["sales_sa"], data["sales"] / factors, rtol=1e-12)
    assert round(data["sales_sa"]["2023Q4"], 1) == 119.2 and "119.2" in text
    made = data.variables["sales_sa"].derived_from  # the Variables tab's account of how it was made
    assert made == "sales seasonally adjusted (STL(seasonal=7, robust))" and made in text
    assert round(s["left"].test("wald", "seasonal[Q2] = 0, seasonal[Q3] = 0, seasonal[Q4] = 0").p_value, 2) == 0.47
    from pymetrix import Equation

    raw = Equation("dlog(sales) ~ seasonal()").fit(data).test("wald", "seasonal[Q2] = 0, seasonal[Q3] = 0, seasonal[Q4] = 0")
    assert raw.p_value < 1e-6 and "it rejects at any level" in text
    stl = pytest.importorskip("statsmodels.tsa.seasonal").STL
    ref = stl(np.log(data["sales"].to_numpy()), period=4, seasonal=7, robust=True).fit()
    np.testing.assert_allclose(np.exp(ref.seasonal), factors.to_numpy(), rtol=1e-10)
    assert not s["steadier"]["sales_seasonal"].equals(factors)


def test_bring_in_an_eviews_model(ran):
    """The scenario's impact against the multiplier 1 / (1 - c(2)); the program's estimates against least squares by
    NumPy on its sample; the text's coefficients are those estimates, rounded."""
    text = _flat(PAGES["bring-in-an-eviews-model"].text)
    s = ran["bring-in-an-eviews-model"]
    macro, data, base, effect = s["macro"], s["data"], s["base"], s["effect"]
    assert macro.model.params == {"c_1": 11.468, "c_2": 0.75145, "c_3": 4.9437, "c_4": 0.10745, "c_5": -0.06499}
    assert s["model"].check(data, ("2020Q1", "2024Q4")).ok
    assert effect["2020Q1"] == pytest.approx(5 / (1 - 0.75145), rel=1e-9) and round(effect["2020Q1"], 1) == 20.1
    assert round(1 / (1 - 0.75145), 2) == 4.02 and "4.02: 20.1" in text
    assert (round(base.solved()["y"]["2024Q4"], 1), round(data["y"]["2024Q4"], 1)) == (593.3, 584.3) and "593.3" in text
    frame = data.frame.loc["2001Q1":"2019Q4"]
    X = np.column_stack([np.ones(len(frame)), frame["y"]])
    cons = np.linalg.lstsq(X, frame["cons"].to_numpy(), rcond=None)[0]
    lagged = data.frame.shift().loc["2001Q1":"2019Q4"]
    X = np.column_stack([np.ones(len(frame)), lagged["y"], lagged["inv"]])
    inv = np.linalg.lstsq(X, frame["inv"].to_numpy(), rcond=None)[0]
    fitted = [np.asarray(eq.params.values) for eq in s["fitted"].model.equations[:2]]
    np.testing.assert_allclose(fitted[0], cons, rtol=1e-9)
    np.testing.assert_allclose(fitted[1], inv, rtol=1e-9)
    text_values = [11.468, 0.75145, 4.9437, 0.10745, -0.06499]
    np.testing.assert_allclose(np.concatenate([cons, inv]), text_values, rtol=5e-4)  # the text: the estimates, rounded
    from pymetrix.models.source import ModelSource

    problems = [d.message for d in ModelSource(_files(PAGES["bring-in-an-eviews-model"].text)["macro.prg"], "eviews").check().diagnostics]
    assert any("is estimated: give data" in p for p in problems) and "is estimated: give data" in text


def test_run_a_script_in_the_app(ran):
    text = _flat(PAGES["run-a-script-in-the-app"].text)
    s = ran["run-a-script-in-the-app"]
    session, found = s["session"], s["found"]
    assert round(found["speed"], 2) == 0.75 and session["speed"] == found["speed"] and "speed of adjustment: 0.75" in text
    assert sorted(session.objects) == ["data", "ecm", "speed"] and "['data', 'ecm', 'speed']" in text
    script = session.script()
    assert script.startswith("from pymetrix import *") and 'ecm = Equation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]").fit(data)' in script
    ns: dict = {}
    exec(script, ns)  # noqa: S102 - View → Script, run as it is
    assert ns["speed"] == found["speed"]


def test_make_a_report(ran):
    """Both reports, from code and from the outline, have the page's headings, caption, tables and appendix."""
    s = ran["make-a-report"]
    text = _flat(PAGES["make-a-report"].text)
    pages = [(s["__folder__"] / name).read_text(encoding="utf-8") for name in ("consumption.html", "outline.html")]
    for html in pages:
        headings = re.findall(r"<h[1-3][^>]*>(.*?)</h", html)
        assert headings == ["Consumption", "The consumption function", "Appendix: where the numbers come from"]
        plain = re.sub(r"<[^>]+>", " ", html)
        assert "Tests of the residuals" in plain and "Breusch-Godfrey" in plain and "Consumption returns" in plain
        assert re.search(r"OLS\s+2000Q2 to 2024Q4\s+99", plain) and "Forecasting team" in plain
    assert round(s["ecm"].params["log(c[-1])"], 2) == -0.75 and "-0.75" in text
    assert "report.save('outline.html')" in s["session"].script() and "report.save('outline.html')" in text
    assert len(s["session"].report_outline) == 4


def test_check_a_forecasts_record(ran):
    """The AR(1)'s one-step forecasts made by hand at each origin, by least squares on the data to that origin."""
    text = _flat(PAGES["check-a-forecasts-record"].text)
    s = ran["check-a-forecasts-record"]
    bt_var, bt_ar, dm, data = s["bt_var"], s["bt_ar"], s["dm"], s["data"]
    acc_var, acc_ar = bt_var.accuracy().to_frame(), bt_ar.accuracy().to_frame()
    assert (acc_var["n"] == 32).all() and round(acc_var.loc["dly, h=1", "RMSE"], 3) == 0.833 and "RMSE 0.833" in text
    assert round(acc_ar.loc["dly, h=1", "RMSE"], 3) == 0.829 and "RMSE 0.829" in text and "0.829" in text
    assert round(acc_var.loc["dly, h=1", "mean error"], 2) == -0.25 and "mean error -0.25" in text
    assert (round(dm.statistic, 2), round(dm.p_value, 2)) == (0.37, 0.71) and "statistic 0.37, p-value 0.71" in text
    assert round(bt_var.efficiency("dly", 1).p_value, 2) == 0.19 and "p-value 0.19" in text
    errors = bt_ar.errors
    assert np.allclose(errors["error"], errors["actual"] - errors["forecast"])  # the outcome less the forecast
    dly = data.frame["dly"].dropna()
    by_hand = []
    for origin in pd.period_range("2015Q1", "2022Q4", freq="Q"):
        y = dly.loc[:origin]
        X = np.column_stack([np.ones(len(y) - 1), y.to_numpy()[:-1]])
        b = np.linalg.lstsq(X, y.to_numpy()[1:], rcond=None)[0]
        by_hand.append(dly[origin + 1] - (b[0] + b[1] * y.iloc[-1]))
    one = errors[errors["horizon"] == 1]
    np.testing.assert_allclose(one["error"].to_numpy(), by_hand, rtol=1e-9)
    assert np.sqrt(np.mean(np.square(by_hand))) == pytest.approx(acc_ar.loc["dly, h=1", "RMSE"], rel=1e-12)


def test_reconcile_components_to_a_total(ran):
    """The reconciliation against the same problem solved through its KKT system: proportional Denton's criterion,
    the components adding to the total every quarter, each free component's annual totals, gov held."""
    text = _flat(PAGES["reconcile-components-to-a-total"].text)
    s = ran["reconcile-components-to-a-total"]
    rec, q = s["rec"], s["q"]
    files = _files(PAGES["reconcile-components-to-a-total"].text)
    raw = pd.read_csv(pd.io.common.StringIO(files["quarterly.csv"]), index_col=0)
    annual = pd.read_csv(pd.io.common.StringIO(files["annual.csv"]), index_col=0)
    gap = raw[["cons", "inv", "gov"]].sum(axis=1) - raw["gdp"]
    assert round(gap.abs().max(), 1) == 5.2 and "5.2" in text
    table = rec.diagnostics().to_frame()
    assert table.loc["gdp (sum of components)", "largest gap after"] < 1e-9
    assert round(rec.adjustments["cons"]["2021Q4"], 2) == -4.85 and "-4.85" in text
    np.testing.assert_allclose(q.frame[["cons", "inv", "gov"]].sum(axis=1), raw["gdp"].to_numpy(), atol=1e-9)
    for c in ("cons", "inv"):
        np.testing.assert_allclose(q[c].groupby(q.frame.index.year).sum().to_numpy(), annual[c].to_numpy(), atol=1e-9)
    np.testing.assert_array_equal(q["gov"].to_numpy(), raw["gov"].to_numpy())
    assert "607.2 and 166.8 in 2021" in text
    # by hand: minimise sum_c || D diag(1/p_c) (x_c - p_c) ||^2 subject to the constraints, through the KKT system
    n, p = 12, np.concatenate([raw["cons"].to_numpy(), raw["inv"].to_numpy()])
    D = (np.eye(n) - np.eye(n, k=-1))[1:]
    blocks = [D @ np.diag(1 / raw[c].to_numpy()) for c in ("cons", "inv")]
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n], Q[n:, n:] = blocks[0].T @ blocks[0], blocks[1].T @ blocks[1]
    years = np.kron(np.eye(3), np.ones(4))
    G = np.vstack([np.hstack([np.eye(n), np.eye(n)]), np.hstack([years, np.zeros((3, n))]), np.hstack([np.zeros((3, n)), years])])
    g = np.concatenate([raw["gdp"].to_numpy() - raw["gov"].to_numpy(), annual["cons"].to_numpy(), annual["inv"].to_numpy()])
    K = np.block([[2 * Q, G.T], [G, np.zeros((len(g), len(g)))]])
    rhs = np.concatenate([2 * Q @ p, g])
    x = np.linalg.lstsq(K, rhs, rcond=None)[0][: 2 * n]
    np.testing.assert_allclose(rec.series["cons"].to_numpy(), x[:n], atol=1e-8)
    np.testing.assert_allclose(rec.series["inv"].to_numpy(), x[n:], atol=1e-8)


def test_translate_an_eviews_program(ran):
    """The report's rows and counts, the script against the steps, and the equation against least squares by NumPy on
    the CSV file the page gives."""
    s = _ran(ran, "translate-an-eviews-program")
    text = _flat(PAGES["translate-an-eviews-program"].text)
    t, session, made = s["t"], s["session"], s["made"]
    assert t.report.counts() == {"translated": 5, "translated with a note": 3, "left out": 1, "error": 0}
    assert "translated (5), translated with a note (3), left out (1), error (0)" in text and "nine rows" in text
    assert [r.status for r in t.report][5] == "translated with a note" and "HAC" in t.report.line(6).reason
    assert t.report.line(9).status == "left out" and "Data → Plot series" in t.report.line(9).reason
    assert sorted(session.objects) == ["eq_c", "eq_c_auto", "macro"]
    eq = session["eq_c"]
    assert eq.provenance.sample == ("2018Q2", "2023Q4") and len(eq.resid) == 23 and "23 quarters" in text
    assert [round(v, 2) for v in eq.params.values] == [-0.03, 1.00] and "-0.03 and 1.00" in text
    assert made["mpc"] == eq.params["dly"] and round(made["mpc"], 2) == 1.00
    frame = pd.read_csv(s["__folder__"] / "macro.csv", index_col=0)
    dlc, dly = (100 * np.log(frame["cons"]).diff()).iloc[1:], (100 * np.log(frame["inc"]).diff()).iloc[1:]
    b = np.linalg.lstsq(np.column_stack([np.ones(len(dly)), dly]), dlc.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(eq.params.values, b, rtol=1e-9)
    ns: dict = {}
    here = os.getcwd()
    os.chdir(s["__folder__"])
    try:
        exec(t.script, ns)  # noqa: S102 - the script runs on its own and gives what the steps give
    finally:
        os.chdir(here)
    np.testing.assert_array_equal(ns["eq_c"].params.values, eq.params.values)
    assert ns["eq_c_auto"].statistic == session["eq_c_auto"].statistic and ns["mpc"] == made["mpc"]
    assert (s["__folder__"] / "pymetrix_export.prg").exists()
    assert "pip install -e packages/pymetrix-eviews" in text and "pip install --pre pymetrix-eviews" in text


def test_the_eviews_pages_dialogs(ran):
    """The plugin's dialogs, as the page names their fields; the panel's buttons and filters."""
    _ran(ran, "translate-an-eviews-program")
    from pymetrix.forms import form_spec
    from pymetrix_eviews.app import ImportEViews, TranslateEViewsProgram
    from pymetrix_eviews.program import STATUSES

    text = _flat(PAGES["translate-an-eviews-program"].text)
    specs = {s.label: s for s in form_spec(ImportEViews)}
    what = specs["Import"]
    assert [what.display(c) for c in what.choices] == ["Workfile bundle", "Get the export program (experimental)"]
    for shown in ("Import: Get the export program (experimental)", "Import: Workfile bundle", "Bundle: macro.evbundle"):
        assert shown in text
    assert {"Save the program as", "Bundle"} <= set(specs)
    assert {"Program", "Workfile data", "Date column", "Frequency", "Program arguments"} == {s.label for s in form_spec(TranslateEViewsProgram)}
    assert list(STATUSES) == ["translated", "translated with a note", "left out", "error"]
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.extension import AppInterface
    from pymetrix_eviews.panel import create_panel

    s = Session()
    panel = create_panel(AppInterface(lambda: s, lambda: None, lambda cls, initial: None, s.run, lambda m: None))
    assert {"Copy", "Copy script", "Run as steps"} <= {b.text() for b in panel.findChildren(QtWidgets.QPushButton)}
    for word in ("Copy script", "Run as steps", "Copy"):
        assert word in text
    panel.deleteLater()


# -- the steps in the app: the dialogs' commands, with the fields as the page names them ------------------------------

# Each step: the command, the page's "Label: value" for each field (labels as the dialog shows them; True for a box the
# page says is ticked), and the fields of its settings sub-form (Test settings, Method settings, Newton's) the same way.
# "{folder}" is the folder of the page's files. A step "@name" is not a dialog: RAW[name] does it, as the page says.
EXAMPLE = ("LoadExample", {"Example dataset": "sample_macro", "Save as": "data"}, {})
GROWTH = [("DeriveSeries", {"Data": "data", "New series name": "dly", "Formula": "100*dlog(y)"}, {}),
          ("DeriveSeries", {"Data": "data", "New series name": "dli", "Formula": "100*dlog(i)"}, {})]
ECM = ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", "Save as": "ecm"}, {})
STEPS = {
    "estimate-and-test-an-equation": [
        EXAMPLE,
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + dlog(c)[-1]", "Method": "OLS", "Save as": "fit"}, {}),
        ("RunTest", {"Object to test": "fit", "Test": "breusch_godfrey", "Save as": "bg"}, {"Lags": "4"}),
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", "Save as": "ecm"}, {}),
        ("FitEquation", {"Formula": "dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", "Standard errors": "White (HC1)",
                         "Save as": "ecm_white"}, {}),
    ],
    "load-data-with-browse-series": [
        ("LoadFromConnector", {"Data service": "fred", "Series codes": "GDPC1", "Save as": "gdp"}, {}),
        ("LoadFromConnector", {"Data service": "worldbank", "Series codes": "NY.GDP.MKTP.CD",
                               "Other parameters": "country=WSM;TON", "Save as": "wb"}, {}),
    ],
    "estimate-a-var": [
        EXAMPLE, *GROWTH,
        ("FitVAR", {"Series": "dly, dli", "Lags": "bic", "Deterministic terms": "Constant", "Save as": "var"}, {}),
        ("ImpulseResponse", {"Estimated VAR": "var", "Horizon": "12", "Identification": "Cholesky", "Save as": "irf"}, {}),
        ("Forecast", {"Estimated model": "var", "Forecast start": "2025Q1", "Forecast end": "2026Q4", "Save as": "fc"}, {}),
    ],
    "bring-in-an-iris-or-dynare-model": [
        ("ImportIris", {"IRIS file": "{folder}/nk.model", "Parameters": "phi=1.5", "Save as": "nk"}, {}),
        ("SolveRationalExpectations", {"Equations": "nk", "Save as": "sol"}, {}),
        ("ImportDynare", {"Dynare file": "{folder}/nk.mod", "Save as": "mod"}, {}),
        ("SolveRationalExpectations", {"Equations": "mod", "Save as": "dyn"}, {}),
    ],
    "run-a-forecast-round": [
        EXAMPLE, *GROWTH,
        ("FitVAR", {"Series": "dly, dli", "Lags": "1", "Save as": "var"}, {}),
        ("Forecast", {"Estimated model": "var", "Forecast start": "2025Q1", "Forecast end": "2026Q4", "Save as": "fc"}, {}),
        ("MakeRound", {"Forecast or solution": "fc", "Round name": "2025Q1 baseline",
                       "Notes": "VAR(1) in output and investment growth", "Save as": "baseline"}, {}),
        ("AdjustRound", {"Round": "baseline", "Variable": "dly", "Changes by period": "2025Q1=-0.3, 2025Q2=-0.1",
                         "Reason": "a strike in January", "Adjust by": "Add to the path"}, {}),
        ("ConditionalForecast", {"Estimated VAR": "var", "Forecast start": "2025Q1", "Forecast end": "2026Q4",
                                 "Conditions": "dli[2025Q1:2025Q4]=-2", "Save as": "weak"}, {}),
        ("MakeRound", {"Forecast or solution": "weak", "Round name": "2025Q1 weak investment", "Save as": "scenario"}, {}),
        ("SaveRound", {"Round": "baseline", "Store folder": "forecasts"}, {}),
        ("SaveRound", {"Round": "scenario", "Store folder": "forecasts"}, {}),
        ("CompareRounds", {"Earlier round": "2025Q1 baseline", "Later round": "2025Q1 weak investment",
                           "Store folder": "forecasts", "Save as": "revisions"}, {}),
        ("ExportReport", {"Results": "baseline, scenario", "Write to": "round.html", "Report title": "Forecast round 2025Q1",
                          "Views": "forecast_round"}, {}),
    ],
    "save-and-share-a-project": [EXAMPLE, ECM, ("@typed", {}, {})],
    "cointegration-in-a-short-sample": [
        EXAMPLE,
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
    "seasonally-adjust-a-series": [
        ("LoadData", {"Data file": "{folder}/sales.csv", "Time column": "quarter", "Frequency": "Quarterly", "Save as": "data"}, {}),
        ("SeasonalAdjust", {"Series": "sales", "Data": "data", "Method": "STL", "Keep up to date": True}, {"Seasonal": "7", "Robust": True}),
    ],
    "bring-in-an-eviews-model": [
        EXAMPLE,
        ("DeriveSeries", {"Data": "data", "New series name": "cons", "Formula": "c"}, {}),
        ("DeriveSeries", {"Data": "data", "New series name": "inv", "Formula": "i"}, {}),
        ("@model_editor", {}, {}),
        ("CheckModel", {"Equations": "macro", "Data": "data", "Start": "2020Q1", "End": "2024Q4"}, {}),
        ("SolveModel", {"Equations": "macro", "Start": "2020Q1", "End": "2024Q4", "Save as": "base"}, {}),
        ("SolveModel", {"Equations": "macro", "Start": "2020Q1", "End": "2024Q4", "Add": "g=5", "Save as": "more_g"}, {}),
    ],
    "make-a-report": [EXAMPLE, ECM, ("@outline", {}, {})],
    "check-a-forecasts-record": [
        EXAMPLE, *GROWTH,
        ("FitVAR", {"Series": "dly, dli", "Lags": "1", "Save as": "var"}, {}),
        ("FitEquation", {"Formula": "dly ~ dly[-1]", "Save as": "ar"}, {}),
        ("RunBacktest", {"Estimated model": "var", "Start": "2015Q1", "End": "2022Q4", "Longest horizon": "4",
                         "Save as": "bt_var"}, {}),
        ("RunBacktest", {"Estimated model": "ar", "Start": "2015Q1", "End": "2022Q4", "Longest horizon": "4",
                         "Save as": "bt_ar"}, {}),
        ("CompareBacktests", {"First backtest": "bt_var", "Second backtest": "bt_ar", "Variable forecast": "dly",
                              "Horizon": "1", "Test": "Diebold-Mariano", "Loss": "Squared", "Save as": "dm"}, {}),
    ],
    "solve-a-large-backward-looking-model": [
        ("@first_block", {}, {}),
        ("CheckModel", {"Equations": "model", "Data": "data", "Start": "2001Q1", "End": "2010Q4"}, {}),
        ("SolveModel", {"Equations": "model", "Start": "2001Q1", "End": "2010Q4", "Solver": "Newton", "Save as": "solution",
                        "Ordering": "Blocks", "Fallback": True}, {"Convergence": "Step", "Refresh": "Period"}),
    ],
    "reconcile-components-to-a-total": [
        ("LoadData", {"Data file": "{folder}/quarterly.csv", "Time column": "quarter", "Frequency": "Quarterly", "Save as": "q"}, {}),
        ("LoadData", {"Data file": "{folder}/annual.csv", "Time column": "year", "Frequency": "Annual", "Save as": "annual"}, {}),
        ("Reconcile", {"Components": "cons, inv, gov", "Total (high frequency)": "gdp", "High-frequency data": "q",
                       "Low-frequency totals": "annual", "Components held fixed": "gov", "Method": "Denton (proportional)",
                       "Conversion": "Sum (flows)", "Save as": "rec"}, {}),
    ],
    "translate-an-eviews-program": [
        ("TranslateEViewsProgram", {"Program": "{folder}/consumption.prg", "Workfile data": "{folder}/macro.csv",
                                    "Date column": "obs", "Frequency": "Quarterly"}, {}),
        ("@run_as_steps", {}, {}),
    ],
}
# what the pages say is under Advanced settings (Newton's Convergence and Refresh are, in the Solver settings section)
ADVANCED = {"Maximum lags", "Other parameters", "Ordering", "Fallback", "Convergence", "Refresh"}
DEFAULTS = {("Save as", "dyn")}  # what a page leaves at its default, so does not name


def _typed(session, code, folder):  # View → Immediate: the line typed
    session.run_code("speed = -float(ecm.params['log(c[-1])'])")


def _model_editor(session, code, folder):  # Model → Model editor…, Syntax: EViews, Name: macro, the text, Apply
    from pymetrix.commands import COMMANDS

    define = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "DefineModel")
    session.run(define(text=(folder / "macro.txt").read_text(encoding="utf-8"), syntax="eviews", name="macro"))


def _outline(session, code, folder):  # Add to report, the Report panel's Heading…, Text…, a caption; Export…
    from pymetrix.commands import ExportReport

    session.add_to_report("ecm")
    session.add_to_report("ecm", "Diagnostics", caption="Tests of the residuals")
    session.add_report_heading("The consumption function")
    session.move_report_item(2, 0)  # Up, twice: the heading first
    session.add_report_text("Consumption returns to its long-run ratio to output.")
    session.run(ExportReport.from_outline(session, path="consumption.html", title="Consumption", author="Forecasting team"))


def _first_block(session, code, folder):  # File → New script, the first block, ▶ Run
    session.run_code(_code(PAGES["solve-a-large-backward-looking-model"].text)[0], "editor")


def _run_as_steps(session, code, folder):  # the EViews translation panel's Run as steps
    from pymetrix_eviews.app import translations

    (cmd, translation), = translations(session)
    translation.run(session)


RAW = {"@typed": _typed, "@model_editor": _model_editor, "@outline": _outline, "@first_block": _first_block,
       "@run_as_steps": _run_as_steps}


def _value(spec, shown):
    """A field's value from what the dialog shows: a fixed choice by its words, a number field's number, a ticked box,
    else the text typed."""
    if shown is True:
        assert spec.kind == "checkbox", spec.label
        return True
    if spec.choices:
        found = [c for c in spec.choices if spec.display(c) == shown]
        assert found, (spec.label, shown, [spec.display(c) for c in spec.choices])
        return found[0]
    return int(shown) if spec.kind == "number" else shown


def _said(label: str, shown, text: str) -> None:
    """The page names the field as the dialog shows it: "Label: value", or "Label ticked"."""
    if shown is True:
        assert f"{label} ticked" in text or f"{label} stays ticked" in text, f"the page says {label} is ticked"
    elif (label, shown) not in DEFAULTS:
        named = Path(shown).name if shown.startswith("{folder}") else shown
        assert f"{label}: {named}" in text, f"the page names {label}: {named}"


def _command(cls, fields: dict, sub: dict, session, folder: Path, text: str):
    from pymetrix.forms import form_spec
    from pymetrix.plugins import registry

    specs = {s.label: s for s in form_spec(cls)}
    values = {}
    for label, shown in fields.items():
        assert label in specs, (cls.__name__, label, sorted(specs))
        assert specs[label].advanced == (label in ADVANCED), (cls.__name__, label)
        _said(label, shown, text)
        values[specs[label].name] = _value(specs[label], shown.format(folder=folder) if isinstance(shown, str) else shown)
    if sub:  # the settings sub-form of the choice made (the test, the method, the solver)
        options = next(s for s in specs.values() if s.kind == "options")
        chosen = values.get(options.depends_on, next(s for s in specs.values() if s.name == options.depends_on).default)
        owner = registry(options.source or "tests").get(chosen).obj
        subspecs = {s.label: s for s in form_spec(owner)}
        parts = []
        for label, shown in sub.items():
            assert label in subspecs, (owner.__name__, label, sorted(subspecs))
            assert subspecs[label].advanced == (label in ADVANCED), (owner.__name__, label)
            _said(label, shown, text)
            parts.append(f"{subspecs[label].name}={_value(subspecs[label], shown)!r}")
        values[options.name] = ", ".join(parts)
    return cls(**values)


def _same(key: str, obj, mine) -> None:
    """A result of the steps in the app equals the code's of the same name."""
    from pymetrix import Dataset, DynareModel, FittedEquation, Model
    from pymetrix.models.linear import LinearSolution
    from pymetrix.models.templates import Template
    from pymetrix.reconciliation import Reconciliation
    from pymetrix.stats.base import TestResult

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
    elif isinstance(obj, DynareModel) or (isinstance(obj, Template) and obj.shock_sd):
        assert obj.shock_sd == mine.shock_sd and obj.solve().rules().to_frame().equals(mine.solve().rules().to_frame())
    elif isinstance(obj, Template):  # a backward-looking model's text: the same equations and coefficients
        assert repr(obj.model) == repr(mine.model), key
    elif hasattr(obj, "solved"):  # a model's solution
        pd.testing.assert_frame_equal(obj.solved().frame, mine.solved().frame, check_freq=False, rtol=1e-10)
    elif isinstance(obj, Reconciliation):
        pd.testing.assert_frame_equal(obj.series, mine.series, check_freq=False, rtol=1e-12)
    elif isinstance(obj, (list, tuple, str, int, float, Model)):  # typed code's own objects
        assert repr(obj) == repr(mine), key
    elif isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
        assert obj.equals(mine), key
    else:  # rounds, forecasts, responses, backtests, tables: their numbers
        pd.testing.assert_frame_equal(obj.to_frame(), mine.to_frame(), check_freq=False, rtol=1e-12)


# objects the steps make that the page's code does not (its check, its data under another name), by page
UNMATCHED = {"cointegration-in-a-short-sample": {"data"}, "bring-in-an-eviews-model": {"model_check"},
             "solve-a-large-backward-looking-model": {"model_check"}}


@pytest.mark.parametrize("stem", sorted(STEPS))
def test_the_steps_in_the_app_give_the_codes_results(stem, ran, tmp_path):
    from pymetrix.commands import COMMANDS, Session

    code = dict(_ran(ran, stem))
    if isinstance(code.get("session"), Session):  # the page's own session: its objects too
        code.update({k: v for k, v in code["session"].objects.items() if k not in code})
    by_name = {i.cls.__name__: i.cls for i in COMMANDS.values()}
    if PAGES[stem].plugin:
        from pymetrix_eviews.app import TranslateEViewsProgram

        by_name["TranslateEViewsProgram"] = TranslateEViewsProgram
    text = _flat(PAGES[stem].text)
    for name, content in _files(PAGES[stem].text).items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    session = Session()
    here = os.getcwd()
    os.chdir(tmp_path)  # what the steps write (a store, a report) goes here
    try:
        with warnings.catch_warnings(), _offline(tmp_path / "home"):
            warnings.simplefilter("ignore", FutureWarning)
            for name, fields, sub in STEPS[stem]:
                if name.startswith("@"):
                    RAW[name](session, code, tmp_path)
                else:
                    session.run(_command(by_name[name], fields, sub, session, tmp_path, text))
            unmatched = set()
            for key, obj in session.objects.items():
                if key in code:
                    _same(key, obj, code[key])
                else:
                    unmatched.add(key)
            assert unmatched == UNMATCHED.get(stem, set())
            ns = dict(session.initial)
            exec(session.script(), ns)  # noqa: S102 - the script the steps recorded replays
            assert set(session.objects) <= set(ns)
    finally:
        os.chdir(here)


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


def test_what_the_new_pages_say_of_the_windows(ran):
    """The views the later pages tell readers to read are the results' own."""
    from pymetrix.views import sections

    def views(stem, name):
        return [t for t, _ in sections(ran[stem][name])]

    assert {"Summary", "Roots", "Residuals", "Residual correlogram"} <= set(views("estimate-a-var", "var"))
    assert views("estimate-a-var", "irf")[0] == "Graph" and "Graph" in views("estimate-a-var", "fc")
    assert views("run-a-forecast-round", "baseline") == ["Paths", "Graph", "Judgement"]
    assert {"Accuracy", "Graph", "Tests", "Forecasts"} <= set(views("check-a-forecasts-record", "bt_var"))
    assert {"Summary", "Graph", "Series", "Adjustments"} <= set(views("reconcile-components-to-a-total", "rec"))
    assert {"Convergence", "Graph", "Data"} <= set(views("bring-in-an-eviews-model", "base"))
    assert "Diagnostics" in views("make-a-report", "ecm")
    from pymetrix.commands import COMMANDS
    from pymetrix.forms import form_spec

    report = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "ExportReport")
    specs = {s.label: s for s in form_spec(report)}
    assert specs["Provenance appendix"].advanced and specs["Date"].advanced and not specs["Use the report outline"].advanced
    var = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "FitVAR")
    assert {s.label: s for s in form_spec(var)}["Maximum lags"].default == 8  # "up to Maximum lags (8, ...)"
    irf = {s.label: s for s in form_spec(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "ImpulseResponse"))}
    assert irf["Draws or replications"].default == 500 and irf["Draws or replications"].advanced and irf["Orthogonalised shocks"].advanced
    backtest = {s.label: s for s in form_spec(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "RunBacktest"))}
    assert backtest["Vintages"].advanced and backtest["Window (0 = expanding)"].advanced
    reconcile = {s.label: s for s in form_spec(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "Reconcile"))}
    assert reconcile["Component weights"].advanced and reconcile["Replace the components in the data"].default is True
    method = reconcile["Method"]
    assert {"Denton (additive)", "Weighted least squares (in proportion to the weights)"} <= {method.display(c) for c in method.choices}


def test_the_windows_the_pages_name(window, tmp_path):
    """The Code window's Run and Run selection, the model editor's syntax, name and tabs, the Report panel's buttons,
    a result window's Add to report, and the project trust dialog's buttons, as the pages name them."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    files = _files(PAGES["run-a-script-in-the-app"].text)
    (tmp_path / "consumption.py").write_text(files["consumption.py"], encoding="utf-8")
    session = window.state["session"]
    editor = window.open_editor(path=str(tmp_path / "consumption.py"))
    assert [a.text() for a in editor.actions_by_name.values()][:2] == ["▶ Run", "Run selection"]
    assert editor.actions_by_name["run"].shortcut().toString() == "F5" and editor.actions_by_name["run_selection"].shortcut().toString() == "F9"
    assert editor.title.startswith("consumption.py")
    editor.run_all()  # ▶ Run: the script's statements, through the Immediate window
    for _ in range(200):
        app.processEvents()
        if "speed" in session.objects:
            break
    assert round(session["speed"], 2) == 0.75 and {"data", "ecm"} <= set(session.objects)
    assert [r.command.source for r in session.history[-3:]] == ["editor"] * 3  # a step each
    editor.parentWidget().close()
    from pymetrix.ui.model_editor_qt import LABELS, ModelEditorWindow

    assert LABELS["eviews"] == "EViews (.prg / model text)"
    model_editor = ModelEditorWindow(text=_files(PAGES["bring-in-an-eviews-model"].text)["macro.txt"], syntax="eviews", name="macro")
    tabs = [model_editor.sidecars.tabText(i) for i in range(model_editor.sidecars.count())]
    assert tabs[:2] == ["Problems", "Symbols"] and model_editor.apply_action.text() == "Apply"
    assert {w.text().strip() for w in model_editor.findChildren(QtWidgets.QLabel)} >= {"Syntax", "Name"}
    rows = [model_editor.problems.topLevelItem(i).text(0) for i in range(model_editor.problems.topLevelItemCount())]
    assert len(rows) == 1 and rows[0].startswith("No problems")  # the text reads
    symbols = {model_editor.symbols.topLevelItem(i).text(0) for i in range(model_editor.symbols.topLevelItemCount())}
    assert {"c_1", "c_2", "c_3", "c_4", "c_5"} <= symbols
    assert model_editor.command().syntax == "eviews" and model_editor.command().name == "macro"
    model_editor.close()
    from pymetrix.ui.report_qt import report_panel

    panel = report_panel(lambda: session, lambda: None)
    assert [panel.buttons[k].text() for k in ("up", "down", "remove", "heading", "text", "export")] == \
        ["Up", "Down", "Remove", "Heading…", "Text…", "Export…"]
    panel.deleteLater()
    import inspect

    from pymetrix.ui import qt as ui

    source = inspect.getsource(ui)
    assert 'Qt.QPushButton("Add to report")' in source
    assert '"Run the code and trust this project"' in source and '"Open without running the code"' in source


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
                        ("dynare model", "bring-in-an-iris-or-dynare-model"), ("iris", "bring-in-an-iris-or-dynare-model"),
                        ("browse series", "load-data-with-browse-series"), ("forecast round", "run-a-forecast-round"),
                        ("backtest", "check-a-forecasts-record"), ("seasonally adjust", "seasonally-adjust-a-series"),
                        ("translate eviews program", "translate-an-eviews-program"), ("reconcile", "reconcile-components-to-a-total")):
        found = [e for e in rank(entries, words) if e.kind == "Help" and e.title == PAGES[stem].title]
        assert found and found[0].detail == "How do I…", words
    titles = [e.title for e in rank(entries, "how do i")]
    assert "Help → How do I…" in titles and "How do I…" in titles
    assert rank(entries, "var")[0].title == "Estimate → VARs and local projections → VAR…"  # menu items still lead


def test_a_dialogs_help_lists_the_task_pages_that_open_it():
    """A dialog's help page ends its description with the task pages whose steps in the app open it (found from the
    menu paths each page writes), and leaves out the dialog the pages open only to have example data."""
    from pymetrix.commands import FitVAR, LoadExample, Reconcile, RunBacktest
    from pymetrix.help import help_page
    from pymetrix.howto import dialogs, pages, pages_for_dialog

    by_id = {p.id: p for p in pages()}
    assert "FitVAR" in dialogs(by_id["howto-estimate-a-var"]) and "Reconcile" in dialogs(by_id["howto-reconcile-components-to-a-total"])
    assert [p.id for p in pages_for_dialog("RunBacktest")] == ["howto-check-a-forecasts-record"]
    assert {p.id for p in pages_for_dialog("FitVAR")} >= {"howto-estimate-a-var", "howto-run-a-forecast-round"}
    text = help_page(FitVAR).to_markdown()
    assert "## How do I…" in text and "(pymetrix:topic/howto-estimate-a-var)" in text
    assert "(pymetrix:topic/howto-reconcile-components-to-a-total)" in help_page(Reconcile).to_markdown()
    assert "(pymetrix:topic/howto-check-a-forecasts-record)" in help_page(RunBacktest).to_markdown()
    assert pages_for_dialog("LoadExample") == [] and "How do I…" not in help_page(LoadExample).to_markdown()
    for p in pages():  # every dialog a page opens is a dialog of the app
        assert all(d for d in dialogs(p)), p.id
