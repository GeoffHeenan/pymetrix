"""The finance plugin's finishing: a report template registered through the extension API and used for finance
results; the Finance workspace's quick actions are its dialogs; the help overview; and the tutorial's notebook is
exactly what the script makes from its page."""

import importlib.util
import json
from pathlib import Path

import pytest

from pymetrix import Dataset, Report, performance, value_at_risk
from pymetrix.report import report_templates

ROOT = Path(__file__).resolve().parents[1]  # the package


def test_the_finance_report_template(tmp_path):
    assert "finance_summary" in report_templates() and "estimation_summary" in report_templates()
    ff = Dataset.example("french")
    perf = performance(ff, ["mkt_rf", "hml"], kind="simple")
    risk = value_at_risk(ff, "mkt_rf", method="normal", window=120)
    rep = Report.from_template("finance_summary", {"Factor performance": perf, "Market VaR": risk}, title="Finance")
    tables = [b.value.title for b in rep.blocks if b.kind == "table"]
    assert tables[:2] == ["Performance", next(t for t in tables if t.startswith("Backtests"))]  # each result's own view first
    assert sum(b.kind == "figure" for b in rep.blocks) == 2  # and its graph
    assert rep.save(tmp_path / "finance.md").exists()
    from pymetrix.plugins import registry

    entry = next(e for e in registry("report_templates").entries() if e.name == "finance_summary")
    assert entry.plugin.name == "pymetrix-finance"


def test_the_workspace_and_the_overview():
    from pymetrix import help as hp
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.extension import Workspace
    from pymetrix.plugins import registry

    ws = next(e.obj for e in registry("workspaces").entries() if isinstance(e.obj, Workspace) and e.obj.key == "finance")
    assert len(ws.tools) == 6 and all(name in COMMANDS for name in ws.tools)  # toolbar shortcuts to its dialogs
    page = hp.page_for_link(hp.link("topic", "finance"))
    assert "which tool for which question" in page.lower() and "Strategy backtest" in page


def test_the_notebook_is_the_page():
    spec = importlib.util.spec_from_file_location("make_tutorial_notebook", ROOT / "scripts" / "make_tutorial_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    page = (ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")
    committed = json.loads((ROOT / "docs" / "tutorial.ipynb").read_text(encoding="utf-8"))
    assert committed == mod.notebook(page), "the notebook differs from the page: run python scripts/make_tutorial_notebook.py in packages/pymetrix-finance"
    code = ["".join(c["source"]) for c in committed["cells"] if c["cell_type"] == "code"]
    assert len(code) == page.count("```python")


FINANCE_RESULTS = ("RiskForecast", "FittedMGARCH", "Performance", "FactorModel", "Portfolio", "OptionValuation", "VolatilitySmile",
                   "YieldCurve", "RiskContributions", "AbnormalReturns", "FittedHAR", "StrategyBacktest", "FittedDNS", "FittedCopula",
                   "MertonModel", "CDSCurve", "FittedVolSurface", "BlackLittermanPosterior")


@pytest.mark.parametrize("name", FINANCE_RESULTS)
def test_every_finance_figure_draws(name):
    pytest.importorskip("matplotlib")
    from pymetrix.views import sections
    from _gallery import BUILDERS

    from pymetrix.figures import Figure

    figures = [v for _, v in sections(BUILDERS[name]()) if isinstance(v, Figure)]
    assert all(f.png()[:4] == b"\x89PNG" for f in figures)  # drawn, not only built
