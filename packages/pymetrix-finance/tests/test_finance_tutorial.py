"""The finance tutorial (docs/tutorial.md) runs as written: its code blocks, in order, as one script, on the
finance datasets; and its claims about them hold.

The Fama-French data are downloaded on first use, so here ``Dataset.example("french")`` is the simulated stand-in
(conftest.py) and the claims about the real returns are checked only with PYMETRIX_FRENCH set to a french.parquet the
plugin downloaded (such as ~/.pymetrix/cache/datasets/french.parquet)."""

import os
import re
from pathlib import Path

import numpy as np

DOC = Path(__file__).resolve().parents[1] / "docs" / "tutorial.md"


def test_the_finance_tutorial_runs_as_written():
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) == 10
    scope: dict = {"__name__": "finance_tutorial"}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f"{DOC.name}, block {i}", "exec"), scope)
    assert len(scope["ff"].frame) == 819 and len(scope["risk"].frame) == 699  # the months the text counts
    if os.environ.get("PYMETRIX_FRENCH"):  # the real Fama-French returns: the text's claims about them
        perf = scope["perf"].values
        assert perf.loc["Sharpe ratio"].idxmax() == "mom" and perf.loc["skewness"].idxmin() == "mom"  # as the text says
        assert scope["p_grs"] < 0.01 and scope["fm"].fits["hlth"].params["const"] > 0
        assert scope["port"].weights.argmax() == 0 and scope["rc"].frame()["share of VaR"].idxmax() == "nodur"
        sharpe = scope["bt"].performance().to_frame().loc["Sharpe ratio"]
        assert sharpe.idxmax() == "min_variance" and sharpe.idxmin() == "max_sharpe"
        assert scope["risk"].violations / len(scope["risk"].frame) > 0.02  # nearly three times the 1% expected
        from pymetrix_finance.risk import kupiec

        c = scope["counts"]
        assert c["Cornish-Fisher"] < c["t-GJR GARCH"] < c["normal GARCH"]
        assert kupiec(scope["fat"].frame["violation"].to_numpy(), 0.01)[1] < 0.05 < kupiec(scope["cf"].frame["violation"].to_numpy(), 0.01)[1]
    assert len(scope["jump_days"]) == 10
    assert abs(scope["mean_car"] - 0.02) < 0.01
    surf = scope["surf"]
    assert (surf.quotes["fitted vol"] - surf.quotes["market vol"]).abs().max() < 1e-3
    assert (surf.arbitrage().to_frame()["free of arbitrage"] == "yes").all() and scope["heston"].at_bounds
    assert np.nanmax(np.abs(scope["yc"].residuals.to_numpy())) < 1.5 and scope["sv"].residuals["m3"].std() < scope["yc"].residuals["m3"].std()
