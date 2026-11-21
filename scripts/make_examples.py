"""Build the cross-section and panel example datasets in src/pymetrix/datasets (run from the repository root).

    python scripts/make_examples.py

Real datasets come from libraries whose terms allow redistribution, each recorded in the dataset's ``license``:
statsmodels' public-domain datasets, and linearmodels' (NCSA licence) copies of datasets from Wooldridge's
Econometric Analysis of Cross Section and Panel Data (the finance plugin's datasets are made by its own script). Simulated datasets are made with fixed seeds and their true
effects are stated in their descriptions. Needs statsmodels and linearmodels; the results are committed, so users
need neither.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from pymetrix import Dataset
from pymetrix.metadata import DatasetMeta, Variable

OUT = Path(__file__).resolve().parents[1] / "src" / "pymetrix" / "datasets"
LINEARMODELS = "linearmodels (NCSA licence), from Wooldridge, Econometric Analysis of Cross Section and Panel Data"


def _labels(descr: str, columns: list[str]) -> dict[str, str]:
    """Variable labels from a dataset description of lines like 'educ  years of schooling'."""
    out = {}
    for line in descr.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] in columns and parts[0] not in out:
            out[parts[0]] = parts[1].strip()
    return out


def _save(name: str, frame: pd.DataFrame, title: str, description: str, license: str, labels: dict[str, str],
          entity: str | None = None, time: str | None = None, freq: str | None = None) -> None:
    variables = {c: Variable(c, label=labels.get(c, "")) for c in frame.columns if c not in (entity, time)}
    data = Dataset.from_frame(frame, entity=entity, time=time, freq=freq, variables=variables,
                              meta=DatasetMeta(title=title, description=description, license=license))
    data.write(str(OUT / f"{name}.parquet"))
    print(f"{name}: {len(frame)} rows, {len(frame.columns)} columns")


def linearmodels_sets() -> None:
    for name, entity, time, title, extra in (
        ("mroz", None, None, "Married women's labour supply, 1975 (Mroz 1987)", "Tobit hours, Heckman wages, probit participation"),
        ("card", "id", None, "Returns to schooling and college proximity (Card 1995)", "instrumental variables: nearc4 for educ"),
        ("wage_panel", "nr", "year", "Young men's wages and unions, 1980-1987 (Vella and Verbeek 1998)", "panel wage equations"),
        ("jobtraining", "fcode", "year", "Michigan job-training grants to firms, 1987-1989 (Holzer et al. 1993)",
         "panel and difference-in-differences: grant on scrap rates"),
    ):
        module = importlib.import_module(f"linearmodels.datasets.{name}")
        frame = module.load()
        if name == "mroz":
            frame = frame.reset_index(drop=True)
            frame.insert(0, "woman", np.arange(1, len(frame) + 1))
            entity = "woman"
        if name == "card":  # this copy's id repeats (992 values for 3,010 people): number the rows, keep it as it is
            frame = frame.reset_index(drop=True).rename(columns={"id": "id_original"})
            frame.insert(0, "id", np.arange(1, len(frame) + 1))
        frame = frame.reset_index(drop=True)
        labels = _labels(module.DESCR or "", list(frame.columns))
        _save(name, frame, title, f"{extra}. {' '.join((module.DESCR or '').split()[:40])}…", LINEARMODELS, labels,
              entity=entity, time=time, freq="Y" if time else None)


def statsmodels_sets() -> None:
    nile = importlib.import_module("statsmodels.datasets.nile").load_pandas().data
    frame = pd.DataFrame({"year": nile["year"].astype(int), "volume": nile["volume"].astype(float)})
    _save("nile", frame, "Annual flow of the Nile at Aswan, 1871-1970", "the classic change point: a fall in the flow from 1898",
          "public domain (statsmodels' nile dataset; Cobb 1978)", {"volume": "annual flow, 10^8 cubic metres"}, time="year", freq="Y")
    rand = importlib.import_module("statsmodels.datasets.randhie").load_pandas().data.reset_index(drop=True)
    rand.insert(0, "person", np.arange(1, len(rand) + 1))
    _save("randhie", rand, "RAND Health Insurance Experiment: doctor visits", "counts: mdvis, the number of outpatient visits",
          "public domain (statsmodels' randhie dataset; RAND Health Insurance Experiment)",
          {"mdvis": "outpatient visits to a doctor", "lncoins": "log(coinsurance + 1)", "idp": "1 if individual deductible plan",
           "lpi": "log(max(1, annual participation incentive payment))", "fmde": "log(max(1, maximum dollar expenditure)) if no deductible",
           "physlm": "1 if physical limitation", "disea": "number of chronic diseases", "hlthg": "1 if good health",
           "hlthf": "1 if fair health", "hlthp": "1 if poor health"}, entity="person")
    anes = importlib.import_module("statsmodels.datasets.anes96").load_pandas().data.reset_index(drop=True)
    anes.insert(0, "respondent", np.arange(1, len(anes) + 1))
    _save("anes96", anes, "American National Election Study, 1996", "ordered and multinomial: PID, party identification from 0 "
          "(strong Democrat) to 6 (strong Republican); vote", "public domain (statsmodels' anes96 dataset; ANES)",
          {"PID": "party identification, 0 strong Democrat to 6 strong Republican", "popul": "population of the respondent's area, 1000s",
           "TVnews": "days a week watching TV news", "selfLR": "self-placement, 1 extremely liberal to 7 extremely conservative",
           "ClinLR": "placement of Clinton, 1 to 7", "DoleLR": "placement of Dole, 1 to 7", "age": "age in years",
           "educ": "education, 1 to 7", "income": "income group, 1 to 24", "vote": "1 if voted for Dole, 0 for Clinton"}, entity="respondent")
    grunfeld = importlib.import_module("statsmodels.datasets.grunfeld").load_pandas().data
    grunfeld = grunfeld.assign(year=grunfeld["year"].astype(int))
    _save("grunfeld", grunfeld, "Grunfeld's investment data, 11 US firms, 1935-1954", "panel and systems of equations",
          "public domain (statsmodels' grunfeld dataset; Kleiber and Zeileis)",
          {"invest": "gross investment", "value": "market value of the firm", "capital": "stock of plant and equipment"},
          entity="firm", time="year", freq="Y")


def simulated_sets() -> None:
    rng = np.random.default_rng(20261025)
    # difference-in-differences and event study: 50 states, 2000-2019; half adopt in 2010; the effect is 2.0
    states, years = 50, np.arange(2000, 2020)
    unit = rng.normal(0, 2, states)
    year_effect = np.cumsum(rng.normal(0.2, 0.3, len(years)))
    rows = []
    for s in range(states):
        treated = s < states // 2
        x = rng.normal(size=len(years))
        for j, yr in enumerate(years):
            post = int(treated and yr >= 2010)
            rows.append({"state": s + 1, "year": int(yr), "treated": int(treated), "post": int(yr >= 2010), "d": post, "x": float(x[j]),
                         "y": float(5 + unit[s] + year_effect[j] + 0.5 * x[j] + 2.0 * post + rng.normal(0, 1))})
    _save("policy_panel", pd.DataFrame(rows), "A policy adopted by half the states in 2010 (simulated)",
          "difference-in-differences and event studies: y = state + year effects + 0.5 x + 2.0 d + noise, d = treated and post (the true effect is 2.0)",
          "simulated by pymetrix (BSD-3)", {"treated": "1 for the states that adopt the policy", "post": "1 from 2010",
                                            "d": "1 where the policy is in force (treated and post)", "x": "a control", "y": "the outcome"},
          entity="state", time="year", freq="Y")
    # regression discontinuity: the score's cutoff at 0 raises the outcome by 1.5
    n = 2000
    score = rng.uniform(-1, 1, n)
    y = 1 + 0.8 * score + 0.5 * score**2 + 1.5 * (score >= 0) + rng.normal(0, 0.5, n)
    _save("rd_sample", pd.DataFrame({"id": np.arange(1, n + 1), "score": score, "above": (score >= 0).astype(int), "y": y}),
          "A programme given to those scoring at least 0 (simulated)",
          "regression discontinuity: y = 1 + 0.8 score + 0.5 score^2 + 1.5 above + noise (the true jump at 0 is 1.5)",
          "simulated by pymetrix (BSD-3)", {"score": "the running variable; the cutoff is 0", "above": "1 if score >= 0", "y": "the outcome"},
          entity="id")
    # synthetic control: region 1 adopts in 2005, lowering its outcome by 3; the others follow common factors
    regions, years = 20, np.arange(1990, 2016)
    factors = np.cumsum(rng.normal(0, 1, (len(years), 2)), axis=0)
    loads = rng.uniform(0.5, 1.5, (regions, 2))
    loads[0] = loads[1:4].mean(axis=0)  # region 1 within the donors' range, as synthetic control assumes
    rows = []
    for r in range(regions):
        for j, yr in enumerate(years):
            effect = -3.0 if r == 0 and yr >= 2005 else 0.0
            rows.append({"region": r + 1, "year": int(yr), "y": float(50 + loads[r] @ factors[j] + effect + rng.normal(0, 0.3))})
    _save("synth_panel", pd.DataFrame(rows), "One region's policy from 2005 (simulated)",
          "synthetic control: region 1 is treated from 2005 and its outcome falls by 3.0; regions 2-20 are the donors",
          "simulated by pymetrix (BSD-3)", {"y": "the outcome"}, entity="region", time="year", freq="Y")



if __name__ == "__main__":
    linearmodels_sets()
    statsmodels_sets()
    simulated_sets()
