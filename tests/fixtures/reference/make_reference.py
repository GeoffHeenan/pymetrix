"""Rebuild the reference fixtures from the copies of the data and published results that statsmodels ships (BSD-3).

    python tests/fixtures/reference/make_reference.py

The JSON files are committed; this script documents where each number came from and lets anyone check the
transcription. It reads statsmodels' installed test data, never a network source.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import statsmodels

HERE = Path(__file__).parent
SM = Path(statsmodels.__file__).parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # statsmodels' own result tables: plain numbers
    return module


def longley() -> dict:
    data = pd.read_csv(SM / "datasets" / "longley" / "longley.csv")
    ref = _load(SM / "regression" / "tests" / "results" / "results_regression.py", "results_regression").Longley()
    names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR", "const"]  # NIST's order: b1..b6, then b0
    return {
        "name": "NIST StRD Longley",
        "source": "NIST Statistical Reference Datasets, Longley (higher difficulty), certified values; as transcribed in "
                  "statsmodels/regression/tests/results/results_regression.py. Log likelihood, AIC and BIC from Stata (same file).",
        "formula": "TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR",
        "data": {c: data[c].astype(float).tolist() for c in ["TOTEMP", "GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]},
        "first_period": "1947", "freq": "Y",
        "certified": {"params": dict(zip(names, ref.params)), "std_errors": dict(zip(names, ref.bse)), "ssr": ref.ssr,
                      "rsquared": ref.rsquared, "rsquared_adj": ref.rsquared_adj, "residual_sd": ref.scale ** 0.5,
                      "df_resid": ref.df_resid},
        "other_package": {"loglike": ref.llf, "aic": ref.aic, "bic": ref.bic, "white_hc1_se": dict(zip(names, ref.HC1_se))},
    }


def arma() -> dict:
    results = SM / "tsa" / "tests" / "results"
    module = _load(results / "results_arma.py", "results_arma")
    data = pd.read_csv(results / "y_arma_data.csv")
    cases = {}
    for column, cls, ar, ma in (("y_arma11c", "Y_arma11c", 1, 1), ("y_arma22c", "Y_arma22c", 2, 2)):
        ref = getattr(module, cls)()
        names = ["const", *[f"ar[{i}]" for i in range(1, ar + 1)], *[f"ma[{i}]" for i in range(1, ma + 1)]]
        cases[column] = {"ar": ar, "ma": ma, "data": data[column].astype(float).tolist(),
                         "params": dict(zip(names, ref.params)), "std_errors": dict(zip(names, ref.bse)),
                         "loglike": float(ref.llf), "sigma2": float(ref.sigma2)}
    return {"name": "ARMA(p, q) with a constant, exact maximum likelihood",
            "source": "gretl (exact ML), as recorded in statsmodels/tsa/tests/results/results_arma.py, on y_arma_data.csv",
            "cases": cases}


def _jmulti(text: str, variables: list[str]) -> tuple[dict, dict]:
    """Coefficients and standard errors from JMulTi's printed VAR estimates (three decimals)."""
    coef: dict[str, dict[str, float]] = {v: {} for v in variables}
    se: dict[str, dict[str, float]] = {v: {} for v in variables}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(\w+)\s*\(t-(\d+)\)\|(.*)$", line) or re.match(r"^(CONST)\s*\|(.*)$", line)
        if not m:
            continue
        if m.group(1) == "CONST":
            term, values = "const", m.group(2)
        else:
            term, values = f"{m.group(1)}[-{m.group(2)}]", m.group(3)
        estimates = [float(x) for x in values.split()]
        errors = [float(x) for x in re.findall(r"\(([-\d.]+)\)", lines[i + 1])]
        for v, b, s in zip(variables, estimates, errors):
            coef[v][term], se[v][term] = b, s
    return coef, se


def var() -> dict:
    macro = pd.read_csv(SM / "datasets" / "macrodata" / "macrodata.csv")
    variables = ["realcons", "realgdp", "realinv"]
    text = (SM / "tsa" / "vector_ar" / "tests" / "JMulTi_results" / "macrodata_jmulti_c.txt").read_text()
    coef, se = _jmulti(text, variables)
    return {"name": "VAR(4) with a constant, least squares",
            "source": "JMulTi, as recorded in statsmodels/tsa/vector_ar/tests/JMulTi_results/macrodata_jmulti_c.txt "
                      "(printed to three decimals), on statsmodels' macrodata in levels, 1959Q1 to 2009Q3",
            "variables": variables, "lags": 4, "first_period": "1959Q1", "freq": "Q", "digits": 3,
            "data": {v: macro[v].astype(float).tolist() for v in variables}, "params": coef, "std_errors": se}


if __name__ == "__main__":
    for name, make in (("longley", longley), ("arma_gretl", arma), ("var_jmulti", var)):
        (HERE / f"{name}.json").write_text(json.dumps(make(), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)
