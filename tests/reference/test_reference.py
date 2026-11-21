"""Reference conformance: key estimators against certified and published values from other sources.

Each case is checked twice: through the public API, and as a recorded call (pymetrix.calls) saved to JSON and
replayed in a fresh session, so the two paths must give the same certified numbers. The fixtures, with their
sources, are built by tests/fixtures/reference/make_reference.py.

- OLS: NIST StRD Longley (certified values), with Stata's log likelihood, AIC and BIC and White (HC1) errors.
- ARMA: gretl's exact maximum likelihood, ARMA(1,1) and ARMA(2,2) with a constant.
- VAR: JMulTi's VAR(4) with a constant (three decimals).

Other NIST StRD sets (Norris, Pontius, NoInt1/2, Filip, Wampler1-5) are added when their files can be
fetched from NIST (see docs/design/object-model-plan.md, decision D6).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import ARMA, OLS, VAR, Dataset, Equation
from pymetrix.commands import Session
from pymetrix.commands.registry import from_json, to_json

FIXTURES = Path(__file__).parents[1] / "fixtures" / "reference"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def lre(estimate: float, certified: float) -> float:
    """The log relative error: the number of correct significant digits (NIST's measure)."""
    if estimate == certified:
        return 15.0
    scale = abs(certified) if certified != 0 else 1.0
    return float(-np.log10(abs(estimate - certified) / scale))


def _printed(value: float) -> float:
    """Half a unit in the last decimal place printed: the most a correctly rounded printed value can be off."""
    text = repr(float(value))
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text and not float(value).is_integer() else 0
    return 0.5 * 10 ** -decimals + 1e-12


def _replayed(code: str, objects: dict, output: str):
    """Run ``code`` in a session (recorded as a call), save the call as JSON, replay it in a fresh session."""
    recorded = Session()
    recorded.objects.update(objects)
    recorded.run_code(code)
    step = recorded.history[-1]
    assert type(step.command).__name__ == "Call", f"{code!r} was not recorded as a call"
    call = from_json(json.loads(json.dumps(to_json(step.command))))
    fresh = Session()
    fresh.objects.update(objects)
    fresh.run(call)
    return recorded[output], fresh[output]


# -- OLS: NIST StRD Longley ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def longley():
    ref = _fixture("longley")
    frame = pd.DataFrame(ref["data"], index=pd.period_range(ref["first_period"], periods=len(ref["data"]["TOTEMP"]), freq=ref["freq"]))
    return ref, Dataset.from_frame(frame)


def _check_longley(fit, ref, digits: float) -> None:
    cert = ref["certified"]
    names = list(fit.params.names)
    for name, value in cert["params"].items():
        assert lre(fit.params[name], value) >= digits, f"{name}: LRE {lre(fit.params[name], value):.1f}"
    se = dict(zip(names, fit.params.se))
    for name, value in cert["std_errors"].items():
        assert lre(se[name], value) >= digits - 1, f"s.e. {name}: LRE {lre(se[name], value):.1f}"
    assert lre(fit.stats["ssr"], cert["ssr"]) >= digits
    assert lre(fit.stats["r2"], cert["rsquared"]) >= digits
    assert lre(fit.stats["se_regression"], cert["residual_sd"]) >= digits
    assert fit.stats["df_resid"] == cert["df_resid"]


def test_longley_through_the_api(longley):
    ref, data = longley
    fit = Equation(ref["formula"]).fit(data)
    _check_longley(fit, ref, digits=9)  # NIST's "higher difficulty" set: at least 9 correct digits
    other = ref["other_package"]
    assert abs(fit.stats["loglike"] - other["loglike"]) < 1e-4  # Stata, printed to 4 decimals
    assert abs(fit.stats["aic"] - other["aic"]) < 1e-4 and abs(fit.stats["bic"] - other["bic"]) < 1e-3


def test_longley_white_errors(longley):
    ref, data = longley
    fit = Equation(ref["formula"]).fit(data, method=OLS(cov="white"))
    se = dict(zip(fit.params.names, fit.params.se))
    for name, value in ref["other_package"]["white_hc1_se"].items():
        # Stata, as printed; on this ill-conditioned problem its sandwich and ours also differ by up to 3e-8 relatively
        assert abs(se[name] - value) <= max(_printed(value), 1e-7 * abs(value)), name


def test_longley_replayed_as_a_call(longley):
    ref, data = longley
    live, replayed = _replayed(f"fit = Equation({ref['formula']!r}).fit(data)", {"data": data}, "fit")
    _check_longley(replayed, ref, digits=9)
    assert np.array_equal(live.params.values, replayed.params.values)


# -- ARMA: gretl -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["y_arma11c", "y_arma22c"])
def test_arma_against_gretl(case):
    ref = _fixture("arma_gretl")["cases"][case]
    data = Dataset.from_frame(pd.DataFrame({"y": ref["data"]}, index=pd.period_range("1800", periods=len(ref["data"]), freq="Y")))
    code = f"fit = Equation('y ~ 1', errors=ARMA(ar={ref['ar']}, ma={ref['ma']})).fit(data)"
    api = Equation("y ~ 1", errors=ARMA(ar=ref["ar"], ma=ref["ma"])).fit(data)
    live, replayed = _replayed(code, {"data": data}, "fit")
    for fit in (api, replayed):
        se = dict(zip(fit.params.names, fit.params.se))
        for name, value in ref["params"].items():
            assert abs(fit.params[name] - value) < 1e-5, name
            assert np.isclose(se[name], ref["std_errors"][name], rtol=1e-3), f"s.e. {name}"  # gretl: a numerical Hessian
        assert abs(fit.stats["loglike"] - ref["loglike"]) < 1e-6
        assert abs(fit.params["sigma2"] - ref["sigma2"]) < 1e-5
    assert np.array_equal(live.params.values, replayed.params.values)


# -- VAR: JMulTi -----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jmulti():
    ref = _fixture("var_jmulti")
    frame = pd.DataFrame(ref["data"], index=pd.period_range(ref["first_period"], periods=len(ref["data"]["realcons"]), freq=ref["freq"]))
    return ref, Dataset.from_frame(frame)


def _check_var(fit, ref) -> None:
    tol = 0.5 * 10 ** -ref["digits"] + 1e-9  # JMulTi prints three decimals
    coefs = fit.coefficients()
    assert fit.provenance.nobs == 199
    for eq in ref["variables"]:
        rows = coefs[coefs["equation"] == eq].set_index("term")
        for term, value in ref["params"][eq].items():
            assert abs(rows.loc[term, "estimate"] - value) <= tol, f"{eq}: {term}"
            assert abs(rows.loc[term, "std_error"] - ref["std_errors"][eq][term]) <= tol, f"{eq}: s.e. {term}"


def test_var_against_jmulti(jmulti):
    ref, data = jmulti
    _check_var(VAR(ref["variables"], lags=ref["lags"]).fit(data), ref)


def test_var_replayed_as_a_call(jmulti):
    ref, data = jmulti
    live, replayed = _replayed(f"var = VAR({ref['variables']!r}, lags={ref['lags']}).fit(data)", {"data": data}, "var")
    _check_var(replayed, ref)
    assert np.array_equal(live.params.values, replayed.params.values)
