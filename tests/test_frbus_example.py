"""The FRB/US worked example (docs/tutorials/frbus.md): the Federal Reserve Board's public FRB/US model simulated in
pymetrix as the Fed's own example programs do (programs/example1-4.prg, pings.prg).

What is checked, with the Fed's packages (``PYMETRIX_FRBUS``: the frbus_package folder; ``PYMETRIX_FRBUS_DATA``: the
data_only_package folder; skipped without them, as the packages are not part of pymetrix):

- tracking, VAR expectations: with add factors from LONGBASE (``Model.add_factors``), the model solved by Newton from
  blanked endogenous values over the 40 quarters after the history reproduces the baseline projection (1e-6 relative);
- solver agreement: Newton and Gauss-Seidel give the same response to a funds-rate shock (1e-8 relative);
- the funds-rate shock of example1.prg and the federal-purchases ping of pings.prg (1 percent of GDP): signs and sizes
  against the responses the Fed's own documentation plots (A New FRB/US Price-Wage Sector, Figure 3), and a doubled
  fiscal shock giving a response within 5 percent of twice the single one;
- alternative policy rules through the switch variables (``dmpex``, ``dmptay``): with the funds rate exogenous the
  fiscal shock does more, and the plain Taylor rule undoes a shock to it at once;
- model-consistent expectations (asset pricing and price-wage setting, example2.prg's "-mcap+wp"): the stacked
  perfect-foresight solution over 240 quarters, terminal values from LONGBASE, reproduces the baseline from blanked
  values (with no shock it is the baseline), satisfies every equation in every period, and gives the funds-rate shock
  responses of the right sign and size next to those under VAR expectations;
- the zero lower bound and liftoff thresholds (example3.prg, VAR expectations, from 2003Q3): the floor never breached
  and binding where the rule is below it, the switches turning on exactly when the thresholds are crossed, every
  equation satisfied, Newton and Gauss-Seidel agreeing;
- a new policy rule appended (example4.prg, model-consistent expectations): the rule holding along the solution, and
  with the inertial Taylor rule's coefficients the responses of the inertial Taylor rule;
- stochastic simulations (stochsim.prg, VAR expectations): the historical errors of the stochastic equations
  resampled a quarter at a time, the draws equal to the baseline with zero errors, reproducible by seed and
  generator, each quarter's errors one historical quarter's, the median near the baseline;
- optimal control (ocpolicy.prg, VAR expectations): the first-order conditions at the optimum, the gradient against
  central differences, the loss below the Taylor rule's.

Without the packages, the same steps run on the excerpt of FRB/US in the repository (six equations and their data,
tests/fixtures/eviews: from the Fed's package, provenance in the files): add factors that make every solver
(Newton, Gauss-Seidel, perfect foresight with model-consistent expectations) reproduce the data, and the compiled
period-by-period solvers equal to the interpreted ones; and the machinery of the last two programs on small models
whose answers are known: the variance of a resampled AR(1) against its analytic value, and the optimal-control path
of a linear-quadratic model against pymetrix's analytic optimal policy under commitment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix.data import Dataset
from pymetrix.estimation import Equation
from pymetrix.models import GaussSeidel, Model, Newton, PerfectForesight, Scenario, fast
from pymetrix.models.eviews import read_eviews_xml

FIXTURES = Path(__file__).parent / "fixtures" / "eviews"
FRBUS = os.environ.get("PYMETRIX_FRBUS")  # the frbus_package folder (with mods/model.xml)
FRBUS_DATA = os.environ.get("PYMETRIX_FRBUS_DATA")  # the data_only_package folder (with LONGBASE.TXT, HISTDATA.TXT)
needs_frbus = pytest.mark.skipif(not (FRBUS and FRBUS_DATA), reason="set PYMETRIX_FRBUS and PYMETRIX_FRBUS_DATA to the FRB/US packages")

MONETARY = ("dmpex", "dmprr", "dmptay", "dmptlr", "dmpintay", "dmpalt", "dmpgen")  # the Fed's set_mp switches
FISCAL = ("dfpex", "dfpsrp", "dfpdbt")  # set_fp's


def with_add_factor_series(frame: pd.DataFrame, model) -> pd.DataFrame:
    """The ``_aerr`` series every FRB/US equation carries, at zero where the data have none (as the Fed's programs)."""
    missing = [v for v in model.exogenous if v.endswith("_aerr") and v not in frame.columns]
    return pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1)


def policy(frame: pd.DataFrame, monetary: str = "dmpintay", fiscal: str = "dfpsrp", rstar_from: pd.Period | None = None) -> pd.DataFrame:
    """The settings of the Fed's examples: one monetary and one fiscal switch on, no zero bound or thresholds, the
    policymaker's equilibrium real rate exogenous (endogenous from ``rstar_from`` on, as in example2.prg)."""
    frame = frame.copy()
    for k in MONETARY:
        if k in frame.columns:
            frame[k] = float(k == monetary)
    for k in FISCAL:
        frame[k] = float(k == fiscal)
    frame["dmptrsh"], frame["rffmin"], frame["drstar"] = 0.0, -9999.0, 0.0
    if rstar_from is not None:
        frame.loc[rstar_from:, "drstar"] = 1.0
    return frame


def worst_gap(solved: pd.DataFrame, baseline: pd.DataFrame, names, rows) -> tuple[float, str]:
    """The largest gap relative to max(1, |baseline|), and where."""
    worst, where = 0.0, ""
    for v in names:
        got, want = solved.loc[rows, v].to_numpy(float), baseline.loc[rows, v].to_numpy(float)
        gap = float(np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))))
        if not gap <= worst:
            worst, where = gap, v
    return worst, where


def max_residual(model, solved: pd.DataFrame, factors, rows) -> float:
    """Every equation in every period of ``rows``: the largest residual at the solution, add factors included."""
    at = model.add_factors(Dataset.from_frame(solved), (str(rows[0]), str(rows[-1])))
    return max(float(np.max(np.abs(at[name].reindex(rows).to_numpy() - factors[name].reindex(rows).fillna(0.0).to_numpy())))
               for name in at)


# ---------------------------------------------------------------------------------------------
# without the packages: the excerpt in the repository


def _excerpt(mce=False):
    read = read_eviews_xml(FIXTURES / "frbus_excerpt.xml", mce=mce)
    frame = pd.read_csv(FIXTURES / "frbus_longbase_excerpt.csv", comment="#")
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q", name="time")
    return read.model, with_add_factor_series(frame, read.model)


@pytest.mark.parametrize("solver", [Newton(), GaussSeidel(tol=1e-12), PerfectForesight(terminal="data")], ids=["newton", "gauss_seidel", "perfect_foresight"])
def test_excerpt_add_factors_reproduce_the_baseline(solver):
    model, frame = _excerpt(mce=["mcap"] if isinstance(solver, PerfectForesight) else False)
    assert model.has_leads == isinstance(solver, PerfectForesight)
    span = ("2026Q2", "2027Q2")
    factors = model.add_factors(Dataset.from_frame(frame), span)
    assert set(factors) == {e.name for e in model.model_equations} and len(factors["zdivgr"]) == 5
    blank = frame.copy()
    blank.loc["2026Q2":"2027Q2", list(model.endogenous)] = np.nan
    solved = model.solve(Dataset.from_frame(blank), span, Scenario(adjust=factors), solver=solver).frame
    rows = pd.period_range("2026Q2", "2027Q2", freq="Q")
    worst, where = worst_gap(solved, frame, model.endogenous, rows)
    assert worst < 1e-9, where
    without = model.solve(Dataset.from_frame(blank), span, solver=solver).frame  # no add factors: the equations' own paths
    assert worst_gap(without, frame, model.endogenous, rows)[0] > 1e-4


@pytest.mark.parametrize("solver", [Newton(), GaussSeidel(tol=1e-13)], ids=["newton", "gauss_seidel"])
def test_compiled_period_solvers_are_the_interpreter(solver, monkeypatch):
    model = Model([Equation("c = 0.5*y + 0.3*c[-1] + 0.1*log(w)", name="cons"),  # a simultaneous block (c, y, i)
                   Equation("i = 0.2*y - 2*max(r - 0.5, 0) + 0.4*i[-1]", name="inv"),
                   Equation("y = c + i + g", name="gdp"), Equation("w = 0.9*w[-1] + 0.1*y", name="wage")])
    index = pd.period_range("2000Q1", periods=24, freq="Q", name="time")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({v: 1.0 + 0.1 * rng.standard_normal(len(index)) for v in ("c", "i", "y", "w", "g")}, index=index)
    frame["r"] = 0.45 + 0.05 * rng.standard_normal(len(index))  # the shock takes it past the kink of max(r - 0.5, 0)
    data = Dataset.from_frame(frame)
    scenario = Scenario(shift={"r": {"2001Q1": 0.3, "2001Q2": 0.2}}, adjust={"cons": 0.1})
    compiled = model.solve(data, ("2001Q1", "2005Q4"), scenario, solver=solver)
    monkeypatch.setattr(fast, "ENABLED", False)
    interpreted = model.solve(data, ("2001Q1", "2005Q4"), scenario, solver=solver)
    np.testing.assert_allclose(compiled.frame.to_numpy(float), interpreted.frame.to_numpy(float), atol=1e-11)
    assert compiled.iterations == interpreted.iterations


def test_newton_takes_a_saturated_logistic_switch():
    """FRB/US's threshold switches are logistic, 1/(1+exp(25*(lur-lurtrsh))), and example3.prg parks the thresholds at
    -9999 and 9999 where they are not to act: exp overflows, and the analytic derivative is inf/inf. Newton takes a
    forward difference for such entries (0 here, where the switch is flat) instead of failing."""
    model = Model([Equation("s = 1/(1 + exp(25*(u - k)))", name="switch"), Equation("u = 0.5*u[-1] + 0.5*s + e", name="u")])
    index = pd.period_range("2000Q1", periods=8, freq="Q", name="time")
    frame = pd.DataFrame({"s": 0.0, "u": 5.0, "k": -9999.0, "e": 0.0}, index=index)
    solved = model.solve(Dataset.from_frame(frame), ("2000Q2", "2001Q4"), Scenario(shift={"e": {"2000Q2": 1.0}})).frame
    np.testing.assert_allclose(solved.loc["2000Q2":, "u"], [3.5, 1.75, 0.875, 0.4375, 0.21875, 0.109375, 0.0546875])
    assert np.all(solved.loc["2000Q2":, "s"] == 0.0)
    frame["k"] = 1.0  # the switch where it acts: the same solution as by Gauss-Seidel
    newton = model.solve(Dataset.from_frame(frame), ("2000Q2", "2001Q4"), Scenario(shift={"e": {"2000Q2": -4.0}})).frame
    gauss = model.solve(Dataset.from_frame(frame), ("2000Q2", "2001Q4"), Scenario(shift={"e": {"2000Q2": -4.0}}),
                        solver=GaussSeidel(tol=1e-12, max_iter=5000)).frame
    np.testing.assert_allclose(newton.to_numpy(float), gauss.to_numpy(float), atol=1e-10)


def test_perfect_foresight_line_search_is_relative_to_each_equation():
    """A level identity in the thousands (big = 10000*exp(x)) next to an equation in logs: the Newton step leaves a
    second-order residual of hundreds in the identity, small next to its level. The line search judges it relative
    to the equation's terms, so the full step is taken: 2 iterations (judged by the raw norm, the steps were cut
    until the 50 iterations ran out)."""
    model = Model([Equation("x = 0.5*x[+1] + e", name="x"), Equation("big = 10000*exp(x)", name="big"),
                   Equation("z = 0.9*z[-1] + 0.0001*(big - 10000)", name="z")])
    index = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    frame = pd.DataFrame({"x": 0.0, "big": 10000.0, "z": 0.0, "e": 0.0}, index=index)
    solved = model.solve(Dataset.from_frame(frame), ("2000Q2", "2009Q3"), Scenario(shift={"e": {"2000Q2": 0.5}}),
                         solver=PerfectForesight(terminal="data"))
    assert solved.iterations <= 3
    assert solved.frame.loc["2000Q2", "big"] == pytest.approx(10000 * np.exp(0.5), rel=1e-12)
    assert solved.frame.loc["2000Q2", "z"] == pytest.approx(0.0001 * 10000 * (np.exp(0.5) - 1), rel=1e-10)


# ---------------------------------------------------------------------------------------------
# with the Fed's packages


QUARTERS = 40  # VAR expectations: the 40 quarters after the history (example1.prg runs 24)
HORIZON = 240  # model-consistent expectations: 60 years, as example2.prg (terminal values from LONGBASE beyond)


@pytest.fixture(scope="module")
def frbus():
    """The standard (VAR-expectations) model, LONGBASE with the examples' policy settings, and the tracking add factors."""
    model = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml").model
    frame = pd.read_csv(Path(FRBUS_DATA) / "LONGBASE.TXT")
    frame.columns = [c.lower() for c in frame.columns]
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q", name="time")
    start = pd.Period(pd.read_csv(Path(FRBUS_DATA) / "HISTDATA.TXT", usecols=[0]).iloc[-1, 0], "Q") + 1
    frame = with_add_factor_series(policy(frame), model)
    span = (str(start), str(start + QUARTERS - 1))
    factors = model.add_factors(Dataset.from_frame(frame), span)
    return {"model": model, "frame": frame, "start": start, "span": span, "factors": factors,
            "rows": pd.period_range(start, periods=QUARTERS, freq="Q")}


def _solve(f, quarters=QUARTERS, frame=None, solver=None, model=None, factors=None, **changes) -> pd.DataFrame:
    start = f["start"]
    model = model or f["model"]
    data = Dataset.from_frame(f["frame"] if frame is None else frame)
    scenario = Scenario(adjust=f["factors"] if factors is None else factors, **changes)
    solved = model.solve(data, (str(start), str(start + quarters - 1)), scenario, solver=solver or Newton())
    return solved.frame.loc[start:start + quarters - 1]


def _first(f, value: float) -> dict[str, float]:
    return {str(f["start"]): value}


@needs_frbus
def test_var_baseline_is_tracked(frbus):
    blank = frbus["frame"].copy()
    blank.loc[frbus["rows"], list(frbus["model"].endogenous)] = np.nan
    solved = _solve(frbus, frame=blank)
    worst, where = worst_gap(solved, frbus["frame"], frbus["model"].endogenous, frbus["rows"])
    assert worst < 1e-6, where  # 2.1e-8 found (wps, a level in the tens of thousands)


@needs_frbus
def test_newton_and_gauss_seidel_agree(frbus):
    shock = {"rffintay_aerr": _first(frbus, 1.0)}
    newton = _solve(frbus, 12, shift=shock)
    gauss = _solve(frbus, 12, shift=shock, solver=GaussSeidel(tol=1e-9, max_iter=5000))
    worst, where = worst_gap(gauss, newton, frbus["model"].endogenous, newton.index)
    assert worst < 1e-8, where  # 6e-10 found


@needs_frbus
def test_funds_rate_shock(frbus):
    """example1.prg: 100 basis points on the inertial Taylor rule for one quarter. The Fed's own responses (Brayton,
    "A New FRB/US Price-Wage Sector", 2013, in the package's documentation folder: Figure 3, page 16, first row: VAR
    expectations) have the output gap falling to about -0.55 after a year and a half, the funds rate back to zero
    after two years and a little below it after, and core inflation down by about a tenth; the model has been
    re-estimated since, so the ranges here are generous."""
    base = _solve(frbus)
    d = _solve(frbus, shift={"rffintay_aerr": _first(frbus, 1.0)}) - base
    assert d.rff.iloc[0] == pytest.approx(1.0, abs=0.01) and 0.5 < d.rff.iloc[1] < 1.0  # inertial: the rate comes down slowly
    trough = d.xgap2.iloc[:12].min()
    assert -0.9 < trough < -0.25 and 3 <= int(d.xgap2.iloc[:12].argmin()) <= 10  # -0.50 in the 8th quarter found
    assert d.lur.iloc[:12].max() > 0.1 and d.pic4.iloc[4:12].max() < 0 and d.rff.iloc[8:20].min() < 0


@needs_frbus
def test_federal_purchases_multiplier(frbus):
    """pings.prg: federal purchases (egfe) up by 1 percent of baseline GDP for one quarter, through egfe_aerr. The
    Fed's own response (Brayton, "A New FRB/US Price-Wage Sector", 2013: Figure 3, page 16, second row: VAR
    expectations): the output gap up by about 1.0 percentage point on impact, the spending, as a share of GDP, up by
    about 1.1, both decaying, the gap below zero (about -0.25) from the third year. Here: the impact on the output gap
    per point of the spending share within 0.5 to 1.5 (0.80 found), the gap negative within five years, and a doubled
    shock giving within 5 percent of twice the response (0.7 percent for the output gap found)."""
    start, frame = frbus["start"], frbus["frame"]
    size = 0.01 * frame.loc[start, "xgdpn"] / frame.loc[start, "egfen"]  # pings.prg's egfe_aerr shift
    base = _solve(frbus)
    once = _solve(frbus, shift={"egfe_aerr": _first(frbus, size)})
    twice = _solve(frbus, shift={"egfe_aerr": _first(frbus, 2 * size)})
    share = 100 * (once.egfen / once.xgdpn - base.egfen / base.xgdpn)
    d1, d2 = once - base, twice - base
    assert 0.9 < share.iloc[0] < 1.3  # 1.07 found
    assert 0.5 < d1.xgap2.iloc[0] / share.iloc[0] < 1.5 and d1.xgap2.iloc[0] == d1.xgap2.max()
    assert d1.xgap2.iloc[:20].min() < 0 and d1.rff.iloc[:8].max() > 0 and d1.lur.iloc[0] < 0
    for v in ("xgap2", "xgdp", "lur", "rff"):
        assert float((d2[v] - 2 * d1[v]).abs().max()) < 0.05 * float((2 * d1[v]).abs().max()), v


@needs_frbus
def test_policy_rules(frbus):
    """The monetary-policy switches (set_mp in the Fed's master_library.prg): with the funds rate exogenous (dmpex)
    the fiscal shock above raises output more than under the inertial Taylor rule, and a shock to the plain Taylor
    rule (dmptay) is undone the next quarter, where the inertial rule takes years."""
    start, frame, model = frbus["start"], frbus["frame"], frbus["model"]
    size = 0.01 * frame.loc[start, "xgdpn"] / frame.loc[start, "egfen"]

    def rule(name):
        return {k: float(k == name) for k in MONETARY if k in model.exogenous}

    def response(name, shift):
        return _solve(frbus, 20, set=rule(name), shift=shift) - _solve(frbus, 20, set=rule(name))

    fiscal = {"egfe_aerr": _first(frbus, size)}
    inertial, fixed = response("dmpintay", fiscal), response("dmpex", fiscal)
    assert float(fixed.rff.abs().max()) < 1e-8 and inertial.rff.iloc[:8].min() > 0
    assert fixed.xgap2.iloc[:8].sum() > 1.1 * inertial.xgap2.iloc[:8].sum()  # 4.5 against 3.5 found
    taylor = response("dmptay", {"rfftay_aerr": _first(frbus, 1.0)})
    inertial = response("dmpintay", {"rffintay_aerr": _first(frbus, 1.0)})
    assert taylor.rff.iloc[0] == pytest.approx(1.0, abs=0.01) and abs(taylor.rff.iloc[1]) < 0.3 < inertial.rff.iloc[1]


@pytest.fixture(scope="module")
def frbus_mce(frbus):
    """The model with model-consistent expectations in asset pricing and price-wage setting (example2.prg's
    "-mcap+wp"), the equilibrium real rate endogenous after 40 quarters, its add factors over 240 quarters."""
    model = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml", mce=["mcap", "mcwp"]).model
    start = frbus["start"]
    frame = with_add_factor_series(policy(frbus["frame"], rstar_from=start + 40), model)
    rows = pd.period_range(start, periods=HORIZON, freq="Q")
    factors = model.add_factors(Dataset.from_frame(frame), (str(rows[0]), str(rows[-1])))
    return {**frbus, "model": model, "frame": frame, "factors": factors, "rows": rows}


PF = PerfectForesight(terminal="data", tol=1e-8)  # FRB/US's levels are in the thousands: 1e-10 is below rounding


@needs_frbus
def test_mce_baseline_is_tracked(frbus_mce):
    f = frbus_mce
    blank = f["frame"].copy()
    blank.loc[f["rows"][:40], list(f["model"].endogenous)] = np.nan  # starting values far off for ten years
    solved = _solve(f, HORIZON, frame=blank, solver=PF)
    worst, where = worst_gap(solved, f["frame"], f["model"].endogenous, f["rows"])
    assert worst < 1e-6, where  # 8e-9 found
    assert max_residual(f["model"], solved, f["factors"], f["rows"]) < 1e-8


@needs_frbus
def test_mce_funds_rate_shock_against_var(frbus_mce):
    """example2.prg: the funds-rate shock with model-consistent expectations in asset pricing and price-wage setting.
    The Fed's responses under these expectations (Brayton 2013, Figure 5, page 18, first row) are smaller than under
    VAR expectations (Figure 3): the output gap falls to about -0.3; long rates move less on impact, as markets see
    the shock fade."""
    f = frbus_mce
    shock = {"rffintay_aerr": _first(f, 1.0)}
    mce = _solve(f, HORIZON, shift=shock, solver=PF)
    assert max_residual(f["model"], mce, f["factors"], f["rows"]) < 1e-8
    d_mce = (mce - f["frame"].loc[f["rows"]]).iloc[:40]
    var_frame = with_add_factor_series(policy(f["frame"], rstar_from=f["start"] + 40), f["model"])
    var = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml").model
    var_factors = var.add_factors(Dataset.from_frame(var_frame), (str(f["rows"][0]), str(f["rows"][39])))
    d_var = _solve(f, 40, frame=var_frame, model=var, factors=var_factors, shift=shock) - var_frame.loc[f["rows"][:40]]
    for d in (d_mce, d_var):
        assert d.rff.iloc[0] == pytest.approx(1.0, abs=0.01) and d.xgap2.iloc[:12].min() < -0.1 and d.lur.iloc[:12].max() > 0
    assert -0.8 < d_mce.xgap2.iloc[:12].min() < -0.1  # -0.19 found (VAR: -0.50)
    assert 0 < d_mce.rg10.iloc[0] < d_var.rg10.iloc[0]  # 0.15 against 0.33 found


# ---------------------------------------------------------------------------------------------
# example3.prg: the zero lower bound and the liftoff thresholds (VAR expectations), on the history from 2003Q3


ZLB_START = pd.Period("2003Q3", "Q")  # example3.prg: 24 quarters from 2003Q3, the plain Taylor rule (rfftay)
ZLB_QUARTERS = 24
# its shocks, the equation errors of 2008Q4-2009Q3, in the four quarters from 2003Q3 (each replaces the add factor)
ZLB_SHOCKS = {"eco": (-.006, -.006, -.011, -.001), "ecd": (-.091, -.018, -.021, .029), "eh": (-.076, -.078, -.040, .073),
              "ebfi": (-.114, -.108, .050, .049), "rbbbp": (2.70, 0.38, -0.89, -1.35)}
ZLB_POLICY = ("rfftay", "rffrule", "rff", "dmptpi", "dmptlur", "dmptmax", "dmptr")  # their add factors set to zero


def zlb_setup(f, floor: bool, thresholds: bool) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """example3.prg's settings: the plain Taylor rule, rffmin = 0.125 (or -9999: no floor), the thresholds (lurtrsh =
    7, pitrsh = 3) from the fifth quarter with dmptr zero before, tracking add factors to 2012Q4 with those of the
    policy equations set to zero, and the shocks in place of the add factors of their four quarters."""
    model, start = f["model"], ZLB_START
    frame = policy(f["frame"], monetary="dmptay")
    frame["rffmin"] = 0.125 if floor else -9999.0
    if thresholds:
        frame.loc[start - 1:start + 3, ["dmptrsh", "lurtrsh", "pitrsh", "dmptr"]] = [0.0, -9999.0, 9999.0, 0.0]
        frame.loc[start + 4:, ["dmptrsh", "lurtrsh", "pitrsh"]] = [1.0, 7.0, 3.0]
    factors = model.add_factors(Dataset.from_frame(frame), (str(start), "2012Q4"))
    for name in ZLB_POLICY:
        factors[name] = 0.0 * factors[name]
    first4 = pd.period_range(start, periods=4, freq="Q")
    for name, values in ZLB_SHOCKS.items():
        factors[name] = factors[name].copy()
        factors[name].loc[first4] = 0.0
        frame.loc[first4, f"{name}_aerr"] = values
    return frame, factors


@pytest.fixture(scope="module")
def frbus_zlb(frbus):
    """The scenario of example3.prg solved by Newton three ways: no floor, the floor, the floor and the thresholds."""
    rows = pd.period_range(ZLB_START, periods=ZLB_QUARTERS, freq="Q")
    out = {"rows": rows}
    for key, floor, thresholds in (("free", False, False), ("zlb", True, False), ("thresholds", True, True)):
        frame, factors = zlb_setup(frbus, floor, thresholds)
        solved = frbus["model"].solve(Dataset.from_frame(frame), (str(rows[0]), str(rows[-1])), Scenario(adjust=factors)).frame
        out[key] = (solved, frame, factors)
    return out


@needs_frbus
def test_zlb_floor_binds(frbus, frbus_zlb):
    """The floor holds in every quarter (1e-8) and binds: without it the funds rate goes to -5.2; with it the rate is
    0.125 for 2003Q3-2005Q2 and again from 2008Q4. Every equation holds in every quarter."""
    rows, model = frbus_zlb["rows"], frbus["model"]
    free = frbus_zlb["free"][0].loc[rows]
    assert free.rff.min() < -5  # -5.21 found (2009Q1)
    for key in ("zlb", "thresholds"):
        solved, frame, factors = frbus_zlb[key]
        rff = solved.loc[rows, "rff"]
        assert rff.min() >= 0.125 - 1e-8, key
        binding = rows[np.abs(rff.to_numpy() - 0.125) < 1e-8]
        assert binding[0] == rows[0] and pd.Period("2008Q4", "Q") in binding, key
        assert max_residual(model, solved, factors, rows) < 1e-8, key
    zlb = frbus_zlb["zlb"][0].loc[rows]
    binding = rows[np.abs(zlb.rff.to_numpy() - 0.125) < 1e-8]
    assert [str(p) for p in binding] == [str(p) for p in pd.period_range("2003Q3", "2005Q2", freq="Q")] + ["2008Q4", "2009Q1", "2009Q2"]
    below = zlb.rffrule < 0.125  # where the rule asks for less than the floor, the floor holds; elsewhere the rule
    assert np.allclose(zlb.rff[below], 0.125, atol=1e-8) and np.allclose(zlb.rff[~below], zlb.rffrule[~below], atol=1e-8)


@needs_frbus
def test_zlb_newton_and_gauss_seidel_agree(frbus, frbus_zlb):
    """The kinks (the max of the rule and the floor, the thresholds' max) solved by Gauss-Seidel too (damped: undamped
    it overshoots in these quarters to a negative stock-market wealth): the same solution, 1e-8 relative."""
    rows = frbus_zlb["rows"][:12]
    for key in ("zlb", "thresholds"):
        newton, frame, factors = frbus_zlb[key]
        gauss = frbus["model"].solve(Dataset.from_frame(frame), (str(rows[0]), str(rows[-1])), Scenario(adjust=factors),
                                     solver=GaussSeidel(tol=1e-9, max_iter=20000, damping=0.5)).frame
        worst, where = worst_gap(gauss, newton, frbus["model"].endogenous, rows)
        assert worst < 1e-8, (key, where)


@needs_frbus
def test_zlb_thresholds_switch_when_crossed(frbus, frbus_zlb):
    """Period by period: from the fifth quarter, dmptlur is above 1/2 exactly when unemployment is below 7, dmptpi
    exactly when expected inflation (zpic58) is above 3, dmptr is the max of dmptmax and its own lag, and the funds
    rate stays at the floor until the quarter after dmptr turns on (liftoff in 2005Q4, a quarter later than with the
    floor alone)."""
    rows = frbus_zlb["rows"]
    s = frbus_zlb["thresholds"][0]
    active = rows[4:]
    lur, zpic, dl, dp = (s.loc[active, v].to_numpy() for v in ("lur", "zpic58", "dmptlur", "dmptpi"))
    assert np.array_equal(dl > 0.5, lur < 7.0) and np.array_equal(dp > 0.5, zpic > 3.0)
    assert np.allclose(s.loc[rows, "dmptmax"], np.maximum(s.loc[rows, "dmptlur"], s.loc[rows, "dmptpi"]), atol=1e-10)
    lag = s["dmptr"].shift(1).loc[rows].to_numpy()
    assert np.allclose(s.loc[rows, "dmptr"], np.maximum(s.loc[rows, "dmptmax"], lag), atol=1e-10)
    assert np.allclose(s.loc[rows[:4], "dmptr"], 0.0, atol=1e-12)
    crossed = active[(lur < 7.0) | (zpic > 3.0)][0]
    on = rows[s.loc[rows, "dmptr"].to_numpy() > 0.5][0]
    assert crossed == on == pd.Period("2005Q3", "Q")
    expected = np.maximum(lag * s.loc[rows, "rffrule"] + (1 - lag) * 0.125, 0.125)
    assert np.allclose(s.loc[rows[4:], "rff"], expected[4:], atol=1e-8)
    assert np.allclose(s.loc[rows[:rows.get_loc(on)], "rff"], 0.125, atol=1e-8)  # at the floor until it turns on
    # in that quarter too, but for the logistic's leak: dmptr is small the quarter before (0.003, unemployment 7.23,
    # in the data the test was written with; the size of the leak moves with each release of LONGBASE)
    assert lag[rows.get_loc(on)] < 0.01 and s.loc[on, "rff"] == pytest.approx(0.125, abs=1e-3)
    zlb = frbus_zlb["zlb"][0]
    lift = lambda frame: rows[(frame.loc[rows, "rff"].to_numpy() > 0.13)][0]  # noqa: E731
    assert lift(zlb) == pd.Period("2005Q3", "Q") and lift(s) == pd.Period("2005Q4", "Q")


# ---------------------------------------------------------------------------------------------
# example4.prg: a new policy rule appended, with new model-consistent expectations


PI4 = "(picxfe + picxfe[-1] + picxfe[-2] + picxfe[-3])/4"
NEW_RULE = (f"rffgen ~ g_rff*rff[-1] + (1 - g_rff)*(rstar + {PI4} + g_pi*({PI4} - pitarg) + g_gap*xgap2)"
            " + g_zpi*(zpic43 - pitarg) + g_zgap*(zgap3 - zgap3[-4])")
FED_RULE = {"g_rff": 1.0, "g_pi": 0.0, "g_gap": 0.0, "g_zpi": 0.5, "g_zgap": 0.5}  # example4.prg's first-difference rule
INERTIAL = {"g_rff": 0.85, "g_pi": 0.5, "g_gap": 1.0, "g_zpi": 0.0, "g_zgap": 0.0}  # rffintay's coefficients


def with_new_rule(model: Model) -> Model:
    """example4.prg's changes: the rule rffgen and the expectations zgap3 = xgap2(3) and zpic43 = picx4(3) added, and
    rffrule given the dmpgen * rffgen term it selects with (the model of October 2026 has dropped it)."""
    eqs = [Equation(e.text + " + dmpgen*rffgen", name="rffrule") if e.name == "rffrule" else e for e in model.equations]
    eqs += [Equation(NEW_RULE, params=list(FED_RULE), name="rffgen"),
            Equation("zgap3 = xgap2[+3]", name="zgap3"), Equation("zpic43 = picx4[+3]", name="zpic43")]
    return Model(eqs, params={**model.params, **FED_RULE})


@needs_frbus
def test_new_policy_rule(frbus_mce):
    """example4.prg: a consumption shock (eco up 1 percent for a quarter) under model-consistent expectations with
    the appended first-difference rule. The appended rule holds along the solution, as does every other equation;
    with the new rule's coefficients set to the inertial Taylor rule's, the responses equal those under dmpintay
    (1e-8); the new rule raises the funds rate sooner (0.50 against 0.20 after three quarters)."""
    f = frbus_mce
    model = with_new_rule(f["model"])
    rows, start = f["rows"], f["start"]
    frame = f["frame"].copy()
    frame["rffgen"], frame["zgap3"], frame["zpic43"] = frame.rff, frame.xgap2.shift(-3), frame.picx4.shift(-3)

    def run(m, rule):
        fr = policy(frame, monetary=rule, rstar_from=start + 40)
        fr["dmpgen"] = float(rule == "dmpgen")
        data = Dataset.from_frame(fr)
        factors = m.add_factors(data, (str(rows[0]), str(rows[-1])))
        solved = m.solve(data, (str(rows[0]), str(rows[-1])), Scenario(adjust=factors, shift={"eco_aerr": _first(f, 0.01)}),
                         solver=PF).frame
        return solved, (solved - fr).loc[rows], factors

    new, d_new, factors = run(model, "dmpgen")
    assert max_residual(model, new, factors, rows) < 1e-8
    rule = new.rff.shift(1) + 0.5 * (new.zpic43 - new.pitarg) + 0.5 * (new.zgap3 - new.zgap3.shift(4))
    assert np.allclose(new.rffgen.loc[rows], (rule + factors["rffgen"]).loc[rows], atol=1e-8)
    assert np.allclose(new.rffrule.loc[rows], (new.rffgen + factors["rffrule"]).loc[rows], atol=1e-8)  # its add factor: LONGBASE's rffrule is not rff
    assert np.allclose(new.zgap3.loc[rows], new.xgap2.shift(-3).loc[rows], atol=1e-8)
    same, _, _ = run(model.with_parameters(INERTIAL), "dmpgen")
    old, d_old, _ = run(model, "dmpintay")
    worst, where = worst_gap(same, old, f["model"].endogenous, rows)  # relative to the levels: some are in the thousands
    assert worst < 1e-8, where
    assert 0.4 < d_new.rff.iloc[2] < 0.6 and 0.1 < d_old.rff.iloc[2] < 0.3  # 0.50 and 0.20 found
    assert d_new.xgap2.iloc[0] == pytest.approx(d_old.xgap2.iloc[0], abs=1e-3) and d_new.xgap2.iloc[2] < d_old.xgap2.iloc[2]


# ---------------------------------------------------------------------------------------------
# stochsim.prg and ocpolicy.prg: the machinery checked first on small models, then FRB/US


def _toy_ar(rows: int = 13) -> tuple[Model, Dataset]:
    model = Model([Equation("y = 0.5*y[-1] + 0.2*x", name="y"), Equation("z = exp(y)", name="z")])
    index = pd.period_range("2000Q1", periods=rows, freq="Q", name="time")
    return model, Dataset.from_frame(pd.DataFrame({"y": 0.0, "z": 1.0, "x": 0.0}, index=index))


def _residual_sample(n: int = 300, seed: int = 5) -> pd.DataFrame:
    e = np.random.default_rng(seed).standard_t(5, n)
    return pd.DataFrame({"y": e - e.mean()}, index=pd.period_range("1950Q1", periods=n, freq="Q"))


def test_resampled_residuals_give_the_analytic_variance():
    """y = 0.5 y[-1] + e, e resampled from a residual sample (Student t(5), de-meaned): after h periods from a known
    start, Var y = s2 (1 + 0.25 + ... + 0.25^(h-1)), s2 the sample's variance (the bootstrap's own). 20000 draws: within
    5 percent in every period (2.3 percent at most found); the nonlinear z = exp(y) is solved in every draw."""
    model, data = _toy_ar()
    sample = _residual_sample()
    sol = model.solve(data, ("2000Q2", "2003Q1"), draws=20000, seed=3, errors=sample)
    var = sol.simulation["y"].var(axis=1, ddof=1).to_numpy()
    analytic = sample["y"].var(ddof=0) * np.cumsum(0.25 ** np.arange(12))
    np.testing.assert_allclose(var, analytic, rtol=0.05)
    np.testing.assert_allclose(sol.simulation["z"].to_numpy(), np.exp(sol.simulation["y"].to_numpy()), rtol=1e-9)
    drawn = sol.simulation.error("y").to_numpy()
    assert np.isin(drawn, sample["y"].to_numpy()).all()  # every error drawn is a residual of the sample, exactly
    assert sol.simulation.info["periods resampled"] == 300 and sol.simulation.info["residuals from"] == "1950Q1 to 2024Q4"


def test_resampled_residuals_seed_scale_zero_and_workers():
    """Reproducible by seed and generator; error_scale multiplies a period's errors; a sample of zeros gives the
    deterministic solution in every draw; workers solve the same draws in other processes."""
    from pymetrix import random_generator
    from pymetrix.errors import SpecificationError

    model, data = _toy_ar()
    sample = _residual_sample()
    span = ("2000Q2", "2001Q4")

    def run(**kw):
        return model.solve(data, span, **{"draws": 50, "seed": 7, "errors": sample, **kw}).simulation

    a, b, c = run(), run(), run(seed=8)
    assert np.array_equal(a.values, b.values) and not np.array_equal(a.values, c.values)
    with random_generator("mt19937"):
        m1, m2 = run(), run()
    assert np.array_equal(m1.values, m2.values) and not np.array_equal(m1.values, a.values)
    half = run(error_scale={"2000Q2": 0.5})
    np.testing.assert_allclose(half.errors[:, 0], 0.5 * a.errors[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(half.errors[:, 1:], a.errors[:, 1:], rtol=0, atol=0)
    assert half.info["error_scale"] == {"2000Q2": 0.5}
    zero = run(errors=0.0 * sample)
    base = model.solve(data, span).frame.loc["2000Q2":"2001Q4"]
    assert np.max(np.abs(zero.values - base[list(zero.names)].to_numpy()[None])) < 1e-12
    parallel = run(workers=2)
    assert parallel.info["workers"] == 2 and np.array_equal(parallel.errors, a.errors)
    np.testing.assert_allclose(parallel.values, a.values, rtol=0, atol=1e-12)
    for kwargs, message in (({"errors": sample.rename(columns={"y": "w"})}, "not equations solved here"),
                            ({"errors": sample, "shock_sd": {"y": 1.0}}, "scale its columns"),
                            ({"errors": sample.iloc[:1]}, "at least 2 periods"),
                            ({"errors": sample, "error_scale": {"1999Q1": 0.5}}, "not in the simulation span")):
        with pytest.raises(SpecificationError, match=message):
            model.solve(data, span, draws=10, **kwargs)
    with pytest.raises(SpecificationError, match="for the stochastic simulation"):
        model.solve(data, span, errors=sample)


def _toy_policy_model() -> Model:
    from pymetrix.metadata import Role

    return Model([Equation("y ~ a*y[-1] - s*(i[-1] - pi[-1]) + ey", params=["a", "s"], name="is"),
                  Equation("pi ~ pi[-1] + k*y + epi", params=["k"], name="pc"),
                  Equation("i ~ phi*pi + i_aerr", params=["phi"], name="rule"),
                  Equation("di = i - i[-1]", name="di")],
                 params={"a": 0.8, "s": 0.3, "k": 0.2, "phi": 1.5}, roles={"ey": Role.SHOCK, "epi": Role.SHOCK})


def test_optimal_control_equals_commitment_on_a_linear_quadratic_model():
    """ocpolicy.prg's method checked where the answer is known: a small backward-looking model (an IS curve, a
    Phillips curve, a rule and the rate's change) with the loss y^2 + pi^2 + di^2, discounted at 0.99. The funds-rate
    path chosen by optimal_control (Gauss-Newton on the path of the rule's add factor, over 200 quarters) after an
    inflation shock is pymetrix's analytic optimal policy under commitment (optimal_policy, its impulse response):
    1e-6 over the first 40 quarters (5e-15 found). The gradient vanishes at the optimum, and the loss is below the
    rule's."""
    from pymetrix import optimal_control, optimal_policy

    model = _toy_policy_model()
    loss = {"y": 1.0, "pi": 1.0, "di": 1.0}
    analytic = optimal_policy(model, "i", loss, kind="commitment", beta=0.99).irf(60, {"epi": 1.0})
    index = pd.period_range("1999Q4", periods=201, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame(0.0, index=index, columns=["y", "pi", "i", "di", "ey", "epi", "i_aerr"]))
    oc = optimal_control(model, data, ("2000Q1", str(index[-1])), ["i_aerr"], loss, discount=0.99,
                         scenario=Scenario(shift={"epi": {"2000Q1": 1.0}}))
    path = oc.solution.frame.loc["2000Q1":].iloc[:41]
    for v in ("i", "pi", "y", "di"):
        np.testing.assert_allclose(path[v].to_numpy(), analytic[v, "epi"].to_numpy()[:41], rtol=0, atol=1e-6)
    assert oc.iterations == 1 and oc.gradient_norm < 1e-8 and oc.loss < oc.baseline_loss
    rule = oc.baseline.frame.loc["2000Q1":]
    assert np.allclose(rule.i, 1.5 * rule.pi)  # the baseline: the rule, with no moves
    assert np.allclose(oc.solution.frame.loc["2000Q1":, "i"] - 1.5 * oc.solution.frame.loc["2000Q1":, "pi"],
                       oc.solution.frame.loc["2000Q1":, "i_aerr"])


def test_optimal_control_refusals():
    from pymetrix import optimal_control
    from pymetrix.errors import SpecificationError

    model = _toy_policy_model()
    index = pd.period_range("1999Q4", periods=21, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame(0.0, index=index, columns=["y", "pi", "i", "di", "ey", "epi", "i_aerr"]))
    span = ("2000Q1", "2004Q4")
    for kwargs, message in (({"instruments": ["i"]}, "exogenous series"), ({"loss": {"ey": 1.0}}, "not endogenous"),
                            ({"loss": {"y": -1.0}}, "0 or more"), ({"desired": {"pi": 2.0}, "loss": {"y": 1.0}}, "not in the loss"),
                            ({"control": ("2005Q1", "2005Q4")}, "must lie in the span"), ({"discount": 1.5}, "discount")):
        args = {"instruments": ["i_aerr"], "loss": {"y": 1.0, "pi": 1.0}, **kwargs}
        with pytest.raises(SpecificationError, match=message):
            optimal_control(model, data, span, **args)
    leads = Model([Equation("x = 0.5*x[+1] + u", name="x")])
    with pytest.raises(SpecificationError, match="backward-looking"):
        optimal_control(leads, data, span, ["u"], {"x": 1.0})


# with the Fed's packages: stochsim.prg


STOCH_START_RESIDUALS = ("1970Q1", "2014Q4")  # stochsim.prg's %residstart and %residend
STOCH_TOL = Newton(tol=1e-8)  # FRB/US's levels are in the tens of thousands: 1e-10 is below rounding


@pytest.fixture(scope="module")
def frbus_stochsim(frbus):
    """stochsim.prg's settings: the inertial Taylor rule (half of it in the first quarter, half the funds rate held),
    exogenous fiscal policy (dfpex), the floor at 0.25, a 4-quarter GDP growth equation appended, and the de-meaned
    historical errors (1970Q1-2014Q4) of the stochastic equations (stochastic type not NO, leo left out)."""
    read = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml")
    model = Model([*read.model.equations, Equation("anngr = 100*(xgdp/xgdp[-4] - 1)", name="anngr")], params=read.model.params)
    start = frbus["start"]
    frame = policy(frbus["frame"], fiscal="dfpex")
    frame["anngr"] = 100 * (frame.xgdp / frame.xgdp.shift(4) - 1)
    frame["rffmin"] = 0.25
    frame.loc[start, ["dmpintay", "dmpex"]] = 0.5
    data = Dataset.from_frame(frame)
    stochastic = [v for v, kind in read.stochastic.items() if kind != "NO" and v != "leo"]
    history = model.add_factors(data, STOCH_START_RESIDUALS)
    errors = pd.DataFrame({v: history[v] for v in stochastic})
    errors = errors - errors.mean()
    span = (str(start), str(start + 11))
    factors = model.add_factors(data, span)
    rows = pd.period_range(start, periods=12, freq="Q")
    sims = model.solve(data, span, Scenario(adjust=factors), solver=STOCH_TOL, draws=200, seed=12345, errors=errors,
                       error_scale={str(start): 0.5})
    return {"model": model, "frame": frame, "data": data, "errors": errors, "factors": factors, "span": span, "rows": rows,
            "sims": sims, "stochastic": stochastic, "start": start}


def _stoch(f, **kw):
    return f["model"].solve(f["data"], f["span"], Scenario(adjust=f["factors"]), solver=STOCH_TOL,
                            **{"draws": 20, "seed": 12345, "errors": f["errors"], "error_scale": {str(f["start"]): 0.5}, **kw})


@needs_frbus
def test_stochsim_stochastic_equations(frbus_stochsim):
    """The model file's stochastic types: 51 equations shocked, their 180 quarters of errors complete and de-meaned."""
    f = frbus_stochsim
    assert len(f["stochastic"]) == 51 and {"eco", "ebfi", "picxfe", "rg10p", "lurnat"} <= set(f["stochastic"])
    assert f["errors"].shape == (180, 51) and np.isfinite(f["errors"].to_numpy()).all()
    assert float(f["errors"].mean().abs().max()) < 1e-12


@needs_frbus
def test_stochsim_zero_errors_is_the_baseline(frbus_stochsim):
    """With every error zero, every draw is the deterministic baseline, which is LONGBASE (1e-8 relative)."""
    f = frbus_stochsim
    sims = _stoch(f, draws=4, errors=0.0 * f["errors"]).simulation
    names = [v for v in f["model"].endogenous if v in f["frame"]]
    base = f["frame"].loc[f["rows"], names].to_numpy(float)
    got = sims.values[:, :, [sims.names.index(v) for v in names]]
    assert np.max(np.abs(got - base[None]) / np.maximum(1.0, np.abs(base[None]))) < 1e-8


def _same_paths(a, b) -> bool:
    """The same paths, to the last digits rounding leaves: OpenBLAS's kernels can round differently from one call to
    the next (5e-11 relative seen with NumPy 1.26), so solutions are compared to 1e-9 relative, the draws exactly."""
    return bool(np.all(np.abs(a.values - b.values) <= 1e-9 * np.maximum(1.0, np.abs(a.values))))


@needs_frbus
def test_stochsim_reproducible_by_seed_and_generator(frbus_stochsim):
    """The same seed and generator give the same draws (exactly) and paths; another seed, or the Mersenne Twister,
    others."""
    from pymetrix import random_generator

    f = frbus_stochsim
    a, b, c = (_stoch(f, seed=s).simulation for s in (12345, 12345, 54321))
    assert np.array_equal(a.errors, b.errors) and _same_paths(a, b)
    assert not np.array_equal(a.errors, c.errors) and not _same_paths(a, c)
    with random_generator("mt19937"):
        m1, m2 = _stoch(f).simulation, _stoch(f).simulation
    assert np.array_equal(m1.errors, m2.errors) and _same_paths(m1, m2)
    assert not np.array_equal(m1.errors, a.errors) and not _same_paths(m1, a)


@needs_frbus
def test_stochsim_draws_are_historical_quarters(frbus_stochsim):
    """Each draw's errors in each quarter are one historical quarter's errors, all 51 together (the first quarter's
    halved); and the add factors that a draw's solution implies are the tracking add factors plus those errors."""
    f = frbus_stochsim
    sims = f["sims"].simulation
    assert sims.error_names == tuple(f["stochastic"]) and sims.info["failed"] == 0
    history = f["errors"].to_numpy()
    scale = np.ones(len(f["rows"]))
    scale[0] = 0.5
    for d in range(0, 200, 7):
        for t in range(len(f["rows"])):
            row = sims.errors[d, t] / scale[t]  # halving and doubling are exact
            assert np.any(np.all(row[None] == history, axis=1)), (d, t)
    for d in (0, 101):
        path = f["frame"].copy()
        path.loc[f["rows"], list(sims.names)] = sims.values[d]
        implied = f["model"].add_factors(Dataset.from_frame(path), f["span"])
        for j, name in enumerate(f["stochastic"]):
            np.testing.assert_allclose((implied[name] - f["factors"][name]).loc[f["rows"]].to_numpy(), sims.errors[d, :, j],
                                       rtol=0, atol=1e-6)


@needs_frbus
def test_stochsim_distribution(frbus_stochsim):
    """The distribution's shape: bands widen with the horizon, the median stays near the baseline (shocks
    de-meaned: within a third of the 90 percent band's half-width), and the floor binds in some draws or none (a
    probability). The Fed's documentation gives no numbers for this program: sanity checks only."""
    f = frbus_stochsim
    sol = f["sims"]
    bands = sol.bands((0.7, 0.9), ["rff", "lur", "picx4", "xgap2"])
    base = f["frame"].loc[f["rows"]]
    for v in ("rff", "lur", "picx4", "xgap2"):
        width = bands[f"{v}_hi90"] - bands[f"{v}_lo90"]
        assert width.iloc[-1] > 2 * width.iloc[0] > 0, v
        assert np.all(bands[f"{v}_lo90"] <= bands[f"{v}_lo70"]) and np.all(bands[f"{v}_hi70"] <= bands[f"{v}_hi90"])
        assert float((bands[v] - base[v]).abs().max()) < float((width / 6).max()), v
    floor = sol.simulation["rffrule"] < 0.25
    assert 0.0 <= float(floor.any().mean()) < 0.5
    assert float(np.nanmin(sol.simulation["rff"].to_numpy() - f["factors"]["rff"].loc[f["rows"]].to_numpy()[:, None])) >= 0.25 - 1e-8


# with the Fed's packages: ocpolicy.prg


@pytest.fixture(scope="module")
def frbus_ocpolicy(frbus):
    """ocpolicy.prg's settings under VAR expectations: the plain Taylor rule, the equilibrium real rate endogenous,
    exogenous fiscal policy for 20 quarters and the debt rule after, a ugap equation appended; the funds-rate path
    (rfftay_aerr) chosen for 60 quarters to minimise the discounted (0.99) sum over 80 quarters of ugap^2,
    (pic4 - 2)^2 and delrff^2."""
    from pymetrix import optimal_control

    read = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml")
    model = Model([*read.model.equations, Equation("ugap = lur - lurnat", name="ugap")], params=read.model.params)
    start = frbus["start"]
    frame = policy(frbus["frame"], monetary="dmptay", fiscal="dfpex", rstar_from=start)
    frame.loc[start + 20:, ["dfpex", "dfpdbt"]] = [0.0, 1.0]
    frame["ugap"] = frame.lur - frame.lurnat
    data = Dataset.from_frame(frame)
    span = (str(start), str(start + 79))
    factors = model.add_factors(data, span)
    loss = {"ugap": 1.0, "pic4": 1.0, "delrff": 1.0}
    oc = optimal_control(model, data, span, ["rfftay_aerr"], loss, desired={"pic4": 2.0}, control=(str(start), str(start + 59)),
                         discount=0.99, scenario=Scenario(adjust=factors), solver=STOCH_TOL)
    return {"oc": oc, "model": model, "data": data, "span": span, "factors": factors, "frame": frame, "loss": loss, "start": start}


@needs_frbus
def test_ocpolicy_first_order_conditions(frbus_ocpolicy):
    """At the optimum the gradient of the loss in the 60 moves is 2e-7 of its size at the baseline (1e-5 asked); the
    gradient from the linearised model equals central differences of the loss at the baseline (1e-4 relative; 1e-6
    found); the loss is below the Taylor rule's. Its size depends on the release of LONGBASE (10.81 against 11.76 with
    the data the test was written with, 11.62 against 14.51 with the release downloaded on 27 November 2026), so it is
    not fixed."""
    from pymetrix.models.control import _Problem

    f = frbus_ocpolicy
    oc = f["oc"]
    problem = _Problem(f["model"], f["data"], f["span"], Scenario(adjust=f["factors"]), STOCH_TOL, ["rfftay_aerr"],
                       oc.moves.index, f["loss"], {"pic4": 2.0}, 0.99)
    zero = np.zeros(len(oc.moves))
    at_base = problem.solve(zero)
    r0 = problem.residuals(at_base)
    gradient = 2 * problem.sensitivities(at_base).T @ r0
    assert float(r0 @ r0) == pytest.approx(oc.baseline_loss, rel=1e-12)
    assert oc.gradient_norm < 1e-5 * np.linalg.norm(gradient)
    for j in (0, 8, 30):
        step = np.zeros_like(zero)
        step[j] = 1e-3
        up, down = problem.residuals(problem.solve(step)), problem.residuals(problem.solve(-step))
        assert (up @ up - down @ down) / 2e-3 == pytest.approx(gradient[j], rel=1e-4), j
    assert oc.loss < oc.baseline_loss
    base = oc.baseline.frame.loc[f["start"]:f["start"] + 79]
    assert np.allclose(base.rff, f["frame"].rff.loc[base.index], atol=1e-6)  # the baseline: LONGBASE, tracked


@needs_frbus
def test_ocpolicy_path(frbus_ocpolicy):
    """The optimal path against the Taylor rule's baseline: the moves only in the 60 control quarters, the funds rate
    away from the rule's, and the equations satisfied along it. Which way it moves in the first year depends on where
    the baseline starts, so on the release of LONGBASE (below the rule's with the data the test was written with,
    above it with the release downloaded on 27 November 2026), so the direction is not checked."""
    f = frbus_ocpolicy
    oc = f["oc"]
    s, start = oc.solution.frame, f["start"]
    after = s.loc[start + 60:start + 79, "rfftay_aerr"] - f["frame"].rfftay_aerr.loc[start + 60:start + 79]
    assert float(after.abs().max()) == 0.0 and float(oc.moves.abs().max().max()) > 0.05
    assert float((s.rff - oc.baseline.frame.rff).loc[start:start + 3].abs().max()) > 1e-3
    assert max_residual(f["model"], s, f["factors"], pd.period_range(start, periods=80, freq="Q")) < 1e-6
