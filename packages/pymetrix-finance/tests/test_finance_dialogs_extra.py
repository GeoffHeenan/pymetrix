"""The finance dialogs for Black-Litterman, bond measures, a Heston option price, Merton at a point and a copula
portfolio's VaR: each equals the call it stands for, records the call that typing its code records, and the session's
script replays to the same result."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, black_litterman, bond_measures, heston_price, merton_point
from pymetrix.commands import Session
from pymetrix.commands.registry import COMMANDS


def _cmd(dialog, /, *args, **kw):
    return COMMANDS[dialog].cls(*args, **kw)


def _start():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2010-01", periods=150, freq="M")
    s = Session()
    s.objects["data"] = Dataset.from_frame(pd.DataFrame(rng.normal(0.005, 0.04, size=(150, 3)), index=idx, columns=["a", "b", "c"]))
    return s


def _replay(s, name):
    ns = dict(s.objects)
    exec("from pymetrix_finance import *\n" + s.script(), ns)  # the plugin's functions, as its scripts import them
    return ns[name]


def _typed_again(s, start=_start):
    record = s.history[-1]
    assert record.ok, record.error
    assert record.call is not None and record.code == record.call.to_code()
    other = start()
    other.run_code("from pymetrix_finance import *")  # the plugin's functions, as its scripts import them
    for step in s.history[:-1]:
        if step.code not in other.script():
            other.run_code(step.code)
    other.run_code(record.code)
    typed = other.history[-1]
    assert typed.ok, typed.error
    assert type(typed.command).__name__ == "Call" and typed.call.same_as(record.call)


def test_black_litterman():
    s = _start()
    s.run(_cmd("BlackLitterman", "a, b, c", "a - b = 0.002, 0.5 c + 0.5 a = 0.006", market_weights="0.5, 0.3, 0.2",
               confidences="0.0001, 0.0002", tau=0.1, delta=3.0, name="bl"))
    cov = s["data"].frame[["a", "b", "c"]].cov().to_numpy()
    mu, sigma = black_litterman(cov, [0.5, 0.3, 0.2], [[1, -1, 0], [0.5, 0, 0.5]], [0.002, 0.006], omega=np.diag([0.0001, 0.0002]),
                                delta=3.0, tau=0.1)
    np.testing.assert_allclose(s["bl"][0], mu, rtol=1e-12)
    np.testing.assert_allclose(s["bl"][1], sigma, rtol=1e-12)
    assert s.history[-1].code.startswith("bl = black_litterman(data.cov(['a', 'b', 'c']), [0.5, 0.3, 0.2], views=[[1.0, -1.0, 0.0], [0.5, 0.0, 0.5]]")
    _typed_again(s)
    np.testing.assert_allclose(_replay(s, "bl")[0], mu, rtol=1e-12)
    # equal market weights and the default omega when left empty; views must name the assets
    s.run(_cmd("BlackLitterman", "a, b", "b = 0.01", name="bl2"))
    np.testing.assert_allclose(s["bl2"][0], black_litterman(s["data"].frame[["a", "b"]].cov().to_numpy(), [0.5, 0.5], [[0, 1]], [0.01])[0])
    assert "not one of the assets" in _cmd("BlackLitterman", "a, b", "z = 0.01").validate(s)[0]
    assert _cmd("BlackLitterman", "a, b", "a = 0.01", confidences="0.1, 0.2").validate(s)


def test_bond_measures():
    s = _start()
    s.run(_cmd("BondMeasures", coupon=4.0, years=2.0, freq=2, zeros="0.03, 0.032, 0.034, 0.035", name="bond"))
    by_api = bond_measures(np.array([0.5, 1.0, 1.5, 2.0]), np.array([2.0, 2.0, 2.0, 102.0]), np.array([0.03, 0.032, 0.034, 0.035]))
    assert s["bond"] == pytest.approx(by_api, rel=1e-12)
    _typed_again(s)
    assert _replay(s, "bond") == pytest.approx(by_api, rel=1e-12)
    s.run(_cmd("BondMeasures", coupon=0.0, years=5.0, freq=1, zeros="0.04", name="zero"))
    assert s["zero"]["duration"] == pytest.approx(5.0) and s["zero"]["price"] == pytest.approx(100 * np.exp(-0.2))
    assert _cmd("BondMeasures", zeros="0.03, 0.04").validate(s)  # neither one rate nor one for each payment


def test_heston_price_and_merton_at_a_point():
    s = _start()
    s.run(_cmd("HestonOptionPrice", spot=100, strike=95, expiry=0.5, rate=0.02, dividend=0.01, v0=0.05, kappa=1.5, theta=0.04,
               sigma=0.4, rho=-0.6, kind="put", name="h"))
    by_api = heston_price(100, 95, 0.5, 0.02, 0.01, 0.05, 1.5, 0.04, 0.4, -0.6, kind="put")
    np.testing.assert_allclose(s["h"], by_api, rtol=1e-12)
    _typed_again(s)
    np.testing.assert_allclose(_replay(s, "h"), by_api, rtol=1e-12)
    s.run(_cmd("MertonPoint", equity=40.0, equity_vol=0.5, debt=70.0, rate=0.02, horizon=2.0, name="firm"))
    assert s["firm"] == pytest.approx(merton_point(40.0, 0.5, 70.0, 0.02, 2.0), rel=1e-12)
    _typed_again(s)
    assert _replay(s, "firm") == pytest.approx(s["firm"], rel=1e-12)


def test_copula_portfolio_var():
    s = _start()
    s.run_code("from pymetrix_finance import *")
    s.run_code("cop = copula(data, ['a', 'b'], family='clayton')")
    s.run(_cmd("CopulaPortfolioVaR", "cop", "0.7, 0.3", level=0.95, margins="empirical", draws=20_000, seed=3, name="pvar"))
    by_api = s["cop"].portfolio_var([0.7, 0.3], level=0.95, n=20_000, margins="empirical", seed=3)
    assert s["pvar"] == pytest.approx(by_api, rel=1e-12)
    _typed_again(s)
    assert _replay(s, "pvar") == pytest.approx(by_api, rel=1e-12)
    assert _cmd("CopulaPortfolioVaR", "cop", "1.0").validate(s)
    assert _cmd("CopulaPortfolioVaR", "data", "0.5, 0.5").validate(s)


def test_the_results_windows_and_forms():
    from pymetrix.views import view_sections

    s = _start()
    s.run(_cmd("MertonPoint"))
    s.run(_cmd("BlackLitterman", "a, b", "a = 0.01"))
    assert [t for t, _ in view_sections(s["merton_firm"])] == ["Values"]
    assert [t for t, _ in view_sections(s["black_litterman_result"])] == ["Summary", "Views", "Graph", "Posterior covariance", "View weights",
                                                                          "Provenance"]
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    for name in ("BlackLitterman", "BondMeasures", "HestonOptionPrice", "MertonPoint", "CopulaPortfolioVaR"):
        assert command_dialog(COMMANDS[name].cls, s) is not None
    assert command_dialog(COMMANDS["CopulaPortfolioVaR"].cls, s).on_advanced_tab("seed")


def test_a_copula_s_next_step_is_its_portfolio_var():
    from pymetrix.ui.next_steps import prefilled, steps_for

    s = _start()
    s.run(COMMANDS["FitCopula"].cls("a, b", family="gaussian", name="cop"))
    (label, cls, field, workspace), = steps_for(s["cop"])
    assert (label, cls.__name__, workspace) == ("Portfolio VaR…", "CopulaPortfolioVaR", "finance")
    assert prefilled(cls, **{field: "cop"}).copula == "cop"


def test_the_finance_menu():
    from pymetrix.commands.menus import menu_entries

    def flat(entries, path=""):  # the items in order, each with its submenu: "Copulas/Copula…"
        for e in entries:
            if e.kind == "command":
                yield f"{path}{e.label}"
            elif e.kind == "submenu":
                yield from flat(e.entries, f"{path}{e.label}/")

    labels = list(flat(menu_entries("Finance", list(COMMANDS.values()))))
    assert labels[:2] == ["Returns…", "Performance…"]
    for before, after in (("Copulas/Copula…", "Copulas/Compare copula families…"), ("Portfolios/Portfolio…", "Portfolios/Black-Litterman…"),
                          ("Options/Option price…", "Options/Heston option price…"),
                          ("Fixed income/Dynamic Nelson-Siegel…", "Fixed income/Bond measures…"),
                          ("Credit/Merton distance to default…", "Credit/Merton at a point…")):
        assert labels.index(after) == labels.index(before) + 1
    assert "Copulas/Copula portfolio VaR…" in labels and "Risk/Value at risk…" in labels
