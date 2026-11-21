"""Historical decompositions of panel VARs (entity by entity) and proxy SVARs (the identified shock, the other shocks
together, the baseline).

Checks: the parts add up to the data to 1e-10 for every entity and every draw; a one-entity panel by within least
squares decomposes exactly as the ordinary VAR with a constant does (the fixed effect is the constant); with the true
shock as the proxy, in a simulated SVAR, the identified shock's contribution is close to the true shock's; the bands
are ordered and hold the point decomposition."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset
from pymetrix.commands import FitVAR, HistoricalDecomposition, Session
from pymetrix.commands.builtin import ProxySVARCommand
from pymetrix.errors import NotFoundError, SpecificationError
from pymetrix.historical_extra import PanelDecomposition, _panel_parts, _proxy_parts, propagate
from pymetrix.proxy import bootstrap_draws

A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
B0 = np.array([[1.0, 0.0], [0.5, 0.8]])


def _panel(entities=6, T=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(entities):
        mu = rng.normal(size=2)
        y = np.zeros(2)
        for t in range(T + 20):
            y = mu + A1 @ y + B0 @ rng.normal(size=2)
            if t >= 20:
                rows.append({"firm": f"f{i}", "time": pd.Period("1990Q1", "Q") + (t - 20), "a": y[0], "b": y[1]})
    return Dataset.from_frame(pd.DataFrame(rows), time="time", entity="firm")


@pytest.fixture(scope="module")
def panel():
    return _panel()


def _adds_up(d, data_frame):
    total = d.baseline + pd.DataFrame({v: f.sum(axis=1) for v, f in d.contributions.items()})
    assert np.abs(total.to_numpy() - data_frame.to_numpy()).max() < 1e-10


@pytest.mark.parametrize("transform", ["fod", "within"])
def test_panel_parts_add_up_for_every_entity_and_draw(panel, transform):
    fit = VAR(["a", "b"], lags=2, by="firm").fit(panel, transform=transform)
    hd = fit.historical_decomposition()
    assert isinstance(hd, PanelDecomposition) and hd.entities == [f"f{i}" for i in range(6)]
    for key in hd.entities:
        rows = fit.levels.xs(key, level="firm")[["a", "b"]]
        _adds_up(hd[key], rows)
        assert hd[key].residual.abs().to_numpy().max() < 1e-10
    rng = np.random.default_rng(1)
    for _ in range(5):  # any coefficients and impact matrix: the parts still add up
        B = fit.B + 0.05 * rng.normal(size=fit.B.shape)
        P = np.linalg.cholesky(fit.sigma) @ np.linalg.qr(rng.normal(size=(2, 2)))[0]
        for key, (c, base, _, _) in _panel_parts(fit, B, P).items():
            Y = fit.levels.xs(key, level="firm")[["a", "b"]].to_numpy()
            assert np.abs(c.sum(axis=2) + base - Y).max() < 1e-10


def test_one_entity_within_is_the_var_with_a_constant():
    panel = _panel(entities=1, T=80, seed=3)
    pv = VAR(["a", "b"], lags=2, by="firm").fit(panel, transform="within")
    series = Dataset.from_frame(panel.frame.droplevel("firm"))
    var = VAR(["a", "b"], lags=2).fit(series)
    assert np.allclose(pv.coefs, var.coefs, atol=1e-10)
    got, want = pv.historical_decomposition()["f0"], var.historical_decomposition()
    assert list(got.deviations.index) == list(want.deviations.index)
    for v in ("a", "b"):  # contributions do not depend on how sigma is scaled (T or T - k)
        assert np.abs(got.contributions[v].to_numpy() - want.contributions[v].to_numpy()).max() < 1e-9
    assert np.abs(got.baseline.to_numpy() - want.baseline.to_numpy()).max() < 1e-9
    const = var.estimate.B[0]  # the fixed effect is the VAR's constant
    assert np.allclose(pv.historical_decomposition().fixed_effects.loc["f0"].to_numpy(), const, atol=1e-10)


def test_panel_identification_bands_and_selection(panel):
    from pymetrix import Cholesky

    fit = VAR(["a", "b"], lags=1, by="firm").fit(panel).identify(Cholesky(["b", "a"]))
    hd = fit.historical_decomposition(draws=60, alpha=0.1, seed=2, entity="f3")
    assert hd.entity == "f3"
    d = hd["f3"]
    assert d.has_bands and hd.has_bands and d.coverage == pytest.approx(0.9)
    for v in ("a", "b"):
        lo, med, up, pt = (x[v].to_numpy() for x in (d.lower, d.median, d.upper, d.contributions))
        assert np.all(lo <= med + 1e-12) and np.all(med <= up + 1e-12)
        assert np.mean((pt >= lo - 1e-12) & (pt <= up + 1e-12)) > 0.9
    # the impact under the ordering b, a: shock b moves a on impact, shock a does not move b
    P = fit.impact()
    assert abs(P[1, 0]) < 1e-14 and abs(P[0, 1]) > 0
    assert hd.select("f1").entity == "f1" and hd.select("f1").selected is hd["f1"]
    with pytest.raises(NotFoundError):
        hd.select("nobody")
    from pymetrix.views import titles

    names = titles(hd)
    assert names[0] == "Entities" and "Table: a" in names and "Bands: a" in names
    assert "firm f3" in hd.table("a").title and "firm f1" in hd.select("f1").table("a").title
    long = hd.to_frame()
    assert set(long.columns) == {"firm", "variable", "period", "part", "value"} and len(long) == 6 * 2 * 2 * len(d.deviations)
    same = fit.historical_decomposition(draws=60, alpha=0.1, seed=2)
    assert np.array_equal(same["f0"].upper["a"].to_numpy(), hd["f0"].upper["a"].to_numpy())  # seeded


def _proxy_dgp(T=600, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(T, 2))
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.3 + A1 @ y[t - 1] + B0 @ eps[t]
    frame = pd.DataFrame({"r": y[:, 0], "x": y[:, 1], "z": eps[:, 0] + noise * rng.normal(size=T)},
                         index=pd.period_range("1960-01", periods=T, freq="M", name="time"))
    return Dataset.from_frame(frame), eps


def test_proxy_parts_add_up_and_match_the_true_shock():
    data, eps = _proxy_dgp(T=2000)
    var = VAR(["r", "x"], lags=1).fit(data)
    ps = var.proxy_svar(data, "z", draws=0)
    hd = ps.historical_decomposition()
    shock = "r shock (z)"
    assert list(hd.contributions["r"].columns) == [shock, "other shocks"]
    _adds_up(hd, pd.DataFrame(var.arrays.Y, index=var.arrays.index, columns=["r", "x"]))
    # the true contribution of shock 1: B0[:, 0] eps_1(t) through the true VAR, from the start of the sample; the
    # shock's sample mean is demeaned, as the estimated constant takes it into the baseline
    e1 = eps[1:, 0] - eps[1:, 0].mean()
    true = propagate(A1[None], (B0[:, 0][None, :] * e1[:, None])[:, :, None])[:, :, 0]
    est = np.column_stack([hd.contributions[v][shock].to_numpy() for v in ("r", "x")])
    for j in range(2):
        assert np.corrcoef(est[:, j], true[:, j])[0, 1] > 0.99
        assert np.sqrt(np.mean((est[:, j] - true[:, j]) ** 2)) < 0.1 * true[:, j].std()
    assert np.corrcoef(hd.shocks[shock].to_numpy(), eps[1:, 0])[0, 1] > 0.995
    unit = var.proxy_svar(data, "z", draws=0, scale="unit").historical_decomposition()  # the scale does not matter
    assert np.allclose(unit.contributions["x"].to_numpy(), hd.contributions["x"].to_numpy(), atol=1e-10)


def test_proxy_bands_from_the_block_bootstrap():
    data, _ = _proxy_dgp(T=300, seed=5, noise=1.0)
    var = VAR(["r", "x"], lags=1).fit(data)
    ps = var.proxy_svar(data, "z", draws=40, seed=7)
    hd = ps.historical_decomposition(draws=40, alpha=0.1, seed=7)
    Y = var.arrays.Y
    for A, Bd, sigma, b in bootstrap_draws(var, ps.z, 0, ps.scale, 40, ps.block, 7):  # every draw adds up
        c, base, _ = _proxy_parts(var, A, Bd, sigma, b)
        assert np.abs(c.sum(axis=2) + base - Y).max() < 1e-10
    assert "moving-block bootstrap" in hd.band_source
    for v in ("r", "x"):
        lo, med, up, pt = (x[v].to_numpy() for x in (hd.lower, hd.median, hd.upper, hd.contributions))
        assert np.all(lo <= med + 1e-12) and np.all(med <= up + 1e-12)
        assert np.mean((pt >= lo - 1e-12) & (pt <= up + 1e-12)) > 0.9
    with pytest.raises(SpecificationError, match="alpha"):
        ps.historical_decomposition(draws=5, alpha=2)


def test_the_dialog_for_a_panel_var(panel):
    session = Session()
    session.objects["panel"] = panel
    session.objects["pvar"] = VAR(["a", "b"], lags=1, by="firm").fit(panel)
    cmd = HistoricalDecomposition("pvar", ordering="b, a", draws=30, entity="f2")
    assert not cmd.validate(session)
    session.run(cmd)
    direct = session["pvar"].identify(__import__("pymetrix").Cholesky(["b", "a"])).historical_decomposition(draws=30, entity="f2")
    got = session["hd"]
    assert got.entity == "f2"
    assert np.array_equal(got["f2"].contributions["a"].to_numpy(), direct["f2"].contributions["a"].to_numpy())
    assert np.array_equal(got["f2"].upper["a"].to_numpy(), direct["f2"].upper["a"].to_numpy())
    script = session.script()
    assert "historical_decomposition(draws=30, entity='f2')" in script
    ns = {"panel": panel, "pvar": session["pvar"]}
    exec(script, ns)  # noqa: S102
    assert np.array_equal(ns["hd"]["f2"].upper["b"].to_numpy(), got["f2"].upper["b"].to_numpy())


def test_the_dialog_for_a_proxy_svar():
    data, _ = _proxy_dgp(T=200, seed=2, noise=0.5)
    session = Session()
    session.objects["data"] = data
    session.run(FitVAR("r, x", lags="1", name="var"))
    session.run(ProxySVARCommand("z", draws=0, name="ps"))
    assert HistoricalDecomposition("ps", ordering="x, r").validate(session)  # the proxy identifies the shock
    assert HistoricalDecomposition("ps", entity="f1").validate(session)  # entities are for panel VARs
    session.run(HistoricalDecomposition("ps", draws=20))
    direct = session["ps"].historical_decomposition(draws=20)
    assert np.array_equal(session["hd"].upper["x"].to_numpy(), direct.upper["x"].to_numpy())
    script = session.script()
    assert "ps.historical_decomposition(draws=20)" in script
    ns = {"data": session["data"]}
    exec(script, ns)  # noqa: S102
    assert np.array_equal(ns["hd"].upper["x"].to_numpy(), session["hd"].upper["x"].to_numpy())
