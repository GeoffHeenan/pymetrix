"""Shift-share IV: 2SLS with a Bartik instrument, against pymetrix's IV; Borusyak-Hull-Jaravel's shock-level IV equal to
the regional one exactly; AKM and AKM0 against the formulas coded with loops on a tiny example; Rotemberg weights that
sum to one and reproduce the estimate."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from pymetrix import IV, Dataset, Equation, SpecificationError, shift_share


def _regions(n=300, K=15, seed=0, weights=False):
    rng = np.random.default_rng(seed)
    S = rng.dirichlet(np.full(K, 0.5), size=n) * rng.uniform(0.7, 1.0, size=(n, 1))  # incomplete shares
    g = rng.normal(size=K)
    u = rng.normal(size=n)
    w = rng.normal(size=n)
    x = S @ g + 0.5 * u + 0.3 * w + rng.normal(size=n)
    y = 2 * x + 0.5 * w + u
    names = [f"s{k}" for k in range(K)]
    frame = pd.DataFrame(S, columns=names).assign(y=y, x=x, w=w, total=S.sum(axis=1), pop=rng.uniform(1, 5, n), id=range(n))
    return Dataset.from_frame(frame, entity="id"), names, dict(zip(names, g))


def test_the_coefficient_is_2sls_with_the_built_instrument():
    data, names, shocks = _regions()
    fit = shift_share(data, "y", "x", names, shocks, controls=["w", "total"])
    B = data.frame[names].to_numpy() @ np.array([shocks[k] for k in names])
    with_b = data.with_series(bartik=pd.Series(B, index=data.frame.index))
    tsls = Equation("y ~ x + w + total").fit(with_b, method=IV(endog=["x"], instruments=["bartik"]))
    assert fit.beta == pytest.approx(tsls.params["x"], abs=1e-10)
    assert np.allclose(fit.instrument.to_numpy(), B)


@pytest.mark.parametrize("weights", [None, "pop"])
def test_bhj_shock_level_iv_equals_the_regional_iv_exactly(weights):
    data, names, shocks = _regions(seed=1)
    fit = shift_share(data, "y", "x", names, list(shocks.values()), controls=["w"], weights=weights)
    f = data.frame
    e = f[weights].to_numpy() if weights else np.ones(len(f))
    W = np.column_stack([np.ones(len(f)), f["w"]])
    sw = np.sqrt(e)
    resid = lambda v: v - W @ np.linalg.lstsq(W * sw[:, None], v * sw, rcond=None)[0]  # noqa: E731
    yt, xt = resid(f["y"].to_numpy()), resid(f["x"].to_numpy())
    S = f[names].to_numpy()
    sk = S.T @ e
    ybar, xbar = S.T @ (e * yt) / sk, S.T @ (e * xt) / sk
    g = np.array(list(shocks.values()))
    # the shock-level IV: ybar on xbar, instrument g, weights s_k, no constant
    shock_beta = np.sum(sk * g * ybar) / np.sum(sk * g * xbar)
    assert shock_beta == pytest.approx(fit.beta, abs=1e-12)
    assert np.allclose(fit.shock_frame["ybar"], ybar) and np.allclose(fit.shock_frame["exposure"], sk)
    assert float(fit.shock_level().to_frame()["estimate"].iloc[0]) == pytest.approx(fit.beta, abs=1e-12)
    # its exposure-robust error: sum_k s_k^2 g_k^2 ebar_k^2 / (sum_k s_k g_k xbar_k)^2
    ebar = ybar - shock_beta * xbar
    assert fit.se_bhj == pytest.approx(np.sqrt(np.sum((sk * g * ebar) ** 2)) / abs(np.sum(sk * g * xbar)), rel=1e-10)


def test_akm_and_akm0_against_loops_on_a_tiny_example():
    """Six regions, three industries, no controls but the constant: AKM's variance written out with loops."""
    S = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.7, 0.1, 0.2], [0.2, 0.2, 0.6], [0.4, 0.4, 0.2], [0.3, 0.1, 0.6]])
    g = np.array([1.0, -0.5, 0.3])
    x = np.array([0.9, -0.2, 1.1, 0.1, 0.4, 0.5])
    y = np.array([2.0, -0.1, 2.5, 0.6, 0.7, 1.4])
    frame = pd.DataFrame(S, columns=["a", "b", "c"]).assign(y=y, x=x, id=range(6))
    fit = shift_share(Dataset.from_frame(frame, entity="id"), "y", "x", ["a", "b", "c"], {"a": 1.0, "b": -0.5, "c": 0.3})
    n, K = S.shape
    B = [sum(S[i, k] * g[k] for k in range(K)) for i in range(n)]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    Bt = [b - mean(B) for b in B]
    xt = [v - mean(x) for v in x]
    yt = [v - mean(y) for v in y]
    D = sum(Bt[i] * xt[i] for i in range(n))
    beta = sum(Bt[i] * yt[i] for i in range(n)) / D
    eps = [yt[i] - beta * xt[i] for i in range(n)]
    Xhat = np.linalg.solve(S.T @ S, S.T @ np.array(Bt))  # the residualized instrument on the shares
    R = [sum(S[i, k] * eps[i] for i in range(n)) for k in range(K)]
    se_akm = np.sqrt(sum(Xhat[k] ** 2 * R[k] ** 2 for k in range(K))) / abs(D)
    assert fit.beta == pytest.approx(beta, abs=1e-12) and fit.se_akm == pytest.approx(se_akm, rel=1e-10)
    assert fit.se_robust == pytest.approx(np.sqrt(sum(Bt[i] ** 2 * eps[i] ** 2 for i in range(n))) / abs(D), rel=1e-10)
    # with three shocks the null-imposed set is the whole line: sum_k Xhat_k b_k = D, so z^2 sum (Xhat_k b_k)^2 / D^2 >= z^2 / 3 > 1
    assert fit.akm0_interval == (-np.inf, np.inf)


def test_akm0_set_inverts_the_null_imposed_test():
    data, names, shocks = _regions(seed=5)
    fit = shift_share(data, "y", "x", names, shocks)
    f = data.frame
    S, g = f[names].to_numpy(), np.array(list(shocks.values()))
    yt, xt = f["y"].to_numpy() - f["y"].mean(), f["x"].to_numpy() - f["x"].mean()
    Bt = S @ g - (S @ g).mean()
    D = Bt @ xt
    Xhat = np.linalg.lstsq(S, Bt, rcond=None)[0]
    z2 = sps.norm.ppf(0.975) ** 2

    def stat(b0):  # beta0 is in the set when (beta - beta0)^2 <= z^2 sum_k Xhat_k^2 R_k(beta0)^2 / D^2
        return (fit.beta - b0) ** 2 - z2 * np.sum((Xhat * (S.T @ (yt - b0 * xt))) ** 2) / D**2

    lo, hi = fit.akm0_interval
    assert lo < fit.beta < hi and abs(stat(lo)) < 1e-9 and abs(stat(hi)) < 1e-9
    assert stat(lo - 0.01) > 0 and stat(hi + 0.01) > 0 and stat(fit.beta) < 0


def test_shock_clusters():
    data, names, shocks = _regions(seed=2)
    clusters = {k: i % 4 for i, k in enumerate(names)}
    fit = shift_share(data, "y", "x", names, shocks, shock_clusters=clusters)
    single = shift_share(data, "y", "x", names, shocks, shock_clusters={k: i for i, k in enumerate(names)})
    plain = shift_share(data, "y", "x", names, shocks)
    assert single.se_akm == pytest.approx(plain.se_akm) and single.se_bhj == pytest.approx(plain.se_bhj)  # one industry per cluster
    assert fit.shock_clusters == 4 and fit.se_akm != pytest.approx(plain.se_akm)


def test_rotemberg_weights_sum_to_one_and_reproduce_the_estimate():
    data, names, shocks = _regions(seed=3)
    fit = shift_share(data, "y", "x", names, shocks, controls=["w"])
    r = fit.rotemberg()
    assert r["weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (r["weight"] * r["beta_k"]).sum() == pytest.approx(fit.beta, abs=1e-10)
    # beta_k is the just-identified 2SLS with share k alone as the instrument
    k = r.index[0]
    tsls = Equation("y ~ x + w").fit(data, method=IV(endog=["x"], instruments=[k]))
    assert r.loc[k, "beta_k"] == pytest.approx(tsls.params["x"], abs=1e-10)
    assert "Weights sum to 1" in fit.rotemberg_table().notes[0]


def test_monte_carlo_akm_covers_with_correlated_errors():
    """Errors correlated across regions with similar shares (an industry component): the region-level robust interval
    undercovers badly, AKM much less (it is known to undercover a little with few shocks), AKM0 is near 95%."""
    rng = np.random.default_rng(10)
    n, K = 200, 20
    S = rng.dirichlet(np.full(K, 0.3), size=n)
    names = [f"s{k}" for k in range(K)]
    akm, robust, akm0 = [], [], []
    for _ in range(150):
        g = rng.normal(size=K)
        nu = rng.normal(size=K) * 3  # industry-level error component
        u = S @ nu + rng.normal(size=n)
        x = S @ g + 0.5 * u + rng.normal(size=n)
        y = 1.0 * x + u
        frame = pd.DataFrame(S, columns=names).assign(y=y, x=x, id=range(n))
        fit = shift_share(Dataset.from_frame(frame, entity="id"), "y", "x", names, g)
        akm.append(abs(fit.beta - 1) < 1.96 * fit.se_akm)
        robust.append(abs(fit.beta - 1) < 1.96 * fit.se_robust)
        akm0.append(fit.in_akm0_set(1.0))
    assert np.mean(robust) < 0.8 and np.mean(akm) > np.mean(robust) + 0.08 and 0.9 <= np.mean(akm0) <= 0.99


def test_errors():
    data, names, shocks = _regions()
    with pytest.raises(SpecificationError, match="no shock"):
        shift_share(data, "y", "x", names, {"s0": 1.0})
    with pytest.raises(SpecificationError, match="shocks for"):
        shift_share(data, "y", "x", names, [1.0, 2.0])
    with pytest.raises(SpecificationError, match="at least two"):
        shift_share(data, "y", "x", ["s0"], [1.0])
    with pytest.raises(SpecificationError, match="no cluster"):
        shift_share(data, "y", "x", names, shocks, shock_clusters={"s0": 1})


def test_the_intervals_are_labelled_by_alpha():
    """With alpha=0.1 the tables' intervals are 90% ones: their columns say so (they were labelled [0.025, 0.975])."""
    data, names, shocks = _regions()
    fit = shift_share(data, "y", "x", names, shocks, alpha=0.1)
    q = sps.norm.ppf(0.95)
    for table in (fit.summary(), fit.shock_level()):
        frame = table.to_frame()
        assert list(frame.columns[-2:]) == ["[0.05", "0.95]"]
        row = frame.iloc[0]
        assert row["[0.05"] == pytest.approx(row["estimate"] - q * row["std err"])
    assert list(shift_share(data, "y", "x", names, shocks).summary().to_frame().columns[-2:]) == ["[0.025", "0.975]"]


def test_the_dialog_records_the_call_and_the_script_replays():
    from pymetrix.commands import COMMANDS, Session

    data, names, shocks = _regions(K=3, seed=4)
    s = Session()
    s.objects["data"] = data
    text = ", ".join(f"{k}: {v}" for k, v in shocks.items())
    s.run(COMMANDS["ShiftShareIV"].cls("y", "x", "s0, s1, s2", text, controls="w", shock_clusters="s0: a, s1: a, s2: b", se="bhj"))
    assert s.history[-1].code.startswith("bartik = shift_share(data, 'y', 'x', ['s0', 's1', 's2'], {'s0': ")
    direct = shift_share(data, "y", "x", names, shocks, controls=["w"], shock_clusters={"s0": "a", "s1": "a", "s2": "b"}, se="bhj")
    got = s.objects["bartik"]
    assert got.beta == direct.beta and got.se_bhj == direct.se_bhj and got.params.se[0] == direct.se_bhj
    namespace = {"data": data}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the estimate
    assert namespace["bartik"].beta == direct.beta
    ordered = COMMANDS["ShiftShareIV"].cls("y", "x", "s0, s1, s2", ", ".join(str(v) for v in shocks.values()))
    assert ordered._arguments()["shocks"] == shocks and ordered.validate(s) == []
    assert COMMANDS["ShiftShareIV"].cls("y", "x", "s0, s1", "1, 2, 3").validate(s)
