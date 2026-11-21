"""Synthetic difference-in-differences, checked independently: the estimate equal to two-way fixed effects weighted
by omega_i lambda_t (statsmodels WLS), as Arkhangelsky et al. show; the reported DiD equal to the 2x2 difference of
means; the unit weights the minimiser of their penalised objective (a Frank-Wolfe solution no better); in a factor
model where the treated units load more on the factors, DiD biased and SDID near the truth; placebo standard errors
for a single treated unit; and refusals."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, synthetic_did
from pymetrix.errors import SpecificationError

sd = importlib.import_module("pymetrix.synthdid")


def _panel(N=40, T=20, n_tr=5, t0=15, seed=0, effect=2.0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 2)).cumsum(axis=0) * 0.5
    load = rng.normal(size=(N, 2))
    load[-n_tr:] += 1.0  # the treated load more on the trending factors: parallel trends fail
    Y = 5 + load @ f.T + rng.normal(size=(N, T)) * 0.3 + rng.normal(size=(N, 1))
    D = np.zeros((N, T))
    D[-n_tr:, t0:] = 1
    Y = Y + effect * D
    rows = [(i, 2000 + t, Y[i, t], D[i, t]) for i in range(N) for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows, columns=["unit", "year", "y", "d"]), entity="unit", time="year", freq="Y"), Y, D


def test_the_weighted_twfe_identity_and_did():
    sm = pytest.importorskip("statsmodels.api")
    data, Y, D = _panel()
    fit = synthetic_did(data, "y", "d")
    N, T, n_tr, t0 = 40, 20, 5, 15
    w = np.zeros((N, T))
    for i in range(N):
        wi = fit.unit_weights.get(i, 0.0) if i < N - n_tr else 1 / n_tr
        for t in range(T):
            w[i, t] = wi * (fit.time_weights.iloc[t] if t < t0 else 1 / (T - t0))
    X = np.column_stack([D.reshape(-1)] + [np.kron(np.eye(N)[:, j], np.ones(T)) for j in range(N)]
                        + [np.tile(np.eye(T)[:, s], N) for s in range(1, T)])
    keep = w.reshape(-1) > 0
    Xk = X[keep][:, X[keep].any(axis=0)]  # without the dummies of units given no weight, so the reference is of full rank
    ref = sm.WLS(Y.reshape(-1)[keep], Xk, weights=w.reshape(-1)[keep]).fit()
    assert fit.att == pytest.approx(ref.params[0], abs=1e-10)
    tr, co = Y[-n_tr:], Y[:-n_tr]
    did = (tr[:, t0:].mean() - tr[:, :t0].mean()) - (co[:, t0:].mean() - co[:, :t0].mean())
    assert fit.did == pytest.approx(did, abs=1e-10)
    assert fit.att == pytest.approx(2.0, abs=3 * fit.se) and abs(fit.did - 2.0) > 1.0  # DiD is biased here
    assert fit.unit_weights.sum() == pytest.approx(1) and fit.time_weights.sum() == pytest.approx(1) and fit.se_method == "jackknife"


def test_the_unit_weights_minimise_their_objective():
    data, Y, _ = _panel(seed=1)
    fit = synthetic_did(data, "y", "d")
    Yco, Ytr = Y[:35, :15], Y[35:, :15].mean(axis=0)
    A, b = Yco.T - Yco.T.mean(axis=0), Ytr - Ytr.mean()
    pen = fit.zeta**2 * 15

    def objective(w):
        return float(np.sum((A @ w - b) ** 2) + pen * w @ w)

    w = np.full(35, 1 / 35)
    for k in range(20000):  # Frank-Wolfe on the simplex: an independent solver
        g = 2 * (A.T @ (A @ w - b) + pen * w)
        s = np.zeros(35)
        s[np.argmin(g)] = 1
        w = w + 2 / (k + 2) * (s - w)
    ours = fit.unit_weights.reindex(range(35)).fillna(0).to_numpy()
    assert objective(ours) <= objective(w) + 1e-8


def test_one_treated_unit_uses_placebos():
    data, _, _ = _panel(N=30, n_tr=1, seed=2)
    fit = synthetic_did(data, "y", "d", placebos=50)
    assert fit.se_method == "placebo" and fit.se > 0 and fit.summary().title.startswith("Synthetic difference")
    assert list(fit.paths.columns) == ["treated", "synthetic"]


def test_refusals():
    data, Y, D = _panel()
    frame = data.frame.reset_index()
    frame.loc[(frame["unit"] == 39) & (frame["year"].astype(str) == "2016"), "d"] = 0  # treatment switching off
    bad = Dataset.from_frame(frame.assign(year=frame["year"].astype(str).astype(int)), entity="unit", time="year", freq="Y")
    with pytest.raises(SpecificationError, match="same period"):
        synthetic_did(bad, "y", "d")
    with pytest.raises(SpecificationError, match="jackknife"):
        synthetic_did(_panel(N=30, n_tr=1, seed=3)[0], "y", "d", se="jackknife")


def test_the_dialog():
    from pymetrix.commands import COMMANDS, Session

    s = Session()
    s.objects["data"] = _panel(N=20, seed=4)[0]
    s.run(COMMANDS["FitSyntheticDiD"].cls("y", "d"))
    assert s.history[-1].code == "sdid = synthetic_did(data, 'y', 'd')" and s.objects["sdid"].se_method == "jackknife"
